import math
import shutil

import cv2
from omnigibson.utils.mvs_utils import batch_np_matrix_to_pycolmap_wo_track, create_pixel_coordinate_grid, build_pycolmap_intri
import omnigibson.utils.camera_utils as ilutils
import omnigibson.utils.transform_utils as T
import os
import trimesh
import h5py
import numpy as np
import torch
import json
import tqdm
import fpsample


def hdf5_to_colmap(scene_dir):
    images_dir = os.path.join(scene_dir, "images")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory {images_dir} does not exist. Please run the data collection script first.")

    MAX_IMAGES = 10000

    # Open the HDF5 file in read mode
    print("Loading HDF5 file...")
    hdf5_file_path = os.path.join(scene_dir, "camera_data.hdf5")
    if not os.path.exists(hdf5_file_path):
        raise FileNotFoundError(f"HDF5 file {hdf5_file_path} does not exist. Please run the data collection script first.")
    with h5py.File(hdf5_file_path, 'r') as f:
        rgb_data = f['rgb']                 # shape: (TOTAL_IMAGES, HEIGHT, WIDTH, 4), dtype: uint8
        depth_data = f['depth']             # shape: (TOTAL_IMAGES, HEIGHT, WIDTH), dtype: float32
        segmentation_data = f['segmentation']  # shape: (TOTAL_IMAGES, HEIGHT, WIDTH), dtype: int32
        
        camera_pose_data = f['camera_pose']       # shape: (TOTAL_IMAGES, 4, 4), dtype: float32

        # Load the segmentation labels (stored as a JSON string in the file attributes)
        # segmentation_labels = json.loads(f.attrs['segmentation_labels'])
        # print(len(segmentation_labels), "segmentation labels found in the HDF5 file.")
        # assert len(segmentation_labels) < 256, "Only 256 segmentation labels are supported by Gaga."

        # Find the image files in the images directory
        image_files = os.listdir(images_dir)
        image_files_by_idx = {int(os.path.splitext(os.path.basename(f))[0].split("_")[1]): f for f in image_files}

        # Create a directory for storing the depth images
        depth_dir = os.path.join(scene_dir, "depth")
        shutil.rmtree(depth_dir, ignore_errors=True)
        os.makedirs(depth_dir, exist_ok=True)

        # Create a directory for storing the segmentation images
        segmentation_dir = os.path.join(scene_dir, "segmentation")
        shutil.rmtree(segmentation_dir, ignore_errors=True)
        os.makedirs(segmentation_dir, exist_ok=True)

        # Convert the data to a point cloud format
        all_points_3d = []
        all_points_rgb = []
        all_points_xyf = []
        image_paths = []
        extrinsics = []
        depth_params = {}

        cam_intrinsics = np.array(json.loads(f.attrs["camera_intrinsics"]), dtype=np.float32)[None, :, :]

        # Total number of desired points
        MAX_POINTS = 1000000
        MAX_POINTS_FACTOR = 10  # Get this many more points than needed to ensure we have enough after filtering
        image_count = min(rgb_data.shape[0], MAX_IMAGES)
        points_per_image = MAX_POINTS * MAX_POINTS_FACTOR // image_count

        for i in tqdm.trange(image_count, desc="Processing images"):
            points_rgb = rgb_data[i][..., :3]
            depth = depth_data[i]
            seg = segmentation_data[i]
            cam_pose = camera_pose_data[i]
            cam_pos, cam_orn = T.mat2pose(torch.tensor(cam_pose))

            # Generate point cloud
            quat_for_conversion = ilutils.convert_camera_frame_orientation_convention(
                cam_orn, "opengl", "ros"
            ).reshape(4)
            points_3d = ilutils.create_pointcloud_from_depth(
                torch.as_tensor(cam_intrinsics),
                torch.as_tensor(depth)[None],
                True,
                position=cam_pos[None],
                orientation=quat_for_conversion[None],
            )
            points_3d = points_3d.reshape((points_rgb.shape[1], points_rgb.shape[0], 3)).permute(
                1, 0, 2
            ).numpy()

            # Flatten point cloud and related data
            points_xyf = create_pixel_coordinate_grid(1, points_rgb.shape[0], points_rgb.shape[1])[0]
            points_xyf[..., 2] = i   # Set the image index as the third coordinate

            assert points_3d.shape[:2] == points_xyf.shape[:2] == points_rgb.shape[:2], \
                "Mismatch in shape of point cloud, pixel coordinates, and RGB data."

            # Remove all the invalid points in the point cloud
            finite_mask = np.isfinite(points_3d).all(axis=-1)
            points_3d = points_3d[finite_mask]
            points_rgb = points_rgb[finite_mask]
            points_xyf = points_xyf[finite_mask]

            # Subsample points if there are too many
            if points_3d.shape[0] > points_per_image:
                indices = np.random.choice(points_3d.shape[0], points_per_image, replace=False)
                points_3d = points_3d[indices]
                points_rgb = points_rgb[indices]
                points_xyf = points_xyf[indices]

            all_points_rgb.append(points_rgb)
            all_points_3d.append(points_3d)
            all_points_xyf.append(points_xyf)
            image_paths.append(image_files_by_idx[i])

            # Move the camera extrinsics to the OpenCV-based frame
            flip_yz = np.diag([1, -1, -1])  # 4x4 matrix
            cam_rotation = cam_pose[:3, :3]
            cam_rotation_opencv = cam_rotation @ flip_yz
            cam_extrinsic_opencv = np.eye(4)
            cam_extrinsic_opencv[:3, :3] = cam_rotation_opencv
            cam_extrinsic_opencv[:3, 3] = cam_pose[:3, 3]  # Translation part
            extrinsics.append(np.linalg.inv(cam_extrinsic_opencv))

            # Save the depth and seg image also.
            depth_path = os.path.join(depth_dir, os.path.basename(image_files_by_idx[i]))
            finite_depth = depth[np.isfinite(depth) & (depth > 0)]
            max_depth = np.max(finite_depth) if finite_depth.size > 0 else 0
            min_depth = np.min(finite_depth) if finite_depth.size > 0 else 0
            depth_range = max_depth - min_depth
            depth_zero_to_one = (depth - min_depth) / depth_range if depth_range > 0 else np.zeros_like(depth)
            depth_params[os.path.splitext(os.path.basename(image_files_by_idx[i]))[0]] = {"scale": float(depth_range), "offset": float(min_depth)}
            cv2.imwrite(depth_path, (depth_zero_to_one * 255).astype(np.uint8))

            seg_path = os.path.join(segmentation_dir, os.path.basename(image_files_by_idx[i]))
            cv2.imwrite(seg_path, seg.astype(np.uint8))

        print("Stacking points and colors...")
        all_points_3d = np.concatenate(all_points_3d, axis=0)
        all_points_rgb = np.concatenate(all_points_rgb, axis=0)
        all_points_xyf = np.concatenate(all_points_xyf, axis=0)
        extrinsics = np.stack(extrinsics, axis=0)

        print("Reducing point cloud size...")

        # Limit the number of points
        max_points = min(MAX_POINTS, all_points_3d.shape[0])
        if len(all_points_3d) > max_points:
            indices = np.random.choice(all_points_3d.shape[0], max_points, replace=False)
            # indices = fpsample.bucket_fps_kdline_sampling(points_3d.reshape(-1, 3), max_points, h=5)
            all_points_3d = all_points_3d[indices]
            all_points_rgb = all_points_rgb[indices]
            all_points_xyf = all_points_xyf[indices]

        assert all_points_3d.shape == all_points_rgb.shape == all_points_xyf.shape == (max_points, 3), \
            "Mismatch in number of points, colors, and pixel coordinates."

        print("Creating COLMAP reconstruction...")
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            all_points_3d,
            all_points_xyf,
            all_points_rgb,
            extrinsics,
            cam_intrinsics,
            image_paths,
            rgb_data.shape[1],
            rgb_data.shape[2],
            shared_camera=True,
            camera_type="PINHOLE",
        )

        print(f"Saving reconstruction to {scene_dir}/sparse")
        sparse_reconstruction_dir = os.path.join(scene_dir, "sparse", "0")
        shutil.rmtree(sparse_reconstruction_dir, ignore_errors=True)
        os.makedirs(sparse_reconstruction_dir, exist_ok=True)
        reconstruction.write(sparse_reconstruction_dir)

        # Write the depth parameters to a JSON file as expected by the Gaussian Splatting code
        with open(os.path.join(sparse_reconstruction_dir, "depth_params.json"), "w") as f:
            json.dump(depth_params, f, indent=2)

        # Save point cloud for fast visualization
        print(f"Exporting point cloud to {sparse_reconstruction_dir}/points.ply")
        trimesh.PointCloud(all_points_3d, colors=all_points_rgb).export(os.path.join(sparse_reconstruction_dir, "points.ply"))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert HDF5 data to COLMAP format.")
    parser.add_argument("scene_dir", type=str, help="Path to the scene directory containing the HDF5 file.")
    args = parser.parse_args()

    hdf5_to_colmap(args.scene_dir)