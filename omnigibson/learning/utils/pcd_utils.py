# Helper script for exporting a point cloud from an hdf5 file containing RGB + Depth images

import fpsample
import numpy as np
import omnigibson.utils.transform_utils as T
import open3d as o3d
import torch as th
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


# Intrinsics for external camera
# Focal length is 17mm, sensor width is 40mm, resolution is 240x240
EXTERNAL_INTRINSICS = np.array([[102.000, 0.000, 120.000], [0.000, 102.000, 120.000], [0.000, 0.000, 1.000]])

# Intrinsics for wrist cameras
# Focal length is 17mm, sensor width is 20.995mm, resolution is 240x240
WRIST_INTRINSICS = np.array([[194.332, 0.000, 120.000], [0.000, 194.332, 120.000], [0.000, 0.000, 1.000]])


def color_pcd_vis(color_pcd):
    # visualize with open3D
    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(color_pcd[:, :3])
    pcd.points = o3d.utility.Vector3dVector(color_pcd[:, 3:])
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, axis])
    print("number points", color_pcd.shape[0])


def depth_to_pcd(
    depth,
    pose,
    base_link_pose,
    K,
    max_depth=20,
):
    # get the homogeneous transformation matrix from quaternion
    pos = pose[:3]
    quat = pose[3:]
    rot = R.from_quat(quat)  # scipy expects [x, y, z, w]
    rot_add = R.from_euler("x", np.pi).as_matrix()  # handle the cam_to_img transformation
    rot_matrix = rot.as_matrix() @ rot_add  # 3x3 rotation matrix
    world_to_cam_tf = np.eye(4)
    world_to_cam_tf[:3, :3] = rot_matrix
    world_to_cam_tf[:3, 3] = pos

    # filter depth
    mask = depth > max_depth
    depth[mask] = 0
    h, w = depth.shape
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij", sparse=False)
    assert depth.min() >= 0
    u = x
    v = y
    uv = np.dstack((u, v, np.ones_like(u)))  # (img_width, img_height, 3)

    Kinv = np.linalg.inv(K)

    pc = depth.reshape(-1, 1) * (uv.reshape(-1, 3) @ Kinv.T)
    pc = pc.reshape(h, w, 3)
    pc = np.concatenate([pc.reshape(-1, 3), np.ones((h * w, 1))], axis=-1)  # shape (H*W, 4)
    if isinstance(base_link_pose, np.ndarray):
        base_link_pose = th.from_numpy(base_link_pose)
    world_to_robot_tf = T.pose2mat((base_link_pose[:3], base_link_pose[3:])).numpy()
    robot_to_world_tf = np.linalg.inv(world_to_robot_tf)
    pc = (pc @ world_to_cam_tf.T @ robot_to_world_tf.T)[:, :3].reshape(h, w, 3)

    return pc


def downsample_pcd(color_pcd, num_points):
    if color_pcd.shape[0] > num_points:
        pc = color_pcd[:, 3:]
        color_img = color_pcd[:, :3]
        kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(pc, num_points, h=5)
        pc = pc[kdline_fps_samples_idx]
        color_img = color_img[kdline_fps_samples_idx]
        color_pcd = np.concatenate([color_img, pc], axis=-1)
    else:
        # randomly sample points
        pad_number_of_points = num_points - color_pcd.shape[0]
        random_idx = np.random.choice(color_pcd.shape[0], pad_number_of_points, replace=True)
        pad_pcd = color_pcd[random_idx]
        color_pcd = np.concatenate([color_pcd, pad_pcd], axis=0)
        # raise ValueError("color_pcd shape is smaller than num_points_to_sample")
    return color_pcd


def process_fused_point_cloud(obs, pcd_num_points: int = 4096) -> np.ndarray:
    base_link_pose = obs["robot_r1::robot_base_link_pose"][:]
    left_cam_pose = obs["robot_r1::left_cam_pose"][:]
    right_cam_pose = obs["robot_r1::right_cam_pose"][:]
    external_cam_pose = obs["robot_r1::external_cam_pose"][:]
    left_rgb = obs["robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"][..., :3]
    right_rgb = obs["robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"][..., :3]
    external_rgb = obs["external::external_sensor0::rgb"][..., :3]
    left_depth = obs["robot_r1::robot_r1:left_realsense_link:Camera:0::depth_linear"][:]
    right_depth = obs["robot_r1::robot_r1:right_realsense_link:Camera:0::depth_linear"][:]
    external_depth = obs["external::external_sensor0::depth_linear"][:]
    data_size = base_link_pose.shape[0] if len(base_link_pose.shape) > 1 else 1
    if data_size == 1:
        # Left point cloud
        left_pcd = depth_to_pcd(left_depth, left_cam_pose, base_link_pose, K=WRIST_INTRINSICS)
        left_rgb_pcd = np.concatenate([left_rgb / 255.0, left_pcd], axis=-1).reshape(-1, 6)

        # Right point cloud
        right_pcd = depth_to_pcd(right_depth, right_cam_pose, base_link_pose, K=WRIST_INTRINSICS)
        right_rgb_pcd = np.concatenate([right_rgb / 255.0, right_pcd], axis=-1).reshape(-1, 6)

        # External camera point cloud
        external_pcd = depth_to_pcd(external_depth, external_cam_pose, base_link_pose, K=EXTERNAL_INTRINSICS)
        external_rgb_pcd = np.concatenate([external_rgb / 255.0, external_pcd], axis=-1).reshape(-1, 6)

        # Fuse all point clouds and downsample
        fused_pcd_all = np.concatenate([left_rgb_pcd, right_rgb_pcd, external_rgb_pcd], axis=0)
        fused_pcd = downsample_pcd(fused_pcd_all, pcd_num_points).astype(np.float32)
    else:
        fused_pcd = np.zeros((data_size, pcd_num_points, 6), dtype=np.float32)  # Initialize empty point cloud
        for i in tqdm(range(data_size)):
            # Left point cloud
            left_pcd = depth_to_pcd(left_depth[i], left_cam_pose[i], base_link_pose[i], K=WRIST_INTRINSICS)
            left_rgb_pcd = np.concatenate([left_rgb[i] / 255.0, left_pcd], axis=-1).reshape(-1, 6)

            # Right point cloud
            right_pcd = depth_to_pcd(right_depth[i], right_cam_pose[i], base_link_pose[i], K=WRIST_INTRINSICS)
            right_rgb_pcd = np.concatenate([right_rgb[i] / 255.0, right_pcd], axis=-1).reshape(-1, 6)

            # External camera point cloud
            external_pcd = depth_to_pcd(
                external_depth[i], external_cam_pose[i], base_link_pose[i], K=EXTERNAL_INTRINSICS
            )
            external_rgb_pcd = np.concatenate([external_rgb[i] / 255.0, external_pcd], axis=-1).reshape(-1, 6)

            # Fuse all point clouds and downsample
            fused_pcd_all = np.concatenate([left_rgb_pcd, right_rgb_pcd, external_rgb_pcd], axis=0)
            fused_pcd[i] = downsample_pcd(fused_pcd_all, pcd_num_points)

    return fused_pcd
