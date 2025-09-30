import numpy as np
import pycolmap


def batch_np_matrix_to_pycolmap_wo_track(
    points3d,
    points_xyf,
    points_rgb,
    extrinsics,
    intrinsics,
    filepaths,
    image_height,
    image_width,
    shared_camera=False,
    camera_type="SIMPLE_PINHOLE",
):
    """
    Convert Batched NumPy Arrays to PyCOLMAP

    Different from batch_np_matrix_to_pycolmap, this function does not use tracks.

    It saves points3d to colmap reconstruction format only to serve as init for Gaussians or other nvs methods.

    Do NOT use this for BA.
    """
    # points3d: Px3
    # points_xyf: Px3, with x, y coordinates and frame indices
    # points_rgb: Px3, rgb colors
    # extrinsics: Nx3x4
    # intrinsics: Nx3x3
    # image_size: 2, assume all the frames have been padded to the same size
    # where N is the number of frames and P is the number of tracks

    N = len(extrinsics)
    P = len(points3d)

    # Reconstruction object, following the format of PyCOLMAP/COLMAP
    reconstruction = pycolmap.Reconstruction()

    for vidx in range(P):
        reconstruction.add_point3D(points3d[vidx], pycolmap.Track(), points_rgb[vidx])

    camera = None
    # frame idx
    for fidx in range(N):
        # set camera
        if camera is None or (not shared_camera):
            pycolmap_intri = build_pycolmap_intri(fidx, intrinsics, camera_type)

            camera = pycolmap.Camera(
                model=camera_type, width=image_width, height=image_height, params=pycolmap_intri, camera_id=fidx + 1
            )

            # add camera
            reconstruction.add_camera(camera)

        # set image
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsics[fidx][:3, :3]), extrinsics[fidx][:3, 3]
        )  # Rot and Trans

        image = pycolmap.Image(
            id=fidx + 1, name=filepaths[fidx], camera_id=camera.camera_id, cam_from_world=cam_from_world
        )

        points2D_list = []

        point2D_idx = 0

        points_belong_to_fidx = points_xyf[:, 2].astype(np.int32) == fidx
        points_belong_to_fidx = np.nonzero(points_belong_to_fidx)[0]

        for point3D_batch_idx in points_belong_to_fidx:
            point3D_id = point3D_batch_idx + 1
            point2D_xyf = points_xyf[point3D_batch_idx]
            point2D_xy = point2D_xyf[:2]
            points2D_list.append(pycolmap.Point2D(point2D_xy, point3D_id))

            # add element
            track = reconstruction.points3D[point3D_id].track
            track.add_element(fidx + 1, point2D_idx)
            point2D_idx += 1

        assert point2D_idx == len(points2D_list)

        try:
            image.points2D = pycolmap.ListPoint2D(points2D_list)
            image.registered = True
        except:
            print(f"frame {fidx + 1} does not have any points")
            image.registered = False

        # add image
        reconstruction.add_image(image)

    return reconstruction


def build_pycolmap_intri(fidx, intrinsics, camera_type, extra_params=None):
    """
    Helper function to get camera parameters based on camera type.

    Args:
        fidx: Frame index
        intrinsics: Camera intrinsic parameters
        camera_type: Type of camera model
        extra_params: Additional parameters for certain camera types

    Returns:
        pycolmap_intri: NumPy array of camera parameters
    """
    if camera_type == "PINHOLE":
        pycolmap_intri = np.array(
            [intrinsics[fidx][0, 0], intrinsics[fidx][1, 1], intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]]
        )
    elif camera_type == "SIMPLE_PINHOLE":
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        pycolmap_intri = np.array([focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]])
    elif camera_type == "SIMPLE_RADIAL":
        raise NotImplementedError("SIMPLE_RADIAL is not supported yet")
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        pycolmap_intri = np.array([focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2], extra_params[fidx][0]])
    else:
        raise ValueError(f"Camera type {camera_type} is not supported yet")

    return pycolmap_intri


def create_pixel_coordinate_grid(num_frames, height, width):
    """
    Creates a grid of pixel coordinates and frame indices for all frames.
    Returns:
        tuple: A tuple containing:
            - points_xyf (numpy.ndarray): Array of shape (num_frames, height, width, 3)
                                            with x, y coordinates and frame indices
            - y_coords (numpy.ndarray): Array of y coordinates for all frames
            - x_coords (numpy.ndarray): Array of x coordinates for all frames
            - f_coords (numpy.ndarray): Array of frame indices for all frames
    """
    # Create coordinate grids for a single frame
    y_grid, x_grid = np.indices((height, width), dtype=np.float32)
    x_grid = x_grid[np.newaxis, :, :]
    y_grid = y_grid[np.newaxis, :, :]

    # Broadcast to all frames
    x_coords = np.broadcast_to(x_grid, (num_frames, height, width))
    y_coords = np.broadcast_to(y_grid, (num_frames, height, width))

    # Create frame indices and broadcast
    f_idx = np.arange(num_frames, dtype=np.float32)[:, np.newaxis, np.newaxis]
    f_coords = np.broadcast_to(f_idx, (num_frames, height, width))

    # Stack coordinates and frame indices
    points_xyf = np.stack((x_coords, y_coords, f_coords), axis=-1)

    return points_xyf
