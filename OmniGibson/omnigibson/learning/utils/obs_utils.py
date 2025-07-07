import av
import fpsample
import numpy as np
import omnigibson.utils.transform_utils as T
import open3d as o3d
import torch as th
from scipy.spatial.transform import Rotation as R
from typing import Dict, Tuple


MIN_DEPTH = 0.0
MAX_DEPTH = 10.0
DEPTH_SHIFT = 0.1

def quantize_depth(
    depth: th.Tensor, 
    min_depth: float = MIN_DEPTH, 
    max_depth: float = MAX_DEPTH, 
    shift: float=DEPTH_SHIFT
) -> th.Tensor:
    """
    Quantizes depth values to a 14-bit range (0 to 16383) based on the specified min and max depth.
    
    Args:
        depth (th.Tensor): Depth tensor.
        min_depth (float): Minimum depth value.
        max_depth (float): Maximum depth value.
        shift (float): Small value to shift depth to avoid log(0).
    Returns:
        th.Tensor: Quantized depth tensor.
    """
    qmax = (1 << 14) - 1
    log_min = th.log(th.tensor(min_depth + shift, dtype=depth.dtype, device=depth.device))
    log_max = th.log(th.tensor(max_depth + shift, dtype=depth.dtype, device=depth.device))

    log_depth = th.log(depth + shift)
    log_norm = (log_depth - log_min) / (log_max - log_min)
    quantized_depth = th.clamp((log_norm * qmax).round(), 0, qmax).to(th.uint16)

    return quantized_depth


def dequantize_depth(
    quantized_depth: th.Tensor, 
    min_depth: float = MIN_DEPTH, 
    max_depth: float = MAX_DEPTH, 
    shift: float=DEPTH_SHIFT
) -> th.Tensor:
    """
    Dequantizes a 14-bit depth tensor back to the original depth values.
    
    Args:
        quantized_depth (th.Tensor): Quantized depth tensor.
        min_depth (float): Minimum depth value.
        max_depth (float): Maximum depth value.
        shift (float): Small value to shift depth to avoid log(0).
    Returns:
        th.Tensor: Dequantized depth tensor.
    """
    qmax = (1 << 14) - 1
    log_min = th.log(th.tensor(min_depth + shift, dtype=th.float32, device=quantized_depth.device))
    log_max = th.log(th.tensor(max_depth + shift, dtype=th.float32, device=quantized_depth.device))

    log_norm = quantized_depth.to(th.float32) / qmax
    log_depth = log_norm * (log_max - log_min) + log_min
    depth = th.clamp(th.exp(log_depth) - shift, min=min_depth, max=max_depth)

    return depth


class VideoLoader:
    def __init__(self, path: str, type: str = "rgb", streaming: bool = False):
        """
        Initialize VideoLoader for loading RGB or depth videos.
        
        Args:
            path (str): Path to the video file
            video_type (str): Either "rgb" or "depth"
            streaming (bool): If True, load frames on demand. If False, load all frames at once.
        """
        self.path = path
        self.type = type
        self.streaming = streaming
        
        if not streaming:
            # Load all frames at once
            if type == "rgb":
                self.frames = load_rgb_video(path)
            elif type == "depth":
                self.frames = load_depth_video(path)
            else:
                raise ValueError(f"Unsupported video type: {type}")
        else:
            # Initialize for streaming
            self.container = av.open(path)
            self.stream = self.container.streams.video[0]
            self.frame_iterator = None
    
    def __iter__(self):
        """Make the loader iterable for streaming mode."""
        if not self.streaming:
            # Return all frames as a single tensor
            yield self.frames
        else:
            # Stream frames one by one
            self.frame_iterator = self.container.demux(self.stream)
            for packet in self.frame_iterator:
                for frame in packet.decode():
                    if self.type == "rgb":
                        rgb = frame.to_ndarray(format="rgb24")
                        yield th.from_numpy(rgb).float()
                    elif self.type == "depth":
                        frame_gray16 = frame.reformat(format='gray16le').to_ndarray()
                        depth_frame = th.from_numpy(frame_gray16)
                        yield dequantize_depth(depth_frame)
    
    def __len__(self):
        """Return the number of frames."""
        if not self.streaming:
            return self.frames.shape[0]
        else:
            # For streaming, we need to count frames
            if not hasattr(self, '_frame_count'):
                temp_container = av.open(self.path)
                temp_stream = temp_container.streams.video[0]
                self._frame_count = temp_stream.frames
                temp_container.close()
            return self._frame_count
    
    def close(self):
        """Close the video container (only needed for streaming mode)."""
        if self.streaming and hasattr(self, 'container'):
            self.container.close()


def load_rgb_video(path: str) -> th.Tensor:
    container = av.open(path)
    stream = container.streams.video[0]
    
    frames = []
    for packet in container.demux(stream):
        for frame in packet.decode():
            rgb = frame.to_ndarray(format="rgb24")  # (H, W, 3), dtype=uint8
            rgb = th.from_numpy(rgb).unsqueeze(0)  # (1, H, W, 3)
            frames.append(rgb)
    
    container.close()
    return th.cat(frames, dim=0).float()  # (T, 3, H, W)


def load_depth_video(path: str, min_depth: float=MIN_DEPTH, max_depth: float=MAX_DEPTH, shift: float=DEPTH_SHIFT) -> th.Tensor:
    container = av.open(path)
    stream = container.streams.video[0]
    
    frames = []
    for packet in container.demux(stream):
        for frame in packet.decode():
            # Decode Y (luma) channel only; YUV420 → grayscale image
            frame_gray16 = frame.reformat(format='gray16le').to_ndarray()
            frames.append(th.from_numpy(frame_gray16).unsqueeze(0))  # (1, H, W)
    
    container.close()
    video = th.cat(frames, dim=0)  # (T, H, W)
    depth = dequantize_depth(video, min_depth=min_depth, max_depth=max_depth, shift=shift)
    return depth


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


def downsample_pcd(color_pcd, num_points) -> Tuple[np.ndarray, np.ndarray]:
    """
    If the current number of point cloud is smaller than num_points, will downsample it
    Otherwise, randomly sample current points to reach num_points
    """
    if color_pcd.shape[0] > num_points:
        pc = color_pcd[:, 3:]
        color_img = color_pcd[:, :3]
        kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(pc, num_points, h=5)
        pc = pc[kdline_fps_samples_idx]
        color_img = color_img[kdline_fps_samples_idx]
        color_pcd = np.concatenate([color_img, pc], axis=-1)
        sampled_idx = kdline_fps_samples_idx
    else:
        # randomly sample points
        pad_number_of_points = num_points - color_pcd.shape[0]
        random_idx = np.random.choice(color_pcd.shape[0], pad_number_of_points, replace=True)
        pad_pcd = color_pcd[random_idx]
        color_pcd = np.concatenate([color_pcd, pad_pcd], axis=0)
        sampled_idx = np.concatenate([np.arange(color_pcd.shape[0]), random_idx])
    return color_pcd, sampled_idx


def process_fused_point_cloud(
    obs: dict,
    robot_name: str, 
    camera_intrinsics: Dict[str, np.ndarray],
    pcd_num_points: int = 4096
) -> Tuple[np.ndarray, np.ndarray]:
	base_link_pose = obs[f"{robot_name}::robot_base_link_pose"]
	data_size = base_link_pose.shape[0] if len(base_link_pose.shape) > 1 else 1
	camera_depth, camera_rgb, camera_seg, camera_pose = dict(), dict(), dict(), dict()
	if data_size == 1:
		for camera_name in camera_intrinsics.keys():
			camera_depth[camera_name] = obs[f"{camera_name}::depth_linear"]
			camera_rgb[camera_name] = obs[f"{camera_name}::rgb"]
			camera_seg[camera_name] = obs[f"{camera_name}::seg_semantic"]
			camera_pose[camera_name] = obs[f"{camera_name}::pose"]
		rgb_pcd, seg_pcd = [], []
		for camera_name, intrinsics in camera_intrinsics.items():
			pcd = depth_to_pcd(camera_depth[camera_name][:], camera_pose[camera_name][:], base_link_pose[:], K=intrinsics)
			rgb_pcd.append(
				np.concatenate([camera_rgb[camera_name][..., :3] / 255.0, pcd], axis=-1).reshape(-1, 6)
			)  # shape (H*W, 6) 
			seg_pcd.append(camera_seg[camera_name][:].reshape(-1)) # shape (H*W)
		# Fuse all point clouds and downsample
		fused_pcd_all = np.concatenate(rgb_pcd, axis=0)
		fused_pcd, sampled_idx = downsample_pcd(fused_pcd_all, pcd_num_points)
		fused_pcd = fused_pcd.astype(np.float32)
		fused_seg = np.concatenate(seg_pcd, axis=0)[sampled_idx].astype(np.uint32)
	else:
		for camera_name in camera_intrinsics.keys():
			camera_depth[camera_name] = iter(obs[f"{camera_name}::depth_linear"])
			camera_rgb[camera_name] = iter(obs[f"{camera_name}::rgb"])
			camera_seg[camera_name] = obs[f"{camera_name}::seg_semantic"]
			camera_pose[camera_name] = obs[f"{camera_name}::pose"]
		fused_pcd = np.zeros((data_size, pcd_num_points, 6), dtype=np.float32)  # Initialize empty point cloud
		fused_seg = np.zeros((data_size, pcd_num_points), dtype=np.uint32)
		for i in range(data_size):
			if i % 500 == 0:
				print(f"Processing frame {i} of {data_size}")
			rgb_pcd, seg_pcd = [], []
			for camera_name, intrinsics in camera_intrinsics.items():
				pcd = depth_to_pcd(next(camera_depth[camera_name]), camera_pose[camera_name][i], base_link_pose[i], K=intrinsics)
				rgb_pcd.append(
					np.concatenate([next(camera_rgb[camera_name]) / 255.0, pcd], axis=-1).reshape(-1, 6)
				)  # shape (H*W, 6) 
				seg_pcd.append(camera_seg[camera_name][i].reshape(-1)) # shape (H*W) 
			# Fuse all point clouds and downsample
			fused_pcd_all = np.concatenate(rgb_pcd, axis=0)
			fused_pcd[i], sampled_idx = downsample_pcd(fused_pcd_all, pcd_num_points)
			fused_seg[i] = np.concatenate(seg_pcd, axis=0)[sampled_idx]

	return fused_pcd, fused_seg
