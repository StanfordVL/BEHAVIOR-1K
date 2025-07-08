import av
# from torch_cluster import fps
import numpy as np
import omnigibson.utils.transform_utils as T
import open3d as o3d
import torch as th
from scipy.spatial.transform import Rotation as R
from sympy import mod_inverse
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


PRIME = 2654435761  # Knuth's multiplicative hash
MOD = 2**24
PRIME_INV = int(mod_inverse(PRIME, MOD))

def id_to_rgb_scrambled(id_map: np.ndarray) -> np.ndarray:
    """Map uint32 instance IDs to uint8 RGB via reversible hash."""
    id_map = id_map.astype(np.uint32)
    hashed = (id_map * PRIME) % MOD
    r = (hashed >> 16) & 0xFF
    g = (hashed >> 8) & 0xFF
    b = hashed & 0xFF
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def rgb_to_id_scrambled(rgb: np.ndarray) -> np.ndarray:
    """Recover uint32 instance IDs from RGB."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    hashed = ((r.astype(np.uint32) << 16) |
              (g.astype(np.uint32) << 8) |
              b.astype(np.uint32))
    ids = (hashed * PRIME_INV) % MOD
    return ids.astype(np.uint32)


class VideoLoader:
    def __init__(self, path: str, type: str = "rgb", streaming: bool = False):
        """
        Initialize VideoLoader for loading RGB or depth videos.
        
        Args:
            path (str): Path to the video file
            video_type (str): Either "rgb", "depth" or "seg"
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
            elif type == "seg":
                self.frames = load_seg_video(path)
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
                    elif self.type == "seg":
                        rgb_id = frame.to_ndarray(format="rgb24")
                        yield th.from_numpy(rgb_to_id_scrambled(rgb_id))
    
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
    """
    Load RGB video with robust frame extraction.
    
    Args:
        path (str): Path to the video file
    Returns:
        th.Tensor: (T, H, W, 3) RGB video tensor
    """
    container = av.open(path)
    stream = container.streams.video[0]
    
    frames = []
    for frame in container.decode(stream):
        rgb = frame.to_ndarray(format="rgb24")  # (H, W, 3), dtype=uint8
        rgb = th.from_numpy(rgb).unsqueeze(0)  # (1, H, W, 3)
        frames.append(rgb)
    
    container.close()
    
    if not frames:
        raise ValueError(f"No frames found in video: {path}")
    
    video = th.cat(frames, dim=0)  # (T, H, W, 3)
    return video


def load_depth_video(path: str, min_depth: float=MIN_DEPTH, max_depth: float=MAX_DEPTH, shift: float=DEPTH_SHIFT) -> th.Tensor:
    container = av.open(path)
    stream = container.streams.video[0]
    
    frames = []
    for frame in container.decode(stream):
        # Decode Y (luma) channel only; YUV420 → grayscale image
        frame_gray16 = frame.reformat(format='gray16le').to_ndarray()
        frames.append(th.from_numpy(frame_gray16).unsqueeze(0))  # (1, H, W)
    
    container.close()
    video = th.cat(frames, dim=0)  # (T, H, W)
    depth = dequantize_depth(video, min_depth=min_depth, max_depth=max_depth, shift=shift)
    return depth


def load_seg_video(path: str) -> th.Tensor:
    container = av.open(path)
    stream = container.streams.video[0]
    
    frames = []
    for frame in container.decode(stream):
        rgb = frame.to_ndarray(format="rgb24")
        id = rgb_to_id_scrambled(rgb)
        frames.append(th.from_numpy(id).unsqueeze(0))  # (1, H, W)
    
    container.close()
    return th.cat(frames, dim=0)  # (T, H, W)


def color_pcd_vis(color_pcd):
    # visualize with open3D
    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(color_pcd[:, :3])
    pcd.points = o3d.utility.Vector3dVector(color_pcd[:, 3:])
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, axis])
    print("number points", color_pcd.shape[0])


def depth_to_pcd(
    depth: th.Tensor,  # (B, H, W) or (H, W)
    pose: th.Tensor,   # (B, 7) or (7,)
    base_link_pose: th.Tensor,  # (B, 7) or (7,)
    K: th.Tensor,      # (B, 3, 3) or (3, 3)
    max_depth=20,
) -> th.Tensor:
    """
    Convert depth images to point clouds with batch processing support.
    Args:
        depth: (B, H, W) or (H, W) depth tensor
        pose: (B, 7) or (7,) camera pose tensor [pos, quat]
        base_link_pose: (B, 7) or (7,) robot base pose tensor [pos, quat]
        K: (B, 3, 3) or (3, 3) camera intrinsics tensor
        max_depth: maximum depth value to filter
    Returns:
        pc: (B, H, W, 3) or (H, W, 3) point cloud tensor
    """
    # Handle single vs batch inputs
    is_batch = len(depth.shape) == 3
    if not is_batch:
        depth = depth.unsqueeze(0)
        pose = pose.unsqueeze(0)
        base_link_pose = base_link_pose.unsqueeze(0)
        K = K.unsqueeze(0)
    B, H, W = depth.shape
    device = depth.device

    # Get poses and convert to transformation matrices
    pos = pose[:, :3]  # (B, 3)
    quat = pose[:, 3:]  # (B, 4)
    rot = T.quat2mat(quat)  # (B, 3, 3)
    rot_add = T.euler2mat(th.tensor([np.pi, 0, 0], device=device))  # (3, 3)
    rot_matrix = th.matmul(rot, rot_add)  # (B, 3, 3)

    # Create world_to_cam transformation matrices
    world_to_cam_tf = th.eye(4, device=device).unsqueeze(0).expand(B, 4, 4).clone()
    world_to_cam_tf[:, :3, :3] = rot_matrix
    world_to_cam_tf[:, :3, 3] = pos

    # Filter depth
    mask = depth > max_depth
    depth = depth.clone()
    depth[mask] = 0

    # Create pixel coordinates
    y, x = th.meshgrid(th.arange(H, device=device), th.arange(W, device=device), indexing="ij")
    u = x.unsqueeze(0).expand(B, H, W)
    v = y.unsqueeze(0).expand(B, H, W)
    uv = th.stack([u, v, th.ones_like(u)], dim=-1).float()  # (B, H, W, 3)

    # Compute inverse of camera intrinsics
    Kinv = th.linalg.inv(K)  # (B, 3, 3)

    # Convert to point cloud
    pc = depth.unsqueeze(-1) * th.matmul(uv, Kinv.transpose(-2, -1))  # (B, H, W, 3)

    # Add homogeneous coordinate
    pc_homo = th.cat([pc, th.ones_like(pc[..., :1])], dim=-1)  # (B, H, W, 4)

    # Transform to robot base frame
    world_to_robot_tf = T.pose2mat((base_link_pose[:, :3], base_link_pose[:, 3:]))  # (B, 4, 4)
    robot_to_world_tf = th.linalg.inv(world_to_robot_tf)

    # Apply transformations
    pc_homo_flat = pc_homo.view(B, -1, 4)  # (B, H*W, 4)
    pc_cam = th.matmul(pc_homo_flat, world_to_cam_tf.transpose(-2, -1))  # (B, H*W, 4)
    pc_transformed = th.matmul(pc_cam, robot_to_world_tf.transpose(-2, -1))  # (B, H*W, 4)
    pc_transformed = pc_transformed[..., :3].view(B, H, W, 3)  # (B, H, W, 3)

    # Return in original format
    if not is_batch:
        pc_transformed = pc_transformed.squeeze(0)
    return pc_transformed


def downsample_pcd(color_pcd, num_points) -> Tuple[np.ndarray, np.ndarray]:
    """
    If the current number of point cloud is smaller than num_points, will downsample it
    Otherwise, randomly sample current points to reach num_points
    """
    N = color_pcd.shape[0]
    device = color_pcd.device
    if color_pcd.shape[0] > num_points:
        # Use FPS on xyz
        xyz = color_pcd[:, 3:6].contiguous()
        # fps expects (N, 3) and returns indices
        # ratio = num_points / N
        batch = th.zeros(N, dtype=th.long, device=device)  # single batch
        idx = fps(xyz, batch, ratio=float(num_points) / N, random_start=True)
        if idx.shape[0] > num_points:
            idx = idx[:num_points]
        sampled = color_pcd[idx]
    else:
        # Pad with random samples
        pad_num = num_points - N
        idx = th.cat([th.arange(N, device=device), th.randint(0, N, (pad_num,), device=device)])
        sampled = color_pcd[idx]
    return sampled, idx


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
