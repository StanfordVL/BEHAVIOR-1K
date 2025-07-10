import av
import numpy as np
import omnigibson.utils.transform_utils as T
import open3d as o3d
import torch as th
from torch_cluster import fps
from tqdm import trange, tqdm
from typing import Dict, Tuple


#==============================================
# Depth
#==============================================

MIN_DEPTH = 0.0
MAX_DEPTH = 10.0
DEPTH_SHIFT = 3.5

def quantize_depth(
    depth: np.ndarray, 
    min_depth: float = MIN_DEPTH, 
    max_depth: float = MAX_DEPTH, 
    shift: float=DEPTH_SHIFT
) -> np.ndarray:
    """
    Quantizes depth values to a 14-bit range (0 to 16383) based on the specified min and max depth.
    
    Args:
        depth (np.ndarray): Depth tensor.
        min_depth (float): Minimum depth value.
        max_depth (float): Maximum depth value.
        shift (float): Small value to shift depth to avoid log(0).
    Returns:
        np.ndarray: Quantized depth tensor.
    """
    qmax = (1 << 14) - 1
    log_min = np.log(min_depth + shift)
    log_max = np.log(max_depth + shift)

    log_depth = np.log(depth + shift)
    log_norm = (log_depth - log_min) / (log_max - log_min)
    quantized_depth = np.clip((log_norm * qmax).round(), 0, qmax).astype(np.uint16)

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


def generate_yuv_palette(num_ids: int) -> np.ndarray:
    """
    Generate `num_ids` equidistant YUV colors in the valid YUV space.
    """
    # Y in [16, 235], U, V in [16, 240] for 8-bit YUV standards (BT.601)
    Y_vals = np.linspace(16, 235, int(np.ceil(num_ids ** (1/3))))
    U_vals = np.linspace(16, 240, int(np.ceil(num_ids ** (1/3))))
    V_vals = np.linspace(16, 240, int(np.ceil(num_ids ** (1/3))))

    palette = []
    for y in Y_vals:
        for u in U_vals:
            for v in V_vals:
                palette.append([y, u, v])
                if len(palette) >= num_ids:
                    return np.array(palette, dtype=np.uint8)
    return np.array(palette[:num_ids], dtype=np.uint8)


#==============================================
# Video I/O
#==============================================

def create_video_writer(
    fpath, 
    resolution, 
    codec_name="libx264", 
    rate=30, 
    pix_fmt="yuv420p",
    stream_options=None,
    context_options=None, 
):
    """
    Creates a video writer to write video frames to when playing back the dataset using PyAV

    Args:
        fpath (str): Absolute path that the generated video writer will write to. Should end in .mp4 or .mkv
        resolution (tuple): Resolution of the video frames to write (height, width)
        codec_name (str): Codec to use for the video writer. Default is "libx264"
        rate (int): Frame rate of the video writer. Default is 30
        pix_fmt (str): Pixel format to use for the video writer. Default is "yuv420p"
        stream_options (dict): Additional stream options to pass to the video writer. Default is None
        context_options (dict): Additional context options to pass to the video writer. Default is None
    Returns:
        av.Container: PyAV container object that can be used to write video frames
        av.Stream: PyAV stream object that can be used to write video frames
    """
    assert fpath.endswith(".mp4") or fpath.endswith(".mkv"), f"Video writer fpath must end with .mp4 or .mkv! Got: {fpath}"
    container = av.open(fpath, mode='w')
    stream = container.add_stream(codec_name, rate=rate)
    stream.height = resolution[0]
    stream.width = resolution[1]
    stream.pix_fmt = pix_fmt
    if stream_options is not None:
        stream.options = stream_options
    if context_options is not None:
        stream.codec_context.options = context_options
    return container, stream

def write_video(obs, video_writer, mode="rgb", **kwargs) -> None:
    """
    Writes videos to the specified video writers using the current trajectory history

    Args:
        obs (torch.Tensor): Observation tensor
        video_writer (container, stream): PyAV container and stream objects to write video frames to
        mode (str): Mode to write video frames to. Only "rgb", "depth" and "seg" are supported.
        kwargs (dict): Additional keyword arguments to pass to the video writer.
    """
    container, stream = video_writer
    if mode == "rgb":
        for frame in obs[:]:
            frame = av.VideoFrame.from_ndarray(frame[..., :3], format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
    elif mode == "depth":
        quantized_depth = quantize_depth(obs[:])
        for frame in quantized_depth:
            frame = av.VideoFrame.from_ndarray(frame, format='gray16le')
            for packet in stream.encode(frame):
                container.mux(packet)
    elif mode == "seg":
        seg_ids = kwargs["seg_ids"]
        palette = th.from_numpy(generate_yuv_palette(len(seg_ids)))
        # Vectorized mapping - much faster than loop
        max_id = seg_ids.max().item() + 1
        instance_id_to_idx = th.full((max_id,), -1, dtype=th.long)
        instance_id_to_idx[seg_ids] = th.arange(len(seg_ids))
        seg_colored = palette[instance_id_to_idx[obs[:]]].numpy()
        for frame in seg_colored:
            frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
    else:
        raise ValueError(f"Unsupported video mode: {mode}.")


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
    for frame in tqdm(container.decode(stream)):
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
    for frame in tqdm(container.decode(stream)):
        # Decode Y (luma) channel only; YUV420 → grayscale image
        frame_gray16 = frame.reformat(format='gray16le').to_ndarray()
        frames.append(th.from_numpy(frame_gray16).unsqueeze(0))  # (1, H, W)
    
    container.close()
    video = th.cat(frames, dim=0)  # (T, H, W)
    depth = dequantize_depth(video, min_depth=min_depth, max_depth=max_depth, shift=shift)
    return depth


def load_seg_video(path: str, id_list: th.Tensor, device="cpu") -> th.Tensor:
    container = av.open(path)
    stream = container.streams.video[0]

    palette = th.from_numpy(generate_yuv_palette(len(id_list))).to(device).float()
    seg_original_keys = id_list.to(device) 
    
    frames = []
    for frame in tqdm(container.decode(stream)):
        rgb = th.tensor(frame.to_ndarray(format="rgb24"), dtype=th.float32, device=device)  # (H, W, 3)
        # For each rgb pixel, find the index of the nearest color in the equidistant bins
        rgb_flat = rgb.reshape(-1, 3)  # (H*W, 3)
        distances = th.cdist(rgb_flat[None, :, :], palette[None, :, :], p=2)[0]  # (H*W, N_ids)
        ids = th.argmin(distances, dim=-1)  # (H*W,)
        ids = seg_original_keys[ids].reshape((rgb.shape[0], rgb.shape[1]))  # (H, W)
        frames.append(ids.unsqueeze(0).cpu())  # (1, H, W)

    container.close()
    return th.cat(frames, dim=0)  # (T, H, W)


#==============================================
# Point Cloud
#==============================================

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
    world_to_robot_tf = T.pose2mat((base_link_pose[:, :3], base_link_pose[:, 3:])).to(device=device)  # (B, 4, 4)
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


def downsample_pcd(color_pcd, num_points) -> Tuple[th.Tensor, th.Tensor]:
    """
    Downsample point clouds with true batch FPS processing.
    
    Args:
        color_pcd: (B, N, 6) point cloud tensor [rgb, xyz] for each batch
        num_points: target number of points
    Returns:
        color_pcd: (B, num_points, 6) downsampled point cloud
        sampled_idx: (B, num_points) sampled indices
    """
    B, N, C = color_pcd.shape
    device = color_pcd.device
    
    # Initialize output tensors
    output_pcd = th.zeros(B, num_points, C, device=device, dtype=color_pcd.dtype)
    output_idx = th.zeros(B, num_points, device=device, dtype=th.long)
    
    if N > num_points:
        # True batch FPS - process all batches together
        xyz = color_pcd[:, :, 3:6].contiguous()  # (B, N, 3)
        xyz_flat = xyz.view(-1, 3)  # (B*N, 3)
        # Create batch indices for all points
        batch_indices = th.arange(B, device=device).repeat_interleave(N)  # (B*N,)
        # Single FPS call for all batches
        idx_flat = fps(xyz_flat, batch_indices, ratio=float(num_points) / N, random_start=True)
        # Vectorized post-processing
        local_idx = idx_flat % N   # Local index within each batch
        batch_idx = idx_flat // N  # Which batch each index belongs to
        for b in trange(B):
            batch_mask = batch_idx == b
            if batch_mask.sum() > 0:
                batch_local_indices = local_idx[batch_mask][:num_points]
                output_pcd[b, :len(batch_local_indices)] = color_pcd[b][batch_local_indices]
                output_idx[b, :len(batch_local_indices)] = batch_local_indices
            
    else:
        pad_num = num_points - N
        random_idx = th.randint(0, N, (B, pad_num), device=device)  # (B, pad_num)
        seq_idx = th.arange(N, device=device).unsqueeze(0).expand(B, N)  # (B, N)
        full_idx = th.cat([seq_idx, random_idx], dim=1)  # (B, num_points)
        batch_indices = th.arange(B, device=device).unsqueeze(1).expand(B, num_points)  # (B, num_points)
        output_pcd = color_pcd[batch_indices, full_idx]  # (B, num_points, C)
        output_idx = full_idx
    
    return output_pcd, output_idx


def process_fused_point_cloud(
    obs: dict,
    robot_name: str, 
    camera_intrinsics: Dict[str, th.Tensor],
    pcd_num_points: int = 4096
) -> Tuple[th.Tensor, th.Tensor]:
    base_link_pose = obs[f"{robot_name}::robot_base_link_pose"]
    camera_depth, camera_rgb, camera_seg, camera_pose = dict(), dict(), dict(), dict()
    for camera_name in camera_intrinsics.keys():
        camera_depth[camera_name] = obs[f"{camera_name}::depth_linear"]
        camera_rgb[camera_name] = obs[f"{camera_name}::rgb"]
        camera_seg[camera_name] = obs[f"{camera_name}::seg_semantic"]
        camera_pose[camera_name] = obs[f"{camera_name}::pose"]
    rgb_pcd, seg_pcd = [], []
    for camera_name, intrinsics in camera_intrinsics.items():
        pcd = depth_to_pcd(
            camera_depth[camera_name], 
            camera_pose[camera_name], 
            base_link_pose, 
            intrinsics
        )
        num_points = pcd.shape[0]
        rgb_pcd.append(
            th.cat([camera_rgb[camera_name][..., :3] / 255.0, pcd], dim=-1).reshape(num_points, -1, 6)
        )  # shape (N, H*W, 6) 
        seg_pcd.append(camera_seg[camera_name].reshape(num_points, -1)) # shape (N, H*W)
    # Fuse all point clouds and downsample
    fused_pcd_all = th.cat(rgb_pcd, dim=1)
    fused_pcd, sampled_idx = downsample_pcd(fused_pcd_all, pcd_num_points)
    fused_pcd = fused_pcd.float().to(device="cpu")
    sampled_idx = sampled_idx.to(device="cpu")
    fused_seg = th.gather(th.cat(seg_pcd, dim=1), 1, sampled_idx)

    return fused_pcd, fused_seg


def color_pcd_vis(color_pcd):
    # visualize with open3D
    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(color_pcd[:, :3])
    pcd.points = o3d.utility.Vector3dVector(color_pcd[:, 3:])
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, axis])
    print("number points", color_pcd.shape[0])