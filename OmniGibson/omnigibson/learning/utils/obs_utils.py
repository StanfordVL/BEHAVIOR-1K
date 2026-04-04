import av
import math
import numpy as np
import omnigibson.utils.transform_utils as T
import torch as th
from av.container import Container
from av.stream import Stream
from tqdm import trange
from typing import Dict, Optional, Tuple, List
from omnigibson.utils.constants import semantic_class_name_to_id
from omnigibson.utils.ui_utils import create_module_logger

logger = create_module_logger("obs_utils")

try:
    from torch_cluster import fps
except ImportError:
    fps = None

# ==============================================
# Depth
# ==============================================

MIN_DEPTH = 0.01
MAX_DEPTH = 10.0
DEPTH_SHIFT = 3.5


def quantize_depth(
    depth: np.ndarray | th.Tensor,
    min_depth: float = MIN_DEPTH,
    max_depth: float = MAX_DEPTH,
    shift: float = DEPTH_SHIFT,
) -> np.ndarray:
    """
    Quantizes depth values to a 12-bit range (0 to 4096) based on the specified min and max depth.

    Args:
        depth (np.ndarray or th.tensor): Depth tensor.
        min_depth (float): Minimum depth value.
        max_depth (float): Maximum depth value.
        shift (float): Small value to shift depth to avoid log(0).
    Returns:
        np.ndarray: Quantized depth tensor.
    """
    # convert to numpy if input is torch tensor
    if isinstance(depth, th.Tensor):
        depth = depth.cpu().numpy()
    qmax = (1 << 12) - 1
    log_min = math.log(min_depth + shift)
    log_max = math.log(max_depth + shift)

    log_depth = np.log(depth + shift)
    log_norm = (log_depth - log_min) / (log_max - log_min)
    quantized_depth = np.clip((log_norm * qmax).round(), 0, qmax).astype(np.uint16)
    return quantized_depth


def dequantize_depth(
    quantized_depth: np.ndarray | th.Tensor,
    min_depth: float = MIN_DEPTH,
    max_depth: float = MAX_DEPTH,
    shift: float = DEPTH_SHIFT,
) -> np.ndarray | th.Tensor:
    """
    Dequantizes a 12-bit depth tensor back to the original depth values.

    Args:
        quantized_depth (np.ndarray or th.tensor): Quantized depth tensor.
        min_depth (float): Minimum depth value.
        max_depth (float): Maximum depth value.
        shift (float): Small value to shift depth to avoid log(0).
    Returns:
        np.ndarray or th.tensor: Dequantized depth tensor.
    """
    backend = np if isinstance(quantized_depth, np.ndarray) else th
    qmax = (1 << 12) - 1
    log_min = math.log(min_depth + shift)
    log_max = math.log(max_depth + shift)

    log_norm = quantized_depth / qmax
    log_depth = log_norm * (log_max - log_min) + log_min
    depth = backend.clip(backend.exp(log_depth) - shift, min_depth, max_depth)

    return depth


def encode_depth_frame(depth: np.ndarray | th.Tensor) -> av.VideoFrame:
    """
    Encodes a depth frame by quantizing it to a 12-bit range and then packing it into YUV420p12le format.

    Args:
        depth (np.ndarray or th.tensor): Depth tensor of shape (H, W) with float values.
    Returns:
        av.VideoFrame: Encoded depth frame in YUV420p12le format, where the Y plane contains the quantized depth values.
    """
    quantized_depth = quantize_depth(depth)
    H, W = quantized_depth.shape[:2]
    # Write depth into the Y plane; set U/V to neutral chroma.
    frame = av.VideoFrame(width=W, height=H, format="yuv420p12le")
    frame.planes[0].update(quantized_depth.tobytes())
    # Bit depth of 12 → neutral = 2^11 = 2048.
    neutral_chroma = np.full((H // 2, W // 2), 2048, dtype=np.uint16)
    frame.planes[1].update(neutral_chroma.tobytes())
    frame.planes[2].update(neutral_chroma.tobytes())
    return frame


def decode_depth_frame(frame: np.ndarray) -> np.ndarray:
    """
    Decodes a depth frame by extracting the quantized depth from the Y plane and then dequantizing it back to float values.
    Args:
        frame (np.ndarray): Encoded depth tensor of shape (H, W) with uint16 values.
    Returns:
        np.ndarray: Decoded depth tensor of shape (H, W) with float values.
    """
    quantized_depth = frame.astype(np.uint16)
    depth = dequantize_depth(quantized_depth)
    return depth


# ==============================================
# Video I/O
# ==============================================


def create_video_writer(
    fpath,
    resolution,
    codec_name="libx264",
    rate=30,
    pix_fmt="yuv420p",
    stream_options=None,
    context_options=None,
) -> Tuple[Container, Stream]:
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
    assert fpath.endswith(".mp4") or fpath.endswith(
        ".mkv"
    ), f"Video writer fpath must end with .mp4 or .mkv! Got: {fpath}"
    container = av.open(fpath, mode="w")
    stream = container.add_stream(codec_name, rate=rate)
    stream.height = resolution[0]
    stream.width = resolution[1]
    stream.pix_fmt = pix_fmt
    if stream_options is not None:
        stream.options = stream_options
    if context_options is not None:
        stream.codec_context.options = context_options
    return container, stream


def write_video(obs, video_writer, mode="rgb", batch_size=None, **kwargs) -> None:
    """
    Writes videos to the specified video writers using the current trajectory history

    Args:
        obs (np.ndarray): Observation data
        video_writer (container, stream): PyAV container and stream objects to write video frames to
        mode (str): Mode to write video frames to. Only "rgb", "depth" and "seg" are supported.
        batch_size (int): Batch size to write video frames to. If None, write video frames to the entire video.
        kwargs (dict): Additional keyword arguments to pass to the video writer.
    """
    container, stream = video_writer
    batch_size = batch_size or obs.shape[0]
    if mode == "rgb":
        for i in range(0, obs.shape[0], batch_size):
            for frame in obs[i : i + batch_size]:
                frame = av.VideoFrame.from_ndarray(frame[..., :3], format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
    elif mode == "depth" or mode == "depth_linear":
        for i in range(0, obs.shape[0], batch_size):
            quantized_depth = quantize_depth(obs[i : i + batch_size])
            for frame in quantized_depth:
                frame = av.VideoFrame.from_ndarray(frame, format="gray16le")
                for packet in stream.encode(frame):
                    container.mux(packet)
    else:
        raise ValueError(f"Unsupported video mode: {mode}.")


# ==============================================
# Point Cloud
# ==============================================


def depth_to_pcd(
    depth: th.Tensor,  # (B, [T], H, W)
    rel_pose: th.Tensor,  # (B, [T], 7) relative pose from camera to base [pos, quat]
    K: th.Tensor,  # (3, 3)
) -> th.Tensor:
    """
    Convert depth images to point clouds with batch processing support.
    Args:
        depth: (B, H, W) depth tensor
        rel_pose: (B, 7) relative pose from camera to base tensor [pos, quat]
        K: (3, 3) camera intrinsics tensor
        max_depth: maximum depth value to filter
    Returns:
        pc: (B, H, W, 3) point cloud tensor in base frame
    """
    original_shape = depth.shape
    depth = depth.view(-1, original_shape[-2], original_shape[-1])  # (B, H, W)
    rel_pose = rel_pose.view(-1, 7)  # (B, 7)
    B, H, W = depth.shape
    device = depth.device

    # Get relative pose and convert to transformation matrix
    rel_pos = rel_pose[:, :3]  # (B, 3)
    rel_quat = rel_pose[:, 3:]  # (B, 4)
    rel_rot = T.quat2mat(rel_quat)  # (B, 3, 3)

    # Add camera coordinate system adjustment (180 degree rotation around X-axis)
    rot_add = T.euler2mat(th.tensor([np.pi, 0, 0], device=device))  # (3, 3)
    rel_rot_matrix = th.matmul(rel_rot, rot_add)  # (B, 3, 3)

    # Create camera_to_base transformation matrix
    camera_to_base_tf = th.eye(4, device=device).unsqueeze(0).expand(B, 4, 4).clone()
    camera_to_base_tf[:, :3, :3] = rel_rot_matrix
    camera_to_base_tf[:, :3, 3] = rel_pos

    # Create pixel coordinates
    y, x = th.meshgrid(th.arange(H, device=device), th.arange(W, device=device), indexing="ij")
    u = x.unsqueeze(0).expand(B, H, W)
    v = y.unsqueeze(0).expand(B, H, W)
    uv = th.stack([u, v, th.ones_like(u)], dim=-1).float()  # (B, H, W, 3)

    # Compute inverse of camera intrinsics
    Kinv = th.linalg.inv(K).to(device)  # (3, 3)

    # Convert to point cloud in camera frame
    pc_camera = depth.unsqueeze(-1) * th.matmul(uv, Kinv.transpose(-2, -1))  # (B, H, W, 3)

    # Add homogeneous coordinate
    pc_camera_homo = th.cat([pc_camera, th.ones_like(pc_camera[..., :1])], dim=-1)  # (B, H, W, 4)

    # Transform from camera frame to base frame
    pc_camera_homo_flat = pc_camera_homo.view(B, -1, 4)  # (B, H*W, 4)
    pc_base = th.matmul(pc_camera_homo_flat, camera_to_base_tf.transpose(-2, -1))  # (B, H*W, 4)
    pc_base = pc_base[..., :3].view(*original_shape, 3)  # (B, [T], H, W, 3)

    return pc_base


def downsample_pcd(color_pcd, num_points, use_fps=True) -> Tuple[th.Tensor, th.Tensor]:
    """
    Downsample point clouds with batch FPS processing or random sampling.

    Args:
        color_pcd: (B, [T], N, 6) point cloud tensor [rgb, xyz] for each batch
        num_points: target number of points
    Returns:
        color_pcd: (B, num_points, 6) downsampled point cloud
        sampled_idx: (B, num_points) sampled indices
    """
    original_shape = color_pcd.shape
    color_pcd = color_pcd.view(-1, original_shape[-2], original_shape[-1])  # (B, N, 6)
    B, N, C = color_pcd.shape
    device = color_pcd.device

    if N > num_points:
        if use_fps:
            # Initialize output tensors
            output_pcd = th.zeros(B, num_points, C, device=device, dtype=color_pcd.dtype)
            output_idx = th.zeros(B, num_points, device=device, dtype=th.long)
            # True batch FPS - process all batches together
            xyz = color_pcd[:, :, 3:6].contiguous()  # (B, N, 3)
            xyz_flat = xyz.view(-1, 3)  # (B*N, 3)
            # Create batch indices for all points
            batch_indices = th.arange(B, device=device).repeat_interleave(N)  # (B*N,)
            # Single FPS call for all batches
            assert (
                fps is not None
            ), "torch_cluster.fps is not available! Please make sure you have omnigibson setup with eval dependencies."
            idx_flat = fps(xyz_flat, batch_indices, ratio=float(num_points) / N, random_start=True)
            # Vectorized post-processing
            batch_idx = idx_flat // N  # Which batch each index belongs to
            local_idx = idx_flat % N  # Local index within each batch
            for b in range(B):
                batch_mask = batch_idx == b
                if batch_mask.sum() > 0:
                    batch_local_indices = local_idx[batch_mask][:num_points]
                    output_pcd[b, : len(batch_local_indices)] = color_pcd[b][batch_local_indices]
                    output_idx[b, : len(batch_local_indices)] = batch_local_indices
        else:
            # Randomly sample num_points indices without replacement for each batch
            output_idx = th.stack(
                [th.randperm(N, device=device)[:num_points] for _ in range(B)], dim=0
            )  # (B, num_points)
            # Use proper batch indexing
            batch_indices = th.arange(B, device=device).unsqueeze(1).expand(B, num_points)
            output_pcd = color_pcd[batch_indices, output_idx]  # (B, num_points, C)
    else:
        pad_num = num_points - N
        random_idx = th.randint(0, N, (B, pad_num), device=device)  # (B, pad_num)
        seq_idx = th.arange(N, device=device).unsqueeze(0).expand(B, N)  # (B, N)
        full_idx = th.cat([seq_idx, random_idx], dim=1)  # (B, num_points)
        batch_indices = th.arange(B, device=device).unsqueeze(1).expand(B, num_points)  # (B, num_points)
        output_pcd = color_pcd[batch_indices, full_idx]  # (B, num_points, C)
        output_idx = full_idx

    output_pcd = output_pcd.view(*original_shape[:-2], num_points, C)  # (B, [T], num_points, 6)
    return output_pcd, output_idx


def process_fused_point_cloud(
    obs: dict,
    camera_intrinsics: Dict[str, th.Tensor],
    pcd_range: Tuple[float, float, float, float, float, float],  # x_min, x_max, y_min, y_max, z_min, z_max
    pcd_num_points: Optional[int] = None,
    use_fps: bool = True,
    verbose: bool = False,
) -> Tuple[th.Tensor, Optional[th.Tensor]]:
    """
    Given a dictionary of observations, process the fused point cloud from all cameras and return the final point cloud tensor in robot base frame.
    Args:
        obs (dict): Dictionary of observations containing point cloud data from different cameras.
        camera_intrinsics (Dict[str, th.Tensor]): Dictionary of camera intrinsics for each camera.
        pcd_range (Tuple[float, float, float, float, float, float]): Range of the point cloud to filter [x_min, x_max, y_min, y_max, z_min, z_max].
        pcd_num_points (Optional[int]): Number of points to sample from the point cloud. If None, no downsampling is performed.
        use_fps (bool): Whether to use farthest point sampling for point cloud downsampling. Default is True.
        verbose (bool): Whether to print verbose output during processing. Default is False.
    """
    if verbose:
        print("Processing fused point cloud from observations...")
    rgb_pcd = []
    for idx, (camera_name, intrinsics) in enumerate(camera_intrinsics.items()):
        if f"{camera_name}::pointcloud" in obs:
            # should already be in robot base frame, see BaseRobot._get_obs()
            pcd = obs[f"{camera_name}::pointcloud"]
            rgb_pcd.append(th.cat([pcd[:, :3] / 255.0, pcd[:, 3:]], dim=-1))
        else:
            # need to convert from depth to point cloud in robot base frame
            pcd = depth_to_pcd(
                obs[f"{camera_name}::depth_linear"], obs["cam_rel_poses"][..., 7 * idx : 7 * idx + 7], intrinsics
            )
            rgb_pcd.append(
                th.cat([obs[f"{camera_name}::rgb"][..., :3] / 255.0, pcd], dim=-1).flatten(-3, -2)
            )  # shape (B, [T], H*W, 6)
    # Fuse all point clouds together
    fused_pcd_all = th.cat(rgb_pcd, dim=-2).to(device="cuda")
    # Now, clip the point cloud to the specified range
    x_min, x_max, y_min, y_max, z_min, z_max = pcd_range
    mask = (
        (fused_pcd_all[..., 3] >= x_min)
        & (fused_pcd_all[..., 3] <= x_max)
        & (fused_pcd_all[..., 4] >= y_min)
        & (fused_pcd_all[..., 4] <= y_max)
        & (fused_pcd_all[..., 5] >= z_min)
        & (fused_pcd_all[..., 5] <= z_max)
    )
    fused_pcd_all[~mask] = 0.0
    # Now, downsample the point cloud if needed
    if pcd_num_points is not None:
        if verbose:
            print(
                f"Downsampling point cloud to {pcd_num_points} points using {'FPS' if use_fps else 'random sampling'}"
            )
        fused_pcd = downsample_pcd(fused_pcd_all, pcd_num_points, use_fps=use_fps)[0]
        fused_pcd = fused_pcd.float()
    else:
        fused_pcd = fused_pcd_all.float()

    return fused_pcd


def color_pcd_vis(color_pcd: np.ndarray):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])

    # Rotation matrices
    Rz_90 = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, np.pi / 2])  # 90 deg about z
    Rx_m90 = o3d.geometry.get_rotation_matrix_from_axis_angle([-np.pi / 2, 0, 0])  # -90 deg about x

    # visualize with open3D
    if color_pcd.ndim == 2:
        colors = color_pcd[:, :3]
        points = color_pcd[:, 3:]
        points = (points @ Rz_90.T) @ Rx_m90.T
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.visualization.draw_geometries([pcd, axis])
        print("number points", color_pcd.shape[0])
    else:
        # realtime streaming
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for i in trange(color_pcd.shape[0]):
            colors = color_pcd[i, :, :3]
            points = color_pcd[i, :, 3:]
            points = (points @ Rz_90.T) @ Rx_m90.T
            pcd.colors = o3d.utility.Vector3dVector(colors)
            pcd.points = o3d.utility.Vector3dVector(points)
            vis.clear_geometries()
            vis.add_geometry(pcd)
            vis.add_geometry(axis)
            vis.poll_events()
            vis.update_renderer()
        vis.destroy_window()


# ==============================================
# Segmentation
# ==============================================


def instance_id_to_instance(
    obs: th.Tensor, instance_id_mapping: Dict[int, str], unique_ins_ids: List[int]
) -> Tuple[th.Tensor, Dict[int, str]]:
    """
    Instance_id segmentation map each unique visual meshes of objects (e.g. /World/scene_name/object_name/visual_mesh_0)
    This function merges all visual meshes of the same object instance to a single instance id.
    Args:
        obs (th.Tensor): (N, H, W) instance_id segmentation
        instance_id_mapping (Dict[int, str]): Dict mapping instance_id ids to instance names
    Returns:
        instance_seg (th.Tensor): (N, H, W) instance segmentation
        instance_mapping (Dict[int, str]): Dict mapping instance ids to instance names
    """
    # trim the instance ids mapping to the valid instance ids
    instance_id_mapping = {k: v for k, v in instance_id_mapping.items() if k in unique_ins_ids}
    # extract the actual instance name, which is located at /World/scene_name/object_name
    # Note that 0, 1 are special cases for background and unlabelled, respectivelly
    instance_id_to_instance = {k: v.split("/")[3] for k, v in instance_id_mapping.items() if k not in [0, 1]}
    # get all unique instance names
    instance_names = set(instance_id_to_instance.values())
    # construct a new instance mapping from instance names to instance ids
    instance_mapping = {0: "background", 1: "unlabelled"}
    instance_mapping.update({k + 2: v for k, v in enumerate(instance_names)})  # {i: object_name}
    reversed_instance_mapping = {v: k for k, v in instance_mapping.items()}  # {object_name: i}
    # put back the background and unlabelled
    instance_id_to_instance.update({0: "background", 1: "unlabelled"})
    # Now, construct the instance segmentation
    instance_seg = th.zeros_like(obs)
    # Create lookup tensor for faster indexing
    lookup = th.full((max(unique_ins_ids) + 1,), -1, dtype=th.long, device=obs.device)
    for instance_id in unique_ins_ids:
        lookup[instance_id] = reversed_instance_mapping[instance_id_to_instance[instance_id]]
    instance_seg = lookup[obs]
    # Note that now the returned instance mapping will be unique (i.e. no unused instance ids)
    return instance_seg, instance_mapping


def instance_to_semantic(
    obs, instance_mapping: Dict[int, str], unique_ins_ids: List[int], is_instance_id: bool = True
) -> th.Tensor:
    """
    Convert instance / instance id segmentation to semantic segmentation.
    Args:
        obs (th.Tensor): (N, H, W) instance / instance_id segmentation
        instance_mapping (Dict[int, str]): Dict mapping instance IDs to instance names
        unique_ins_ids (List[int]): List of unique instance IDs
        is_instance_id (bool): Whether the input is instance id segmentation
    Returns:
        semantic_seg (th.Tensor): (N, H, W) semantic segmentation
    """
    # trim the instance ids mapping to the valid instance ids
    instance_mapping = {k: v for k, v in instance_mapping.items() if k in unique_ins_ids}
    # we remove 0: background, 1: unlabelled from the instance mapping for now
    instance_mapping.pop(0, None)
    instance_mapping.pop(1, None)
    # get semantic name from instance mapping
    if is_instance_id:
        instance_mapping = {k: v.split("/")[3] for k, v in instance_mapping.items()}
    # instance names are of category_model_id, so we extract the category name
    # with the exception of robot. We assume that robot is the only instance with "robot" in the name
    instance_to_semantic = {}
    for k, v in instance_mapping.items():
        if "robot" in v:
            instance_to_semantic[k] = "agent"
        else:
            instance_to_semantic[k] = v.rsplit("_", 2)[0]
    instance_to_semantic.update({0: "background", 1: "unlabelled"})
    # Now, construct the semantic segmentation
    semantic_seg = th.zeros_like(obs)
    semantic_name_to_id = semantic_class_name_to_id()
    # Create lookup tensor for faster indexing
    lookup = th.full((max(unique_ins_ids) + 1,), -1, dtype=th.long, device=obs.device)
    for instance_id in instance_mapping:
        lookup[instance_id] = semantic_name_to_id[instance_to_semantic[instance_id]]
    semantic_seg = lookup[obs]
    return semantic_seg


def instance_to_bbox(
    obs: th.Tensor, instance_mapping: Dict[int, str], unique_ins_ids: List[int]
) -> List[List[Tuple[int, int, int, int, int]]]:
    """
    Convert instance segmentation to bounding boxes.

    Args:
        obs (th.Tensor): (N, H, W) tensor of instance IDs
        instance_mapping (Dict[int, str]): Dict mapping instance IDs to instance names
            Note: this does not need to include all instance IDs, only the ones that we want to generate bbox for
        unique_ins_ids (List[int]): List of unique instance IDs
    Returns:
        List of N lists, each containing tuples (x_min, y_min, x_max, y_max, instance_id) for each instance
    """
    if len(obs.shape) == 2:
        obs = obs.unsqueeze(0)  # Add batch dimension if single frame
    N = obs.shape[0]
    bboxes = [[] for _ in range(N)]
    valid_ids = [id for id in instance_mapping if id in unique_ins_ids]
    for instance_id in valid_ids:
        # Create mask for this instance
        mask = obs == instance_id  # (N, H, W)
        # Find bounding boxes for each frame
        for n in range(N):
            frame_mask = mask[n]  # (H, W)
            if not frame_mask.any():
                continue
            # Find non-zero indices (where instance exists)
            y_coords, x_coords = th.where(frame_mask)
            if len(y_coords) == 0:
                continue
            # Calculate bounding box
            x_min = x_coords.min().item()
            x_max = x_coords.max().item()
            y_min = y_coords.min().item()
            y_max = y_coords.max().item()
            bboxes[n].append((x_min, y_min, x_max, y_max, instance_id))

    return bboxes
