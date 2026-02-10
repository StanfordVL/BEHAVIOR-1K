import av
import cv2
import h5py
import json
import os
import numpy as np
import omnigibson.utils.transform_utils as T
import pandas as pd
import torch as th
import torch.nn.functional as F
from av.container import Container
from av.stream import Stream
from tqdm import trange
from typing import Dict, Optional, Tuple, Generator, List
from omnigibson.learning.utils.eval_utils import (
    CAMERA_INTRINSICS,
    HEAD_RESOLUTION,
    WRIST_RESOLUTION,
    ROBOT_CAMERA_NAMES,
)
from omnigibson.learning.utils.dataset_utils import makedirs_with_mode
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

MIN_DEPTH = 0.0
MAX_DEPTH = 10.0
DEPTH_SHIFT = 3.5


def quantize_depth(
    depth: np.ndarray, min_depth: float = MIN_DEPTH, max_depth: float = MAX_DEPTH, shift: float = DEPTH_SHIFT
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
    quantized_depth: np.ndarray, min_depth: float = MIN_DEPTH, max_depth: float = MAX_DEPTH, shift: float = DEPTH_SHIFT
) -> np.ndarray:
    """
    Dequantizes a 14-bit depth tensor back to the original depth values.

    Args:
        quantized_depth (np.ndarray): Quantized depth tensor.
        min_depth (float): Minimum depth value.
        max_depth (float): Maximum depth value.
        shift (float): Small value to shift depth to avoid log(0).
    Returns:
        np.ndarray: Dequantized depth tensor.
    """
    qmax = (1 << 14) - 1
    log_min = np.log(min_depth + shift)
    log_max = np.log(max_depth + shift)

    log_norm = quantized_depth / qmax
    log_depth = log_norm * (log_max - log_min) + log_min
    depth = np.clip(np.exp(log_depth) - shift, min_depth, max_depth)

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
    elif mode == "seg":
        seg_ids = kwargs["seg_ids"]
        palette = th.from_numpy(generate_yuv_palette(len(seg_ids)))
        # Vectorized mapping - much faster than loop
        max_id = seg_ids.max().item() + 1
        instance_id_to_idx = th.full((max_id,), -1, dtype=th.long)
        instance_id_to_idx[seg_ids] = th.arange(len(seg_ids))
        for i in range(0, obs.shape[0], batch_size):
            seg_colored = palette[instance_id_to_idx[obs[i : i + batch_size]]].numpy()
            for frame in seg_colored:
                frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
    elif mode == "bbox":
        bbox_2d_data = kwargs["bbox"]
        for i in range(0, obs.shape[0], batch_size):
            for j, frame in enumerate(obs[i : i + batch_size].numpy()):
                # overlay bboxes with names
                frame = overlay_bboxes_with_names(
                    frame,
                    bbox_2d_data=bbox_2d_data[i + j],
                    instance_mapping=kwargs["instance_mapping"],
                    task_relevant_objects=kwargs["task_relevant_objects"],
                )
                frame = av.VideoFrame.from_ndarray(frame[..., :3], format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
    else:
        raise ValueError(f"Unsupported video mode: {mode}.")


class VideoLoader:
    def __init__(
        self,
        *args,
        path: str,
        batch_size: Optional[int] = None,
        stride: int = 1,
        output_size: Tuple[int, int] = None,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        start_idx_is_keyframe: bool = False,
        fps: int = 30,
        downsample_factor: int = 1,
        **kwargs,
    ):
        """
        Sequentially load RGB video with robust frame extraction.

        Args:
            path (str): Path to the video file
            batch_size (int): Batch size to load the video into memory. If None, load the entire video into memory.
            stride (int): Stride to load the video into memory.
                i.e. if batch_size=3 and stride=1, __iter__ will return [0, 1, 2], [1, 2, 3], [2, 3, 4], ...
            output_size (Tuple[int, int]): Output size of the video frames to resize to.
            start_idx (int): Frame to start loading the video from. Default is 0.
            end_idx (Optional[int]): Frame to stop loading the video at. If None, will load until video end.
                NOTE: end idx is not inclusive, i.e. if end_idx=10, the last frame will be 9.
            start_idx_is_keyframe (bool): Whether the start index is a keyframe.
                Set this to True if you know the start index is a keyframe, which will allow for faster seeking to the start index.
            fps (int): Frames per second of the video. Default is 30.
            downsample_factor (int): Factor to downsample the video frames by. Default is 1 (no downsampling).
        Returns:
            th.Tensor: (T, H, W, 3) RGB video tensor
        """
        self.container = av.open(path.replace(":", "+"))
        self.stream = self.container.streams.video[0]
        self._frames = []
        self.batch_size = batch_size
        self.stride = stride
        self._frame_iter = None
        self._done = False
        self.output_size = output_size
        self._start_frame = start_idx
        self._end_frame = end_idx if end_idx is not None else self.stream.frames
        self._start_idx_is_keyframe = start_idx_is_keyframe
        self._current_frame = start_idx
        self._time_base = self.stream.time_base
        self._fps = fps
        self._downsample_factor = downsample_factor
        # Note that unless start idx is keyframe, we also set start_pts to be a few frames preceding the start_frame if it's not 0,
        # so we can return the correct iterator in reset()
        start_frame = self._start_frame if self._start_idx_is_keyframe else max(0, self._start_frame - 5)
        self._start_pts = int(start_frame / self._fps / self._time_base)
        self.reset()

    def __iter__(self) -> Generator[th.Tensor, None, None]:
        self.reset()
        self._frames = []
        self._done = False
        return self

    def __next__(self):
        if self._done:
            raise StopIteration
        try:
            while True:
                # use downsample factor to skip frames
                for _ in range(self._downsample_factor - 1):
                    next(self._frame_iter)
                frame = next(self._frame_iter)
                processed_frame = self._process_single_frame(frame)
                self._current_frame += 1
                if self._current_frame == self._end_frame:
                    self._done = True
                self._frames.append(processed_frame)
                if (self.batch_size and len(self._frames) == self.batch_size) or self._done:
                    batch = th.cat(self._frames, dim=0)
                    self._frames = self._frames[self.stride :]
                    return batch
        except StopIteration:
            self._done = True
            if len(self._frames) > 0:
                batch = th.cat(self._frames, dim=0)
                self._frames = []
                return batch
            else:
                raise
        except Exception as e:
            self._done = True
            raise e

    def _process_single_frame(self, frame: av.VideoFrame) -> th.Tensor:
        raise NotImplementedError("Subclasses must implement this method")

    def reset(self):
        self._current_frame = self._start_frame
        self.container.seek(self._start_pts, stream=self.stream, backward=True, any_frame=False)
        self._frame_iter = self.container.decode(self.stream)
        if self._start_frame > 0 and not self._start_idx_is_keyframe:
            # Decode forward until we find the start frame
            for frame in self._frame_iter:
                if frame.pts is None:
                    continue
                cur_frame = round(frame.pts * self._time_base * self._fps)
                if cur_frame == self._start_frame - 1:
                    return
                elif cur_frame > self._start_frame - 1:
                    raise ValueError(
                        f"Start frame {self._start_frame} is beyond the video length. Current frame: {cur_frame}"
                    )

    @property
    def frames(self) -> th.Tensor:
        """
        Return all frames at once.
        """
        assert not self.batch_size, "Cannot get all frames at once when batch_size is set"
        return next(iter(self))

    def close(self):
        self.container.close()


class RGBVideoLoader(VideoLoader):
    def __init__(self, data_path: str, task_id: int, camera_id: str, demo_id: str, *args, **kwargs):
        super().__init__(
            path=f"{data_path}/videos/task-{task_id:04d}/observation.images.rgb.{camera_id}/episode_{demo_id}.mp4",
            *args,
            **kwargs,
        )

    def _process_single_frame(self, frame: av.VideoFrame) -> th.Tensor:
        rgb = frame.to_ndarray(format="rgb24")  # (H, W, 3)
        if self.output_size is None:
            return th.from_numpy(rgb).movedim(-1, -3).unsqueeze(0)
        rgb = F.interpolate(
            th.from_numpy(rgb).to(th.uint8).movedim(-1, -3).unsqueeze(0), size=self.output_size, mode="nearest-exact"
        )
        return rgb  # (1, 3, H, W)


class DepthVideoLoader(VideoLoader):
    def __init__(self, data_path: str, task_id: int, camera_id: str, demo_id: str, *args, **kwargs):
        self.min_depth = kwargs.get("min_depth", MIN_DEPTH)
        self.max_depth = kwargs.get("max_depth", MAX_DEPTH)
        self.shift = kwargs.get("shift", DEPTH_SHIFT)
        super().__init__(
            path=f"{data_path}/videos/task-{task_id:04d}/observation.images.depth.{camera_id}/episode_{demo_id}.mp4",
            *args,
            **kwargs,
        )

    def _process_single_frame(self, frame: av.VideoFrame) -> th.Tensor:
        # Decode Y (luma) channel only; YUV420 → grayscale image
        frame_gray16 = frame.reformat(format="gray16le").to_ndarray()  # (H, W)
        depth = dequantize_depth(frame_gray16, min_depth=self.min_depth, max_depth=self.max_depth, shift=self.shift)
        depth = th.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
        if self.output_size is not None:
            depth = F.interpolate(depth, size=self.output_size, mode="nearest-exact")
        return depth.squeeze(0)  # (1, H, W)


class SegVideoLoader(VideoLoader):
    def __init__(self, data_path: str, task_id: int, camera_id: str, demo_id: str, *args, **kwargs):
        self.id_list = kwargs.get("id_list", None)
        assert self.id_list is not None, "id_list must be provided for SegVideoLoader"
        self.id_list = self.id_list.to(device="cuda")  # (N_ids,)
        self.palette = th.from_numpy(generate_yuv_palette(len(self.id_list))).float().to(device="cuda")  # (N_ids, 3)
        super().__init__(
            path=f"{data_path}/videos/task-{task_id:04d}/observation.images.seg_instance_id.{camera_id}/episode_{demo_id}.mp4",
            *args,
            **kwargs,
        )

    def _process_single_frame(self, frame: av.VideoFrame) -> th.Tensor:
        rgb = th.from_numpy(frame.to_ndarray(format="rgb24")).float().to(device="cuda")  # (H, W, 3)
        rgb_flat = rgb.reshape(-1, 3)  # (H*W, 3)
        # For each rgb pixel, find the index of the nearest color in the equidistant bins
        distances = th.cdist(rgb_flat[None, :, :], self.palette[None, :, :], p=2)[0]  # (H*W, N_ids)
        ids = th.argmin(distances, dim=-1)  # (H*W,)
        ids = self.id_list[ids].reshape((rgb.shape[0], rgb.shape[1])).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        if self.output_size is not None:
            ids = F.interpolate(ids.float(), size=self.output_size, mode="nearest-exact")
        return ids.squeeze(0).cpu().to(th.long)  # (1, H, W)


OBS_LOADER_MAP = {
    "rgb": RGBVideoLoader,
    "depth": DepthVideoLoader,  # alias for depth_linear for compatibility
    "depth_linear": DepthVideoLoader,
    "seg_instance_id": SegVideoLoader,
}

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


def rgbd_vid_to_pcd(
    data_folder: str,
    task_id: int,
    demo_id: int,
    episode_id: int,
    robot_camera_names: Dict[str, str] = ROBOT_CAMERA_NAMES["R1Pro"],
    downsample_ratio: int = 4,
    pcd_range: Tuple[float, float, float, float, float, float] = (
        -0.2,
        1.0,
        -1.0,
        1.0,
        -0.2,
        1.5,
    ),  # x_min, x_max, y_min, y_max, z_min, z_max
    pcd_num_points: Optional[int] = 4096,
    batch_size: int = 500,
    use_fps: bool = False,
):
    """
    Generate point cloud data from compressed RGBD data (mp4) in the specified task folder.
    Args:
        data_folder (str): Path to the data folder containing RGBD data.
        task_id (int): Task ID for the task being processed.
        demo_id (int): Demo ID for the episode being processed.
        episode_id (int): Episode ID for the episode being processed.
        robot_camera_names (dict): Dict of camera names to process.
        downsample_ratio (int): Downsample ratio for the camera resolution.
        pcd_range (tuple): Range of the point cloud.
        pcd_num_points (Optional[int]): Number of points to sample from the point cloud. If None, no downsampling is performed.
        batch_size (int): Number of frames to process in each batch.
        use_fps (bool): Whether to use farthest point sampling for point cloud downsampling.
    """
    logger.info(f"Generating point cloud data from RGBD for {demo_id} in {data_folder}")
    output_dir = os.path.join(data_folder, "pcd_vid", f"task-{task_id:04d}")
    makedirs_with_mode(output_dir)

    total_n_points_before_downsample = 0
    # create a new hdf5 file to store the point cloud data
    with h5py.File(f"{output_dir}/episode_{demo_id:08d}.hdf5", "w") as out_f:
        in_f = pd.read_parquet(
            f"{data_folder}/2025-challenge-demos/data/task-{task_id:04d}/episode_{demo_id:08d}.parquet"
        )
        # get observation loaders
        obs_loaders = {}
        for camera_id, robot_camera_name in robot_camera_names.items():
            resolution = HEAD_RESOLUTION if camera_id == "head" else WRIST_RESOLUTION
            output_size = (resolution[0] // downsample_ratio, resolution[1] // downsample_ratio)
            total_n_points_before_downsample += output_size[0] * output_size[1]
            keys = ["rgb", "depth_linear"]
            for key in keys:
                kwargs = {}
                if key == "seg_semantic_id":
                    with open(
                        f"{data_folder}/2025-challenge-demos/meta/episodes/task-{task_id:04d}/episode_{demo_id:08d}.json"
                    ) as f:
                        kwargs["id_list"] = th.tensor(json.load(f)[f"{robot_camera_name}::unique_ins_ids"])
                obs_loaders[f"{robot_camera_name}::{key}"] = iter(
                    OBS_LOADER_MAP[key](
                        data_path=f"{data_folder}/2025-challenge-demos",
                        task_id=task_id,
                        demo_id=f"{demo_id:08d}",
                        camera_id=camera_id,
                        output_size=output_size,
                        batch_size=batch_size,
                        stride=batch_size,
                        **kwargs,
                    )
                )
        # construct out dset
        cam_rel_poses = th.from_numpy(np.array(in_f["observation.cam_rel_poses"].tolist(), dtype=np.float32))
        data_size = cam_rel_poses.shape[0]
        fused_pcd_dset = out_f.create_dataset(
            f"data/demo_{episode_id}/robot_r1::fused_pcd",
            shape=(data_size, pcd_num_points if pcd_num_points is not None else total_n_points_before_downsample, 6),
            compression="lzf",
        )
        # We batch process every batch_size frames
        for i in range(0, data_size, batch_size):
            logger.info(f"Processing batch {i} of {data_size}...")
            obs = dict()  # to store rgbd and pass into process_fused_point_cloud
            obs["cam_rel_poses"] = cam_rel_poses[i : i + batch_size]
            # get all camera intrinsics
            camera_intrinsics = {}
            for camera_id, robot_camera_name in robot_camera_names.items():
                # Calculate the downsampled camera intrinsics
                camera_intrinsics[robot_camera_name] = (
                    th.from_numpy(CAMERA_INTRINSICS["R1Pro"][camera_id]) / downsample_ratio
                )
                camera_intrinsics[robot_camera_name][-1, -1] = 1.0
                obs[f"{robot_camera_name}::rgb"] = next(obs_loaders[f"{robot_camera_name}::rgb"]).movedim(-3, -1)
                obs[f"{robot_camera_name}::depth_linear"] = next(obs_loaders[f"{robot_camera_name}::depth_linear"])
            # process the fused point cloud
            pcd = process_fused_point_cloud(
                obs=obs,
                camera_intrinsics=camera_intrinsics,
                pcd_range=pcd_range,
                pcd_num_points=pcd_num_points,
                use_fps=use_fps,
                verbose=True,
            )
            logger.info("Saving point cloud data...")
            fused_pcd_dset[i : i + batch_size] = pcd.cpu()

    logger.info("Point cloud data saved!")


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


def generate_yuv_palette(num_ids: int) -> np.ndarray:
    """
    Generate `num_ids` equidistant YUV colors in the valid YUV space.
    """
    # Y in [16, 235], U, V in [16, 240] for 8-bit YUV standards (BT.601)
    Y_vals = np.linspace(16, 235, int(np.ceil(num_ids ** (1 / 3))))
    U_vals = np.linspace(16, 240, int(np.ceil(num_ids ** (1 / 3))))
    V_vals = np.linspace(16, 240, int(np.ceil(num_ids ** (1 / 3))))

    palette = []
    for y in Y_vals:
        for u in U_vals:
            for v in V_vals:
                palette.append([y, u, v])
                if len(palette) >= num_ids:
                    return np.array(palette, dtype=np.uint8)
    return np.array(palette[:num_ids], dtype=np.uint8)


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


# ==============================================
# Bounding box
# ==============================================


def find_non_overlapping_text_position(x1, y1, x2, y2, text_size, occupied_regions, img_height, img_width):
    """Find a text position that doesn't overlap with existing text."""
    text_w, text_h = text_size
    padding = 5

    # Try different positions in order of preference
    positions = [
        # Above bbox
        (x1, y1 - text_h - padding),
        # Below bbox
        (x1, y2 + text_h + padding),
        # Right of bbox
        (x2 + padding, y1 + text_h),
        # Left of bbox
        (x1 - text_w - padding, y1 + text_h),
        # Inside bbox (top-left)
        (x1 + padding, y1 + text_h + padding),
        # Inside bbox (bottom-right)
        (x2 - text_w - padding, y2 - padding),
    ]

    for text_x, text_y in positions:
        # Check bounds
        if text_x < 0 or text_y < text_h or text_x + text_w > img_width or text_y > img_height:
            continue

        # Check for overlap with existing text
        text_rect = (text_x - padding, text_y - text_h - padding, text_x + text_w + padding, text_y + padding)

        overlap = False
        for occupied_rect in occupied_regions:
            if (
                text_rect[0] < occupied_rect[2]
                and text_rect[2] > occupied_rect[0]
                and text_rect[1] < occupied_rect[3]
                and text_rect[3] > occupied_rect[1]
            ):
                overlap = True
                break

        if not overlap:
            return text_x, text_y, text_rect

    # Fallback: use the first position even if it overlaps
    text_x, text_y = positions[0]
    text_rect = (text_x - padding, text_y - text_h - padding, text_x + text_w + padding, text_y + padding)
    return text_x, text_y, text_rect


def overlay_bboxes_with_names(
    img: np.ndarray,
    bbox_2d_data: List[Tuple[int, int, int, int, int]],
    instance_mapping: Dict[int, str],
    task_relevant_objects: List[str],
) -> np.ndarray:
    """
    Overlays bounding boxes with object names on the given image.

    Args:
        img (np.ndarray): The input image (RGB) to overlay on.
        bbox_2d_data (List[Tuple[int, int, int, int, int]]): Bounding box data with format (x1, y1, x2, y2, instance_id)
        instance_mapping (Dict[int, str]): Mapping from instance ID to object name
        task_relevant_objects (List[str]): List of task relevant objects
    Returns:
        np.ndarray: The image with bounding boxes and object names overlaid.
    """
    # Create a copy of the image to draw on
    overlay_img = img.copy()
    img_height, img_width = img.shape[:2]

    # Track occupied text regions to avoid overlap
    occupied_text_regions = []

    # Process each bounding box
    for bbox in bbox_2d_data:
        x1, y1, x2, y2, instance_id = bbox
        object_name = instance_mapping[instance_id]
        # Only overlay task relevant objects
        if object_name not in task_relevant_objects:
            continue

        # Generate a consistent color based on instance_id
        color = get_consistent_color(instance_id)

        # Draw the bounding box
        cv2.rectangle(overlay_img, (x1, y1), (x2, y2), color, 2)

        # Draw the object name
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(object_name, font, font_scale, font_thickness)[0]
        # Find non-overlapping position for text
        text_x, text_y, text_rect = find_non_overlapping_text_position(
            x1, y1, x2, y2, text_size, occupied_text_regions, img_height, img_width
        )
        # Add this text region to occupied regions
        occupied_text_regions.append(text_rect)

        # Draw background rectangle for text
        cv2.rectangle(
            overlay_img, (int(text_rect[0]), int(text_rect[1])), (int(text_rect[2]), int(text_rect[3])), color, -1
        )

        # Draw the text
        cv2.putText(
            overlay_img,
            object_name,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    return overlay_img


def get_consistent_color(instance_id):
    import colorsys

    colors = [
        (52, 73, 94),  # Dark blue-gray
        (142, 68, 173),  # Purple
        (39, 174, 96),  # Emerald green
        (230, 126, 34),  # Orange
        (231, 76, 60),  # Red
        (41, 128, 185),  # Blue
        (155, 89, 182),  # Amethyst
        (26, 188, 156),  # Turquoise
        (241, 196, 15),  # Yellow (darker)
        (192, 57, 43),  # Dark red
        (46, 204, 113),  # Green
        (52, 152, 219),  # Light blue
        (155, 89, 182),  # Violet
        (22, 160, 133),  # Dark turquoise
        (243, 156, 18),  # Dark yellow
        (211, 84, 0),  # Dark orange
        (154, 18, 179),  # Dark purple
        (31, 81, 255),  # Royal blue
        (20, 90, 50),  # Forest green
        (120, 40, 31),  # Maroon
    ]

    # Use hash to consistently select a color from the palette
    hash_val = hash(str(instance_id))
    base_color_idx = hash_val % len(colors)
    base_color = colors[base_color_idx]

    # Add slight variation while maintaining sophistication
    # Convert to HSV for easier manipulation
    r, g, b = [c / 255.0 for c in base_color]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    # Add small random variation to hue (±10 degrees) and saturation/value
    hue_variation = ((hash_val >> 8) % 20 - 10) / 360.0  # ±10 degrees
    sat_variation = ((hash_val >> 16) % 20 - 10) / 200.0  # ±5% saturation
    val_variation = ((hash_val >> 24) % 20 - 10) / 200.0  # ±5% value

    # Apply variations with bounds checking
    h = (h + hue_variation) % 1.0
    s = max(0.4, min(0.9, s + sat_variation))  # Keep saturation between 40-90%
    v = max(0.3, min(0.7, v + val_variation))  # Keep value between 30-70% (darker for contrast)

    # Convert back to RGB
    r, g, b = colorsys.hsv_to_rgb(h, s, v)

    # Convert to 0-255 range
    return (int(r * 255), int(g * 255), int(b * 255))
