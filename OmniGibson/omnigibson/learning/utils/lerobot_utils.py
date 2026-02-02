import argparse
import av
# Suppress verbose FFmpeg/libx264 encoder output at module load time
av.logging.set_level(av.logging.PANIC)

import json
import logging
import numpy as np
import os
import pandas as pd
import torch as th
import torchvision
from torch.utils.data import Dataset
import packaging
import glob
import PIL
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    CODEBASE_VERSION as LEROBOT_CODEBASE_VERSION,
)

import lerobot.datasets.image_writer as liw

if LEROBOT_CODEBASE_VERSION == "v2.1":
    from lerobot.constants import HF_LEROBOT_HOME
else:
    from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.datasets.video_utils import get_audio_info, get_safe_default_codec
from lerobot.datasets.compute_stats import _assert_type_and_shape
from omnigibson.learning.utils.dataset_utils import get_credentials
from omnigibson.learning.utils.eval_utils import (
    TASK_NAMES_TO_INDICES,
    PROPRIOCEPTION_INDICES,
)
import omnigibson.learning.utils.obs_utils as OU
from omnigibson.utils.ui_utils import create_module_logger
from pathlib import Path
from PIL import Image as PILImage
from torchvision import transforms
from torchvision.io import VideoReader
from typing import Any, Dict, Tuple, Callable, Sequence
from tqdm import tqdm
import tempfile
import shutil

from lerobot.datasets.utils import (
    DEFAULT_IMAGE_PATH,
    check_delta_timestamps,
    is_valid_version,
    get_delta_indices,
    get_safe_version,
    validate_episode_buffer,
    write_info,
)
from lerobot.datasets.compute_stats import compute_episode_stats
from lerobot.datasets.video_utils import decode_video_frames as native_decode_video_frames
import concurrent.futures


logger = create_module_logger("OmniGibsonLeRobotDataset")


def hf_transform_to_torch(items_dict: dict[th.Tensor | None]):
    """
    Adapted from lerobot.datasets.utils.hf_transform_to_torch
    Preserve float64 for timestamp to avoid precision issues
    Below is the original docstring:
    Get a transform function that convert items from Hugging Face dataset (pyarrow)
    to torch tensors. Importantly, images are converted from PIL, which corresponds to
    a channel last representation (h w c) of uint8 type, to a torch image representation
    with channel first (c h w) of float32 type in range [0,1].
    """
    for key in items_dict:
        if key == "timestamp":
            items_dict[key] = [x if isinstance(x, str) else th.tensor(x, dtype=th.float64) for x in items_dict[key]]
        else:
            first_item = items_dict[key][0]
            if isinstance(first_item, PILImage.Image):
                to_tensor = transforms.ToTensor()
                items_dict[key] = [to_tensor(img) for img in items_dict[key]]
            elif first_item is None:
                pass
            else:
                items_dict[key] = [x if isinstance(x, str) else th.tensor(x) for x in items_dict[key]]
    return items_dict


def get_video_pixel_channels(pix_fmt: str) -> int:
    if "gray" in pix_fmt or "depth" in pix_fmt or "monochrome" in pix_fmt:
        return 1
    elif "rgba" in pix_fmt or "yuva" in pix_fmt:
        return 4
    elif "rgb" in pix_fmt or "yuv" in pix_fmt or "gbrp" in pix_fmt:
        return 3
    else:
        raise ValueError("Unknown format")


def get_video_info(video_path: Path | str) -> dict:
    # Set logging level
    logging.getLogger("libav").setLevel(av.logging.ERROR)

    # Getting video stream information
    video_info = {}
    with av.open(str(video_path), "r") as video_file:
        try:
            video_stream = video_file.streams.video[0]
        except IndexError:
            # Reset logging level
            av.logging.restore_default_callback()
            return {}

        video_info["video.height"] = video_stream.height
        video_info["video.width"] = video_stream.width
        video_info["video.codec"] = video_stream.codec.canonical_name
        video_info["video.pix_fmt"] = video_stream.pix_fmt
        video_info["video.is_depth_map"] = False

        # Calculate fps from r_frame_rate
        video_info["video.fps"] = int(video_stream.base_rate)

        pixel_channels = get_video_pixel_channels(video_stream.pix_fmt)
        video_info["video.channels"] = pixel_channels

    # Reset logging level
    av.logging.restore_default_callback()

    # Adding audio stream information
    video_info.update(**get_audio_info(video_path))

    return video_info


def _encode_video_worker(video_key: str, episode_index: int, root: Path, fps: int) -> Path:
    temp_path = Path(tempfile.mkdtemp(dir=root)) / f"{video_key}_{episode_index:03d}.mp4"
    fpath = DEFAULT_IMAGE_PATH.format(image_key=video_key, episode_index=episode_index, frame_index=0)
    img_dir = (root / fpath).parent
    # Use our custom video encoder
    mode = "depth" if "depth" in video_key else "rgb"
    encode_video_frames(img_dir, temp_path, fps, overwrite=True, mode=mode)

    shutil.rmtree(img_dir)
    return temp_path


def encode_video_frames(
    imgs_dir: Path | str,
    video_path: Path | str,
    fps: int,
    vcodec: str = "h264",
    pix_fmt: str = "yuv420p",
    g: int | None = 2,
    crf: int | None = 24,
    fast_decode: int = 0,
    log_level: int | None = av.logging.ERROR,
    overwrite: bool = False,
    preset: int | None = None,
    mode: str = "rgb",
    stream_options: dict | None = None,
) -> None:
    """More info on ffmpeg arguments tuning on `benchmark/video/README.md`"""
    # Check encoder availability
    if vcodec not in ["h264", "hevc", "libsvtav1", "libx264", "libx265"]:
        raise ValueError(f"Unsupported video codec: {vcodec}. Supported codecs are: h264, hevc, libsvtav1.")

    video_path = Path(video_path)
    imgs_dir = Path(imgs_dir)

    if video_path.exists() and not overwrite:
        logging.warning(f"Video file already exists: {video_path}. Skipping encoding.")
        return

    video_path.parent.mkdir(parents=True, exist_ok=True)

    # Encoders/pixel formats incompatibility check
    if (vcodec == "libsvtav1" or vcodec == "hevc") and pix_fmt == "yuv444p":
        logging.warning(f"Incompatible pixel format 'yuv444p' for codec {vcodec}, auto-selecting format 'yuv420p'")
        pix_fmt = "yuv420p"

    # Encoders/pixel formats incompatibility check
    if mode == "depth":
        if stream_options is None:
            stream_options = dict()
        if OU.USE_DEPTH_RGB_ENCODING:
            _depth_codec = "hevc_nvenc"
            _depth_pix_fmt = "gbrp"
            stream_options["tune"] = "lossless"
            g = max(g, 3)  # Assumes B = 2, and G >= B + 1
        else:
            _depth_codec = "libx265"
            _depth_pix_fmt = "yuv444p12le"
            stream_options["x265-params"] = "lossless=1:log-level=none"
        if vcodec != _depth_codec:
            logging.warning(f"Incompatible codec {vcodec} with mode == 'depth', auto-selecting codec '{_depth_codec}'")
            vcodec = _depth_codec
        if pix_fmt != _depth_pix_fmt:
            logging.warning(
                f"Incompatible pix_fmt {pix_fmt} with mode == 'depth', auto-selecting pix_fmt '{_depth_pix_fmt}'"
            )
            pix_fmt = _depth_pix_fmt

    # Determine info based on lerobot version
    if LEROBOT_CODEBASE_VERSION == "v2.1":
        delimiter = "_"
    else:
        # v3.0
        delimiter = "-"

    # Get input frames
    template = f"frame{delimiter}" + ("[0-9]" * 6) + ".png"
    input_list = sorted(glob.glob(str(imgs_dir / template)), key=lambda x: int(x.split(delimiter)[-1].split(".")[0]))

    # Define video output frame size (assuming all input frames are the same size)
    if len(input_list) == 0:
        raise FileNotFoundError(f"No images found in {imgs_dir}.")
    with PIL.Image.open(input_list[0]) as dummy_image:
        width, height = dummy_image.size

    # Define video codec options
    video_options = {}

    if g is not None:
        video_options["g"] = str(g)

    if crf is not None:
        video_options["crf"] = str(crf)

    if fast_decode:
        key = "svtav1-params" if vcodec == "libsvtav1" else "tune"
        value = f"fast-decode={fast_decode}" if vcodec == "libsvtav1" else "fastdecode"
        video_options[key] = value

    if vcodec == "libsvtav1":
        video_options["preset"] = str(preset) if preset is not None else "12"

    # Set logging level for both Python and native FFmpeg
    if log_level is not None:        
        av.logging.set_level(av.logging.PANIC)
        logging.getLogger("libav").setLevel(log_level)


    # Create and open output file (overwrite by default)
    with av.open(str(video_path), "w") as output:
        output_stream = output.add_stream(vcodec, fps, options=video_options)
        output_stream.pix_fmt = pix_fmt
        output_stream.width = width
        output_stream.height = height
        # output_stream.codec_context.gop_size = int(fps * 2)
        if stream_options is not None:
            output_stream.options = stream_options

        # Loop through input frames and encode them
        for input_data in input_list:
            with PIL.Image.open(input_data) as input_image:
                if OU.USE_DEPTH_RGB_ENCODING or mode == "rgb":
                    input_image = input_image.convert("RGB")
                    input_frame = av.VideoFrame.from_image(input_image)
                elif mode == "depth":
                    input_image = np.array(input_image, dtype=np.uint16)
                    input_frame = av.VideoFrame.from_ndarray(input_image, format="gray16le")
                packet = output_stream.encode(input_frame)
                if packet:
                    output.mux(packet)

        # Flush the encoder
        packet = output_stream.encode()
        if packet:
            output.mux(packet)

    # Reset logging level
    if log_level is not None:
        av.logging.restore_default_callback()

    if not video_path.exists():
        raise OSError(f"Video encoding did not work. File not found: {video_path}.")


def decode_video_frames(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str | None = None,
    min_depth: float | None = None,
    max_depth: float | None = None,
    depth_shift: float | None = None,
) -> th.Tensor:
    if OU.USE_DEPTH_RGB_ENCODING:
        frames = native_decode_video_frames(
            video_path=video_path, timestamps=timestamps, tolerance_s=tolerance_s, backend=backend
        )

        # Convert RGB -> depth
        if "depth" in str(video_path):
            frames = th.from_numpy(
                OU.rgb2depth(
                    (frames.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8),
                    min_depth=min_depth,
                    max_depth=max_depth,
                    disparity=False,
                    err_depth=0.0,
                )
            ).unsqueeze(-3)

        return frames

    else:
        # Use our custom decoder for depth
        if "depth" in str(video_path):
            backend = "pyav"
            return decode_video_frames_torchvision(
                video_path=video_path,
                timestamps=timestamps,
                tolerance_s=tolerance_s,
                backend=backend,
                min_depth=min_depth,
                max_depth=max_depth,
                depth_shift=depth_shift,
            )
        else:
            return native_decode_video_frames(
                video_path=video_path, timestamps=timestamps, tolerance_s=tolerance_s, backend=backend
            )


def decode_video_frames_torchvision(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    log_loaded_timestamps: bool = False,
    backend: str | None = None,
    min_depth: float | None = None,
    max_depth: float | None = None,
    depth_shift: float | None = None,
) -> th.Tensor:
    """
    Adapted from decode_video_frames_vision to handle depth decoding
    """
    video_path = str(video_path)

    # set backend
    keyframes_only = False
    if "depth" in video_path:
        backend = "pyav"
    torchvision.set_video_backend(backend)
    if backend == "pyav":
        keyframes_only = True  # pyav doesn't support accurate seek

    # set a video stream reader
    # TODO(rcadene): also load audio stream at the same time
    if "depth" in video_path:
        reader = DepthVideoReader(
            video_path, "video", min_depth=min_depth, max_depth=max_depth, depth_shift=depth_shift
        )
    else:
        reader = VideoReader(video_path, "video")

    # set the first and last requested timestamps
    # Note: previous timestamps are usually loaded, since we need to access the previous key frame
    first_ts = min(timestamps) - 5  # a little backward to account for timestamp mismatch
    last_ts = max(timestamps)

    # access closest key frame of the first requested frame
    # Note: closest key frame timestamp is usually smaller than `first_ts` (e.g. key frame can be the first frame of the video)
    # for details on what `seek` is doing see: https://pyav.basswood-io.com/docs/stable/api/container.html?highlight=inputcontainer#av.container.InputContainer.seek
    reader.seek(first_ts, keyframes_only=keyframes_only)

    # load all frames until last requested frame
    loaded_frames = []
    loaded_ts = []
    for frame in reader:
        current_ts = frame["pts"]
        if log_loaded_timestamps:
            logging.info(f"frame loaded at timestamp={current_ts:.4f}")
        loaded_frames.append(frame["data"])
        loaded_ts.append(current_ts)
        if current_ts >= last_ts:
            break

    reader.container.close()

    reader = None

    query_ts = th.tensor(timestamps)
    loaded_ts = th.tensor(loaded_ts)

    # compute distances between each query timestamp and timestamps of all loaded frames
    dist = th.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
        "It means that the closest frame that can be loaded from the video is too far away in time."
        "This might be due to synchronization issues with timestamps during data collection."
        "To be safe, we advise to ignore this item during training."
        f"\nqueried timestamps: {query_ts}"
        f"\nloaded timestamps: {loaded_ts}"
        f"\nvideo: {video_path}"
        f"\nbackend: {backend}"
    )

    # get closest frames to the query timestamps
    closest_frames = th.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logging.info(f"{closest_ts=}")

    # convert to the pytorch format which is float32 in [0,1] range (and channel first)
    closest_frames = closest_frames.type(th.float32)
    if "depth" not in video_path:
        closest_frames = closest_frames / 255

    assert len(timestamps) == len(closest_frames)
    return closest_frames


class DepthVideoReader(VideoReader):
    """
    Adapted from torchvision.io.VideoReader to support gray16le decoding for depth
    """

    def __init__(
        self,
        src: str,
        stream: str = "video",
        num_threads: int = 0,
        min_depth: float | None = None,
        max_depth: float | None = None,
        depth_shift: float | None = None,
    ) -> None:
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_shift = depth_shift

        super().__init__(src=src, stream=stream, num_threads=num_threads)

    def __next__(self) -> Dict[str, Any]:
        """Decodes and returns the next frame of the current stream.
        Frames are encoded as a dict with mandatory
        data and pts fields, where data is a tensor, and pts is a
        presentation timestamp of the frame expressed in seconds
        as a float.

        Returns:
            (dict): a dictionary and containing decoded frame (``data``)
            and corresponding timestamp (``pts``) in seconds

        """
        try:
            frame = next(self._c)
            pts = float(frame.pts * frame.time_base)
            if "video" in self.pyav_stream:
                frame = th.as_tensor(
                    OU.dequantize_depth(
                        frame.reformat(format="gray16le").to_ndarray(),
                        min_depth=self.min_depth,
                        max_depth=self.max_depth,
                        shift=self.depth_shift,
                    )
                ).unsqueeze(dim=-3)  # Add back 1 channel
            elif "audio" in self.pyav_stream:
                frame = th.as_tensor(frame.to_ndarray()).permute(1, 0)
            else:
                frame = None
        except av.error.EOFError:
            raise StopIteration

        if frame.numel() == 0:
            raise StopIteration

        return {"data": frame, "pts": pts}


def aggregate_stats(stats_list: list[dict[str, dict]]) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate stats from multiple compute_stats outputs into a single set of stats.

    The final stats will have the union of all data keys from each of the stats dicts.

    For instance:
    - new_min = min(min_dataset_0, min_dataset_1, ...)
    - new_max = max(max_dataset_0, max_dataset_1, ...)
    - new_mean = (mean of all data, weighted by counts)
    - new_std = (std of all data)
    """

    _assert_type_and_shape(stats_list)

    data_keys = {key for stats in stats_list for key in stats}
    aggregated_stats = {key: {} for key in data_keys}

    for key in data_keys:
        stats_with_key = [stats[key] for stats in stats_list if key in stats]
        aggregated_stats[key] = aggregate_feature_stats(stats_with_key)

    return aggregated_stats


def aggregate_feature_stats(stats_ft_list: list[dict[str, dict]]) -> dict[str, dict[str, np.ndarray]]:
    """Aggregates stats for a single feature."""
    means = np.stack([s["mean"] for s in stats_ft_list])
    variances = np.stack([s["std"] ** 2 for s in stats_ft_list])
    counts = np.stack([s["count"] for s in stats_ft_list])
    q01 = np.stack([s["q01"] for s in stats_ft_list])
    q99 = np.stack([s["q99"] for s in stats_ft_list])
    total_count = counts.sum(axis=0)

    # Prepare weighted mean by matching number of dimensions
    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)

    # Compute the weighted mean
    weighted_means = means * counts
    total_mean = weighted_means.sum(axis=0) / total_count

    # Compute the variance using the parallel algorithm
    delta_means = means - total_mean
    weighted_variances = (variances + delta_means**2) * counts
    total_variance = weighted_variances.sum(axis=0) / total_count

    # Compute weighted quantiles
    weighted_q01 = np.percentile(q01, 1, axis=0)
    weighted_q99 = np.percentile(q99, 99, axis=0)

    return {
        "min": np.min(np.stack([s["min"] for s in stats_ft_list]), axis=0),
        "max": np.max(np.stack([s["max"] for s in stats_ft_list]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "q01": weighted_q01,
        "q99": weighted_q99,
        "count": total_count,
    }


def generate_task_json(data_dir: str, credentials_path: str) -> int:
    num_tasks = len(TASK_NAMES_TO_INDICES)
    gc = get_credentials(credentials_path=credentials_path)[0]
    spreadsheet = gc.open("B50 Task Misc")
    ws = spreadsheet.worksheet("Task natural language instruction")
    rows = ws.get_all_values()
    with open(f"{data_dir}/meta/tasks.jsonl", "w") as f:
        for task_name, task_index in tqdm(TASK_NAMES_TO_INDICES.items()):
            # find the corresponding row in the google sheet
            prompt = None
            for row in rows:
                if row[0] == task_name:
                    prompt = row[1].replace("\u2019", "'")
                    break
            assert prompt is not None, f"Natural language instruction not found for task: {task_name}"
            json.dump(
                {
                    "task_index": task_index,
                    "task_name": task_name,
                    "task": prompt,
                },
                f,
            )
            f.write("\n")
    print(f"Generated task JSON for {num_tasks} tasks.")
    return num_tasks


def generate_episode_json(data_dir: str, robot_type: str = "R1Pro") -> Tuple[int, int]:
    assert os.path.exists(f"{data_dir}/meta/tasks.jsonl"), "Task JSON does not exist!"
    assert os.path.exists(f"{data_dir}/meta/episodes"), "Episode metadata directory does not exist!"
    with open(f"{data_dir}/meta/tasks.jsonl", "r") as f:
        task_json = [json.loads(line) for line in f]
    num_frames = 0
    num_episodes = 0
    with open(f"{data_dir}/meta/episodes.jsonl", "w") as out_f:
        with open(f"{data_dir}/meta/episodes_stats.jsonl", "w") as out_stats_f:
            for task_info in tqdm(task_json):
                task_index = task_info["task_index"]
                task_name = task_info["task"]
                if not os.path.exists(f"{data_dir}/meta/episodes/task-{task_index:04d}"):
                    continue
                for episode_name in tqdm(sorted(os.listdir(f"{data_dir}/meta/episodes/task-{task_index:04d}"))):
                    with open(f"{data_dir}/meta/episodes/task-{task_index:04d}/{episode_name}", "r") as f:
                        episode_info = json.load(f)
                        episode_index = int(episode_name.split(".")[0].split("_")[-1])
                        episode_json = {
                            "episode_index": episode_index,
                            "tasks": [task_name],
                            "length": episode_info["num_samples"],
                        }
                        # load the corresponding parquet file
                        episode_df = pd.read_parquet(
                            f"{data_dir}/data/task-{task_index:04d}/episode_{episode_index:08d}.parquet"
                        )
                        episode_stats = {}
                        for key in ["action", "observation.state", "observation.cam_rel_poses"]:
                            if key not in episode_stats:
                                episode_stats[key] = {}
                            values = np.stack(episode_df[key].values)
                            if len(values.shape) == 1:
                                values = values[:, np.newaxis]
                            episode_stats[key]["min"] = values.min(axis=0).tolist()
                            episode_stats[key]["max"] = values.max(axis=0).tolist()
                            episode_stats[key]["mean"] = values.mean(axis=0).tolist()
                            episode_stats[key]["std"] = values.std(axis=0).tolist()
                            episode_stats[key]["q01"] = np.quantile(values, 0.01, axis=0).tolist()
                            episode_stats[key]["q99"] = np.quantile(values, 0.99, axis=0).tolist()
                            episode_stats[key]["count"] = [values.shape[0]]
                            if key == "observation.state":
                                robot_pos = values[:, PROPRIOCEPTION_INDICES[robot_type]["robot_pos"]]
                                episode_json["distance_traveled"] = round(
                                    np.sum(np.linalg.norm(robot_pos[1:, :] - robot_pos[:-1, :], axis=-1)).item(), 4
                                )
                                left_eef_pos = values[:, PROPRIOCEPTION_INDICES[robot_type]["eef_left_pos"]]
                                right_eef_pos = values[:, PROPRIOCEPTION_INDICES[robot_type]["eef_right_pos"]]
                                episode_json["left_eef_displacement"] = round(
                                    np.sum(np.linalg.norm(left_eef_pos[1:, :] - left_eef_pos[:-1, :], axis=-1)).item(),
                                    4,
                                )
                                episode_json["right_eef_displacement"] = round(
                                    np.sum(
                                        np.linalg.norm(right_eef_pos[1:, :] - right_eef_pos[:-1, :], axis=-1)
                                    ).item(),
                                    4,
                                )
                        episode_stats_json = {
                            "episode_index": episode_index,
                            "stats": episode_stats,
                        }
                        num_episodes += 1
                        num_frames += episode_info["num_samples"]
                    json.dump(episode_json, out_f)
                    out_f.write("\n")
                    json.dump(episode_stats_json, out_stats_f)
                    out_stats_f.write("\n")
    print(f"Generated episode JSON for {num_episodes} episodes and {num_frames} frames.")
    return num_episodes, num_frames


def generate_info_json(
    data_dir: str,
    fps: int = 30,
    total_episodes: int = 50,
    total_tasks: int = 50,
    total_frames: int = 50,
):
    info = {
        "codebase_version": "v2.1",
        "robot_type": "R1Pro",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "total_videos": total_episodes * 9,
        "chunks_size": 10000,
        "fps": fps,
        "splits": {
            "train": "0:" + str(total_episodes),
        },
        "data_path": "data/task-{episode_chunk:04d}/episode_{episode_index:08d}.parquet",
        "video_path": "videos/task-{episode_chunk:04d}/{video_key}/episode_{episode_index:08d}.mp4",
        "metainfo_path": "meta/episodes/task-{episode_chunk:04d}/episode_{episode_index:08d}.json",
        "annotation_path": "annotations/task-{episode_chunk:04d}/episode_{episode_index:08d}.json",
        "features": {
            "observation.images.rgb.left_wrist": {
                "dtype": "video",
                "shape": [480, 480, 3],
                "names": ["height", "width", "rgb"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 480,
                    "video.width": 480,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.images.rgb.right_wrist": {
                "dtype": "video",
                "shape": [480, 480, 3],
                "names": ["height", "width", "rgb"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 480,
                    "video.width": 480,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.images.rgb.head": {
                "dtype": "video",
                "shape": [720, 720, 3],
                "names": ["height", "width", "rgb"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 720,
                    "video.width": 720,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.images.depth.left_wrist": {
                "dtype": "video",
                "shape": [480, 480, 3],
                "names": ["height", "width", "depth"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 480,
                    "video.width": 480,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p16le",
                    "video.is_depth_map": True,
                    "has_audio": False,
                },
            },
            "observation.images.depth.right_wrist": {
                "dtype": "video",
                "shape": [480, 480, 3],
                "names": ["height", "width", "depth"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 480,
                    "video.width": 480,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p16le",
                    "video.is_depth_map": True,
                    "has_audio": False,
                },
            },
            "observation.images.depth.head": {
                "dtype": "video",
                "shape": [720, 720, 3],
                "names": ["height", "width", "depth"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 720,
                    "video.width": 720,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p16le",
                    "video.is_depth_map": True,
                    "has_audio": False,
                },
            },
            "observation.images.seg_instance_id.left_wrist": {
                "dtype": "video",
                "shape": [480, 480, 3],
                "names": ["height", "width", "rgb"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 480,
                    "video.width": 480,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.images.seg_instance_id.right_wrist": {
                "dtype": "video",
                "shape": [480, 480, 3],
                "names": ["height", "width", "rgb"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 480,
                    "video.width": 480,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.images.seg_instance_id.head": {
                "dtype": "video",
                "shape": [720, 720, 3],
                "names": ["height", "width", "rgb"],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 720,
                    "video.width": 720,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "action": {"dtype": "float32", "shape": [23], "names": None},
            "timestamp": {"dtype": "float64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "observation.cam_rel_poses": {"dtype": "float32", "shape": [21], "names": None},
            "observation.state": {"dtype": "float32", "shape": [256], "names": None},
            "observation.task_info": {"dtype": "float32", "shape": [None], "names": None},
        },
    }

    with open(f"{data_dir}/meta/info.json", "w") as f:
        json.dump(info, f, indent=4)

    print(f"Generated info JSON for {len(info)} entries.")


class OmniGibsonLeRobotDatasetMetadata(LeRobotDatasetMetadata):
    #
    # @classmethod
    # def create(
    #     cls,
    #     repo_id: str,
    #     fps: int,
    #     features: dict,
    #     robot_type: str | None = None,
    #     root: str | Path | None = None,
    #     use_videos: bool = True,
    # ) -> "LeRobotDatasetMetadata":
    # @classmethod
    # def create(
    #     cls,
    #     repo_id: str,
    #     fps: int,
    #     features: dict,
    #     robot_type: str | None = None,
    #     root: str | Path | None = None,
    #     use_videos: bool = True,
    # ) -> "LeRobotDatasetMetadata":
    #     """Creates metadata for a LeRobotDataset."""
    #     obj = cls.__new__(cls)
    #     obj._features_to_include = None
    #     obj.repo_id = repo_id
    #     obj.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
    #
    #     obj.root.mkdir(parents=True, exist_ok=False)
    #
    #     # TODO(aliberts, rcadene): implement sanity check for features
    #     features = {**features, **DEFAULT_FEATURES}
    #     _validate_feature_names(features)
    #
    #     obj.tasks, obj.task_to_task_index = {}, {}
    #     obj.episodes_stats, obj.stats, obj.episodes = {}, {}, {}
    #     obj.info = create_empty_dataset_info(CODEBASE_VERSION, fps, features, use_videos, robot_type)
    #     if len(obj.video_keys) > 0 and not use_videos:
    #         raise ValueError()
    #     write_json(obj.info, obj.root / INFO_PATH)
    #     obj.revision = None
    #     return obj

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
        features: Sequence[str] | None = None,
    ):
        self._features_to_include = None if features is None else set(features)
        # Call super
        super().__init__(
            repo_id=repo_id,
            root=root,
            revision=revision,
            force_cache_sync=force_cache_sync,
        )

    @property
    def features(self) -> dict[str, dict]:
        """All features contained in the dataset."""
        if not hasattr(self, "_features_to_include") or self._features_to_include is None:
            return self.info["features"]
        else:
            return {k: v for k, v in self.info["features"].items() if k in self._features_to_include}

    def update_video_info(self, video_key=None) -> None:
        """
        Warning: this function writes info from first episode videos, implicitly assuming that all videos have
        been encoded the same way. Also, this means it assumes the first episode exists.
        """
        if video_key is not None and video_key not in self.video_keys:
            raise ValueError(f"Video key {video_key} not found in dataset")

        video_keys = [video_key] if video_key is not None else self.video_keys
        for key in video_keys:
            if not self.features[key].get("info", None):
                rel_path = (
                    self.get_video_file_path(ep_index=0, vid_key=key)
                    if LEROBOT_CODEBASE_VERSION == "v2.1"
                    else self.video_path.format(video_key=key, chunk_index=0, file_index=0)
                )
                video_path = self.root / rel_path
                self.info["features"][key]["info"] = get_video_info(video_path)


class OmniGibsonLeRobotDataset(LeRobotDataset):
    _MIN_DEPTH = None
    _MAX_DEPTH = None
    _DEPTH_SHIFT = None

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        root: str | Path | None = None,
        robot_type: str | None = None,
        use_videos: bool = True,
        tolerance_s: float = 1e-4,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
        min_depth=OU.MIN_DEPTH,
        max_depth=OU.MAX_DEPTH,
        depth_shift=OU.DEPTH_SHIFT,
    ) -> "OmniGibsonLeRobotDataset":
        # Set depth ranges
        cls.set_cls_depth_range(min_depth=min_depth, max_depth=max_depth, depth_shift=depth_shift)

        ##############
        """Create a LeRobot Dataset from scratch in order to record data."""
        obj = cls.__new__(cls)
        obj.meta = OmniGibsonLeRobotDatasetMetadata.create(
            repo_id=repo_id,
            fps=fps,
            robot_type=robot_type,
            features=features,
            root=root,
            use_videos=use_videos,
        )
        obj.repo_id = obj.meta.repo_id
        obj.root = obj.meta.root
        obj.revision = None
        obj.tolerance_s = tolerance_s
        obj.image_writer = None
        obj.batch_encoding_size = batch_encoding_size
        obj.episodes_since_last_encoding = 0

        if image_writer_processes or image_writer_threads:
            obj.start_image_writer(image_writer_processes, image_writer_threads)

        # TODO(aliberts, rcadene, alexander-soare): Merge this with OnlineBuffer/DataBuffer
        obj.episode_buffer = obj.create_episode_buffer()

        obj.episodes = None
        obj.hf_dataset = obj.create_hf_dataset()
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.delta_indices = None
        obj.video_backend = video_backend if video_backend is not None else get_safe_default_codec()

        if LEROBOT_CODEBASE_VERSION == "v2.1":
            obj.episode_data_index = None
        else:
            obj.writer = None
            obj.latest_episode = None
            obj._current_file_start_frame = None
            # Initialize tracking for incremental recording
            obj._lazy_loading = False
            obj._recorded_frames = 0
            obj._writer_closed_for_reading = False

        ##############

        # Add og metadata dict
        obj.min_depth = min_depth
        obj.max_depth = max_depth
        obj.depth_shift = depth_shift
        obj.meta.info["omnigibson"] = dict()

        return obj

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
        features: Sequence[str] | None = None,
        min_depth=None,
        max_depth=None,
        depth_shift=None,
    ):
        ###################
        Dataset.__init__(self)
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else LEROBOT_CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.delta_indices = None
        self.batch_encoding_size = batch_encoding_size
        self.episodes_since_last_encoding = 0

        # Unused attributes
        self.image_writer = None
        self.episode_buffer = None

        if LEROBOT_CODEBASE_VERSION != "v2.1":
            # Assume v3.0
            assert LEROBOT_CODEBASE_VERSION == "v3.0"
            self.writer = None
            self.latest_episode = None
            self._current_file_start_frame = None  # Track the starting frame index of the current parquet file

        self.root.mkdir(exist_ok=True, parents=True)

        # Load metadata
        self.meta = OmniGibsonLeRobotDatasetMetadata(
            self.repo_id,
            self.root,
            self.revision,
            force_cache_sync=force_cache_sync,
            features=features,
        )

        if LEROBOT_CODEBASE_VERSION == "v2.1":
            if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
                episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
                self.stats = aggregate_stats(episodes_stats)

            # Load actual data
            try:
                if force_cache_sync:
                    raise FileNotFoundError
                assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
                self.hf_dataset = self.load_hf_dataset()
            except (AssertionError, FileNotFoundError, NotADirectoryError):
                self.revision = get_safe_version(self.repo_id, self.revision)
                self.download_episodes(download_videos)
                self.hf_dataset = self.load_hf_dataset()

            # Import here because it is v2.1 specific import
            from lerobot.datasets.utils import get_episode_data_index

            self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)

        else:
            # Track dataset state for efficient incremental writing
            self._lazy_loading = False
            self._recorded_frames = self.meta.total_frames
            self._writer_closed_for_reading = False

            # Load actual data
            try:
                if force_cache_sync:
                    raise FileNotFoundError
                self.hf_dataset = self.load_hf_dataset()
                # Check if cached dataset contains all requested episodes
                if not self._check_cached_episodes_sufficient():
                    raise FileNotFoundError("Cached dataset doesn't contain all requested episodes")
            except (AssertionError, FileNotFoundError, NotADirectoryError):
                if is_valid_version(self.revision):
                    self.revision = get_safe_version(self.repo_id, self.revision)
                self.download(download_videos)
                self.hf_dataset = self.load_hf_dataset()

            # Create mapping from absolute indices to relative indices when only a subset of the episodes are loaded
            # Build a mapping: absolute_index -> relative_index_in_filtered_dataset
            self._absolute_to_relative_idx = None
            if self.episodes is not None:
                self._absolute_to_relative_idx = {
                    abs_idx.item() if isinstance(abs_idx, th.Tensor) else abs_idx: rel_idx
                    for rel_idx, abs_idx in enumerate(self.hf_dataset["index"])
                }

        # Setup delta_indices
        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

        ###################

        # If depth values not set, load from dataset's meta info and set them internally
        og_info = self.meta.info.get("omnigibson", {})
        self.min_depth = og_info.get("min_depth", OU.MIN_DEPTH) if min_depth is None else min_depth
        self.max_depth = og_info.get("max_depth", OU.MAX_DEPTH) if max_depth is None else max_depth
        self.depth_shift = og_info.get("depth_shift", OU.DEPTH_SHIFT) if depth_shift is None else depth_shift

        # Update depth values
        self.set_depth_range(min_depth=self.min_depth, max_depth=self.max_depth, depth_shift=self.depth_shift)

    def __getitem__(self, idx) -> dict:
        item = super().__getitem__(idx)

        for name, val in item.items():
            if "depth" in name and val.ndim > 1:
                # Decode
                if OU.USE_DEPTH_RGB_ENCODING:
                    shape = list(range(val.ndim - 3)) + [-2, -1, -3]
                    val = th.from_numpy(
                        OU.rgb2depth(
                            rgb=(val.permute(shape).numpy() * 255).astype(
                                np.uint8
                            ),  # Prune final dimension so shape is (H, W)
                            min_depth=self.min_depth,
                            max_depth=self.max_depth,
                            err_depth=0.0,
                        )
                    ).unsqueeze(-3)
                else:
                    val = OU.dequantize_depth(
                        val,
                        min_depth=self.min_depth,
                        max_depth=self.max_depth,
                        shift=self.depth_shift,
                    )

                item[name] = val

        return item

    @classmethod
    def set_cls_depth_range(cls, min_depth, max_depth, depth_shift):
        cls._MIN_DEPTH = min_depth
        cls._MAX_DEPTH = max_depth
        cls._DEPTH_SHIFT = depth_shift

        # Also make sure to set the root class's values since that is what is used when overwriting the lerobot function
        if cls != OmniGibsonLeRobotDataset:
            OmniGibsonLeRobotDataset.set_cls_depth_range(
                min_depth=min_depth, max_depth=max_depth, depth_shift=depth_shift
            )

    def set_depth_range(self, min_depth, max_depth, depth_shift):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_shift = depth_shift
        self.set_cls_depth_range(min_depth=min_depth, max_depth=max_depth, depth_shift=depth_shift)

    def set_omnigibson_metadata(self, key, value):
        """
        Adds omnigibson-relevant metadata to this dataset's metadata info dict

        Args:
            key (str): Entry to add / modify
            value (Any): Value to store
        """
        if "omnigibson" not in self.meta.info:
            self.meta.info["omnigibson"] = dict()

        self.meta.info["omnigibson"][key] = value

    def get_omnigibson_metadata(self, key):
        """
        Grabs omnigibson-relevant metadata from this dataset's metadata info dict. Returns None if not found

        Args:
            key (str): Entry to grab

        Returns:
            Any: Corresponding metadata value
        """
        assert "omnigibson" in self.meta.info, "Found no omnigibson metadata in this dataset!"
        return self.meta.info["omnigibson"].get(key, None)

    @classmethod
    def image_array_to_pil_image(cls, image_array: np.ndarray, range_check: bool = True) -> PIL.Image.Image:
        # Overwrites the native LeRobot implementation to include depth processing
        # TODO(aliberts): handle 1 channel and 4 for depth images
        if image_array.ndim != 3:
            raise ValueError(f"The array has {image_array.ndim} dimensions, but 3 is expected for an image.")

        if image_array.shape[0] in {1, 3}:  # Handle RGB or Depth
            # Transpose from pytorch convention (C, H, W) to (H, W, C)
            image_array = image_array.transpose(1, 2, 0)

        elif image_array.shape[-1] not in {1, 3}:
            raise NotImplementedError(
                f"The image has {image_array.shape[-1]} channels, but 1 or 3 is required for now."
            )

        if image_array.dtype != np.uint8:
            if image_array.shape[-1] == 3:
                if range_check:
                    # RGB, handle edge cases
                    max_ = image_array.max().item()
                    min_ = image_array.min().item()
                    if max_ > 1.0 or min_ < 0.0:
                        raise ValueError(
                            "The image data type is float, which requires values in the range [0.0, 1.0]. "
                            f"However, the provided range is [{min_}, {max_}]. Please adjust the range or "
                            "provide a uint8 image with values in the range [0, 255]."
                        )
                image_array = (image_array * 255).astype(np.uint8)
            else:
                # This is depth, so normalize before storing as image
                if OU.USE_DEPTH_RGB_ENCODING:
                    image_array = OU.depth2rgb(
                        depth=image_array[:, :, 0],  # Prune final dimension so shape is (H, W)
                        min_depth=cls._MIN_DEPTH,
                        max_depth=cls._MAX_DEPTH,
                        disparity=False,
                    )
                else:
                    image_array = OU.quantize_depth(
                        depth=image_array,
                        min_depth=cls._MIN_DEPTH,
                        max_depth=cls._MAX_DEPTH,
                        shift=cls._DEPTH_SHIFT,
                    )[:, :, 0]  # Prune final dimension so shape is (H, W)

        return PIL.Image.fromarray(image_array)


# Define separate classes for V2 vs V3 versions of the dataset
class OmniGibsonLeRobotV2Dataset(OmniGibsonLeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        # video_backend: str = "pyav",
        batch_encoding_size: int = 1,
        # chunk_streaming_using_keyframe: bool = True,
        # shuffle: bool = True,
        # seed: int = 0,
        features: Sequence[str] | None = None,
        min_depth=None,
        max_depth=None,
        depth_shift=None,
    ):
        # Call super
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            revision=revision,
            force_cache_sync=force_cache_sync,
            download_videos=download_videos,
            video_backend=video_backend,
            batch_encoding_size=batch_encoding_size,
            # chunk_streaming_using_keyframe=chunk_streaming_using_keyframe,
            # shuffle=shuffle,
            # seed=seed,
            features=features,
            min_depth=min_depth,
            max_depth=max_depth,
            depth_shift=depth_shift,
        )

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        # See: https://github.com/huggingface/lerobot/issues/93#issuecomment-3182826868
        posted_data = {}
        for key, indices in query_indices.items():
            if key not in self.meta.video_keys:
                posted_data[key] = th.stack([self.hf_dataset[idx][key] for idx in indices])
        return posted_data

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                timestamps = self.hf_dataset[query_indices[key]]["timestamp"]
                query_timestamps[key] = th.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, th.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.
        """
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            # Use our custom decode_video_frames function
            frames = decode_video_frames(
                video_path,
                query_ts,
                self.tolerance_s,
                self.video_backend,
                self.min_depth,
                self.max_depth,
                self.depth_shift,
            )
            item[vid_key] = frames.squeeze(0)

        return item

    def encode_episode_videos(self, episode_index: int) -> None:
        """
        Use ffmpeg to convert frames stored as png into mp4 videos.
        Note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
        since video encoding with ffmpeg is already using multithreading.

        This method handles video encoding steps:
        - Video encoding via ffmpeg
        - Video info updating in metadata
        - Raw image cleanup

        Args:
            episode_index (int): Index of the episode to encode.
        """
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            if video_path.is_file():
                # Skip if video is already encoded. Could be the case when resuming data recording.
                continue
            img_dir = self._get_image_file_path(episode_index=episode_index, image_key=key, frame_index=0).parent
            # Use our own custom video encoder function
            mode = "depth" if "depth" in key else "rgb"
            encode_video_frames(img_dir, video_path, self.fps, overwrite=True, mode=mode)
            shutil.rmtree(img_dir)

        # Update video info (only needed when first episode is encoded since it reads from episode 0)
        if len(self.meta.video_keys) > 0 and episode_index == 0:
            self.meta.update_video_info()
            write_info(self.meta.info, self.meta.root)  # ensure video info always written properly

    def save_episode(self, episode_data: dict | None = None) -> None:
        # Run super
        super().save_episode(episode_data=episode_data)

        # Update depth values
        self.meta.info["omnigibson"]["min_depth"] = self.min_depth
        self.meta.info["omnigibson"]["max_depth"] = self.max_depth
        self.meta.info["omnigibson"]["depth_shift"] = self.depth_shift


class OmniGibsonLeRobotV3Dataset(OmniGibsonLeRobotDataset):
    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, th.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.
        """
        ep = self.meta.episodes[ep_idx]
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            # Episodes are stored sequentially on a single mp4 to reduce the number of files.
            # Thus we load the start timestamp of the episode on this mp4 and,
            # shift the query timestamp accordingly.
            from_timestamp = ep[f"videos/{vid_key}/from_timestamp"]
            shifted_query_ts = [from_timestamp + ts for ts in query_ts]

            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            # Use our custom decode_video_frames function
            frames = decode_video_frames(
                video_path,
                shifted_query_ts,
                self.tolerance_s,
                self.video_backend,
                self.min_depth,
                self.max_depth,
                self.depth_shift,
            )
            item[vid_key] = frames.squeeze(0)

        return item

    def _encode_temporary_episode_video(self, video_key: str, episode_index: int) -> Path:
        # Use our custom video worker
        return _encode_video_worker(video_key, episode_index, self.root, self.fps)

    def save_episode(
        self,
        episode_data: dict | None = None,
        parallel_encoding: bool = True,
    ) -> None:
        """
        This will save to disk the current episode in self.episode_buffer.

        Video encoding is handled automatically based on batch_encoding_size:
        - If batch_encoding_size == 1: Videos are encoded immediately after each episode
        - If batch_encoding_size > 1: Videos are encoded in batches.

        Args:
            episode_data (dict | None, optional): Dict containing the episode data to save. If None, this will
                save the current episode in self.episode_buffer, which is filled with 'add_frame'. Defaults to
                None.
            parallel_encoding (bool, optional): If True, encode videos in parallel using ProcessPoolExecutor.
                Defaults to True on Linux, False on macOS as it tends to use all the CPU available already.
        """
        episode_buffer = episode_data if episode_data is not None else self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        # size and task are special cases that won't be added to hf_dataset
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # Update tasks and task indices with new tasks if any
        self.meta.save_episode_tasks(episode_tasks)

        # Given tasks in natural language, find their corresponding task indices
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            # index, episode_index, task_index are already processed above, and image and video
            # are processed separately by storing image path and frame info as meta data
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        # Wait for image writer to end, so that episode stats over images can be computed
        self._wait_image_writer()

        # TODO: Handle depth
        ep_stats = compute_episode_stats(episode_buffer, self.features)

        ep_metadata = self._save_episode_data(episode_buffer)
        has_video_keys = len(self.meta.video_keys) > 0
        use_batched_encoding = self.batch_encoding_size > 1

        # We cannot use parallel encoding if we're using depth and writing to RGB -- this is because the GPU resource
        # handling seems to crash upon initialization in the multi-process setting

        if parallel_encoding and any("depth" in key for key in self.meta.video_keys) and OU.USE_DEPTH_RGB_ENCODING:
            logging.warning("Cannot use parallel_encoding with depth keys and depth->RGB encoding. Setting to False.")
            parallel_encoding = False

        if has_video_keys and not use_batched_encoding:
            num_cameras = len(self.meta.video_keys)
            if parallel_encoding and num_cameras > 1:
                # TODO(Steven): Ideally we would like to control the number of threads per encoding such that:
                # num_cameras * num_threads = (total_cpu -1)
                with concurrent.futures.ProcessPoolExecutor(max_workers=num_cameras) as executor:
                    future_to_key = {
                        executor.submit(
                            _encode_video_worker,
                            video_key,
                            episode_index,
                            self.root,
                            self.fps,
                        ): video_key
                        for video_key in self.meta.video_keys
                    }

                    results = {}
                    for future in concurrent.futures.as_completed(future_to_key):
                        video_key = future_to_key[future]
                        try:
                            temp_path = future.result()
                            results[video_key] = temp_path
                        except Exception as exc:
                            logging.error(f"Video encoding failed for {video_key}: {exc}")
                            raise exc

                for video_key in self.meta.video_keys:
                    temp_path = results[video_key]
                    ep_metadata.update(self._save_episode_video(video_key, episode_index, temp_path=temp_path))
            else:
                for video_key in self.meta.video_keys:
                    ep_metadata.update(self._save_episode_video(video_key, episode_index))

        # `meta.save_episode` need to be executed after encoding the videos
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats, ep_metadata)

        if has_video_keys and use_batched_encoding:
            # Check if we should trigger batch encoding
            self.episodes_since_last_encoding += 1
            if self.episodes_since_last_encoding == self.batch_encoding_size:
                start_ep = self.num_episodes - self.batch_encoding_size
                end_ep = self.num_episodes
                self._batch_save_episode_video(start_ep, end_ep)
                self.episodes_since_last_encoding = 0

        if not episode_data:
            # Reset episode buffer and clean up temporary images (if not already deleted during video encoding)
            self.clear_episode_buffer(delete_images=len(self.meta.image_keys) > 0)

        # Update depth values
        self.meta.info["omnigibson"]["min_depth"] = self.min_depth
        self.meta.info["omnigibson"]["max_depth"] = self.max_depth
        self.meta.info["omnigibson"]["depth_shift"] = self.depth_shift


# Save internal methods to force handling depth during image saving
liw.image_array_to_pil_image = OmniGibsonLeRobotDataset.image_array_to_pil_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="/scr/behavior/2025-challenge-demos")
    args = parser.parse_args()

    # expand root
    data_dir = os.path.expanduser(args.data_dir)
    print("Generating task JSON...")
    num_tasks = generate_task_json(data_dir, credentials_path="~/Documents/credentials")
    print("Generating episode JSON...")
    num_episodes, num_frames = generate_episode_json(data_dir)
    print(num_tasks, num_episodes, num_frames)
    print("Generating info JSON...")
    generate_info_json(data_dir, fps=30, total_episodes=num_episodes, total_tasks=num_tasks, total_frames=num_frames)
