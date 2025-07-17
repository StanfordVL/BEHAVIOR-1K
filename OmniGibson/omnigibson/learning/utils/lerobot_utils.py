import argparse
import json
import os
from typing import Tuple
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class BehaviorLeRobotDataset(LeRobotDataset):
    def get_episodes_file_paths(self) -> list[str]:
        """
        Overwrite the original method to use the episodes indices instead of range(self.meta.total_episodes)
        """
        episodes = self.episodes if self.episodes is not None else list(self.meta.episodes.keys())
        fpaths = [str(self.meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        if len(self.meta.video_keys) > 0:
            video_files = [
                str(self.meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self.meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files

        return fpaths


def generate_task_json(data_dir: str) -> int:
    num_tasks = len(TASK_INDICES)
    
    with open(f"{data_dir}/meta/tasks.jsonl", "w") as f:
        for task_index, task_name in TASK_INDICES.items():
            json.dump({"task_index": task_index, "task": task_name}, f)
            f.write("\n")
    return num_tasks


def generate_episode_json(data_dir: str) -> Tuple[int, int]:
    assert os.path.exists(f"{data_dir}/meta/tasks.jsonl"), "Task JSON does not exist!"
    assert os.path.exists(f"{data_dir}/meta/episodes"), "Episode metadata directory does not exist!"
    with open(f"{data_dir}/meta/tasks.jsonl", "r") as f:
        task_json = [json.loads(line) for line in f]
    num_frames = 0
    num_episodes = 0
    with open(f"{data_dir}/meta/episodes.jsonl", "w") as out_f:
        with open(f"{data_dir}/meta/episodes_stats.jsonl", "w") as out_stats_f:
            for task_info in task_json:
                task_index = task_info["task_index"]
                task_name = task_info["task"]
                for episode_name in sorted(os.listdir(f"{data_dir}/meta/episodes/task-{task_index:04d}")):
                    with open(f"{data_dir}/meta/episodes/task-{task_index:04d}/{episode_name}", "r") as f:
                        episode_info = json.load(f)
                        episode_index = int(episode_name.split(".")[0].split("_")[-1])
                        episode_json = {
                            "episode_index": episode_index,
                            "tasks": [task_name],
                            "length": episode_info["num_samples"]
                        }
                        episode_stats_json = {
                            "episode_index": episode_index,
                            "tasks": [task_name],
                            "stats": {
                                "obs": {
                                    "min": np.array([episode_info["num_samples"]]).tolist(),
                                    "max": np.array([episode_info["num_samples"]]).tolist(),
                                    "mean": np.array([episode_info["num_samples"]]).tolist(),
                                    "std": np.array([episode_info["num_samples"]]).tolist(),
                                    "count": np.array([episode_info["num_samples"]]).tolist()
                                }
                            }
                        }
                        num_episodes += 1
                        num_frames += episode_info["num_samples"]
                    json.dump(episode_json, out_f)
                    out_f.write("\n")
                    json.dump(episode_stats_json, out_stats_f)
                    out_stats_f.write("\n")
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
        "total_videos": total_episodes * 6,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {
            "train": "0:" + str(total_episodes),
        },
        "data_path": "data/task-{episode_chunk:04d}/episode_{episode_index:07d}.parquet",
        "video_path": "videos/task-{episode_chunk:04d}/{video_key}/episode_{episode_index:07d}.mp4",
        "features": {
            "observation.images.rgb.left_wrist": {
                "dtype": "video",
                "shape": [
                    480,
                    480,
                    3
                ],
                "names": [
                    "height",
                    "width",
                    "rgb"
                ],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 480,
                    "video.width": 480,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False
                }
            },
            "observation.images.rgb.right_wrist": {
                "dtype": "video",
                "shape": [
                    480,
                    480,
                    3
                ],
                "names": [
                    "height",
                    "width",
                    "rgb"
                ],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 480,
                    "video.width": 480,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False
                }
            },
            "observation.images.rgb.head": {
                "dtype": "video",
                "shape": [
                    720,
                    720,
                    3
                ],
                "names": [
                    "height",
                    "width",
                    "rgb"
                ],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 720,
                    "video.width": 720,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False
                }
            },
            "observation.images.depth.left_wrist": {
                "dtype": "video",
                "shape": [
                    480,
                    480,
                    3
                ],
                "names": [
                    "height",
                    "width",
                    "depth"
                ],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 480,
                    "video.width": 480,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p16le",
                    "video.is_depth_map": True,
                    "has_audio": False
                }
            },
            "observation.images.depth.right_wrist": {
                "dtype": "video",
                "shape": [
                    480,
                    480,
                    3
                ],
                "names": [
                    "height",
                    "width",
                    "depth"
                ],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 480,
                    "video.width": 480,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p16le",
                    "video.is_depth_map": True,
                    "has_audio": False
                }
            },
            "observation.images.depth.head": {
                "dtype": "video",
                "shape": [
                    720,
                    720,
                    3
                ],
                "names": [
                    "height",
                    "width",
                    "depth"
                ],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 720,
                    "video.width": 720,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p16le",
                    "video.is_depth_map": True,
                    "has_audio": False
                }
            },
            "observation.images.seg_instance_id.left_wrist": {
                "dtype": "video",
                "shape": [
                    480,
                    480,
                    3
                ],
                "names": [
                    "height",
                    "width",
                    "seg_instance_id"
                ],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 480,
                    "video.width": 480,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p", 
                    "has_audio": False
                }
            },
            "observation.images.seg_instance_id.right_wrist": {
                "dtype": "video",
                "shape": [
                    480,
                    480,
                    3
                ],
                "names": [
                    "height",
                    "width",
                    "seg_instance_id"
                ],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 480,
                    "video.width": 480,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p", 
                    "has_audio": False
                }
            },
            "observation.images.seg_instance_id.head": {
                "dtype": "video",
                "shape": [
                    720,
                    720,
                    3
                ],
                "names": [
                    "height",
                    "width",
                    "seg_instance_id"
                ],
                "info": {
                    "video.fps": 30.0,
                    "video.height": 720,
                    "video.width": 720,
                    "video.channels": 3,
                    "video.codec": "libx265",
                    "video.pix_fmt": "yuv420p", 
                    "has_audio": False
                }
            },
            "action": {
                "dtype": "float32",
                "shape": [
                    23
                ],
                "names": None
            },
            "timestamp": {
                "dtype": "float64",
                "shape": [
                    1
                ],
                "names": None
            },
            "episode_index": {
                "dtype": "int64",
                "shape": [
                    1
                ],
                "names": None
            },
            "index": {
                "dtype": "int64",
                "shape": [
                    1
                ],
                "names": None
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [None],
                "names": None
            },
        }
    }

    with open(f"{data_dir}/meta/info.json", "w") as f:
        json.dump(info, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, default="/home/svl/Documents/Files/behavior")
    args = parser.parse_args()

    num_tasks = generate_task_json(args.data_dir)
    num_episodes, num_frames = generate_episode_json(args.data_dir)
    print(num_tasks, num_episodes, num_frames)

    generate_info_json(args.data_dir, fps=30, total_episodes=num_episodes, total_tasks=num_tasks, total_frames=num_frames)
