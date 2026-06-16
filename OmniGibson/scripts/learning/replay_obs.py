import argparse
import csv
import omnigibson as og
import os
import yaml
from omnigibson.envs import HDF5PlaybackWrapper, LeRobotPlaybackWrapper
from omnigibson.eval.utils.dataset_utils import makedirs_with_mode
from omnigibson.eval.utils.eval_utils import PROPRIOCEPTION_INDICES
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import create_module_logger


log = create_module_logger(module_name="replay_obs")
log.setLevel(20)

gm.RENDER_VIEWER_CAMERA = False
gm.DEFAULT_VIEWER_WIDTH = 128
gm.DEFAULT_VIEWER_HEIGHT = 128


def _load_challenge_available_tasks() -> dict:
    available_tasks = {}
    for year in ("2025", "2026"):
        task_cfg_path = os.path.join(
            gm.DATA_PATH, f"{year}-challenge-task-instances", "metadata", "available_tasks.yaml"
        )
        if not os.path.exists(task_cfg_path):
            continue
        with open(task_cfg_path, "r") as f:
            available_tasks.update(yaml.safe_load(f))
    return available_tasks


def _load_challenge_task_ids() -> dict[str, int]:
    task_ids = {}
    for year, filename in (("2025", "B50_task_misc.csv"), ("2026", "B100_task_misc.csv")):
        task_misc_path = os.path.join(gm.DATA_PATH, f"{year}-challenge-task-instances", "metadata", filename)
        if not os.path.exists(task_misc_path):
            continue
        with open(task_misc_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                task_ids[row["Task"]] = int(row["Task ID"])
    return task_ids


def _infer_task_id_from_demo_id(demo_id: int) -> int:
    return demo_id // 10000


def _get_task_name_from_task_id(task_id: int) -> str:
    task_names_by_id = {task_id: task_name for task_name, task_id in _load_challenge_task_ids().items()}
    if task_id not in task_names_by_id:
        raise KeyError(f"Task ID {task_id} inferred from demo_id was not found in challenge task metadata")
    return task_names_by_id[task_id]


def _find_full_scene_file(task_name: str, scene_model: str) -> str:
    for year in ("2026", "2025"):
        task_scene_file_folder = os.path.join(
            gm.DATA_PATH, f"{year}-challenge-task-instances", "scenes", scene_model, "json"
        )
        if not os.path.isdir(task_scene_file_folder):
            continue
        for file in os.listdir(task_scene_file_folder):
            if task_name in file and file.endswith(".json") and "partial_rooms" not in file:
                return os.path.join(task_scene_file_folder, file)
    raise FileNotFoundError(f"No full scene file found for task '{task_name}' and scene '{scene_model}'")


def _load_room_instances(task_name: str) -> list[str]:
    for year, filename in (("2026", "B100_task_misc.csv"), ("2025", "B50_task_misc.csv")):
        task_misc_path = os.path.join(gm.DATA_PATH, f"{year}-challenge-task-instances", "metadata", filename)
        if not os.path.exists(task_misc_path):
            continue
        with open(task_misc_path, newline="", encoding="utf-8") as f:
            task_misc_csv = csv.reader(f, delimiter=",", quotechar='"')
            for row in task_misc_csv:
                if task_name in row[1]:
                    return row[2].strip().split("\n")
    raise FileNotFoundError(f"No task misc room instances found for task '{task_name}'")


def replay_hdf5_file(
    data_folder: str,
    task_id: int,
    demo_id: int,
    output_format: str,
    flush_every_n_steps: int,
    lerobot_repo_id: str | None = None,
    lerobot_root_dir: str | None = None,
    overwrite_lerobot: bool = True,
    use_longest_demo: bool = False,
) -> int:
    """
    Replays a single HDF5 file and saves data to the specified format.

    Args:
        data_folder: data folder
        task_id: ID of the task to replay
        demo_id: ID of the demo to replay
        output_format: Output format, "hdf5" or "lerobot"
        flush_every_n_steps: Number of steps to flush the data after
        use_longest_demo: If True, replay the demo with the most steps instead of the last demo group

    Returns:
        episode_id: ID of the episode
    """
    task_name = _get_task_name_from_task_id(task_id)
    replay_dir = os.path.join(data_folder, "replayed")
    makedirs_with_mode(replay_dir)

    gm.ENABLE_TRANSITION_RULES = False

    robot_sensor_config = {
        "VisionSensor": {
            "sensor_kwargs": {
                "image_height": 480,
                "image_width": 480,
            },
        },
        "zed_link:Camera:0": {
            "sensor_kwargs": {
                "horizontal_aperture": 40.0,
                "image_height": 720,
                "image_width": 720,
            },
        },
    }
    available_tasks = _load_challenge_available_tasks()
    if task_name not in available_tasks:
        raise KeyError(f"Task '{task_name}' not found in available challenge task metadata")
    scene_model = available_tasks[task_name][0]["scene_model"]
    full_scene_file = _find_full_scene_file(task_name=task_name, scene_model=scene_model)
    load_room_instances = _load_room_instances(task_name=task_name)

    input_path = f"{data_folder}/2026-challenge-rawdata/task-{task_id:04d}/episode_{demo_id:08d}.hdf5"

    common_kwargs = dict(
        input_path=input_path,
        full_scene_file=full_scene_file,
        load_room_instances=load_room_instances,
        robot_sensor_config=robot_sensor_config,
        n_render_iterations=1,
        flush_every_n_steps=flush_every_n_steps,
        flush_every_n_traj=1,
        include_robot_control=False,
        robot_proprio_keys=list(PROPRIOCEPTION_INDICES["R1Pro"].keys()),
        robot_obs_modalities=["proprio", "rgb", "depth_linear"],
        include_contacts=False,
    )

    if output_format == "hdf5":
        output_path = os.path.join(replay_dir, f"episode_{demo_id:08d}.hdf5")
        kwargs = dict(
            **common_kwargs,
            output_path=output_path,
            compression={"compression": "lzf"},
        )
        env = HDF5PlaybackWrapper.create_from_hdf5(**kwargs)
    else:
        output_path = f"b1k/{task_name}"
        if lerobot_repo_id is not None:
            output_path = lerobot_repo_id
        root_dir = lerobot_root_dir or os.path.join(data_folder, "lerobot")
        makedirs_with_mode(root_dir)
        kwargs = dict(
            **common_kwargs,
            output_path=output_path,
            root_dir=root_dir,
            overwrite=overwrite_lerobot,
            robot_type="R1Pro",
            task_name=task_name,
            include_task_obs=False,
        )
        env = LeRobotPlaybackWrapper.create_from_hdf5(**kwargs)

    env.load_observation_space()

    demo_ids = sorted(int(key.split("_", 1)[1]) for key in env.input_hdf5["data"].keys() if key.startswith("demo_"))
    if not demo_ids:
        raise ValueError(f"No demo groups found in {input_path}")

    if use_longest_demo:
        episode_id = max(
            demo_ids,
            key=lambda cur_episode_id: env.input_hdf5["data"][f"demo_{cur_episode_id}"].attrs["num_samples"],
        )
        selection_reason = "most steps"
    else:
        episode_id = demo_ids[-1]
        selection_reason = "last demo"
    num_samples = env.input_hdf5["data"][f"demo_{episode_id}"].attrs["num_samples"]
    log.info(f" >>> Replaying episode {episode_id} ({selection_reason}) with {num_samples} steps")

    env.playback_episode(episode_id=episode_id, record_data=True)

    log.info("Playback complete. Saving data...")
    env.save_data()

    log.info(f"Successfully processed episode_{demo_id:08d}")
    return episode_id


def main():
    parser = argparse.ArgumentParser(description="Replay HDF5 files and save data")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to the data folder")
    parser.add_argument("--demo_id", type=int, required=True, help="Demo ID to process")
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["hdf5", "lerobot"],
        default="hdf5",
        help="Output format: hdf5, lerobot",
    )
    parser.add_argument("--flush_every_n_steps", type=int, default=1000, help="Flush data every N steps")
    parser.add_argument(
        "--lerobot_repo_id",
        type=str,
        default=None,
        help="LeRobot repo id / relative output path. Defaults to b1k/<task_name>.",
    )
    parser.add_argument(
        "--lerobot_root_dir",
        type=str,
        default=None,
        help="Root directory for LeRobot output. Defaults to <data_folder>/lerobot.",
    )
    parser.add_argument(
        "--resume_lerobot",
        action="store_true",
        help="Append to an existing LeRobot dataset instead of overwriting it.",
    )
    parser.add_argument(
        "--use_longest_demo",
        "--use-longest-demo",
        action="store_true",
        help="Replay the demo with the most steps instead of the last demo group.",
    )

    args = parser.parse_args()
    task_id = _infer_task_id_from_demo_id(args.demo_id)

    if not os.path.exists(
        f"{args.data_folder}/2026-challenge-rawdata/task-{task_id:04d}/episode_{args.demo_id:08d}.hdf5"
    ):
        raise FileNotFoundError(f"Error: File episode_{args.demo_id:08d}.hdf5 does not exist under {args.data_folder}")

    _ = replay_hdf5_file(
        data_folder=args.data_folder,
        task_id=task_id,
        demo_id=args.demo_id,
        output_format=args.output_format,
        flush_every_n_steps=args.flush_every_n_steps,
        lerobot_repo_id=args.lerobot_repo_id,
        lerobot_root_dir=args.lerobot_root_dir,
        overwrite_lerobot=not args.resume_lerobot,
        use_longest_demo=args.use_longest_demo,
    )

    log.info("All done!")
    og.shutdown()


if __name__ == "__main__":
    main()
