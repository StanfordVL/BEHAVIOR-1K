import argparse
import csv
import omnigibson as og
import os
import time
from omnigibson.envs import HDF5PlaybackWrapper, LeRobotPlaybackWrapper
from omnigibson.learning.utils.dataset_utils import update_google_sheet, makedirs_with_mode
from omnigibson.learning.utils.eval_utils import (
    PROPRIOCEPTION_INDICES,
    TASK_NAMES_TO_INDICES,
    TASK_INDICES_TO_NAMES,
    HEAD_RESOLUTION,
    WRIST_RESOLUTION,
)
from omnigibson.macros import gm
from omnigibson.utils.ui_utils import create_module_logger


log = create_module_logger(module_name="replay_obs")
log.setLevel(20)

gm.RENDER_VIEWER_CAMERA = False
gm.DEFAULT_VIEWER_WIDTH = 128
gm.DEFAULT_VIEWER_HEIGHT = 128

FLUSH_EVERY_N_STEPS = 500


def replay_hdf5_file(
    data_folder: str,
    task_id: int,
    demo_id: int,
    output_format: str = "hdf5",
    flush_every_n_steps: int = 500,
) -> int:
    """
    Replays a single HDF5 file and saves data to the specified format.

    Args:
        data_folder: data folder
        task_id: ID of the task to replay
        demo_id: ID of the demo to replay
        output_format: Output format, either "hdf5" or "lerobot"
        flush_every_n_steps: Number of steps to flush the data after

    Returns:
        episode_id: ID of the episode
    """
    task_name = TASK_INDICES_TO_NAMES[task_id]
    replay_dir = os.path.join(data_folder, "replayed")
    makedirs_with_mode(replay_dir)

    gm.ENABLE_TRANSITION_RULES = False

    modalities = ["rgb", "depth_linear"]

    robot_sensor_config = {
        "VisionSensor": {
            "modalities": modalities,
            "sensor_kwargs": {
                "image_height": WRIST_RESOLUTION[0],
                "image_width": WRIST_RESOLUTION[1],
            },
        },
    }

    task_scene_file_folder = os.path.join(
        os.path.dirname(os.path.dirname(og.__path__[0])), "joylo", "sampled_task", task_name
    )
    full_scene_file = None
    for file in os.listdir(task_scene_file_folder):
        if file.endswith(".json") and "partial_rooms" not in file:
            full_scene_file = os.path.join(task_scene_file_folder, file)
    assert full_scene_file is not None, f"No full scene file found in {task_scene_file_folder}"

    load_room_instances = None
    try:
        with open(
            f"{gm.DATA_PATH}/2025-challenge-task-instances/metadata/B50_task_misc.csv", newline="", encoding="utf-8"
        ) as f:
            task_misc_csv = csv.reader(f, delimiter=",", quotechar='"')
            for row in task_misc_csv:
                if task_name in row[1]:
                    load_room_instances = row[2].strip().split("\n")
                    break
    except FileNotFoundError as e:
        log.error(
            "No B50_task_misc.csv file found in 2025-challenge-task-instances/metadata folder. Please ensure the dataset is up to date."
        )
        raise e
    assert load_room_instances is not None, "load room instance not found!"

    input_path = f"{data_folder}/2025-challenge-rawdata/task-{task_id:04d}/episode_{demo_id:08d}.hdf5"

    if output_format == "hdf5":
        output_path = os.path.join(replay_dir, f"episode_{demo_id:08d}.hdf5")
        env = HDF5PlaybackWrapper.create_from_hdf5(
            input_path=input_path,
            output_path=output_path,
            compression={"compression": "lzf"},
            robot_obs_modalities=["proprio"],
            robot_proprio_keys=list(PROPRIOCEPTION_INDICES["R1Pro"].keys()),
            robot_sensor_config=robot_sensor_config,
            external_sensors_config=dict(),
            n_render_iterations=3,
            flush_every_n_traj=1,
            flush_every_n_steps=flush_every_n_steps,
            full_scene_file=full_scene_file,
            include_robot_control=False,
            include_contacts=False,
            load_room_instances=load_room_instances,
        )
    else:
        output_path = f"{task_name}_episode_{demo_id:08d}"
        root_dir = os.path.join(data_folder, "lerobot")
        makedirs_with_mode(root_dir)
        env = LeRobotPlaybackWrapper.create_from_hdf5(
            input_path=input_path,
            output_path=output_path,
            root_dir=root_dir,
            robot_type="R1Pro",
            task_name=task_name,
            robot_obs_modalities=["proprio"],
            robot_proprio_keys=list(PROPRIOCEPTION_INDICES["R1Pro"].keys()),
            robot_sensor_config=robot_sensor_config,
            external_sensors_config=dict(),
            n_render_iterations=3,
            flush_every_n_traj=1,
            flush_every_n_steps=flush_every_n_steps,
            full_scene_file=full_scene_file,
            include_robot_control=False,
            include_contacts=False,
            load_room_instances=load_room_instances,
        )

    env.robots[0].sensors["robot_r1:zed_link:Camera:0"].horizontal_aperture = 40.0
    env.robots[0].sensors["robot_r1:zed_link:Camera:0"].image_height = HEAD_RESOLUTION[0]
    env.robots[0].sensors["robot_r1:zed_link:Camera:0"].image_width = HEAD_RESOLUTION[1]
    env.load_observation_space()

    num_samples = [env.input_hdf5["data"][key].attrs["num_samples"] for key in env.input_hdf5["data"].keys()]
    episode_id = num_samples.index(max(num_samples))
    log.info(f" >>> Replaying episode {episode_id}")

    env.playback_episode(
        episode_id=episode_id,
        record_data=True,
    )

    log.info("Playback complete. Saving data...")
    env.save_data()

    log.info(f"Successfully processed episode_{demo_id:08d}")
    return episode_id


def main():
    parser = argparse.ArgumentParser(description="Replay HDF5 files and save data")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to the data folder")
    parser.add_argument("--data_url", type=str, default="", required=False, help="URL to raw data")
    parser.add_argument("--task_name", type=str, required=True, help="Task name to process")
    parser.add_argument("--demo_id", type=int, required=True, help="Demo ID to process")
    parser.add_argument(
        "--output_format", type=str, choices=["hdf5", "lerobot"], default="hdf5", help="Output format: hdf5 or lerobot"
    )
    parser.add_argument("--flush_every_n_steps", type=int, default=500, help="Flush data every N steps")
    parser.add_argument("--update_sheet", action="store_true", help="Include this flag to update the Google Sheet")
    parser.add_argument("--row", type=int, required=False, help="Row number to update")

    args = parser.parse_args()
    task_id = TASK_NAMES_TO_INDICES[args.task_name]

    if not os.path.exists(
        f"{args.data_folder}/2025-challenge-rawdata/task-{task_id:04d}/episode_{args.demo_id:08d}.hdf5"
    ):
        if args.data_url:
            from omnigibson.learning.utils.dataset_utils import download_and_extract_data

            instance_id = int((args.demo_id % 1e4) // 10)
            traj_id = int(args.demo_id % 10)
            download_and_extract_data(args.data_url, args.data_folder, args.task_name, instance_id, traj_id)
        else:
            raise FileNotFoundError(
                f"Error: File episode_{args.demo_id:08d}.hdf5 does not exists under {args.data_folder}"
            )

    _ = replay_hdf5_file(
        data_folder=args.data_folder,
        task_id=task_id,
        demo_id=args.demo_id,
        output_format=args.output_format,
        flush_every_n_steps=args.flush_every_n_steps,
    )

    if args.output_format == "hdf5":
        try:
            os.remove(f"{args.data_folder}/replayed/episode_{args.demo_id:08d}.hdf5")
        except FileNotFoundError:
            log.warning(f"File {args.data_folder}/replayed/episode_{args.demo_id:08d}.hdf5 not found")

    if args.update_sheet:
        try:
            import gspread
        except ImportError:
            log.warning("gspread not installed, skipping Google Sheet update")
        else:
            credentials_path = f"{os.environ.get('HOME')}/Documents/credentials"
            sheet_update_success = False
            for _ in range(5):
                try:
                    update_google_sheet(credentials_path, args.task_name, args.row)
                    sheet_update_success = True
                    break
                except gspread.exceptions.APIError as e:
                    log.error(f"Failed to update Google Sheet: {e}")
                    time.sleep(60)
            if not sheet_update_success:
                update_google_sheet(credentials_path, args.task_name, args.row)

    log.info("All done!")
    og.shutdown()


if __name__ == "__main__":
    main()
