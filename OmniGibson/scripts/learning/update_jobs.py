import argparse
import getpass
import os
import subprocess
import time
from omnigibson.learning.utils.dataset_utils import get_credentials, VALID_USER_NAME, get_urls_from_lightwheel
from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES


user = getpass.getuser()
home = os.environ.get("HOME")
MAX_JOBS = {"vision": 64, "viscam": 32}  # Maximum number of jobs allowed
MAX_TRAJ_PER_TASK = 200
credentials_path = f"{home}/Documents/credentials"


def main(args):
    assert user in VALID_USER_NAME, f"Invalid user {user}!"
    gc, lightwheel_api_credentials, lw_token = get_credentials(credentials_path=credentials_path)
    tracking_spreadsheet = gc.open("B1K Challenge 2025 Data Replay Tracking Sheet")
    task_misc_spreadsheet = gc.open("B50 Task Misc")
    tasks_misc_rows = task_misc_spreadsheet.worksheet("Task Misc").get_all_values()

    if not args.local:
        partition = "viscam" if args.viscam else "svl,napoli-gpu"
        node = "viscam" if args.viscam else "vision"
        data_dir = "/vision/group/behavior/2025-challenge-demos"
        # Get number of running or pending jobs for the current user
        cmd = (
            "/usr/local/bin/sacct --format=JobID,State --user={} --state=RUNNING,PENDING --partition {} --noheader "
            "| awk '$2 ~ /RUNNING|PENDING/ {{ split($1, a, \".\"); print a[1] }}' "
            "| sort -u "
            "| wc -l"
        ).format(user, partition)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        running_jobs = int(result.stdout.strip())
        print(f"Running jobs: {running_jobs}")
        if running_jobs >= MAX_JOBS[node]:
            print(f"SLURM job limit reached: {running_jobs}. Exiting...")
            exit(0)
        job_quota = MAX_JOBS[node] - running_jobs
    else:
        data_dir = os.path.expanduser("~/Documents/Files/behavior")
        job_quota = 1

    task_list = list(TASK_NAMES_TO_INDICES.keys())
    worksheets = tracking_spreadsheet.worksheets()
    for ws in worksheets:
        task_name = ws.title.split(" - ")[-1]
        if task_name in task_list:
            task_id = TASK_NAMES_TO_INDICES[task_name]
            assert task_name in tasks_misc_rows[task_id + 1][1], f"Task name {task_name} mismatch!"
            col_to_check = 4 if args.local else 3
            if tasks_misc_rows[task_id + 1][col_to_check].strip() == "1":
                # Iterate through all the rows, find the unprocessed ones
                all_rows = ws.get_all_values()
                num_process_traj = 0
                for row_idx, row in enumerate(all_rows[1:], start=2):  # Skip header, row numbers start at 2
                    if num_process_traj >= MAX_TRAJ_PER_TASK:
                        break
                    elif row and row[4].strip().lower() in ["pending", "done"]:
                        num_process_traj += 1
                    elif row and (
                        (row[4].strip().lower() == "unprocessed" and int(row[1]) == 0)
                        or (row[4].strip().lower() == "failed" and row[5].strip() == user and len(row[7].strip()) < 2)
                    ):  # currently only generate unique task instance
                        instance_id, traj_id, resource_uuid = int(row[0]), int(row[1]), row[2]
                        url = get_urls_from_lightwheel([resource_uuid], lightwheel_api_credentials, lw_token=lw_token)[
                            0
                        ]
                        if not args.local:
                            node = "viscam" if args.viscam else "vision"
                            cmd = (
                                "cd /vision/u/{}/BEHAVIOR-1K && "
                                '/usr/local/bin/sbatch OmniGibson/omnigibson/learning/scripts/replay_data_{}.sbatch.sh --data_url "{}" --data_folder {} --task_name {} --demo_id {} --update_sheet --row {}'
                            ).format(
                                user,
                                node,
                                url,
                                data_dir,
                                task_name,
                                int(f"{task_id:04d}{instance_id:03d}{traj_id:01d}"),
                                row_idx,
                            )
                            # Run the command
                            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                            if result.returncode == 0:
                                print(
                                    f"Scheduled job for episode {task_id:04d}{instance_id:03d}{traj_id:01d}",
                                    result.stdout,
                                )
                                ws.update(
                                    range_name=f"E{row_idx}:G{row_idx}",
                                    values=[["pending", user, time.strftime("%Y-%m-%d %H:%M:%S")]],
                                )
                            else:
                                print(
                                    f"Failed to schedule job for episode {task_id:04d}{instance_id:03d}{traj_id:01d}",
                                    result.stderr,
                                )
                                exit(0)
                        else:
                            gpu_id = args.gpu if args.gpu is not None else 0
                            ws.update(
                                range_name=f"E{row_idx}:G{row_idx}",
                                values=[["pending", user, time.strftime("%Y-%m-%d %H:%M:%S")]],
                            )
                            cmd = (
                                "cd ~/Research/BEHAVIOR-1K && "
                                "OMNIGIBSON_GPU_ID={} python OmniGibson/omnigibson/learning/scripts/replay_obs.py "
                                '--data_url "{}" '
                                "--data_folder {} "
                                "--task_name {} "
                                "--demo_id {} "
                                "--update_sheet "
                                "--low_dim --rgbd --seg "
                                "--row {}"
                            ).format(
                                gpu_id,
                                url,
                                data_dir,
                                task_name,
                                int(f"{task_id:04d}{instance_id:03d}{traj_id:01d}"),
                                row_idx,
                            )
                            subprocess.run(cmd, shell=True, check=True)
                        job_quota -= 1
                        num_process_traj += 1
                        if job_quota <= 0:
                            print("Reached job limit, exiting...")
                            exit(0)
                    # Remove this if we don't want failed trajectories to count
                    elif row and row[4].strip().lower() == "failed":
                        num_process_traj += 1
                # rate limit
                time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Whether to run locally or with slurm")
    parser.add_argument("--viscam", action="store_true", help="Whether to run with viscam")
    parser.add_argument("--gpu", required=False, type=int, help="(For local) GPU ID to use")
    args = parser.parse_args()

    main(args)
