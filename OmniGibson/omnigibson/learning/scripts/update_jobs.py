import argparse
import getpass
import os
import requests
import subprocess
import time
from omnigibson.learning.scripts.common import get_credentials
from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES
from typing import List


user = getpass.getuser()
home = os.environ.get("HOME")
MAX_JOBS = {"vision": 64, "viscam": 32}  # Maximum number of jobs allowed
credentials_path = f"{home}/Documents/credentials"


def get_urls_from_lightwheel(uuids: List[str], lightwheel_api_credentials: dict, lw_token: str):
    header = {
        "UserName": lightwheel_api_credentials["username"],
        "Authorization": lw_token,
    }
    body = {"versionUuids": uuids, "projectUuid": lightwheel_api_credentials["projectUuid"]}
    response = requests.post(
        "https://assetserver.lightwheel.net/api/asset/v1/teleoperation/download", headers=header, json=body
    )
    response.raise_for_status()
    urls = [res["files"][0]["url"] for res in response.json()["downloadInfos"]]
    return urls


def main(args):
    gc, lightwheel_api_credentials, lw_token = get_credentials(credentials_path=credentials_path)
    spreadsheet = gc.open("B1K Challenge 2025 Data Replay Tracking Sheet")

    if not args.local:
        partition = "viscam" if args.viscam else "svl,napoli-gpu"
        data_dir = f"/vision/u/{user}/data/behavior"
        # Get number of running or pending jobs for the current user
        cmd = (
            "/usr/local/bin/sacct --format=JobID,State --user={} --state=RUNNING,PENDING --partition {} --noheader "
            "| awk '$2 ~ /RUNNING|PENDING/ {{ split($1, a, \".\"); print a[1] }}' "
            "| sort -u "
            "| wc -l"
        ).format(user, partition)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        running_jobs = int(result.stdout.strip())
        if running_jobs >= MAX_JOBS:
            print(f"SLURM job limit reached: {running_jobs}. Exiting...")
            exit(0)
        job_quota = MAX_JOBS - running_jobs
    else:
        data_dir = os.path.expanduser("~/Documents/Files/behavior")
        job_quota = 1

    task_list = list(TASK_NAMES_TO_INDICES.keys())
    worksheets = spreadsheet.worksheets()
    for ws in worksheets:
        task_name = ws.title.split(" - ")[-1]
        if task_name in task_list:
            task_id = TASK_NAMES_TO_INDICES[task_name]
            # Iterate through all the rows, find the unprocessed ones
            all_rows = ws.get_all_values()
            for row_idx, row in enumerate(all_rows[1:], start=2):  # Skip header, row numbers start at 2
                if row and row[3].strip().lower() == "unprocessed":
                    instance_id, traj_id, resource_uuid = int(row[0]), int(row[1]), row[2]
                    url = get_urls_from_lightwheel([resource_uuid], lightwheel_api_credentials, lw_token=lw_token)[0]
                    print(f"Scheduling job for episode {task_id:04d}{instance_id:03d}{traj_id:01d}")
                    ws.update(
                        range_name=f"D{row_idx}:F{row_idx}",
                        values=[["pending", user, time.strftime("%Y-%m-%d %H:%M:%S")]],
                    )
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
                        subprocess.run(cmd, shell=True)
                    else:
                        cmd = (
                            "cd ~/Research/BEHAVIOR-1K && "
                            "python OmniGibson/omnigibson/learning/scripts/replay_obs.py "
                            '--data_url "{}" '
                            "--data_folder {} "
                            "--task_name {} "
                            "--demo_id {} "
                            "--update_sheet "
                            "--low_dim --rgbd --seg --bbox --pcd_gt --pcd_vid "
                            "--row {}"
                        ).format(url, data_dir, task_name, int(f"{task_id:04d}{instance_id:03d}{traj_id:01d}"), row_idx)
                        subprocess.run(cmd, shell=True, check=True)
                    job_quota -= 1
                    if job_quota <= 0:
                        print(f"Reached job limit, exiting...")
                        exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Whether to run locally or with slurm")
    parser.add_argument("--viscam", action="store_true", help="Whether to run with viscam")
    args = parser.parse_args()

    main(args)
