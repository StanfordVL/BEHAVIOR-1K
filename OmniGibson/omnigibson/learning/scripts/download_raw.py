import os
import time
from omnigibson.learning.scripts.common import (
    get_credentials,
    download_and_extract_data,
    get_urls_from_lightwheel,
)
from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES

# We download a maximum of 200 trajectories for each task
MAX_TRAJ_PER_TASK = 200

task_list = list(TASK_NAMES_TO_INDICES.keys())
data_dir = "/vision/group/behavior"
home = os.environ.get("HOME")
gc, lightwheel_api_credentials, lw_token = get_credentials(credentials_path=f"{home}/Documents/credentials")

tracking_spreadsheet = gc.open("B1K Challenge 2025 Data Replay Tracking Sheet")
worksheets = tracking_spreadsheet.worksheets()
for ws in worksheets:
    file_downloaded = 0
    traj_downloaded = 0
    task_name = ws.title.split(" - ")[-1]
    if task_name in task_list:
        task_id = TASK_NAMES_TO_INDICES[task_name]
        all_rows = ws.get_all_values()
        for row in all_rows[1:]:
            if row and int(row[1]) == 0:  # We download a maximum of one trajectory for each task instance id
                resource_uuid = row[2]
                instance_id = int(row[0])
                # check whether raw file already exists
                if not os.path.exists(
                    os.path.join(
                        data_dir, "raw", f"task-{task_id:04d}", f"episode_{task_id:04d}{instance_id:03d}0.hdf5"
                    )
                ):
                    url = get_urls_from_lightwheel([resource_uuid], lightwheel_api_credentials, lw_token=lw_token)[0]
                    try:
                        download_and_extract_data(url, data_dir, task_name, instance_id, 0)
                        file_downloaded += 1
                        traj_downloaded += 1
                    except AssertionError as e:
                        print(
                            f"Error downloading or extracting data for {task_name} resource uuid {resource_uuid}: {e}"
                        )
                else:
                    traj_downloaded += 1
            if traj_downloaded >= MAX_TRAJ_PER_TASK:
                break
        time.sleep(1)

    print(f"Finished processing task: {ws.title}, {file_downloaded} files downloaded.")

print("All tasks processed.")
