import getpass
import gspread
import time
import os
import requests
from datetime import datetime
from omnigibson.learning.scripts.common import get_credentials, VALID_USER_NAME
from collections import Counter
from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES
from typing import Tuple


MAX_ENTRIES_PER_TASK = 300
home = os.environ.get("HOME")
credentials_path = f"{home}/Documents/credentials"


def get_all_instance_id_for_task(lw_token: str, lightwheel_api_credentials: dict, task_name: str) -> Tuple[int, str]:
    """
    Given task name, fetch all instance IDs for that task.
    Args:
        lw_token (str): Lightwheel API token.
        lightwheel_api_credentials (dict): Lightwheel API credentials.
        task_name (str): Name of the task.
    Returns:
        Tuple[int, str]: instance_id and resourceUuid
    """
    header = {
        "UserName": lightwheel_api_credentials["username"],
        "Authorization": lw_token,
    }
    body = {
        "searchRequest": {
            "whereEqFields": {
                "projectUuid": lightwheel_api_credentials["projectUuid"],
                "level1": task_name,
                "taskType": 2,
                "isEnd": True,
                "passed": True,
                "resourceType": 3,
            },
            "selectedFields": [],
            "sortFields": {"createdAt": 2, "difficulty": 2},
            "isDeleted": False,
        },
        "page": 1,
        "pageSize": 300,
    }
    response = requests.post("https://assetserver.lightwheel.net/api/asset/v1/task/get", headers=header, json=body)
    response.raise_for_status()
    return [(item["level2"], item["resourceUuid"]) for item in response.json().get("data", [])]


def is_more_than_x_hours_ago(dt_str, x, fmt="%Y-%m-%d %H:%M:%S"):
    dt = datetime.strptime(dt_str, fmt)
    diff_hours = (datetime.now() - dt).total_seconds() / 3600
    return diff_hours > x


def main():
    assert getpass.getuser() in VALID_USER_NAME, f"Invalid user {getpass.getuser()}"
    gc, lightwheel_api_credentials, lw_token = get_credentials(credentials_path)
    spreadsheet = gc.open("B1K Challenge 2025 Data Replay Tracking Sheet")
    # Update main sheet
    main_worksheet = spreadsheet.worksheet("Main")
    main_worksheet.update(range_name="A5:A5", values=[[f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}"]])

    for task_name, task_index in TASK_NAMES_TO_INDICES.items():
        worksheet_name = f"{task_index} - {task_name}"
        # Get or create the worksheet
        try:
            task_worksheet = spreadsheet.worksheet(worksheet_name)
        except gspread.exceptions.WorksheetNotFound:
            task_worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows="1", cols="7")
            header = ["Instance ID", "Traj ID", "Resource UUID", "Status", "Worker ID", "Last Updated", "Misc"]
            task_worksheet.update(range_name="A1:G1", values=[header])

        # Get all ids from lightwheel
        lw_ids = get_all_instance_id_for_task(lw_token, lightwheel_api_credentials, task_name)

        # Get all resource uuids
        rows = task_worksheet.get_all_values()
        resource_uuids = set(row[2] for row in rows[1:] if len(row) > 2)
        counter = Counter(row[0] for row in rows[1:] if len(row) > 0)
        for lw_id in lw_ids:
            num_entries = task_worksheet.row_count - 1
            if MAX_ENTRIES_PER_TASK is not None and num_entries >= MAX_ENTRIES_PER_TASK:
                break
            if lw_id[1] not in resource_uuids:
                # append new row with unprocessed status
                new_row = [
                    lw_id[0],
                    counter[lw_id[0]],
                    lw_id[1],
                    "unprocessed",
                    "",
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    "",
                ]
                task_worksheet.append_row(new_row, value_input_option="USER_ENTERED")
                counter[lw_id[0]] += 1
                # rate limit
                time.sleep(1)
        # now iterate through entires and find failure ones
        for row_idx, row in enumerate(rows[1:], start=2):
            hours_to_check = 12
            if row and row[3].strip().lower() == "pending" and is_more_than_x_hours_ago(row[5], hours_to_check):
                print(
                    f"Row {row_idx} in {worksheet_name} is pending for more than {hours_to_check} hours, marking as failed."
                )
                # change row[3] to failed and append 'a' to row[6]
                task_worksheet.update(
                    range_name=f"D{row_idx}:G{row_idx}",
                    values=[["failed", row[4].strip(), time.strftime("%Y-%m-%d %H:%M:%S"), row[6].strip() + "a"]],
                )
                time.sleep(1)  # rate limit
        # rate limit
        time.sleep(1)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] All tasks updated successfully.")


if __name__ == "__main__":
    main()
