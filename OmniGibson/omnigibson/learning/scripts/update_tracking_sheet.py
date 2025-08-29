import getpass
import gspread
import time
import os
from tqdm import tqdm
from datetime import datetime
from omnigibson.learning.scripts.common import (
    get_credentials,
    VALID_USER_NAME,
    get_all_instance_id_for_task,
    get_urls_from_lightwheel,
    get_timestamp_from_lightwheel,
)
from collections import Counter
from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES


MAX_ENTRIES_PER_TASK = 300
home = os.environ.get("HOME")
credentials_path = f"{home}/Documents/credentials"


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

    for task_name, task_index in tqdm(TASK_NAMES_TO_INDICES.items()):
        worksheet_name = f"{task_index} - {task_name}"
        # Get or create the worksheet
        try:
            task_worksheet = spreadsheet.worksheet(worksheet_name)
        except gspread.exceptions.WorksheetNotFound:
            task_worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows="1", cols="8")
            header = [
                "Instance ID",
                "Traj ID",
                "Resource UUID",
                "Timestamp",
                "Status",
                "Worker ID",
                "Last Updated",
                "Misc",
            ]
            task_worksheet.update(range_name="A1:H1", values=[header])

        # Get all ids from lightwheel
        lw_ids = get_all_instance_id_for_task(lw_token, lightwheel_api_credentials, task_name)

        # Get all resource uuids
        rows = task_worksheet.get_all_values()
        if len(rows) != len(lw_ids) + 1:
            print(f"Row count mismatch for task {task_name}: {len(rows)} != {len(lw_ids) + 1}")
        resource_uuids = set(row[2] for row in rows[1:] if len(row) > 2)
        counter = Counter(row[0] for row in rows[1:] if len(row) > 0)
        for lw_id in lw_ids:
            num_entries = task_worksheet.row_count - 1
            if MAX_ENTRIES_PER_TASK is not None and num_entries >= MAX_ENTRIES_PER_TASK:
                break
            if lw_id[1] not in resource_uuids:
                url = get_urls_from_lightwheel([lw_id[1]], lightwheel_api_credentials, lw_token)
                timestamp = str(get_timestamp_from_lightwheel(url)[0])
                # append new row with unprocessed status
                new_row = [
                    lw_id[0],
                    counter[lw_id[0]],
                    lw_id[1],
                    timestamp,
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
            hours_to_check = 24
            if row and row[4].strip().lower() == "pending" and is_more_than_x_hours_ago(row[6], hours_to_check):
                print(
                    f"Row {row_idx} in {worksheet_name} is pending for more than {hours_to_check} hours, marking as failed."
                )
                # change row[4] to failed and append 'a' to row[7]
                task_worksheet.update(
                    range_name=f"E{row_idx}:H{row_idx}",
                    values=[["failed", row[5].strip(), time.strftime("%Y-%m-%d %H:%M:%S"), row[7].strip() + "a"]],
                )
                time.sleep(1)  # rate limit
        # rate limit
        time.sleep(1)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] All tasks updated successfully.")


if __name__ == "__main__":
    main()
