import gspread
import time
import os
from google.oauth2.service_account import Credentials
from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES


user = os.environ.get("USER")
task_list = list(TASK_NAMES_TO_INDICES.keys())

# 1. Set up credentials and authorize
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
SERVICE_ACCOUNT_FILE = "/afs/cs.stanford.edu/u/{}/google_credentials.json".format(user)

credentials = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
gc = gspread.authorize(credentials)


spreadsheet = gc.open("B1K Challenge 2025 Data Replay Tracking Sheet")

for task_name in task_list:
    print(f"Processing task: {task_name}")

    # 4. Get or create the worksheet
    try:
        worksheet = spreadsheet.worksheet(task_name)
        print(f"Worksheet '{task_name}' already exists.")
    except gspread.exceptions.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title=task_name, rows="100", cols="20")
        print(f"Worksheet '{task_name}' created.")

    existing_data = worksheet.get_all_values()  # returns list of rows (each row is a list of cell values)
    for row in existing_data:
        if row["status"].strip().lower() == "unprocessed":
            print(row["file_id"])
            break

    # 6. Data to append (you can modify this)
    new_rows = [
        ["Episode ID", "Task Instance ID", "Status", "User", "Timestamp"],
        ["episode_001", "1", "Unprocessed", "", time.strftime("%Y-%m-%d %H:%M:%S")],
    ]

    # 7. Append rows to the bottom of the sheet
    for row in new_rows:
        worksheet.append_row(row, value_input_option="USER_ENTERED")

    print(f"{task_name} worksheet updated.")

print("All tasks updated successfully.")
