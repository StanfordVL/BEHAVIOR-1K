import getpass
import gspread
import json
import requests
import os
import tarfile
import time
from typing import Tuple
from google.oauth2.service_account import Credentials
from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES
from omnigibson.learning.scripts.replay_obs import makedirs_with_mode

VALID_USER_NAME = ["wsai", "yinhang", "svl"]


def get_credentials(credentials_path: str) -> Tuple[gspread.Client, str]:
    # authorize with Google Sheets API
    SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    SERVICE_ACCOUNT_FILE = f"{credentials_path}/google_credentials.json"
    credentials = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    gc = gspread.authorize(credentials)

    # fetch lightwheel API token
    LIGHTWHEEL_API_FILE = f"{credentials_path}/lightwheel_credentials.json"
    LIGHTWHEEL_LOGIN_URL = "http://authserver.lightwheel.net/api/authenticate/v1/user/login"
    with open(LIGHTWHEEL_API_FILE, "r") as f:
        lightwheel_api_credentials = json.load(f)

    response = requests.post(
        LIGHTWHEEL_LOGIN_URL,
        json={"username": lightwheel_api_credentials["username"], "password": lightwheel_api_credentials["password"]},
    )
    response.raise_for_status()
    lw_token = response.json().get("token")
    return gc, lightwheel_api_credentials, lw_token


def update_google_sheet(credentials_path: str, task_name: str, row_idx: int):
    assert getpass.getuser() in VALID_USER_NAME, f"Invalid user {getpass.getuser()}"
    # authorize with Google Sheets API
    SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    SERVICE_ACCOUNT_FILE = f"{credentials_path}/google_credentials.json"
    credentials = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    gc = gspread.authorize(credentials)
    spreadsheet = gc.open("B1K Challenge 2025 Data Replay Tracking Sheet")
    worksheet_name = f"{TASK_NAMES_TO_INDICES[task_name]} - {task_name}"
    task_worksheet = spreadsheet.worksheet(worksheet_name)
    # get row data
    row_data = task_worksheet.row_values(row_idx)
    assert row_data[3] == "pending"
    assert row_data[4] == getpass.getuser()
    # update status and timestamp
    task_worksheet.update(
        range_name=f"D{row_idx}:F{row_idx}",
        values=[["done", getpass.getuser(), time.strftime("%Y-%m-%d %H:%M:%S")]],
    )


def download_and_extract_data(
    url: str,
    data_dir: str,
    task_name: str,
    instance_id: int,
    traj_id: int,
):
    makedirs_with_mode(f"{data_dir}/raw/task-{TASK_NAMES_TO_INDICES[task_name]:04d}")
    # Download zip file
    response = requests.get(url)
    response.raise_for_status()
    base_name = os.path.basename(url).split("?")[0]  # remove ?Expires... suffix
    file_name = os.path.join(data_dir, "raw", base_name)
    base_name = base_name.split(".")[0]  # remove .tar suffix
    with open(file_name, "wb") as f:
        f.write(response.content)
    # unzip file
    with tarfile.open(file_name, "r:*") as tar_ref:
        tar_ref.extractall(f"{data_dir}/raw")
    # rename and move to "raw" folder
    assert os.path.exists(
        f"{data_dir}/raw/{base_name}/{task_name}.hdf5"
    ), f"File not found: {data_dir}/raw/{base_name}/{task_name}.hdf5"
    # check running_args.json
    with open(f"{data_dir}/raw/{base_name}/running_args.json", "r") as f:
        running_args = json.load(f)
        assert running_args["task_name"] == task_name, f"Task name mismatch: {running_args['task_name']} != {task_name}"
        assert (
            running_args["instance_id"] == instance_id
        ), f"Instance ID mismatch: {running_args['instance_id']} != {instance_id}"
    os.rename(
        f"{data_dir}/raw/{base_name}/{task_name}.hdf5",
        f"{data_dir}/raw/task-{TASK_NAMES_TO_INDICES[task_name]:04d}/episode_{TASK_NAMES_TO_INDICES[task_name]:04d}{instance_id:03d}{traj_id:01d}.hdf5",
    )
    # remove tar file and
    os.remove(file_name)
    os.remove(f"{data_dir}/raw/{base_name}/running_args.json")
    os.rmdir(f"{data_dir}/raw/{base_name}")
