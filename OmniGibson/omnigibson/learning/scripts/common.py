import getpass
import gspread
import json
import requests
import os
import tarfile
import time
from typing import Tuple, List
from tqdm import tqdm
from google.oauth2.service_account import Credentials
from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES
from urllib.parse import urlparse

VALID_USER_NAME = ["wsai", "yinhang", "svl", "wpai", "qinengw", "wsai-yfj", "jdw"]


def makedirs_with_mode(path, mode=0o2775):
    """
    Recursively create directories with specified mode applied to all newly created dirs.
    Existing directories keep their current permissions.
    """
    # Normalize path
    path = os.path.abspath(path)
    parts = path.split(os.sep)
    if parts[0] == "":
        parts[0] = os.sep  # for absolute paths on Unix

    current_path = parts[0]
    for part in parts[1:]:
        current_path = os.path.join(current_path, part)
        if not os.path.exists(current_path):
            try:
                os.makedirs(current_path, exist_ok=True)
                # Apply mode explicitly because os.mkdir may be affected by umask
                os.chmod(current_path, mode)
            except Exception as e:
                print(f"Failed to create directory {current_path}: {e}")
        else:
            pass


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
    assert row_data[4] == "pending"
    assert row_data[5] == getpass.getuser()
    # update status and timestamp
    task_worksheet.update(
        range_name=f"E{row_idx}:G{row_idx}",
        values=[["done", getpass.getuser(), time.strftime("%Y-%m-%d %H:%M:%S")]],
    )


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


def get_urls_from_lightwheel(uuids: List[str], lightwheel_api_credentials: dict, lw_token: str) -> List[str]:
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


def get_timestamp_from_lightwheel(urls: List[str]) -> List[str]:
    timestamps = []
    for url in tqdm(urls):
        resp = requests.head(url, allow_redirects=True)
        cd = resp.headers.get("content-disposition")
        if cd and "filename=" in cd:
            # e.g. 'attachment; filename="episode_00001234.parquet"'
            fname = cd.split("filename=")[-1].strip('"; ')
        else:
            # fallback: use last part of the URL path
            fname = urlparse(resp.url).path.split("/")[-1]
        # extract timestamp from filename, which is of the format "`taskname`_`timestamp``.tar"
        timestamp = fname.rsplit("_", 1)[1].split(".")[0]
        assert len(timestamp) == 16, f"Invalid timestamp format: {timestamp}"
        timestamps.append(timestamp)
    return timestamps


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
        ), f"Instance ID mismatch: {running_args['instance_id']} in running_args.json != {instance_id} from LW API"
    os.rename(
        f"{data_dir}/raw/{base_name}/{task_name}.hdf5",
        f"{data_dir}/raw/task-{TASK_NAMES_TO_INDICES[task_name]:04d}/episode_{TASK_NAMES_TO_INDICES[task_name]:04d}{instance_id:03d}{traj_id:01d}.hdf5",
    )
    # remove tar file and
    os.remove(file_name)
    os.remove(f"{data_dir}/raw/{base_name}/running_args.json")
    os.rmdir(f"{data_dir}/raw/{base_name}")
