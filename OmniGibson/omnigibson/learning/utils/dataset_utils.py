import getpass
import json
import numpy as np
import os
import omnigibson as og
import pandas as pd
import random
import re
import requests
import shutil
import tarfile
import time
import zipfile
from collections import Counter
from datetime import datetime
from typing import Any, Tuple, List, Optional
from tqdm import tqdm
from google.oauth2.service_account import Credentials
from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES
from urllib.parse import urlparse

VALID_USER_NAME = ["wsai", "yinhang", "svl", "wsai-yfj", "wpai", "qinengw", "jdw"]


def makedirs_with_mode(path, mode=0o2775) -> None:
    """
    Recursively create directories with specified mode applied to all newly created dirs.
    Args:
        path (str): The directory path to create.
        mode (int): The mode to apply to newly created directories.
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


def get_credentials(credentials_path: str = "~/Documents/credentials") -> Tuple[Any, dict, str]:
    """
    [Internal use only] Get Google Sheets and Lightwheel API credentials.
    Args:
        credentials_path (str): Path to the credentials directory.
    Returns:
        Tuple[gspread.Client, dict, str]: Google Sheets client and Lightwheel API credentials and token.
    """
    import gspread

    credentials_path = os.path.expanduser(credentials_path)
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


def update_google_sheet(credentials_path: str, task_name: str, row_idx: int) -> None:
    """
    [Internal use only] update internal data replay tracking sheet.
    Args:
        credentials_path (str): Path to the credentials directory.
        task_name (str): Name of the task to update.
        row_idx (int): Row index to update.
    """
    import gspread

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
    [Internal use only] Given task name, fetch all instance IDs for that task.
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
    """
    [Internal use only] Given a list of UUIDs, fetch their download URLs from Lightwheel API.
    Args:
        uuids (List[str]): List of version UUIDs.
        lightwheel_api_credentials (dict): Lightwheel API credentials.
        lw_token (str): Lightwheel API token.
    Returns:
        List[str]: List of download URLs.
    """
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
    """
    [Internal use only] Given a list of URLs, fetch their timestamps (on the filename) from Lightwheel API.
    Args:
        urls (List[str]): List of download URLs.
    Returns:
        List[str]: List of timestamps.
    """
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
) -> None:
    """
    [Internal use only] Download and extract data from a Lightwheel API URL.
    Args:
        url (str): The download URL.
        data_dir (str): The directory to save the data.
        task_name (str): The name of the task.
        instance_id (int): The instance ID.
        traj_id (int): The trajectory ID.
    """
    makedirs_with_mode(f"{data_dir}/2025-challenge-rawdata/task-{TASK_NAMES_TO_INDICES[task_name]:04d}")
    # Download zip file
    response = requests.get(url)
    response.raise_for_status()
    base_name = os.path.basename(url).split("?")[0]  # remove ?Expires... suffix
    file_name = os.path.join(data_dir, "2025-challenge-rawdata", base_name)
    base_name = base_name.split(".")[0]  # remove .tar suffix
    with open(file_name, "wb") as f:
        f.write(response.content)
    # unzip file
    with tarfile.open(file_name, "r:*") as tar_ref:
        tar_ref.extractall(f"{data_dir}/2025-challenge-rawdata")
    # rename and move to "raw" folder
    assert os.path.exists(
        f"{data_dir}/2025-challenge-rawdata/{base_name}/{task_name}.hdf5"
    ), f"File not found: {data_dir}/2025-challenge-rawdata/{base_name}/{task_name}.hdf5"
    # check running_args.json
    with open(f"{data_dir}/2025-challenge-rawdata/{base_name}/running_args.json", "r") as f:
        running_args = json.load(f)
        assert running_args["task_name"] == task_name, f"Task name mismatch: {running_args['task_name']} != {task_name}"
        assert (
            running_args["instance_id"] == instance_id
        ), f"Instance ID mismatch: {running_args['instance_id']} in running_args.json != {instance_id} from LW API"
    os.rename(
        f"{data_dir}/2025-challenge-rawdata/{base_name}/{task_name}.hdf5",
        f"{data_dir}/2025-challenge-rawdata/task-{TASK_NAMES_TO_INDICES[task_name]:04d}/episode_{TASK_NAMES_TO_INDICES[task_name]:04d}{instance_id:03d}{traj_id:01d}.hdf5",
    )
    # remove tar file and
    os.remove(file_name)
    os.remove(f"{data_dir}/2025-challenge-rawdata/{base_name}/running_args.json")
    os.rmdir(f"{data_dir}/2025-challenge-rawdata/{base_name}")


def reorder_sheet(worksheet) -> None:
    """
    Reorder rows in the worksheet based on column B and column A.

    Rules:
    0. First row is header row -> keep as-is.
    1. Rows with B == 0 → first group, sorted by A.
    2. Rows with B != -1 (and not 0) → second group, sorted by A.
    3. Rows with B == -1 → last group, sorted by A.
    """

    # Get all values
    all_values = worksheet.get_all_values()
    if not all_values:
        return  # empty sheet

    header, rows = all_values[0], all_values[1:]

    # Parse into (A, B, rest_of_row)
    def parse_row(row):
        row[0] = int(row[0])
        row[1] = int(row[1])
        return row[0], row[1], row

    parsed = [parse_row(row) for row in rows]

    # Grouping
    group_b0 = [r for r in parsed if r[1] == 0]
    group_notm1 = [r for r in parsed if r[1] > 0]
    group_m1 = [r for r in parsed if r[1] < 0]

    # Sort each group by A
    group_b0.sort(key=lambda x: x[0])
    group_notm1.sort(key=lambda x: x[0])
    group_m1.sort(key=lambda x: x[0])

    # Rebuild ordered rows
    new_rows = [r[2] for r in group_b0 + group_notm1 + group_m1]

    # Write back in one batch
    worksheet.update("A1", [header] + new_rows)
    print("Reordered rows in worksheet:", worksheet.title)
    time.sleep(1)  # to avoid rate limiting


def remove_failed_episodes(worksheet, data_dir: str) -> None:
    """
    For the given worksheet and data_dir:
    0. Ignore the first row (header)
    1. Extract task_id from ws.title, which is "{task_id} - {task_name}"
    2. For each row with column B == -1:
       - take demo_id = int(column A)
       - construct episode_name = f"episode_{task_id:04d}{demo_id:04d}"
       - remove corresponding files from data_dir in all subfolders
    """
    # --- Step 1: get task_id from sheet title ---
    title = worksheet.title
    task_id_str, _ = title.split(" - ", 1)
    task_id = int(task_id_str)

    # --- Step 2: read all rows (ignore header) ---
    all_values = worksheet.get_all_values()
    rows = all_values[1:]
    total_removed = 0
    for row in rows:
        if len(row) < 2:
            continue
        try:
            demo_id = int(row[0])
            b_val = int(row[1])
        except ValueError:
            continue

        if b_val == -1:
            episode_name = f"episode_{task_id:04d}{demo_id:03d}0"

            # Files to remove
            files = [
                os.path.join(data_dir, f"data/task-{task_id:04d}/{episode_name}.parquet"),
                os.path.join(data_dir, f"meta/episodes/task-{task_id:04d}/{episode_name}.json"),
                os.path.join(data_dir, f"raw/task-{task_id:04d}/{episode_name}.hdf5"),
                os.path.join(data_dir, f"videos/task-{task_id:04d}/observation.images.depth.head/{episode_name}.mp4"),
                os.path.join(
                    data_dir, f"videos/task-{task_id:04d}/observation.images.depth.left_wrist/{episode_name}.mp4"
                ),
                os.path.join(
                    data_dir, f"videos/task-{task_id:04d}/observation.images.depth.right_wrist/{episode_name}.mp4"
                ),
                os.path.join(data_dir, f"videos/task-{task_id:04d}/observation.images.rgb.head/{episode_name}.mp4"),
                os.path.join(
                    data_dir, f"videos/task-{task_id:04d}/observation.images.rgb.left_wrist/{episode_name}.mp4"
                ),
                os.path.join(
                    data_dir, f"videos/task-{task_id:04d}/observation.images.rgb.right_wrist/{episode_name}.mp4"
                ),
                os.path.join(
                    data_dir, f"videos/task-{task_id:04d}/observation.images.seg_instance_id.head/{episode_name}.mp4"
                ),
                os.path.join(
                    data_dir,
                    f"videos/task-{task_id:04d}/observation.images.seg_instance_id.left_wrist/{episode_name}.mp4",
                ),
                os.path.join(
                    data_dir,
                    f"videos/task-{task_id:04d}/observation.images.seg_instance_id.right_wrist/{episode_name}.mp4",
                ),
            ]
            n_removed = 0
            for f in files:
                if os.path.exists(f):
                    os.remove(f)
                    n_removed += 1
            total_removed += n_removed
    print(f"Total removed files for task {task_id}: {total_removed}")
    return total_removed


def extract_annotations(
    data_dir: str,
    annotation_data_dir: str,
    credentials_path: str = "~/Documents/credentials",
    remove_memory_prefix: bool = False,
) -> None:
    """
    Extract annotations from the annotation data directory and store in the data directory.
    If remove_memory_prefix is True, remove "memory_prefix" field in skill annotations.
    """
    data_dir = os.path.expanduser(data_dir)
    makedirs_with_mode(f"{data_dir}/annotations")
    annotation_data_dir = os.path.expanduser(annotation_data_dir)
    # get tracking worksheet
    gc = get_credentials(credentials_path)[0]
    spreadsheet = gc.open("B1K Challenge 2025 Data Replay Tracking Sheet")

    task_processed = 0
    # iterate through all files under annotation_data_dir,
    for file in os.listdir(annotation_data_dir):
        if file.endswith(".zip"):
            # extract filename
            filename = file[:-4]
            if filename not in TASK_NAMES_TO_INDICES:
                print(f"Invalid task name: {filename}")
                continue
            # unzip the file
            with zipfile.ZipFile(os.path.join(annotation_data_dir, file), "r") as zip_ref:
                zip_ref.extractall(f"{data_dir}/annotations")
            # rename folder based on task indices
            task_index = TASK_NAMES_TO_INDICES[filename]
            os.rename(f"{data_dir}/annotations/{filename}", f"{data_dir}/annotations/task-{task_index:04d}")
            # now, assert there are 200 files in the task folder
            assert (
                len(os.listdir(f"{data_dir}/annotations/task-{task_index:04d}")) == 200
            ), f"Task {task_index} does not have 200 files."
            # now, fetch all timestamp - task indices correspondance from worksheet
            worksheet = spreadsheet.worksheet(f"{task_index} - {filename}")
            rows = worksheet.get_all_values()[1:]  # skip header
            for row in rows:
                if row and row[4] == "done":
                    instance_id, traj_id, timestamp = int(row[0]), int(row[1]), row[3]
                    assert os.path.isfile(
                        f"{data_dir}/annotations/task-{task_index:04d}/{filename}_{timestamp}.json"
                    ), f"Missing annotation for {instance_id}"
                    # rename episode
                    os.rename(
                        f"{data_dir}/annotations/task-{task_index:04d}/{filename}_{timestamp}.json",
                        f"{data_dir}/annotations/task-{task_index:04d}/episode_{task_index:04d}{instance_id:03d}{traj_id:01d}.json",
                    )
                    if remove_memory_prefix:
                        # remove "memory" in skill_annotations and primitive_annotations
                        with open(
                            f"{data_dir}/annotations/task-{task_index:04d}/episode_{task_index:04d}{instance_id:03d}{traj_id:01d}.json",
                            "r",
                        ) as f:
                            annotation_data = json.load(f)
                        for skill in annotation_data.get("skill_annotation", []):
                            if "memory_prefix" in skill:
                                del skill["memory_prefix"]
                        for primitive in annotation_data.get("primitive_annotation", []):
                            if "memory_prefix" in primitive:
                                del primitive["memory_prefix"]
                        with open(
                            f"{data_dir}/annotations/task-{task_index:04d}/episode_{task_index:04d}{instance_id:03d}{traj_id:01d}.json",
                            "w",
                        ) as f:
                            json.dump(annotation_data, f, indent=4)
            print(f"Finished processing task {task_index} - {filename}")
            task_processed += 1
            time.sleep(1.5)  # to avoid rate limiting

    # remove __MACOSX folder
    shutil.rmtree(f"{data_dir}/annotations/__MACOSX")
    print(f"Finished processing {task_processed} tasks.")


def check_leaf_folders_have_n(data_dir: str, n: int = 200) -> Tuple[dict, int]:
    """
    Recursively find all leaf folders under data_dir.
    A leaf folder is one that contains only files (no subdirectories).
    For each leaf folder, check it has exactly n files.
    Args:
        data_dir (str): The root directory to start searching.
        n (int): The exact number of files each leaf folder should have.
    Returns:
        Tuple[dict, int]: A tuple containing:
            - A dictionary mapping leaf folder paths to their file counts.
            - The total file count across all leaf folders.
    """
    data_dir = os.path.expanduser(data_dir)
    results = {}
    total_count = 0
    for root, dirs, files in os.walk(data_dir):
        # ignore hidden folders
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        # leaf folder: contains files but no subdirs
        if not dirs:
            count = len([f for f in files if os.path.isfile(os.path.join(root, f))])
            results[root] = count
            total_count += count
            if count == n:
                print(f"✅ {root} has exactly {n} files.")
            else:
                raise Exception(f"❌ {root} has {count} files (expected {n}).")
    print(f"Total files across all leaf folders: {total_count}")
    return results, total_count


def update_sheet_counts(worksheet) -> None:
    """
    [Internal use only] Updates the worksheet:
    1. For rows with B != 0:
       - E = "ignored"
       - F = ""
    2. Replace column B with the number of occurrences of column A
       in previous rows.
    """
    all_values = worksheet.get_all_values()
    if not all_values:
        return

    _, rows = all_values[0], all_values[1:]

    # Track counts of column A values
    counts = {}

    updated_rows = []
    for row in rows:
        row[0] = int(row[0])
        row[1] = int(row[1])
        row[7] = ""

        # --- Step 1: update columns E/F based on original B ---
        if row[1] != 0:
            row[4] = "ignored"  # Column E (0-indexed 4)
            row[5] = ""  # Column F (0-indexed 5)

        # --- Step 2: update column B with previous counts of A ---
        prev_count = counts.get(row[0], 0)
        row[1] = int(prev_count)  # Column B
        counts[row[0]] = prev_count + 1

        updated_rows.append(row)

    # Update the sheet in one batch
    worksheet.update("A2", updated_rows)
    print("Changed worksheet:", worksheet.title)
    time.sleep(1)  # to avoid rate limiting


def assign_test_instances(task_ws, ws_misc, misc_values) -> None:
    """
    For a given task worksheet and the misc spreadsheet:
    1. Get task_id and task_name from worksheet title "{id} - {name}".
    2. Collect unique integers in Column A and compute missing IDs from {1..300}.
    3. Sample up to 20 missing IDs
    4. Write groups into columns C in the matching row of Test Instances tab.
    """
    # --- Step 1: parse task id/name from worksheet title ---
    title = task_ws.title
    task_id_str, task_name = title.split(" - ", 1)
    task_id = int(task_id_str)

    # --- Step 2: collect unique ints in column A ---
    all_values = task_ws.get_all_values()
    rows = all_values[1:]  # ignore header
    col_a_set = set()
    for row in rows:
        if not row or not row[0]:
            continue
        try:
            col_a_set.add(int(row[0]))
        except ValueError:
            continue

    ref_set = set(range(1, 301))
    # assert col_a_set is a subset of ref_set
    assert col_a_set.issubset(ref_set), f"Column A has values outside 1-300: {col_a_set - ref_set}"
    missing = list(ref_set - col_a_set)
    assert len(missing) >= 20, f"Not enough missing IDs to sample 20: only {len(missing)} missing."

    # --- Step 3: sample up to 20 ---
    sample_missing = random.sample(missing, 20)
    random.shuffle(sample_missing)

    # --- Step 4: open misc sheet and find correct row ---

    # First row is header
    target_row = misc_values[task_id + 1]
    assert (
        int(target_row[0]) == task_id and target_row[1].strip() == task_name
    ), f"Row mismatch for task {task_id} - {task_name}: found {target_row[0]} - {target_row[1]}"

    # --- Step 5: update in one batch ---
    ws_misc.update(range_name=f"C{task_id + 2}:C{task_id + 2}", values=[[", ".join(map(str, sample_missing))]])
    time.sleep(1)

    print(f"✅ Updated task {task_id} - {task_name} with test instances.")


def update_parquet_indices(root_dir: str):
    """For every parquet file named episode_XXXXXXXX.parquet, update episode_index and task_index."""
    pat = re.compile(r"episode_(\d{8})\.parquet$")

    for dirpath, _, filenames in os.walk(root_dir):
        print(dirpath)
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)

            m = pat.search(fname)
            if not m:
                continue  # not a matching parquet

            episode_num = int(m.group(1))
            task_num = int(m.group(1)[:4])
            try:
                df = pd.read_parquet(fpath)

                assert "episode_index" in df.columns
                df["episode_index"] = episode_num
                assert "task_index" in df.columns
                df["task_index"] = task_num

                # overwrite parquet
                df.to_parquet(fpath, index=False)

            except Exception as e:
                print(f"Skipping {fpath}, error: {e}")


def remove_grasp_state(root_dir: str):
    """
    For every parquet file named episode_XXXXXXXX.parquet,
    If observation.state has dim 258, remove dim 193 and 233 (grasp_left and grasp_right) and save the parquet back to disk.
    """
    pat = re.compile(r"episode_(\d{8})\.parquet$")

    for dirpath, _, filenames in os.walk(root_dir):
        print(dirpath)
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)

            m = pat.search(fname)
            if not m:
                continue  # not a matching parquet

            try:
                df = pd.read_parquet(fpath)

                assert "observation.state" in df.columns
                obs = np.array(df["observation.state"].tolist())
                if obs.ndim == 2 and obs.shape[1] == 258:
                    obs = np.delete(obs, [193, 233], axis=1)
                    df["observation.state"] = obs.tolist()

                    # overwrite parquet
                    df.to_parquet(fpath, index=False)

            except Exception as e:
                print(f"Skipping {fpath}, error: {e}")


def fix_permissions(root_dir: str):
    """Recursively set rw-rw-r-- for all files owned by the current user."""
    for dirpath, _, filenames in os.walk(root_dir):
        print(dirpath)
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            try:
                os.chmod(fpath, 0o664)  # rw-rw-r--
            except (PermissionError, FileNotFoundError):
                continue


def download_raw(credentials_path: str = "~/Documents/credentials", max_traj_per_task: int = 200):
    task_list = list(TASK_NAMES_TO_INDICES.keys())
    data_dir = "/vision/group/behavior/2025-challenge-rawdata"
    gc, lightwheel_api_credentials, lw_token = get_credentials(credentials_path=credentials_path)

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
                        url = get_urls_from_lightwheel([resource_uuid], lightwheel_api_credentials, lw_token=lw_token)[
                            0
                        ]
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
                if traj_downloaded >= max_traj_per_task:
                    break
            time.sleep(1)

        print(f"Finished processing task: {ws.title}, {file_downloaded} files downloaded.")

    print("All tasks processed.")


def is_more_than_x_hours_ago(dt_str, x, fmt="%Y-%m-%d %H:%M:%S"):
    dt = datetime.strptime(dt_str, fmt)
    diff_hours = (datetime.now() - dt).total_seconds() / 3600
    return diff_hours > x


def update_tracking_sheet(
    credentials_path: str = "~/Documents/credentials", max_entries_per_task: Optional[int] = None
) -> None:
    """
    [Internal use only] Updates the tracking sheet with the latest information from lightwheel.
    Args:
        credentials_path (str): The path to the credentials file.
        max_entries_per_task (Optional[int]): The maximum number of entries to process per task.
    """
    import gspread

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
            if max_entries_per_task is not None and num_entries >= max_entries_per_task:
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
    # check_leaf_folders_have_n("~/behavior", 200)
    # gc = get_credentials("~/Documents/credentials")[0]
    # tracking_spreadsheet = gc.open("B1K Challenge 2025 Data Replay Tracking Sheet")
    # misc_sheet = gc.open("B50 Task Misc")
    # misc_ws = misc_sheet.worksheet("Test Instances")
    # misc_values = misc_ws.get_all_values()
    # for task_name, task_index in tqdm(TASK_NAMES_TO_INDICES.items()):
    #     task_ws = tracking_spreadsheet.worksheet(f"{task_index} - {task_name}")
    #     assign_test_instances(task_ws, misc_ws, misc_values)
    #     time.sleep(1)
    # extract_annotations(
    #     "/scr/behavior/2025-challenge-demos", "/home/svl/Downloads/annotations", remove_memory_prefix=True
    # )
    og.shutdown()
