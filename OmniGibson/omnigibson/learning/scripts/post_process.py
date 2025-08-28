# import gspread
import os

# import omnigibson as og
import re
import random
import pandas as pd
import time
# from omnigibson.learning.scripts.common import (
#     get_credentials,
#     download_and_extract_data,
#     get_urls_from_lightwheel,
# )
# from omnigibson.learning.utils.eval_utils import TASK_NAMES_TO_INDICES


def reorder_sheet(worksheet):
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


def remove_failed_episodes(worksheet, data_dir: str):
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


def check_leaf_folders_have_200(data_dir: str):
    """
    Recursively find all leaf folders under data_dir.
    A leaf folder is one that contains only files (no subdirectories).
    For each leaf folder, check it has exactly 200 files.
    Prints results and returns a dict {folder_path: count}.
    """
    results = {}
    total_count = 0
    for root, dirs, files in os.walk(data_dir):
        # leaf folder: contains files but no subdirs
        if not dirs:
            if "Trash-1000" not in root:  # ignore trash folder
                count = len([f for f in files if os.path.isfile(os.path.join(root, f))])
                results[root] = count
                total_count += count
                if count == 200:
                    print(f"✅ {root} has exactly 200 files.")
                else:
                    raise Exception(f"❌ {root} has {count} files (expected 200).")
    print(f"Total files across all leaf folders: {total_count}")
    return results, total_count


def update_sheet_counts(worksheet):
    """
    Updates the worksheet:
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


def assign_test_instances(task_ws, ws_misc, misc_values):
    """
    For a given task worksheet and the misc spreadsheet:
    1. Get task_id and task_name from worksheet title "{id} - {name}".
    2. Collect unique integers in Column A and compute missing IDs from {1..300}.
    3. Sample up to 20 missing IDs, shuffle, split into 2 groups of 10.
    4. Write groups into columns C and D in the matching row of Test Instances tab.
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
    group1, group2 = sample_missing[:10], sample_missing[10:]

    # --- Step 4: open misc sheet and find correct row ---

    # First row is header
    target_row = misc_values[task_id + 1]
    assert (
        int(target_row[0]) == task_id and target_row[1].strip() == task_name
    ), f"Row mismatch for task {task_id} - {task_name}: found {target_row[0]} - {target_row[1]}"

    # --- Step 5: update in one batch ---
    ws_misc.update(
        range_name=f"C{task_id + 2}:D{task_id + 2}", values=[[", ".join(map(str, group1)), ", ".join(map(str, group2))]]
    )
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


if __name__ == "__main__":
    update_parquet_indices("/home/svl/behavior")
    # gc, lightwheel_api_credentials, lw_token = get_credentials(credentials_path="/home/svl/Documents/credentials")
    # tracking_spreadsheet = gc.open("B1K Challenge 2025 Data Replay Tracking Sheet")
    # worksheets = tracking_spreadsheet.worksheets()
    # misc_spreadsheet = gc.open("B50 Task Misc")
    # ws_misc = misc_spreadsheet.worksheet("Test Instances")
    # misc_values = ws_misc.get_all_values()
    # for ws in worksheets:
    #     if ws.title != "Main":
    #         assign_test_instances(ws, ws_misc, misc_values)
