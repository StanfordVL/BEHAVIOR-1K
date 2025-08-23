import os
import re
import pandas as pd


def fix_permissions(root_dir: str):
    """Recursively set rw-rw-r-- for all files owned by the current user."""
    my_uid = os.getuid()
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            try:
                st = os.stat(fpath, follow_symlinks=False)
                if st.st_uid == my_uid:  # check ownership
                    os.chmod(fpath, 0o664)  # rw-rw-r--
            except (PermissionError, FileNotFoundError):
                continue


def update_parquet_indices(root_dir: str):
    """For every parquet file named episode_XXXXXXXX.parquet, update episode_index and task_index."""
    pat = re.compile(r"episode_(\d{8})\.parquet$")

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)

            m = pat.search(fname)
            if not m:
                continue  # not a matching parquet

            episode_num = int(m.group(1))
            task_num = int(str(episode_num)[:4])

            try:
                df = pd.read_parquet(fpath)

                if "episode_index" in df.columns:
                    df["episode_index"] = episode_num
                if "task_index" in df.columns:
                    df["task_index"] = task_num

                # overwrite parquet
                df.to_parquet(fpath, index=False)

            except Exception as e:
                print(f"Skipping {fpath}, error: {e}")
