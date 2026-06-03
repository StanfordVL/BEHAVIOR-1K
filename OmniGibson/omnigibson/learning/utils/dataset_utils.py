import os
from typing import Tuple


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
