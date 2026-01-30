import pathlib
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def check_dir(obj_dir: pathlib.Path):
    success_file = obj_dir / "import.success"
    if not success_file.exists():
        print(f"Success file not found for {obj_dir}")

def main():
    dataset_root = pathlib.Path("/checkpoint/clear/cgokmen/behavior-data-2") / "vid2room"

    scenes_dir = dataset_root / "scenes"
    for scene_dir in scenes_dir.iterdir():
        check_dir(scene_dir)

    objects_dir = dataset_root / "objects"
    for category_dir in objects_dir.iterdir():
        for obj_dir in category_dir.iterdir():
            check_dir(obj_dir)