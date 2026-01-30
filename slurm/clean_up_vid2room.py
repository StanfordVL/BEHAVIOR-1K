import pathlib
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import shutil

def check_dir(obj_dir: pathlib.Path):
    success_file = obj_dir / "import.success"
    if not success_file.exists():
        # print(f"Success file not found for {obj_dir}")
        shutil.rmtree(obj_dir)
        return False
    return True

def main():
    dataset_roots = pathlib.Path("/checkpoint/clear/cgokmen/behavior-data2")

    with ProcessPoolExecutor(max_workers=10) as executor:
        bad_files = []
        futures = {}

        for dataset_name in ["spoc", "hssd", "ai2thor"]:
            dataset_root = dataset_roots / dataset_name

            scenes_dir = dataset_root / "scenes"
            for scene_dir in scenes_dir.iterdir():
                future = executor.submit(check_dir, scene_dir)
                futures[future] = scene_dir

            objects_dir = dataset_root / "objects"
            for category_dir in objects_dir.iterdir():
                for obj_dir in category_dir.iterdir():
                    future = executor.submit(check_dir, obj_dir)
                    futures[future] = obj_dir

        for future in tqdm(as_completed(futures), total=len(futures), desc="Checking directories"):
            if not future.result():
                bad_files.append(str(futures[future]))

        print(f"Found {len(bad_files)} bad files out of {len(futures)} total files")
        with open("bad_files.json", "w") as f:
            json.dump(sorted(bad_files), f, indent=2, sort_keys=True)

if __name__ == "__main__":
    main()