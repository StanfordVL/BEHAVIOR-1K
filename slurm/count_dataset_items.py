import pathlib
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import shutil

def main():
    dataset_roots = pathlib.Path("/checkpoint/clear/cgokmen/behavior-data2")

    for dataset_name in ["spoc", "vid2room"]:
        with open(f"object_files_{dataset_name}.txt", "w") as object_f, open(f"scene_files_{dataset_name}.txt", "w") as scene_f:
            dataset_root = dataset_roots / dataset_name

            scenes_dir = dataset_root / "scenes"
            for scene_dir in scenes_dir.iterdir():
                scene_f.write(str(scene_dir) + "\n")

            objects_dir = dataset_root / "objects"
            for category_dir in objects_dir.iterdir():
                for obj_dir in category_dir.iterdir():
                    object_f.write(str(obj_dir) + "\n")

if __name__ == "__main__":
    main()