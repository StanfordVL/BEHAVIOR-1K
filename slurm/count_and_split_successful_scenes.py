import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import json
import hashlib

VAL_FRACTION_ONE_EVERY_N_SCENES = 10

def check_scene_success(scene_path):
    return (scene_path / "import.success").exists()

def main():
    scenes_dir = pathlib.Path("/checkpoint/clear/cgokmen/behavior-data2/vid2room/scenes")

    successful_scenes = []
    failed_scenes = []
    with ProcessPoolExecutor() as executor:
        futures = {}
        for scene in scenes_dir.iterdir():
            future = executor.submit(check_scene_success, scene)
            futures[future] = scene
        for future in tqdm(as_completed(futures), total=len(futures), desc="Counting successful scenes"):
            result = future.result()
            scene = futures[future]
            if result:
                successful_scenes.append(scene.name)
            else:
                failed_scenes.append(scene.name)

    print(f"Successful scenes: {len(successful_scenes)}")
    print(f"Failed scenes: {len(failed_scenes)}")

    train_scenes = []
    val_scenes = []
    for scene in successful_scenes:
        if int(hashlib.md5((str(scene) + "tomato").encode()).hexdigest(), 16) % VAL_FRACTION_ONE_EVERY_N_SCENES == 0:
            val_scenes.append(scene)
        else:
            train_scenes.append(scene)

    print(f"Train scenes: {len(train_scenes)}")
    print(f"Val scenes: {len(val_scenes)}")

    with open("successful_scenes_train.json", "w") as f:
        json.dump(train_scenes, f, indent=2, sort_keys=True)
    with open("successful_scenes_val.json", "w") as f:
        json.dump(val_scenes, f, indent=2, sort_keys=True)

if __name__ == "__main__":
    main()