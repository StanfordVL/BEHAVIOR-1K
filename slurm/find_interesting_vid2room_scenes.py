import pathlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
import itertools

import sys
sys.path.append("/home/cgokmen/projects/BEHAVIOR-1K/vid2room_policy")

from vid2scene_policy.data_collection.object_classifier import ObjectClassifier


OBJECT_CLASSIFIER = ObjectClassifier("/checkpoint/clear/cgokmen/vid2room/RealEstate10K/object_embeddings.npz", "/home/cgokmen/projects/BEHAVIOR-1K/vid2room_policy/vid2scene_policy/data_collection/classifiers")

def is_object_interesting(obj_category: str) -> bool:
    """
    Check if an object is interesting.
    """
    return OBJECT_CLASSIFIER.is_support(obj_category) and "cabinet" not in obj_category.lower()

def is_scene_interesting(scene_path: pathlib.Path) -> bool:
    """
    Check if a scene is interesting.
    """
    
    # Get the list of objects in this scene
    object_files = list(scene_path.glob("obj_meshes_v9_pointmap/*.json"))
    object_categories = Counter([obj_file.stem.rsplit("-", 2)[0] for obj_file in object_files])
    objects_interesting = {category: count for category, count in object_categories.items() if is_object_interesting(category)}
    if len(objects_interesting) < 2:
        return None
    return dict(objects_interesting)

def main():
    dataset_root = pathlib.Path("/checkpoint/clear/cgokmen/vid2room/RealEstate10K")

    with ProcessPoolExecutor(max_workers=64) as executor:
        futures = {}
        good_scenes = {}
        for scene_path in tqdm(dataset_root.glob("*/rooms/*/association.success"), desc="Queueing scenes"):
            scene_root = scene_path.parent
            future = executor.submit(is_scene_interesting, scene_root)
            futures[future] = scene_root
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing scenes"):
            result = future.result()
            if result is None:
                continue
            scene_root = str(futures[future])
            good_scenes[scene_root] = result

        print(f"Found {len(good_scenes)} interesting scenes out of {len(futures)} total")

    with open("interesting_scenes.json", "w") as f:
        json.dump(good_scenes, f, indent=2, sort_keys=True)

if __name__ == "__main__":
    main()