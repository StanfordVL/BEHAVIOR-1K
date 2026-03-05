import pathlib
import json
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

INCLUDE_ROOM_TYPES = {"living_room", "bedroom", "dining_room", "office"}

def load_segmentation(scene_path):
    room_name = scene_path.name
    room_type = room_name.rsplit("_", 1)[0]
    if room_type not in INCLUDE_ROOM_TYPES:
        return None
    segmentation_path = scene_path / "segmentation3d/objects_to_segmentation_maps.json"
    if not segmentation_path.exists():
        return None
    with open(segmentation_path, "r") as f:
        return [k for k in json.load(f)]

def main():
    interesting_scenes = [pathlib.Path(p) for p in json.load(open("/cvgl2/u/cgokmen/BEHAVIOR-1K/slurm/interesting_scenes_full.json"))]
    vid2room_segmentations = {}
    with ProcessPoolExecutor() as executor:
        futures = {}
        for scene_path in tqdm(interesting_scenes):
            future = executor.submit(load_segmentation, scene_path)
            futures[future] = scene_path
        for future in tqdm(as_completed(futures), total=len(futures)):
            scene_path = futures[future]
            segmentation = future.result()
            if segmentation is not None:
                vid2room_segmentations[scene_path] = segmentation

    with open("vid2room_segmentations.json", "w") as f:
        json.dump(vid2room_segmentations, f, indent=2, sort_keys=True)

if __name__ == "__main__":
    main()