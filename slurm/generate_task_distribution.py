from collections import Counter
import os
import random
import json

config = {
    "behavior-1k-assets": {
        "train": {
            "scenes": [
                "Beechwood_0_int",
                "Benevolence_1_int",
                "Ihlen_0_int",
                "Ihlen_1_int",
                "Merom_0_int",
                "Merom_1_int",
                "Pomaria_1_int",
                "Pomaria_2_int",
                "Wainscott_0_int",
                "Wainscott_1_int",
                "house_single_floor",
            ],
            "episodes_per_scene": 10,
        },
        "val": {
            "scenes": [
                "Rs_int",
                "house_double_floor_lower",
                "Pomaria_0_int",
                "Benevolence_2_int",
                "Beechwood_1_int",
            ],
            "episodes_per_scene": 10,
        },
    },

    "spoc": {
        "train": {
            "scenes": sorted([x for x in os.listdir("/cvgl2/u/cgokmen/BEHAVIOR-1K/datasets/spoc/scenes") if "train" in x]),
            "episodes_per_scene": 10,
        },
        "val2": {
            "scenes": sorted([x for x in os.listdir("/cvgl2/u/cgokmen/BEHAVIOR-1K/datasets/spoc/scenes") if "val" in x]),
            "episodes_per_scene": 10,
        },
    },

    "vid2room": {
        "train": {
            "scenes": json.load(open("successful_scenes_train.json")),
            "episodes_per_scene": 10,
        },
        "val": {
            "scenes": json.load(open("successful_scenes_val.json")),
            "episodes_per_scene": 10,
        },
    }
}

def main():
    jobs = []
    for dataset, dataset_config in config.items():
        for split, split_config in dataset_config.items():
            episodes_per_scene = split_config["episodes_per_scene"]
            print(f"Dataset: {dataset}, Split: {split}, Scenes: {len(split_config['scenes'])}, Episodes per scene: {episodes_per_scene}")
            scenes = list(split_config["scenes"])
            for scene in split_config["scenes"]:
                jobs.append((dataset, split, scene, episodes_per_scene))
    
    # Save the jobs to a json file
    random.shuffle(jobs)
    for i, job in enumerate(jobs):
        with open(f"rollout_jobs/{i}.csv", "w") as f:
            f.write(f"{job[0]},{job[1]},{job[2]},{job[3]}\n")

if __name__ == "__main__":
    main()