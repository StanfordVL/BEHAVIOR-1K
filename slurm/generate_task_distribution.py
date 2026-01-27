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
                "Rs_int",
                "Wainscott_0_int",
                "Wainscott_1_int",
                "house_single_floor",
            ],
            "episodes": 12000,
        },
        "val": {
            "scenes": [
                "house_double_floor_lower",
                "Pomaria_0_int",
                "Benevolence_2_int",
                "Beechwood_1_int",
            ],
            "episodes": 1200,
        },
    },

    "spoc": {
        "train": {
            "scenes": sorted([x for x in os.listdir("/checkpoint/clear/cgokmen/behavior-data2/spoc/scenes") if "train" in x]),
            "episodes": 12000,
        },
        "val": {
            "scenes": sorted([x for x in os.listdir("/checkpoint/clear/cgokmen/behavior-data2/spoc/scenes") if "val" in x]),
            "episodes": 1200,
        },
    },
}

num_jobs = 256 * 4

def main():
    all_episodes = []
    for dataset, dataset_config in config.items():
        for split, split_config in dataset_config.items():
            split_episodes = []
            episodes_per_scene = split_config["episodes"] // len(split_config["scenes"]) + 1
            print(f"Dataset: {dataset}, Split: {split}, Scenes: {len(split_config['scenes'])}, Episodes per scene: {episodes_per_scene}")
            scenes = list(split_config["scenes"])
            random.shuffle(scenes)
            for scene in split_config["scenes"]:
                for i in range(episodes_per_scene):
                    if len(split_episodes) >= split_config["episodes"]:
                        break
                    split_episodes.append((dataset, split, scene))

            all_episodes.extend(split_episodes)
    
    # Divide up the episodes into num_jobs
    batch_size = len(all_episodes) // num_jobs
    batches = []
    for i in range(num_jobs):
        this_batch = all_episodes[i * batch_size:(i + 1) * batch_size]

        # Count the batch and unpack back into the original format with a count
        batch_counter = Counter(this_batch)
        batches.append([(*k, v) for k, v in batch_counter.items()])

    # Save the batches to a json file
    for i, batch in enumerate(batches):
        with open(f"rollout_jobs/{i}.csv", "w") as f:
            for item in batch:
                f.write(f"{item[0]},{item[1]},{item[2]},{item[3]}\n")

if __name__ == "__main__":
    main()