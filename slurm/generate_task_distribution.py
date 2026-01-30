from collections import Counter
import os
import random
import json

config = {
    "behavior-1k-assets": {
        # "train": {
        #     "scenes": [
        #         "Beechwood_0_int",
        #         "Benevolence_1_int",
        #         "Ihlen_0_int",
        #         "Ihlen_1_int",
        #         "Merom_0_int",
        #         "Merom_1_int",
        #         "Pomaria_1_int",
        #         "Pomaria_2_int",
        #         "Rs_int",
        #         "Wainscott_0_int",
        #         "Wainscott_1_int",
        #         "house_single_floor",
        #     ],
        #     "episodes": 20000,
        # },
        "val2": {
            "scenes": [
                "house_double_floor_lower",
                "Pomaria_0_int",
                "Benevolence_2_int",
                "Beechwood_1_int",
            ],
            "episodes": 2000,
        },
    },

    "spoc": {
        # "train": {
        #     "scenes": sorted([x for x in os.listdir("/checkpoint/clear/cgokmen/behavior-data2/spoc/scenes") if "train" in x]),
        #     "episodes": 20000,
        # },
        "val2": {
            "scenes": sorted([x for x in os.listdir("/checkpoint/clear/cgokmen/behavior-data2/spoc/scenes") if "val" in x]),
            "episodes": 2000,
        },
    },

    "vid2room": {
        # "train": {
        #     "scenes": json.load(open("successful_scenes_train.json")),
        #     "episodes": 20000,
        # },
        "val2": {
            "scenes": json.load(open("successful_scenes_val.json")),
            "episodes": 2000,
        },
    }
}

num_jobs = (1024) * 1
max_episodes_per_line = 10

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

    # Divide each batch into max_episodes_per_line
    smaller_batches = []
    for batch in batches:
        smaller_items_batch = []
        for item in batch:
            # Convert this item to multiple items if its count is greater than max_episodes_per_line
            if item[3] > max_episodes_per_line:
                total_remaining = item[3]
                while total_remaining > 0:
                    smaller_items_batch.append((item[0], item[1], item[2], min(total_remaining, max_episodes_per_line)))
                    total_remaining -= max_episodes_per_line
            else:
                smaller_items_batch.append(item)

        # Shuffle the batch
        random.shuffle(smaller_items_batch)

        smaller_batches.append(smaller_items_batch)
    
    # Save the batches to a json file
    random.shuffle(smaller_batches)
    for i, batch in enumerate(smaller_batches):
        with open(f"rollout_jobs/{i}.csv", "w") as f:
            for item in batch:
                f.write(f"{item[0]},{item[1]},{item[2]},{item[3]}\n")

if __name__ == "__main__":
    main()