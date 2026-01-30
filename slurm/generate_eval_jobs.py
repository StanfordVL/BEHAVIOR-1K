from collections import Counter
import os
import random
import json
import pathlib

policy_paths = {
    "pi05ddp-b": "/checkpoint/clear/cgokmen/policies-bigrun2/pi05ddp-b-2123187/checkpoints/004000/pretrained_model",
    "pi05ddp-bv": "/checkpoint/clear/cgokmen/policies-bigrun2/pi05ddp-bv-2123675/checkpoints/004000/pretrained_model",
    "pi05ddp-bp": "/checkpoint/clear/cgokmen/policies-bigrun2/pi05ddp-bp-2123619/checkpoints/004000/pretrained_model",
    "pi05ddp-bpv": "/checkpoint/clear/cgokmen/policies-bigrun2/pi05ddp-bpv-2123618/checkpoints/003000/pretrained_model",
    "actddp-bpv": "/checkpoint/clear/cgokmen/policies-bigrun2/actddp-bpv-2123242/checkpoints/004000/pretrained_model",
    # "customdp-bpv": "/checkpoint/clear/cgokmen/policies-bigrun2/customdpddp-bpv-212888/x/pretrained_model",
    "customdp-bv": "/checkpoint/clear/cgokmen/policies-bigrun/customdp-bv-2117603/checkpoints/010000/pretrained_model",
    # "dpddp-bpv": "/checkpoint/clear/cgokmen/policies-bigrun2/dpddp-bpv-2123522/x/pretrained_model",
    "dpddp-b": "/checkpoint/clear/cgokmen/policies-bigrun2/dpddp-b-2123185/checkpoints/004000/pretrained_model",
}

num_jobs = (350) * 1
max_eval_instances = 50
outputs_path = pathlib.Path("/checkpoint/clear/cgokmen/eval-results")

def main():
    all_episodes = []
    eval_json_files = [f for f in pathlib.Path("/home/cgokmen/projects/BEHAVIOR-1K/slurm/eval-starts").glob("*.json")]
    random.seed(42)
    random.shuffle(eval_json_files)
    eval_json_files = eval_json_files[:max_eval_instances]

    for policy_name, policy_checkpoint_path in policy_paths.items():
        assert os.path.exists(policy_checkpoint_path), f"Policy checkpoint path does not exist: {policy_checkpoint_path}"
        policy_output_path = outputs_path / policy_name
        policy_output_path.mkdir(parents=True, exist_ok=True)
        for eval_json_file in eval_json_files:
            target_output_path = policy_output_path / eval_json_file.stem
            all_episodes.append((policy_checkpoint_path, str(eval_json_file), str(target_output_path)))
    
    # Divide up the episodes into num_jobs
    batch_size = len(all_episodes) // num_jobs
    assert batch_size * num_jobs == len(all_episodes), f"Number of episode {len(all_episodes)} must be divisible by number of jobs {num_jobs}"
    batches = []
    for i in range(num_jobs):
        this_batch = all_episodes[i * batch_size:(i + 1) * batch_size]
        batches.append(this_batch)
    
    # Save the batches to a json file
    for i, batch in enumerate(batches):
        with open(f"eval_jobs/{i}.csv", "w") as f:
            for item in batch:
                f.write(f"{item[0]},{item[1]},{item[2]}\n")

if __name__ == "__main__":
    main()