from collections import Counter
import os
import random
import json
import pathlib

policy_names = {
    "behavior-1k-assets-1",
    "behavior-1k-assets-5",
    "behavior-1k-assets-10",
    "spoc-1",
    "spoc-5",
    "spoc-10",
    "spoc-20",
    "spoc-50",
    "vid2room-1",
    "vid2room-5",
    "vid2room-10",
    "vid2room-20",
    "vid2room-50",
}

outputs_path = pathlib.Path("/cvgl2/u/cgokmen/BEHAVIOR-1K/slurm/eval_jobs_norgb")
assert not outputs_path.exists(), f"Outputs path already exists: {outputs_path}"
outputs_path.mkdir(parents=True, exist_ok=True)

def main():
    with open("/cvgl2/u/cgokmen/BEHAVIOR-1K/slurm/eval_configurations.json", "r") as f:
        eval_configurations = json.load(f)

    all_eval_jobs = []
    for policy_name in policy_names:
        policy_checkpoint_path = pathlib.Path("/vision/group/vid2room/vid2room_pick_policies_norgb") / policy_name / "checkpoints" / "step_03000.pt"
        assert os.path.exists(policy_checkpoint_path), f"Policy checkpoint path does not exist: {policy_checkpoint_path}"

        for eval_configuration in eval_configurations:
            job = dict(eval_configuration)
            job["checkpoint_path"] = str(policy_checkpoint_path)
            all_eval_jobs.append(job)

    random.shuffle(all_eval_jobs)

    for i, job in enumerate(all_eval_jobs):
        with open(outputs_path / f"{i}.json", "w") as f:
            json.dump(job, f)

if __name__ == "__main__":
    main()