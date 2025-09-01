import argparse
import json
import os
from huggingface_hub import HfApi, login

print(os.cpu_count())

home = os.environ.get("HOME")
with open(f"{home}/Documents/credentials/hf_credentials.json") as f:
    credentials = json.load(f)
login(token=credentials["token"])


parser = argparse.ArgumentParser()
parser.add_argument("--folder_path", type=str, default="/scr/behavior/2025-challenge-demos")
parser.add_argument("--task_id", type=int, required=True)
args = parser.parse_args()

hub_api = HfApi()


hub_api.create_branch(
    repo_id="behavior-1k/B50",
    branch=f"task-{args.task_id:04d}",
    repo_type="dataset",
    exist_ok=True,
)

print(f"Branch task-{args.task_id:04d} created.")

hub_api.upload_large_folder(
    repo_id="behavior-1k/2025-challenge-demos",
    folder_path=args.folder_path,
    repo_type="dataset",
    revision=f"task-{args.task_id:04d}",
    private=True,
    num_workers=os.cpu_count() - 2,
    allow_patterns=[f"**/task-{args.task_id:04d}/**"],
    ignore_patterns=["raw/**", "meta/**"],
)

print(f"Upload complete for task {args.task_id}!")
