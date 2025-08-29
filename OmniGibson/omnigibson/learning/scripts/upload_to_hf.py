import os
from huggingface_hub import HfApi

print(os.cpu_count())

hub_api = HfApi()


hub_api.upload_large_folder(
    repo_id="behavior-1k/B50",
    folder_path="~/behavior",
    repo_type="dataset",
    private=True,
    ignore_patterns=["raw/**"],
)
