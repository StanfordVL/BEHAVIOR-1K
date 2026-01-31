import pathlib
import shutil
from tqdm.auto import tqdm

def main():
    input_dir = pathlib.Path("/checkpoint/clear/cgokmen/policies-bigrun2")
    output_dir = pathlib.Path("/checkpoint/clear/cgokmen/policies-bigrun2-last-checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)

    last_dirs = list(input_dir.glob("*/checkpoints/last"))
    for last_dir in tqdm(last_dirs):
        policy_name = last_dir.parts[-3]
        shutil.copytree(last_dir, output_dir / policy_name)

if __name__ == "__main__":
    main()