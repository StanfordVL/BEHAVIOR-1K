#!/usr/bin/env python3
"""
Create a sample zip of a dataset with randomly sampled scenes.

This script reads scene JSONs, extracts unique object categories, and zips:
  - <dataset_name>/scenes/<scene_name>/ for each sampled scene
  - <dataset_name>/objects/<category>/ for each used category
Plus any top-level files in <dataset_name>/.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import shutil

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a dataset sample zip with randomly sampled scenes."
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default="/cvgl2/u/cgokmen/BEHAVIOR-1K/datasets",
        help="Path to the datasets directory.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="vid2room",
        help="Name of the dataset to sample the scene from.",
    )
    parser.add_argument(
        "--num-scenes",
        "-n",
        type=int,
        default=20,
        help="Number of scenes to randomly sample (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="./sample.zip",
        help="Output directory or zip path. Defaults to ./sample.zip",
    )
    return parser.parse_args()


def get_available_scenes(dataset_root: Path) -> list[str]:
    """Get all available scene names from the dataset."""
    scenes_dir = dataset_root / "scenes"
    if not scenes_dir.exists():
        raise FileNotFoundError(f"Scenes directory not found: {scenes_dir}")
    
    scenes = []
    for scene_dir in scenes_dir.iterdir():
        if (scene_dir / "import.success").exists():
            scenes.append(scene_dir.name)
    return sorted(scenes)


def find_scene_json(dataset_root: Path, scene_name: str) -> tuple[Path, str]:
    scene_json_dir = dataset_root / "scenes" / scene_name / "json"
    if not scene_json_dir.exists():
        raise FileNotFoundError(f"Scene json dir not found: {scene_json_dir}")

    best_json = scene_json_dir / f"{scene_name}_best.json"
    if best_json.exists():
        return best_json, scene_name

    json_files = sorted(scene_json_dir.glob("*.json"))
    if len(json_files) == 1:
        return json_files[0], scene_name
    if not json_files:
        raise FileNotFoundError(f"No scene json files in: {scene_json_dir}")
    raise ValueError(f"Multiple json files found in {scene_json_dir}.")


def extract_used_models(scene_json: Path) -> set[tuple[str, str, str]]:
    with scene_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    objects_info = data.get("objects_info", {}).get("init_info", {})
    models = set()
    for obj_info in objects_info.values():
        args = obj_info.get("args", {})
        models.add((args["dataset_name"], args["category"], args["model"]))
    return models


def add_directory(zipf: ZipFile, dir_path: Path, base_parent: Path) -> None:
    for path in dir_path.rglob("*"):
        if path.is_file():
            arcname = path.relative_to(base_parent)
            zipf.write(path, arcname)


def main() -> int:
    args = parse_args()
    datasets_dir = args.datasets_dir
    dataset_root = datasets_dir / args.dataset_name

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)

    # Get all available scenes
    try:
        available_scenes = get_available_scenes(dataset_root)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not available_scenes:
        print(f"Error: no scenes found in {dataset_root / 'scenes'}", file=sys.stderr)
        return 1

    # Sample N scenes
    num_scenes = min(args.num_scenes, len(available_scenes))
    if num_scenes < args.num_scenes:
        print(f"Warning: only {len(available_scenes)} scenes available, sampling all.", file=sys.stderr)
    
    random.shuffle(available_scenes)

    # Collect all models from all sampled scenes
    sampled_scenes = []
    all_models: set[tuple[str, str, str]] = set()
    scene_jsons: list[tuple[str, Path]] = []

    for scene_name in available_scenes:
        try:
            scene_json, _ = find_scene_json(dataset_root, scene_name)
            models = extract_used_models(scene_json)
            for dataset_name, category, model in models:
                model_dir = datasets_dir / dataset_name / "objects" / category / model
                assert model_dir.exists(), f"Model directory not found: {model_dir}"
            all_models.update(models)
            sampled_scenes.append(scene_name)
            scene_jsons.append((scene_name, scene_json))
            if len(sampled_scenes) == num_scenes:
                break
        except (FileNotFoundError, ValueError, AssertionError) as exc:
            print(f"Warning: skipping scene {scene_name}: {exc}", file=sys.stderr)
            continue

    print(f"Sampled {num_scenes} scenes: {sampled_scenes}")

    if not scene_jsons:
        print("Error: no valid scenes could be processed", file=sys.stderr)
        return 1

    output_path = args.output

    if str(output_path).endswith(".zip"):
        with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as zipf:
            # Include all sampled scene directories
            for scene_name, scene_json in scene_jsons:
                scene_dir = dataset_root / "scenes" / scene_name
                if scene_dir.exists():
                    add_directory(zipf, scene_dir, datasets_dir)
                    print(f"  Added scene: {scene_name}")

            # Include only used object categories
            for dataset_name, category, model in sorted(all_models):
                model_dir = datasets_dir / dataset_name / "objects" / category / model
                assert model_dir.exists(), f"Model directory not found: {model_dir}"
                add_directory(zipf, model_dir, datasets_dir)

    else:
        output_path.mkdir(parents=True, exist_ok=True)
        for scene_name, scene_json in scene_jsons:
            scene_dir = dataset_root / "scenes" / scene_name
            if scene_dir.exists():
                shutil.copytree(scene_dir, output_path / "scenes" / scene_name)

        # Copy all object categories
        for dataset_name, category, model in sorted(all_models):
            model_dir = datasets_dir / dataset_name / "objects" / category / model
            assert model_dir.exists(), f"Model directory not found: {model_dir}"
            shutil.copytree(model_dir, output_path / "objects" / category / model, dirs_exist_ok=True)

    print(f"\nWrote sample zip: {output_path}")
    print(f"Scenes included: {len(scene_jsons)}")
    print(f"Unique object models: {len(all_models)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
