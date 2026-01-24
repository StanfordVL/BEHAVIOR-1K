#!/usr/bin/env python3
"""
Create a one-scene sample zip of the spoc dataset.

This script reads a scene JSON, extracts unique object categories, and zips:
  - spoc/scenes/<scene_name>/
  - spoc/objects/<category>/ for each used category
Plus any top-level files in spoc/.
"""

import argparse
import json
import sys
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a one-scene spoc dataset sample zip."
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default="/home/cgokmen/projects/BEHAVIOR-1K/datasets",
        help="Path to the datasets directory.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="spoc",
        help="Name of the dataset to sample the scene from.",
    )
    parser.add_argument(
        "--scene-name",
        type=str,
        default="train_505",
        help="Scene directory name under scenes/ (e.g., train_505).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="spoc_sample.zip",
        help="Output zip path. Defaults to ./spoc_sample.zip",
    )
    return parser.parse_args()


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

    try:
        scene_json, scene_name = find_scene_json(dataset_root, args.scene_name)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    models = extract_used_models(scene_json)
    if not models:
        print(f"Error: no object models found in {scene_json}", file=sys.stderr)
        return 1

    scene_dir = dataset_root / "scenes" / scene_name
    if not scene_dir.exists():
        print(f"Error: scene dir not found: {scene_dir}", file=sys.stderr)
        return 1

    output_path = args.output

    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as zipf:
        # # Include top-level files in spoc/
        # for item in dataset_root.iterdir():
        #     if item.is_file():
        #         zipf.write(item, item.relative_to(base_parent))

        # Include the selected scene directory
        add_directory(zipf, scene_dir, datasets_dir)

        # Include only used object categories
        for dataset_name, category, model in sorted(models):
            model_dir = datasets_dir / dataset_name / "objects" / category / model
            assert model_dir.exists(), f"Model directory not found: {model_dir}"
            add_directory(zipf, model_dir, datasets_dir)

    print(f"Wrote sample zip: {output_path}")
    print(f"Scene: {scene_name} (json: {scene_json})")
    print(f"Used object models: {len(models)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

