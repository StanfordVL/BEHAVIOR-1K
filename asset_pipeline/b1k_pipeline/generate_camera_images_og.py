"""Render scene preview images using OmniGibson.

This is the OmniGibson-based counterpart to b1k_pipeline/max/generate_camera_images.py.
Instead of rendering inside 3ds Max, it:

1. Reads the scene camera poses that max/object_list.py exported into the scene's
   object_list.json (the "cameras" entry: position in mm, rotation as an xyzw
   quaternion, and a room/layer assignment).
2. Unpacks artifacts/og_dataset.zip from pack_dataset and loads the scene from
   that copy (InteractiveTraversableScene), rendering an RGB image from each
   camera pose using the viewer camera.
3. Writes the images and metadata into camera_images_og.zip in the scene's
   artifacts directory.

Example:
    OMNIGIBSON_HEADLESS=1 python -m b1k_pipeline.generate_camera_images_og restaurant_diner
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.append(r"D:\BEHAVIOR-1K\asset_pipeline")

import numpy as np
from PIL import Image

import b1k_pipeline.utils

OUTPUT_FILENAME = "{0}.png"
JSON_FILENAME = "generate_camera_images_og.json"


def _load_cameras(scene_name):
    """Load the scene camera poses from the scene's object_list.json."""
    object_list_path = (
        b1k_pipeline.utils.PIPELINE_ROOT
        / "cad"
        / "scenes"
        / scene_name
        / "artifacts"
        / "object_list.json"
    )
    assert object_list_path.exists(), f"Could not find object list at {object_list_path}"
    with open(object_list_path, "r") as f:
        object_list = json.load(f)
    cameras = object_list.get("cameras", {})
    assert cameras, f"No cameras found in {object_list_path}"
    return cameras


def _save_rgb(rgb, output_path):
    """Convert an OmniGibson RGB observation to a uint8 PNG on disk."""
    if hasattr(rgb, "cpu"):
        rgb = rgb.cpu()
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    rgb = rgb[..., :3]
    Image.fromarray(rgb).save(output_path)


def _unpack_dataset(scene_name, data_root):
    """Stage pack_dataset's output and resolve its verified or WIP scene name."""
    dataset_dir = Path(data_root) / "behavior-1k-assets"
    archive_path = b1k_pipeline.utils.PIPELINE_ROOT / "artifacts" / "og_dataset.zip"
    with zipfile.ZipFile(archive_path) as archive:
        filenames = set(archive.namelist())
        for scene_model in (scene_name, f"WIP_{scene_name}"):
            if f"scenes/{scene_model}/json/{scene_model}_best.json" in filenames:
                break
        else:
            raise FileNotFoundError(f"Scene {scene_name} is missing from {archive_path}")
        print(f"Unpacking {archive_path} to {dataset_dir}")
        archive.extractall(dataset_dir)
    return scene_model


def _render_camera_images(scene_name, scene_model, data_root, scene_output_dir, resolution=1024, focal_length=None):
    cameras = _load_cameras(scene_name)

    # Import OmniGibson lazily (and headless) so that simply importing this module
    # (e.g. for _load_cameras) does not spin up the simulator. We set the viewer
    # resolution via gm *before* launching so the viewer camera is created at the
    # desired resolution -- changing it afterwards goes through og.sim.editing_usd()
    # which invalidates the physics view and can crash the RTX renderer.
    from omnigibson.macros import gm

    # pack_dataset contains the scene/object/system assets, but the runtime's
    # robot assets and encryption key are distributed separately.
    installed_data_root = Path(gm.DATA_PATH)
    shutil.copytree(installed_data_root / "omnigibson-robot-assets", Path(data_root) / "omnigibson-robot-assets")
    shutil.copy2(installed_data_root / "omnigibson.key", Path(data_root) / "omnigibson.key")

    with gm.unlocked():
        gm.DATA_PATH = str(data_root)
        gm.USE_ENCRYPTED_ASSETS = True
        gm.HEADLESS = True
        gm.DEFAULT_VIEWER_HEIGHT = resolution
        gm.DEFAULT_VIEWER_WIDTH = resolution

    import omnigibson as og
    import torch as th

    env = og.Environment(
        configs={
            "scene": {
                "type": "InteractiveTraversableScene",
                "scene_model": scene_model,
            }
        }
    )

    cam = og.sim.viewer_camera
    cam.add_modality("rgb")
    if focal_length is not None:
        cam.focal_length = focal_length

    rendered = {}
    for camera_id, camera_info in cameras.items():
        # object_list stores positions in mm; OmniGibson works in meters. The
        # rotation is the max node rotation (obj.rotation), which is already the
        # correct world-frame orientation (no inversion needed) and shares the
        # -Z look / +Y up convention with USD/Isaac cameras.
        position = np.array(camera_info["position"], dtype=np.float64) / 1000.0
        orientation = np.array(camera_info["rotation"], dtype=np.float64)

        cam.set_position_orientation(
            position=th.tensor(position, dtype=th.float32),
            orientation=th.tensor(orientation, dtype=th.float32),
        )

        # This is a static scene preview, so we don't step physics; a few render
        # passes let the RTX renderer converge before capture.
        for _ in range(5):
            og.sim.render()

        obs, _ = cam.get_obs()
        image_path = os.path.abspath(
            os.path.join(scene_output_dir, OUTPUT_FILENAME.format(camera_id))
        )
        _save_rgb(obs["rgb"], image_path)
        print("Saved", image_path)
        rendered[camera_id] = {
            "image": os.path.relpath(image_path, scene_output_dir),
            "room": camera_info.get("room", ""),
        }

    with open(os.path.join(scene_output_dir, JSON_FILENAME), "w") as f:
        json.dump({"success": True, "cameras": rendered}, f, indent=4)

    og.clear()
    return scene_output_dir


def generate_camera_images(scene_name, output_dir=None, resolution=1024, focal_length=None):
    # DVC scene targets include the scenes/ prefix; OmniGibson expects just the name.
    scene_name = scene_name.removeprefix("scenes/")
    if output_dir is None:
        output_dir = b1k_pipeline.utils.PIPELINE_ROOT / "cad" / "scenes" / scene_name / "artifacts"
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Render in a fresh temporary directory so reruns cannot include stale images
    # and leave only the archive in the scene's artifacts directory.
    b1k_pipeline.utils.TMP_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="camera_images_og_", dir=b1k_pipeline.utils.TMP_DIR) as temp_dir:
        data_root = Path(temp_dir) / "data"
        scene_model = _unpack_dataset(scene_name, data_root)
        image_dir = Path(temp_dir) / "images"
        image_dir.mkdir()
        _render_camera_images(
            scene_name, scene_model, data_root, image_dir, resolution=resolution, focal_length=focal_length
        )
        return shutil.make_archive(os.path.join(output_dir, "camera_images_og"), "zip", image_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Render scene preview images from object_list.json camera poses using OmniGibson."
    )
    parser.add_argument("scene_name", help="Scene name or DVC target (e.g. restaurant_diner or scenes/restaurant_diner).")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for camera_images_og.zip. Defaults to the scene's artifacts directory.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Square image resolution in pixels.",
    )
    parser.add_argument(
        "--focal-length",
        type=float,
        default=None,
        help="Optional camera focal length (mm). Defaults to the viewer camera default.",
    )
    args = parser.parse_args()

    archive_path = generate_camera_images(
        args.scene_name,
        output_dir=args.output_dir,
        resolution=args.resolution,
        focal_length=args.focal_length,
    )
    print(f"Done. Images written to {archive_path}")


if __name__ == "__main__":
    main()
