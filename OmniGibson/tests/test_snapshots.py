"""
Deterministic snapshot tests.
Each test loads a fixed scene and compares sensor output against a stored .npy reference.

To regenerate all snapshots:
    SNAPSHOT_UPDATE=1 OMNIGIBSON_HEADLESS=1 pytest tests/test_snapshots.py

A companion .png is written alongside each .npy on every test run for visual
inspection.  Commit both the .npy and .png files together.
"""

import os
from pathlib import Path

import numpy as np
import omnigibson as og
import pytest
import torch as th
from omnigibson.objects import DatasetObject
from PIL import Image

SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"
GOLDEN_DIR = SNAPSHOTS_DIR / "golden"
ACTUAL_DIR = SNAPSHOTS_DIR / "actual"

# Camera pose used across multiple tests, positioned to view the Rs_int kitchen area.
_CAM_POS = th.tensor([1.5, -4, 2.25])
_CAM_ORN = th.tensor([0.56829048, 0.09569975, 0.13571846, 0.80589577])


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------


def _id_to_rgb(semantic_id: int) -> tuple[int, int, int]:
    """Map a semantic class ID to a stable RGB colour via a simple hash spread."""
    h = int(semantic_id) & 0xFFFFFFFF
    h = ((h >> 16) ^ h) * 0x45D9F3B
    h = ((h >> 16) ^ h) * 0x45D9F3B
    h = (h >> 16) ^ h
    return ((h & 0xFF0000) >> 16, (h & 0x00FF00) >> 8, h & 0x0000FF)


def _array_to_png(array: np.ndarray, png_path: Path) -> None:
    """
    Write a 2-D numpy array as a colour PNG.

    Integer arrays (seg_semantic, seg_instance) are false-coloured by unique ID.
    Float arrays (depth_linear, depth) are normalised to grayscale.
    """
    if np.issubdtype(array.dtype, np.integer):
        h, w = array.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for uid in np.unique(array):
            rgb[array == uid] = _id_to_rgb(int(uid))
    else:
        finite = array[np.isfinite(array)]
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
        normalized = np.clip((array - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        gray = (normalized * 255).astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)
    Image.fromarray(rgb).save(png_path)


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------


def _update_mode():
    return os.environ.get("SNAPSHOT_UPDATE", "").lower() in ("1", "true", "yes")


def _capture(camera_pos, camera_orn, modalities):
    """Step the sim, render, and return a dict of {modality: numpy array}."""
    cam = og.sim.viewer_camera
    cam.set_position_orientation(position=camera_pos, orientation=camera_orn)
    for m in modalities:
        cam.add_modality(m)

    og.sim.step()
    for _ in range(3):
        og.sim.render()

    obs, _ = cam.get_obs()
    out = {}
    for m in modalities:
        data = obs[m]
        if hasattr(data, "cpu"):
            data = data.cpu()
        out[m] = np.array(data)
    return out


def _check(name, array):
    """Compare array against stored golden reference, or save it if SNAPSHOT_UPDATE is set.

    On mismatch the actual output is written to snapshots/actual/ for debugging.
    """
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    path = GOLDEN_DIR / f"{name}.npy"

    if _update_mode():
        np.save(path, array)
        png_path = path.with_suffix(".png")
        _array_to_png(array, png_path)
        print(f"Image can be viewed at : {png_path}")
        pytest.skip(f"Snapshot updated: {path}")

    if not path.exists():
        pytest.fail(
            f"Missing golden snapshot '{name}': {path}. " "Set SNAPSHOT_UPDATE=1 to generate it and commit the result."
        )

    reference = np.load(path)
    shape_match = array.shape == reference.shape
    pixel_match = shape_match and np.array_equal(array, reference)

    if not pixel_match:
        ACTUAL_DIR.mkdir(parents=True, exist_ok=True)
        actual_npy = ACTUAL_DIR / f"{name}.npy"
        actual_png = ACTUAL_DIR / f"{name}.png"
        np.save(actual_npy, array)
        _array_to_png(array, actual_png)

    if not shape_match:
        pytest.fail(f"Shape mismatch for '{name}': got {array.shape}, expected {reference.shape}")
    assert pixel_match, f"Snapshot mismatch for '{name}'. Set SNAPSHOT_UPDATE=1 to regenerate the reference."


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


# skip this test for now since it's flaky on CI
# TODO @stefren: investigate and re-enable
@pytest.mark.skip(reason="Flaky test, investigate and re-enable")
def test_snapshot_rs_int():
    """
    Semantic segmentation of Rs_int with a bowl and apple placed at fixed positions.
    Covers: interactive scene loading, object placement, seg_semantic modality.
    """
    env = og.Environment(configs={"scene": {"type": "InteractiveTraversableScene", "scene_model": "Rs_int"}})

    bowl = DatasetObject(name="bowl", category="bowl", model="ajzltc")
    env.scene.add_object(bowl)
    bowl.set_position_orientation(position=th.tensor([0.5, -1.5, 0.5]), frame="scene")

    apple = DatasetObject(name="apple", category="apple", model="agveuv")
    env.scene.add_object(apple)
    apple.set_position_orientation(position=th.tensor([0.8, -1.5, 0.5]), frame="scene")

    obs = _capture(_CAM_POS, _CAM_ORN, ["seg_semantic"])
    og.clear()

    _check("rs_int_with_objects_seg_semantic", obs["seg_semantic"])


@pytest.mark.skip(reason="Flaky test, investigate and re-enable")
def test_snapshot_items_in_scene():
    """
    Semantic segmentation of a Fetch robot and a few objects in an empty scene.
    Covers: robot loading, robot segmentation class, viewer camera setup.
    """
    env = og.Environment(
        configs={
            "scene": {"type": "Scene"},
            "robots": [{"type": "Fetch", "obs_modalities": [], "position": [0, 0, 0], "orientation": [0, 0, 0, 1]}],
        }
    )
    for name, category, model, pos in [
        ("bowl", "bowl", "ajzltc", [-0.5, 0.0, 0.1]),
        ("apple", "apple", "agveuv", [0.0, 1.0, 0.1]),
        ("microwave", "microwave", "hjjxmi", [0.5, 3.0, 0.1]),
        ("stove", "stove", "dusebh", [1.0, 0.0, 0.1]),
    ]:
        obj = DatasetObject(name=name, category=category, model=model)
        env.scene.add_object(obj)
        obj.set_position_orientation(position=th.tensor(pos), frame="scene")

    obs = _capture(_CAM_POS, _CAM_ORN, ["seg_semantic"])
    og.clear()

    _check("robot_in_empty_scene_seg_semantic", obs["seg_semantic"])
