"""
Render converted vid2room scenes from the original camera viewpoints.

For each scene in the interesting_scenes list, loads the converted OmniGibson
scene and re-renders every frame from the original video's camera trajectory.
Outputs:
  - rgb/*.png       (RGBA frames)
  - depth/*.npz     (float32 linear depth in meters)
  - seg/*.npz       (int instance segmentation)
  - poses/*.npz     (camera poses in both OpenCV and OpenGL conventions)
  - metadata.json   (intrinsics, resolution, filenames)
  - seg_info.json   (segmentation ID -> object name mapping)
"""

import argparse
import hashlib
import json
import pathlib
import traceback

import numpy as np
import torch as th
from PIL import Image
from scipy.spatial.transform import Rotation as R

from omnigibson.macros import gm

gm.HEADLESS = True
gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_OBJECT_STATES = False
gm.ENABLE_TRANSITION_RULES = False

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.utils.asset_utils import get_dataset_path

from omnigibson.examples.scenes.load_vid2room_scene import get_scene_id

# 180-degree rotation around X: converts OpenCV camera convention (Y-down, Z-forward)
# to OpenGL camera convention (Y-up, Z-backward).
_OPENCV_TO_OPENGL = np.eye(4)
_OPENCV_TO_OPENGL[:3, :3] = R.from_euler("x", [180], degrees=True).as_matrix()

RENDER_WARMUP_STEPS = 5
RENDER_SETTLE_STEPS = 3


def get_original_resolution(room_dir, filenames):
    """Detect original video resolution from an image in the room directory."""
    for fn in filenames:
        img_path = room_dir / "images" / fn
        if img_path.exists():
            with Image.open(img_path) as img:
                return img.size  # (width, height)
    return None


def setup_camera_from_intrinsics(intrinsics_3x3, pm_h, pm_w, render_w, render_h):
    """
    Scale intrinsics from pointmap resolution to render resolution
    and derive OmniGibson camera parameters (focal_length, horizontal_aperture).

    Returns (scaled_K, focal_length_mm, horizontal_aperture_mm).
    """
    K = intrinsics_3x3.copy()
    K[0] *= render_w / pm_w
    K[1] *= render_h / pm_h

    fx = K[0, 0]
    focal_length_mm = 17.0
    horizontal_aperture_mm = focal_length_mm * render_w / fx
    return K, focal_length_mm, horizontal_aperture_mm


def render_scene(room_dir, output_dir, scene_id, render_w=None, render_h=None):
    """
    Load one pre-converted vid2room scene and render every original camera viewpoint.

    Args:
        room_dir: pathlib.Path to the vid2room room directory (for camera data).
        output_dir: Where to write rendered outputs.
        scene_id: OmniGibson scene name (e.g. vid_xxx_room_0).
        render_w, render_h: Override render resolution. If None, uses original video resolution.
    """
    room_dir = pathlib.Path(room_dir)
    output_dir = pathlib.Path(output_dir)

    # ── Load camera data from the reconstruction ──
    cnp = np.load(room_dir / "sparse_pi3x/0/cameras_and_points.npz")
    filenames = list(cnp["filenames"])
    camera_poses = cnp["camera_poses"]  # (N, 4, 4) c2w in OpenCV convention
    intrinsics = cnp["intrinsics"][0].copy()
    _, pm_h, pm_w, _ = cnp["local_points"].shape

    # ── Determine render resolution ──
    if render_w is None or render_h is None:
        orig_res = get_original_resolution(room_dir, filenames)
        if orig_res is not None:
            render_w, render_h = orig_res
        else:
            render_w, render_h = pm_w, pm_h
            print(f"  Warning: could not detect original resolution, using pointmap resolution {pm_w}x{pm_h}")

    K, focal_length_mm, h_aperture_mm = setup_camera_from_intrinsics(
        intrinsics, pm_h, pm_w, render_w, render_h
    )

    # ── Create output directories ──
    rgb_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth"
    seg_dir = output_dir / "seg"
    poses_dir = output_dir / "poses"
    for d in (rgb_dir, depth_dir, seg_dir, poses_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── Load the pre-converted scene via OmniGibson Environment ──
    cfg = {
        "render": {"viewer_width": int(render_w), "viewer_height": int(render_h)},
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": scene_id,
            "scene_instance": f"{scene_id}_best",
            "dataset_name": "vid2room",
        },
    }
    env = og.Environment(configs=cfg)
    og.sim.step()

    # ── Configure renderer ──
    lazy.carb.settings.get_settings().set_string("/rtx/rendermode", "RaytracedLighting")
    lazy.carb.settings.get_settings().set_bool("/rtx/useViewLightingMode", True)
    lazy.carb.settings.get_settings().set_bool("/rtx/post/histogram/enabled", True)
    lazy.carb.settings.get_settings().set_float("/rtx/post/histogram/whiteScale", 5.0)

    # ── Set camera intrinsics ──
    cam = og.sim.viewer_camera
    cam.focal_length = focal_length_mm
    cam.horizontal_aperture = h_aperture_mm

    for _ in range(RENDER_WARMUP_STEPS):
        og.sim.render()

    actual_K = cam.intrinsic_matrix.cpu().numpy()
    print(f"  Resolution: {render_w}x{render_h}")
    print(f"  Target fx={K[0, 0]:.1f}  Actual fx={actual_K[0, 0]:.1f}")

    # ── Save metadata ──
    metadata = {
        "intrinsics_target": K.tolist(),
        "intrinsics_actual": actual_K.tolist(),
        "render_width": int(render_w),
        "render_height": int(render_h),
        "focal_length_mm": focal_length_mm,
        "horizontal_aperture_mm": h_aperture_mm,
        "num_frames": len(camera_poses),
        "filenames": filenames,
        "room_dir": str(room_dir),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # ── Render loop ──
    seg_infos = {}
    for i, (c2w_opencv, fname) in enumerate(zip(camera_poses, filenames)):
        frame_id = f"{i:05d}"

        c2w_opengl = c2w_opencv @ _OPENCV_TO_OPENGL
        pos, quat = T.mat2pose(th.tensor(c2w_opengl, dtype=th.float32))
        cam.set_position_orientation(position=pos, orientation=quat)

        for _ in range(RENDER_SETTLE_STEPS):
            og.sim.render()

        obs, obs_info = cam.get_obs()
        rgb = obs["rgb"].detach().cpu().numpy()
        depth = obs["depth_linear"].detach().cpu().numpy()
        seg = obs["seg_instance"].detach().cpu().numpy()
        seg_infos.update(obs_info["seg_instance"])

        Image.fromarray(rgb).save(str(rgb_dir / f"{frame_id}.png"))
        np.savez_compressed(str(depth_dir / f"{frame_id}.npz"), depth=depth)
        np.savez_compressed(str(seg_dir / f"{frame_id}.npz"), seg=seg)
        np.savez_compressed(
            str(poses_dir / f"{frame_id}.npz"),
            c2w_opencv=c2w_opencv,
            c2w_opengl=c2w_opengl,
            filename=fname,
        )

        if (i + 1) % 20 == 0 or i == len(camera_poses) - 1:
            print(f"  Frame {i + 1}/{len(camera_poses)}")

    # ── Save segmentation mapping ──
    (output_dir / "seg_info.json").write_text(
        json.dumps({str(k): v for k, v in seg_infos.items()}, indent=2)
    )


def main():
    parser = argparse.ArgumentParser(description="Render vid2room scenes from original camera viewpoints")
    parser.add_argument(
        "--scene-list",
        type=str,
        default="/cvgl2/u/cgokmen/BEHAVIOR-1K/slurm/interesting_scenes.json",
    )
    parser.add_argument("--output-root", type=str, default=None,
                        help="Root directory for rendered outputs. Defaults to <dataset_path>/vid2room/renders")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--total-tasks", type=int, default=1)
    parser.add_argument("--restart-every", type=int, default=8,
                        help="Restart the process after this many scenes (0 = no limit)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--success-file", type=str, default="")
    parser.add_argument("--render-width", type=int, default=None,
                        help="Override render width (default: original video resolution)")
    parser.add_argument("--render-height", type=int, default=None,
                        help="Override render height (default: original video resolution)")
    args = parser.parse_args()

    dataset_root = pathlib.Path(get_dataset_path("vid2room"))
    output_root = pathlib.Path(args.output_root) if args.output_root else dataset_root / "renders"
    output_root.mkdir(parents=True, exist_ok=True)

    processed = 0
    scene_list = json.loads(pathlib.Path("/cvgl2/u/cgokmen/BEHAVIOR-1K/slurm/interesting_scenes.json").read_text())
    scene_list = [pathlib.Path(k) for k in scene_list if int(hashlib.md5(f"{k}-potato".encode()).hexdigest(), 16) % args.total_tasks == args.task_id]

    for room_dir in scene_list:
        scene_id = get_scene_id(room_dir)

        # Check that the scene has been converted (JSON exists)
        scene_json = dataset_root / "scenes" / scene_id / "json" / f"{scene_id}_best.json"
        if not scene_json.exists():
            print(f"Skipping {scene_id}: scene not converted (no {scene_json})")
            continue

        output_dir = output_root / scene_id
        success_path = output_dir / "render.success"

        if success_path.exists() and not args.overwrite:
            print(f"Skipping {scene_id}: already rendered")
            continue

        print(f"Rendering {scene_id} ...")
        try:
            render_scene(room_dir, output_dir, scene_id, args.render_width, args.render_height)
            success_path.touch()
            processed += 1
            print(f"  Done ({processed} scenes so far)")
        except Exception:
            exc = traceback.format_exc()
            print(f"Error rendering {scene_id}:\n{exc}")
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "render.error").write_text(exc)
        finally:
            og.clear()

        if args.restart_every and processed >= args.restart_every:
            break

    print(f"Rendered {processed} scenes total")
    if args.success_file:
        pathlib.Path(args.success_file).touch()

    og.shutdown()


if __name__ == "__main__":
    main()
