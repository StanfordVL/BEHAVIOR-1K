"""
Create side-by-side MP4 videos comparing original vs rendered frames for all scenes.

Uses ProcessPoolExecutor to process scenes in parallel. Requires render_vid2room_scenes
to have been run first (render.success must exist per scene).
"""

import argparse
import json
import pathlib

import cv2
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

DEFAULT_SCENE_LIST = "/cvgl2/u/cgokmen/BEHAVIOR-1K/slurm/interesting_scenes.json"


def get_scene_id(room_dir):
    """Generate scene ID from room path: vid_XXXXX_room_type_N"""
    room_dir = pathlib.Path(room_dir)
    room_name = room_dir.name
    video_id = room_dir.parent.parent.name
    assert video_id.startswith("vid_"), f"Video ID {video_id} does not start with 'vid_'"
    return f"{video_id}_{room_name}"


def iter_vid2room_scenes(scene_list):
    """Yield room dirs from scene list, excluding bathrooms and incomplete rooms."""
    with open(scene_list, "r") as f:
        room_dirs = json.load(f)
    room_dirs = [pathlib.Path(k) for k in room_dirs]
    room_dirs = [x for x in room_dirs if "bathroom" not in str(x)]
    room_dirs = [x for x in room_dirs if (x / "floorplan.success").exists()]
    return room_dirs


def create_comparison_video(args):
    """
    Worker: create side-by-side MP4 for one scene.
    args: (room_dir, dataset_root, output_dir, fps)
    """
    room_dir, dataset_root, output_dir, fps = args
    room_dir = pathlib.Path(room_dir)
    dataset_root = pathlib.Path(dataset_root)
    output_dir = pathlib.Path(output_dir)

    scene_id = get_scene_id(room_dir)
    render_dir = dataset_root / "vid2room" / "renders" / scene_id

    metadata_path = render_dir / "metadata.json"
    if not metadata_path.exists():
        return scene_id, False, f"No metadata at {metadata_path}"

    metadata = json.loads(metadata_path.read_text())
    filenames = metadata["filenames"]
    num_frames = metadata["num_frames"]
    render_w = metadata["render_width"]
    render_h = metadata["render_height"]

    rgb_dir = render_dir / "rgb"
    images_dir = room_dir / "images"
    if not rgb_dir.exists():
        return scene_id, False, f"No rgb dir at {rgb_dir}"
    if not images_dir.exists():
        return scene_id, False, f"No images dir at {images_dir}"

    output_mp4 = output_dir / f"{scene_id}_original_vs_rendered.mp4"
    target_h = render_h
    target_w_per_side = render_w
    out_w = target_w_per_side * 2
    out_h = target_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_mp4), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        return scene_id, False, f"Failed to open VideoWriter for {output_mp4}"

    try:
        for i in range(num_frames):
            orig_path = images_dir / filenames[i]
            if not orig_path.exists():
                writer.release()
                output_mp4.unlink(missing_ok=True)
                return scene_id, False, f"Original frame not found: {orig_path}"

            orig = np.array(Image.open(orig_path).convert("RGB"))
            if orig.shape[:2] != (target_h, target_w_per_side):
                orig = cv2.resize(orig, (target_w_per_side, target_h), interpolation=cv2.INTER_LINEAR)

            rend_path = rgb_dir / f"{i:05d}.png"
            if not rend_path.exists():
                writer.release()
                output_mp4.unlink(missing_ok=True)
                return scene_id, False, f"Rendered frame not found: {rend_path}"

            rend = np.array(Image.open(rend_path).convert("RGB"))
            if rend.shape[:2] != (target_h, target_w_per_side):
                rend = cv2.resize(rend, (target_w_per_side, target_h), interpolation=cv2.INTER_LINEAR)

            frame = np.hstack([orig, rend])
            cv2.putText(frame, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, "Rendered", (target_w_per_side + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        writer.release()
        return scene_id, True, str(output_mp4)
    except Exception as e:
        writer.release()
        output_mp4.unlink(missing_ok=True)
        return scene_id, False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Create original vs rendered comparison videos for all scenes")
    parser.add_argument("--scene-list", type=str, default=DEFAULT_SCENE_LIST)
    parser.add_argument("--dataset-root", type=str, default="/cvgl2/u/cgokmen/BEHAVIOR-1K/datasets")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Where to save MP4s. Default: dataset_root/vid2room/comparison_videos")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    dataset_root = pathlib.Path(args.dataset_root)
    output_dir = pathlib.Path(args.output_dir) if args.output_dir else dataset_root / "vid2room" / "comparison_videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    room_dirs = iter_vid2room_scenes(args.scene_list)
    renders_root = dataset_root / "vid2room" / "renders"

    # Only process scenes that have been rendered
    tasks = []
    for room_dir in room_dirs:
        scene_id = get_scene_id(room_dir)
        success_file = renders_root / scene_id / "render.success"
        if not success_file.exists():
            continue
        tasks.append((room_dir, dataset_root, output_dir, args.fps))

    if not tasks:
        print("No scenes with render.success found. Run render_vid2room_scenes first.")
        return

    print(f"Processing {len(tasks)} scenes")
    ok = 0
    fail = 0
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(create_comparison_video, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scenes"):
            scene_id, success, msg = future.result()
            if success:
                ok += 1
            else:
                fail += 1
                tqdm.write(f"FAIL {scene_id}: {msg}")

    print(f"Done: {ok} ok, {fail} failed")

if __name__ == "__main__":
    main()