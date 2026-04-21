"""Render a labeled, color-coded floor plan PNG for a BEHAVIOR-1K scene.

The image shows every room instance in a distinct color with its instance name
drawn at the room centroid, and overlays the traversability map so that walls
and obstacles (furniture, fixtures) are visible against the room colors.

Run as a script:

    python scripts/sampling/floor_plan_visualization.py \\
        --scene house_single_floor --output house_single_floor_floorplan.png
"""

import argparse
import os

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from omnigibson.utils.asset_utils import get_dataset_path, get_scene_path


def _load_room_categories():
    path = os.path.join(get_dataset_path("behavior-1k-assets"), "metadata", "room_categories.txt")
    with open(path, "r") as fp:
        return [line.rstrip() for line in fp.readlines()]


def _parse_segmentation(scene_dir, floor):
    """Parse instance/semantic segmentation the same way SegmentationMap does.

    Returns the raw instance map, semantic map, and an ins_id -> ins_name dict.
    """
    layout_dir = os.path.join(scene_dir, "layout")
    img_ins = cv2.imread(os.path.join(layout_dir, f"floor_insseg_{floor}.png"), cv2.IMREAD_GRAYSCALE)
    img_sem = cv2.imread(os.path.join(layout_dir, f"floor_semseg_{floor}.png"), cv2.IMREAD_GRAYSCALE)
    assert img_ins is not None and img_sem is not None, f"Missing segmentation maps in {layout_dir}"
    assert img_ins.shape == img_sem.shape, "Instance and semantic maps have different sizes"

    room_cats = _load_room_categories()

    # For each semantic id, enumerate its instance ids (in sorted order, matching SegmentationMap).
    sem_id_to_ins_ids = {}
    for ins_id in np.unique(img_ins):
        if ins_id == 0:
            continue
        ys, xs = np.where(img_ins == ins_id)
        sem_id = int(img_sem[ys[0], xs[0]])
        sem_id_to_ins_ids.setdefault(sem_id, []).append(int(ins_id))

    ins_id_to_name = {}
    for sem_id, ins_ids in sem_id_to_ins_ids.items():
        sem_name = room_cats[sem_id - 1]
        for i, ins_id in enumerate(sorted(ins_ids)):
            ins_id_to_name[ins_id] = f"{sem_name}_{i}"

    return img_ins, img_sem, ins_id_to_name


def _load_trav(scene_dir, floor):
    path = os.path.join(scene_dir, "layout", f"floor_trav_{floor}.png")
    trav = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert trav is not None, f"Missing traversability map: {path}"
    return trav


def _downsample_to(img, target_size, interpolation):
    """Resize so the largest dimension is target_size, preserving aspect."""
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    if scale >= 1.0:
        return img
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=interpolation)


def render_floor_plan(scene_dir, output_path, floor=0, target_size=1500, dpi=200, crop_margin_px=80):
    ins, _, ins_id_to_name = _parse_segmentation(scene_dir, floor)
    trav = _load_trav(scene_dir, floor)

    # Crop the full-scene maps to the tight bounding box around the rooms so
    # the rendered image isn't dominated by empty exterior pixels.
    ys, xs = np.where(ins > 0)
    if len(xs):
        y0 = max(int(ys.min()) - crop_margin_px, 0)
        y1 = min(int(ys.max()) + crop_margin_px + 1, ins.shape[0])
        x0 = max(int(xs.min()) - crop_margin_px, 0)
        x1 = min(int(xs.max()) + crop_margin_px + 1, ins.shape[1])
        ins = ins[y0:y1, x0:x1]
        trav = trav[y0:y1, x0:x1]

    # Keep the two maps pixel-aligned while downsampling for a manageable figure.
    ins = _downsample_to(ins, target_size, cv2.INTER_NEAREST)
    trav = _downsample_to(trav, target_size, cv2.INTER_NEAREST)

    # Image rows run top-to-bottom while world y runs bottom-to-top, so flip rows.
    ins = ins[::-1, :]
    trav = trav[::-1, :]

    # Prefer landscape output: if the cropped map is taller than wide, rotate 90 CCW.
    if ins.shape[0] > ins.shape[1]:
        ins = np.rot90(ins, k=1)
        trav = np.rot90(trav, k=1)

    ins_ids = sorted(ins_id_to_name.keys())
    cmap = plt.get_cmap("tab20", max(len(ins_ids), 1))
    ins_id_to_color = {ins_id: cmap(i)[:3] for i, ins_id in enumerate(ins_ids)}

    # Build an RGB canvas: white background, colored per-room fill, dark obstacles.
    canvas = np.ones((*ins.shape, 3), dtype=np.float32)
    for ins_id, color in ins_id_to_color.items():
        canvas[ins == ins_id] = color

    # Obstacle overlay: where the traversability map is black inside any room,
    # darken that pixel so furniture / fixtures stand out against the room color.
    obstacle_mask = (trav == 0) & (ins > 0)
    canvas[obstacle_mask] *= 0.35

    # Walls (black in trav, outside any room) render as solid black for contrast.
    wall_mask = (trav == 0) & (ins == 0)
    canvas[wall_mask] = 0.0

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(canvas, interpolation="nearest")

    # Label each room at its centroid. Use a bbox so labels stay readable on any color.
    for ins_id, name in ins_id_to_name.items():
        ys, xs = np.where(ins == ins_id)
        if len(xs) == 0:
            continue
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        ax.text(
            cx,
            cy,
            f"{name}\n(id={ins_id})",
            ha="center",
            va="center",
            fontsize=7,
            color="black",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="black", alpha=0.8, linewidth=0.5),
        )

    # Legend keyed by instance name, sorted for stable output.
    handles = [mpatches.Patch(color=ins_id_to_color[i], label=f"{ins_id_to_name[i]} (id={i})") for i in ins_ids]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)

    scene_name = os.path.basename(os.path.normpath(scene_dir))
    ax.set_title(f"{scene_name} — floor {floor} room segmentation with obstacles")
    ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", default="house_single_floor", help="Scene name under behavior-1k-assets/scenes")
    parser.add_argument(
        "--scene-dir", default=None, help="Override scene directory directly (takes precedence over --scene)"
    )
    parser.add_argument("--floor", type=int, default=0)
    parser.add_argument("--output", default=None, help="Output PNG path (default: <scene>_floorplan.png)")
    parser.add_argument("--target-size", type=int, default=1500, help="Max pixels per side before rendering")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    scene_dir = args.scene_dir or get_scene_path(args.scene)
    output = args.output or f"{args.scene}_floorplan.png"
    out = render_floor_plan(scene_dir, output, floor=args.floor, target_size=args.target_size, dpi=args.dpi)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
