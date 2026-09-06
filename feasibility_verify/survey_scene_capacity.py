"""How many R1s fit in each scene? CPU-only survey of the baked trav maps.

The traversability maps are plain PNGs in the dataset (``floor_trav_0.png``,
0.01 m per pixel), so this needs no Isaac, no GPU and no scene load.

For each scene it erodes the map by the robot's radius, takes the largest
connected traversable component, and greedily packs robot positions into it at
a given centre-to-centre separation. The greedy count is a lower bound on
capacity -- a real placement can do at least this well.

Why separation matters even though symbolic navigation teleports: the teleport
is still into a live physics scene. Two interpenetrating R1s get shoved apart by
PhysX, which is what produced the ``holding=agent_1`` corruption and the
500-tick settle loops. So bodies must genuinely not overlap -- but none of
cuRobo's or the trav map's extra path-planning margin is needed.

Run:
    python feasibility_verify/survey_scene_capacity.py --robots 9
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

SCENES_DIR = os.path.expanduser("~/behavior-workspace/BEHAVIOR-1K/datasets/behavior-1k-assets/scenes")
ROOM_CATEGORIES = os.path.expanduser(
    "~/behavior-workspace/BEHAVIOR-1K/datasets/behavior-1k-assets/metadata/room_categories.txt"
)
#: Rooms that are not where a household cooperation benchmark belongs. The
#: largest connected free region of house_single_floor is garden_0, and placing
#: agents there defeats the point of choosing a multi-room house.
OUTDOOR = ("garden", "lawn", "yard", "porch", "patio", "driveway", "deck", "balcony")
METERS_PER_PIXEL = 0.01  # TraversableMap.map_default_resolution

#: Measured on R1: norm(reset_joint_pos_aabb_extent[:2]) / 2. This is the
#: circumscribed radius, so it is correct for a robot at any yaw. An oriented
#: box test could pack tighter, at the cost of real code.
R1_RADIUS = 0.62


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robots", type=int, default=9, help="How many robots we need to place.")
    parser.add_argument("--radius", type=float, default=R1_RADIUS, help="Robot circumscribed radius (m).")
    parser.add_argument(
        "--separation",
        type=float,
        default=None,
        help="Centre-to-centre minimum (m). Default 2*radius, i.e. just-touching.",
    )
    parser.add_argument("--map", default="floor_trav_0.png", help="Which baked map to use.")
    parser.add_argument("--top", type=int, default=18, help="How many scenes to list.")
    parser.add_argument(
        "--by-room",
        action="store_true",
        help=(
            "Measure capacity of the biggest INDOOR ROOM instead of the biggest connected free "
            "region. This is the number that matters: the free region is often the garden, and "
            "indoor rooms shrink drastically once eroded by the robot radius."
        ),
    )
    return parser.parse_args()


def room_instance_areas(scene, eroded):
    """``[(room_name, mask), ...]`` for every room instance in @scene.

    Mirrors ``SegmentationMap._load_map``: instance ids come from
    floor_insseg_0.png, their category from the co-located pixel in
    floor_semseg_0.png, indexed into room_categories.txt (1-based).
    """
    layout = os.path.join(SCENES_DIR, scene, "layout")
    ins = cv2.imread(os.path.join(layout, "floor_insseg_0.png"), cv2.IMREAD_GRAYSCALE)
    sem = cv2.imread(os.path.join(layout, "floor_semseg_0.png"), cv2.IMREAD_GRAYSCALE)
    if ins is None or sem is None or ins.shape != sem.shape:
        return []
    if ins.shape != eroded.shape:
        ins = cv2.resize(ins, (eroded.shape[1], eroded.shape[0]), interpolation=cv2.INTER_NEAREST)
        sem = cv2.resize(sem, (eroded.shape[1], eroded.shape[0]), interpolation=cv2.INTER_NEAREST)
    with open(ROOM_CATEGORIES) as handle:
        categories = [line.rstrip() for line in handle]
    out = []
    for ins_id in np.unique(ins):
        if ins_id == 0:
            continue
        mask = ins == ins_id
        rows, cols = np.nonzero(mask)
        sem_id = int(sem[rows[0], cols[0]])
        if not (1 <= sem_id <= len(categories)):
            continue
        out.append((categories[sem_id - 1], mask))
    return out


def largest_component(mask):
    """(area_px, component_mask) of the biggest connected traversable blob."""
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
    best_area, best = 0, None
    for label in range(1, count):
        component = labels == label
        area = int(component.sum())
        if area > best_area:
            best_area, best = area, component
    return best_area, best


def greedy_pack(component, separation_px, limit):
    """Greedily place up to @limit points in @component, >= separation apart.

    Scans on a coarse lattice rather than every pixel: at 1 cm resolution a
    pixel-by-pixel scan of a whole house is pointless, and a lattice step of
    separation/2 cannot miss a feasible packing by more than a constant factor.
    """
    rows, cols = np.nonzero(component)
    if len(rows) == 0:
        return []
    step = max(1, int(separation_px / 2))
    placed = []
    for row in range(rows.min(), rows.max() + 1, step):
        for col in range(cols.min(), cols.max() + 1, step):
            if not component[row][col]:
                continue
            if all((row - r) ** 2 + (col - c) ** 2 >= separation_px**2 for r, c in placed):
                placed.append((row, col))
                if len(placed) >= limit:
                    return placed
    return placed


def main() -> None:
    args = parse_args()
    separation = args.separation if args.separation is not None else 2 * args.radius
    radius_px = int(round(args.radius / METERS_PER_PIXEL))
    separation_px = separation / METERS_PER_PIXEL
    # A disk, not cv2.erode's square kernel: the robot's footprint is round
    # (circumscribed), and a square kernel of side r is not even a radius-r
    # erosion.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1))

    print(f"robots={args.robots}  radius={args.radius} m  separation={separation:.2f} m  map={args.map}")
    print(f"erosion disk radius {radius_px} px ({args.radius} m), lattice step {int(separation_px / 2)} px\n")

    rows = []
    for scene in sorted(os.listdir(SCENES_DIR)):
        path = os.path.join(SCENES_DIR, scene, "layout", args.map)
        if not os.path.exists(path):
            continue
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        free = (image == 255).astype(np.uint8)
        raw_area = float(free.sum()) * METERS_PER_PIXEL**2
        eroded = cv2.erode(free, kernel)

        if args.by_room:
            best = (0.0, 0, None)
            for name, mask in room_instance_areas(scene, eroded):
                if any(tag in name.lower() for tag in OUTDOOR):
                    continue
                area_px, component = largest_component(eroded * mask)
                if component is None:
                    continue
                placed = greedy_pack(component, separation_px, args.robots)
                area = area_px * METERS_PER_PIXEL**2
                if len(placed) > best[1] or (len(placed) == best[1] and area > best[0]):
                    best = (area, len(placed), name)
            rows.append((f"{scene} [{best[2]}]" if best[2] else scene, raw_area, best[0], best[1]))
            continue

        area_px, component = largest_component(eroded)
        component_area = area_px * METERS_PER_PIXEL**2
        if component is None:
            rows.append((scene, raw_area, 0.0, 0))
            continue
        placed = greedy_pack(component, separation_px, args.robots)
        rows.append((scene, raw_area, component_area, len(placed)))

    rows.sort(key=lambda r: (-r[3], -r[2]))
    label = "best indoor room m2" if args.by_room else "largest free m2"
    print(f"{'scene':<44}{'trav m2':>10}{label:>21}{'robots placed':>15}")
    print("-" * 92)
    for scene, raw_area, component_area, placed in rows[: args.top]:
        flag = "  OK" if placed >= args.robots else ""
        print(f"{scene:<44}{raw_area:>10.1f}{component_area:>21.1f}{placed:>15}{flag}")

    fits = [r for r in rows if r[3] >= args.robots]
    print(f"\n{len(fits)} / {len(rows)} scenes fit {args.robots} robots in one connected free region.")
    for scene, _, component_area, placed in rows:
        if scene.startswith("Rs_int") or scene.startswith("house_single_floor"):
            print(f"{scene}: {component_area:.1f} m2, placed {placed} robots.")


if __name__ == "__main__":
    main()
