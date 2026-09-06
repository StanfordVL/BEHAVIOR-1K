"""How often does the symbolic sampler teleport a robot somewhere it cannot fit?

`_sample_pose_near_object` picks a base pose uniformly in a 0-1.5 m annulus
around the target and teleports there with no validity check at all -- the
cuRobo `_validate_poses` stage was dropped when the sampler was made
cuRobo-free. This measures the consequence, per scene, straight from the baked
trav map PNGs (0.01 m/px). No Isaac, no GPU.

Two numbers matter:

* **bad-pose rate** -- fraction of sampled candidates that land where an R1
  does not fit. These are the "teleported into a wall and fell over" cases.
* **dead-target rate** -- fraction of targets for which *no* pose in the whole
  annulus is valid. For these, adding a validity filter does not help: the
  sampler exhausts its attempts and NAVIGATE_TO raises PLANNING_ERROR. This is
  the number that decides whether a scene is usable at all.

Run:
    python feasibility_verify/measure_teleport_risk.py --scenes Rs_int house_single_floor
"""

from __future__ import annotations

import argparse
import math
import os

import cv2
import numpy as np

SCENES_DIR = os.path.expanduser("~/behavior-workspace/BEHAVIOR-1K/datasets/behavior-1k-assets/scenes")
METERS_PER_PIXEL = 0.01
R1_RADIUS = 0.62


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", nargs="+", default=["Rs_int", "house_single_floor", "Beechwood_0_int"])
    parser.add_argument("--radius", type=float, default=R1_RADIUS)
    parser.add_argument("--targets", type=int, default=300, help="Random target positions per scene.")
    parser.add_argument("--candidates", type=int, default=64, help="Candidate poses sampled per target.")
    parser.add_argument("--reach-lo", type=float, default=0.0, help="Annulus inner radius (current: 0.0).")
    parser.add_argument("--reach-hi", type=float, default=1.5, help="Annulus outer radius.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load(scene, radius):
    path = os.path.join(SCENES_DIR, scene, "layout", "floor_trav_0.png")
    if not os.path.exists(path):
        return None, None
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    free = (image == 255).astype(np.uint8)
    radius_px = int(round(radius / METERS_PER_PIXEL))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1))
    return free, cv2.erode(free, kernel)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    print(f"radius={args.radius} m  annulus=[{args.reach_lo}, {args.reach_hi}] m  "
          f"{args.targets} targets x {args.candidates} candidates\n")
    print(f"{'scene':<24}{'bad-pose rate':>15}{'dead targets':>15}{'usable':>9}")
    print("-" * 64)

    for scene in args.scenes:
        free, fits = load(scene, args.radius)
        if free is None:
            print(f"{scene:<24}{'(no trav map)':>15}")
            continue
        rows, cols = np.nonzero(free)
        if len(rows) == 0:
            print(f"{scene:<24}{'(empty map)':>15}")
            continue
        picks = rng.integers(0, len(rows), size=args.targets)
        bad = total = dead = 0
        for index in picks:
            row0, col0 = int(rows[index]), int(cols[index])
            valid_here = 0
            for _ in range(args.candidates):
                distance = rng.uniform(args.reach_lo, args.reach_hi) / METERS_PER_PIXEL
                yaw = rng.uniform(-math.pi, math.pi)
                row = int(round(row0 + distance * math.sin(yaw)))
                col = int(round(col0 + distance * math.cos(yaw)))
                total += 1
                inside = 0 <= row < fits.shape[0] and 0 <= col < fits.shape[1]
                if inside and fits[row][col]:
                    valid_here += 1
                else:
                    bad += 1
            if valid_here == 0:
                dead += 1
        bad_rate = bad / total if total else 1.0
        dead_rate = dead / args.targets
        verdict = "yes" if dead_rate < 0.05 else ("marginal" if dead_rate < 0.25 else "NO")
        print(f"{scene:<24}{bad_rate:>14.1%}{dead_rate:>15.1%}{verdict:>9}")

    print("\nbad-pose rate = candidates landing where an R1 does not fit (a validity")
    print("                filter turns these into retries -- survivable)")
    print("dead targets  = targets with NO valid pose anywhere in the annulus (the")
    print("                filter cannot help; NAVIGATE_TO fails outright)")


if __name__ == "__main__":
    main()
