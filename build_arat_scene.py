#!/usr/bin/env python3
"""
Interactive scene builder for ARAT (Action Research Arm Test) tasks.

Workflow:
  1. Edit the OBJECTS list below to add/remove/reposition objects.
  2. Run:  conda run -n behavior python build_arat_scene.py [--hand right|left]
  3. Use the viewer to inspect. Ctrl+C to exit.
  4. Dimensions are printed to the console on startup.
  5. Use --save scene.json to dump the scene state for later reloading.
  6. Use --list-category <name> to browse available models for a category.

Tips:
  - Position is [x, y, z] in meters (Z is up).
  - Orientation is [qx, qy, qz, qw] quaternion (default: upright).
  - Scale is [sx, sy, sz] multiplier on the native asset size.
  - For PrimitiveObjects: size/radius/height set the base dimensions.
  - To find object categories:
      ls datasets/behavior-1k-assets/objects/ | grep <keyword>
  - To find model IDs within a category:
      ls datasets/behavior-1k-assets/objects/<category>/
"""

import argparse
import json
import sys
import torch as th
import omnigibson as og
from omnigibson.macros import gm

gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = False
gm.USE_PBR_MATERIALS = True


# ============================================================================
# EDIT THIS: Objects to place in the scene
# ============================================================================
# Each entry is a dict. Required keys depend on "type":
#
#   DatasetObject:
#     type, name, category, model, position
#     optional: orientation, scale, fixed_base
#
#   PrimitiveObject:
#     type, name, primitive_type, position
#     optional: size, radius, height, rgba, scale, fixed_base
#
# Supported primitive_type: Sphere, Cube, Cylinder, Cone, Torus, Disk

OBJECTS = [
    # --- Table (fixed) ---
    {
        "type": "DatasetObject",
        "name": "table",
        "category": "breakfast_table",
        "model": "bhszwe",
        "position": [0.6, 0.0, 0.0],
        "orientation": [0, 0, 0, 1],
        "fixed_base": True,
    },

    # --- Example graspable objects (adjust positions to sit on table) ---
    # Small cube (primitive) - like an ARAT block
    # {
    #     "type": "PrimitiveObject",
    #     "name": "block_small",
    #     "primitive_type": "Cube",
    #     "size": 0.025,
    #     "rgba": [1.0, 0.2, 0.2, 1.0],
    #     "position": [0.5, -0.10, 0.8],
    #     "fixed_base": False,
    # },
    # # Medium cube
    # {
    #     "type": "PrimitiveObject",
    #     "name": "block_medium",
    #     "primitive_type": "Cube",
    #     "size": 0.05,
    #     "rgba": [0.2, 0.6, 1.0, 1.0],
    #     "position": [0.5, 0.0, 0.8],
    #     "fixed_base": False,
    # },
    # # Large cube
    # {
    #     "type": "PrimitiveObject",
    #     "name": "block_large",
    #     "primitive_type": "Cube",
    #     "size": 0.075,
    #     "rgba": [0.2, 0.8, 0.2, 1.0],
    #     "position": [0.5, 0.10, 0.8],
    #     "fixed_base": False,
    # },

    # # Sphere (like a ball grasp test)
    # {
    #     "type": "PrimitiveObject",
    #     "name": "ball",
    #     "primitive_type": "Sphere",
    #     "radius": 0.0375,
    #     "rgba": [1.0, 0.8, 0.0, 1.0],
    #     "position": [0.6, -0.10, 0.8],
    #     "fixed_base": False,
    # },

    # # Cylinder (like a peg / dowel)
    # {
    #     "type": "PrimitiveObject",
    #     "name": "cylinder_peg",
    #     "primitive_type": "Cylinder",
    #     "radius": 0.01,
    #     "height": 0.10,
    #     "rgba": [0.6, 0.4, 0.2, 1.0],
    #     "position": [0.6, 0.10, 0.8],
    #     "fixed_base": False,
    # },

    # # Washer (torus, flattened)
    # {
    #     "type": "PrimitiveObject",
    #     "name": "washer",
    #     "primitive_type": "Torus",
    #     "size": 0.02,
    #     "rgba": [0.7, 0.7, 0.7, 1.0],
    #     "position": [0.7, 0.0, 0.8],
    #     "scale": [1.0, 1.0, 0.3],
    #     "fixed_base": False,
    # },
]

# ============================================================================
# ROBOT CONFIG
# ============================================================================
HAND_TO_ROBOT = {
    "right": "FrankaMountedSharpaRight",
    "left": "FrankaMountedSharpaLeft",
}


def build_object_cfgs(object_list):
    """Convert the OBJECTS list into OmniGibson config dicts."""
    cfgs = []
    for spec in object_list:
        obj_type = spec["type"]
        cfg = {"type": obj_type, "name": spec["name"]}

        if obj_type == "DatasetObject":
            cfg["category"] = spec["category"]
            cfg["model"] = spec["model"]
        elif obj_type == "PrimitiveObject":
            cfg["primitive_type"] = spec["primitive_type"]
            if "rgba" in spec:
                cfg["rgba"] = spec["rgba"]
            for key in ("size", "radius", "height"):
                if key in spec:
                    cfg[key] = spec[key]

        cfg["position"] = spec.get("position", [0, 0, 0.5])
        cfg["orientation"] = spec.get("orientation", [0, 0, 0, 1])
        cfg["fixed_base"] = spec.get("fixed_base", False)
        if "scale" in spec:
            cfg["scale"] = spec["scale"]

        cfgs.append(cfg)
    return cfgs


def print_dimensions(env, object_list):
    """Print AABB dimensions for every object."""
    print("\n" + "=" * 80)
    print(f"{'OBJECT DIMENSIONS':^80}")
    print("=" * 80)
    print(f"  {'Name':<22} {'Type':<16} {'X (cm)':>8} {'Y (cm)':>8} {'Z (cm)':>8}   Position (m)")
    print("  " + "-" * 76)

    for spec in object_list:
        obj = env.scene.object_registry("name", spec["name"])
        if obj is None:
            print(f"  {spec['name']:<22} (not found)")
            continue
        extent = obj.aabb_extent
        pos = obj.get_position_orientation()[0]
        otype = spec.get("primitive_type", spec.get("category", "?"))
        print(
            f"  {spec['name']:<22} {otype:<16} "
            f"{extent[0]*100:8.2f} {extent[1]*100:8.2f} {extent[2]*100:8.2f}   "
            f"[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"
        )

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="ARAT scene builder -- edit OBJECTS list, run, inspect, repeat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--hand", choices=["right", "left"], default="right",
        help="Which Sharpa hand to use (default: right)",
    )
    parser.add_argument(
        "--no-robot", action="store_true",
        help="Load scene without robot (objects only)",
    )
    parser.add_argument(
        "--save", type=str, default=None, metavar="PATH",
        help="Save scene state to JSON after loading (e.g. --save my_scene.json)",
    )
    parser.add_argument(
        "--list-category", type=str, default=None, metavar="CAT",
        help="Print available models for a category and exit",
    )
    args = parser.parse_args()

    if args.list_category:
        import os
        cat_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "datasets", "behavior-1k-assets", "objects", args.list_category,
        )
        if not os.path.isdir(cat_dir):
            print(f"Category '{args.list_category}' not found at: {cat_dir}")
            sys.exit(1)
        models = sorted(os.listdir(cat_dir))
        print(f"Models for '{args.list_category}' ({len(models)}):")
        for m in models:
            print(f"  {m}")
        sys.exit(0)

    # Build config
    obj_cfgs = build_object_cfgs(OBJECTS)

    # Add lighting
    obj_cfgs.extend([
        {"type": "LightObject", "light_type": "Sphere", "name": "light0",
         "radius": 0.01, "intensity": 1e5, "position": [-2, -2, 2]},
        {"type": "LightObject", "light_type": "Sphere", "name": "light1",
         "radius": 0.01, "intensity": 1e5, "position": [2, 2, 2]},
    ])

    cfg = {
        "scene": {"type": "Scene"},
        "objects": obj_cfgs,
    }

    if not args.no_robot:
        robot_class = HAND_TO_ROBOT[args.hand]
        cfg["robots"] = [{
            "type": robot_class,
            "name": f"franka_sharpa_{args.hand}",
            "obs_modalities": [],
            "fixed_base": True,
            "self_collisions": False,
            "load_config": {"xform_props_pre_loaded": False},
        }]
        print(f"\n*** Robot: {robot_class} ***")

    print(f"*** Objects: {len(OBJECTS)} ***\n")

    env = og.Environment(configs=cfg)

    # Step a few times so physics settles
    for _ in range(10):
        og.sim.step()

    # Print dimensions
    print_dimensions(env, OBJECTS)

    # Save if requested
    if args.save:
        env.scene.save(args.save)
        print(f"\nScene saved to: {args.save}")

    # Camera setup
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([1.5, 1.2, 1.5]),
        orientation=th.tensor([0.2706, 0.2706, 0.6533, 0.6533]),
    )
    og.sim.enable_viewer_camera_teleoperation()

    print("\n" + "=" * 60)
    print("  Scene loaded! Camera controls:")
    print("    Alt + Left-click  = Orbit")
    print("    Alt + Right-click = Zoom")
    print("    Alt + Middle-click = Pan")
    print("    Ctrl+C to exit")
    print("=" * 60 + "\n")

    # Render warmup
    for _ in range(100):
        og.sim.render()

    while True:
        og.sim.step()


if __name__ == "__main__":
    main()
