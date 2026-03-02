#!/usr/bin/env python3
"""
End-to-end pipeline for building Franka + SharpaWave dexterous hand robots.

This script orchestrates all steps needed to go from raw URDFs to a fully
renderable OmniGibson robot.  Each step can be run independently or as a
full pipeline.

Steps:
  1. MERGE   -- Merge Franka arm URDF with SharpaWave hand URDF
  2. CONVERT -- Convert merged URDF to USD via BEHAVIOR's import_custom_robot
                (requires Isaac Sim / conda environment)
  3. POSTPROCESS -- Fix shared-mesh deduplication + Y-up→Z-up rotation
  4. TEST    -- Load and visualize in OmniGibson (requires Isaac Sim)

Usage:
    # Full pipeline for right hand:
    python run_sharpa_pipeline.py --hand right

    # Full pipeline for both hands:
    python run_sharpa_pipeline.py --hand both

    # Only specific steps:
    python run_sharpa_pipeline.py --hand right --steps merge
    python run_sharpa_pipeline.py --hand right --steps convert
    python run_sharpa_pipeline.py --hand right --steps postprocess
    python run_sharpa_pipeline.py --hand right --steps merge,postprocess

    # Steps 2 (convert) and 4 (test) require Isaac Sim and must be run
    # inside the conda environment:
    conda run -n behavior python run_sharpa_pipeline.py --hand right --steps convert
    conda run -n behavior python run_sharpa_pipeline.py --hand right --steps test

Notes:
    - Steps 1 (merge) and 3 (postprocess) run with plain Python (no Isaac Sim).
    - Steps 2 (convert) and 4 (test) require Isaac Sim via conda.
    - Step 2 sets overwrite=true to allow re-running the conversion.
    - The pipeline is idempotent -- safe to re-run any step.
"""

import argparse
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Config file paths for URDF→USD conversion
CONVERSION_CONFIGS = {
    "right": os.path.join(
        SCRIPT_DIR,
        "datasets/omnigibson-robot-assets/models/franka/franka_mounted_sharpa_right/franka_mounted_sharpa_right.yaml",
    ),
    "left": os.path.join(
        SCRIPT_DIR,
        "datasets/omnigibson-robot-assets/models/franka/franka_mounted_sharpa_left/franka_mounted_sharpa_left.yaml",
    ),
}

ALL_STEPS = ["merge", "convert", "postprocess", "test"]
ISAAC_STEPS = {"convert", "postprocess", "test"}  # These require Isaac Sim


def run_cmd(cmd, desc, cwd=None):
    """Run a command and stream output.  Returns True on success."""
    print(f"\n{'─' * 60}")
    print(f"  {desc}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'─' * 60}\n")

    result = subprocess.run(cmd, cwd=cwd or SCRIPT_DIR)

    if result.returncode != 0:
        print(f"\n  *** FAILED (exit code {result.returncode}) ***\n")
        return False
    return True


def step_merge(hand_side):
    """Step 1: Merge Franka arm URDF + SharpaWave hand URDF."""
    return run_cmd(
        [sys.executable, "merge_franka_sharpa.py", "--hand", hand_side],
        f"Step 1: Merging Franka + SharpaWave {hand_side} hand URDFs",
    )


def step_convert(hand_side):
    """Step 2: Convert merged URDF to USD via BEHAVIOR's import_custom_robot."""
    config_path = CONVERSION_CONFIGS[hand_side]
    if not os.path.exists(config_path):
        print(f"  ERROR: Conversion config not found: {config_path}")
        return False

    # We use the OmniGibson import_custom_robot entry point with --config
    return run_cmd(
        [
            sys.executable,
            "-m", "omnigibson.examples.robots.import_custom_robot",
            "--config", config_path,
        ],
        f"Step 2: Converting {hand_side} hand URDF → USD (requires Isaac Sim)",
        cwd=os.path.join(SCRIPT_DIR, "OmniGibson"),
    )


def step_postprocess(hand_side):
    """Step 3: Post-process USD (fix shared meshes + rotation)."""
    return run_cmd(
        [sys.executable, "postprocess_sharpa_usd.py", "--hand", hand_side],
        f"Step 3: Post-processing {hand_side} hand USD",
    )


def step_test(hand_side):
    """Step 4: Test the robot in OmniGibson."""
    return run_cmd(
        [sys.executable, "test_sharpa_robot.py", "--hand", hand_side],
        f"Step 4: Testing {hand_side} hand in OmniGibson (requires Isaac Sim)",
    )


STEP_FUNCS = {
    "merge": step_merge,
    "convert": step_convert,
    "postprocess": step_postprocess,
    "test": step_test,
}


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline for Franka + SharpaWave hand",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_sharpa_pipeline.py --hand right                      # Full pipeline
  python run_sharpa_pipeline.py --hand both                       # Both hands
  python run_sharpa_pipeline.py --hand right --steps merge        # Only merge
  python run_sharpa_pipeline.py --hand right --steps merge,postprocess
  conda run -n behavior python run_sharpa_pipeline.py --hand right --steps convert
  conda run -n behavior python run_sharpa_pipeline.py --hand right --steps test
        """,
    )
    parser.add_argument(
        "--hand",
        choices=["right", "left", "both"],
        default="right",
        help="Which hand to build (default: right)",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated list of steps to run: merge,convert,postprocess,test "
             "(default: all steps)",
    )
    parser.add_argument(
        "--skip-isaac",
        action="store_true",
        help="Skip steps that require Isaac Sim (convert, test)",
    )

    args = parser.parse_args()

    # Parse steps
    if args.steps:
        steps = [s.strip().lower() for s in args.steps.split(",")]
        for s in steps:
            if s not in ALL_STEPS:
                print(f"ERROR: Unknown step '{s}'. Valid steps: {ALL_STEPS}")
                sys.exit(1)
    else:
        steps = list(ALL_STEPS)

    if args.skip_isaac:
        steps = [s for s in steps if s not in ISAAC_STEPS]

    hands = ["right", "left"] if args.hand == "both" else [args.hand]

    print("=" * 60)
    print("Franka + SharpaWave Pipeline")
    print(f"  Hands: {', '.join(hands)}")
    print(f"  Steps: {', '.join(steps)}")
    print("=" * 60)

    # Check for Isaac Sim steps
    isaac_steps_requested = [s for s in steps if s in ISAAC_STEPS]
    if isaac_steps_requested:
        print(f"\n  NOTE: Steps {isaac_steps_requested} require Isaac Sim.")
        print(f"  Make sure you're running inside: conda run -n behavior ...\n")

    all_ok = True
    for hand_side in hands:
        print(f"\n{'=' * 60}")
        print(f"  Processing: {hand_side} hand")
        print(f"{'=' * 60}")

        for step_name in steps:
            func = STEP_FUNCS[step_name]
            ok = func(hand_side)
            if not ok:
                print(f"\n  Pipeline FAILED at step '{step_name}' for {hand_side} hand.")
                all_ok = False
                break

    print(f"\n{'=' * 60}")
    if all_ok:
        print("Pipeline completed successfully!")
    else:
        print("Pipeline failed (see errors above).")
        sys.exit(1)


if __name__ == "__main__":
    main()
