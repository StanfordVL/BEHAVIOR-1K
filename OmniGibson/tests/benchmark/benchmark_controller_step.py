"""
Benchmark: Batched vs Sequential controller step.

Sets up an environment with multiple robots, applies actions, then times the
controller-step portion (begin + step_controller_class + deploy) in isolation.

Usage:
    python tests/benchmark/benchmark_controller_step.py [--arm-controller IK|OSC] [--warmup 50] [--iters 200] [--num-robots 5]

    --num-robots N  will replicate fetch robots N times (minimum 5 uses the
                    mixed-robot set from test_controllers; above 5 adds extra fetch robots).
"""

import argparse
import time

import numpy as np
import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.controllers.controller_base import Controller, ControllerType


def _run_controller_step():
    """Execute the full controller step pipeline once."""
    Controller.begin_controller_step()
    Controller.step_controller_class()
    Controller.deploy_controller_step()


def _benchmark(n_warmup, n_iters, label):
    """Time controller step. Returns total_times array."""
    for _ in range(n_warmup):
        _run_controller_step()
    th.cuda.synchronize() if th.cuda.is_available() else None

    total_times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        _run_controller_step()
        th.cuda.synchronize() if th.cuda.is_available() else None
        total_times.append((time.perf_counter() - t0) * 1000)

    arr = np.array(total_times)
    print(f"\n--- {label} ({n_iters} iters) ---")
    print(f"  Total:  mean={arr.mean():.4f}  median={np.median(arr):.4f}  std={arr.std():.4f}  min={arr.min():.4f}  max={arr.max():.4f} ms")
    return arr


BASE_ROBOTS = [
    {"model": "franka", "name": "robot0", "obs_modalities": [], "position": [150, 150, 100],
     "orientation": [0, 0, 0, 1], "action_normalize": False, "fixed_base": True},
    {"model": "fetch", "name": "robot1", "obs_modalities": [], "position": [150, 150, 105],
     "orientation": [0, 0, 0, 1], "action_normalize": False, "fixed_base": False},
    {"model": "tiago", "name": "robot2", "obs_modalities": [], "position": [150, 150, 110],
     "orientation": [0, 0, 0, 1], "action_normalize": False},
    {"model": "a1", "name": "robot3", "obs_modalities": [], "position": [150, 150, 115],
     "orientation": [0, 0, 0, 1], "action_normalize": False, "fixed_base": True},
    {"model": "r1", "name": "robot4", "obs_modalities": [], "position": [150, 150, 120],
     "orientation": [0, 0, 0, 1], "action_normalize": False},
]


def _build_robot_configs(num_robots):
    """Build robot config list. First 5 are the mixed set, extras are fetch clones."""
    robots = list(BASE_ROBOTS[:min(num_robots, len(BASE_ROBOTS))])
    for i in range(len(BASE_ROBOTS), num_robots):
        robots.append({
            "model": "fetch",
            "name": f"robot{i}",
            "obs_modalities": [],
            "position": [150, 150, 100 + i * 5],
            "orientation": [0, 0, 0, 1],
            "action_normalize": False,
            "fixed_base": False,
        })
    return robots


def main():
    parser = argparse.ArgumentParser(description="Benchmark batched vs sequential controller step")
    parser.add_argument("--arm-controller", choices=["IK", "OSC"], default="IK",
                        help="Arm controller to benchmark (default: IK)")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=200, help="Timed iterations")
    parser.add_argument("--num-robots", type=int, default=5,
                        help="Number of robots (first 5 are mixed types, extras are fetch clones)")
    args = parser.parse_args()

    controller_name = {
        "IK": "InverseKinematicsController",
        "OSC": "OperationalSpaceController",
    }[args.arm_controller]

    robot_cfgs = _build_robot_configs(args.num_robots)
    cfg = {
        "scene": {"type": "Scene"},
        "objects": [],
        "robots": robot_cfgs,
    }

    print(f"Loading environment with {args.num_robots} robots, arm controller = {controller_name} ...")
    env = og.Environment(configs=cfg)

    for i, robot in enumerate(env.robots):
        robot.set_position_orientation(
            position=th.tensor([0.0, i * 5.0, 0.0]),
            orientation=T.euler2quat(th.tensor([0.0, 0.0, np.pi / 3])),
        )
        robot.reset()

    for _ in range(10):
        og.sim.step()

    for robot in env.robots:
        robot.keep_still()
        for cname in robot.controllers:
            Controller.reset(robot._controller_id(cname))

    env.scene.update_initial_file()
    env.scene.reset()

    controller_kwargs = {"mode": "pose_delta_ori"}
    for robot in env.robots:
        controller_config = {
            f"arm_{arm}": {"name": controller_name, **controller_kwargs}
            for arm in robot.arm_names
        }
        robot.reload_controllers(controller_config)

    env.scene.update_initial_file()
    env.scene.reset()

    # Apply random actions so controllers have non-trivial goals
    for robot in env.robots:
        action = th.rand(robot.action_dim) * 0.1 - 0.05
        robot_action = robot.prepare_action(action)
        idx = 0
        for cname in robot.controllers:
            cid = robot._controller_id(cname)
            cmd_dim = Controller.command_dim(cid)
            Controller.apply_action(controller_id=cid, action=robot_action[idx:idx + cmd_dim])
            idx += cmd_dim

    for _ in range(5):
        og.sim.step()

    # Count active controller instances per type
    type_counts = {}
    for cid, ctype in Controller._types.items():
        type_counts[ctype.name] = type_counts.get(ctype.name, 0) + 1
    print(f"\nActive controllers per type ({args.num_robots} robots):")
    for name, count in sorted(type_counts.items()):
        print(f"  {name}: {count}")

    # ---- Benchmark BATCHED (current code) ----
    batched_total = _benchmark(args.warmup, args.iters, f"BATCHED ({args.arm_controller})")

    # ---- Monkey-patch step_batch to SEQUENTIAL fallback ----
    original_step_batch = Controller.step_batch

    @classmethod
    def _sequential_step_batch(cls, controller_ids, controller_type):
        for cid in controller_ids:
            if cls._goals[cid] is None:
                cls._goals[cid] = cls.compute_no_op_goal(cid, cls._control_dicts[cid])
        results = []
        for cid in controller_ids:
            control = cls.compute_control(controller_id=cid, control_dict=cls._control_dicts[cid])
            control = cls.clip_control(cid, control)
            cls._controls[cid] = control
            results.append(control)
        return results

    Controller.step_batch = _sequential_step_batch

    seq_total = _benchmark(args.warmup, args.iters, f"SEQUENTIAL ({args.arm_controller})")

    # Restore
    Controller.step_batch = original_step_batch

    # ---- Summary ----
    print("\n========== SUMMARY ==========")
    print(f"Arm controller:   {controller_name}")
    print(f"Robots:           {len(env.robots)}")
    print(f"Iterations:       {args.iters}")
    print(f"Batched mean:     {batched_total.mean():.4f} ms")
    print(f"Sequential mean:  {seq_total.mean():.4f} ms")
    print(f"Speedup:          {seq_total.mean()/batched_total.mean():.2f}x")
    print("=============================\n")

    og.shutdown()


if __name__ == "__main__":
    main()
