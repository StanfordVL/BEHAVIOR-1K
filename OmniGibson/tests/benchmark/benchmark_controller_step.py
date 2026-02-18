"""
Benchmark: Batched vs Sequential controller step.

Sets up an environment with multiple robots, applies actions, then times the
controller-step portion (begin + step_controller_class + deploy) in isolation.
Also reports per-controller-class breakdown.

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
from omnigibson.controllers import REGISTERED_CONTROLLERS
from omnigibson.controllers.controller_base import BaseController


def _run_controller_step():
    """Execute the full controller step pipeline once."""
    BaseController.begin_controller_step()
    for controller_cls in REGISTERED_CONTROLLERS.values():
        controller_cls.step_controller_class()
    BaseController.deploy_controller_step()


def _run_controller_step_per_class():
    """Like _run_controller_step but returns per-class wall times (ms)."""
    BaseController.begin_controller_step()
    class_times = {}
    for name, controller_cls in REGISTERED_CONTROLLERS.items():
        if not controller_cls._configs:
            continue
        t0 = time.perf_counter()
        controller_cls.step_controller_class()
        th.cuda.synchronize() if th.cuda.is_available() else None
        class_times[name] = (time.perf_counter() - t0) * 1000
    BaseController.deploy_controller_step()
    return class_times


def _benchmark(n_warmup, n_iters, label, per_class=False):
    """Time controller step. Returns (total_times, per_class_times_dict)."""
    for _ in range(n_warmup):
        _run_controller_step()
    th.cuda.synchronize() if th.cuda.is_available() else None

    total_times = []
    class_accum = {}

    for _ in range(n_iters):
        if per_class:
            t0 = time.perf_counter()
            ct = _run_controller_step_per_class()
            th.cuda.synchronize() if th.cuda.is_available() else None
            total_times.append((time.perf_counter() - t0) * 1000)
            for k, v in ct.items():
                class_accum.setdefault(k, []).append(v)
        else:
            t0 = time.perf_counter()
            _run_controller_step()
            th.cuda.synchronize() if th.cuda.is_available() else None
            total_times.append((time.perf_counter() - t0) * 1000)

    arr = np.array(total_times)
    print(f"\n--- {label} ({n_iters} iters) ---")
    print(f"  Total:  mean={arr.mean():.4f}  median={np.median(arr):.4f}  std={arr.std():.4f}  min={arr.min():.4f}  max={arr.max():.4f} ms")

    if per_class:
        for name in sorted(class_accum.keys()):
            ca = np.array(class_accum[name])
            n = len(REGISTERED_CONTROLLERS[name]._configs)
            print(f"    {name:40s} ({n:2d} instances):  mean={ca.mean():.4f}  median={np.median(ca):.4f} ms")

    return arr, class_accum


def _make_sequential_step_batch(cls):
    """Return a sequential fallback step_batch matching BaseController's default."""

    @classmethod
    def _sequential_step_batch(cls_inner, controller_ids):
        for cid in controller_ids:
            if cls_inner._goals[cid] is None:
                cls_inner._goals[cid] = cls_inner.compute_no_op_goal(cid, cls_inner._control_dicts[cid])
        results = []
        for cid in controller_ids:
            control = cls_inner.compute_control(controller_id=cid, control_dict=cls_inner._control_dicts[cid])
            control = cls_inner.clip_control(cid, control)
            cls_inner._controls[cid] = control
            results.append(control)
        return results

    return _sequential_step_batch


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
            controller_cls = robot._get_controller_class(cname)
            controller_cls.reset(robot._controller_id(cname))

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
            cls = robot._get_controller_class(cname)
            cid = robot._controller_id(cname)
            cmd_dim = cls.command_dim(cid)
            cls.apply_action(controller_id=cid, action=robot_action[idx:idx + cmd_dim])
            idx += cmd_dim

    # Take a few env steps to populate caches
    for _ in range(5):
        og.sim.step()

    # Count active controller instances per class
    print(f"\nActive controllers per class ({args.num_robots} robots):")
    for name, cls in REGISTERED_CONTROLLERS.items():
        if cls._configs:
            print(f"  {name}: {len(cls._configs)}")

    # ---- Benchmark BATCHED (current code) ----
    batched_total, batched_per_class = _benchmark(
        args.warmup, args.iters, f"BATCHED ({args.arm_controller})", per_class=True)

    # ---- Monkey-patch to SEQUENTIAL fallback ----
    saved = {}
    for name, cls in REGISTERED_CONTROLLERS.items():
        if hasattr(cls, "step_batch") and "step_batch" in cls.__dict__:
            saved[name] = cls.__dict__["step_batch"]
            cls.step_batch = _make_sequential_step_batch(cls)

    seq_total, seq_per_class = _benchmark(
        args.warmup, args.iters, f"SEQUENTIAL ({args.arm_controller})", per_class=True)

    # Restore batched methods
    for name, method in saved.items():
        REGISTERED_CONTROLLERS[name].step_batch = method

    # ---- Summary ----
    print("\n========== SUMMARY ==========")
    print(f"Arm controller:   {controller_name}")
    print(f"Robots:           {len(env.robots)}")
    print(f"Iterations:       {args.iters}")
    print(f"{'':40s}  {'Batched':>10s}  {'Sequential':>10s}  {'Speedup':>8s}")
    print(f"{'TOTAL':40s}  {batched_total.mean():10.4f}  {seq_total.mean():10.4f}  {seq_total.mean()/batched_total.mean():8.2f}x")

    all_classes = sorted(set(list(batched_per_class.keys()) + list(seq_per_class.keys())))
    for name in all_classes:
        b = np.array(batched_per_class.get(name, [0])).mean()
        s = np.array(seq_per_class.get(name, [0])).mean()
        n = len(REGISTERED_CONTROLLERS[name]._configs)
        speedup = s / b if b > 0 else float("inf")
        print(f"  {name:38s}  {b:10.4f}  {s:10.4f}  {speedup:8.2f}x  ({n} instances)")

    print("=============================\n")

    og.shutdown()


if __name__ == "__main__":
    main()
