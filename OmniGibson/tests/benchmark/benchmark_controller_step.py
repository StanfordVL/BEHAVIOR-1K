"""
Benchmark: Batched vs Sequential controller step.

Sets up an environment with multiple robots, applies actions, then times the
controller-step portion (begin + step_controller_class + deploy) in isolation.
Reports both total time and per-controller-type breakdowns.

Usage:
    python tests/benchmark/benchmark_controller_step.py [--arm-controller IK|OSC] [--warmup 50] [--iters 200] [--num-robots 5]

    --num-robots N  will replicate fetch robots N times (minimum 5 uses the
                    mixed-robot set from test_controllers; above 5 adds extra fetch robots).
"""

import argparse
import time
from collections import defaultdict

import numpy as np
import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.controllers import Controller, ControlType
from omnigibson.utils.backend_utils import _compute_backend as cb


def _run_controller_step():
    """Execute the full controller step pipeline once."""
    Controller.begin_controller_step()
    Controller.step_controller_class()
    Controller.deploy_controller_step()


def _run_controller_step_timed():
    """Execute the full controller step pipeline once, returning per-type timings (ms)."""
    Controller.begin_controller_step()

    # --- Replicate the gather phase from step_controller_class ---
    Controller._control_dicts.clear()
    active_ids = []
    for controller_id in list(Controller._configs.keys()):
        Controller._goals.setdefault(controller_id, None)
        Controller._controls.setdefault(controller_id, None)
        robot = Controller._robots.get(controller_id, None)
        if robot is None or not robot.control_enabled:
            continue
        if robot._articulation_view_direct is None or not robot._articulation_view_direct.initialized:
            continue
        robot_name = robot.name
        cache = Controller._ROBOT_CONTROL_STEP_CACHE.get(robot_name, None)
        if cache is None:
            control_dict = robot.get_control_dict()
            u_vec = cb.zeros(robot.n_dof)
            u_type_vec = cb.array([ControlType.EFFORT] * robot.n_dof)
            cache = {
                "robot": robot,
                "control_dict": control_dict,
                "u_vec": u_vec,
                "u_type_vec": u_type_vec,
            }
            Controller._ROBOT_CONTROL_STEP_CACHE[robot_name] = cache
        Controller._control_dicts[controller_id] = cache["control_dict"]
        active_ids.append(controller_id)

    type_timings = {}
    if active_ids:
        type_groups = {}
        for cid in active_ids:
            ctype = Controller._types[cid]
            type_groups.setdefault(ctype, []).append(cid)

        all_results = {}
        for ctype, cids in type_groups.items():
            th.cuda.synchronize() if th.cuda.is_available() else None
            t0 = time.perf_counter()
            controls = Controller.step_batch(cids, ctype)
            th.cuda.synchronize() if th.cuda.is_available() else None
            type_timings[ctype] = (time.perf_counter() - t0) * 1000
            for cid, control in zip(cids, controls):
                all_results[cid] = control

        for controller_id in active_ids:
            control = all_results[controller_id]
            robot_name = Controller._robots[controller_id].name
            cache = Controller._ROBOT_CONTROL_STEP_CACHE[robot_name]
            idx = Controller.dof_idx(controller_id)
            cache["u_vec"][idx] = control
            cache["u_type_vec"][idx] = Controller.control_type(controller_id)

    Controller.deploy_controller_step()
    return type_timings


def _benchmark(n_warmup, n_iters, label, per_type=False):
    """Time controller step. Returns (total_times, per_type_times) arrays."""
    for _ in range(n_warmup):
        _run_controller_step()
    th.cuda.synchronize() if th.cuda.is_available() else None

    total_times = []
    per_type_times = defaultdict(list)

    for _ in range(n_iters):
        if per_type:
            t0 = time.perf_counter()
            type_timings = _run_controller_step_timed()
            total_times.append((time.perf_counter() - t0) * 1000)
            for ctype, dt in type_timings.items():
                per_type_times[ctype].append(dt)
        else:
            t0 = time.perf_counter()
            _run_controller_step()
            th.cuda.synchronize() if th.cuda.is_available() else None
            total_times.append((time.perf_counter() - t0) * 1000)

    arr = np.array(total_times)
    print(f"\n--- {label} ({n_iters} iters) ---")
    print(f"  Total:  mean={arr.mean():.4f}  median={np.median(arr):.4f}  "
          f"std={arr.std():.4f}  min={arr.min():.4f}  max={arr.max():.4f} ms")

    per_type_arrays = {}
    if per_type:
        for ctype in sorted(per_type_times.keys(), key=lambda c: c.name):
            a = np.array(per_type_times[ctype])
            per_type_arrays[ctype] = a
            print(f"    {ctype.name:40s}  mean={a.mean():.4f}  median={np.median(a):.4f} ms")

    return arr, per_type_arrays


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

    type_counts = {}
    for cid, ctype in Controller._types.items():
        type_counts[ctype.name] = type_counts.get(ctype.name, 0) + 1
    print(f"\nActive controllers per type ({args.num_robots} robots):")
    for name, count in sorted(type_counts.items()):
        print(f"  {name}: {count}")

    # ---- Benchmark BATCHED (current code, with per-type timing) ----
    batched_total, batched_per_type = _benchmark(
        args.warmup, args.iters, f"BATCHED ({args.arm_controller})", per_type=True
    )

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

    seq_total, seq_per_type = _benchmark(
        args.warmup, args.iters, f"SEQUENTIAL ({args.arm_controller})", per_type=True
    )

    Controller.step_batch = original_step_batch

    # ---- Summary with per-type breakdown ----
    print("\n========== SUMMARY ==========")
    print(f"Arm controller:   {controller_name}")
    print(f"Robots:           {len(env.robots)}")
    print(f"Iterations:       {args.iters}")

    header = f"{'':40s}  {'Batched':>8s}  {'Sequential':>10s}  {'Speedup':>8s}"
    print(header)
    print(f"{'TOTAL':40s}  {batched_total.mean():8.4f}  {seq_total.mean():10.4f}  "
          f"{seq_total.mean()/batched_total.mean():7.2f}x")

    all_types = sorted(
        set(list(batched_per_type.keys()) + list(seq_per_type.keys())),
        key=lambda c: c.name,
    )
    for ctype in all_types:
        b_mean = batched_per_type[ctype].mean() if ctype in batched_per_type else float("nan")
        s_mean = seq_per_type[ctype].mean() if ctype in seq_per_type else float("nan")
        count = type_counts.get(ctype.name, 0)
        speedup = s_mean / b_mean if b_mean > 0 else float("nan")
        print(f"  {ctype.name:38s}  {b_mean:8.4f}  {s_mean:10.4f}  "
              f"{speedup:7.2f}x  ({count} instances)")

    print("=============================\n")

    og.shutdown()


if __name__ == "__main__":
    main()
