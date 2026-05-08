import argparse
import cProfile
import json
import os
import time

import psutil
import torch as th

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.profiling_utils import get_vram_usage

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n-envs", type=int, default=1)
parser.add_argument("-r", "--rendering", action="store_true")
parser.add_argument("-t", "--task-type", choices=["simple", "behavior", "navigation"], default="simple")
parser.add_argument("-o", "--object-states", action="store_true")
parser.add_argument("-g", "--gpu-dynamics", action="store_true")
parser.add_argument("-d", "--deep-profiling", action="store_true")

NUM_STEPS = 300


def apply_macros(args):
    gm.ENABLE_OBJECT_STATES = args.object_states
    gm.ENABLE_TRANSITION_RULES = args.object_states
    gm.USE_GPU_DYNAMICS = args.gpu_dynamics
    gm.ENABLE_DEEP_PROFILING = args.deep_profiling


def make_config(args):
    cfg = {
        "env": {
            "action_frequency": 30,
            "physics_frequency": 120,
        },
    }

    cfg["scene"] = {
        "type": "InteractiveTraversableScene",
        "scene_model": "Rs_int",
        "load_object_categories": ["floors", "breakfast_table"],
    }
    cfg["robots"] = [
        {
            "model": "r1pro",
            "obs_modalities": ["rgb"] if args.rendering else ["proprio"],
            "position": [-1.3, 0.5, 0.0],
            "orientation": [0.0, 0.0, 0.7071, -0.7071],
        }
    ]
    if args.task_type == "behavior":
        cfg["task"] = {
            "type": "BehaviorTask",
            "activity_name": "laying_wood_floors",
            "activity_definition_id": 0,
            "online_object_sampling": True,
            "use_presampled_robot_pose": False,
        }
    elif args.task_type == "navigation":
        cfg["task"] = {
            "type": "PointNavigationTask",
            "robot_idn": 0,
            "floor": 0,
            "goal_tolerance": 0.36,
            "path_range": [1.0, 10.0],
            "visualize_goal": False,
            "visualize_path": False,
        }
    else:
        cfg["task"] = {"type": "DummyTask"}

    return cfg


def main():
    args = parser.parse_args()

    apply_macros(args)

    load_start = time.time()
    og.launch()

    cfg = make_config(args)
    cfg["env"]["num_envs"] = args.n_envs
    if args.deep_profiling:
        load_profiler = cProfile.Profile()
        load_profiler.enable()
    env = og.Environment(configs=cfg)
    env.reset()
    if args.deep_profiling:
        load_profiler.disable()
        load_profiler.dump_stats("load.prof")
    total_load_time = time.time() - load_start

    og.sim._step_profiler.reset()
    og.sim._pre_physics_step_profiler.reset()
    og.sim._post_physics_step_profiler.reset()
    og.sim._non_physics_step_profiler.reset()

    action_lo, action_hi = -0.3, 0.3
    action_dims = [scene.robots[0].action_dim for scene in env.scenes]
    for _ in range(NUM_STEPS):
        actions = th.stack([th.rand(d) * (action_hi - action_lo) + action_lo for d in action_dims])
        env.step(actions)

    n_steps = og.sim._step_profiler.call_count
    avg_total_ms = og.sim._step_profiler.average_time * 1e3
    avg_og_ms = (
        (
            og.sim._pre_physics_step_profiler.total_time
            + og.sim._post_physics_step_profiler.total_time
            + og.sim._non_physics_step_profiler.total_time
        )
        / n_steps
        * 1e3
    )
    avg_isaac_ms = avg_total_ms - avg_og_ms
    fps = 1000.0 / avg_total_ms * args.n_envs
    memory_gb = psutil.Process(os.getpid()).memory_info().rss / 1024**3
    vram_gb = get_vram_usage()

    label = f"{args.n_envs} {'env' if args.n_envs == 1 else 'envs'}"
    label += ", rendering" if args.rendering else ", no rendering"
    label += f", {args.task_type} task"
    label += ", states on" if args.object_states else ", states off"

    output = [
        {"name": label, "unit": "time (s)", "value": total_load_time, "extra": ["Loading time", "Loading time"]},
        {"name": label, "unit": "fps", "value": fps, "extra": ["FPS", "FPS"]},
        {"name": label, "unit": "time (ms)", "value": avg_isaac_ms, "extra": ["Isaac step time", "Isaac step time"]},
        {
            "name": label,
            "unit": "time (ms)",
            "value": avg_og_ms,
            "extra": ["Non-Isaac step time", "Non-Isaac step time"],
        },
        {"name": label, "unit": "GB", "value": memory_gb, "extra": ["Memory usage", "Memory usage"]},
        {"name": label, "unit": "GB", "value": vram_gb, "extra": ["Vram usage", "Vram usage"]},
    ]

    ret = []
    if os.path.exists("vector_output.json"):
        with open("vector_output.json") as f:
            ret = json.load(f)
    ret.extend(output)
    with open("vector_output.json", "w") as f:
        json.dump(ret, f, indent=4)

    print(
        f"[{label}] load={total_load_time:.1f}s  FPS={fps:.1f}  Isaac={avg_isaac_ms:.2f}ms  "
        f"OG={avg_og_ms:.2f}ms  RAM={memory_gb:.2f}GB  VRAM={vram_gb:.2f}GB"
    )

    if args.deep_profiling:
        og.sim._step_profiler.dump_stats("step.prof")
        og.sim._pre_physics_step_profiler.dump_stats("pre_physics_step.prof")
        og.sim._post_physics_step_profiler.dump_stats("post_physics_step.prof")
        og.sim._non_physics_step_profiler.dump_stats("non_physics_step.prof")

    og.shutdown()


if __name__ == "__main__":
    main()
