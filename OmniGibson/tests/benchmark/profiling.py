import argparse
import cProfile
import json
import math
import os
import time

import psutil
import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.object_states import Covered
from omnigibson.utils.constants import PrimType
from omnigibson.utils.profiling_utils import get_vram_usage

parser = argparse.ArgumentParser()

parser.add_argument("-r", "--robot", type=int, default=0)
parser.add_argument("-s", "--scene", default="")
parser.add_argument("-c", "--cloth", action="store_true")
parser.add_argument("-w", "--fluids", action="store_true")
parser.add_argument("-g", "--gpu_dynamics", action="store_true")
parser.add_argument("-p", "--macro_particle_system", action="store_true")
parser.add_argument("-d", "--deep-profiling", action="store_true")

PROFILING_FIELDS = ["FPS", "Isaac step time", "Non-Isaac step time", "Memory usage", "Vram usage"]
NUM_CLOTH = 5
NUM_SLICE_OBJECT = 3

SCENE_OFFSET = {
    "": [0, 0],
    "Rs_int": [0, 0],
    "Pomaria_0_garden": [0.3, 0],
    "grocery_store_cafe": [-3.5, 3.5],
    "house_single_floor": [-3, -1],
    "Ihlen_0_int": [-1, 2],
}


def main():
    args = parser.parse_args()
    # Modify macros settings
    # gm.ENABLE_HQ_RENDERING = args.fluids  # Temporarily disabled, since it requires >= 60 FPS
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = True
    gm.USE_GPU_DYNAMICS = args.gpu_dynamics
    gm.ENABLE_DEEP_PROFILING = args.deep_profiling

    cfg = {
        "env": {
            "action_frequency": 30,
            "physics_frequency": 120,
        }
    }
    if args.robot > 0:
        cfg["robots"] = []
        for i in range(args.robot):
            cfg["robots"].append(
                {
                    "model": "r1pro",
                    "obs_modalities": ["rgb"],
                    "position": [-1.3 + 0.75 * i + SCENE_OFFSET[args.scene][0], 0.5 + SCENE_OFFSET[args.scene][1], 0],
                    "orientation": [0.0, 0.0, 0.7071, -0.7071],
                }
            )

    if args.scene:
        assert args.scene in SCENE_OFFSET, f"Scene {args.scene} not found in SCENE_OFFSET"
        cfg["scene"] = {
            "type": "InteractiveTraversableScene",
            "scene_model": args.scene,
        }
    else:
        cfg["scene"] = {"type": "Scene"}

    cfg["objects"] = [
        {
            "type": "DatasetObject",
            "name": "table",
            "category": "breakfast_table",
            "model": "rjgmmy",
            "fixed_base": True,
            "scale": [0.75] * 3,
            "position": [0.5 + SCENE_OFFSET[args.scene][0], -1 + SCENE_OFFSET[args.scene][1], 0.3],
            "orientation": [0.0, 0.0, 0.7071, -0.7071],
        }
    ]

    if args.cloth:
        cfg["objects"].extend(
            [
                {
                    "type": "DatasetObject",
                    "name": f"cloth_{n}",
                    "category": "t_shirt",
                    "model": "kvidcx",
                    "prim_type": PrimType.CLOTH,
                    "abilities": {"cloth": {}},
                    "bounding_box": [0.3, 0.5, 0.7],
                    "position": [-0.4, -1, 0.7 + n * 0.4],
                    "orientation": [0.7071, 0.0, 0.7071, 0.0],
                }
                for n in range(NUM_CLOTH)
            ]
        )

    cfg["objects"].extend(
        [
            {
                "type": "DatasetObject",
                "name": f"apple_{n}",
                "category": "apple",
                "model": "agveuv",
                "scale": [1.5] * 3,
                "position": [0.5 + SCENE_OFFSET[args.scene][0], -1.25 + SCENE_OFFSET[args.scene][1] + n * 0.2, 0.5],
                "abilities": {"diceable": {}} if args.macro_particle_system else {},
            }
            for n in range(NUM_SLICE_OBJECT)
        ]
    )
    cfg["objects"].extend(
        [
            {
                "type": "DatasetObject",
                "name": f"knife_{n}",
                "category": "table_knife",
                "model": "jxdfyy",
                "scale": [2.5] * 3,
            }
            for n in range(NUM_SLICE_OBJECT)
        ]
    )

    load_start = time.time()

    # Launch OG before setting up the profiler. If we don't do this then the carb profiler
    # overtakes the profiler and we don't get any useful data.
    og.launch()

    if args.deep_profiling:
        load_profiler = cProfile.Profile()
        load_profiler.enable()
    env = og.Environment(configs=cfg)
    table = env.scene.object_registry("name", "table")
    apples = [env.scene.object_registry("name", f"apple_{n}") for n in range(NUM_SLICE_OBJECT)]
    knifes = [env.scene.object_registry("name", f"knife_{n}") for n in range(NUM_SLICE_OBJECT)]
    if args.cloth:
        clothes = [env.scene.object_registry("name", f"cloth_{n}") for n in range(NUM_CLOTH)]
        for cloth in clothes:
            cloth.root_link.mass = 1.0
    env.reset()

    for n, knife in enumerate(knifes):
        knife.set_position_orientation(
            position=apples[n].get_position_orientation()[0] + th.tensor([-0.15, 0.0, 0.1 * (n + 2)]),
            orientation=T.euler2quat(th.tensor([-math.pi / 2, 0, 0], dtype=th.float32)),
        )
        knife.keep_still()
    if args.fluids:
        table.states[Covered].set_value(env.scene.get_system("water"), True)

    output = []

    # Update the simulator's viewer camera's pose so it points towards the robot
    og.sim.viewer_camera.set_position_orientation(
        position=[SCENE_OFFSET[args.scene][0], -3 + SCENE_OFFSET[args.scene][1], 1]
    )
    # record total load time
    if args.deep_profiling:
        load_profiler.disable()
        load_profiler.dump_stats("load.prof")
    total_load_time = time.time() - load_start

    # Reset profiler counters so we only measure the benchmark loop
    og.sim._step_profiler.reset()
    og.sim._pre_physics_step_profiler.reset()
    og.sim._post_physics_step_profiler.reset()
    og.sim._non_physics_step_profiler.reset()

    for i in range(300):
        if args.robot:
            action_lo, action_hi = -0.3, 0.3
            env.step(
                th.stack(
                    [th.rand(env.robots[i].action_dim) * (action_hi - action_lo) + action_lo for i in range(args.robot)]
                ).flatten()
            )
        else:
            env.step(None)

    # Compute timing metrics from simulator profilers (convert to ms)
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
    memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024**3
    vram_usage = get_vram_usage()

    result_values = [avg_total_ms, avg_isaac_ms, avg_og_ms, memory_usage, vram_usage]

    if n_steps % 100 == 0 or n_steps == 300:
        print(
            "total time: {:.3f} ms, Isaac time: {:.3f} ms, Non-Isaac time: {:.3f} ms, memory: {:.3f} GB, vram: {:.3f} GB.".format(
                *result_values
            )
        )

    field = f"{args.scene}" if args.scene else "Empty scene"
    if args.robot:
        field += f", with {args.robot} Fetch"
    if args.cloth:
        field += ", cloth"
    if args.fluids:
        field += ", fluids"
    if args.macro_particle_system:
        field += ", macro particles"
    output.append(
        {"name": field, "unit": "time (ms)", "value": total_load_time, "extra": ["Loading time", "Loading time"]}
    )
    for i, title in enumerate(PROFILING_FIELDS):
        unit = "time (ms)" if "time" in title else "GB"
        value = result_values[i]
        if title == "FPS":
            value = 1000 / value
            unit = "fps"
        output.append({"name": field, "unit": unit, "value": value, "extra": [title, title]})

    ret = []
    if os.path.exists("output.json"):
        with open("output.json", "r") as f:
            ret = json.load(f)
    ret.extend(output)
    with open("output.json", "w") as f:
        json.dump(ret, f, indent=4)

    # Save the simulation profilers
    if args.deep_profiling:
        og.sim._step_profiler.dump_stats("step.prof")
        og.sim._pre_physics_step_profiler.dump_stats("pre_physics_step.prof")
        og.sim._post_physics_step_profiler.dump_stats("post_physics_step.prof")
        og.sim._non_physics_step_profiler.dump_stats("non_physics_step.prof")

    og.shutdown()


if __name__ == "__main__":
    main()
