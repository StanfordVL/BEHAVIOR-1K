import os
import argparse
import json
import omnigibson as og
import numpy as np
import omnigibson.utils.transform_utils as T
import time

from omnigibson.macros import gm
from omnigibson.systems import get_system
from omnigibson.object_states import Covered
from omnigibson.utils.profiling_utils import ProfilingEnv
from omnigibson.utils.constants import PrimType

parser = argparse.ArgumentParser()

parser.add_argument("-r", "--robot", type=int, default=0)
parser.add_argument("-s", "--scene", default="")
parser.add_argument("-c", "--cloth", action='store_true')
parser.add_argument("-w", "--fluids", action='store_true')
parser.add_argument("-g", "--gpu_denamics", action='store_true')
parser.add_argument("-p", "--macro_particle_system", action='store_true')

PROFILING_FIELDS = ["FPS", "Omni step time", "Non-omni step time", "Memory usage", "Vram usage"]
NUM_CLOTH = 5
NUM_SLICE_OBJECT = 3

SCENE_OFFSET = {
    "": [0, 0],
    "Rs_int": [0, 0],
    "Pomaria_0_garden": [0.3, 0],
    "grocery_store_cafe": [-3.5, 3.5],
    "house_single_floor": [-3, -1],
    "Ihlen_0_int": [-1, 2]
}


def main():
    args = parser.parse_args()
    # Modify macros settings
    gm.ENABLE_HQ_RENDERING = args.fluids
    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = True
    gm.ENABLE_FLATCACHE = not args.cloth
    gm.USE_GPU_DYNAMICS = args.gpu_denamics

    cfg = {
        "env": {
            "action_frequency": 30,
            "physics_frequency": 120,
        }
    }
    if args.robot > 0:
        cfg["robots"] = []
        for i in range(args.robot):
            cfg["robots"].append({
                "type": "Fetch",
                "obs_modalities": "all",
                "position": [-1.3 + 0.75 * i + SCENE_OFFSET[args.scene][0], 0.5 + SCENE_OFFSET[args.scene][1], 0],
                "orientation": [0., 0., 0.7071, -0.7071]
            })

    if args.scene:
        assert args.scene in SCENE_OFFSET, f"Scene {args.scene} not found in SCENE_OFFSET"
        cfg["scene"] = {
            "type": "InteractiveTraversableScene",
            "scene_model": args.scene,
        }
    else:
        cfg["scene"] = {"type": "Scene"}

    cfg["objects"] = [{
        "type": "DatasetObject",
        "name": "table",
        "category": "breakfast_table",
        "model": "rjgmmy",
        "fixed_base": True,
        "scale": [0.75] * 3,
        "position": [0.5 + SCENE_OFFSET[args.scene][0], -1 + SCENE_OFFSET[args.scene][1], 0.3],
        "orientation": [0., 0., 0.7071, -0.7071]
    }]
    
    if args.cloth:
        cfg["objects"].extend([{
            "type": "DatasetObject",
            "name": f"cloth_{n}",
            "category": "t_shirt",
            "model": "kvidcx",
            "prim_type": PrimType.CLOTH,
            "abilities": {"cloth": {}},
            "bounding_box": [0.3, 0.5, 0.7],
            "position": [-0.4, -1, 0.7 + n * 0.4],
            "orientation": [0.7071, 0., 0.7071, 0.],
        } for n in range(NUM_CLOTH)])
    
    cfg["objects"].extend([{
        "type": "DatasetObject",
        "name": f"apple_{n}",
        "category": "apple",
        "model": "agveuv",
        "scale": [1.5] * 3,
        "position": [0.5 + SCENE_OFFSET[args.scene][0], -1.25 + SCENE_OFFSET[args.scene][1] + n * 0.2, 0.5],
        "abilities": {"diceable": {}} if args.macro_particle_system else {}
    } for n in range(NUM_SLICE_OBJECT)])
    cfg["objects"].extend([{
        "type": "DatasetObject",
        "name": f"knife_{n}",
        "category": "table_knife",
        "model": "jxdfyy",
        "scale": [2.5] * 3
    } for n in range(NUM_SLICE_OBJECT)])

    load_start = time.time()
    env = ProfilingEnv(configs=cfg)
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
            position=apples[n].get_position() + np.array([-0.15, 0.0, 0.1 * (n + 2)]),
            orientation=T.euler2quat([-np.pi / 2, 0, 0]),
        )
        knife.keep_still()
    if args.fluids:
        table.states[Covered].set_value(get_system("water"), True)

    output, results = [], []

    # Update the simulator's viewer camera's pose so it points towards the robot
    og.sim.viewer_camera.set_position([SCENE_OFFSET[args.scene][0], -3 + SCENE_OFFSET[args.scene][1], 1])
    # record total load time
    total_load_time = time.time() - load_start
    
    for i in range(300):
        if args.robot:
            result = env.step(np.array([np.random.uniform(-0.3, 0.3, env.robots[i].action_dim) for i in range(args.robot)]).flatten())[4]
        else:
            result = env.step(None)[4]
        results.append(result)

    field = f"{args.scene}" if args.scene else "Empty scene"
    if args.robot:
        field += f", with {args.robot} Fetch"
    if args.cloth:
        field += ", cloth" 
    if args.fluids:
        field += ", fluids"
    if args.macro_particle_system:
        field += ", macro particles"
    output.append({
        "name": field,
        "unit": "time (ms)",
        "value": total_load_time,
        "extra": ["Loading time", "Loading time"]
    })
    results = np.array(results)
    for i, title in enumerate(PROFILING_FIELDS):
        unit = "time (ms)" if 'time' in title else "GB"
        value = np.mean(results[:, i])
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
    og.shutdown()

if __name__ == "__main__":
    main()
