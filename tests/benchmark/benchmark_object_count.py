"""Script to benchmark speed vs. no. of objects in the scene."""

import os
import time

import matplotlib.pyplot as plt
import numpy as np

from igibson import app, Simulator
from igibson.objects.primitive_object import PrimitiveObject
from igibson.scenes.empty_scene import EmptyScene
from igibson.utils.assets_utils import get_ig_assets_version


# Params to be set as needed.
MAX_NUM_OBJS = 1000      # Maximum no. of objects to add.
NUM_OBJS_PER_ITER = 20   # No. of objects to add per iteration.
NUM_STEPS_PER_ITER = 30  # No. of steps to take for each n of objects.
OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Desktop")

# Internal constants.
_N_PER_ROW = int(np.sqrt(MAX_NUM_OBJS))
_MIN_VAL = -2.0
_MAX_VAL = 2.0
_STEP_SIZE = (_MAX_VAL - _MIN_VAL) / _N_PER_ROW


def benchmark_scene(sim):
    assets_version = get_ig_assets_version()
    print("assets_version", assets_version)

    scene = EmptyScene(floor_plane_visible=True)
    sim.import_scene(scene)
    sim.play()

    xs = []
    ys = []
    yerrs = []
    for i in range(MAX_NUM_OBJS // NUM_OBJS_PER_ITER):
        new_objs = []
        for j in range(NUM_OBJS_PER_ITER):
            obj_idx = i * NUM_OBJS_PER_ITER + j
            obj = PrimitiveObject(
                prim_path=f"/World/obj{obj_idx}",
                primitive_type="Sphere",
                name=f"obj{obj_idx}",
                scale=0.05,
                visual_only=False,
            )
            sim.import_object(obj=obj, auto_initialize=False)
            x = _MIN_VAL + _STEP_SIZE * (obj_idx % _N_PER_ROW)
            y = _MIN_VAL + _STEP_SIZE * (obj_idx // _N_PER_ROW)
            obj.set_position(position=np.array([x, y, 0.1]))
            new_objs.append(obj)

        # Take a step to initialize the new objects (done in _non_physics_step()).
        sim.step()
        step_freqs = []
        for _ in range(NUM_STEPS_PER_ITER):
            start = time.time()
            sim.step()
            end = time.time()
            step_freqs.append(1 / (end - start))

        xs.append(i * NUM_OBJS_PER_ITER)
        max_freq, min_freq = np.max(step_freqs), np.min(step_freqs)
        ys.append(np.mean((max_freq, min_freq)))
        yerrs.append(max_freq - ys[-1])

    plt.figure(figsize=(9, 6))
    ax = plt.subplot(1, 1, 1)
    plt.errorbar(xs, ys, yerr=yerrs, elinewidth=0.75)
    ax.set_xlabel("No. of objects")
    ax.set_ylabel("Render fps")
    ax.set_title(f"Version {assets_version}")
    plt.tight_layout()
    plt.savefig(os.path.join(
        OUTPUT_DIR, f"scene_objs_benchmark_{MAX_NUM_OBJS}.png"))


def main():
    assert MAX_NUM_OBJS <= 1000

    sim = Simulator()
    benchmark_scene(sim)
    app.close()


if __name__ == "__main__":
    main()
