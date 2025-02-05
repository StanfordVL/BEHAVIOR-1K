from omnigibson.utils.constants import PrimType
from omnigibson.object_states import Folded, Unfolded
from omnigibson.macros import gm
import numpy as np

import omnigibson as og
import json
import torch

# Make sure object states and GPU dynamics are enabled (GPU dynamics needed for cloth)
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True


def main(random_selection=False, headless=False, short_exec=False):
    """
    Demo of cloth objects that can potentially be folded.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    cloth_category_models = [
        ("hoodie", "agftpm"),
    ]

    for cloth in cloth_category_models:
        category = cloth[0]
        model = cloth[1]
        if model != "agftpm":
            continue
        print(f"\nCategory: {category}, Model: {model}!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # Create the scene config to load -- empty scene + custom cloth object
        cfg = {
            "scene": {
                "type": "Scene",
            },
            "objects": [
                {
                    "type": "DatasetObject",
                    "name": model,
                    "category": category,
                    "model": model,
                    # "bounding_box": [0.897, 0.568, 0.012],
                    "prim_type": PrimType.CLOTH,
                    "abilities": {"cloth": {}},
                    # "position": [0, 0, 0.5],
                    "orientation": [0.7071, 0.0, 0.7071, 0.0],
                },
            ],
        }

        # Create the environment
        env = og.Environment(configs=cfg)

        # Grab object references
        carpet = env.scene.object_registry("name", model)
        objs = [carpet]

        # Set viewer camera
        og.sim.viewer_camera.set_position_orientation(
            position=np.array([0.46382895, -2.66703958, 1.22616824]),
            orientation=np.array([0.58779174, -0.00231237, -0.00318273, 0.80900271]),
        )

        for _ in range(100):
            og.sim.step()

        print("\nCloth state:\n")

        if not short_exec:
            # Get particle positions
            increments = 100
            obj = objs[0]

            # Fold - first stage
            pos = np.asarray(obj.root_link.compute_particle_positions())
            y_min = np.min(pos[:, 1])
            y_max = np.max(pos[:, 1])
            y_mid_bottom = y_min + (y_max - y_min) / 3
            y_mid_top = y_min + 2 * (y_max - y_min) / 3
            bottom_indices = np.where(pos[:, 1] < y_mid_bottom)[0]
            bottom_start = np.copy(pos[bottom_indices])
            bottom_end = np.copy(bottom_start)
            bottom_end[:, 1] = 2 * y_mid_bottom - bottom_end[:, 1]  # Mirror bottom to the above of y_mid_bottom
            for ctrl_pts in np.linspace(bottom_start, bottom_end, increments):
                obj.root_link.set_particle_positions(torch.tensor(ctrl_pts), idxs=bottom_indices)
                og.sim.step()

            # Fold - second stage
            pos = np.asarray(obj.root_link.compute_particle_positions())
            y_min = np.min(pos[:, 1])
            y_max = np.max(pos[:, 1])
            y_mid_bottom = y_min + (y_max - y_min) / 3
            y_mid_top = y_min + 2 * (y_max - y_min) / 3
            top_indices = np.where(pos[:, 1] > y_mid_top)[0]
            top_start = np.copy(pos[top_indices])
            top_end = np.copy(top_start)
            top_end[:, 1] = 2 * y_mid_top - top_end[:, 1]  # Mirror top to the below of y_mid_top
            for ctrl_pts in np.linspace(top_start, top_end, increments):
                obj.root_link.set_particle_positions(torch.tensor(ctrl_pts), idxs=top_indices)
                og.sim.step()

            # Fold - third stage
            pos = np.asarray(obj.root_link.compute_particle_positions())
            x_min = np.min(pos[:, 0])
            x_max = np.max(pos[:, 0])
            x_mid = (x_min + x_max) / 2
            indices = np.argsort(pos, axis=0)[:, 0][-len(pos) // 2 :]
            start = np.copy(pos[indices])
            end = np.copy(start)
            end[:, 0] = 2 * x_mid - end[:, 0]
            for ctrl_pts in np.linspace(start, end, increments):
                obj.root_link.set_particle_positions(torch.tensor(ctrl_pts), idxs=indices)
                og.sim.step()

            while True:
                print(f"\nCategory: {category}, Model: {model}!!!!!!!!!!!!!!!!!!!!!!!!!!")
                env.step(np.array([]))

        # Shut down env at the end
        print()
        env.close()


if __name__ == "__main__":
    main()
