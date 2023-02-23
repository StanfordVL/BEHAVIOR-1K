from omnigibson.utils.constants import PrimType
from omnigibson.object_states import Folded, Unfolded
from omnigibson.macros import gm
import logging
import numpy as np

import omnigibson as og

# Make sure object states and GPU dynamics are enabled (GPU dynamics needed for cloth)
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True


def main(random_selection=False, headless=False, short_exec=False):
    """
    Demo of cloth objects that can potentially be folded.
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    # Create the scene config to load -- empty scene + custom cloth object
    cfg = {
        "scene": {
            "type": "Scene",
        },
        "objects": [
            {
                "type": "DatasetObject",
                "name": "carpet",
                "category": "carpet",
                "model": "carpet_0",
                "prim_type": PrimType.CLOTH,
                "abilities": {"foldable": {}, "unfoldable": {}},
                "position": [0, 0, 0.5],
            },
            {
                "type": "DatasetObject",
                "name": "dishtowel",
                "category": "dishtowel",
                "model": "Tag_Dishtowel_Basket_Weave_Red",
                "prim_type": PrimType.CLOTH,
                "scale": 5.0,
                "abilities": {"foldable": {}, "unfoldable": {}},
                "position": [1, 1, 0.5],
            },
            {
                "type": "DatasetObject",
                "name": "shirt",
                "category": "t-shirt",
                "model": "t-shirt_000",
                "prim_type": PrimType.CLOTH,
                "scale": 0.05,
                "abilities": {"foldable": {}, "unfoldable": {}},
                "position": [-1, 1, 0.5],
                "orientation": [0.7071, 0., 0.7071, 0.],
            },
        ],
    }

    # Create the environment
    env = og.Environment(configs=cfg, action_timestep=1 / 60., physics_timestep=1 / 60.)

    # Grab object references
    carpet = env.scene.object_registry("name", "carpet")
    dishtowel = env.scene.object_registry("name", "dishtowel")
    shirt = env.scene.object_registry("name", "shirt")
    objs = [carpet, dishtowel, shirt]

    def print_state():
        folded = carpet.states[Folded].get_value()
        unfolded = carpet.states[Unfolded].get_value()
        info = "carpet: [folded] %d [unfolded] %d" % (folded, unfolded)

        folded = dishtowel.states[Folded].get_value()
        unfolded = dishtowel.states[Unfolded].get_value()
        info += " || dishtowel: [folded] %d [unfolded] %d" % (folded, unfolded)

        folded = shirt.states[Folded].get_value()
        unfolded = shirt.states[Unfolded].get_value()
        info += " || tshirt: [folded] %d [unfolded] %d" % (folded, unfolded)

        print(info)

    for _ in range(100):
        og.sim.step()

    if not short_exec:
        # Fold all three cloths along the x-axis
        for i in range(3):
            obj = objs[i]
            pos = obj.root_link.particle_positions
            x_min, x_max = np.min(pos, axis=0)[0], np.max(pos, axis=0)[0]
            x_extent = x_max - x_min
            # Get indices for the bottom 10 percent vertices in the x-axis
            indices = np.argsort(pos, axis=0)[:, 0][:(pos.shape[0] // 10)]
            start = np.copy(pos[indices])

            # lift up a bit
            mid = np.copy(start)
            mid[:, 2] += x_extent * 0.2

            # move towards x_max
            end = np.copy(mid)
            end[:, 0] += x_extent * 0.9

            increments = 25
            for ctrl_pts in np.concatenate([np.linspace(start, mid, increments), np.linspace(mid, end, increments)]):
                pos = obj.root_link.particle_positions
                pos[indices] = ctrl_pts
                obj.root_link.particle_positions = pos
                og.sim.step()
                print_state()

        # Fold the t-shirt twice again along the y-axis
        for direction in [-1, 1]:
            obj = shirt
            pos = obj.root_link.particle_positions
            y_min, y_max = np.min(pos, axis=0)[1], np.max(pos, axis=0)[1]
            y_extent = y_max - y_min
            if direction == 1:
                indices = np.argsort(pos, axis=0)[:, 1][:(pos.shape[0] // 20)]
            else:
                indices = np.argsort(pos, axis=0)[:, 1][-(pos.shape[0] // 20):]
            start = np.copy(pos[indices])

            # lift up a bit
            mid = np.copy(start)
            mid[:, 2] += y_extent * 0.2

            # move towards y_max
            end = np.copy(mid)
            end[:, 1] += direction * y_extent * 0.4

            increments = 25
            for ctrl_pts in np.concatenate([np.linspace(start, mid, increments), np.linspace(mid, end, increments)]):
                pos = obj.root_link.particle_positions
                pos[indices] = ctrl_pts
                obj.root_link.particle_positions = pos
                og.sim.step()
                print_state()

        while True:
            og.sim.step()
            print_state()

    # Shut down env at the end
    env.close()


if __name__ == "__main__":
    main()
