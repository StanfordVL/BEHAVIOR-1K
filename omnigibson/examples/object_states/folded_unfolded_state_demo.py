from omnigibson.utils.constants import PrimType
from omnigibson.object_states import Folded, Unfolded
from omnigibson.macros import gm
import numpy as np

import omnigibson as og

# Make sure object states and GPU dynamics are enabled (GPU dynamics needed for cloth)
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True


def main(random_selection=False, headless=False, short_exec=False):
    """
    Demo of cloth objects that can potentially be folded.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

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
                "model": "ctclvd",
                "bounding_box": [0.897, 0.568, 0.012],
                "prim_type": PrimType.CLOTH,
                "abilities": {"cloth": {}},
                "position": [0, 0, 0.5],
            },
            {
                "type": "DatasetObject",
                "name": "dishtowel",
                "category": "dishtowel",
                "model": "dtfspn",
                "bounding_box": [0.852, 1.1165, 0.174],
                "prim_type": PrimType.CLOTH,
                "abilities": {"cloth": {}},
                "position": [1, 1, 0.5],
            },
            {
                "type": "DatasetObject",
                "name": "shirt",
                "category": "t_shirt",
                "model": "kvidcx",
                "bounding_box": [0.472, 1.243, 1.158],
                "prim_type": PrimType.CLOTH,
                "abilities": {"cloth": {}},
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

    # Set viewer camera
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([0.46382895, -2.66703958, 1.22616824]),
        orientation=np.array([0.58779174, -0.00231237, -0.00318273, 0.80900271]),
    )

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

        print(f"{info}{' ' * (110 - len(info))}", end="\r")

    for _ in range(100):
        og.sim.step()

    print("\nCloth state:\n")

    if not short_exec:
        # Fold all three cloths along the x-axis
        for i in range(3):
            obj = objs[i]
            pos = obj.root_link.compute_particle_positions()
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
                obj.root_link.set_particle_positions(ctrl_pts, idxs=indices)
                og.sim.step()
                print_state()

        # Fold the t-shirt twice again along the y-axis
        for direction in [-1, 1]:
            obj = shirt
            pos = obj.root_link.compute_particle_positions()
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
                obj.root_link.set_particle_positions(ctrl_pts, idxs=indices)
                env.step(np.array([]))
                print_state()

        while True:
            env.step(np.array([]))
            print_state()

    # Shut down env at the end
    print()
    env.close()


if __name__ == "__main__":
    main()
