import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.object_states import Folded, Unfolded
from omnigibson.utils.constants import PrimType
from omnigibson.utils.python_utils import multi_dim_linspace
from omnigibson.utils.ui_utils import KeyboardEventHandler

# Make sure object states and GPU dynamics are enabled (GPU dynamics needed for cloth)
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_TRANSITION_RULES = False


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
                "orientation": [0.7071, 0.0, 0.7071, 0.0],
            },
        ],
    }

    # Create the environment
    env = og.Environment(configs=cfg)

    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.ESCAPE,
        callback_fn=lambda: og.shutdown(),
    )

    # Grab object references
    carpet = env.scene.object_registry("name", "carpet")
    dishtowel = env.scene.object_registry("name", "dishtowel")
    shirt = env.scene.object_registry("name", "shirt")
    objs = [carpet, dishtowel, shirt]

    # Set viewer camera
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([0.46382895, -2.66703958, 1.22616824]),
        orientation=th.tensor([0.58779174, -0.00231237, -0.00318273, 0.80900271]),
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
            x_min, x_max = th.min(pos, dim=0).values[0], th.max(pos, dim=0).values[0]
            x_extent = x_max - x_min
            # Get indices for the bottom 10 percent vertices in the x-axis
            indices = th.argsort(pos, dim=0)[:, 0][: (pos.shape[0] // 10)]
            start = th.clone(pos[indices])

            # lift up a bit
            mid = th.clone(start)
            mid[:, 2] += x_extent * 0.2

            # move towards x_max
            end = th.clone(mid)
            end[:, 0] += x_extent * 0.9

            increments = 25
            for ctrl_pts in th.cat(
                [multi_dim_linspace(start, mid, increments), multi_dim_linspace(mid, end, increments)]
            ):
                obj.root_link.set_particle_positions(ctrl_pts, idxs=indices)
                og.sim.step()
                print_state()

        # Fold the t-shirt twice again along the y-axis
        for direction in [-1, 1]:
            obj = shirt
            pos = obj.root_link.compute_particle_positions()
            y_min, y_max = th.min(pos, dim=0).values[1], th.max(pos, dim=0).values[1]
            y_extent = y_max - y_min
            if direction == 1:
                indices = th.argsort(pos, dim=0)[:, 1][: (pos.shape[0] // 20)]
            else:
                indices = th.argsort(pos, dim=0)[:, 1][-(pos.shape[0] // 20) :]
            start = th.clone(pos[indices])

            # lift up a bit
            mid = th.clone(start)
            mid[:, 2] += y_extent * 0.2

            # move towards y_max
            end = th.clone(mid)
            end[:, 1] += direction * y_extent * 0.4

            increments = 25
            for ctrl_pts in th.cat(
                [multi_dim_linspace(start, mid, increments), multi_dim_linspace(mid, end, increments)]
            ):
                obj.root_link.set_particle_positions(ctrl_pts, idxs=indices)
                env.step(th.empty(0))
                print_state()

        while True:
            env.step([])
            print_state()

    # Shut down env at the end
    og.shutdown()


if __name__ == "__main__":
    main()
