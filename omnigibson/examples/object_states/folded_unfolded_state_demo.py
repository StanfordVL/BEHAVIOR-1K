from omnigibson.utils.constants import PrimType
from omnigibson.object_states import Folded, Unfolded
from omnigibson.macros import gm
import logging

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

    max_steps = 100 if short_exec else -1
    steps = 0

    while steps != max_steps:
        og.sim.step()

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
        steps += 1

    # Shut down env at the end
    env.close()


if __name__ == "__main__":
    main()
