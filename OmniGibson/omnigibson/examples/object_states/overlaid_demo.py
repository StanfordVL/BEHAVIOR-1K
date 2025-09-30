import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.object_states import Overlaid
from omnigibson.utils.constants import PrimType
from omnigibson.utils.ui_utils import KeyboardEventHandler

# Make sure object states and GPU dynamics are enabled (GPU dynamics needed for cloth)
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True


def main(random_selection=False, headless=False, short_exec=False):
    """
    Demo of cloth objects that can be overlaid on rigid objects.

    Loads a carpet on top of a table. Initially Overlaid will be True because the carpet largely covers the table.
    If you drag the carpet off the table or even just fold it into half, Overlaid will become False.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Create the scene config to load -- empty scene + custom cloth object + custom rigid object
    cfg = {
        "scene": {
            "type": "Scene",
        },
        "objects": [
            {
                "type": "DatasetObject",
                "name": "carpet",
                "category": "carpet",
                "model": "vlccww",
                "scale": 0.5,
                "prim_type": PrimType.CLOTH,
                "abilities": {"cloth": {}},
                "position": [0, 0, 1.0],
            },
            {
                "type": "DatasetObject",
                "name": "breakfast_table",
                "category": "breakfast_table",
                "model": "rjgmmy",
                "bounding_box": [1.36, 1.081, 0.84],
                "prim_type": PrimType.RIGID,
                "position": [0, 0, 0.58],
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
    breakfast_table = env.scene.object_registry("name", "breakfast_table")

    # Set camera pose
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([0.88215526, -1.40086216, 2.00311063]),
        orientation=th.tensor([0.42013364, 0.12342107, 0.25339685, 0.86258043]),
    )

    max_steps = 100 if short_exec else -1
    steps = 0

    print("\nTry dragging cloth around with CTRL + Left-Click to see the Overlaid state change:\n")

    while steps != max_steps:
        print(f"Overlaid {carpet.states[Overlaid].get_value(breakfast_table)}    ", end="\r")
        env.step([])
        steps += 1

    # Shut down env at the end
    og.shutdown()


if __name__ == "__main__":
    main()
