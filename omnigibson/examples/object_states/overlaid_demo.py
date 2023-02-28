from omnigibson.utils.constants import PrimType
from omnigibson.object_states import Overlaid
from omnigibson.macros import gm

import omnigibson as og

# Make sure object states and GPU dynamics are enabled (GPU dynamics needed for cloth)
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True


def main(random_selection=False, headless=False, short_exec=False):
    """
    Demo of cloth objects that can be overlaid on rigid objects.

    Loads a carpet on top of a table. Initially Overlaid will be True because the carpet largely covers the table.
    If you drag the carpet off the table or even just fold it into half, Overlaid will become False.
    """
    og.log.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

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
                "model": "carpet_0",
                "prim_type": PrimType.CLOTH,
                "abilities": {"foldable": {}},
                "position": [0, 0, 1.0],
                "scale": [1.5] * 3,
            },
            {
                "type": "DatasetObject",
                "name": "breakfast_table",
                "category": "breakfast_table",
                "model": "19203",
                "prim_type": PrimType.RIGID,
                "position": [0, 0, 0.6],
            },
        ],
    }

    # Create the environment
    env = og.Environment(configs=cfg, action_timestep=1 / 60., physics_timestep=1 / 60.)

    # Grab object references
    carpet = env.scene.object_registry("name", "carpet")
    breakfast_table = env.scene.object_registry("name", "breakfast_table")

    max_steps = 100 if short_exec else -1
    steps = 0

    while steps != max_steps:
        print(f"Overlaid {carpet.states[Overlaid].get_value(breakfast_table)}")
        og.sim.step()

    # Shut down env at the end
    env.close()


if __name__ == "__main__":
    main()
