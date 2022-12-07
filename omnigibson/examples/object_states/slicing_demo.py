import logging
import numpy as np

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.objects import DatasetObject, LightObject
import omnigibson.utils.transform_utils as T
from omnigibson.utils.ui_utils import disclaimer


def main(random_selection=False, headless=False, short_exec=False):
    """
    Demo to use the raycasting-based sampler to load objects onTop and/or inside another
    Loads a cabinet, a microwave open on top of it, and two plates with apples on top, one inside and one on top of the cabinet
    Then loads a shelf and cracker boxes inside of it
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    # Make sure object states are enabled
    assert gm.ENABLE_OBJECT_STATES, f"Object states must be enabled in macros.py in order to use this demo!"
    assert gm.ENABLE_GLOBAL_CONTACT_REPORTING, f"Global contact reporting must be enabled in macros.py in order to use this demo!"
    assert gm.ENABLE_TRANSITION_RULES, f"Transition rules must be enabled in macros.py in order to use this demo!"
    assert not gm.ENABLE_OMNI_PARTICLES, f"Cannot use GPU dynamics with slicing demo due to an NVIDIA bug!"
    disclaimer(f"We are attempting to showcase Slicer / Sliceable object states, which requires deleting objects from "
               f"the simulator at runtime.\n"
               f"Currently, Omniverse has a bug when using GPU dynamics where a segfault will occur if an object in "
               f"contact with another object is attempted to be removed.\n"
               f"This bug should be fixed by the next Omniverse release.\n")

    # Create the scene config to load -- empty scene
    cfg = {
        "scene": {
            "type": "EmptyScene",
        }
    }

    # Create the environment
    env = og.Environment(configs=cfg, action_timestep=1/60., physics_timestep=1/60.)
    env.step(np.array([]))

    # Update the simulator's viewer camera's pose so it points towards the table
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([ 0.544888, -0.412084,  1.11569 ]),
        orientation=np.array([0.54757518, 0.27792802, 0.35721896, 0.70378409]),
    )

    # Create a table, knife, and apple
    table = DatasetObject(
        prim_path="/World/table",
        name="table",
        category="breakfast_table",
        model="19203",
        scale=0.9,
    )
    og.sim.import_object(table)
    table.set_position([0, 0, 0.598])
    env.step(np.array([]))

    apple = DatasetObject(
        prim_path="/World/apple",
        name="apple",
        category="apple",
        model="00_0",
        scale=1.5,
    )
    og.sim.import_object(apple)
    apple_pos = np.array([0.085, 0,  0.90])
    apple.set_position(apple_pos)

    # Let apple settle
    for _ in range(50):
        env.step(np.array([]))

    knife = DatasetObject(
        prim_path="/World/knife",
        name="knife",
        category="table_knife",
        model="1",
        scale=2.5,
    )
    og.sim.import_object(knife)
    knife.set_position_orientation(
        position=apple_pos + np.array([-0.15, 0, 0.2]),
        orientation=T.euler2quat([-np.pi / 2, 0, 0]),
    )
    env.step(np.array([]))

    # Create lights
    light0 = LightObject(
        prim_path="/World/light0",
        name="light0",
        light_type="Sphere",
        radius=0.01,
        intensity=4000.0,
    )
    og.sim.import_object(light0)
    light0.set_position(np.array([1.217, -0.848, 1.388]))

    light1 = LightObject(
        prim_path="/World/light1",
        name="light1",
        light_type="Sphere",
        radius=0.01,
        intensity=4000.0,
    )
    og.sim.import_object(light1)
    light1.set_position(np.array([-1.217, 0.848, 1.388]))

    for _ in range(10):
        env.step(np.array([]))

    input("The knife will fall on the apple and slice it. Press [ENTER] to continue.")

    # Step simulation for a bit so that apple is sliced
    for i in range(500):
        env.step(np.array([]))

    input("Apple has been sliced! Press [ENTER] to terminate the demo.")

    # Always close environment at the end
    env.close()


if __name__ == "__main__":
    main()
