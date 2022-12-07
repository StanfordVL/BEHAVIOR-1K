import logging
import numpy as np

import omnigibson as og
from omnigibson import object_states
from omnigibson.macros import gm
from omnigibson.objects import PrimitiveObject
from omnigibson.systems import DustSystem, StainSystem


def main(random_selection=False, headless=False, short_exec=False):
    """
    Demo of a cleaning task
    Loads an interactive scene and sets all object surface to be dirty
    Loads also a cleaning tool that can be soaked in water and used to clean objects if moved manually
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    # Make sure object states are enabled
    assert gm.ENABLE_OBJECT_STATES, f"Object states must be enabled in macros.py in order to use this demo!"

    # Create the scene config to load -- Rs_int with only a few object categories loaded
    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "load_object_categories": ["floors", "walls", "ceilings", "breakfast_table", "bottom_cabinet", "sink", "stove", "fridge", "window"],
        }
    }

    env = og.Environment(configs=cfg, action_timestep=1/60., physics_timestep=1/60.)

    # Load a cleaning tool (a block with the ability to be soaked and is a cleaning tool)
    block = PrimitiveObject(
        prim_path="/World/block",
        name="block",
        primitive_type="Cube",
        scale=[0.15, 0.1, 0.03],
        rgba=[0.5, 1.0, 1.0, 1.0],
        abilities={"soakable": {}, "cleaningTool": {}},
    )
    og.sim.import_object(block)
    block.set_position([-1.4, 3.0, 1.5])

    # Set everything that can go dirty and activate the water sources
    dusty_objects = env.scene.object_registry("category", "breakfast_table")
    stained_objects = env.scene.object_registry("category", "bottom_cabinet")
    water_source_objects = env.scene.get_objects_with_state(object_states.WaterSource)

    for obj in dusty_objects:
        logging.info(f"Setting object {obj.name} to be Dusty")
        obj.states[object_states.Covered].set_value(DustSystem, True)

    for obj in stained_objects:
        logging.info(f"Setting object {obj.name} to be Stained")
        obj.states[object_states.Covered].set_value(StainSystem, True)

    for obj in water_source_objects:
        if object_states.ToggledOn in obj.states:
            logging.info(f"Setting water source object {obj} to be ToggledOn")
            obj.states[object_states.ToggledOn].set_value(True)

    # Set the camera to be in a good position
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([-0.825556,  2.42499 ,  1.04104 ]),
        orientation=np.array([0.56919735, 0.09896035, 0.13981109, 0.80416049]),
    )

    max_steps = -1 if not short_exec else 1000
    step = 0
    try:
        for i in range(200):
            env.step(np.array([]))
        while step != max_steps:
            env.step(np.array([]))      # Empty action since no robots in the environment
            step += 1
    finally:
        # Always close environment at the end
        env.close()


if __name__ == "__main__":
    main()
