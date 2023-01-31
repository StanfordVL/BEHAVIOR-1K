import logging
from collections import OrderedDict
import numpy as np

import omnigibson as og
from omnigibson import object_states
from omnigibson.macros import gm

# Make sure object states are enabled
gm.ENABLE_OBJECT_STATES = True


def main(random_selection=False, headless=False, short_exec=False):
    """
    Demo of on fire state.
    Loads a stove (toggled on), and two apples.
    The first apple will be ignited by the stove first, then the second apple will be ignited by the first apple.
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    # Define specific objects we want to load in with the scene directly
    obj_configs = []

    # Light
    obj_configs.append(OrderedDict(
        type="LightObject",
        light_type="Sphere",
        name="light",
        radius=0.01,
        intensity=1e5,
        position=[-2.0, -2.0, 1.0],
    ))

    # Stove
    obj_configs.append(OrderedDict(
        type="DatasetObject",
        name="stove",
        category="stove",
        model="101943",
        position=[0, 0, 0.65],
    ))

    # 2 Apples
    for i in range(2):
        obj_configs.append(OrderedDict(
            type="DatasetObject",
            name=f"apple{i}",
            category="apple",
            model="00_0",
            position=[0, i * 0.05, 1.65],
            abilities={"flammable": {"ignition_temperature": 100, "distance_threshold": 0.5}},
        ))

    # Create the scene config to load -- empty scene with desired objects
    cfg = {
        "scene": {
            "type": "Scene",
        },
        "objects": obj_configs,
    }

    # Create the environment
    env = og.Environment(configs=cfg, action_timestep=1 / 60., physics_timestep=1 / 60.)

    # Get reference to relevant objects
    stove = env.scene.object_registry("name", "stove")
    apples = list(env.scene.object_registry("category", "apple"))

    # Set camera to appropriate viewing pose
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([ 0.46938863, -3.97887141,  1.64106008]),
        orientation=np.array([0.63311689, 0.00127259, 0.00155577, 0.77405359]),
    )

    # Let objects settle
    for _ in range(25):
        env.step(np.array([]))

    # Turn on the stove
    stove.states[object_states.ToggledOn].set_value(True)

    # The first apple will be affected by the stove
    apples[0].set_position(stove.states[object_states.HeatSourceOrSink].get_link_position() + np.array([-0.15, 0, 0.1]))

    # The second apple will NOT be affected by the stove, but will be affected by the first apple once it's on fire.
    apples[1].set_position(stove.states[object_states.HeatSourceOrSink].get_link_position() + np.array([-0.3, 0, 0.1]))

    steps = 0
    max_steps = -1 if not short_exec else 1000

    # Main recording loop
    while steps != max_steps:
        env.step(np.array([]))
        temps = [f"{apple.states[object_states.Temperature].get_value():>20.2f}" for apple in apples]
        print(f"{'Apple temperature:':<20}", *temps, end="\r")
        steps += 1

    # Always close env at the end
    env.close()


if __name__ == "__main__":
    main()
