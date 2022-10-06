import logging

import numpy as np

import igibson as ig
from igibson import object_states
from igibson.macros import gm
from igibson.objects import DatasetObject, LightObject


def main(random_selection=False, headless=False, short_exec=False):
    """
    Demo of temperature change
    Loads a stove, a microwave and an oven, all toggled on, and five frozen apples
    The user can move the apples to see them change from frozen, to normal temperature, to cooked and burnt
    This demo also shows how to load objects ToggledOn and how to set the initial temperature of an object
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    # Make sure object states are enabled
    assert gm.ENABLE_OBJECT_STATES, f"Object states must be enabled in macros.py in order to use this demo!"

    # Create the scene config to load -- empty scene
    cfg = {
        "scene": {
            "type": "EmptyScene",
            "floor_plane_visible": True,
        }
    }

    # Create the environment
    env = ig.Environment(configs=cfg, action_timestep=1 / 60., physics_timestep=1 / 60.)

    # Set camera to appropriate viewing pose
    ig.sim.viewer_camera.set_position_orientation(
        position=np.array([ 0.46938863, -3.97887141,  1.64106008]),
        orientation=np.array([0.63311689, 0.00127259, 0.00155577, 0.77405359]),
    )

    # Create a light object
    light = LightObject(
        prim_path="/World/sphere_light",
        light_type="Sphere",
        name="sphere_light",
        radius=0.01,
        intensity=1e5,
    )
    ig.sim.import_object(light)
    light.set_position(np.array([-2.0, -2.0, 1.0]))

    # Load stove ON
    stove = DatasetObject(
        prim_path="/World/stove",
        name="stove",
        category="stove",
        model="101943",
    )
    ig.sim.import_object(stove)
    stove.set_position([0, 0, 0.65])

    # Load microwave ON
    microwave = DatasetObject(
        prim_path="/World/microwave",
        name="microwave",
        category="microwave",
        model="7128",
        scale=0.25,
    )
    ig.sim.import_object(microwave)
    microwave.set_position([2.5, 0, 0.094])

    # Load oven ON
    oven = DatasetObject(
        prim_path="/World/oven",
        name="oven",
        category="oven",
        model="7120",
    )
    ig.sim.import_object(oven)
    oven.set_position([-1.25, 0, 0.80])

    # Load tray
    tray = DatasetObject(
        prim_path="/World/tray",
        name="tray",
        category="tray",
        model="tray_000",
        scale=0.15,
    )
    ig.sim.import_object(tray)
    tray.set_position([0, 0, 1.24])

    # Load fridge
    fridge = DatasetObject(
        prim_path="/World/fridge",
        name="fridge",
        category="fridge",
        model="12252",
        abilities={
            "coldSource": {
                "temperature": -100.0,
                "requires_inside": True,
            }
        },
    )
    ig.sim.import_object(fridge)
    fridge.set_position([1.25, 0, 0.80])

    # Load 5 apples
    apples = []
    for i in range(5):
        apple = DatasetObject(
            prim_path=f"/World/apple{i}",
            name=f"apple{i}",
            category="apple",
            model="00_0",
        )
        ig.sim.import_object(apple)
        apple.set_position([0, i * 0.05, 1.65])
        apples.append(apple)

    # Take an environment step so that all objects are initialized properly
    env.step(np.array([]))

    # Turn on all scene objects
    stove.states[object_states.ToggledOn].set_value(True)
    microwave.states[object_states.ToggledOn].set_value(True)
    oven.states[object_states.ToggledOn].set_value(True)

    # Set initial temperature of the apples to -50 degrees Celsius, and move the apples to different objects
    for apple in apples:
        apple.states[object_states.Temperature].set_value(-50)
    apples[0].states[object_states.Inside].set_value(oven, True, use_ray_casting_method=True)
    apples[1].set_position(stove.links["heat_source_link"].get_position() + np.array([0, 0, 0.1]))
    apples[2].states[object_states.OnTop].set_value(tray, True, use_ray_casting_method=True)
    apples[3].states[object_states.Inside].set_value(fridge, True, use_ray_casting_method=True)
    apples[4].states[object_states.Inside].set_value(microwave, True, use_ray_casting_method=True)

    steps = 0
    max_steps = -1 if not short_exec else 1000

    # Main recording loop
    locations = [f'{loc:>20}' for loc in ["Inside oven", "On stove", "On tray", "Inside fridge", "Inside microwave"]]
    print()
    print(f"{'Apple location:':<20}", *locations)
    while steps != max_steps:
        env.step(np.array([]))
        temps = [f"{apple.states[object_states.Temperature].get_value():>20.2f}" for apple in apples]
        print(f"{'Apple temperature:':<20}", *temps, end="\r")
        steps += 1

    # Always close env at the end
    env.close()


if __name__ == "__main__":
    main()
