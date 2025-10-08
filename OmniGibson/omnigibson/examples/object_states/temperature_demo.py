import math

import torch as th

import omnigibson as og
from omnigibson import object_states
from omnigibson.macros import gm
import omnigibson.utils.transform_utils as T

# Make sure object states are enabled
gm.ENABLE_OBJECT_STATES = True


def main(random_selection=False, headless=False, short_exec=False):
    """
    Demo of temperature change
    Loads a stove, a microwave and an oven, all toggled on, and five frozen apples
    The user can move the apples to see them change from frozen, to normal temperature, to cooked and burnt
    This demo also shows how to load objects ToggledOn and how to set the initial temperature of an object
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Define specific objects we want to load in with the scene directly
    obj_configs = []

    # Light
    obj_configs.append(
        dict(
            type="LightObject",
            light_type="Sphere",
            name="light",
            radius=0.01,
            intensity=1e8,
            position=[-2.0, -2.0, 1.0],
        )
    )

    # Stove
    obj_configs.append(
        dict(
            type="DatasetObject",
            name="stove",
            category="stove",
            model="ykretu",
            position=[0, 0, 0.61],
            orientation=T.euler2quat(th.tensor([0.0, 0.0, -math.pi / 2.0])),
        )
    )

    # Microwave
    obj_configs.append(
        dict(
            type="DatasetObject",
            name="microwave",
            category="microwave",
            model="abzvij",
            position=[2.5, 0, 0.15],
            orientation=T.euler2quat(th.tensor([0.0, 0.0, -math.pi / 2.0])),
        )
    )

    # Oven
    obj_configs.append(
        dict(
            type="DatasetObject",
            name="oven",
            category="oven",
            model="ffitak",
            position=[-1.25, 0, 0.36],
            orientation=T.euler2quat(th.tensor([0.0, 0.0, -math.pi / 2.0])),
        )
    )

    # Tray
    obj_configs.append(
        dict(
            type="DatasetObject",
            name="tray",
            category="tray",
            model="xzcnjq",
            position=[-2.25, -0.12, 0.05],
        )
    )

    # Fridge
    obj_configs.append(
        dict(
            type="DatasetObject",
            name="fridge",
            category="fridge",
            model="petcxr",
            abilities={
                "coldSource": {
                    "temperature": -100.0,
                    "requires_inside": True,
                }
            },
            position=[1.25, 0, 1.0],
            orientation=T.euler2quat(th.tensor([0.0, 0.0, -math.pi / 2.0])),
        )
    )

    # 5 Apples
    for i in range(5):
        obj_configs.append(
            dict(
                type="DatasetObject",
                name=f"apple{i}",
                category="apple",
                model="agveuv",
                position=[0, i * 0.1, 5.0],
            )
        )

    # Create the scene config to load -- empty scene with desired objects
    cfg = {
        "scene": {
            "type": "Scene",
        },
        "objects": obj_configs,
    }

    # Create the environment
    env = og.Environment(configs=cfg)

    # Get reference to relevant objects
    stove = env.scene.object_registry("name", "stove")
    microwave = env.scene.object_registry("name", "microwave")
    oven = env.scene.object_registry("name", "oven")
    tray = env.scene.object_registry("name", "tray")
    fridge = env.scene.object_registry("name", "fridge")
    apples = list(env.scene.object_registry("category", "apple"))

    # Set camera to appropriate viewing pose
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([0.1759, -4.8260, 1.7778]),
        orientation=th.tensor([0.63311689, 0.00127259, 0.00155577, 0.77405359]),
    )

    # Let objects settle
    for _ in range(25):
        env.step(th.empty(0))

    # Turn on all scene objects
    stove.states[object_states.ToggledOn].set_value(True)
    microwave.states[object_states.ToggledOn].set_value(True)
    oven.states[object_states.ToggledOn].set_value(True)

    # Set initial temperature of the apples to -50 degrees Celsius, and move the apples to different objects
    for apple in apples:
        apple.states[object_states.Temperature].set_value(-50)
    apples[0].states[object_states.Inside].set_value(oven, True)
    apples[1].set_position_orientation(
        position=stove.states[object_states.HeatSourceOrSink].link.get_position_orientation()[0]
        + th.tensor([0, 0, 0.1])
    )
    apples[2].states[object_states.OnTop].set_value(tray, True)
    apples[3].states[object_states.Inside].set_value(fridge, True)
    apples[4].states[object_states.Inside].set_value(microwave, True)

    steps = 0
    max_steps = -1 if not short_exec else 1000

    # Main recording loop
    locations = ["Oven", "Stove", "Tray", "Fridge", "Microwave"]
    print()
    while steps != max_steps:
        env.step(th.empty(0))
        temp_info = [
            f"{loc}: {apple.states[object_states.Temperature].get_value():.1f}°C"
            for loc, apple in zip(locations, apples)
        ]
        print(f"Apple temps: {' | '.join(temp_info):<80}", end="\r")
        steps += 1

    # Always close env at the end
    og.shutdown()


if __name__ == "__main__":
    main()
