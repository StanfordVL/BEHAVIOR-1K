import math

import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm

# Make sure object states and transition rules are enabled
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = True


def main(random_selection=False, headless=False, short_exec=False):
    """
    Demo of slicing an apple into two apple slices
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Create the scene config to load -- empty scene with table, knife, and apple
    table_cfg = dict(
        type="DatasetObject",
        name="table",
        category="breakfast_table",
        model="rjgmmy",
        bounding_box=[1.36, 1.081, 0.84],
        position=[0, 0, 0.58],
    )

    apple_cfg = dict(
        type="DatasetObject",
        name="apple",
        category="apple",
        model="agveuv",
        position=[0.085, 0, 0.92],
    )

    knife_cfg = dict(
        type="DatasetObject",
        name="knife",
        category="table_knife",
        model="lrdmpf",
        bounding_box=[0.401, 0.044, 0.009],
        position=[0, 0, 20.0],
    )

    light0_cfg = dict(
        type="LightObject",
        name="light0",
        light_type="Sphere",
        radius=0.01,
        intensity=4000.0,
        position=[1.217, -0.848, 1.388],
    )

    light1_cfg = dict(
        type="LightObject",
        name="light1",
        light_type="Sphere",
        radius=0.01,
        intensity=4000.0,
        position=[-1.217, 0.848, 1.388],
    )

    cfg = {
        "scene": {
            "type": "Scene",
        },
        "objects": [table_cfg, apple_cfg, knife_cfg, light0_cfg, light1_cfg],
    }

    # Create the environment
    env = og.Environment(configs=cfg)

    # Grab reference to apple and knife
    apple = env.scene.object_registry("name", "apple")
    knife = env.scene.object_registry("name", "knife")

    # Update the simulator's viewer camera's pose so it points towards the table
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([0.544888, -0.412084, 1.11569]),
        orientation=th.tensor([0.54757518, 0.27792802, 0.35721896, 0.70378409]),
    )

    # Let apple settle
    for _ in range(50):
        env.step([])

    knife.keep_still()
    knife.set_position_orientation(
        position=apple.get_position_orientation()[0] + th.tensor([-0.15, 0.0, 0.2], dtype=th.float32),
        orientation=T.euler2quat(th.tensor([-math.pi / 2, 0, 0], dtype=th.float32)),
    )

    if not short_exec:
        input("The knife will fall on the apple and slice it. Press [ENTER] to continue.")

    # Step simulation for a bit so that apple is sliced
    for i in range(1000):
        env.step([])

    if not short_exec:
        input("Apple has been sliced! Press [ENTER] to terminate the demo.")

    # Always close environment at the end
    og.shutdown()


if __name__ == "__main__":
    main()
