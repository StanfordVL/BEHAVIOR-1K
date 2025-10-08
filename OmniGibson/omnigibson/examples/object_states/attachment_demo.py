import math

import torch as th
import yaml

import omnigibson as og
from omnigibson.macros import gm, macros
import omnigibson.utils.transform_utils as T

# Make sure object states are enabled
gm.ENABLE_OBJECT_STATES = True


def main(random_selection=False, headless=False, short_exec=False):
    """
    Demo of attachment of different parts of a bookcase
    """
    cfg = yaml.load(open(f"{og.example_config_path}/default_cfg.yaml", "r"), Loader=yaml.FullLoader)
    # Add objects that we want to create
    obj_cfgs = []

    base_z = 0.2

    obj_cfgs.append(
        dict(
            type="DatasetObject",
            name="bookcase_back_panel",
            category="bookcase_back",
            model="gjsnrt",
            position=[0, 0, 0.01],
            abilities={"attachable": {}},
        )
    )

    ys = [-0.93, -0.61, -0.29, 0.03, 0.35, 0.68]
    for i in range(6):
        obj_cfgs.append(
            dict(
                type="DatasetObject",
                name=f"bookcase_shelf_{i}",
                category="bookcase_shelf",
                model="ymtnqa",
                position=[0, ys[i], base_z],
                orientation=T.euler2quat(th.tensor([0.0, -math.pi / 2.0, math.pi / 2.0])),
                abilities={"attachable": {}},
            )
        )

    obj_cfgs.append(
        dict(
            type="DatasetObject",
            name="bookcase_top",
            category="bookcase_top",
            model="pfiole",
            position=[0, 1.0, base_z],
            orientation=T.euler2quat(th.tensor([-math.pi / 2.0, math.pi / 2.0, -math.pi])),
            abilities={"attachable": {}},
        )
    )

    cfg["objects"] = obj_cfgs

    env = og.Environment(configs=cfg)

    # Lower the break force to make it easier to break apart the bookcase
    with macros.unlocked():
        macros.object_states.attached_to.DEFAULT_BREAK_FORCE = 500.0
        macros.object_states.attached_to.DEFAULT_BREAK_TORQUE = 500.0

    # Set viewer camera pose
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([-1.689292, -2.11718198, 0.93332228]),
        orientation=th.tensor([0.57687967, -0.22995655, -0.29022759, 0.72807814]),
    )

    for _ in range(10):
        env.step([])

    input(
        "\n\nbookcase parts fall to their correct poses and get automatically attached to the back panel.\n"
        "You can try to drag (Shift + Left-CLICK + Drag) parts of the bookcase to break it apart (you may need to zoom out and drag with a larger force).\n"
        "Press [ENTER] to continue.\n"
    )

    steps = 0
    max_steps = -1 if not short_exec else 1000

    while steps != max_steps:
        env.step([])

    og.shutdown()


if __name__ == "__main__":
    main()
