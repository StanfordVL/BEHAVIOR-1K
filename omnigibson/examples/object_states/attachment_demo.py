import torch as th
import yaml

import omnigibson as og
from omnigibson.macros import gm

# Make sure object states are enabled
gm.ENABLE_OBJECT_STATES = True


def main(random_selection=False, headless=False, short_exec=False):
    """
    Demo of attachment of different parts of a shelf
    """
    cfg = yaml.load(open(f"{og.example_config_path}/default_cfg.yaml", "r"), Loader=yaml.FullLoader)
    # Add objects that we want to create
    obj_cfgs = []
    obj_cfgs.append(
        dict(
            type="LightObject",
            name="light",
            light_type="Sphere",
            radius=0.01,
            intensity=5000,
            position=[0, 0, 1.0],
        )
    )

    base_z = 0.2
    delta_z = 0.01

    idx = 0
    obj_cfgs.append(
        dict(
            type="DatasetObject",
            name="shelf_back_panel",
            category="shelf_back_panel",
            model="gjsnrt",
            position=[0, 0, 0.01],
            abilities={"attachable": {}},
        )
    )
    idx += 1

    obj_cfgs.append(
        dict(
            type="DatasetObject",
            name=f"shelf_side_left",
            category="shelf_side",
            model="bxfkjj",
            position=[-0.4, 0, base_z + delta_z * idx],
            abilities={"attachable": {}},
        )
    )
    idx += 1

    obj_cfgs.append(
        dict(
            type="DatasetObject",
            name=f"shelf_side_right",
            category="shelf_side",
            model="yujrmw",
            position=[0.4, 0, base_z + delta_z * idx],
            abilities={"attachable": {}},
        )
    )
    idx += 1

    ys = [-0.93, -0.61, -0.29, 0.03, 0.35, 0.68]
    for i in range(6):
        obj_cfgs.append(
            dict(
                type="DatasetObject",
                name=f"shelf_shelf_{i}",
                category="shelf_shelf",
                model="ymtnqa",
                position=[0, ys[i], base_z + delta_z * idx],
                abilities={"attachable": {}},
            )
        )
        idx += 1

    obj_cfgs.append(
        dict(
            type="DatasetObject",
            name="shelf_top_0",
            category="shelf_top",
            model="pfiole",
            position=[0, 1.0, base_z + delta_z * idx],
            abilities={"attachable": {}},
        )
    )
    idx += 1

    obj_cfgs.append(
        dict(
            type="DatasetObject",
            name=f"shelf_baseboard",
            category="shelf_baseboard",
            model="hlhneo",
            position=[0, -10.97884506, base_z + delta_z * idx],
            abilities={"attachable": {}},
        )
    )
    idx += 1

    cfg["objects"] = obj_cfgs

    env = og.Environment(configs=cfg)

    # Set viewer camera pose
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([-1.689292, -2.11718198, 0.93332228]),
        orientation=th.tensor([0.57687967, -0.22995655, -0.29022759, 0.72807814]),
    )

    for _ in range(10):
        env.step([])

    shelf_baseboard = env.scene.object_registry("name", "shelf_baseboard")
    shelf_baseboard.set_position_orientation(position=[0, -0.979, 0.21], orientation=[0, 0, 0, 1])
    shelf_baseboard.keep_still()
    # Lower the mass of the baseboard - otherwise, the gravity will create enough torque to break the joint
    shelf_baseboard.root_link.mass = 0.1

    input(
        "\n\nShelf parts fall to their correct poses and get automatically attached to the back panel.\n"
        "You can try to drag (Shift + Left-CLICK + Drag) parts of the shelf to break it apart (you may need to zoom out and drag with a larger force).\n"
        "Press [ENTER] to continue.\n"
    )

    for _ in range(5000):
        og.sim.step()

    og.shutdown()


if __name__ == "__main__":
    main()
