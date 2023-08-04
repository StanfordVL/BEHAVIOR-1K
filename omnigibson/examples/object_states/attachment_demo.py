import yaml
import numpy as np

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
    obj_cfgs.append(dict(
        type="LightObject",
        name="light",
        light_type="Sphere",
        radius=0.01,
        intensity=5000,
        position=[0, 0, 1.0],
    ))

    base_z = 0.2
    delta_z = 0.01

    idx = 0
    obj_cfgs.append(dict(
        type="DatasetObject",
        name="shelf_back_panel",
        category="shelf_back_panel",
        model="gjsnrt",
        bounding_box=[0.8, 2.02, 0.02],
        position=[0, 0, 0.01],
        abilities={"attachable": {}},
    ))
    idx += 1

    xs = [-0.4, 0.4]
    for i in range(2):
        obj_cfgs.append(dict(
            type="DatasetObject",
            name=f"shelf_side_{i}",
            category="shelf_side",
            model="bxfkjj",
            bounding_box=[0.03, 2.02, 0.26],
            position=[xs[i], 0, base_z + delta_z * idx],
            abilities={"attachable": {}},
        ))
        idx += 1

    ys = [-0.93, -0.61, -0.29, 0.03, 0.35, 0.68]
    for i in range(6):
        obj_cfgs.append(dict(
            type="DatasetObject",
            name=f"shelf_shelf_{i}",
            category="shelf_shelf",
            model="ymtnqa",
            bounding_box=[0.74, 0.023, 0.26],
            position=[0, ys[i], base_z + delta_z * idx],
            abilities={"attachable": {}},
        ))
        idx += 1

    obj_cfgs.append(dict(
        type="DatasetObject",
        name="shelf_top_0",
        category="shelf_top",
        model="pfiole",
        bounding_box=[0.74, 0.04, 0.26],
        position=[0, 1.0, base_z + delta_z * idx],
        abilities={"attachable": {}},
    ))
    idx += 1

    obj_cfgs.append(dict(
        type="DatasetObject",
        name=f"shelf_baseboard",
        category="shelf_baseboard",
        model="hlhneo",
        bounding_box=[0.742, 0.067, 0.02],
        position=[0, -0.97884506, base_z + delta_z * idx],
        abilities={"attachable": {}},
    ))
    idx += 1

    cfg["objects"] = obj_cfgs

    env = og.Environment(configs=cfg)

    # Set viewer camera pose
    og.sim.viewer_camera.set_position_orientation(
        position=np.array([-1.689292, -2.11718198, 0.93332228]),
        orientation=np.array([0.57687967, -0.22995655, -0.29022759, 0.72807814]),
    )

    for _ in range(10):
        env.step([])

    shelf_baseboard = og.sim.scene.object_registry("name", "shelf_baseboard")
    shelf_baseboard.set_position_orientation([0, -0.979, 0.26], [0, 0, 0, 1])
    shelf_baseboard.keep_still()
    shelf_baseboard.set_linear_velocity([-0.2, 0, 0])

    input("\n\nShelf parts fall to their correct poses and get automatically attached to the back panel.\n"
          "You can try to drag the shelf to hit the floor to break it apart. Press [ENTER] to continue.\n")

    for _ in range(1000):
        og.sim.step()

    og.shutdown()

if __name__ == "__main__":
    main()
