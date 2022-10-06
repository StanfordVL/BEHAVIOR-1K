import logging
import os

import numpy as np

import igibson as ig
from igibson import object_states
from igibson.macros import gm
from igibson.objects import DatasetObject


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

    # Create the scene config to load -- empty scene
    cfg = {
        "scene": {
            "type": "EmptyScene",
        }
    }

    # Create the environment
    env = ig.Environment(configs=cfg, action_timestep=1/60., physics_timestep=1/60.)

    # Sample microwave and boxes
    sample_boxes_on_shelf(env)
    sample_microwave_plates_apples(env)

    max_steps = 100 if short_exec else -1
    step = 0
    while step != max_steps:
        env.step(np.array([]))
        step += 1

    # Always close environment at the end
    env.close()


def sample_microwave_plates_apples(env):
    # Load cabinet, set position manually, and step 100 times
    logging.info("Loading cabinet")
    cabinet = DatasetObject(
        prim_path="/World/cabinet",
        name="cabinet",
        category="bottom_cabinet",
        model="46380",
        # scale=2.0,
    )
    ig.sim.import_object(cabinet)
    z_offset = -cabinet.aabb_center[2] + cabinet.aabb_extent[2] / 2
    cabinet.set_position(np.array([1.0, 0, z_offset]))
    env.step(np.array([]))              # One step is needed for the object to be fully initialized
    for _ in range(50):
        env.step(np.array([]))

    # Load microwave, set position on top of the cabinet, open it, and step 100 times
    logging.info("Loading microwave Open and OnTop of the cabinet")
    microwave = DatasetObject(
        prim_path="/World/microwave",
        name="microwave",
        category="microwave",
        model="7128",
        scale=0.5,
    )
    ig.sim.import_object(microwave)
    env.step(np.array([]))              # One step is needed for the object to be fully initialized
    assert microwave.states[object_states.OnTop].set_value(cabinet, True, use_ray_casting_method=True)
    assert microwave.states[object_states.Open].set_value(True)
    logging.info("Microwave loaded and placed")
    for _ in range(50):
        env.step(np.array([]))

    logging.info("Loading plates")
    n_plates = 3
    for i in range(n_plates):
        plate = DatasetObject(
            prim_path=f"/World/plate{i}",
            name=f"plate{i}",
            category="plate",
            model="plate_000",
            bounding_box=np.array([0.25, 0.25, 0.05]),
        )
        ig.sim.import_object(plate)
        env.step(np.array([]))              # One step is needed for the object to be fully initialized

        # Put the 1st plate in the microwave
        if i == 0:
            logging.info("Loading plate Inside the microwave")
            assert plate.states[object_states.Inside].set_value(microwave, True, use_ray_casting_method=True)
        else:
            logging.info("Loading plate OnTop the microwave")
            assert plate.states[object_states.OnTop].set_value(microwave, True, use_ray_casting_method=True)

        logging.info("Plate %d loaded and placed." % i)
        for _ in range(50):
            env.step(np.array([]))

        logging.info("Loading three apples OnTop of the plate")
        for j in range(3):
            apple = DatasetObject(
                prim_path=f"/World/apple{i * n_plates + j}",
                name=f"apple{i * n_plates + j}",
                category="apple",
                model="00_0",
            )
            ig.sim.import_object(apple)
            env.step(np.array([]))  # One step is needed for the object to be fully initialized
            assert apple.states[object_states.OnTop].set_value(plate, True, use_ray_casting_method=True)
            logging.info("Apple %d loaded and placed." % j)
            for _ in range(50):
                env.step(np.array([]))


def sample_boxes_on_shelf(env):
    shelf = DatasetObject(
        prim_path=f"/World/shelf",
        name=f"shelf",
        category="shelf",
        model="1170df5b9512c1d92f6bce2b7e6c12b7",
        bounding_box=np.array([1.0, 0.4, 2.0]),
    )
    ig.sim.import_object(shelf)
    z_offset = -shelf.aabb_center[2] + shelf.aabb_extent[2] / 2
    shelf.set_position(np.array([-1.0, 0, z_offset]))
    env.step(np.array([]))  # One step is needed for the object to be fully initialized

    logging.info("Shelf placed")
    for _ in range(50):
        env.step(np.array([]))

    for i in range(10):
        box = DatasetObject(
            prim_path=f"/World/box{i}",
            name=f"box{i}",
            category="cracker_box",
            model="cracker_box_000",
            bounding_box=np.array([0.2, 0.05, 0.3]),
        )
        ig.sim.import_object(box)
        env.step(np.array([]))  # One step is needed for the object to be fully initialized
        box.states[object_states.Inside].set_value(shelf, True, use_ray_casting_method=True)
        logging.info(f"Box {i} placed.")

        for _ in range(50):
            env.step(np.array([]))


if __name__ == "__main__":
    main()
