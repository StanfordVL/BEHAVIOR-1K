import os

import torch as th

import omnigibson as og
from omnigibson import object_states
from omnigibson.macros import gm
from omnigibson.objects import DatasetObject

# Make sure object states are enabled
gm.ENABLE_OBJECT_STATES = True


def main(random_selection=False, headless=False, short_exec=False):
    """
    Demo to use the raycasting-based sampler to load objects onTop and/or inside another
    Loads a cabinet, a microwave open on top of it, and two plates with apples on top, one inside and one on top of the cabinet
    Then loads a shelf and cracker boxes inside of it
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Create the scene config to load -- empty scene
    cfg = {
        "scene": {
            "type": "Scene",
        },
    }

    # Define objects we want to sample at runtime
    microwave_cfg = dict(
        type="DatasetObject",
        name="microwave",
        category="microwave",
        model="hjjxmi",
        bounding_box=[0.768, 0.512, 0.392],
    )

    cabinet_cfg = dict(
        type="DatasetObject",
        name="cabinet",
        category="bottom_cabinet",
        model="bamfsz",
        bounding_box=[1.075, 1.131, 1.355],
    )

    plate_cfgs = [
        dict(
            type="DatasetObject",
            name=f"plate{i}",
            category="plate",
            model="iawoof",
            bounding_box=th.tensor([0.20, 0.20, 0.05]),
        )
        for i in range(2)
    ]

    apple_cfgs = [
        dict(
            type="DatasetObject",
            name=f"apple{i}",
            category="apple",
            model="agveuv",
            bounding_box=[0.065, 0.065, 0.077],
        )
        for i in range(4)
    ]

    shelf_cfg = dict(
        type="DatasetObject",
        name=f"shelf",
        category="shelf",
        model="pkgbcp",
        bounding_box=th.tensor([1.0, 0.4, 2.0]),
    )

    box_cfgs = [
        dict(
            type="DatasetObject",
            name=f"box{i}",
            category="box_of_crackers",
            model="cmdigf",
            bounding_box=th.tensor([0.2, 0.05, 0.3]),
        )
        for i in range(5)
    ]

    # Compose objects cfg
    objects_cfg = [
        microwave_cfg,
        cabinet_cfg,
        *plate_cfgs,
        *apple_cfgs,
        shelf_cfg,
        *box_cfgs,
    ]

    # Update their spawn positions so they don't collide immediately
    for i, obj_cfg in enumerate(objects_cfg):
        obj_cfg["position"] = [100 + i, 100 + i, 100 + i]

    cfg["objects"] = objects_cfg

    # Create the environment
    env = og.Environment(configs=cfg)
    env.step([])

    # Sample microwave and boxes
    sample_boxes_on_shelf(env)
    sample_microwave_plates_apples(env)

    max_steps = 100 if short_exec else -1
    step = 0
    while step != max_steps:
        env.step(th.empty(0))
        step += 1

    # Always close environment at the end
    og.clear()


def sample_microwave_plates_apples(env):
    microwave = env.scene.object_registry("name", "microwave")
    cabinet = env.scene.object_registry("name", "cabinet")
    plates = list(env.scene.object_registry("category", "plate"))
    apples = list(env.scene.object_registry("category", "apple"))

    # Place the cabinet at a pre-determined location on the floor
    og.log.info("Placing cabinet on the floor...")
    cabinet.set_orientation([0, 0, 0, 1.0])
    env.step(th.empty(0))
    offset = cabinet.get_position_orientation()[0][2] - cabinet.aabb_center[2]
    cabinet.set_position_orientation(position=th.tensor([1.0, 0, cabinet.aabb_extent[2] / 2]) + offset)
    env.step(th.empty(0))

    # Set microwave on top of the cabinet, open it, and step 100 times
    og.log.info("Placing microwave OnTop of the cabinet...")
    assert microwave.states[object_states.OnTop].set_value(cabinet, True)
    assert microwave.states[object_states.Open].set_value(True)
    og.log.info("Microwave placed.")
    for _ in range(50):
        env.step(th.empty(0))

    og.log.info("Placing plates")
    n_apples_per_plate = int(len(apples) / len(plates))
    for i, plate in enumerate(plates):
        # Put the 1st plate in the microwave
        if i == 0:
            og.log.info(f"Placing plate {i} Inside the microwave...")
            assert plate.states[object_states.Inside].set_value(microwave, True)
        else:
            og.log.info(f"Placing plate {i} OnTop the microwave...")
            assert plate.states[object_states.OnTop].set_value(microwave, True)

        og.log.info(f"Plate {i} placed.")
        for _ in range(50):
            env.step(th.empty(0))

        og.log.info(f"Placing {n_apples_per_plate} apples OnTop of the plate...")
        for j in range(n_apples_per_plate):
            apple_idx = i * n_apples_per_plate + j
            apple = apples[apple_idx]
            assert apple.states[object_states.OnTop].set_value(plate, True)
            og.log.info(f"Apple {apple_idx} placed.")
            for _ in range(50):
                env.step(th.empty(0))


def sample_boxes_on_shelf(env):
    shelf = env.scene.object_registry("name", "shelf")
    boxes = list(env.scene.object_registry("category", "box_of_crackers"))
    # Place the shelf at a pre-determined location on the floor
    og.log.info("Placing shelf on the floor...")
    shelf.set_orientation([0, 0, 0, 1.0])
    env.step(th.empty(0))
    offset = shelf.get_position_orientation()[0][2] - shelf.aabb_center[2]
    shelf.set_position_orientation(position=th.tensor([-1.0, 0, shelf.aabb_extent[2] / 2]) + offset)
    env.step(th.empty(0))  # One step is needed for the object to be fully initialized

    og.log.info("Shelf placed.")
    for _ in range(50):
        env.step(th.empty(0))

    og.log.info("Placing boxes...")
    for i, box in enumerate(boxes):
        box.states[object_states.Inside].set_value(shelf, True)
        og.log.info(f"Box {i} placed.")

        for _ in range(50):
            env.step(th.empty(0))


if __name__ == "__main__":
    main()
