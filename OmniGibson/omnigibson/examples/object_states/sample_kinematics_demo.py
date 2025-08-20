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
    Demo to use the raycasting-based sampler to load objects onTop and/or inside another
    Loads a cabinet, a microwave open on top of it, and two plates with apples on top, one inside and one on top of the cabinet
    Then loads a bookcase and cracker boxes inside of it
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
    )

    cabinet_cfg = dict(
        type="DatasetObject",
        name="cabinet",
        category="bottom_cabinet",
        model="songes",
    )

    plate_cfgs = [
        dict(
            type="DatasetObject",
            name=f"plate{i}",
            category="plate",
            model="iawoof",
            scale=0.6,
            # bounding_box=th.tensor([0.20, 0.20, 0.02]),
        )
        for i in range(2)
    ]

    apple_cfgs = [
        dict(
            type="DatasetObject",
            name=f"apple{i}",
            category="apple",
            model="agveuv",
        )
        for i in range(2)
    ]

    bookcase_cfg = dict(
        type="DatasetObject",
        name="bookcase",
        category="bookcase",
        model="gsksby",
    )

    box_cfgs = [
        dict(
            type="DatasetObject",
            name=f"box{i}",
            category="box_of_crackers",
            model="cmdigf",
        )
        for i in range(5)
    ]

    # Compose objects cfg
    objects_cfg = [
        microwave_cfg,
        cabinet_cfg,
        *plate_cfgs,
        *apple_cfgs,
        bookcase_cfg,
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
    sample_boxes_on_bookcase(env)
    sample_microwave_plates_apples(env)

    max_steps = 100 if short_exec else -1
    step = 0
    while step != max_steps:
        env.step(th.empty(0))
        step += 1

    # Always close environment at the end
    og.shutdown()


def sample_microwave_plates_apples(env):
    microwave = env.scene.object_registry("name", "microwave")
    cabinet = env.scene.object_registry("name", "cabinet")
    plates = list(env.scene.object_registry("category", "plate"))
    apples = list(env.scene.object_registry("category", "apple"))

    # Place the cabinet at a pre-determined location on the floor
    og.log.info("Placing cabinet on the floor...")
    env.step(th.empty(0))
    offset = cabinet.get_position_orientation()[0][2] - cabinet.aabb_center[2]
    cabinet.set_position_orientation(
        position=th.tensor([0.5, 0, cabinet.aabb_extent[2] / 2]) + offset,
        orientation=T.euler2quat(th.tensor([0.0, 0.0, -math.pi / 2.0])),
    )
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


def sample_boxes_on_bookcase(env):
    bookcase = env.scene.object_registry("name", "bookcase")
    boxes = list(env.scene.object_registry("category", "box_of_crackers"))
    # Place the bookcase at a pre-determined location on the floor
    og.log.info("Placing bookcase on the floor...")
    env.step(th.empty(0))
    offset = bookcase.get_position_orientation()[0][2] - bookcase.aabb_center[2]
    bookcase.set_position_orientation(
        position=th.tensor([-0.5, 0, bookcase.aabb_extent[2] / 2]) + offset,
        orientation=T.euler2quat(th.tensor([0.0, 0.0, -math.pi / 2.0])),
    )
    env.step(th.empty(0))  # One step is needed for the object to be fully initialized

    og.log.info("bookcase placed.")
    for _ in range(50):
        env.step(th.empty(0))

    og.log.info("Placing boxes...")
    for i, box in enumerate(boxes):
        box.states[object_states.Inside].set_value(bookcase, True)
        og.log.info(f"Box {i} placed.")

        for _ in range(50):
            env.step(th.empty(0))


if __name__ == "__main__":
    main()
