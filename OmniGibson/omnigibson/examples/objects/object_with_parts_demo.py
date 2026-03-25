"""
Demo of add_object_with_parts: spawning objects along with their annotated
connectedpart (cap attached to a bottle) and extrapart (pillow placed on a bed).
"""

import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.object_states.attached_to import AttachedTo
from omnigibson.utils.object_utils import add_object_with_parts

gm.ENABLE_OBJECT_STATES = True


def main(random_selection=False, headless=False, short_exec=False):
    """
    Demo showing how add_object_with_parts spawns a main object together with
    its connectedpart and extrapart children defined in metadata.
    """
    cfg = {
        "scene": {"type": "Scene"},
        "objects": [
            dict(
                type="DatasetObject",
                name="table",
                category="breakfast_table",
                model="rjgmmy",
                bounding_box=[1.36, 1.081, 0.84],
                position=[0, 0, 0.42],
            ),
            dict(
                type="LightObject",
                name="light0",
                light_type="Sphere",
                radius=0.01,
                intensity=4000.0,
                position=[1.2, -0.8, 1.5],
            ),
            dict(
                type="LightObject",
                name="light1",
                light_type="Sphere",
                radius=0.01,
                intensity=4000.0,
                position=[-1.2, 0.8, 1.5],
            ),
        ],
    }

    env = og.Environment(configs=cfg)
    scene = env.scene

    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([0.0, -2.5, 1.8]),
        orientation=T.euler2quat(th.tensor([th.pi / 6, 0.0, 0.0])),
    )

    # --- connectedpart example: vodka bottle + cap ---
    bottle = DatasetObject(
        name="vodka_bottle",
        category="vodka_bottle",
        model="bojwlu",
        abilities={"attachable": {}},
    )
    bottle_objs = add_object_with_parts(
        scene=scene,
        obj=bottle,
        pos=th.tensor([0.0, 0.0, 1.2]),  # on the table top
        orn=th.tensor([0.0, 0.0, 0.0, 1.0]),
    )

    cap = next((o for o in bottle_objs if o is not bottle), None)
    print(f"\nBottle + connectedpart spawned: {[o.name for o in bottle_objs]}")
    if cap is not None and AttachedTo in cap.states:
        print(f"  Cap '{cap.name}' attached to bottle: {cap.states[AttachedTo].get_value(bottle)}")

    # --- extrapart example: bed + pillow ---
    bed = DatasetObject(
        name="bed",
        category="bed",
        model="zrumze",
    )
    bed_objs = add_object_with_parts(
        scene=scene,
        obj=bed,
        pos=th.tensor([3.0, 0.0, 0.3]),
        orn=th.tensor([0.0, 0.0, 0.0, 1.0]),
    )

    pillow = next((o for o in bed_objs if o is not bed), None)
    print(f"\nBed + extrapart spawned: {[o.name for o in bed_objs]}")
    if pillow is not None:
        print(f"  Pillow '{pillow.name}' position: {pillow.get_position_orientation()[0].tolist()}")

    # Let physics settle
    for _ in range(100):
        env.step([])

    if not headless:
        input(
            "\n\nThe bottle's cap is attached via a joint (connectedpart).\n"
            "The bed's pillow is placed at the annotated offset (extrapart) and rests freely.\n"
            "Try Shift+Left-Click+Drag on the cap to break it off the bottle.\n"
            "Press [ENTER] to continue."
        )

    steps = 0
    max_steps = -1 if not short_exec else 500
    while steps != max_steps:
        env.step([])
        steps += 1

    og.shutdown()


if __name__ == "__main__":
    main()
