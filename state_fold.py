from omnigibson.utils.constants import PrimType
from omnigibson.macros import gm

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.utils.ui_utils import KeyboardEventHandler
import torch as th

# Make sure object states and GPU dynamics are enabled (GPU dynamics needed for cloth)
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True
gm.DATASET_PATH = "/scr/cloth-test"


def main(random_selection=False, headless=False, short_exec=False):
    """
    Demo of cloth objects that can potentially be folded.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    cloth_category_models = [
        ("jeans", "pvzxyp"),
    ]

    for cloth in cloth_category_models:
        category = cloth[0]
        model = cloth[1]
        print(f"\nCategory: {category}, Model: {model}!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # Create the scene config to load -- empty scene + custom cloth object
        cfg = {
            "scene": {
                "type": "Scene",
            },
            "objects": [
                {
                    "type": "DatasetObject",
                    "name": model,
                    "category": category,
                    "model": model,
                    "prim_type": PrimType.CLOTH,
                    "abilities": {"cloth": {}},
                    "position": [0, 0, 0.5],
                    "orientation": [0, 0, 0, 1],
                    "load_config": {
                        "default_configuration": "settled",
                    },
                },
            ],
        }

        # Create the environment
        env = og.Environment(configs=cfg)

        # Grab object references
        obj = env.scene.object_registry("name", model)

        # Set viewer camera
        og.sim.viewer_camera.set_position_orientation(
            position=th.tensor([0.46382895, -2.66703958, 1.22616824]),
            orientation=th.tensor([0.58779174, -0.00231237, -0.00318273, 0.80900271]),
        )

        def reset_points_to_configuration(configuration):
            obj.root_link.reset_points_to_configuration(configuration)
            obj.set_position_orientation(position=th.zeros(3), orientation=th.tensor([0.0, 0.0, 0.0, 1.0]))
            obj.set_position_orientation(
                position=th.tensor([0, 0, obj.aabb_extent[2] / 2.0 - obj.aabb_center[2]]),
                orientation=th.tensor([0.0, 0.0, 0.0, 1.0]),
            )

        KeyboardEventHandler.initialize()
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.Q,
            callback_fn=lambda: reset_points_to_configuration("default"),
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.W,
            callback_fn=lambda: reset_points_to_configuration("settled"),
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.E,
            callback_fn=lambda: reset_points_to_configuration("folded"),
        )
        KeyboardEventHandler.add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.R,
            callback_fn=lambda: reset_points_to_configuration("crumpled"),
        )

        while True:
            og.sim.render()

        # Shut down env at the end
        print()
        env.close()


if __name__ == "__main__":
    main()
