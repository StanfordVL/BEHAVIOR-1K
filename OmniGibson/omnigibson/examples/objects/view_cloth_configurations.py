from omnigibson.utils.asset_utils import get_all_object_categories, get_all_object_category_models
from omnigibson.macros import gm

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.utils.constants import PrimType
from omnigibson.utils.ui_utils import KeyboardEventHandler, choose_from_options
import torch as th
from bddl.object_taxonomy import ObjectTaxonomy

# Make sure object states and GPU dynamics are enabled (GPU dynamics needed for cloth)
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = True


def main(random_selection=False, headless=False, short_exec=False):
    """
    Demo showing the multiple configurations stored on each cloth object.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Select a category to load
    available_obj_categories = get_all_object_categories()
    object_taxonomy = ObjectTaxonomy()
    cloth_obj_categories = [
        category
        for category in available_obj_categories
        if object_taxonomy.get_synset_from_category(category)
        and "cloth" in object_taxonomy.get_abilities(object_taxonomy.get_synset_from_category(category))
    ]
    obj_category = choose_from_options(
        options=cloth_obj_categories, name="object category", random_selection=random_selection
    )

    # Select a model to load
    available_obj_models = get_all_object_category_models(obj_category)
    obj_model = choose_from_options(
        options=available_obj_models, name="object model", random_selection=random_selection
    )

    # Create and load this object into the simulator
    obj_cfg = {
        "type": "DatasetObject",
        "name": "cloth",
        "category": obj_category,
        "model": obj_model,
        "prim_type": PrimType.CLOTH,
        "position": [0, 0, 0.5],
        "orientation": [0, 0, 0, 1],
        "load_config": {
            "default_configuration": "settled",
        },
    }

    # Create the scene config to load -- empty scene + custom cloth object
    cfg = {
        "scene": {
            "type": "Scene",
        },
        "objects": [obj_cfg],
    }

    # Create the environment
    env = og.Environment(configs=cfg)

    # Grab object references
    obj = env.scene.object_registry("name", "cloth")

    # Set viewer camera
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([0.46382895, -2.66703958, 1.22616824]),
        orientation=th.tensor([0.58779174, -0.00231237, -0.00318273, 0.80900271]),
    )

    def reset_points_to_configuration(configuration):
        print(f"Resetting to {configuration} configuration")
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
    print("Press Q to reset to default configuration")

    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.W,
        callback_fn=lambda: reset_points_to_configuration("settled"),
    )
    print("Press W to reset to settled configuration")

    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.E,
        callback_fn=lambda: reset_points_to_configuration("folded"),
    )
    print("Press E to reset to folded configuration")

    KeyboardEventHandler.add_keyboard_callback(
        key=lazy.carb.input.KeyboardInput.R,
        callback_fn=lambda: reset_points_to_configuration("crumpled"),
    )
    print("Press R to reset to crumpled configuration")

    # Step through the environment
    max_steps = 100 if short_exec else 10000
    for i in range(max_steps):
        env.step(th.empty(0))

    # Always close the environment at the end
    og.clear()


if __name__ == "__main__":
    main()
