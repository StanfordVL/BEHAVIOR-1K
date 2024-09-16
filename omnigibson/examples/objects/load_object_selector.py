import torch as th

import omnigibson as og
from omnigibson.utils.asset_utils import (
    get_all_object_categories,
    get_all_object_category_models,
    get_og_avg_category_specs,
)
from omnigibson.utils.ui_utils import choose_from_options


def main(random_selection=False, headless=False, short_exec=False):
    """
    This demo shows how to load any scaled objects from the OG object model dataset
    The user selects an object model to load
    The objects can be loaded into an empty scene or an interactive scene (OG)
    The example also shows how to use the Environment API or directly the Simulator API, loading objects and robots
    and executing actions
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)
    scene_options = ["Scene", "InteractiveTraversableScene"]
    scene_type = choose_from_options(options=scene_options, name="scene type", random_selection=random_selection)

    # -- Choose the object to load --

    # Select a category to load
    available_obj_categories = get_all_object_categories()
    obj_category = choose_from_options(
        options=available_obj_categories, name="object category", random_selection=random_selection
    )

    # Select a model to load
    available_obj_models = get_all_object_category_models(obj_category)
    obj_model = choose_from_options(
        options=available_obj_models, name="object model", random_selection=random_selection
    )

    # Load the specs of the object categories, e.g., common scaling factor
    avg_category_spec = get_og_avg_category_specs()

    # Create and load this object into the simulator
    obj_cfg = dict(
        type="DatasetObject",
        name="obj",
        category=obj_category,
        model=obj_model,
        position=[0, 0, 50.0],
    )

    cfg = {
        "scene": {
            "type": scene_type,
        },
        "objects": [obj_cfg],
    }
    if scene_type == "InteractiveTraversableScene":
        cfg["scene"]["scene_model"] = "Rs_int"

    # Create the environment
    env = og.Environment(configs=cfg)

    # Place the object so it rests on the floor
    obj = env.scene.object_registry("name", "obj")
    center_offset = obj.get_position_orientation()[0] - obj.aabb_center + th.tensor([0, 0, obj.aabb_extent[2] / 2.0])
    obj.set_position_orientation(position=center_offset)

    # Step through the environment
    max_steps = 100 if short_exec else 10000
    for i in range(max_steps):
        env.step(th.empty(0))

    # Always close the environment at the end
    og.clear()


if __name__ == "__main__":
    main()
