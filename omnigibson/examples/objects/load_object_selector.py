import numpy as np
import omnigibson as og
from omnigibson.utils.asset_utils import (
    get_all_object_categories,
    get_og_avg_category_specs,
    get_object_models_of_category,
)
from omnigibson.utils.ui_utils import choose_from_options


def main(random_selection=False, headless=False, short_exec=False):
    """
    This demo shows how to load any scaled objects from the iG object model dataset
    The user selects an object model to load
    The objects can be loaded into an empty scene, an interactive scene (iG) or a static scene (Gibson)
    The example also shows how to use the Environment API or directly the Simulator API, loading objects and robots
    and executing actions
    """
    og.log.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    scene_options = ["Scene", "InteractiveTraversableScene"]
    scene_type = choose_from_options(options=scene_options, name="scene type", random_selection=random_selection)

    # -- Choose the object to load --

    # Select a category to load
    available_obj_categories = get_all_object_categories()
    obj_category = choose_from_options(options=available_obj_categories, name="object category", random_selection=random_selection)

    # Select a model to load
    available_obj_models = get_object_models_of_category(obj_category)
    obj_model = choose_from_options(options=available_obj_models, name="object model", random_selection=random_selection)

    # Load the specs of the object categories, e.g., common scaling factor
    avg_category_spec = get_og_avg_category_specs()

    # Create and load this object into the simulator
    obj_cfg = dict(
        type="DatasetObject",
        name="obj",
        category=obj_category,
        model=obj_model,
        bounding_box=avg_category_spec.get(obj_category),
        fit_avg_dim_volume=True,
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
    env = og.Environment(configs=cfg, action_timestep=1 / 60., physics_timestep=1 / 60.)

    # Place the object so it rests on the floor
    obj = env.scene.object_registry("name", "obj")
    center_offset = obj.get_position() - obj.aabb_center + np.array([0, 0, obj.aabb_extent[2] / 2.0])
    obj.set_position(center_offset)

    # Step through the environment
    max_steps = 100 if short_exec else 10000
    for i in range(max_steps):
        env.step(np.array([]))

    # Always close the environment at the end
    env.close()


if __name__ == "__main__":
    main()
