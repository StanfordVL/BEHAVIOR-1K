import logging
import os

import igibson
from igibson.objects.usd_object import URDFObject
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import (
    get_all_object_categories,
    get_ig_avg_category_specs,
    get_ig_model_path,
    get_object_models_of_category,
)
from igibson.utils.utils import let_user_pick


def main(random_selection=False, headless=False, short_exec=False):
    """
    Minimal example to visualize all the models available in the iG dataset
    It queries the user to select an object category and a model of that category, loads it and visualizes it
    No physical simulation
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    settings = MeshRendererSettings(enable_shadow=True, msaa=False, optimized=True)
    s = Simulator(
        mode="gui_interactive" if not headless else "headless",
        image_width=512,
        image_height=512,
        vertical_fov=70,
        rendering_settings=settings,
    )
    scene = EmptyScene(floor_plane_rgba=[0.6, 0.6, 0.6, 1])
    # scene.load_object_categories(benchmark_names)
    s.import_scene(scene)
    s.renderer.set_light_position_direction([0, 0, 10], [0, 0, 0])

    # Select a category to load
    available_obj_categories = get_all_object_categories()
    obj_category = available_obj_categories[
        let_user_pick(available_obj_categories, random_selection=random_selection) - 1
    ]

    # Select a model to load
    available_obj_models = get_object_models_of_category(obj_category)
    obj_model = available_obj_models[let_user_pick(available_obj_models, random_selection=random_selection) - 1]

    logging.info("Visualizing category {}, model {}".format(obj_category, obj_model))

    # Load the specs of the object categories, e.g., common scaling factor
    avg_category_spec = get_ig_avg_category_specs()

    try:
        # Create the full path combining the path for all models and the name of the model
        model_path = get_ig_model_path(obj_category, obj_model)
        filename = os.path.join(model_path, obj_model + ".urdf")

        # Create a unique name for the object instance
        obj_name = "{}_{}".format(obj_category, 0)

        # Create and import the object
        simulator_obj = URDFObject(
            filename,
            name=obj_name,
            category=obj_category,
            model_path=model_path,
            avg_obj_dims=avg_category_spec.get(obj_category),
            fit_avg_dim_volume=True,
            texture_randomization=False,
            overwrite_inertial=True,
        )
        s.import_object(simulator_obj)
        simulator_obj.set_position([0.5, -0.5, 1.01])

        # Set a better viewing direction
        if not headless:
            s.viewer.initial_pos = [2.0, 0, 1.6]
            s.viewer.initial_view_direction = [-1, 0, 0]
            s.viewer.reset_viewer()

            # Visualize object
            max_steps = 100 if short_exec else -1
            step = 0
            while step != max_steps:
                s.viewer.update()
                step += 1

    finally:
        s.disconnect()


if __name__ == "__main__":
    main()
