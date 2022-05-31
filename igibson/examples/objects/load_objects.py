import logging
import os
from sys import platform

import numpy as np
import yaml

import igibson
from igibson.envs.igibson_env import iGibsonEnv
# from igibson.external.pybullet_tools.utils import quat_from_euler
from igibson.objects.usd_object import URDFObject
from igibson.objects.ycb_object import YCBObject
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.render.profiler import Profiler
from igibson.robots.turtlebot import Turtlebot
from igibson.scenes.empty_scene import EmptyScene
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_ig_avg_category_specs, get_ig_category_path, get_ig_model_path
from igibson.utils.utils import let_user_pick, parse_config


def main(random_selection=False, headless=False, short_exec=False):
    """
    This demo shows how to load scaled objects from the iG object model dataset and
    additional objects from the YCB dataset in predefined locations
    Loads a concrete object model of a table, and a random one of the same category, and 10 cracker boxes
    The objects can be loaded into an empty scene, an interactive scene (iG) or a static scene (Gibson)
    The example also shows how to use the Environment API or directly the Simulator API, loading objects and robots
    and executing actions
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    scene_options = ["Empty scene", "Interactive scene (iG)", "Static scene (Gibson)"]
    type_of_scene = let_user_pick(scene_options, random_selection=random_selection) - 1

    if type_of_scene == 0:  # Empty
        config = parse_config(os.path.join(igibson.example_config_path, "turtlebot_static_nav.yaml"))
        settings = MeshRendererSettings(enable_shadow=False, msaa=False, texture_scale=0.5)
        s = Simulator(
            mode="gui_interactive" if not headless else "headless",
            image_width=512,
            image_height=512,
            rendering_settings=settings,
        )
        scene = EmptyScene(floor_plane_rgba=[0.6, 0.6, 0.6, 1])
        # scene.load_object_categories(benchmark_names)
        s.import_scene(scene)
        robot_config = config["robot"]
        robot_config.pop("name")
        turtlebot = Turtlebot(**robot_config)
        s.import_object(turtlebot)

    elif type_of_scene == 1:  # iG
        config_filename = os.path.join(igibson.example_config_path, "turtlebot_nav.yaml")
        config_data = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
        config_data["load_object_categories"] = []  # Uncomment this line to accelerate loading with only the building
        config_data["visible_target"] = False
        config_data["visible_path"] = False
        # Reduce texture scale for Mac.
        if platform == "darwin":
            config_data["texture_scale"] = 0.5
        env = iGibsonEnv(config_file=config_data, mode="gui_interactive" if not headless else "headless")
        s = env.simulator

    elif type_of_scene == 2:  # Gibson
        config = parse_config(os.path.join(igibson.example_config_path, "turtlebot_static_nav.yaml"))
        settings = MeshRendererSettings(enable_shadow=False, msaa=False)
        # Reduce texture scale for Mac.
        if platform == "darwin":
            settings.texture_scale = 0.5
        s = Simulator(
            mode="gui_interactive" if not headless else "headless",
            image_width=512,
            image_height=512,
            rendering_settings=settings,
        )

        scene = StaticIndoorScene("Rs", build_graph=True, pybullet_load_texture=False)
        s.import_scene(scene)
        robot_config = config["robot"]
        robot_config.pop("name")
        turtlebot = Turtlebot(**robot_config)
        s.import_object(turtlebot)

    if not headless:
        # Set a better viewing direction
        s.viewer.initial_pos = [-2, 1.4, 1.2]
        s.viewer.initial_view_direction = [0.6, -0.8, 0.1]
        s.viewer.reset_viewer()

    # Objects to load: two tables, the first one is predefined model, the second, random for the same category
    table_objects_to_load = {
        "table_1": {
            "category": "breakfast_table",
            "model": "1b4e6f9dd22a8c628ef9d976af675b86",
            "pos": (0.0, -0.2, 1.01),
            "orn": (0, 0, 90),
        },
        "table_2": {
            "category": "breakfast_table",
            "pos": (0.5, -2.0, 1.01),
            "orn": (0, 0, 45),
        },
    }

    # Load the specs of the object categories, e.g., common scaling factor
    avg_category_spec = get_ig_avg_category_specs()

    scene_objects = {}
    try:
        for obj in table_objects_to_load.values():
            category = obj["category"]
            if category in scene_objects:
                scene_objects[category] += 1
            else:
                scene_objects[category] = 1

            # Get the path for all models of this category
            category_path = get_ig_category_path(category)

            # If the specific model is given, we use it. If not, we select one randomly
            if "model" in obj:
                model = obj["model"]
            else:
                model = np.random.choice(os.listdir(category_path))

            # Create the full path combining the path for all models and the name of the model
            model_path = get_ig_model_path(category, model)
            filename = os.path.join(model_path, model + ".urdf")

            # Create a unique name for the object instance
            obj_name = "{}_{}".format(category, scene_objects[category])

            # Create and import the object
            simulator_obj = URDFObject(
                filename,
                name=obj_name,
                category=category,
                model_path=model_path,
                avg_obj_dims=avg_category_spec.get(category),
                fit_avg_dim_volume=True,
                texture_randomization=False,
                overwrite_inertial=True,
            )
            s.import_object(simulator_obj)
            simulator_obj.set_position_orientation(obj["pos"], quat_from_euler(obj["orn"]))

        for _ in range(10):
            obj = YCBObject("003_cracker_box")
            s.import_object(obj)
            obj.set_position_orientation(np.append(np.random.uniform(low=0, high=2, size=2), [1.8]), [0, 0, 0, 1])

        if type_of_scene == 1:
            max_iterations = 1 if short_exec else 10
            for j in range(max_iterations):
                logging.info("Resetting environment")
                env.reset()
                for i in range(100):
                    with Profiler("Environment action step"):
                        # action = env.action_space.sample()
                        state, reward, done, info = env.step([0.1, 0.1])
                        if done:
                            logging.info("Episode finished after {} timesteps".format(i + 1))
                            break
        else:
            max_steps = 100 if short_exec else 10000
            for i in range(max_steps):
                with Profiler("Simulator step"):
                    turtlebot.apply_action([0.1, 0.1])
                    s.step()
                    rgb = s.renderer.render_robot_cameras(modes=("rgb"))

    finally:
        if type_of_scene == 1:
            env.close()
        else:
            s.disconnect()


if __name__ == "__main__":
    main()
