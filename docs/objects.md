# Objects

### Overview
We provide a wide variety of **Objects** that can be imported into the **Simulator**.
- `YCBObject`
- `RBOObject`
- `ShapeNetObject`
- `Pedestrian`
- `ArticulatedObject`
- `URDFObject`
- `SoftObject`
- `Cube`
- `VisualMarker`

Typically, they take in the name or the path of an object (in `igibson.assets_path`) and provide a `load` function that be invoked externally (usually by `import_object` and `import_object` of `Simulator`). The `load` function imports the object into PyBullet. Some **Objects** (e.g. `ArticulatedObject`) also provide APIs to get and set the object pose.

Most of the code can be found here: [igibson/objects](https://github.com/StanfordVL/iGibson/blob/master/igibson/objects).

### BEHAVIOR Dataset of Objects

We use a new dataset of everyday objects, the BEHAVIOR Dataset of Objects. In total, we curate 1217 object models across 391 object categories, to support 100 BEHAVIOR activities. The categories range from food items to tableware, from home decorations to office supplies, and from apparel to cleaning tools.

To maintain high visual realism, all object models include material information (metallic, roughness) that can be rendered by iGibson 2.0 renderer. To maintain high physics realism, object models are annotated with size, mass, center of mass, moment of inertia, and also stable orientations. The collision mesh is a simplified version of the visual mesh, obtained with a convex decomposition using the VHACD algorithm. Object models with a shape close to a box are annotated with a primitive box collision mesh, much more efficient and robust for collision checking.

All models in the BEHAVIOR Dataset are organized following the WordNet, associating them to synsets. This structure allows us to define properties for all models of the same categories, but it also facilitates more general sampling of activity instances fulfilling initial conditions such as onTop(fruit, table) that can be achieved using any model within the branch fruit of WordNet.

Please see below for an updated file structure for BEHAVIOR Dataset

```bash
OBJECT_NAME
│   # Unified Robot Description Format (URDF)
│   # http://wiki.ros.org/urdf
│   # It defines the object model (parts, articulation, dynamics properties etc.).
│   OBJECT_NAME.urdf 
│
└───shape
│   └───visual 
│   │    │   # Directory containing visual meshes (vm) of the object. Used for iGibson's rendering. Encrypted
│   │    │   # All objs are UV mapped onto the same texture, linked by default.mtl. All faces are triangles.
│   │    │  vm1.encrypted.obj
│   │    │  vm2.encrypted.obj
│   │    │  …
│   │    │  default.mtl (links the geometry to the texture files)
│   │ 
│   └───collision
│   │    │   # Directory containing collision meshes (cm) of the objects. Used for iGibson's physics simulation.
│   │    │   # Each obj represents a unique link of the object.
│   │    │   # For example, link_1_cm.obj represents the collision mesh of link_1. 
│   │    │  cm1.obj
│   │    │  cm2.obj
│   │    │  …
│
└───material
│   │   # Contains 4 default channels:
│   │   # 	DIFFUSE.png (RGB albedo map)
│   │   # 	METALLIC.png (metallic map)
│   │   # 	NORMAL.png (tangent normal map)
│   │   # 	ROUGHNESS.png (roughness map)
|   |   # Also contains diffuse texture maps that will be used when some object state changes happen, e.g. cooked, burnt, or soaked.
│   │   DIFFUSE.encrypted.png
│   │   METALLIC.encrypted.png	
│   │   NORMAL.encrypted.png
│   │   ROUGHNESS.encrypted.png
│   │   DIFFUSE_Frozen.encrypted.png
│   │   DIFFUSE_Cooked.encrypted.png
│   │   DIFFUSE_Burnt.encrypted.png
│   │   DIFFUSE_Soaked.encrypted.png
│   │   DIFFUSE_ToggledOn.encrypted.png
│
└───misc
│   │   # contains bounding box information of the object, its stable orientations, object state-related link annotation (e.g. toggle button, water/heat/cold source, slicer/cleaning_tool, etc)
│   │   metadata.json
│   │   # contains the object’s material annotation of what kinds of material each link can have.
│   │   material_groups.json 
│   │   # contains info about the minimum volume oriented bounding box that best fits the object
│   │   mvbb_meta.json 
│
└───visualizations
│   │   # An image and a video of the object rendered with iG renderer
│   │   00.png
│   │   OBJECT_NAME.mp4

```



### Adding other objects to iGibson
We provide detailed instructions and scripts to import your own objects (non-articulated) into iGibson. 

Instruction can be found here: [External Objects](https://github.com/StanfordVL/iGibson/blob/master/igibson/utils/data_utils/ext_object). 


### Examples
In this example, we import a few objects into iGibson. The code can be found here: [igibson/examples/objects/load_objects.py](https://github.com/StanfordVL/iGibson/blob/master/igibson/examples/objects/load_objects.py).

```python
import logging
import os
from sys import platform

import numpy as np
import yaml

import igibson
from igibson.envs.igibson_env import iGibsonEnv
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


def main():
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
    type_of_scene = let_user_pick(scene_options) - 1

    if type_of_scene == 0:  # Empty
        config = parse_config(os.path.join(igibson.example_config_path, "turtlebot_static_nav.yaml"))
        settings = MeshRendererSettings(enable_shadow=False, msaa=False, texture_scale=0.5)
        s = Simulator(mode="gui_interactive", image_width=512, image_height=512, rendering_settings=settings)
        scene = EmptyScene(render_floor_plane=True, floor_plane_rgba=[0.6, 0.6, 0.6, 1])
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
        env = iGibsonEnv(config_file=config_data, mode="gui_interactive")
        s = env.simulator

    elif type_of_scene == 2:  # Gibson
        config = parse_config(os.path.join(igibson.example_config_path, "turtlebot_static_nav.yaml"))
        settings = MeshRendererSettings(enable_shadow=False, msaa=False)
        # Reduce texture scale for Mac.
        if platform == "darwin":
            settings.texture_scale = 0.5
        s = Simulator(mode="gui_interactive", image_width=512, image_height=512, rendering_settings=settings)

        scene = StaticIndoorScene("Rs", build_graph=True, pybullet_load_texture=False)
        s.import_scene(scene)
        robot_config = config["robot"]
        robot_config.pop("name")
        turtlebot = Turtlebot(**robot_config)
        s.import_object(turtlebot)

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
                initial_pos=obj["pos"],
                initial_orn=obj["orn"],
            )
            s.import_object(simulator_obj)

        for _ in range(10):
            obj = YCBObject("003_cracker_box")
            s.import_object(obj)
            obj.set_position_orientation(np.append(np.random.uniform(low=0, high=2, size=2), [1.8]), [0, 0, 0, 1])

        if type_of_scene == 1:
            for j in range(10):
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
            for i in range(10000):
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

```