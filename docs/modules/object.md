---
icon: material/food-apple-outline
---

# 🍎 **Object**

Objects, such as furnitures, are essential to building manipulation environments. We designed the MujocoObject interfaces to standardize and simplify the procedure for importing 3D models into the scene or procedurally generate new objects. MuJoCo defines models via the MJCF XML format. These MJCF files can either be stored as XML files on disk and loaded into simulator, or be created on-the-fly by code prior to simulation. 


## Usage

### Importing Objects

Objects can be added to a given `Environment` instance by specifying them in the config that is passed to the environment constructor via the `objects` key. This is expected to be a list of dictionaries, each of which specifies the desired configuration for a single object to be created. For each dict, the `type` key is required and specifies the desired object class, and global `position` and `orientation` (in (x,y,z,w) quaternion form) can also be specified. Additional keys can be specified and will be passed directly to the specific robot class constructor. An example of a robot configuration is shown below in `.yaml` form:


??? code "single_object_config_example.yaml"
    ``` yaml linenums="1"
    robots:
      - type: USDObject
        name: some_usd_object
        usd_path: your_path_to_model.usd
        visual_only: False
        position: [0, 0, 0]
        orientation: [0, 0, 0, 1]
        scale: [0.5, 0.6, 0.7]
    ```

`OmniGibson` supports 6 types of objects shown as follows:

- `ControllableObject`: This class represents objects that can be controller through joint controllers. It is used as the parent class of the robot classes and provide functionalities to apply control actions to the objects. In general, users should not create object of this class, but rather directly spawn the desired robot type in the `robots` section of the config.

- `StatefulObject`: This class represents objects that comes with object states. For more information regarding object states please take a look at `object_states`. This is also meant to be a parent class, and should generally not be instantiated directly.

- `PrimitiveObject`: This class represents primitive shape objects (Cubes, Spheres, Cones, etc.) This are usually used as visual objects in the scene. For example, users can instantiate a sphere object to visualize the target location of a robot reaching task, and set it's property `visual_only` to true to disable it's kinematics and collision with other objects.

- `LightObject`: This class specifically represents lights in the scene, and provide funtionalities to modify the properties of lights. There are 6 types of lights users can instantiate in OmniGibson, cylinder light, disk light, distant light, dome light, geometry ligtht, rectangle light, and sphere light. Users can choose whichever type of light that works for the best, and set the `intensity` property to control the brightness of it. 

- `USDObject`: This class represents objects loaded through a USD file. This is useful when users want to load a custom USD asset into the simulator. Users should specify the `usd_path` parameter of the `USDObject` in order to load the desired file of their choice.

- `DatasetObject`: This class inherits from `USDObject` and represents object from the OmniGibson dataset. Users should specify the category of objects they want to load, as well as the model id, which is a 6 character string unique to each dataset object. For the possible categories and models, please refer to our [Knowledgebase Dashboard](https://behavior.stanford.edu/knowledgebase/)


### Runtime

Usually, objects are instantiated upon startup. We can modify certain properties of the object when the simulator is running. For example, one might desire to teleop the object from one place to another, then simply call `object.set_position_orientation(new_pos, new_orn)` will do the job. Another example might be to highlight an object by setting `object.highlighed = True`, the object when then be highlighted in pick in the scene.

To access the objects from the environment, one can call `env.scene.object_registry`. Here are a couple examples:

- `env.scene.object_registry("name", OBJECT_NAME): get the object by its name

- `env.scene.object_registry("category", CATEGORY): get the object by its category

- `env.scene.object_registry("prim_path", PRIM_PATH): get the object by its prim path 
