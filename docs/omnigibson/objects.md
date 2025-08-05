# :material-food-apple-outline: **Object**

## Description

In **`OmniGibson`**, `Object`s define entities that can be placed arbitrarily within a given [`Scene`](./scenes.md). These entities can range from arbitrarily imported USD assets, to BEHAVIOR-specific dataset assets, to procedurally generated lights. `Object`s serve as the main building block of a given `Scene` instance, and provide a unified interface for quickly prototyping scenes.

## Usage

### Importing Objects

Objects can be added to a given `Environment` instance by specifying them in the config that is passed to the environment constructor via the `objects` key. This is expected to be a list of dictionaries, each of which specifies the desired configuration for a single object to be created. For each dict, the `type` key is required and specifies the desired object class, and global `position` and `orientation` (in (x,y,z,w) quaternion form) can also be specified. Additional keys can be specified and will be passed directly to the specific object class constructor. An example of an object configuration is shown below in `.yaml` form:


??? code "object_config_example.yaml"
    ``` yaml linenums="1"
    objects:
      - type: USDObject         # For your custom imported object
        name: my_object
        usd_path: your_path_to_model.usd
        visual_only: False
        position: [0, 0, 0]
        orientation: [0, 0, 0, 1]
        scale: [0.5, 0.6, 0.7]
      - type: DatasetObject     # For a pre-existing BEHAVIOR-1K object
        name: apple0
        category: apple
        model: agveuv
        position: [0, 0, 0.5]
        orientation: [0, 0, 0, 1]
        scale: [0.4, 0.4, 0.4]
    ```

Alternatively, an object can be directly imported at runtime by first creating the object class instance (e.g.: `obj = DatasetObject(...)`) and then importing it via `og.sim.import_object(obj)`. This can be useful for iteratively prototyping a desired scene configuration.

### Runtime

Once an object is imported into the simulator / environment, we can directly query and set variious properties. For example, to teleport the object, simply call `object.set_position_orientation(new_pos, new_orn)`. Setting a desired joint configuration can be done via `obj.set_joint_positions(new_joint_pos)`.

??? warning annotate "Some attributes require sim cycling"

    For properties that fundamentally alter an object's physical behavior (such as scale, enabled collisions, or collision filter pairs), values set at runtime will **not** propagate until the simulator is stopped (`og.sim.stop()`) and re-started (`og.sim.play()`).


All objects are tracked and organized by the underlying scene, and can quickly be [queried by relevant properties](./scenes.md#runtime).


## Types
**`OmniGibson`** directly supports multiple `Object` classes, which are intended to encapsulate different types of objects with varying functionalities. The most basic is [`BaseObject`](../reference/objects/object_base.md), which can capture any arbitrary object and thinly wraps an [`EntityPrim`](../reference/prims/entity_prim.md). The more specific classes are shown below:

<table markdown="span">
    <tr>
        <td valign="top">
            [**`StatefulObject`**](../reference/objects/stateful_object.md)<br><br>
            Encapsulates an object that owns a set of [object states](./object_states.md). In general, this is intended to be a parent class, and not meant to be instantiated directly.<br><br>
        </td>
    </tr>
    <tr>
        <td valign="top">
            [**`USDObject`**](../reference/objects/usd_object.md)<br><br>
            Encapsulates an object imported from a usd file. Useful when loading custom USD assets into **`OmniGibson`**. Users should specify the absolute `usd_path` to the desired file to import.<br><br>
        </td>
    </tr>
    <tr>
        <td valign="top">
            [**`DatasetObject`**](../reference/objects/dataset_object.md)<br><br>
            This inherits from `USDObject` and encapsulates an object from the BEHAVIOR-1K dataset. Users should specify the `category` and `model` of object to load, where `model` is a 6 character string unique to each dataset object. For an overview of all possible categories and models, please refer to our [Knowledgebase Dashboard](https://behavior.stanford.edu/knowledgebase/)<br><br>
        </td>
    </tr>
    <tr>
        <td valign="top">
            [**`PrimitiveObject`**](../reference/objects/primitive_object.md)<br><br>
            Encapsulates an object defined by a single primitive geom, such a sphere, cube, or cylinder. These are often used as visual objects (via `visual_only=True`) in the scene, e.g., for visualizing the target location of a robot reaching task.<br><br>
        </td>
    </tr>
    <tr>
        <td valign="top">
            [**`LightObject`**](../reference/objects/light_object.md)<br><br>
            Encapsulates a virtual light source, where both the shape (sphere, disk, dome, etc.), size, and intensity can be specified.<br><br>
        </td>
    </tr>
    <tr>
        <td valign="top">
            [**`ControllableObject`**](../reference/objects/controllable_object.md)<br><br>
            Encapsulates an object that is motorized, for example, a conveyer belt, and provides functionality to apply actions and deploy control signals to the motors. However, currently this class is used exclusively as a parent class of `BaseRobot`, and should not be instantiated directly by users.<br><br>
        </td>
    </tr>
</table>
