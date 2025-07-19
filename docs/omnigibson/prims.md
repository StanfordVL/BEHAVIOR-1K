# :material-cube-outline: **Prim**

## Description

A Prim, short for "primitive," is a fundamental building block of Omniverse's underlying scene representation (called a "stage"), and represents a single scene component, such as a rigid body, joint, light, camera, or material. **`OmniGibson`** implements `Prim` classes which directly encapsulate the underlying omniverse `UsdPrim` instances, and provides direct access to omniverse's low level prim APIs.

Each `Prim` instance uniquely wraps a corresponding prim in the current scene stage, and is defined by its corresponding `prim_path`. This filepath-like string defines the prim's name, as well as all of its preceding parent prim names. For example, a `RigidPrim` capturing a robot's gripper link may have a prim path of `/World/robot0/gripper_link`, indicating that the `gripper_link` is a child of the `robot0` prim, which in turn is a child of the `World` prim. 

Additionally, prims carry a range of attributes, including position, rotation, scale, and material properties. These attributes define the properties and characteristics of the objects they represent.

## Usage

### Loading a Prim
Generally, you should not have to directly instantiate any `Prim` class instance, as all of the main entry-point level classes within **`OmniGibson`** either do not require or directly import `Prim` instances themselves. However, you can always create a `Prim` instance directly. This requires two arguments at the minimum: a unique `name`, and a corresponding `prim_path` (which can either point to a pre-existing prim on the Omniverse scene stage or to a novel location where a new prim will be created).

If a prim already exists at `prim_path`, the created `Prim` instance will automatically point to it. However, if it does _not_ exist, you must call `prim.load()` explicitly to load the prim to the omniverse stage at the desired `prim_path`. Note that not all prim classes can be loaded from scratch -- for example, `GeomPrim`s require a pre-existing `prim_path` when created!

After the prim has been created, it may additionally require further initialization via `prim.initialize()`, which _must_ occur at least 1 simulation step after the prim has been loaded. (1)
{ .annotate }

1. This is a fundamental quirk of omniverse and unfortunately cannot be changed ):

### Runtime
Once initialized, a `Prim` instance can be used as a direct interface with the corresponding low-level prim on the omniverse stage. The low-level attributes of the underlying prim can be queried / set via `prim.get_attribute(name)` / `prim.set_attribute(name, val)`. In addition, some `Prim` classes implement higher-level functionality to more easily manipulate the underlying prim, such as `MaterialPrim`'s `bind(prim_path)`, which binds its owned material to the desired prim located at `prim_path`.

## Types
**`OmniGibson`** directly supports multiple `Prim` classes, which are intended to encapsulate different types of prims from the omniverse scene stage. The most basic is [`BasePrim`](../reference/prims/prim_base.md), which can capture any arbitrary prim. The more specific classes are shown below:

<table markdown="span">
    <tr>
        <td valign="top">
            [**`XFormPrim`**](../reference/prims/xform_prim.md)<br><br>
            Encapsulates a transformable prim. This prim can get and set its local or global pose, as well as its own scale.<br><br>
        </td>
    </tr>
    <tr>
        <td valign="top">
            [**`GeomPrim`**](../reference/prims/geom_prim.md#prims.geom_prim.GeomPrim)<br><br>
            Encapsulates a prim defined by a geom (shape or mesh). It is an `XFormPrim` that can additionally owns geometry defined by its set of `points`. Its subclasses [`VisualGeomPrim`](../reference/prims/geom_prim.md) and [`CollisionGeomPrim`](../reference/prims/geom_prim.md#prims.geom_prim.CollisionGeomPrim) implement additional utility for dealing with those respective types of geometries (e.g.: `CollisionGeomPrim.set_collision_approximation(...)`).<br><br>
        </td>
    </tr>
    <tr>
        <td valign="top">
            [**`ClothPrim`**](../reference/prims/cloth_prim.md)<br><br>
            Encapsulates a prim defined by a mesh geom that is to be converted into cloth. It is a `GeomPrim` that dynamically transforms its owned (rigid) mesh into a (compliant, particle-based) cloth. Its methods can be used to query and set its individual particles' state, as well as track a subset of keypoints / keyfaces.<br><br>
        </td>
    </tr>
    <tr>
        <td valign="top">
            [**`RigidPrim`**](../reference/prims/rigid_prim.md)<br><br>
            Encapsulates a prim defined by a rigid body. It is an `XFormPrim` that is subject to physics and gravity, and may belong to an `EntityPrim`. It additionally has attributes to control its own mass, density, and other physics-related behavior.<br><br>
        </td>
    </tr>
    <tr>
        <td valign="top">
            [**`JointPrim`**](../reference/prims/joint_prim.md)<br><br>
            Encapsulates a prim defined by a joint. It belongs to an `EntityPrim` and has attributes to control its own joint state (position, velocity, effort).<br><br>
        </td>
    </tr>
    <tr>
        <td valign="top">
            [**`EntityPrim`**](../reference/prims/entity_prim.md)<br><br>
            Encapsulates the top-level prim of an imported object. Since the underlying object consists of a set of links and joints, this class owns its corresponding set of `RigidPrim`s and `JointPrim`s, and provides high-level functionality to controlling the object's pose, joint state, and physics-related behavior.<br><br>
        </td>
    </tr>
    <tr>
        <td valign="top">
            [**`MaterialPrim`**](../reference/prims/material_prim.md)<br><br>
            Encapsulates a prim defining a material specification. It provides high-level functionality for directly controlling the underlying material's properties and behavior.<br><br>
        </td>
    </tr>
</table>


