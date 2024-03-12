---
icon: material/food-apple-outline
---

# 🍎 **Object**

Objects, such as furnitures, are essential to building manipulation environments. We designed the MujocoObject interfaces to standardize and simplify the procedure for importing 3D models into the scene or procedurally generate new objects. MuJoCo defines models via the MJCF XML format. These MJCF files can either be stored as XML files on disk and loaded into simulator, or be created on-the-fly by code prior to simulation. 

OmniGibson supports 6 types of objects:

- `ControllableObject`: This class represents objects that can be controller through joint controllers. It is used as the parent class of the robot classes and provide functionalities to apply control actions to the objects.

- `StatefulObject`: This class represents objects that comes with object states. For more information regarding object states please take a look at xx.

- `PrimitiveObject`: This class represents primitive shape objects (Cubes, Spheres, Cones, etc.) This are usually 

- `LightObject`: This class specifically represents lights in the scene, and provide funtionalities to modify the properties of lights.

- `USDObject`: This class represents objects loaded through a USD file. This is useful when users want to load a custom USD asset into the simulator.

- `DatasetObject`: This class inherits from `USDObject` and represents object from the OmniGibson dataset.

