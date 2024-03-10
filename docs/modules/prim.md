---
icon: material/cube-outline
---

# 🧱 **Prim**

A Prim, short for "primitive," is a fundamental building block of a scene, representing an individual object or entity within the scene's hierarchy. It is essentially a container that encapsulates data, attributes, and relationships, allowing it to represent various scene components like models, cameras, lights, or groups of prims. These prims are systematically organized into a hierarchical framework, creating a scene graph that depicts the relationships and transformations between objects.

Every prim is uniquely identified by a path, which serves as a locator within the scene graph. This path includes the names of all parent prims leading up to it. For example, a prim's path might be `/World/robot0/gripper_link`, indicating that the `gripper_link` is a child of `robot0`. 

Additionally, prims carry a range of attributes, including position, rotation, scale, and material properties. These attributes define the properties and characteristics of the objects they represent.

