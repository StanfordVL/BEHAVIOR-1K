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
- `VisualShape`

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
|   |   # Also contains diffuse texture maps that will be used when some object state changes happen, e.g. cooked, burnt, or saturated.
│   │   DIFFUSE.encrypted.png
│   │   METALLIC.encrypted.png	
│   │   NORMAL.encrypted.png
│   │   ROUGHNESS.encrypted.png
│   │   DIFFUSE_Frozen.encrypted.png
│   │   DIFFUSE_Cooked.encrypted.png
│   │   DIFFUSE_Burnt.encrypted.png
│   │   DIFFUSE_saturated.encrypted.png
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
In this example, we import three objects into PyBullet, two of which are articulated objects. The code can be found here: [igibson/examples/demo/object_example.py](https://github.com/StanfordVL/iGibson/blob/master/igibson/examples/demo/object_example.py).

```python
from igibson.objects.ycb_object import YCBObject
from igibson.objects.articulated_object import ArticulatedObject
import igibson
import os
import pybullet as p
import pybullet_data
import time


def main():
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1./240.)

    floor = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
    p.loadMJCF(floor)

    cabinet_0007 = os.path.join(
        igibson.assets_path, 'models/cabinet2/cabinet_0007.urdf')
    cabinet_0004 = os.path.join(
        igibson.assets_path, 'models/cabinet/cabinet_0004.urdf')

    obj1 = ArticulatedObject(filename=cabinet_0007)
    obj1.load()
    obj1.set_position([0, 0, 0.5])

    obj2 = ArticulatedObject(filename=cabinet_0004)
    obj2.load()
    obj2.set_position([0, 0, 2])

    obj3 = YCBObject('003_cracker_box')
    obj3.load()
    obj3.set_position_orientation([0, 0, 1.2], [0, 0, 0, 1])

    for _ in range(24000):  # at least 100 seconds
        p.stepSimulation()
        time.sleep(1./240.)

    p.disconnect()


if __name__ == '__main__':
    main()

```

You can open the cabinet and the drawer by dragging your mouse over them. You can even put the cereal box into the drawer like this:
![object](images/object.png)

