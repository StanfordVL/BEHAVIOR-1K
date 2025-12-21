# Physics Engine

### Overview
We use the open-sourced [PyBullet](http://www.pybullet.org/) as our underlying physics engine. It can simulate rigid body collision and joint actuation for robots and articulated objects in an accurate and efficient manner. Since we are using MeshRenderer for rendering and PyBullet for physics simulation, we need to keep them synchronized at all time. Our code have already handled this for you.

Typically, we use `p.createMultiBody` and `p.loadURDF` to load scenes, objects and robots into PyBullet, use `p.resetBasePositionAndOrientation` to set the base pose of robots and objects, `p.resetJointState` to set joint position of robots and articulated objects, and `p.setJointMotorControl2` to control the robots and articulated objects.

More info can be found in here: [PyBullet documentation](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA).

### Examples
In this example, we import a scene, a robot and an object into PyBullet and step through a few seconds of simulation. The code can be found here:[igibson/examples/demo/physics_engine_example.py](https://github.com/StanfordVL/iGibson/blob/master/igibson/examples/demo/physics_engine_example.py).

```python
import pybullet as p
from igibson.utils.assets_utils import get_scene_path, get_texture_file
import igibson

import os
import sys
import time


def main():
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = os.path.join(get_scene_path('Rs'), 'mesh_z_up.obj')

    p.connect(p.GUI)
    p.setGravity(0,0,-9.8)
    p.setTimeStep(1./240.)

    # Load scenes
    collision_id = p.createCollisionShape(p.GEOM_MESH,
                                          fileName=model_path,
                                          meshScale=1.0,
                                          flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
    visual_id = p.createVisualShape(p.GEOM_MESH,
                                    fileName=model_path,
                                    meshScale=1.0)
    texture_filename = get_texture_file(model_path)
    texture_id = p.loadTexture(texture_filename)

    mesh_id = p.createMultiBody(baseCollisionShapeIndex=collision_id,
                                baseVisualShapeIndex=visual_id)

    # Load robots
    turtlebot_urdf = os.path.join(igibson.assets_path, 'models/turtlebot/turtlebot.urdf')
    robot_id = p.loadURDF(turtlebot_urdf, flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)

    # Load objects
    obj_visual_filename = os.path.join(igibson.assets_path, 'models/ycb/002_master_chef_can/textured_simple.obj')
    obj_collision_filename = os.path.join(igibson.assets_path, 'models/ycb/002_master_chef_can/textured_simple_vhacd.obj')
    collision_id = p.createCollisionShape(p.GEOM_MESH,
                                          fileName=obj_collision_filename,
                                          meshScale=1.0)
    visual_id = p.createVisualShape(p.GEOM_MESH,
                                    fileName=obj_visual_filename,
                                    meshScale=1.0)
    object_id = p.createMultiBody(baseCollisionShapeIndex=collision_id,
                                  baseVisualShapeIndex=visual_id,
                                  basePosition=[1.0, 0.0, 1.0],
                                  baseMass=0.1)

    for _ in range(10000):
        p.stepSimulation()

    p.disconnect()


if __name__ == '__main__':
    main()
```

You will see the PyBullet interface like this. In the scene, there is a Turtlebot, together with a blue food can next to the robot.
![physics_engine.png](images/physics_engine.png)

