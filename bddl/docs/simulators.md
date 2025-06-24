# Simulators

### Overview

**Simulator** maintains an instance of **Renderer** and **PhysicsEngine** and provides APIs to import **Scene**, **Object** and **Robot** into both of them and keep them synchronized at all time.

Some key functions are the following:
- `load`: initialize PyBullet physics engine and MeshRenderer
- `import_{scene, ig_scene}`: import the scene into PyBullet by calling `scene.load`, and then import it into MeshRenderer by calling `self.renderer.add_instance`. If `InteractiveIndoorScene` is imported using `import_ig_scene`, all objects in the scene are also imported.
- `import_{object, articulated_object, robot}`: import the object, articulated object and robot into the simulator in a similar manner
- `sync`: synchronize the poses of the dynamic objects (including the robots) between PyBullet and MeshRenderer. Specifically, it calls `update_position` for each object, in which it retrieve the object's pose in PyBullet, and then update its pose accordingly in MeshRenderer.

If `Simulator` uses `gui` mode, by default it will also maintain a `Viewer`, which essentially is a virtual camera in the scene that can render images. More info about the `Viewer` can be found here: [igibson/render/viewer.py](https://github.com/StanfordVL/iGibson/blob/master/igibson/render/viewer.py). Notably, you can press `h` in the `ExternalView` window to show the help menu for mouse/keyboard control.

Most of the code can be found here: [igibson/simulator.py](https://github.com/StanfordVL/iGibson/blob/master/igibson/simulator.py).

### Examples
In this example, we import a `StaticIndoorScene`, a `Turtlebot`, and ten `YCBObject` into the simulator. The code can be found here: [igibson/examples/demo/simulator_example.py](https://github.com/StanfordVL/iGibson/blob/master/igibson/examples/demo/simulator_example.py)

```python
from igibson.robots.turtlebot_robot import Turtlebot
from igibson.simulator import Simulator
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.objects.ycb_object import YCBObject
from igibson.utils.utils import parse_config
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
import numpy as np
from igibson.render.profiler import Profiler
from IPython import embed


def main():
    config = parse_config('../configs/turtlebot_demo.yaml')
    settings = MeshRendererSettings(enable_shadow=False, msaa=False)
    s = Simulator(mode='gui', image_width=256,
                  image_height=256, rendering_settings=settings)

    scene = StaticIndoorScene('Rs',
                              build_graph=True,
                              pybullet_load_texture=True)
    s.import_scene(scene)
    turtlebot = Turtlebot(config)
    s.import_robot(turtlebot)

    for _ in range(10):
        obj = YCBObject('003_cracker_box')
        s.import_object(obj)
        obj.set_position_orientation(np.random.uniform(
            low=0, high=2, size=3), [0, 0, 0, 1])

    print(s.renderer.instances)

    for i in range(10000):
        with Profiler('Simulator step'):
            turtlebot.apply_action([0.1, 0.1])
            s.step()
            rgb = s.renderer.render_robot_cameras(modes=('rgb'))
    s.disconnect()


if __name__ == '__main__':
    main()

```
