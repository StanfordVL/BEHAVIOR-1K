import numpy as np
import igibson as ig
from igibson import object_states
from igibson.macros import gm
from igibson.objects import DatasetObject, LightObject


def main():
    # Make sure object states are enabled
    assert gm.ENABLE_OBJECT_STATES, f"Object states must be enabled in macros.py in order to use this demo!"

    # Create the scene config to load -- empty scene
    cfg = {
        "scene": {
            "type": "EmptyScene",
        }
    }

    # Create the environment
    env = ig.Environment(configs=cfg, action_timestep=1/60., physics_timestep=1/60.)

    # Set camera to appropriate viewing pose
    ig.sim.viewer_camera.set_position_orientation(
        position=np.array([ 0.182103, -2.07295 ,  0.14017 ]),
        orientation=np.array([0.77787037, 0.00267566, 0.00216149, 0.62841535]),
    )

    # Create a light object
    light = LightObject(
        prim_path="/World/sphere_light",
        light_type="Sphere",
        name="sphere_light",
        radius=0.01,
        intensity=1e5,
    )
    ig.sim.import_object(light)
    light.set_position(np.array([-2.0, -2.0, 1.0]))
    env.step(np.array([]))

    # Import bowls of varying sizes
    obj_category = "bowl"
    obj_model = "68_0"
    scales = [0.5, 1.0, 2.0]
    xs = [-0.6, 0, 0.8]
    objs = []

    for i, (scale, x) in enumerate(zip(scales, xs)):
        name = f"{obj_category}{i}"
        obj = DatasetObject(
            prim_path=f"/World/{name}",
            name=name,
            category=obj_category,
            model=obj_model,
            scale=scale,
            abilities={"heatable": {}},
        )
        # Make sure the bowls can be heated
        assert object_states.Heated in obj.states
        ig.sim.import_object(obj)
        obj.set_position(np.array([x, 0, 0]))
        objs.append(obj)

    # Take a step to make sure all objects are fully initialized
    env.step(np.array([]))

    def report_states(objs):
        for obj in objs:
            print("=" * 20)
            print("object:", obj.name)
            print("temperature:", obj.states[object_states.Temperature].get_value())
            print("obj is heated:", obj.states[object_states.Heated].get_value())

    # Report default states
    print("==== Initial state ====")
    report_states(objs)

    # Notify user that we're about to heat the object
    input("Objects will be heated, and steam will slowly rise. Press ENTER to continue.")

    # Heated.
    for obj in objs:
        obj.states[object_states.Temperature].set_value(50)
    env.step(np.array([]))
    report_states(objs)

    # Take a look at the steam effect.
    # After a while, objects will be below the Steam temperature threshold.
    print("==== Objects are now heated... ====")
    print()
    for _ in range(2000):
        env.step(np.array([]))
        # Also print temperatures
        temps = [f"{obj.states[object_states.Temperature].get_value():>7.2f}" for obj in objs]
        print(f"obj temps:", *temps, end="\r")
    print()

    # Objects are not heated anymore.
    print("==== Objects are no longer heated... ====")
    report_states(objs)

    # Close environment at the end
    input("Demo completed. Press ENTER to shutdown environment.")
    env.close()


if __name__ == "__main__":
    main()
