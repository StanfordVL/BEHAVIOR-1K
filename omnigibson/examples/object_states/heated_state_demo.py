import torch as th

import omnigibson as og
from omnigibson import object_states
from omnigibson.macros import gm

# Make sure object states are enabled
gm.ENABLE_OBJECT_STATES = True


def main(random_selection=False, headless=False, short_exec=False):
    # Define object configurations for objects to load -- we want to load a light and three bowls
    obj_configs = []

    obj_configs.append(
        dict(
            type="LightObject",
            light_type="Sphere",
            name="light",
            radius=0.01,
            intensity=1e8,
            position=[-2.0, -2.0, 1.0],
        )
    )

    for i, (scale, x) in enumerate(zip([0.5, 1.0, 2.0], [-0.6, 0, 0.8])):
        obj_configs.append(
            dict(
                type="DatasetObject",
                name=f"bowl{i}",
                category="bowl",
                model="ajzltc",
                bounding_box=th.tensor([0.329, 0.293, 0.168]) * scale,
                abilities={"heatable": {}},
                position=[x, 0, 0.2],
            )
        )

    # Create the scene config to load -- empty scene with light object and bowls
    cfg = {
        "scene": {
            "type": "Scene",
        },
        "objects": obj_configs,
    }

    # Create the environment
    env = og.Environment(configs=cfg)

    # Set camera to appropriate viewing pose
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([0.182103, -2.07295, 0.14017]),
        orientation=th.tensor([0.77787037, 0.00267566, 0.00216149, 0.62841535]),
    )

    # Dim the skybox so we can see the bowls' steam effectively
    og.sim.skybox.intensity = 100.0

    # Grab reference to objects of relevance
    objs = list(env.scene.object_registry("category", "bowl"))

    def report_states(objs):
        for obj in objs:
            print("=" * 20)
            print("object:", obj.name)
            print("temperature:", obj.states[object_states.Temperature].get_value())
            print("obj is heated:", obj.states[object_states.Heated].get_value())

    # Report default states
    print("==== Initial state ====")
    report_states(objs)

    if not short_exec:
        # Notify user that we're about to heat the object
        input("Objects will be heated, and steam will slowly rise. Press ENTER to continue.")

    # Heated.
    for obj in objs:
        obj.states[object_states.Temperature].set_value(50)
    env.step(th.empty(0))
    report_states(objs)

    # Take a look at the steam effect.
    # After a while, objects will be below the Steam temperature threshold.
    print("==== Objects are now heated... ====")
    print()
    max_iterations = 2000 if not short_exec else 100
    for _ in range(max_iterations):
        env.step(th.empty(0))
        # Also print temperatures
        temps = [f"{obj.states[object_states.Temperature].get_value():>7.2f}" for obj in objs]
        print(f"obj temps:", *temps, end="\r")
    print()

    # Objects are not heated anymore.
    print("==== Objects are no longer heated... ====")
    report_states(objs)

    if not short_exec:
        # Close environment at the end
        input("Demo completed. Press ENTER to shutdown environment.")

    og.clear()


if __name__ == "__main__":
    main()
