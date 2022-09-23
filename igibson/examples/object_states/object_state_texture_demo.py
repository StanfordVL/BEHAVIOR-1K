import numpy as np
import igibson as ig
from igibson import object_states
from igibson.objects import DatasetObject, LightObject
from igibson.macros import gm, macros
from igibson.systems import WaterSystem


def main():
    # Make sure object states are enabled
    assert gm.ENABLE_OBJECT_STATES, f"Object states must be enabled in macros.py in order to use this demo!"

    # Create the scene config to load -- empty scene
    cfg = {
        "scene": {
            "type": "EmptyScene",
            "floor_plane_visible": True,
        }
    }

    # Create the environment
    env = ig.Environment(configs=cfg, action_timestep=1/60., physics_timestep=1/60.)

    # Set camera to appropriate viewing pose
    ig.sim.viewer_camera.set_position_orientation(
        position=np.array([ 1.7789 , -1.68822,  1.13551]),
        orientation=np.array([0.57065614, 0.20331904, 0.267029  , 0.74947212]),
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

    # Add a cabinet object
    obj = DatasetObject(
        prim_path="/World/cabinet",
        name="cabinet",
        category="bottom_cabinet",
        model="45087",
        abilities={"freezable": {}, "cookable": {}, "burnable": {}, "soakable": {}, "toggleable": {}},
    )

    # Make sure all the appropriate states are in the object
    assert object_states.Frozen in obj.states
    assert object_states.Cooked in obj.states
    assert object_states.Burnt in obj.states
    assert object_states.Soaked in obj.states
    assert object_states.ToggledOn in obj.states

    # Add the object and take a step to make sure the cabinet is fully initialized
    ig.sim.import_object(obj)
    obj.set_position(np.array([0, 0, 0.55]))
    env.step(np.array([]))

    def report_states():
        # Make sure states are propagated before printing
        for i in range(5):
            env.step(np.array([]))

        print("=" * 20)
        print("temperature:", obj.states[object_states.Temperature].get_value())
        print("obj is frozen:", obj.states[object_states.Frozen].get_value())
        print("obj is cooked:", obj.states[object_states.Cooked].get_value())
        print("obj is burnt:", obj.states[object_states.Burnt].get_value())
        print("obj is soaked:", obj.states[object_states.Soaked].get_value(WaterSystem))
        print("obj is toggledon:", obj.states[object_states.ToggledOn].get_value())
        print("obj textures:", obj.get_textures())

    # Report default states
    print("==== Initial state ====")
    report_states()

    # Notify user that we're about to freeze the object, and then freeze the object
    input("\nObject will be frozen. Press ENTER to continue.")
    obj.states[object_states.Temperature].set_value(-50)
    report_states()

    # Notify user that we're about to cook the object, and then cook the object
    input("\nObject will be cooked. Press ENTER to continue.")
    obj.states[object_states.Temperature].set_value(100)
    report_states()

    # Notify user that we're about to burn the object, and then burn the object
    input("\nObject will be burned. Press ENTER to continue.")
    obj.states[object_states.Temperature].set_value(250)
    report_states()

    # Notify user that we're about to reset the object to its default state, and then reset state
    input("\nObject will be reset to default state. Press ENTER to continue.")
    obj.states[object_states.Temperature].set_value(macros.object_states.temperature.DEFAULT_TEMPERATURE)
    obj.states[object_states.MaxTemperature].set_value(macros.object_states.temperature.DEFAULT_TEMPERATURE)
    report_states()

    # Notify user that we're about to soak the object, and then soak the object
    input("\nObject will be soaked. Press ENTER to continue.")
    obj.states[object_states.Soaked].set_value(WaterSystem, True)
    report_states()

    # Notify user that we're about to unsoak the object, and then unsoak the object
    input("\nObject will be unsoaked. Press ENTER to continue.")
    obj.states[object_states.Soaked].set_value(WaterSystem, False)
    report_states()

    # Notify user that we're about to toggle on the object, and then toggle on the object
    input("\nObject will be toggled on. Press ENTER to continue.")
    obj.states[object_states.ToggledOn].set_value(True)
    report_states()

    # Notify user that we're about to toggle off the object, and then toggle off the object
    input("\nObject will be toggled off. Press ENTER to continue.")
    obj.states[object_states.ToggledOn].set_value(False)
    report_states()

    # Close environment at the end
    input("Demo completed. Press ENTER to shutdown environment.")
    env.close()


if __name__ == "__main__":
    main()