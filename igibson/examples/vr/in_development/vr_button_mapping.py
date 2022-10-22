"""
Demo for figuring out optimal button mapping.
The system will print out button events as they happen.
It will show you the button idx and press id, in a format that
can be directly pasted into the vr_config.yaml.

Please use this if creating a custom action-button mapping for a VR controller
that is neither an HTC Vive controller nor an Oculus controller.
"""
from omnigibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from omnigibson.scenes.empty_scene import EmptyScene
from omnigibson.simulator import Simulator


def main():
    s = Simulator(mode="vr", rendering_settings=MeshRendererSettings(enable_shadow=True, optimized=True))
    scene = EmptyScene()
    s.import_scene(scene)

    # Main simulation loop
    while True:
        s.step()

        # Print all button events that happen each frame
        event_data = s.get_vr_events()
        if event_data:
            print("----- Next set of event data (on current frame): -----")
            for event in event_data:
                readable_event = ["left_controller" if event[0] == 0 else "right_controller", event[1], event[2]]
                print("Event (controller, button_idx, press_id): {}".format(readable_event))


if __name__ == "__main__":
    main()
