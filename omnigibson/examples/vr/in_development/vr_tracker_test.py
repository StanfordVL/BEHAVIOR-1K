"""
Demo for testing VR trackers.
The serial number for a tracker can be found in 
SteamVR settings -> controllers -> manage vive trackers
"""
from omnigibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from omnigibson.scenes.empty_scene import EmptyScene
from omnigibson.simulator import Simulator

# Note: replace this with another tracker serial number to test it
TEST_TRACKER_SERIAL_NUM = "LHR-DF82C682"


def main():
    s = Simulator(mode="vr", rendering_settings=MeshRendererSettings(enable_shadow=True, optimized=True))
    scene = EmptyScene()
    s.import_scene(scene)

    # Main simulation loop
    while True:
        s.step()
        is_valid, trans, rot = s.get_data_for_vr_tracker(TEST_TRACKER_SERIAL_NUM)

        if is_valid:
            print("Data for tracker: {}".format(TEST_TRACKER_SERIAL_NUM))
            print("Translation: {}".format(trans))
            print("Rotation: {}".format(rot))


if __name__ == "__main__":
    main()
