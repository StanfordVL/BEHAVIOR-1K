""" Test code showing how to use a VrConditionSwitcher.
"""

import os



import igibson
from igibson.objects.ycb_object import YCBObject
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_vr import VrConditionSwitcher, VrSettings
from igibson.robots.behavior_robot import BehaviorRobot
from igibson.scenes.interactive_traversable_scene import InteractiveIndoorScene
from igibson.simulator import Simulator

# HDR files for PBR rendering
hdr_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_02.hdr")
hdr_texture2 = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_03.hdr")
light_modulation_map_filename = os.path.join(
    igibson.ig_dataset_path, "scenes", "Rs_int", "layout", "floor_lighttype_0.png"
)
background_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "urban_street_01.jpg")


def main():
    # VR rendering settings
    vr_rendering_settings = MeshRendererSettings(
        optimized=True,
        fullscreen=False,
        env_texture_filename=hdr_texture,
        env_texture_filename2=hdr_texture2,
        env_texture_filename3=background_texture,
        light_modulation_map_filename=light_modulation_map_filename,
        enable_shadow=True,
        enable_pbr=True,
        msaa=True,
        light_dimming_factor=1.0,
    )
    # VR system settings
    # Change use_vr to toggle VR mode on/off
    vr_settings = VrSettings()
    s = Simulator(mode="vr", rendering_settings=vr_rendering_settings, vr_settings=vr_settings)
    scene = InteractiveIndoorScene("Rs_int")
    s.import_scene(scene)

    # Create a BehaviorRobot and it will handle all initialization and importing under-the-hood
    # Change USE_GRIPPER to switch between the BRHand and the BRGripper (see robots/behavior_robot.py for more details)
    vr_agent = BehaviorRobot()
    # Since vr_height_offset is set, we will use the VR HMD true height plus this offset instead of the z coordinate of start_pos
    s.set_vr_start_pos([0, 0, 0], vr_height_offset=-0.1)
    # Create condition switcher to manage condition switching
    # Note: to start with, the overlay is shown but there is no text
    # The user needs to press the "switch" button to make the next condition appear
    vr_cs = VrConditionSwitcher(s)

    # Objects to interact with
    mass_list = [5, 10, 100, 500]
    mustard_start = [-1, 1.55, 1.2]
    for i in range(len(mass_list)):
        mustard = YCBObject("006_mustard_bottle", renderer_params={"use_pbr": False, "use_pbr_mapping": False})
        s.import_object(mustard)
        mustard.set_position([mustard_start[0] + i * 0.2, mustard_start[1], mustard_start[2]])
        body_ids = mustard.get_body_ids()
        assert len(body_ids) == 1, "Object is expected to be single-body."
        body_id = body_ids[0]
        p.changeDynamics(body_id, -1, mass=mass_list[i])

    # Main simulation loop
    while True:
        s.step()

        # Update VR objects
        vr_agent.apply_action()

        # Switch to different condition with right toggle
        if s.query_vr_event("right_controller", "overlay_toggle"):
            vr_cs.switch_condition()

        # Hide/show condition switcher with left toggle
        if s.query_vr_event("left_controller", "overlay_toggle"):
            vr_cs.toggle_show_state()

    s.disconnect()


if __name__ == "__main__":
    main()
