"""
Example script for interacting with OmniGibson scenes with VR and Behaviorbot.
"""
import omnigibson as og
from omnigibson.utils.xr_utils import VRSys

def main():
    # Create the config for generating the environment we want
    env_cfg = {"action_timestep": 1 / 60., "physics_timestep": 1 / 120.}
    scene_cfg = {"type": "InteractiveTraversableScene", "scene_model": "Rs_int"}
    robot0_cfg = {
        "type": "Behaviorbot",
        "controller_config": {
            "gripper_0": {"command_input_limits": "default"},
            "gripper_1": {"command_input_limits": "default"},
        }
    }
    cfg = dict(env=env_cfg, scene=scene_cfg, robots=[robot0_cfg])

    # Create the environment
    env = og.Environment(configs=cfg)
    env.reset()
    # start vrsys 
    vr_robot = env.robots[0]
    vrsys = VRSys(system="SteamVR", vr_robot=vr_robot, enable_touchpad_movement=True)
    vrsys.start()
    # set headset position to be 1m above ground and facing +x
    head_init_transform = vrsys.og2xr(pos=[0, 0, 1], orn=[0, 0, 0, 1])
    vrsys.vr_profile.set_physical_world_to_world_anchor_transform_to_match_xr_device(head_init_transform, vrsys.hmd)

    # main simulation loop
    for _ in range(10000):
        if og.sim.is_playing():
            # step the VR system to get the latest data from VR runtime
            vr_data = vrsys.step()
            # generate robot action and step the environment
            action = vr_robot.gen_action_from_vr_data(vr_data)
            env.step(action)                

    # Shut down the environment cleanly at the end
    vrsys.stop()
    env.close()

if __name__ == "__main__":
    main()