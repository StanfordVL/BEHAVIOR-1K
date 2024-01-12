"""
Example script for interacting with OmniGibson scenes with VR and Behaviorbot.
"""
import omnigibson as og
from omnigibson.utils.teleop_utils import OVXRSystem

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
    vrsys = OVXRSystem(robot=env.robots[0], show_control_marker=False, system="SteamVR", enable_touchpad_movement=True)
    # We want a lower movement speed for controlling with VR headset
    vrsys.base_movement_speed = 0.03
    vrsys.start()
    # set headset position to be 1m above ground and facing +x
    vrsys.set_initial_transform(pos=[0, 0, 1], orn=[0, 0, 0, 1])

    # main simulation loop
    for _ in range(10000):
        # step the VR system to get the latest data from VR runtime
        vrsys.update()
        # generate robot action and step the environment
        action = vrsys.teleop_data_to_action()
        env.step(action)                

    # Shut down the environment cleanly at the end
    vrsys.stop()
    env.close()

if __name__ == "__main__":
    main()