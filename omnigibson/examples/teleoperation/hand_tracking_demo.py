"""
Example script for using hand tracking (OpenXR only) with dexterous hand.
You can set DEBUG_MODE to True to visualize the landmarks of the hands! 
"""
import omnigibson as og
from omnigibson.utils.teleop_utils import OVXRSystem

DEBUG_MODE = True  # set to True to visualize the landmarks of the hands

def main():
    # Create the config for generating the environment we want
    env_cfg = {"action_timestep": 1 / 60., "physics_timestep": 1 / 120.}
    scene_cfg = {"type": "Scene"}
    robot0_cfg = {
        "type": "Behaviorbot",
        "obs_modalities": ["rgb", "depth", "normal", "scan", "occupancy_grid"],
        "action_normalize": False,
        "grasping_mode": "assisted"
    }
    object_cfg = [
        {
            "type": "DatasetObject",
            "prim_path": "/World/breakfast_table",
            "name": "breakfast_table",
            "category": "breakfast_table",
            "model": "kwmfdg",
            "bounding_box": [2, 1, 0.4],
            "position": [0.8, 0, 0.3],
            "orientation": [0, 0, 0.707, 0.707],
        },
        {
            "type": "DatasetObject",
            "prim_path": "/World/apple",
            "name": "apple",
            "category": "apple",
            "model": "omzprq",
            "position": [0.6, 0.1, 0.5],
        },
        {
            "type": "DatasetObject",
            "prim_path": "/World/banana",
            "name": "banana",
            "category": "banana",
            "model": "znakxm",
            "position": [0.6, -0.1, 0.5],
        },
    ]
    if DEBUG_MODE:
        # Add the marker to visualize hand tracking landmarks
        object_cfg.extend([{
            "type": "PrimitiveObject",
            "prim_path": f"/World/marker_{i}",
            "name": f"marker_{i}",  
            "primitive_type": "Cube",
            "size": 0.01,
            "visual_only": True,
            "rgba": [0.0, 1.0, 0.0, 1.0],
        } for i in range(52)])

    cfg = dict(env=env_cfg, scene=scene_cfg, robots=[robot0_cfg], objects=object_cfg)
    # Create the environment
    env = og.Environment(configs=cfg)
    env.reset()

    if DEBUG_MODE:
        markers = [env.scene.object_registry("name", f"marker_{i}") for i in range(52)]
    
    # Start vrsys 
    vrsys = OVXRSystem(robot=env.robots[0], show_control_marker=False, system="OpenXR", use_hand_tracking=True)
    vrsys.start()
    # set headset position to be 1m above ground and facing +x direction
    head_init_transform = vrsys.og2xr(pos=[0, 0, 1], orn=[0, 0, 0, 1])
    vrsys.vr_profile.set_physical_world_to_world_anchor_transform_to_match_xr_device(head_init_transform, vrsys.hmd)

    # main simulation loop
    for _ in range(10000):
        if og.sim.is_playing():
            vrsys.update()
            if DEBUG_MODE:
                if "left" in vrsys.raw_data["hand_data"] and "raw" in vrsys.raw_data["hand_data"]["left"]:
                    for i in range(26):
                        pos = vrsys.raw_data["hand_data"]["left"]["raw"]["pos"][i]
                        orn = vrsys.raw_data["hand_data"]["left"]["raw"]["orn"][i]
                        markers[i].set_position_orientation(pos, orn)
                if "right" in vrsys.raw_data["hand_data"] and "raw" in vrsys.raw_data["hand_data"]["right"]:
                    for i in range(26):
                        pos = vrsys.raw_data["hand_data"]["right"]["raw"]["pos"][i]
                        orn = vrsys.raw_data["hand_data"]["right"]["raw"]["orn"][i]
                        markers[i + 26].set_position_orientation(pos, orn)
            action = vrsys.teleop_data_to_action()
            env.step(action)                

    # Shut down the environment cleanly at the end
    vrsys.stop()
    env.close()

if __name__ == "__main__":
    main()