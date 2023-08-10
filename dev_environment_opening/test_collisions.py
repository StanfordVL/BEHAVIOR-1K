import yaml
import numpy as np
import argparse

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, UndoableContext
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.utils.motion_planning_utils import set_arm_and_detect_collision, set_base_and_detect_collision

import cProfile, pstats, io
import time
import os
import argparse

def pause(time):
    for _ in range(int(time*100)):
        og.sim.step()

def main():
    # Load the config
    config_filename = "test_tiago.yaml"
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config["scene"]["load_object_categories"] = ["floors", "walls", "coffee_table"]

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()


    positions = [
        [0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.5, 0.5, 0]
    ]

    joint_positions = [

    ]

    # from IPython import embed; embed()
    # breakpoint()
    robot.untuck()
    robot.set_position([0, 0, 0.1])
    # pause(100)
    og.sim.step()
    # print(gm.ENABLE_FLATCACHE)
    # meshes = []
    # for link in robot.links.values():
    #     for mesh in link.collision_meshes.values():
    #         if mesh.prim_path == "/World/robot0/l_wheel_link/collisions" or mesh.prim_path == "/World/robot0/r_wheel_link/collisions":
    #             mesh.collision_enabled = False
            # meshes.append(mesh)

    # from IPython import embed; embed()

    # pause(100)

    controller = StarterSemanticActionPrimitives(None, scene, robot)

    # for position in positions:
    #     with UndoableContext(controller.robot, controller.robot_copy, "original", True) as context:
    #         print(set_base_and_detect_collision(context,(position, [0, 0, 0, 1])))
    #         print("--------------------")

    # pause(100)

    # joint_combined_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx["combined"]])
    # initial_joint_pos = np.array(robot.get_joint_positions()[joint_combined_idx])

    pose_2d = [-0.762831, -0.377231, 2.72892]
    pose = controller._get_robot_pose_from_2d_pose(pose_2d)
    robot.set_position_orientation(*pose)
    og.sim.step()

    joint_control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]])
    joint_combined_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx["combined"]])
    initial_joint_pos = np.array(robot.get_joint_positions()[joint_combined_idx])
    control_idx_in_joint_pos = np.where(np.in1d(joint_combined_idx, joint_control_idx))[0]
    joint_pos = initial_joint_pos
    # joint_pos[control_idx_in_joint_pos] = [0.0133727 ,0.216775 ,0.683931 ,2.04371 ,1.88204 ,0.720747 ,1.23276 ,1.72251]
    from IPython import embed; embed()

    with UndoableContext(controller.robot, controller.robot_copy, "original") as context:
        print(set_arm_and_detect_collision(context, joint_pos))
        print("--------------------")
        for i in range(10000):
            og.sim.render()


    pause(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run test script")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="If set, profile code and generate .prof file",
    )
    args = parser.parse_args()
    if args.profile:
        pr = cProfile.Profile()
        pr.enable()
        main()
        pr.disable()
        results = pstats.Stats(pr)
        filename = f'profile-{os.path.basename(__file__)}-{time.strftime("%Y%m%d-%H%M%S")}'
        results.dump_stats(f"./profiles/{filename}.prof")
        os.system(f"flameprof ./profiles/{filename}.prof > ./profiles/{filename}.svg")
        # Run `snakeviz ./profiles/<filename>.prof` to visualize stack trace or open <filename>.svg in a browser
    else:
        main()