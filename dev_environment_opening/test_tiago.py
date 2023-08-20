import yaml
import numpy as np
import argparse

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, UndoableContext
from omnigibson.objects.primitive_object import PrimitiveObject
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject

import cProfile, pstats, io
import time
import os
import argparse
    

def pause(time):
    for _ in range(int(time*100)):
        og.sim.step()

def replay_controller(env, filename):
    actions = yaml.load(open(filename, "r"), Loader=yaml.FullLoader)
    for action in actions:
        env.step(action)

def execute_controller(ctrl_gen, env, filename=None):
    actions = []
    for action in ctrl_gen:
        env.step(action)
        actions.append(action.tolist())
    if filename is not None:
        with open(filename, "w") as f:
            yaml.dump(actions, f)

def main():
    # Load the config
    config_filename = "test_tiago.yaml"
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls", "coffee_table"]

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # table = DatasetObject(
    #     name="table",
    #     category="breakfast_table",
    #     model="rjgmmy",
    #     scale = 0.3
    # )
    # og.sim.import_object(table)
    # table.set_position([-0.7, 0.5, 0.2])

    grasp_obj = DatasetObject(
        name="cologne",
        category="cologne",
        model="lyipur",
        scale=[0.01, 0.01, 0.01]
    )
    og.sim.import_object(grasp_obj)
    grasp_obj.set_position([-0.3, -0.8, 0.5])
    og.sim.step()

    # marker = PrimitiveObject(
    #     prim_path=f"/World/marker",
    #     name="marker",
    #     primitive_type="Cube",
    #     size=0.07,
    #     visual_only=True,
    #     rgba=[1.0, 0, 0, 1.0],
    # )
    # og.sim.import_object(marker)
    # marker.set_position_orientation([-0.29840604, -0.79821703,  0.59273211], [0.        , 0.70710678, 0.        , 0.70710678])
    # og.sim.step()


    # robot.set_position([-2.0, 0.0, 0.0])
    pause(2)
    controller = StarterSemanticActionPrimitives(None, scene, robot)

    def set_start_pose():
        reset_pose_tiago = np.array([
            -1.78029833e-04,  3.20231302e-05, -1.85759447e-07, -1.16488536e-07,
            4.55182843e-08,  2.36128806e-04,  1.50000000e-01,  9.40000000e-01,
            -1.10000000e+00,  0.00000000e+00, -0.90000000e+00,  1.47000000e+00,
            0.00000000e+00,  2.10000000e+00,  2.71000000e+00,  1.50000000e+00,
            1.71000000e+00,  1.30000000e+00, -1.57000000e+00, -1.40000000e+00,
            1.39000000e+00,  0.00000000e+00,  0.00000000e+00,  4.50000000e-02,
            4.50000000e-02,  4.50000000e-02,  4.50000000e-02,
        ])
        robot.set_joint_positions(reset_pose_tiago)
        og.sim.step()

    def test_navigate_to_obj():
        # Need to set start pose to reset_hand because default tuck pose for Tiago collides with itself
        execute_controller(controller._reset_hand(), env)
        execute_controller(controller._navigate_to_obj(table), env)

    def test_grasp_no_navigation():
        # Need to set start pose to reset_hand because default tuck pose for Tiago collides with itself
        set_start_pose()
        pose = controller._get_robot_pose_from_2d_pose([-0.433881, -0.210183, -2.96118])
        robot.set_position_orientation(*pose)
        og.sim.step()
        # replay_controller(env, "./replays/test_grasp_pose.yaml")
        execute_controller(controller._grasp(grasp_obj), env)

    def test_grasp():
        # Need to set start pose to reset_hand because default tuck pose for Tiago collides with itself
        # execute_controller(controller._reset_hand(), env)
        set_start_pose()
        # pause(2)
        execute_controller(controller._grasp(grasp_obj), env)

    def test_place():
        test_grasp()
        pause(1)
        execute_controller(controller._place_on_top(table), env)
    
    # positions = [[0.09988395869731903, -1.0999969244003296, 1.4699827432632446, 2.710009813308716, 1.710004448890686, -1.5700018405914307, 1.3899955749511719, 2.001982011279324e-07], [0.15954897383415584, -0.9759483151785584, 1.051426922254121, 1.3919954813427862, 1.9255247232751793, -0.46858315638703396, 1.135518807525537, 0.5174528326963662], [0.15062408833826937, -0.4437143998615267, 0.8304433521196042, 1.437534104367112, 1.6164805582338932, -0.37533100951328124, 0.6381778036539293, -0.0283867578914061], [0.13373369762078724, 0.5635409634642365, 0.41223083348993295, 1.5237161776241306, 1.0316129205581803, -0.1988508522420569, -0.30304254915485307, -1.0613909301407032], [0.11684330690330513, 1.57079632679, -0.005981685139738268, 1.6098982508811492, 0.4467452828824673, -0.022370694970832543, -1.244262901963635, -2.09439510239]]
    control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx["left"]])
    # robot.set_position_orientation([-1.08215380e+00, -3.35281938e-01, -2.77837131e-07], [ 1.78991655e-07, -4.65450078e-08, -2.67762393e-01,  9.63485003e-01])
    # og.sim.step()
    # for p in positions:
    #     robot.set_joint_positions(p, control_idx)
    #     pause(0.2)
    # pause(100)
    # Work more reliably
    # og.sim._physics_context.set_gravity(value=-0.1)
    # execute_controller(controller._reset_hand(), env)
    # robot.set_position_orientation([-1.08215380e+00, -3.35281938e-01, -2.77837131e-07], [ 1.78991655e-07, -4.65450078e-08, -2.67762393e-01,  9.63485003e-01])
    # og.sim.step()
    # execute_controller(controller._reset_hand(), env)
    # grasp_pose = [-0.29840604, -0.79821703,  0.59273211], [0.        , 0.70710678, 0.        , 0.70710678]
    # execute_controller(controller._reset_hand(), env)
    # execute_controller(controller._move_hand(grasp_pose), env)
    # test_grasp_no_navigation()
    
    # end_joint = [0.11684331,  1.57079633, -0.00598169,  1.60989825,  0.44674528, -0.02237069, -1.2442629, -2.0943951]
    # robot.set_joint_positions(end_joint, control_idx)
    # pause(1)
    # from IPython import embed; embed()
    # og.sim.step()
    # Don't work as reliably

    # pose_2d = [-0.543999, -0.233287,-1.16071]
    # pos = np.array([pose_2d[0], pose_2d[1], 0.05])
    # orn = T.euler2quat([0, 0, pose_2d[2]])
    # robot.set_position_orientation(pos, orn)
    # og.sim.step()
    # pause(10)
    # t_pose = ([-0.29840604, -0.79821703,  0.59273211], [0.        , 0.70710678, 0.        , 0.70710678])
    # execute_controller(controller._reset_hand(), env)
    # execute_controller(controller._move_hand(t_pose), env)

    # try:
    #     test_grasp_no_navigation()
    # except:
    #     pass
    

    # test_grasp_no_navigation()
    test_grasp()

    # test_grasp_no_navigation()
    # set_start_pose()
    # pose_2d = [-0.15, -0.269397, -2.0789]
    # pos = np.array([pose_2d[0], pose_2d[1], 0.05])
    # orn = T.euler2quat([0, 0, pose_2d[2]])
    # robot.set_position_orientation(pos, orn)
    # pause(2)

    # with UndoableContext(robot, "base") as context:        
    #     print(not detect_robot_collision(context, (pos, orn)))
    #     for i in range(10000):
    #         og.sim.render()
    # test_grasp()
    # test_place()

    # replay_controller(env, "./replays/tiago_grasp.yaml")
    # execute_controller(controller.place_on_top(table), env)
    # from IPython import embed; embed()
    # execute_controller(controller._navigate_to_pose([-0.3, -2.3, 0.0]), env)
    # execute_controller(controller._navigate_to_pose(pose_2d), env)

    # pause(100)

    ###################################################################################
    # Random test code below
    ###################################################################################
    # def detect_robot_collision(robot, filter_objs=[]):
    #     filter_categories = ["floors"]
        
    #     obj_in_hand = robot._ag_obj_in_hand[robot.default_arm]
    #     if obj_in_hand is not None:
    #         filter_objs.append(obj_in_hand)

    #     collision_prims = list(robot.states[ContactBodies].get_value(ignore_objs=tuple(filter_objs)))

    #     for col_prim in collision_prims:
    #         tokens = col_prim.prim_path.split("/")
    #         obj_prim_path = "/".join(tokens[:-1])
    #         col_obj = og.sim.scene.object_registry("prim_path", obj_prim_path)
    #         if col_obj.category in filter_categories:
    #             collision_prims.remove(col_prim)

    #     return len(collision_prims) > 0 or detect_self_collision(robot)

    # def detect_self_collision(robot):
    #     contacts = robot.contact_list()
    #     robot_links = [link.prim_path for link in robot.links.values()]
    #     disabled_pairs = [set(p) for p in robot.disabled_collision_pairs]
    #     for c in contacts:
    #         link0 = c.body0.split("/")[-1]
    #         link1 = c.body1.split("/")[-1]
    #         if {link0, link1} not in disabled_pairs and c.body0 in robot_links and c.body1 in robot_links:
    #             return True
    #     return False

    # robot.set_position([-0.1, -0.35, 0.05])
    # robot.set_orientation(T.euler2quat([0, 0,-np.pi/1.5]))
    # og.sim.step()
    # control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx["left"]])
    # joint_pos = [0.284846, 1.22316, 0.323617, 1.72149, 1.4959, -0.31599, -1.4043, 0.152401]
    # robot.set_joint_positions(joint_pos, control_idx)
    # og.sim.step()
    # while True:
    #     coll = []
    #     robot_links = [link.prim_path for link in robot.links.values()]
    #     for c in robot.contact_list():
    #         if c.body0 in robot_links and c.body1 in robot_links:
    #             link0 = c.body0.split("/")[-1]
    #             link1 = c.body1.split("/")[-1]
    #             pair = {link0, link1}
    #             if pair not in coll:
    #                 coll.append(pair)
        
    #     print(coll)
    #     print(detect_robot_collision(robot))
    #     print(detect_self_collision(robot))
    #     print("---------------")
    #     pause(2)


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
        s = io.StringIO()
        results = pstats.Stats(pr)
        filename = f'profile-{os.path.basename(__file__)}-{time.strftime("%Y%m%d-%H%M%S")}'
        results.dump_stats(f"./profiles/{filename}.prof")
        os.system(f"flameprof ./profiles/{filename}.prof > ./profiles/{filename}.svg")
        # Run `snakeviz ./profiles/<filename>.prof` to visualize stack trace or open <filename>.svg in a browser
    else:
        main()



