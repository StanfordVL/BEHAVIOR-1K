import yaml
import numpy as np
import argparse

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
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

    table = DatasetObject(
        name="table",
        category="breakfast_table",
        model="rjgmmy",
        scale = 0.3
    )
    og.sim.import_object(table)
    table.set_position([1.0, 1.0, 0.2])

    grasp_obj = DatasetObject(
        name="potato",
        category="cologne",
        model="lyipur",
        scale=0.01
    )
    og.sim.import_object(grasp_obj)
    grasp_obj.set_position([-0.3, -0.8, 0.5])
    og.sim.step()

    controller = StarterSemanticActionPrimitives(None, scene, robot)


    def test_navigate_to_obj():
        # Need to set start pose to reset_hand because default tuck pose for Tiago collides with itself
        execute_controller(controller._reset_hand(), env)
        execute_controller(controller._navigate_to_obj(table), env)

    def test_grasp_no_navigation():
        # Need to set start pose to reset_hand because default tuck pose for Tiago collides with itself
        execute_controller(controller._reset_hand(), env)
        robot.set_position([-0.1, -0.35, 0.05])
        robot.set_orientation(T.euler2quat([0, 0,-np.pi/1.5]))
        og.sim.step()
        execute_controller(controller.grasp(grasp_obj), env)

    def test_grasp():
        # Need to set start pose to reset_hand because default tuck pose for Tiago collides with itself
        execute_controller(controller._reset_hand(), env)
        execute_controller(controller.grasp(grasp_obj), env)

    def test_place():
        test_grasp()
        pause(1)
        execute_controller(controller.place_on_top(table), env)


    # Work more reliably
    test_navigate_to_obj()
    
    # Don't work as reliably
    # test_grasp_no_navigation()
    # test_grasp()
    # test_place()

    pause(5)

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



