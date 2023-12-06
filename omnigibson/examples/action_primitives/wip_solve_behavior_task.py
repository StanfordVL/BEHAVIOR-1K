"""
Example script demo'ing robot primitive to solve a task
"""
import os
import yaml
import numpy as np

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, StarterSemanticActionPrimitiveSet

# Don't use GPU dynamics and use flatcache for performance boost
# gm.USE_GPU_DYNAMICS = True
# gm.ENABLE_FLATCACHE = True

def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        env.step(action)

def set_start_pose(robot):
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

def main():
    # Load the config
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Update it to run a grocery shopping task
    config["scene"]["scene_model"] = "Benevolence_1_int"
    config["scene"]["load_task_relevant_only"] = True
    config["scene"]["not_load_object_categories"] = ["ceilings"]
    config["task"] = {
        "type": "BehaviorTask",
        "activity_name": "picking_up_trash",
        "activity_definition_id": 0,
        "activity_instance_id": 0,
        "predefined_problem": None,
        "online_object_sampling": False,
    }

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    controller = StarterSemanticActionPrimitives(env)
    set_start_pose(robot)

    # Grasp can of soda
    grasp_obj = scene.object_registry("name", "can_of_soda_89")
    print("Executing controller")
    execute_controller(controller.apply_ref(StarterSemanticActionPrimitiveSet.GRASP, grasp_obj), env)
    print("Finished executing grasp")

    # Place can in trash can
    print("Executing controller")
    trash = scene.object_registry("name", "trash_can_85")
    execute_controller(controller.apply_ref(StarterSemanticActionPrimitiveSet.PLACE_INSIDE, trash), env)
    print("Finished executing place")

if __name__ == "__main__":
    main()