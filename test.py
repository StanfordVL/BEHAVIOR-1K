import os

import yaml
import numpy as np

import omnigibson as og
from omnigibson.utils.ui_utils import choose_from_options
from omnigibson.utils.asset_utils import (
    get_all_object_categories,
    get_og_avg_category_specs,
    get_object_models_of_category,
)
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, UndoableContext
import omnigibson.utils.transform_utils as T
from omnigibson.objects import PrimitiveObject
from omnigibson.object_states import ContactBodies
from omnigibson.objects.dataset_object import DatasetObject

def pause(time):
    for _ in range(int(time*100)):
        # og.sim.render()
        og.sim.step()

def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        # print(action)
        if action is None:
            og.sim.step()
        else:
            env.step(action)
        

def main(random_selection=False, headless=False, short_exec=False):
    """
    Prompts the user to select a type of scene and loads a turtlebot into it, generating a Point-Goal navigation
    task within the environment.

    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)
    # Load the config
    config_filename = "test.yaml"
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls", "coffee_table", "desk"]
    # config["scene"]["load_object_categories"] = None

    # config["objects"] = [obj_cfg]
    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    # table_cfg = dict(
    #     type="DatasetObject",
    #     name="table",
    #     category="breakfast_table",
    #     model="rjgmmy",
    #     scale=0.9,
    #     position=[0, 0, 0.58],
    # )

    table = DatasetObject(
        name="table",
        category="breakfast_table",
        model="rjgmmy",
        scale = 0.3,
        position=[2.0, 2.0, 0.58]
    )
    og.sim.import_object(table)
    table.set_position([1.0, 1.0, 0.58])

    marker = PrimitiveObject(
        prim_path=f"/World/marker",
        name="marker",
        primitive_type="Cube",
        size=0.07,
        visual_only=False,
        rgba=[1.0, 0, 0, 1.0],
    )

    og.sim.import_object(marker)
    marker.set_position([-0.3, -0.8, 0.5])
    og.sim.step()

    
    # from omnigibson.utils.sim_utils import land_object
    # land_object(robot, np.zeros(3), z_offset=0.05)

    # for link_name in ["l_wheel_link", "r_wheel_link"]:
    #     link = robot.links[link_name]
    #     for col_mesh in link.collision_meshes.values():
    #         col_mesh.set_collision_approximation("boundingSphere")
    # og.sim.step()

    # from IPython import embed; embed()

    # robot.set_position([-1.01199755, -0.30455528, 0.0])
    # robot.set_orientation(T.euler2quat([0, 0, 5.64995586]))
    robot.set_position([0.0, -0.5, 0.05])
    robot.set_orientation(T.euler2quat([0, 0, -np.pi/1.5]))
    og.sim.step()


    controller = StarterSemanticActionPrimitives(None, scene, robot)
    # navigate_controller = controller._navigate_to_pose_direct([0.0, -1.0, 0.0], low_precision=True)
    

    # navigate_controller = controller._navigate_to_pose([0.5, 2.5, 0.0])
    # navigate_controller = controller._navigate_to_pose([0.0, 0.5, 0.0])
    navigate_controller_marker = controller._navigate_to_obj(marker)
    navigate_controller_table = controller._navigate_to_obj(table)


    # robot.untuck()
    start_joint_pos = np.array(
        [
            0.0,
            0.0,  # wheels
            0.2,  # trunk
            0.0,
            1.1707963267948966,
            0.0,  # head
            1.4707963267948965,
            -0.4,
            1.6707963267948966,
            0.0,
            1.5707963267948966,
            0.0,  # arm
            0.05,
            0.05,  # gripper
        ]
    )
    robot.set_joint_positions(start_joint_pos)
    og.sim.step()
    hand_controller = controller.grasp(marker)

    # execute_controller(hand_controller, env)

    # execute_controller(navigate_controller_marker, env)
    execute_controller(hand_controller, env)
    pause(1)
    execute_controller(navigate_controller_table, env)

    # while True:
    #     pause(1)
    #     action = next(navigate_controller)
    #     state, reward, done, info = env.step(action)
        # collision_objects = list(filter(lambda obj : "floor" not in obj.name, robot.states[ContactBodies].get_value()))
        # if len(collision_objects) > 0:
        #     print(collision_objects[0].name)
        # print(collision_objects)

    # for action in navigate_controller:
    #     state, reward, done, info = env.step(action)
    #     if done:
    #         og.log.info("Episode finished after {} timesteps".format(i + 1))
    #         break

    # # Always close the environment at the end
    # env.close()


# def check_contact(obj1, obj2):
#     # One way, use the Object State API
#     from omnigibson.object_states import ContactBodies
#     # Other way, directly check for Touching
#     in_contact = obj1.states[Touching].get_value(obj2)

#     # This returns list of links obj1 is in contact with
#     in_contact = len(set(obj2.links.values()).intersection(obj1.states[ContactBodies].get_value())) > 0

#     # Last way (most fine grained)
#     contact_list = obj1.contact_list()


# def check_contact_multiple_times_and_then_revert(robot):
#     # Goal:
#     # 1. Check if robot is in contact with anything N times
#     # 2. Revert state to the initial state deterministically

#     # Pseudocode:
#     # 1. Record sim state
#     # 2. Modify some omni physics flags to "prepare" for checking
#     # 3. Run contact checks
#     # 4. Revert state
#     # 5. Revert physics flags

#     # 1.
#     state = og.sim.dump_state(serialized=False)

#     # 2.
#     # We need to (a) disable gravity, (b) disable "solving contacts", (c) make sure all objects are "kept still"
#     og.sim.set_gravity(0.0)
#     for obj in og.sim.scene.objects:
#         for link in obj.links.values():
#             PhysxSchema.PhysxRigidBodyAPI(link.prim).GetSolveContactAttr().Set(False)
#         obj.keep_still()

#     # 3.
#     for pos in robot_poses:
#         robot.set_position(pos)
#         check_contact(robot)

#     # 4.
#     og.sim.load_state(state, serialized=False)

#     # 5.
#     og.sim.set_gravity(9.81)
#     for obj in og.sim.scene.objects:
#         for link in obj.links.values():
#             link.set_attribute("SolveContacts", True)



if __name__ == "__main__":
    main()
