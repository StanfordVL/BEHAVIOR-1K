import yaml
import numpy as np

import omnigibson as og
from omnigibson.macros import gm
from pxr import PhysxSchema
    

def pause(time):
    for _ in range(int(time*100)):
        # og.sim.render()
        og.sim.step()

def detect_self_collision(robot):
    contacts = robot.contact_list()
    robot_links = [link.prim_path for link in robot.links.values()]
    # impulse_matrix = RigidContactAPI.get_impulses(robot_links, robot_links)
    # return np.max(impulse_matrix) > 0.0
    for c in contacts:
        if c.body0 in robot_links and c.body1 in robot_links:
            return True
    return False

class UndoableContext(object):
    def __init__(self, robot):
        self.robot = robot

    def __enter__(self):
        self.obj_in_hand = self.robot._ag_obj_in_hand[self.robot.default_arm]
        # Store object in hand and the link of the object attached to the robot gripper to manually restore later
        if self.obj_in_hand is not None:
            obj_ag_link_path = self.robot._ag_obj_constraint_params[self.robot.default_arm]['ag_link_prim_path']
            for link in self.obj_in_hand._links.values():
                if link.prim_path == obj_ag_link_path:
                    self.obj_in_hand_link = link
                    break

        self.state = og.sim.dump_state(serialized=False)
        og.sim._physics_context.set_gravity(value=0.0)
        for obj in og.sim.scene.objects:
            for link in obj.links.values():
                PhysxSchema.PhysxRigidBodyAPI(link.prim).GetSolveContactAttr().Set(False)
            obj.keep_still()

    def __exit__(self, *args):
        og.sim.load_state(self.state, serialized=False)
        og.sim.step()
        # from IPython import embed; embed()
        if self.obj_in_hand is not None and not self.robot._ag_obj_constraint_params[self.robot.default_arm]:
            self.robot._establish_grasp(ag_data=(self.obj_in_hand, self.obj_in_hand_link))
        og.sim._physics_context.set_gravity(value=-9.81)
        for obj in og.sim.scene.objects:
            for link in obj.links.values():
                PhysxSchema.PhysxRigidBodyAPI(link.prim).GetSolveContactAttr().Set(True)
        og.sim.step()

def main():
    # Load the config
    config_filename = "test.yaml"
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls", "coffee_table"]

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()


    def test_collision(joint_pos):
        with UndoableContext(robot):
            set_joint_position(joint_pos)
            og.sim.step()
            print(detect_self_collision(robot))
            print("-------")
            pause(2)

    def set_joint_position(joint_pos):
        joint_control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]])
        robot.set_joint_positions(joint_pos, joint_control_idx)


    robot.untuck()
    og.sim.step()
    joint_control_idx = np.concatenate([robot.trunk_control_idx, robot.arm_control_idx[robot.default_arm]])
    joint_pos = robot.get_joint_positions()[joint_control_idx]
    while True:
        test_collision(joint_pos)
        og.sim.step()


if __name__ == "__main__":
    main()
