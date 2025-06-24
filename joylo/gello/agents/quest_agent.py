from typing import Dict

import numpy as np
import quaternion
from dm_control import mjcf
from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from oculus_reader.reader import OculusReader

from gello.agents.agent import Agent
from gello.agents.spacemouse_agent import apply_transfer, mj2ur, ur2mj
from gello.dm_control_tasks.arms.ur5e import UR5e

# cartensian space control, controller <> robot relative pose matters. This extrinsics is based on
# our setup, for details please checkout the project page.
quest2ur = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
ur2quest = np.linalg.inv(quest2ur)

translation_scaling_factor = 2.0


class SingleArmQuestAgent(Agent):
    def __init__(self, robot_type: str, which_hand: str, verbose: bool = False) -> None:
        """Interact with the robot using the quest controller.

        leftTrig: press to start control (also record the current position as the home position)
        leftJS: a tuple of (x,y) for the joystick, only need y to control the gripper
        """
        self.which_hand = which_hand
        assert self.which_hand in ["l", "r"]

        self.oculus_reader = OculusReader()
        if robot_type == "ur5":
            _robot = UR5e()
        else:
            raise ValueError(f"Unknown robot type: {robot_type}")
        self.physics = mjcf.Physics.from_mjcf_model(_robot.mjcf_model)
        self.control_active = False
        self.reference_quest_pose = None
        self.reference_ee_rot_ur = None
        self.reference_ee_pos_ur = None

        self.robot_type = robot_type
        self._verbose = verbose

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        if self.robot_type == "ur5":
            num_dof = 6
        current_qpos = obs["joint_positions"][:num_dof]  # last one dim is the gripper
        current_gripper_angle = obs["joint_positions"][-1]
        # run the fk
        self.physics.data.qpos[:num_dof] = current_qpos
        self.physics.step()

        ee_rot_mj = np.array(
            self.physics.named.data.site_xmat["attachment_site"]
        ).reshape(3, 3)
        ee_pos_mj = np.array(self.physics.named.data.site_xpos["attachment_site"])
        if self.which_hand == "l":
            pose_key = "l"
            trigger_key = "leftTrig"
            # joystick_key = "leftJS"
            # left yx
            gripper_open_key = "Y"
            gripper_close_key = "X"
        elif self.which_hand == "r":
            pose_key = "r"
            trigger_key = "rightTrig"
            # joystick_key = "rightJS"
            # right ba for the key
            gripper_open_key = "B"
            gripper_close_key = "A"
        else:
            raise ValueError(f"Unknown hand: {self.which_hand}")
        # check the trigger button state
        pose_data, button_data = self.oculus_reader.get_transformations_and_buttons()
        if len(pose_data) == 0 or len(button_data) == 0:
            print("no data, quest not yet ready")
            return np.concatenate([current_qpos, [current_gripper_angle]])

        new_gripper_angle = current_gripper_angle
        if button_data[gripper_open_key]:
            new_gripper_angle = 1
        if button_data[gripper_close_key]:
            new_gripper_angle = 0
        arm_not_move_return = np.concatenate([current_qpos, [new_gripper_angle]])
        if len(pose_data) == 0:
            print("no data, quest not yet ready")
            return arm_not_move_return

        trigger_state = button_data[trigger_key][0]
        if trigger_state > 0.5:
            if self.control_active is True:
                if self._verbose:
                    print("controlling the arm")
                current_pose = pose_data[pose_key]
                delta_rot = current_pose[:3, :3] @ np.linalg.inv(
                    self.reference_quest_pose[:3, :3]
                )
                delta_pos = current_pose[:3, 3] - self.reference_quest_pose[:3, 3]
                delta_pos_ur = (
                    apply_transfer(quest2ur, delta_pos) * translation_scaling_factor
                )
                # ? is this the case?
                delta_rot_ur = quest2ur[:3, :3] @ delta_rot @ ur2quest[:3, :3]
                if self._verbose:
                    print(
                        f"delta pos and rot in ur space: \n{delta_pos_ur}, {delta_rot_ur}"
                    )
                next_ee_rot_ur = delta_rot_ur @ self.reference_ee_rot_ur
                next_ee_pos_ur = delta_pos_ur + self.reference_ee_pos_ur

                target_quat = quaternion.as_float_array(
                    quaternion.from_rotation_matrix(ur2mj[:3, :3] @ next_ee_rot_ur)
                )
                ik_result = qpos_from_site_pose(
                    self.physics,
                    "attachment_site",
                    target_pos=apply_transfer(ur2mj, next_ee_pos_ur),
                    target_quat=target_quat,
                    tol=1e-14,
                    max_steps=400,
                )
                self.physics.reset()
                if ik_result.success:
                    new_qpos = ik_result.qpos[:num_dof]
                else:
                    print("ik failed, using the original qpos")
                    return arm_not_move_return
                command = np.concatenate([new_qpos, [new_gripper_angle]])
                return command

            else:  # last state is not in active
                self.control_active = True
                if self._verbose:
                    print("control activated!")
                self.reference_quest_pose = pose_data[pose_key]

                self.reference_ee_rot_ur = mj2ur[:3, :3] @ ee_rot_mj
                self.reference_ee_pos_ur = apply_transfer(mj2ur, ee_pos_mj)
                return arm_not_move_return
        else:
            if self._verbose:
                print("deactive control")
            self.control_active = False
            self.reference_quest_pose = None
            return arm_not_move_return


class DualArmQuestAgent(Agent):
    def __init__(self, robot_type: str) -> None:
        self.left_arm = SingleArmQuestAgent(robot_type, "l")
        self.right_arm = SingleArmQuestAgent(robot_type, "r")

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        pass


if __name__ == "__main__":
    oculus_reader = OculusReader()
    while True:
        """
        example output:
        ({'l': array([[-0.828395 ,  0.541667 , -0.142682 ,  0.219646 ],
        [-0.107737 ,  0.0958919,  0.989544 , -0.833478 ],
        [ 0.549685 ,  0.835106 , -0.0210789, -0.892425 ],
        [ 0.       ,  0.       ,  0.       ,  1.       ]]), 'r': array([[-0.328058,  0.82021 ,  0.468652, -1.8288  ],
        [ 0.070887,  0.516083, -0.8536  , -0.238691],
        [-0.941994, -0.246809, -0.227447, -0.370447],
        [ 0.      ,  0.      ,  0.      ,  1.      ]])},
        {'A': False, 'B': False, 'RThU': True, 'RJ': False, 'RG': False, 'RTr': False, 'X': False, 'Y': False, 'LThU': True, 'LJ': False, 'LG': False, 'LTr': False, 'leftJS': (0.0, 0.0), 'leftTrig': (0.0,), 'leftGrip': (0.0,), 'rightJS': (0.0, 0.0), 'rightTrig': (0.0,), 'rightGrip': (0.0,)})

        """
        pose_data, button_data = oculus_reader.get_transformations_and_buttons()
        if len(pose_data) == 0:
            print("no data")
            continue
        else:
            print(pose_data["l"])
