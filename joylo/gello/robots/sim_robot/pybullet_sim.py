from typing import Dict
import os

import numpy as np
import pybullet as pb
import pybullet_data

from gello.asset_root import ASSET_ROOT
from gello.robots.sim_robot.zmq_server import ZMQRobotServer, ZMQServerThread


class PyBulletRobotServer:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5556,
        print_joints: bool = False,
    ):
        self._client_id = pb.connect(pb.GUI)
        pb.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS, 1)
        pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())

        pb.loadURDF(
            "plane.urdf",
            [0, 0, -0.001],
            physicsClientId=self._client_id,
        )
        # load franka
        self._franka = pb.loadURDF(
            os.path.join(ASSET_ROOT, "panda", "panda.urdf"),
            useFixedBase=1,
            physicsClientId=self._client_id,
            flags=pb.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
        )
        pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 1)
        pb.resetDebugVisualizerCamera(
            cameraDistance=1.4,
            cameraYaw=45,
            cameraPitch=-45,
            cameraTargetPosition=np.array([0.0, 0.0, 0.3]),
            physicsClientId=self._client_id,
        )

        joints = [
            pb.getJointInfo(self._franka, i, physicsClientId=self._client_id)
            for i in range(
                pb.getNumJoints(self._franka, physicsClientId=self._client_id)
            )
        ]
        self._arm_joints = [j[0] for j in joints if j[2] == pb.JOINT_REVOLUTE]
        self._gripper_joints = [j[0] for j in joints if j[2] == pb.JOINT_PRISMATIC]

        self.eef_ind = (
            pb.getNumJoints(self._franka, physicsClientId=self._client_id) - 1
        )

        self._joint_state = np.zeros(8)
        self._joint_cmd = self._joint_state

        self._zmq_server = ZMQRobotServer(robot=self, host=host, port=port)
        self._zmq_server_thread = ZMQServerThread(self._zmq_server)

    def num_dofs(self) -> int:
        return 8

    def get_joint_state(self) -> np.ndarray:
        return self._joint_state

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        assert len(joint_state) == 8, len(joint_state)
        _joint_state = joint_state.copy()
        self._joint_cmd = _joint_state

    def freedrive_enabled(self) -> bool:
        return True

    def set_freedrive_mode(self, enable: bool):
        pass

    def get_observations(self) -> Dict[str, np.ndarray]:
        arm_joint_states = pb.getJointStates(
            self._franka, self._arm_joints, physicsClientId=self._client_id
        )
        arm_joint_pos = np.array([j[0] for j in arm_joint_states])
        arm_joint_vel = np.array([j[1] for j in arm_joint_states])

        gripper_joint_states = pb.getJointStates(
            self._franka, self._gripper_joints, physicsClientId=self._client_id
        )
        gripper_width = gripper_joint_states[0][0] * 2

        joint_positions = np.concatenate([arm_joint_pos, [gripper_width]])
        joint_velocities = np.concatenate([arm_joint_vel, [0]])

        eef_state = pb.getLinkState(
            self._franka,
            self.eef_ind,
            physicsClientId=self._client_id,
        )
        eef_pos, eef_quat = eef_state[0], eef_state[1]
        eef_pos_quat = np.concatenate([eef_pos, eef_quat])

        return {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "gripper_position": gripper_width,
            "ee_pos_quat": eef_pos_quat,
        }

    def serve(self) -> None:
        # start the zmq server
        self._zmq_server_thread.start()
        while True:
            arm_cmd, gripper_cmd = self._joint_cmd[:-1], self._joint_cmd[-1]
            for joint_ind, joint_value in zip(self._arm_joints, arm_cmd):
                pb.resetJointState(
                    self._franka,
                    joint_ind,
                    joint_value,
                    physicsClientId=self._client_id,
                )
            gripper_cmd = (1 - gripper_cmd) * 0.04
            for joint_id in self._gripper_joints:
                pb.resetJointState(
                    self._franka,
                    joint_id,
                    gripper_cmd,
                    physicsClientId=self._client_id,
                )

            pb.stepSimulation(physicsClientId=self._client_id)

    def stop(self) -> None:
        self._zmq_server_thread.join()
        pb.disconnect(physicsClientId=self._client_id)

    def __del__(self) -> None:
        self.stop()


if __name__ == "__main__":
    sim = PyBulletRobotServer()
    sim.serve()
    print("done")
