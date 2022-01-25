import copy

import numpy as np


from igibson.metrics.metric_base import MetricBase
from igibson.robots.manipulation_robot import IsGraspingState


class RobotMetric(MetricBase):
    def __init__(self):
        self.initialized = False

    def step_callback(self, env, _):
        robot = env.robots[0]
        agent_distance = {part: 0 for part in self.agent_pos}

        self.next_state_cache = {
            "body": {"position": robot.get_position()},
        }.update({arm: {"position": robot.get_eef_position(arm)} for arm in robot.arm_names})

        if not self.initialized:
            self.agent_pos = {part: [] for part in ["base"] + robot.arm_names}
            self.agent_grasping = {part: [] for part in robot.arm_names}

            self.agent_local_pos = {part: [] for part in robot.arm_names}

            self.delta_agent_distance = {part: [] for part in ["base"] + robot.arm_names}
            self.delta_agent_grasp_distance = {part: [] for part in robot.arm_names}

            self.state_cache = copy.deepcopy(self.next_state_cache)
            self.initialized = True

        self.agent_pos["body"].append(list(self.state_cache["body"]["position"]))
        distance = np.linalg.norm(
            np.array(self.next_state_cache["body"]["position"]) - self.state_cache["body"]["position"]
        )
        self.delta_agent_distance["body"].append(distance)

        for arm in robot.arm_names:
            self.agent_pos[arm].append(list(self.state_cache[arm]["position"]))
            gripper_distance = np.linalg.norm(
                self.next_state_cache[arm]["position"] - self.state_cache[arm]["position"]
            )
            self.delta_agent_distance[arm].append(gripper_distance)

            self.agent_local_pos[arm].append(robot.get_eef_position(arm).tolist())

            if robot.is_grasping(arm) == IsGraspingState.TRUE:
                self.delta_agent_grasp_distance[arm].append(gripper_distance)
                self.agent_grasping[arm].append(True)
            else:
                self.delta_agent_grasp_distance[arm].append(0)
                self.agent_grasping[arm].append(False)

        self.state_cache = copy.deepcopy(self.next_state_cache)

    def gather_results(self):
        return {
            "agent_distance": {
                "timestep": self.delta_agent_distance,
            },
            "grasp_distance": {
                "timestep": self.delta_agent_grasp_distance,
            },
            "pos": {
                "timestep": self.agent_pos,
            },
            "local_pos": {
                "timestep": self.agent_local_pos,
            },
            "grasping": {
                "timestep": self.agent_grasping,
            },
        }
