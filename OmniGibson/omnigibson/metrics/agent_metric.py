import copy
import torch as th
from omnigibson.metrics.metric_base import MetricBase


class RobotMetric(MetricBase):
    def __init__(self):
        self.initialized = False

    def start_callback(self, env):
        self.initialized = False

    def step_callback(self, env):
        robot = env.robots[0]
        self.next_state_cache = {
            "base": {"position": robot.get_position_orientation()[0]},
            **{arm: {"position": robot.get_eef_position(arm)} for arm in robot.arm_names},
        }

        if not self.initialized:
            self.agent_pos = {part: [] for part in ["base"] + robot.arm_names}
            self.agent_local_pos = {part: [] for part in robot.arm_names}

            self.delta_agent_distance = {part: [] for part in ["base"] + robot.arm_names}

            self.state_cache = copy.deepcopy(self.next_state_cache)
            self.initialized = True

        self.agent_pos["base"].append(list(self.state_cache["base"]["position"]))
        distance = th.linalg.norm(self.next_state_cache["base"]["position"] - self.state_cache["base"]["position"])
        self.delta_agent_distance["base"].append(distance)

        for arm in robot.arm_names:
            self.agent_pos[arm].append(list(self.state_cache[arm]["position"]))
            gripper_distance = th.linalg.norm(
                self.next_state_cache[arm]["position"] - self.state_cache[arm]["position"]
            )
            self.delta_agent_distance[arm].append(gripper_distance)
            self.agent_local_pos[arm].append(robot.get_eef_position(arm).tolist())

        self.state_cache = copy.deepcopy(self.next_state_cache)

    def gather_results(self):
        return {
            "agent_distance": {
                "timestep": self.delta_agent_distance,
            },
            "pos": {
                "timestep": self.agent_pos,
            },
            "local_pos": {
                "timestep": self.agent_local_pos,
            },
        }
