import copy
import torch as th
from omnigibson.metrics.metric_base import MetricBase


class AgentMetric(MetricBase):
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
            self.delta_agent_distance = {part: [] for part in ["base"] + robot.arm_names}
            self.state_cache = copy.deepcopy(self.next_state_cache)
            self.initialized = True

        distance = th.linalg.norm(
            self.next_state_cache["base"]["position"] - self.state_cache["base"]["position"]
        ).item()
        self.delta_agent_distance["base"].append(distance)

        for arm in robot.arm_names:
            eef_distance = th.linalg.norm(
                self.next_state_cache[arm]["position"] - self.state_cache[arm]["position"]
            ).item()
            self.delta_agent_distance[arm].append(eef_distance)

        self.state_cache = copy.deepcopy(self.next_state_cache)

    def gather_results(self):
        return {"agent_distance": {k: sum(v) for k, v in self.delta_agent_distance.items()}}
