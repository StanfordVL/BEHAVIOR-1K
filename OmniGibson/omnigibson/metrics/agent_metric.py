import copy
import torch as th
from omnigibson.metrics.metric_base import MetricBase
from typing import Optional


class AgentMetric(MetricBase):
    def __init__(self, human_stats: Optional[dict] = None):
        self.initialized = False
        self.human_stats = human_stats
        if human_stats is None:
            print("No human stats provided.")
        else:
            self.human_stats = {
                "base": self.human_stats["distance_traveled"],
                "left": self.human_stats["left_eef_displacement"],
                "right": self.human_stats["right_eef_displacement"],
            }

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
        results = {
            "agent_distance": {k: sum(v) for k, v in self.delta_agent_distance.items()},
        }
        results.update(
            {
                "normalized_agent_distance": {
                    k: (self.human_stats[k] / v if v != 0 else float("inf"))
                    for k, v in results["agent_distance"].items()
                }
            }
        )
        return results
