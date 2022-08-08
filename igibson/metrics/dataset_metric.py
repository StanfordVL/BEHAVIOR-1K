from collections import defaultdict

from igibson.metrics.metric_base import MetricBase


class DatasetMetric(MetricBase):
    """
    Used to transform saved demos into (state, action) pairs
    """

    def __init__(self):
        self.prev_state = None
        self.state_history = defaultdict(list)

    def start_callback(self, env, _):
        self.prev_state = env.get_state()

    def step_callback(self, env, log_reader):
        # when called, action is at timestep t and the state is at timestep t-1, so we need to use the previous state
        self.prev_state["action"] = log_reader.get_agent_action("vr_robot")
        for key, value in self.prev_state.items():
            self.state_history[key].append(value)
        self.prev_state = env.get_state()

    def gather_results(self):
        return self.state_history
