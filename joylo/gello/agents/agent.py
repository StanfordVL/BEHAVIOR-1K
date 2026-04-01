from typing import Any, Dict

import numpy as np


class Agent:
    def __init__(self):
        self._started = False

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        raise NotImplementedError

    def reset(self):
        self._started = False

    def start(self):
        self._started = True


class DummyAgent(Agent):
    def __init__(self, num_dofs: int):
        self.num_dofs = num_dofs
        super().__init__()

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        return np.zeros(self.num_dofs)
