from typing import Any, Dict, Protocol, List

import numpy as np
import torch as th


class Agent(Protocol):
    def __init__(self):
        self._started = False

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        """Returns an action given an observation.

        Args:
            obs: observation from the environment.

        Returns:
            action: action to take on the environment.
        """
        raise NotImplementedError

    def reset(self):
        """Executes any necessary reset protocols. Default merely resets internal variables"""
        self._started = False

    def start(self):
        """Executes any necessary starting protocols. Default merely sets the initialized variable"""
        self._started = True


class DummyAgent(Agent):
    def __init__(self, num_dofs: int):
        self.num_dofs = num_dofs

        super().__init__()

    def act(self, obs: Dict[str, Any]) -> th.tensor:
        return np.zeros(self.num_dofs)


class BimanualAgent(Agent):
    def __init__(self, agent_left: Agent, agent_right: Agent):
        self.agent_left = agent_left
        self.agent_right = agent_right

        super().__init__()

    def act(self, obs: Dict[str, Any]) -> th.tensor:
        left_obs = {}
        right_obs = {}
        for key, val in obs.items():
            L = val.shape[0]
            half_dim = L // 2
            assert L == half_dim * 2, f"{key} must be even, something is wrong"
            left_obs[key] = val[:half_dim]
            right_obs[key] = val[half_dim:]
        return np.concatenate(
            [self.agent_left.act(left_obs), self.agent_right.act(right_obs)]
        )


class MultiControllerAgent(Agent):
    def __init__(self, agents: List[Agent]):
        self.agents = agents

        super().__init__()

    def act(self, obs: Dict[str, Any]) -> th.tensor:
        return th.concatenate([agent.act(obs) for agent in self.agents])

    def reset(self):
        super().reset()

        # Reset all owned agents
        for agent in self.agents:
            agent.reset()

    def start(self):
        super().start()

        # Start all owned agents
        for agent in self.agents:
            agent.start()
