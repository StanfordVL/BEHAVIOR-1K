import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import torch as th
from omnigibson.eval.utils.network_utils import PolicyTimeoutError, WebsocketClientPolicy
from typing import Optional


__all__ = [
    "LocalPolicy",
    "MultiWebsocketPolicy",
    "PolicyBatchResult",
    "WebsocketPolicy",
]


@dataclass
class PolicyBatchResult:
    actions: th.Tensor
    failures: dict[int, Exception]


class LocalPolicy:
    """
    Local policy that directly queries action from policy,
        outputs zero delta action if policy is None.
    """

    def __init__(self, *args, action_dim: Optional[int] = None, **kwargs) -> None:
        self.policy = None  # To be set later
        self.action_dim = action_dim

    def set_action_dim(self, action_dim: int) -> None:
        self.action_dim = action_dim

    def act(self, obs: dict) -> th.Tensor:
        return self.forward(obs)

    def forward(self, obs: dict, *args, **kwargs) -> th.Tensor:
        """
        Directly return a zero action tensor of the specified action dimension.
        """
        if self.policy is not None:
            return self.policy.act(obs).detach().cpu()
        else:
            assert self.action_dim is not None
            batch_size = None
            if obs:
                first_obs = next(iter(obs.values()))
                if isinstance(first_obs, th.Tensor) and first_obs.ndim > 0:
                    batch_size = first_obs.shape[0]
            shape = (self.action_dim,) if batch_size is None else (batch_size, self.action_dim)
            return th.zeros(shape, dtype=th.float32)

    def reset(self) -> None:
        if self.policy is not None:
            self.policy.reset()

    def set_deadline(self, deadline: Optional[float]) -> None:
        pass


class WebsocketPolicy:
    """
    Websocket policy for controlling the robot over a websocket connection. ``action_chunk_size`` opts
    into the action-chunk protocol documented in ``docs/challenge/evaluation.md``; it is disabled by
    default because replaying a chunk is correct only for servers that return actions intended for
    open-loop execution from one observation.
    """

    def __init__(
        self,
        *args,
        host: Optional[str] = None,
        port: Optional[int] = None,
        allow_reconnect: bool = False,
        action_chunk_size: int = 0,
        **kwargs,
    ) -> None:
        logging.info(f"Creating websocket client policy with host: {host}, port: {port}")
        self.last_action = None
        self.policy = None
        self._allow_reconnect = allow_reconnect
        self._action_chunk_size = action_chunk_size
        self._deadline = None
        if host is not None or port is not None:
            self.policy = WebsocketClientPolicy(
                host=host,
                port=port,
                allow_reconnect=allow_reconnect,
                action_chunk_size=action_chunk_size,
            )

    def update_host(self, host: str, port: int) -> None:
        self.policy = WebsocketClientPolicy(
            host=host,
            port=port,
            allow_reconnect=self._allow_reconnect,
            action_chunk_size=self._action_chunk_size,
        )
        self.policy.set_deadline(self._deadline)

    def set_deadline(self, deadline: Optional[float]) -> None:
        self._deadline = deadline
        if self.policy is not None:
            self.policy.set_deadline(deadline)

    def forward(self, obs: dict, *args, **kwargs) -> th.Tensor:
        if "need_new_action" in obs and not obs["need_new_action"] and self.last_action is not None:
            return self.last_action
        self.last_action = self.policy.act(obs).detach().cpu()
        return self.last_action

    def reset(self) -> None:
        if self.policy is not None:
            self.policy.reset()
        self.last_action = None


class MultiWebsocketPolicy:
    """Query one policy server per logical environment while OmniGibson steps them together."""

    def __init__(self, endpoints: list[dict], allow_reconnect: bool = True, action_chunk_size: int = 0) -> None:
        self.clients = [
            WebsocketClientPolicy(
                host=endpoint["host"],
                port=endpoint["port"],
                allow_reconnect=allow_reconnect,
                action_chunk_size=action_chunk_size,
            )
            for endpoint in endpoints
        ]
        self.executor = ThreadPoolExecutor(max_workers=len(self.clients))
        self.action_dim = None
        self.failures = {}
        self.time_budget = None
        self.elapsed = [0.0] * len(self.clients)

    def set_action_dim(self, action_dim: int) -> None:
        self.action_dim = action_dim

    def set_time_budget(self, seconds: float) -> None:
        self.time_budget = seconds
        self.elapsed = [0.0] * len(self.clients)

    def _run_client(self, env_idx: int, method, *args):
        start = time.monotonic()
        if self.time_budget is not None:
            remaining = self.time_budget - self.elapsed[env_idx]
            if remaining <= 0:
                raise PolicyTimeoutError("Rollout time budget expired")
            self.clients[env_idx].set_deadline(start + remaining)
        try:
            result = method(*args)
        finally:
            self.elapsed[env_idx] += time.monotonic() - start
        if self.time_budget is not None and self.elapsed[env_idx] >= self.time_budget:
            raise PolicyTimeoutError("Rollout time budget expired")
        return result

    def reset(self) -> None:
        self.failures = {}
        futures = {
            self.executor.submit(self._run_client, env_idx, client.reset): env_idx
            for env_idx, client in enumerate(self.clients)
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                self.failures[futures[future]] = e

    def forward(self, obs: dict, active_env_indices: list[int]) -> PolicyBatchResult:
        assert self.action_dim is not None
        actions = th.zeros((len(self.clients), self.action_dim), dtype=th.float32)
        failures = {env_idx: self.failures[env_idx] for env_idx in active_env_indices if env_idx in self.failures}
        futures = {
            self.executor.submit(
                self._run_client,
                env_idx,
                self.clients[env_idx].act,
                {key: value[env_idx : env_idx + 1] for key, value in obs.items()},
            ): env_idx
            for env_idx in active_env_indices
            if env_idx not in failures
        }
        for future in as_completed(futures):
            env_idx = futures[future]
            try:
                action = future.result()
                if action.ndim == 2 and action.shape[0] == 1:
                    action = action[0]
                if action.shape != (self.action_dim,):
                    raise ValueError(f"Policy {env_idx} returned action shape {tuple(action.shape)}")
                actions[env_idx] = action
            except Exception as e:
                self.failures[env_idx] = failures[env_idx] = e
        return PolicyBatchResult(actions=actions, failures=failures)

    def charge_simulation(self, seconds: float, active_env_indices: list[int]) -> dict[int, Exception]:
        share = seconds / len(self.clients)
        failures = {}
        for env_idx in active_env_indices:
            self.elapsed[env_idx] += share
            if self.time_budget is not None and self.elapsed[env_idx] >= self.time_budget:
                error = PolicyTimeoutError("Rollout time budget expired")
                self.failures[env_idx] = failures[env_idx] = error
        return failures

    def close(self) -> None:
        self.executor.shutdown(wait=True)
        for client in self.clients:
            client.close()
