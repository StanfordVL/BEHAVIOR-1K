import logging
import torch as th
from omnigibson.eval.utils.network_utils import WebsocketClientPolicy
from typing import Optional


__all__ = [
    "LocalPolicy",
    "WebsocketPolicy",
]


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

    def forward(self, obs: dict, *args, **kwargs) -> th.Tensor:
        if "need_new_action" in obs and not obs["need_new_action"] and self.last_action is not None:
            return self.last_action
        self.last_action = self.policy.act(obs).detach().cpu()
        return self.last_action

    def reset(self) -> None:
        if self.policy is not None:
            self.policy.reset()
        self.last_action = None
