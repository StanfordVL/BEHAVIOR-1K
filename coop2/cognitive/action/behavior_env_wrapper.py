"""L2 wrapper: the seam between COOP2's plan layer and ``CooperativeBehaviorEnv``.

``SymbolicEnvWrapper`` converts each symbolic action into a crafter primitive
*integer* and passes a dict of those to ``env.step``. Nothing about that
survives here -- a BEHAVIOR primitive is a generator that runs for 10^2-10^3
ticks, and the facade takes the symbolic action dict itself. So ``step`` is
overridden wholesale rather than adapted.

Everything the plan layer above actually touches is kept identical: the
user-name <-> env-id mapping, ``_map_dict``, ``_last_info``, the per-agent
``check_termination_condition`` call after the step, and the five-tuple return.

The one structural difference the plan layer must live with: an agent whose
primitive is still in flight is *not* re-issued. L3 hands down
``plan.get_current_action()`` on every step, and re-assigning a running
primitive would raise, so repeats are dropped here rather than upstream.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from coop2.cognitive.action.action_env_wrapper import SymbolicEnvWrapper
from coop2.cognitive.action.behavior_action import BehaviorActionExecutor

__all__ = ["BehaviorSymbolicEnvWrapper"]


class BehaviorSymbolicEnvWrapper(SymbolicEnvWrapper):
    """``SymbolicEnvWrapper`` speaking BEHAVIOR primitives."""

    def __init__(self, env, agent_names: List[str], **env_kwargs):
        super().__init__(env, agent_names=agent_names, **env_kwargs)
        self._adopt_facade_executors()

    def _adopt_facade_executors(self) -> None:
        """Share the facade's executors instead of owning a second set.

        There must be exactly one executor per agent. The facade's ``step()``
        calls ``execute()`` on *its* executors, while L3 reads plan progress
        from ``get_action_records()`` on the wrapper's. With two sets the
        wrapper's were never executed, so their history stayed empty,
        ``action_status`` came back None, and L3 never advanced the plan --
        the same action was re-issued forever. It cost a 30-minute timeout to
        find, because from the outside it looks exactly like a slow primitive.
        """
        facade_executors = getattr(self.env, "executors", None)
        if not facade_executors:
            self.agent_actions = {
                user_id: BehaviorActionExecutor(
                    self.name_map[user_id],
                    engine=getattr(self.env, "engine", None),
                    world_state=getattr(self.env, "world", None),
                )
                for user_id in self.user_agent_names
            }
            return
        self.agent_actions = {
            user_id: facade_executors[self.name_map[user_id]]
            for user_id in self.user_agent_names
            if self.name_map[user_id] in facade_executors
        }

    def _rebind_executors(self) -> None:
        """Re-adopt after reset(), which rebuilds the facade's executors."""
        self._adopt_facade_executors()
        for executor in self.agent_actions.values():
            executor.engine = getattr(self.env, "engine", None)
            executor.world_state = getattr(self.env, "world", None)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        result = super().reset(seed=seed, options=options)
        self._rebind_executors()
        return result

    def step(self, symbolic_actions: Dict[str, Dict[str, Any]]):
        """One tick. Issues any newly-available symbolic actions, then advances."""
        self._rebind_executors()

        pending: Dict[str, Dict[str, Any]] = {}
        for user_id, symbolic_action in (symbolic_actions or {}).items():
            if user_id not in self.name_map or not symbolic_action:
                continue
            env_id = self.name_map[user_id]
            engine = getattr(self.env, "engine", None)
            if engine is not None and engine.has_active(env_id):
                # Already running this action; L3 re-sends it every step.
                continue
            action_type = symbolic_action.get("action_type", "wait")
            args = symbolic_action.get("args")
            if args is None:
                args = {k: v for k, v in symbolic_action.items() if k != "action_type"}
            pending[env_id] = {"action_type": action_type, **args}

        obs, rewards, terminated, truncated, info = self.env.step(pending)
        self._env_step_count += 1

        mapped_info = self._map_dict(info)
        world_state = None
        if mapped_info:
            first = next(iter(mapped_info.values()))
            if isinstance(first, dict):
                world_state = first.get("symbolic_world_state")

        for user_id, executor in self.agent_actions.items():
            executor.current_env_step = self._env_step_count
            agent_info = mapped_info.get(user_id, {}) if isinstance(mapped_info, dict) else {}
            action_outcome = agent_info.get("action_outcome") if isinstance(agent_info, dict) else None
            if isinstance(action_outcome, dict):
                action_outcome = dict(action_outcome, agent_id=user_id)
                agent_info["action_outcome"] = action_outcome
            executor.check_termination_condition(world_state, action_outcome=action_outcome)

        self._last_observations = self._map_dict(obs)
        self._last_info = mapped_info
        return (
            self._last_observations,
            self._map_dict(rewards),
            self._map_dict(terminated),
            self._map_dict(truncated),
            mapped_info,
        )
