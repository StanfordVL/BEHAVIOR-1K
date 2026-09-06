"""L1: ``CooperativeBehaviorEnv`` -- the seam COOP2's upper layers plug into.

Replaces ma_crafter's ``macrafter.coop_env.CooperativeEnv``. The shape is fixed
by what L2-L6 already expect (PORTING_PLAN 5.1/5.2) and is deliberately not
negotiable: a five-tuple of dicts from ``step``, and **``info[agent_id]`` as the
real observation channel**. Agents never read ``obs`` -- in crafter it was RGB
that ``observe()`` stored and nothing consumed -- so everything the LLM can see
travels in ``info``:

===========================  ============================================
key                          consumer
===========================  ============================================
``symbolic_world_state``     L2 grounding and termination checks
``symbolic_view`` (str)      the ``## Symbolic View`` prompt section
``target_hints`` (str)       the ``## Current Reachable Targets`` section
``action_outcome`` (dict)    the L2 -> L3 success/failure contract
``task_states`` (dict)       process logging, from L1d (M6)
===========================  ============================================

One ``step()`` is one ``engine.tick()``, i.e. one ``env.step``. That is the
right granularity for the engine but the wrong one for the world model: a
primitive spans 10^2-10^3 ticks, and rebuilding the scene graph on each would
dominate the run. So the model is refreshed only when a primitive actually
terminated, or every ``observation_every`` ticks. Metrics follow the same split
the engine already draws: ``env_step`` counts ticks, ``decision_count`` counts
primitives, and it is ``decision_count`` that is COOP2's denominator.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["CooperativeBehaviorEnv", "CooperativeEnv"]


class CooperativeBehaviorEnv:
    """N robots in one BEHAVIOR scene, driven by symbolic actions.

    Args:
        scene_model: defaults to ``house_single_floor``. Rs_int is measured
            unusable -- 96% of sampled base poses put an R1 where it does not
            fit, and its best room holds one robot.
        n_agents: robots to spawn; placement is derived from the scene.
        length: tick budget, i.e. what ``Timeout`` counts.
        room: room instance to place everyone in. ``None`` picks the largest
            indoor one.
        objects: extra objects to add, as env-config dicts.
        observation_every: refresh the world model at least this often (ticks),
            in addition to whenever a primitive terminates.
    """

    def __init__(
        self,
        scene_model: str = "house_single_floor",
        n_agents: int = 2,
        seed: Optional[int] = None,
        length: int = 20000,
        robot_model: str = "R1",
        room: Optional[str] = None,
        objects: Optional[Sequence[Dict[str, Any]]] = None,
        headless: bool = True,
        observation_every: int = 200,
        use_scene_graph: bool = True,
        coop_config_path: Optional[str] = None,
        **kwargs: Any,
    ):
        # crafter kwargs (area/view/size/n_players/reward) arrive from the
        # copied runners. They describe a 2D grid world and have no meaning
        # here; accept and ignore rather than crash, but record them so a
        # confused caller can see they were dropped.
        self.ignored_kwargs = dict(kwargs)
        if "n_players" in kwargs:
            n_agents = int(kwargs["n_players"])

        self.scene_model = scene_model
        self.n_agents = int(n_agents)
        self.seed = seed
        self.length = int(length)
        self.robot_model = robot_model
        self.room = room
        self.extra_objects = list(objects or [])
        self.headless = headless
        self.observation_every = int(observation_every)
        self.use_scene_graph = use_scene_graph
        self.coop_config_path = coop_config_path

        self.agent_names: List[str] = [f"agent_{i}" for i in range(self.n_agents)]
        self.possible_agents: List[str] = list(self.agent_names)

        self.env = None
        self.engine = None
        self.world = None
        self.controllers: Dict[str, Any] = {}
        self.executors: Dict[str, Any] = {}
        self.placement_room: Optional[str] = None
        self._time_limit_seconds: Optional[float] = None
        self._pending_outcomes: Dict[str, Dict[str, Any]] = {}
        self._last_info: Dict[str, Any] = {}
        self._loaded = False

    # -- lifecycle ---------------------------------------------------------

    @property
    def agents(self) -> List[str]:
        return list(self.agent_names)

    @property
    def current_step(self) -> int:
        return self.engine.env_step if self.engine is not None else 0

    @property
    def decision_count(self) -> int:
        """Primitives issued -- COOP2's metric denominator, not ticks."""
        return self.engine.decision_count if self.engine is not None else 0

    def set_team_score_time_limit(self, seconds: Optional[float]) -> None:
        self._time_limit_seconds = seconds

    def _build(self) -> None:
        import omnigibson as og  # noqa: PLC0415
        from omnigibson.macros import gm  # noqa: PLC0415

        from coop2.behavior_env.env_setup import (  # noqa: PLC0415
            assert_multi_robot_sanity,
            build_multi_robot_config,
            prepare_robots,
        )
        from coop2.behavior_env.placement import place_objects, place_robots  # noqa: PLC0415
        from coop2.behavior_env.primitive_engine import MultiAgentPrimitiveEngine  # noqa: PLC0415
        from coop2.behavior_env.symbolic_contention import (  # noqa: PLC0415
            ContentiousSymbolicActionPrimitives,
        )
        from coop2.behavior_env.world_state import BehaviorWorldState  # noqa: PLC0415
        from coop2.cognitive.action.behavior_action import BehaviorActionExecutor  # noqa: PLC0415

        gm.USE_GPU_DYNAMICS = False
        gm.ENABLE_TRANSITION_RULES = False
        if self.headless:
            gm.HEADLESS = True
            gm.RENDER_VIEWER_CAMERA = False

        config = build_multi_robot_config(
            robot_poses=[([1.5 * i, 0.0, 0.05], [0.0, 0.0, 0.0, 1.0]) for i in range(self.n_agents)],
            robot_model=self.robot_model,
            scene_model=self.scene_model,
            load_object_categories=None,
            objects=self.extra_objects,
            agent_names=self.agent_names,
        )
        self.env = og.Environment(configs=config)

        # Placement before prepare_robots: config poses are placeholders and
        # nothing has stepped yet, so moving here is free.
        _, self.placement_room = place_robots(self.env, seed=self.seed, room=self.room)
        prepare_robots(self.env)
        assert_multi_robot_sanity(self.env, expected_robots=self.n_agents)

        self.controllers = {
            agent_id: ContentiousSymbolicActionPrimitives(self.env, robot)
            for agent_id, robot in zip(self.agent_names, self.env.robots)
        }
        self.engine = MultiAgentPrimitiveEngine(
            self.env,
            agent_ids=self.agent_names,
            attempts=1,
            enable_head_tracking=False,
            controllers=self.controllers,
            verbose=False,
        )

        if self.extra_objects:
            probe = next(iter(self.controllers.values()))
            first = self.env.scene.object_registry("name", self.extra_objects[0]["name"])
            if first is not None:
                place_objects(
                    self.env,
                    [spec["name"] for spec in self.extra_objects],
                    annulus=probe.sampling_range_for(first),
                    seed=self.seed,
                    room=self.placement_room,
                )

        self.world = BehaviorWorldState(self.env, use_scene_graph=self.use_scene_graph)
        self.world.start()
        self.executors = {
            agent_id: BehaviorActionExecutor(agent_id, engine=self.engine, world_state=self.world)
            for agent_id in self.agent_names
        }
        self._loaded = True

    def reset(self, seed: Optional[int] = None, get_obs: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if seed is not None:
            self.seed = seed
        if not self._loaded:
            self._build()
        self._pending_outcomes = {}
        self.world.step()
        info = self._build_info()
        return self._empty_obs(), info

    def close(self) -> None:
        import omnigibson as og  # noqa: PLC0415

        if og.sim is not None:
            og.shutdown()

    def render(self) -> None:
        return None

    # -- observation -------------------------------------------------------

    def _empty_obs(self) -> Dict[str, Any]:
        # Symbolic only: every robot is configured with obs_modalities=[] so it
        # drops out of the observation space entirely. Agents read info.
        return {agent_id: {} for agent_id in self.agent_names}

    def _build_info(self) -> Dict[str, Any]:
        from coop2.behavior_env.symbolic_view import render_symbolic_view, target_hints  # noqa: PLC0415

        info: Dict[str, Any] = {}
        for agent_id in self.agent_names:
            observation = self.world.observation_for(agent_id, max_steps=self.length)
            radius = None
            controller = self.controllers.get(agent_id)
            if controller is not None and observation.entities:
                probe = next(
                    (e for e in observation.entities.values() if not e.is_robot and not e.is_fixed), None
                )
                if probe is not None:
                    obj = self.env.scene.object_registry("name", probe.name)
                    if obj is not None:
                        radius = controller.interaction_radius_for(obj)
            hints = target_hints(observation, interaction_radius=radius)
            info[agent_id] = {
                "symbolic_world_state": observation,
                "symbolic_view": render_symbolic_view(observation, interaction_radius=radius),
                "target_hints": "\n".join(
                    f"{hint.primitive}({hint.target_id})" + (f"  # {hint.note}" if hint.note else "")
                    for hint in hints
                ),
                "action_outcome": self._pending_outcomes.get(agent_id),
                "task_states": {},  # L1d, M6
            }
        self._last_info = info
        return info

    # -- stepping ----------------------------------------------------------

    def step(
        self,
        actions: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """One tick. Assigns any new symbolic actions first, then advances.

        ``actions`` maps agent id to either a symbolic-action dict
        (``{"action_type": ..., "target": ...}``) or None. An agent whose
        primitive is still in flight is left alone -- re-assigning would raise,
        and the plan layer is allowed to send the same intent repeatedly.

        Extra keyword arguments are accepted and ignored: ``SymbolicEnvWrapper``
        probes the signature for ``share_requests``/``place_requests``/etc. and
        treats ``**kwargs`` as support for all of them.
        """
        if not self._loaded:
            raise RuntimeError("Call reset() before step().")

        self._pending_outcomes = {}
        for agent_id, action in (actions or {}).items():
            if not action or self.engine.has_active(agent_id):
                continue
            payload = dict(action)
            action_type = payload.pop("action_type", None) or payload.pop("type", None)
            if action_type is None:
                continue
            self.executors[agent_id].execute(action_type, current_step=self.engine.env_step, **payload)

        outcomes = self.engine.tick()
        for agent_id, outcome in outcomes.items():
            payload = outcome.to_dict()
            self._pending_outcomes[agent_id] = payload
            self.executors[agent_id].submit_outcome(payload)

        refresh = bool(outcomes) or (self.engine.env_step % self.observation_every == 0)
        if refresh:
            self.world.step()
            info = self._build_info()
        else:
            # Reuse the cached view -- but never the cached outcome. An outcome
            # is true for exactly the tick its primitive terminated on, and
            # serving a stale one made every subsequent action report its
            # predecessor's success the moment it was issued, so a plan
            # "completed" without three of its four primitives ever running.
            info = {
                agent_id: {**payload, "action_outcome": self._pending_outcomes.get(agent_id)}
                for agent_id, payload in (self._last_info or self._build_info()).items()
            }

        truncated_all = self.engine.env_step >= self.length
        return (
            self._empty_obs(),
            {agent_id: 0.0 for agent_id in self.agent_names},
            {agent_id: False for agent_id in self.agent_names},
            {agent_id: truncated_all for agent_id in self.agent_names},
            info,
        )

    def wait_for_state_change(self, timeout: float = 0.05) -> None:
        """No-op: this env is synchronous. crafter's runner polls a thread."""
        return None


#: The copied L6 runners import ``CooperativeEnv`` from ``macrafter``. Aliasing
#: rather than renaming keeps those files byte-identical to upstream.
CooperativeEnv = CooperativeBehaviorEnv
