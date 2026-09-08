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
dominate the run. The model is therefore refreshed **only when a primitive
terminates** -- which is precisely when an agent returns to the reasoning stage
and has a reason to look at the world. Nobody reads it in between. Metrics follow the same split
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
        observation_every: optional tick-interval refresh, **off by default**.
            The world model is rebuilt when a primitive terminates -- i.e. when
            an agent returns to the reasoning stage and actually needs to look
            at the world. Rebuilding on a timer instead would recompute the
            scene graph hundreds of times inside a single primitive for nobody
            to read. Set it non-zero only to debug drift.
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
        observation_every: int = 0,
        use_scene_graph: bool = False,
        coop_config_path: Optional[str] = None,
        task_specs=None,
        bddl_activity: Optional[str] = None,
        bddl_instance_id: int = 0,
        video_path: Optional[str] = None,
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
        self.task_specs = task_specs
        # BDDL activity name, e.g. "coop_two_apples_pomaria". When set, the env
        # loads OmniGibson's BehaviorTask from the cached instance and
        # `terminated` follows compiled_task.check_goal -- the activity's own
        # goal expression, evaluated against the simulator.
        self.bddl_activity = bddl_activity
        self.bddl_instance_id = int(bddl_instance_id)

        # Where to write the episode video, or None. The viewer camera has to
        # be enabled *before* the Environment is built (gm.RENDER_VIEWER_CAMERA
        # is read when the camera is created), which is why this is a
        # constructor argument and not a method you call later.
        self.video_path = video_path
        self.recorder = None

        self.goal_reached_at: Optional[int] = None
        self.task_tracker = None
        self._closed = False
        import os as _os
        self.engine_verbose = bool(kwargs.get('engine_verbose') or _os.environ.get('COOP2_ENGINE_VERBOSE'))
        self.outcomes_seen = 0

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
        from coop2.behavior_env.symbolic_navigation import DestinationRegistry  # noqa: PLC0415
        from coop2.behavior_env.cooperative_tasks import (  # noqa: PLC0415
            BehaviorTaskState,
            CoopTaskTracker,
        )
        from coop2.behavior_env.world_state import BehaviorWorldState  # noqa: PLC0415
        from coop2.cognitive.action.behavior_action import BehaviorActionExecutor  # noqa: PLC0415

        gm.USE_GPU_DYNAMICS = False
        gm.ENABLE_TRANSITION_RULES = False
        if self.headless:
            gm.HEADLESS = True
            # Headless and "no viewer camera" are separate: rendering offscreen
            # is exactly how a video gets recorded without a window.
            gm.RENDER_VIEWER_CAMERA = bool(self.video_path)
        if self.video_path:
            from coop2.behavior_env.recording import enable_viewer_rendering  # noqa: PLC0415

            enable_viewer_rendering()

        config = build_multi_robot_config(
            robot_poses=[([1.5 * i, 0.0, 0.05], [0.0, 0.0, 0.0, 1.0]) for i in range(self.n_agents)],
            robot_model=self.robot_model,
            scene_model=self.scene_model,
            load_object_categories=None,
            objects=self.extra_objects,
            agent_names=self.agent_names,
            task=self._task_config(),
        )
        self.env = og.Environment(configs=config)
        self._enforce_controller_config(config)

        # Placement before prepare_robots: config poses are placeholders and
        # nothing has stepped yet, so moving here is free.
        _, self.placement_room = place_robots(self.env, seed=self.seed, room=self.room)
        prepare_robots(self.env)
        assert_multi_robot_sanity(self.env, expected_robots=self.n_agents)

        # One registry for the whole scene: agents reserve the pose they are
        # about to teleport to, so concurrent samplers cannot all pick spots
        # around the same object and interpenetrate on arrival.
        self.destinations = DestinationRegistry()
        self.controllers = {
            agent_id: ContentiousSymbolicActionPrimitives(
                self.env, robot, destinations=self.destinations
            )
            for agent_id, robot in zip(self.agent_names, self.env.robots)
        }
        self.engine = MultiAgentPrimitiveEngine(
            self.env,
            agent_ids=self.agent_names,
            attempts=1,
            enable_head_tracking=False,
            controllers=self.controllers,
            verbose=self.engine_verbose,
        )
        self._start_recording()

        if self.extra_objects:
            place_objects(
                self.env,
                [spec["name"] for spec in self.extra_objects],
                seed=self.seed,
                room=self.placement_room,
                scene_model=self.scene_model,
            )

        self.world = BehaviorWorldState(self.env, use_scene_graph=self.use_scene_graph)
        self.world.start()
        # Before anything asks for an id: the activity's own bindings win, so
        # the ids in the prompt are the ids its goal expression is written in.
        task = getattr(self.env, "task", None)
        if getattr(task, "object_scope", None):
            adopted = self.world.adopt_task_scope(task)
            print(f"[setup] adopted {adopted} entity ids from the BDDL object scope")
        self.executors = {
            agent_id: BehaviorActionExecutor(agent_id, engine=self.engine, world_state=self.world)
            for agent_id in self.agent_names
        }

        # L1d. Exposed as `task_tracker` because plan_log_saver.save_task_log
        # and run_individual reach it by that name on the base env; without it
        # the episode runs to completion and then dies writing its logs.
        self.world.step()
        tasks = self.task_specs if self.task_specs is not None else self._default_tasks()
        self.task_tracker = CoopTaskTracker(self.world, tasks)
        self.capability_history = self.task_tracker.capability_history
        self._loaded = True

    def _start_recording(self) -> None:
        """Frame the task and hook the recorders onto the engine's tick.

        Writes one file per view: an overview of the room, plus a chase view
        per robot. The overview alone was not enough to see what happened --
        two R1s in a living room are small in a shot wide enough to hold the
        whole task -- and a per-robot view shows which object each one is
        actually reaching for.
        """
        if not self.video_path:
            return
        import os as _os  # noqa: PLC0415

        from coop2.behavior_env.recording import (  # noqa: PLC0415
            MultiViewRecorder,
            chain,
            chase_pose,
        )

        base, extension = _os.path.splitext(self.video_path)
        extension = extension or ".mp4"

        # One camera per robot, no room-wide shot. A single overview cannot be
        # framed indoors here: 63 deg cannot hold 4.4 m of task from inside a
        # 4.9 m room, a wide enough lens turns the robots into specks, and the
        # seg map's living_room_0 spans an open-plan boundary so "a traversable
        # cell at the right distance" kept landing behind a wall. A camera over
        # a robot has none of those problems.
        views = {
            robot.name: (lambda robot=robot: chase_pose(robot))
            for robot in self.env.robots
        }

        from coop2.behavior_env.recording import WIDE_FOCAL_LENGTH  # noqa: PLC0415

        self.recorder = MultiViewRecorder(
            views=views,
            path_for=lambda name: f"{base}_{name}{extension}",
            every=4,
            fps=30,
            # Every view is a wide one: the ceiling caps the camera about a
            # metre above the robot's head, and 63 deg from there frames little
            # more than the head.
            focal_lengths={name: WIDE_FOCAL_LENGTH for name in views},
        )
        self.engine.on_tick = chain(self.engine.on_tick, self.recorder)
        print(f"[video] recording {len(views)} views: {', '.join(views)}")

    def _enforce_controller_config(self, config: Dict[str, Any]) -> None:
        """Apply our controller config to whatever robots the scene ended up with.

        ``Environment._load_robots`` skips the entire robots_config when the
        scene already imported robots -- "Only actually load robots if no robot
        has been imported from the scene loading directly yet". A BDDL activity
        sets scene_instance to the cached template, that template contains the
        robots it was sampled with, and so our config is dropped on the floor:
        the robots come up with R1's defaults (IK arms, delta trunk) while
        ``robot._controller_config`` still reports ours. Nothing raises until
        ``q_to_action`` asserts the trunk is a non-delta JointController, which
        happens on the first tick, long after the misconfiguration.

        The symbolic primitives require position-mode JointControllers, so
        assert them here rather than hoping the template was sampled with the
        same robot config.
        """
        wanted = {
            robot_config["name"]: robot_config.get("controller_config")
            for robot_config in config.get("robots", [])
            if robot_config.get("controller_config")
        }
        for robot in self.env.robots:
            controller_config = wanted.get(robot.name)
            if controller_config is None:
                continue
            registered = getattr(robot, "controllers", {}) or {}
            names = sorted(registered)
            if names and all(
                name in controller_config for name in names
            ) and self._controllers_match(robot, controller_config):
                continue
            print(f"[setup] reloading {robot.name}'s controllers to the primitives config")
            robot.reload_controllers(controller_config)

    @staticmethod
    def _controllers_match(robot, controller_config: Dict[str, Any]) -> bool:
        """Do @robot's live controllers already match @controller_config?"""
        from omnigibson.controllers.controller_view import ControllerView  # noqa: PLC0415

        for name, entry in (getattr(robot, "controllers", {}) or {}).items():
            group_key = entry[0] if isinstance(entry, tuple) else entry
            live = ControllerView._controller_groups.get(group_key)
            expected = (controller_config.get(name) or {}).get("name")
            if expected and type(live).__name__ != expected:
                return False
            wants_delta = (controller_config.get(name) or {}).get("use_delta_commands")
            if wants_delta is not None and getattr(live, "use_delta_commands", None) != wants_delta:
                return False
        return True

    def _task_config(self) -> Optional[Dict[str, Any]]:
        """BehaviorTask when a BDDL activity is named, else DummyTask.

        ``online_object_sampling=False`` is the point: the instance was sampled
        once and frozen, so every run loads the same layout. Sampling here
        instead would take ~45 s and lay the objects out differently each time,
        and the whole reason for the cached template is that the contested
        distances stay comparable across runs.
        """
        if not self.bddl_activity:
            return None
        return {
            "type": "BehaviorTask",
            "activity_name": self.bddl_activity,
            "activity_definition_id": 0,
            "activity_instance_id": self.bddl_instance_id,
            "online_object_sampling": False,
            # Must be False, for two independent reasons. The template was
            # sampled without presampled robot poses, so scene metadata has no
            # "robot_poses" key and BehaviorTask.reset dereferences None (the
            # class default is True, despite its docstring saying False). And
            # this facade places robots itself, room-scoped and mutually
            # separated, which a presampled pose would overwrite.
            "use_presampled_robot_pose": False,
        }

    def _default_tasks(self):
        """One "hold this" task per added object.

        A deliberately trivial default so a runner started without a task set
        still produces populated constraint metrics rather than empty ones --
        an empty task list makes every COOP2 constraint read zero, which is
        indistinguishable from a cooperation failure.
        """
        from coop2.behavior_env.cooperative_tasks import BehaviorTaskState  # noqa: PLC0415

        tasks = []
        for spec in self.extra_objects:
            obj = self.env.scene.object_registry("name", spec["name"])
            if obj is None:
                continue
            entity_id = self.world.entity_id_for(obj)
            tasks.append(
                BehaviorTaskState(
                    task_id=f"hold_{entity_id}",
                    target_id=entity_id,
                    goal=("holding", entity_id),
                    required_agents=1,
                )
            )
        return tasks

    def reset(self, seed: Optional[int] = None, get_obs: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if seed is not None:
            self.seed = seed
        if not self._loaded:
            self._build()
        self._pending_outcomes = {}
        self.world.step()
        # Baseline at env_step 0. The tracker only samples when a primitive
        # terminates, so its first snapshot lands hundreds of ticks in, by
        # which time the agents have already walked to their targets -- and
        # _diff records nothing for a snapshot with no predecessor. The most
        # important spatial improvement of the episode, nobody-near to
        # someone-near, was therefore never counted, and C+ read 0 for runs in
        # which both agents did reach an object.
        if self.task_tracker is not None and not self.task_tracker.get_history():
            self.task_tracker.step(0)
        info = self._build_info()
        return self._empty_obs(), info

    def close(self) -> None:
        """Release the env -- but do **not** shut Isaac down here.

        ``og.shutdown()`` ends in ``app.close()``, which terminates the process
        without unwinding Python. crafter's ``close()`` merely tore down an
        object, so the copied runners call it in the middle of their teardown
        and then keep going: ``run_individual`` closes at line 252 and computes
        and writes ``coop2_metrics.json`` at 270-272. Shutting down inside
        ``close()`` meant those twenty lines never ran and the episode produced
        no metrics -- silently, since nothing unwinds far enough to raise.

        So the real shutdown is deferred to interpreter exit. og.sim is a
        process singleton and there is one env per process, so nothing is
        waiting to reuse the GPU in the meantime.
        """
        # The video file is only valid once the encoder is flushed, and the
        # real shutdown below never unwinds, so finalise it here.
        if self.recorder is not None:
            # ViewerRecorder.close() reports the frame count itself.
            self.recorder.close()
            self.recorder = None

        self._closed = True
        self._register_shutdown()

    @staticmethod
    def _register_shutdown() -> None:
        if getattr(CooperativeBehaviorEnv, "_shutdown_registered", False):
            return
        import atexit  # noqa: PLC0415

        def _shutdown() -> None:
            import omnigibson as og  # noqa: PLC0415

            if og.sim is not None:
                og.shutdown()

        atexit.register(_shutdown)
        CooperativeBehaviorEnv._shutdown_registered = True

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
            observation = self.world.observation_for(
                agent_id, max_steps=self.length, env_step=self.engine.env_step
            )
            # Resolve the radius per entity, not once from a probe object. The
            # gate is per-object (a table's radius exceeds an apple's), so a
            # single probe radius -- in practice the smallest object's --
            # labelled large objects "too far" while the gate would have let
            # them through, contradicting what the prompt promises the agent.
            radius = None
            controller = self.controllers.get(agent_id)
            if controller is not None:
                cache: Dict[str, Optional[float]] = {}

                def radius(entity, _controller=controller, _cache=cache):
                    if entity.name not in _cache:
                        obj = self.env.scene.object_registry("name", entity.name)
                        _cache[entity.name] = (
                            _controller.interaction_radius_for(obj) if obj is not None else None
                        )
                    return _cache[entity.name]

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

        # An outcome is exactly the moment an agent goes back to reasoning, so
        # that is the only moment the world model has a reader.
        self.outcomes_seen += len(outcomes)
        if outcomes and self.task_tracker is not None:
            # A terminated primitive is the macro-step boundary: the constraints
            # are defined over decisions, not ticks.
            acting = {agent_id: None for agent_id in outcomes}
            self.task_tracker.step(self.engine.env_step, acting=acting)

        refresh = bool(outcomes) or (
            self.observation_every > 0 and self.engine.env_step % self.observation_every == 0
        )
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
        terminated_all = self._goal_reached()
        return (
            self._empty_obs(),
            {agent_id: 0.0 for agent_id in self.agent_names},
            {agent_id: terminated_all for agent_id in self.agent_names},
            {agent_id: truncated_all for agent_id in self.agent_names},
            info,
        )

    def _goal_reached(self) -> bool:
        """Has the episode's goal been met?

        ``compiled_task.check_goal`` is the only authority: the BDDL activity's
        own goal expression, evaluated against the simulator. A stand-in check
        ("every agent holds an apple") stood here while BDDL was not wired up.
        It is gone: two things deciding `terminated` cannot disagree usefully,
        and a run with no activity should be visibly unbounded rather than
        quietly ending on a proxy for the goal.
        """
        if self.goal_reached_at is not None:
            return True
        if self.bddl_activity:
            return self._bddl_goal_reached()
        return False

    def _bddl_goal_reached(self) -> bool:
        """``compiled_task.check_goal`` against the live scene.

        Note what is *not* called: ``check_initial_conditions``. A BDDL init
        block may contain ``inroom``, which has no entry in
        ``PREDICATE_TO_STATE`` and raises KeyError; the goal expression is a
        different set of predicates and is safe to evaluate every macro-step.
        """
        task = getattr(self.env, "task", None)
        compiled = getattr(task, "compiled_task", None)
        if compiled is None:
            return False
        try:
            met, breakdown = compiled.check_goal(task._evaluate_predicate)
        except Exception as error:  # noqa: BLE001 - a broken predicate must not end the run
            print(f"[goal] check_goal raised {type(error).__name__}: {error}")
            return False
        if met:
            self.goal_reached_at = self.engine.env_step
            print(f"[goal] BDDL goal satisfied at env_step {self.goal_reached_at}: {breakdown}")
        return bool(met)

    def wait_for_state_change(self, timeout: float = 0.05) -> None:
        """No-op: this env is synchronous. crafter's runner polls a thread."""
        return None


#: The copied L6 runners import ``CooperativeEnv`` from ``macrafter``. Aliasing
#: rather than renaming keeps those files byte-identical to upstream.
CooperativeEnv = CooperativeBehaviorEnv
