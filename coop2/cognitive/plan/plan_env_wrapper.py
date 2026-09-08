"""
Planning environment wrapper for MA-Crafter.

This wrapper sits on top of SymbolicEnvWrapper and manages plan execution:
- Steps through agent plans one primitive action at a time
- Tracks plan status (in progress, success, failure) after each step
- Triggers plan regeneration when plans terminate
"""

import time
import re
from typing import Dict, List, Optional, Any, Callable
from coop2._repair_shim import (
    Coop2RepairController,
    Coop2TraceLogger,
    PettingZooParallelAdapter,
)
# The BEHAVIOR subclass, not crafter's base class. The base converts each
# symbolic action into a crafter *integer* action id, which this facade cannot
# execute: it hands over {"agent_0": 0}, the facade's `if not action` skips it
# because 0 is falsy, and the plan then sits in "executing" forever with no
# primitive ever assigned. Nothing raises; the episode just runs out of steps.
from ..action.behavior_env_wrapper import BehaviorSymbolicEnvWrapper as SymbolicEnvWrapper
from .plan import SymbolicPlan, SymbolicPlanExecutor, SymbolicPlanLogger
from .coop2_process_logger import Coop2ProcessLogger
from .coop2_repair_dispatcher import Coop2RepairDispatcher
from .plan_log_saver import PlanningLogSaver
from ..agent import Agent


class PlanningEnvWrapper:
    """
    Wrapper that manages plan-based agent execution.

    Handles:
    - Executing one primitive action per agent per step (from their current plans)
    - Tracking plan status and detecting termination
    - Triggering plan generation when needed
    - Synchronous stepping of all agents

    Usage:
        env = PlanningEnvWrapper(base_env, agent_names=['alice', 'bob'])
        env.set_plan_generators({'alice': alice_planner, 'bob': bob_planner})

        obs, info = env.reset()

        while not done:
            # Wrapper handles plan execution and regeneration internally
            obs, rewards, terminated, truncated, info = env.step()
    """

    def __init__(
        self,
        base_env,
        agent_names: List[str],
        agents: Optional[Dict[str, Agent]] = None,
        logger: Optional[SymbolicPlanLogger] = None,
        coop2_adapter: Optional[Any] = None,
        coop2_precheck_enabled: bool = False,
        **env_kwargs
    ):
        """
        Initialize the planning wrapper.

        Args:
            base_env: The base MA-Crafter environment class or instance
            agent_names: List of agent names
            agents: Optional dict mapping agent_id to Agent instances
            logger: Optional shared logger for plan tracking
            coop2_adapter: Optional adapter for generic COOP2 task/constraint checks
            coop2_precheck_enabled: If True, run the paper-method pre-execution
                constraint check before primitive environment steps. Defaults to
                False so baseline experiments keep the previous execution path.
            **env_kwargs: Additional environment arguments
        """
        # Create the symbolic wrapper
        self.symbolic_env = SymbolicEnvWrapper(base_env, agent_names=agent_names, **env_kwargs)
        self.agent_names = agent_names

        # Store agents dict (initially empty)
        self.agents: Dict[str, Optional[Agent]] = {aid: None for aid in agent_names}

        # Create shared logger
        self.logger = logger or SymbolicPlanLogger()
        self.coop2_trace = Coop2TraceLogger()
        self.coop2_adapter = coop2_adapter or self._default_coop2_adapter()
        self.coop2_repair_controller = Coop2RepairController(
            adapter=self.coop2_adapter,
            trace_logger=self.coop2_trace,
            enabled=coop2_precheck_enabled,
        )

        # Create plan executors for each agent (with None agent references initially)
        self.plan_executors: Dict[str, SymbolicPlanExecutor] = {}
        for agent_id in agent_names:
            self.plan_executors[agent_id] = SymbolicPlanExecutor(
                agent_id=agent_id,
                agent=None,  # Will be set when agents are provided
                plan_generator=None,  # Set later via set_plan_generators or use agents
                logger=self.logger
            )

        # Track previous action results for each agent
        self._prev_action_results: Dict[str, Optional[Dict]] = {aid: None for aid in agent_names}

        # Track current observations for plan generation
        self._current_obs: Dict[str, Any] = {}
        self._current_step: int = 0

        # The agent view is rebuilt only at a decision boundary; see
        # _agent_view_signature. Between boundaries nothing about the plan has
        # changed, so re-serialising it every tick is pure overhead.
        self._cached_agent_views: List[Dict[str, Any]] = []
        self._last_agent_view_signature: Optional[tuple] = None

        # Compact per-step process trace for COOP2 case-study plots/metrics.
        self.coop2_process_logger = Coop2ProcessLogger(
            symbolic_env=self.symbolic_env,
            agent_names=self.agent_names,
            agents=self.agents,
            coop2_adapter=self.coop2_adapter,
        )
        self.coop2_repair_dispatcher = Coop2RepairDispatcher(
            agents=self.agents,
            message_broker_getter=lambda: self._message_broker,
        )
        self.log_saver = PlanningLogSaver(self)

        # Message broker for coordinated message handling (created when agents are set)
        self._message_broker = None

        # Event notification for state changes (event-driven interrupt handling)
        import threading
        self._state_change_event = threading.Condition()

        # Set agents if provided (this creates the message broker and assigns it to agents)
        if agents:
            self.set_agents(agents)

    def set_coop2_precheck_enabled(self, enabled: bool):
        """Enable or disable the COOP2 pre-execution repair gate."""
        self.coop2_repair_controller.set_enabled(enabled)

    def enable_coop2_precheck(self):
        """Enable pre-execution COOP2 constraint checking."""
        self.set_coop2_precheck_enabled(True)

    def disable_coop2_precheck(self):
        """Disable pre-execution COOP2 constraint checking."""
        self.set_coop2_precheck_enabled(False)

    @property
    def coop2_precheck_enabled(self) -> bool:
        """Whether the COOP2 pre-execution repair gate is enabled."""
        return self.coop2_repair_controller.enabled

    @coop2_precheck_enabled.setter
    def coop2_precheck_enabled(self, enabled: bool):
        self.coop2_repair_controller.set_enabled(enabled)

    @property
    def coop2_evaluator(self):
        """Backwards-compatible access to the controller's evaluator."""
        return self.coop2_repair_controller.evaluator

    @property
    def coop2_process_log(self) -> List[Dict[str, Any]]:
        """Backwards-compatible access to compact COOP2 process records."""
        return self.coop2_process_logger.records

    def _default_coop2_adapter(self):
        """Choose the richest available COOP2 adapter for the wrapped env."""
        base_env = getattr(self.symbolic_env, "env", self.symbolic_env)
        if hasattr(base_env, "task_tracker"):
            try:
                from coop2._repair_shim import MacrafterCoopAdapter
                return MacrafterCoopAdapter(self.symbolic_env)
            except ImportError:
                pass
        return PettingZooParallelAdapter(self.symbolic_env)

    def set_plan_generators(self, generators: Dict[str, Callable]):
        """
        Set plan generator functions for agents.

        Args:
            generators: Dict mapping agent_id to generator function
                Generator signature: (agent_id, observation, env_step) -> SymbolicPlan
        """
        for agent_id, generator in generators.items():
            if agent_id in self.plan_executors:
                self.plan_executors[agent_id].plan_generator = generator

    def set_plan_generator(self, agent_id: str, generator: Callable):
        """Set plan generator for a single agent."""
        if agent_id in self.plan_executors:
            self.plan_executors[agent_id].plan_generator = generator

    def set_agents(self, agents: Dict[str, Agent]):
        """
        Set Agent instances for autonomous plan generation.
        Automatically creates a MessageBroker and assigns it to all agents.

        Args:
            agents: Dict mapping agent_id to Agent instance
        """
        for agent_id, agent in agents.items():
            if agent_id in self.agents:
                self.agents[agent_id] = agent
                # Update executor's agent reference
                if agent_id in self.plan_executors:
                    self.plan_executors[agent_id].agent = agent

        # Create message broker and assign to all agents
        from ..messages import MessageBroker
        self._message_broker = MessageBroker(agents)
        self._message_broker.wrapper = self
        for agent in agents.values():
            agent.message_broker = self._message_broker

    def set_agent(self, agent_id: str, agent: Agent):
        """Set a single Agent instance."""
        if agent_id in self.agents:
            self.agents[agent_id] = agent
            # Update executor's agent reference
            if agent_id in self.plan_executors:
                self.plan_executors[agent_id].agent = agent
            if self._message_broker is not None:
                self._message_broker.agents[agent_id] = agent
                agent.message_broker = self._message_broker

    def set_message_broker(self, message_broker):
        """
        Set the message broker for coordinated message handling.

        Args:
            message_broker: MessageBroker instance
        """
        self._message_broker = message_broker
        self._message_broker.wrapper = self
        # Also assign to all agents
        for agent in self.agents.values():
            if agent is not None:
                agent.message_broker = message_broker

    @property
    def message_broker(self):
        """Get the message broker."""
        return self._message_broker

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset the environment. Plans need to be set manually after reset."""
        obs_dict, info = self.symbolic_env.reset(seed=seed, options=options)

        self._current_obs = obs_dict
        self._current_info = info  # Store info for accessing symbolic_view
        self._current_step = 0
        self._prev_action_results = {aid: None for aid in self.agent_names}
        self.coop2_repair_controller.reset()
        self.coop2_process_logger.reset(info)

        # Get coop config from base env if available
        coop_config = None
        base_env = self.symbolic_env.env
        if hasattr(base_env, 'get_config_observation'):
            coop_config = base_env.get_config_observation()

        # Reset agents if they exist
        for agent_id, agent in self.agents.items():
            if agent is not None:
                agent.reset()
                # Set env info if agent supports it
                if hasattr(agent, 'set_env_info'):
                    agent_info = info.get(agent_id, {})
                    symbolic_view = agent_info.get('symbolic_view')
                    target_hints = agent_info.get('target_hints')
                    agent.set_env_info(
                        coop_config=coop_config,
                        symbolic_view=symbolic_view,
                        target_hints=target_hints,
                    )
                # Give initial observation to agents
                agent.observe(obs_dict[agent_id], self._current_step)

        # Reset plan executors to need new plans
        for agent_id in self.agent_names:
            self.plan_executors[agent_id].needs_new_plan = True

        return obs_dict, info

    def wait_for_state_change(self, timeout: float = 0.1) -> bool:
        """
        Wait for any agent state change (event-driven).

        Args:
            timeout: Max time to wait in seconds

        Returns:
            True if notified, False if timeout
        """
        self._sync_committed_plans()
        with self._state_change_event:
            changed = self._state_change_event.wait(timeout)
        self._sync_committed_plans()
        return changed

    def notify_state_change(self):
        """Notify that an agent state has changed."""
        self._sync_committed_plans()
        with self._state_change_event:
            self._state_change_event.notify_all()

    def _sync_committed_plans(self):
        """Record plans once agents commit them, before primitive execution begins."""
        for agent_id, agent in self.agents.items():
            if agent is None or not agent.ready or agent.plan is None:
                continue
            plan = agent.plan
            if plan.status.value != "pending":
                continue
            current = self.logger.current_plans.get(agent_id)
            if current is plan:
                continue
            if current is not None and current.status.value in {"pending", "executing"}:
                self.logger.log_plan_interrupted(current, self._current_step, "Replanned")
                self._reset_symbolic_action_state(agent_id)
            self.logger.log_plan_created(plan)
            self.coop2_trace.log(
                "plan_committed",
                env_step=self._current_step,
                agent_id=agent_id,
                plan_id=plan.plan_id,
                specification=plan.specification,
                remaining_actions=[
                    action.to_dict()
                    for action in plan.actions[plan.current_action_index:]
                ],
            )

    def _reset_symbolic_action_state(self, agent_id: str):
        """Drop the action a replaced plan left in flight.

        Clearing L2's record is not enough: the engine is still running the
        primitive, so ``has_active`` stays true and the *new* plan's first
        action cannot be issued until the abandoned one finishes -- hundreds of
        ticks for a NAVIGATE_TO. The old primitive's outcome then arrives with
        no symbolic action to attach to and is discarded. Abort it here, which
        is what COOP2's X -> I transition means, and the new plan starts on the
        next tick.

        This runs only when a plan is actually *replaced*. An agent that
        interrupts and resumes never reaches this path, so its primitive keeps
        running with the ticks it has already spent.
        """
        base_env = getattr(self.symbolic_env, "env", None)
        engine = getattr(base_env, "engine", None)
        if engine is not None and engine.has_active(agent_id):
            engine.abort(agent_id, retract=False)

        action_handler = self.symbolic_env.agent_actions.get(agent_id)
        if action_handler is not None and hasattr(action_handler, "reset_current_action"):
            action_handler.reset_current_action()
        self._prev_action_results[agent_id] = None

    def _idle_step_return(self, extra_info: Optional[Dict[str, Any]] = None):
        """Return the current observations without advancing the environment."""
        info = {}
        current_info = getattr(self, "_current_info", {}) or {}
        for agent_id in self.agent_names:
            agent_info = current_info.get(agent_id, {}) if isinstance(current_info, dict) else {}
            info[agent_id] = dict(agent_info) if isinstance(agent_info, dict) else {}
            if extra_info:
                info[agent_id].update(extra_info)
        return (
            self._current_obs,
            {agent_id: 0.0 for agent_id in self.agent_names},
            {agent_id: False for agent_id in self.agent_names},
            {agent_id: False for agent_id in self.agent_names},
            info,
        )

    def _agent_view_signature(self, actions: Dict[str, Any]) -> tuple:
        """What must change before the agent view is worth rebuilding.

        A new plan_id means the agent went back to reasoning and returned with
        a fresh plan; a changed status covers interruption and failure; a
        changed action index means the plan advanced. Everything else is a tick
        in which the plan is simply still running.
        """
        signature = []
        for agent_id in self.agent_names:
            agent = self.agents.get(agent_id)
            plan = getattr(agent, "plan", None) if agent is not None else None
            action = actions.get(agent_id) or {}
            signature.append((
                agent_id,
                getattr(plan, "plan_id", None),
                getattr(plan, "current_action_index", None),
                str(getattr(plan, "status", None)),
                getattr(getattr(agent, "state", None), "value", None),
                bool(getattr(agent, "ready", False)),
                action.get("action_type") if isinstance(action, dict) else None,
                action.get("target") if isinstance(action, dict) else None,
            ))
        return tuple(signature)

    def step(self):
        """
        Execute one environment step for all agents.

        Reads plans from agent.plan directly.

        Steps through each agent's plan one primitive action at a time.
        After execution, checks symbolic action status and updates plan status.

        Returns:
            tuple: (observations, rewards, terminated, truncated, info)
        """
        # Check if all agents are ready (in W state) and transition to X before execution
        self._sync_committed_plans()
        managed_agents = any(agent is not None for agent in self.agents.values())
        all_ready = all(
            self.agents[aid] is not None and self.agents[aid].ready
            for aid in self.agent_names
        )
        if managed_agents and not all_ready:
            waiting_for = [
                aid
                for aid in self.agent_names
                if self.agents[aid] is None or not self.agents[aid].ready
            ]
            return self._idle_step_return(
                {"waiting_for_agents": waiting_for}
            )

        if all_ready:
            evaluation = self.coop2_repair_controller.before_execution(
                agents=self.agents,
                env_step=self._current_step,
                current_info=getattr(self, "_current_info", None),
                repair_dispatcher=self.coop2_repair_dispatcher.dispatch,
            )
            if evaluation is not None and evaluation.should_repair:
                return self._idle_step_return(
                    {"coop2_pre_execution": evaluation.to_dict()}
                )

            # Use same timestamp for all agents transitioning together
            transition_time = time.time()
            for agent_id in self.agent_names:
                agent = self.agents[agent_id]
                if agent is not None and agent.state.value == "waiting":
                    agent.start_execution(timestamp=transition_time, env_step=self._current_step)

        # Get next action from each agent's current plan
        actions = {}
        for agent_id in self.agent_names:
            action = self.plan_executors[agent_id].step(
                observation=self._current_obs[agent_id],
                env_step=self._current_step,
                action_results=self._prev_action_results[agent_id]
            )
            actions[agent_id] = action
        # Rebuild only when an agent is back in reasoning, a plan was
        # interrupted, or the action index advanced. record_step mutates the
        # dicts it is handed (it writes action_outcome into them), so hand it
        # shallow copies rather than the cached originals.
        signature = self._agent_view_signature(actions)
        if signature != self._last_agent_view_signature:
            self._cached_agent_views = self.coop2_process_logger.build_agent_views(actions)
            self._last_agent_view_signature = signature
        process_agent_views = [dict(view) for view in self._cached_agent_views]

        # Execute all actions in the environment (one step)
        obs_dict, rewards, terminated, truncated, info = self.symbolic_env.step(actions)

        # Update current state
        self._current_obs = obs_dict
        self._current_step += 1

        # NOW check symbolic action status and update plan status
        for agent_id in self.agent_names:
            # Get the agent FIRST before accessing its plan
            agent = self.agents[agent_id]
            action_handler = self.symbolic_env.agent_actions[agent_id]
            action_records = action_handler.get_action_records()

            # Get the latest symbolic action status
            if action_records:
                latest_record = action_records[-1]
                action_status = {
                    "status": latest_record["status"],
                    "failure_reason": latest_record.get("failure_reason")
                }
                # Update the plan's SymbolicAction with the current primitive_action for visualization
                if agent is not None and agent.plan:
                    current_action = agent.plan.get_current_action()
                    if current_action and latest_record.get("primitive_action"):
                        current_action.primitive_action = latest_record["primitive_action"]
                    # Also sync primitive_action_history for multi-step action visualization
                    if current_action and latest_record.get("primitive_action_history"):
                        current_action.primitive_action_history.update(latest_record["primitive_action_history"])
                    if current_action and latest_record.get("outcome") is not None:
                        current_action.outcome = latest_record["outcome"]
                        self.coop2_trace.log(
                            "action_outcome",
                            env_step=self._current_step,
                            agent_id=agent_id,
                            action=current_action.to_dict(),
                            outcome=latest_record["outcome"],
                        )
            else:
                action_status = None

            # Store for next iteration
            self._prev_action_results[agent_id] = action_status

            # Check plan status and handle completion/failure
            if agent is not None and agent.plan and action_status:
                plan = agent.plan
                executor = self.plan_executors[agent_id]  # Reference for setting needs_new_plan
                current_action = plan.get_current_action()

                # If symbolic action succeeded, advance and check completion
                if action_status["status"] == "success" and current_action is not None and current_action.start_step is not None:
                    # Log action completion
                    self.logger.log_action_completed(plan, current_action, self._current_step, True, None)
                    # Advance to next action
                    plan.advance_action()

                    # Check if plan is complete (was last action)
                    if plan.is_complete():
                        if plan.status.value == "executing":
                            goal_failure_reason = self._plan_goal_failure_reason(plan)
                            if goal_failure_reason:
                                self.logger.log_plan_completed(
                                    plan,
                                    self._current_step,
                                    False,
                                    goal_failure_reason,
                                )
                                if hasattr(agent, "memory"):
                                    agent.memory.record_plan_failure(
                                        plan_id=plan.plan_id,
                                        specification=plan.specification,
                                        failed_action="plan_goal",
                                        failure_reason=goal_failure_reason,
                                        env_step=self._current_step,
                                    )
                            else:
                                plan.complete_success(self._current_step)
                                self.logger.log_plan_completed(plan, self._current_step, True)
                            executor.needs_new_plan = True
                            # Transition agent to R (reasoning) state when plan completes
                            if agent_id in self.agents and self.agents[agent_id] is not None:
                                self.agents[agent_id].set_unready(
                                    reason='plan_terminated',
                                    env_step=self._current_step
                                )
                    # The next action will be marked started when it is actually selected
                    # for primitive execution on a later step.

                # If symbolic action failed, plan fails (e.g., navigate returned no_path)
                elif action_status["status"] == "failed":
                    if plan.status.value == "executing" and current_action is not None:
                        failure_reason = action_status.get("failure_reason", "Action failed")
                        print(f"  [{agent_id}] Action failed: {failure_reason} - terminating plan")
                        self.logger.log_action_completed(plan, current_action, self._current_step, False, failure_reason)
                        plan.complete_failed(self._current_step, failure_reason)
                        self.logger.log_plan_completed(plan, self._current_step, False, failure_reason)
                        if hasattr(agent, "memory"):
                            agent.memory.record_plan_failure(
                                plan_id=plan.plan_id,
                                specification=plan.specification,
                                failed_action=str(current_action),
                                failure_reason=failure_reason,
                                env_step=self._current_step,
                            )
                        executor.needs_new_plan = True
                        # Transition agent to R (reasoning) state when plan fails
                        if agent_id in self.agents and self.agents[agent_id] is not None:
                            self.agents[agent_id].set_unready(
                                reason='plan_terminated',
                                env_step=self._current_step
                            )

        # Sync agent observations and env info
        # Get coop config from base env if available
        coop_config = None
        base_env = self.symbolic_env.env
        if hasattr(base_env, 'get_config_observation'):
            coop_config = base_env.get_config_observation()

        for agent_id, agent in self.agents.items():
            if agent is not None:
                # Set env info if agent supports it
                if hasattr(agent, 'set_env_info'):
                    agent_info = info.get(agent_id, {})
                    symbolic_view = agent_info.get('symbolic_view')
                    target_hints = agent_info.get('target_hints')
                    agent.set_env_info(
                        coop_config=coop_config,
                        symbolic_view=symbolic_view,
                        target_hints=target_hints,
                    )
                agent.observe(obs_dict[agent_id], self._current_step)

        self._current_info = info  # Store info for later access
        self.coop2_process_logger.record_step(
            env_step=self._current_step,
            info=info,
            agent_views=process_agent_views,
        )

        return obs_dict, rewards, terminated, truncated, info

    def _plan_goal_failure_reason(self, plan: SymbolicPlan) -> Optional[str]:
        """Return a failure reason when actions completed but the named task did not."""
        task_name = self._task_name_from_spec(plan.specification)
        if task_name is None:
            return None

        if task_name.startswith("collect_"):
            expected = task_name.removeprefix("collect_")
            if self._plan_collected_item(plan, expected):
                return None
            return f"Plan task not achieved: expected collect_{expected}"

        if task_name.startswith("make_"):
            expected = task_name.removeprefix("make_")
            if self._plan_crafted_item(plan, expected):
                return None
            return f"Plan task not achieved: expected make_{expected}"

        if task_name.startswith("place_"):
            expected = task_name.removeprefix("place_")
            if self._plan_placed_item(plan, expected):
                return None
            return f"Plan task not achieved: expected place_{expected}"

        return None

    def _task_name_from_spec(self, specification: str) -> Optional[str]:
        match = re.match(r"\s*([a-zA-Z0-9_]+)\s*(?:\(|$)", str(specification or ""))
        if not match:
            return None
        return match.group(1).lower()

    def _plan_collected_item(self, plan: SymbolicPlan, expected: str) -> bool:
        expected = expected.lower()
        return any(
            expected in self._collected_items_from_outcome(action.outcome or {})
            for action in plan.actions
        )

    def _plan_crafted_item(self, plan: SymbolicPlan, expected: str) -> bool:
        expected = expected.lower()
        achievement_name = f"make_{expected}"
        for action in plan.actions:
            outcome = action.outcome or {}
            effects = outcome.get("effects") or {}
            inventory_delta = effects.get("inventory_delta") or {}
            achievement_delta = effects.get("achievement_delta") or {}
            if self._positive_delta(inventory_delta.get(expected)):
                return True
            if self._positive_delta(achievement_delta.get(achievement_name)):
                return True
        return False

    def _plan_placed_item(self, plan: SymbolicPlan, expected: str) -> bool:
        expected = expected.lower()
        achievement_name = f"place_{expected}"
        for action in plan.actions:
            outcome = action.outcome or {}
            effects = outcome.get("effects") or {}
            achievement_delta = effects.get("achievement_delta") or {}
            if self._positive_delta(achievement_delta.get(achievement_name)):
                return True
            for place_result in effects.get("place_results") or []:
                if (
                    place_result.get("success")
                    and str(place_result.get("object_type", "")).lower() == expected
                ):
                    return True
        return False

    def _collected_items_from_outcome(self, outcome: Dict[str, Any]) -> set[str]:
        effects = outcome.get("effects") or {}
        collected: set[str] = set()

        for item_name, delta in (effects.get("inventory_delta") or {}).items():
            if self._positive_delta(delta):
                collected.add(str(item_name).lower())

        for achievement_name, delta in (effects.get("achievement_delta") or {}).items():
            normalized = str(achievement_name).lower()
            if self._positive_delta(delta) and normalized.startswith("collect_"):
                collected.add(normalized.removeprefix("collect_"))
            elif self._positive_delta(delta) and normalized == "eat_cow":
                collected.add("food")

        for collection in effects.get("collections") or []:
            resource_type = str(collection.get("resource_type", "")).lower()
            if resource_type:
                collected.add(resource_type)
                if resource_type == "tree":
                    collected.add("wood")
                elif resource_type == "cow":
                    collected.add("food")
            for item_name, amount in (collection.get("received") or {}).items():
                if self._positive_delta(amount):
                    collected.add(str(item_name).lower())

        return collected

    def _positive_delta(self, value: Any) -> bool:
        try:
            return float(value or 0) > 0
        except (TypeError, ValueError):
            return bool(value)

    def get_agent_status(self, agent_id: str) -> Dict:
        """Get current status of an agent's plan execution."""
        if agent_id in self.plan_executors:
            return self.plan_executors[agent_id].get_status()
        return {"status": "unknown"}

    def get_all_agent_status(self) -> Dict[str, Dict]:
        """Get status for all agents."""
        return {
            agent_id: self.plan_executors[agent_id].get_status()
            for agent_id in self.agent_names
        }

    def get_agents_needing_plans(self) -> List[str]:
        """Get list of agents that need new plans."""
        return [
            agent_id for agent_id in self.agent_names
            if self.plan_executors[agent_id].needs_new_plan
        ]

    def agents_need_plans(self) -> Dict[str, bool]:
        """Get dict mapping agent_id to whether they need a new plan."""
        return {
            agent_id: self.plan_executors[agent_id].needs_new_plan
            for agent_id in self.agent_names
        }

    def set_agent_plan(self, agent_id: str, plan: SymbolicPlan):
        """
        Manually set a plan for an agent (external plan generation).

        Args:
            agent_id: Agent to set plan for
            plan: Plan to execute
        """
        if agent_id in self.plan_executors:
            # Get plan_id from agent's plan_count if agent exists
            plan_id = None
            if agent_id in self.agents and self.agents[agent_id] is not None:
                plan_id = self.agents[agent_id].plan_count
            self.plan_executors[agent_id].set_plan(plan, self._current_step, plan_id=plan_id)

    def terminate_unfinished_plans(self):
        """Terminate all unfinished plans (call at episode end)."""
        self.logger.terminate_unfinished_plans(self._current_step)

    def save_plan_logs(self, filename: str):
        """Save plan execution logs."""
        self.logger.save_logs(filename)

    def save_agent_log(self, output_path: str):
        """Save agent state timelines to JSON."""
        self.log_saver.save_agent_log(output_path)

    def save_message_log(self, output_path: str):
        """Save inter-agent messages to JSON."""
        self.log_saver.save_message_log(output_path)

    def save_task_log(self, output_path: str):
        """Save task states history to JSON (only changed tasks to reduce file size)."""
        self.log_saver.save_task_log(output_path)

    def save_capability_log(self, output_path: str):
        """Save capability change history to JSON (if CooperativeEnv is used)."""
        self.log_saver.save_capability_log(output_path)

    def save_team_score_log(self, output_path: str):
        """Save team resource score summary if the base environment provides one."""
        self.log_saver.save_team_score_log(output_path)

    def save_coop2_process_log(self, output_path: str):
        """Save the compact per-step COOP2 process trace."""
        self.coop2_process_logger.save(output_path)

    def save_logs(self, output_dir: str):
        """
        Save all experiment logs and data.

        Saves:
        - plan_logs.json: Plan execution history
        - agent_states.json: Agent state timelines (R/W/X/I transitions)
        - message_log.json: Inter-agent messages
        - task_states.json: Task states history (if CooperativeEnv is used)
        - capability_changes.json: Agent capability changes (if CooperativeEnv is used)
        - coop2_trace.json: Environment-agnostic COOP2 events

        Args:
            output_dir: Directory to save all logs
        """
        self.log_saver.save_all(output_dir)

    def get_plan_statistics(self) -> Dict:
        """Get statistics about plan execution."""
        return self.logger.get_statistics()

    @property
    def current_step(self) -> int:
        """Get current environment step."""
        return self._current_step

    # Forward attribute access to symbolic env
    def __getattr__(self, name):
        """Forward attribute access to the wrapped symbolic environment."""
        return getattr(self.symbolic_env, name)
