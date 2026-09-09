"""
Plan interface for MA-Crafter agents.

A Plan consists of:
- Specification: Goal/description of what the plan aims to achieve
- Actions: Sequence of symbolic actions to execute
- Status tracking: Success/failure of plan and individual actions

Also includes the SymbolicPlanExecutor for managing plan execution.
"""

from typing import List, Dict, Optional, Any, Literal, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from ..action.action import SymbolicAction


class SymbolicPlanStatus(Enum):
    """Status of plan execution."""
    PENDING = "pending"      # Plan not yet started
    EXECUTING = "executing"  # Plan currently executing
    SUCCESS = "success"      # Plan completed successfully
    FAILED = "failed"        # Plan failed during execution
    INTERRUPTED = "interrupted"  # Plan terminated by episode end


@dataclass
class SymbolicPlan:
    """
    A plan consisting of a specification and sequence of symbolic actions.
    
    The Plan class is responsible for tracking its own execution state,
    including which action is currently executing and the overall plan status.
    """
    specification: str  # Goal/description of the plan
    actions: List[SymbolicAction]
    
    # Metadata
    plan_id: int
    agent_id: str
    created_at_step: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Execution tracking (managed internally by the Plan)
    status: SymbolicPlanStatus = SymbolicPlanStatus.PENDING
    current_action_index: int = 0
    start_step: Optional[int] = None
    end_step: Optional[int] = None
    failure_reason: Optional[str] = None
    
    def get_current_action(self) -> Optional[SymbolicAction]:
        """Get the currently executing action."""
        if 0 <= self.current_action_index < len(self.actions):
            return self.actions[self.current_action_index]
        return None
    
    def advance_action(self):
        """Move to the next action in the plan."""
        self.current_action_index += 1
        
        # Auto-check if plan is complete after advancing
        if self.is_complete() and self.status == SymbolicPlanStatus.EXECUTING:
            # Plan completed all actions but not yet marked as success
            # This will be finalized by mark_success() call
            pass
    
    def is_complete(self) -> bool:
        """Check if all actions have been executed."""
        return self.current_action_index >= len(self.actions)
    
    def start(self, step: int):
        """Start plan execution."""
        if self.status == SymbolicPlanStatus.PENDING:
            self.status = SymbolicPlanStatus.EXECUTING
            self.start_step = step
            return True
        return False
    
    def mark_started(self, step: int):
        """Mark plan as started (alias for backward compatibility)."""
        return self.start(step)
    
    def complete_success(self, step: int):
        """Complete plan successfully."""
        if self.status == SymbolicPlanStatus.EXECUTING:
            self.status = SymbolicPlanStatus.SUCCESS
            self.end_step = step
            return True
        return False
    
    def mark_success(self, step: int):
        """Mark plan as successfully completed (alias for backward compatibility)."""
        return self.complete_success(step)
    
    def complete_failed(self, step: int, reason: str):
        """Complete plan with failure."""
        if self.status in [SymbolicPlanStatus.EXECUTING, SymbolicPlanStatus.PENDING]:
            self.status = SymbolicPlanStatus.FAILED
            self.end_step = step
            self.failure_reason = reason
            return True
        return False
    
    def mark_failed(self, step: int, reason: str):
        """Mark plan as failed (alias for backward compatibility)."""
        return self.complete_failed(step, reason)
    
    def complete_interrupted(self, step: int):
        """Mark plan as interrupted (e.g., by episode end).

        The action in flight is marked too. Leaving it at "executing" forever
        made the log unreadable at exactly the moment that matters most: in the
        run where the BDDL goal fired, the placement that *satisfied* it was
        still inside its settle when the episode stopped, so the trace showed
        ``place_on_top[executing]`` and gave no way to tell a cut-short action
        from a stuck one.
        """
        if self.status in [SymbolicPlanStatus.EXECUTING, SymbolicPlanStatus.PENDING]:
            self.status = SymbolicPlanStatus.INTERRUPTED
            self.end_step = step
            self.failure_reason = "Episode ended"

            current = self.get_current_action()
            if current is not None and current.status in (None, "executing", "pending"):
                current.status = "interrupted"
                current.end_step = step
                current.failure_reason = "Episode ended"
            return True
        return False
    
    def start_action(self, action_index: int, step: int) -> bool:
        """
        Start execution of a specific action.
        
        Returns:
            True if action was started, False if invalid index
        """
        if 0 <= action_index < len(self.actions):
            action = self.actions[action_index]
            if action.start_step is None:
                action.start_step = step
                action.status = "executing"
                return True
        return False
    
    def complete_action(
        self,
        action_index: int,
        step: int,
        success: bool,
        failure_reason: Optional[str] = None,
        outcome: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Complete execution of a specific action.
        
        Returns:
            True if action was completed, False if invalid index
        """
        if 0 <= action_index < len(self.actions):
            action = self.actions[action_index]
            action.end_step = step
            action.status = "success" if success else "failed"
            if failure_reason:
                action.failure_reason = failure_reason
            if outcome is not None:
                action.outcome = outcome
            return True
        return False
    
    def get_progress(self) -> str:
        """Get human-readable progress string."""
        return f"{self.current_action_index}/{len(self.actions)}"
    
    def get_duration(self) -> Optional[int]:
        """Get plan execution duration in steps."""
        if self.start_step is not None and self.end_step is not None:
            return self.end_step - self.start_step
        return None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            "plan_id": self.plan_id,
            "agent_id": self.agent_id,
            "specification": self.specification,
            "created_at_step": self.created_at_step,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "status": self.status.value,
            "failure_reason": self.failure_reason,
            "metadata": self.metadata,
            "duration": self.get_duration(),
            "actions": [action.to_dict() for action in self.actions],
            "progress": self.get_progress()
        }


class SymbolicPlanLogger:
    """Logger for tracking plan execution."""
    
    def __init__(self):
        self.plan_history: List[Dict] = []
        self.current_plans: Dict[str, SymbolicPlan] = {}
    
    def log_plan_created(self, plan: SymbolicPlan):
        """Log when a new plan is created."""
        self.current_plans[plan.agent_id] = plan
        print(f"\n[{plan.agent_id}] New Plan #{plan.plan_id} created at step {plan.created_at_step}")
        print(f"  Specification: {plan.specification}")
        print(f"  Actions: {len(plan.actions)}")
        for i, action in enumerate(plan.actions):
            print(f"    {i+1}. {action.action_type}({action.args})")
    
    def log_plan_started(self, plan: SymbolicPlan, step: int):
        """Log when plan execution starts."""
        plan.start(step)
        print(f"\n[{plan.agent_id}] Plan #{plan.plan_id} started at step {step}")
    
    def log_action_started(self, plan: SymbolicPlan, action: SymbolicAction, step: int):
        """Log when an action starts."""
        plan.start_action(plan.current_action_index, step)
        print(f"  [{plan.agent_id}] Action {plan.current_action_index + 1}/{len(plan.actions)}: "
              f"{action.action_type}({action.args}) - step {step}")
    
    def log_action_completed(self, plan: SymbolicPlan, action: SymbolicAction, step: int, 
                           success: bool, failure_reason: Optional[str] = None):
        """Log when an action completes."""
        # Find the index of this action in the plan by matching the action object
        action_index = -1
        for i, plan_action in enumerate(plan.actions):
            if plan_action is action:
                action_index = i
                break
        
        if action_index == -1:
            # Fallback: couldn't find the action, use current_action_index logic
            action_index = plan.current_action_index if plan.current_action_index < len(plan.actions) else len(plan.actions) - 1
        
        # Use the plan's method to complete the action
        plan.complete_action(action_index, step, success, failure_reason, getattr(action, "outcome", None))
        
        status_symbol = "[OK]" if success else "[FAIL]"
        # Calculate duration from when this action actually started to now
        if action.start_step is not None:
            duration = step - action.start_step
        else:
            duration = 0
        print(f"  [{plan.agent_id}] Action {status_symbol} {action.action_type} - "
              f"step {step} (duration: {duration})")
        if failure_reason:
            print(f"    Reason: {failure_reason}")
    
    def log_plan_completed(self, plan: SymbolicPlan, step: int, success: bool, 
                          failure_reason: Optional[str] = None):
        """Log when plan completes."""
        if success:
            plan.complete_success(step)
            duration = plan.get_duration()
            print(f"\n[{plan.agent_id}] [OK] Plan #{plan.plan_id} SUCCEEDED at step {step} "
                  f"(duration: {duration})")
        else:
            plan.complete_failed(step, failure_reason or "Unknown")
            print(f"\n[{plan.agent_id}] [FAIL] Plan #{plan.plan_id} FAILED at step {step}")
            print(f"  Reason: {failure_reason}")
        
        # Archive completed plan
        self.plan_history.append(plan.to_dict())
        if plan.agent_id in self.current_plans:
            del self.current_plans[plan.agent_id]
    
    def get_agent_plans(self, agent_id: str) -> List[Dict]:
        """Get all plans for an agent."""
        return [p for p in self.plan_history if p["agent_id"] == agent_id]
    
    def log_plan_interrupted(self, plan: SymbolicPlan, step: int, reason: str = "Replanned"):
        """Log when a plan is interrupted (e.g., agent decided to replan)."""
        plan.complete_interrupted(step)
        plan.failure_reason = reason
        self.plan_history.append(plan.to_dict())
        print(f"\n[{plan.agent_id}] [INT] Plan #{plan.plan_id} INTERRUPTED at step {step} ({reason})")
        if plan.agent_id in self.current_plans:
            del self.current_plans[plan.agent_id]
    
    def get_all_plans(self) -> List[Dict]:
        """Get all plans including currently executing ones."""
        all_plans = self.plan_history.copy()
        # Add currently executing plans
        for agent_id, plan in self.current_plans.items():
            all_plans.append(plan.to_dict())
        return all_plans
    
    def save_logs(self, filename: str):
        """Save plan logs to file."""
        data = {
            "plan_history": self.plan_history,
            "current_plans": {aid: p.to_dict() for aid, p in self.current_plans.items()}
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nPlan logs saved to {filename}")
    
    def terminate_unfinished_plans(self, step: int):
        """Terminate all unfinished plans (e.g., at episode end)."""
        for agent_id, plan in list(self.current_plans.items()):
            plan.complete_interrupted(step)
            self.plan_history.append(plan.to_dict())
            print(f"\n[{plan.agent_id}] [INT] Plan #{plan.plan_id} INTERRUPTED at step {step} (episode ended)")
        self.current_plans.clear()
    
    def get_statistics(self) -> Dict:
        """Get statistics about plan execution."""
        total_plans = len(self.plan_history)
        if total_plans == 0:
            return {"total_plans": 0}
        
        successful = sum(1 for p in self.plan_history if p["status"] == "success")
        failed = sum(1 for p in self.plan_history if p["status"] == "failed")
        interrupted = sum(1 for p in self.plan_history if p["status"] == "interrupted")
        
        total_duration = sum(p["duration"] for p in self.plan_history if p["duration"] is not None)
        avg_duration = total_duration / total_plans if total_plans > 0 else 0
        
        return {
            "total_plans": total_plans,
            "successful": successful,
            "failed": failed,
            "interrupted": interrupted,
            "success_rate": successful / total_plans if total_plans > 0 else 0,
            "average_duration": avg_duration
        }


class SymbolicPlanExecutor:
    """
    Executor interface for managing plan execution in MA-Crafter.
    
    The executor manages plan lifecycle, executes action sequences,
    and handles plan completion or failure.
    """
    
    def __init__(self, agent_id: str, agent: Any = None, plan_generator: Callable = None, logger: Optional[SymbolicPlanLogger] = None):
        """
        Initialize planning agent.
        
        Args:
            agent_id: Unique identifier for the agent
            agent: Optional reference to Agent instance (to access agent.plan)
            plan_generator: Optional function that generates plans (for backward compatibility)
                Signature: plan_generator(agent_id, observation, env_step) -> Plan
            logger: Optional plan logger
        """
        self.agent_id = agent_id
        self.agent = agent  # Reference to agent for accessing agent.plan
        self.plan_generator = plan_generator
        self.logger = logger or SymbolicPlanLogger()
        
        self.needs_new_plan = True  # Start needing a plan
    
    def set_plan(self, plan: SymbolicPlan, env_step: int, plan_id: int = None):
        """
        Set a new plan to execute (externally generated).
        
        Args:
            plan: The plan to execute
            env_step: Current environment step
            plan_id: Optional plan ID (from agent's plan_count). If None, plan.plan_id must be set.
        """
        if plan_id is not None:
            plan.plan_id = plan_id
        # plan_id should already be set by agent or passed as parameter
        plan.agent_id = self.agent_id
        plan.created_at_step = env_step
        
        # Set plan in agent if available
        if self.agent is not None:
            self.agent.plan = plan
        
        self.needs_new_plan = False
        
        self.logger.log_plan_created(plan)
        self.logger.log_plan_started(plan, env_step)
        return plan
    
    def generate_new_plan(self, observation: Any, env_step: int) -> SymbolicPlan:
        """
        Generate and set a new plan (uses plan_generator).
        For backward compatibility with direct usage.
        """
        if not self.plan_generator:
            raise ValueError("No plan_generator provided")
            
        plan = self.plan_generator(self.agent_id, observation, env_step)
        return self.set_plan(plan, env_step)
    
    def step(self, observation: Any, env_step: int, action_results: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Execute one step of planning/action.
        
        Args:
            observation: Current observation
            env_step: Current environment step
            action_results: Results from last action execution (status, failure_reason)
                           NOTE: This is processed by the planning wrapper now
        
        Returns:
            Symbolic action dict to execute, or noop if waiting for plan
        """
        # Get plan from agent if available
        plan = self.agent.plan if self.agent is not None else None
        
        # If no plan available, wait for one to be set externally
        if plan is None:
            self.needs_new_plan = True
            return {"action_type": "noop"}
        
        # If we have a new plan (PENDING status), start it and log
        if plan.status == SymbolicPlanStatus.PENDING:
            # Check if there's an old plan being replaced (interrupted by replan)
            old_plan = self.logger.current_plans.get(self.agent_id)
            if old_plan is not None and old_plan is not plan:
                if old_plan.status in [SymbolicPlanStatus.PENDING, SymbolicPlanStatus.EXECUTING]:
                    self.logger.log_plan_interrupted(old_plan, env_step, "Replanned")
            
            self.needs_new_plan = False
            if self.logger.current_plans.get(self.agent_id) is not plan:
                self.logger.log_plan_created(plan)
            self.logger.log_plan_started(plan, env_step)
        
        # Get current action to execute
        current_action = plan.get_current_action()
        
        # If current_action is None, plan is complete - return noop and wait for new plan
        if current_action is None:
            return {"action_type": "noop"}
        
        # If this is a new action, log it as starting
        if current_action.start_step is None:
            self.logger.log_action_started(plan, current_action, env_step)
        
        # Execute the current action
        return {
            "action_type": current_action.action_type,
            **current_action.args
        }
    
    def get_status(self) -> Dict:
        """Get current status of the agent."""
        plan = self.agent.plan if self.agent is not None else None
        
        if plan is None:
            return {"status": "no_plan"}
        
        return {
            "status": plan.status.value,
            "plan_id": plan.plan_id,
            "specification": plan.specification,
            "progress": f"{plan.current_action_index}/{len(plan.actions)}",
            "current_action": plan.get_current_action().action_type if plan.get_current_action() else None
        }
