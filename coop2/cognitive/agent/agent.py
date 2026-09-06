"""
Base agent class for planning agents in MA-Crafter.

Defines the interface for agents that generate and execute plans.
This will be used as the foundation for MAEIL and other planning approaches.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict, Union
from enum import Enum
import time
from ..plan.plan import SymbolicPlan
from ..coop2_messages import has_coop2_repair_message
from .memory import AgentMemory


class AgentState(Enum):
    """Agent execution state in the planning cycle."""
    R = "reasoning"      # Agent is reasoning/generating a plan (not ready)
    W = "waiting"        # Agent is ready and waiting for other agents
    X = "executing"      # Agent is executing its plan
    I = "interrupted"    # Agent interrupted by message, deciding to resume or replan


class Agent(ABC):
    """
    Abstract base class for planning agents.
    
    Agents observe the environment and generate plans to achieve goals.
    The agent is responsible for maintaining its own internal state and
    generating plans based on observations.
    """
    
    def __init__(self, agent_id: str, plan_generation_time: float = 0.0, memory_size: int = 10):
        """
        Initialize the agent.
        
        Args:
            agent_id: Unique identifier for this agent
            plan_generation_time: Time in seconds to simulate plan generation (default: 0.0)
            memory_size: Number of recent messages and plans to keep in memory (default: 10)
        """
        self.agent_id = agent_id
        self.observation = None
        self.env_step = 0
        self.plan: Optional[SymbolicPlan] = None
        self.plan_generation_time = plan_generation_time
        self.plan_count = 0  # Track number of plans generated
        self.ready = False  # Agent is ready when it has a valid plan
        self._state = AgentState.R  # Start in reasoning state
        
        # Track state transitions: [(wall_clock_time, env_step, state), ...]
        self.state_history = [(time.time(), 0, AgentState.R)]
        self._start_time = time.time()  # Reference time for relative plotting
        
        # Message system
        self.message_broker = None  # Set by environment
        self.message_buffer = []  # Current unread messages
        self.message_history = []  # All messages ever received
        
        # Buffer indicator: tracks senders of messages in buffer
        # Key = sender_id, Value = True if message from this sender is in buffer
        self.buffer_senders = {}  # {sender_id: True, ...}
        
        # Track the committed plan (plan when agent became ready)
        # Used to detect if handle_interrupt generated a new plan
        self._committed_plan = None
        
        # Track when each plan became active: [(wall_clock_time, env_step, plan_id), ...]
        # Used by visualization to map time ranges to plans
        self.plan_timeline = []
        
        # Shutdown flag for terminating wait loops
        self._shutdown = False
        
        # Memory: stores recent messages and plans in chronological order
        self.memory = AgentMemory(max_size=memory_size)
    
    def shutdown(self):
        """Signal agent to stop waiting and terminate."""
        self._shutdown = True
    
    def wait_for_messages_from(self, senders: List[str]) -> bool:
        """
        Check if buffer contains messages from ALL specified senders.
        
        This method is useful for topologies where an agent needs to wait
        for messages from specific senders before proceeding.
        
        Args:
            senders: List of sender IDs to wait for
        
        Returns:
            True if messages from ALL senders are in buffer, False otherwise
        """
        if not senders:
            return True
        return all(self.buffer_senders.get(sender, False) for sender in senders)
    
    @property
    def state(self) -> AgentState:
        """Get current agent state."""
        return self._state
    
    def _set_state(self, new_state: AgentState, timestamp: float = None, env_step: int = None):
        """
        Internal method to set agent state and record transition.
        
        Use public transition methods instead: set_ready(), set_unready(), etc.
        
        Args:
            new_state: The new state to transition to
            timestamp: Wall clock time (from time.time()). If None, uses current time.
            env_step: Environment step number. If None, uses self.env_step.
        """
        if new_state != self._state:
            self._state = new_state
            if timestamp is None:
                timestamp = time.time()
            if env_step is None:
                env_step = self.env_step
            self.state_history.append((timestamp, env_step, new_state))
    
    # ========================================================================
    # Finite State Machine: Explicit State Transition Methods
    # ========================================================================
    # Valid transitions:
    # R -> W: set_ready() - plan generation complete, agent becomes ready
    # W -> X: start_execution() - all agents ready, begin execution
    # W -> I: set_unready(reason='message_received') - message during wait
    # X -> I: set_unready(reason='message_received') - message during execution
    # X -> R: set_unready(reason='plan_terminated') - plan finished, need new plan
    # I -> W: resume_execution() OR generate new plan then set_ready() - agent becomes ready
    # ========================================================================
    
    def set_ready(self, timestamp: float = None, env_step: int = None):
        """
        Mark agent as ready and transition to W (Waiting) state.
        
        Call this after generate_plan() when agent has finished reasoning.
        Increments plan counter only if the plan changed from the committed plan.
        Transition: R -> W or I -> W
        
        Args:
            timestamp: Wall clock time for state tracking
            env_step: Environment step number for state tracking
        """
        if self._state not in [AgentState.R, AgentState.I]:
            print(f"WARNING [{self.agent_id}]: set_ready() called from invalid state {self._state.value}")
        
        if timestamp is None:
            timestamp = time.time()
        if env_step is None:
            env_step = self.env_step
        
        # Only increment plan counter if plan actually changed (new plan object created)
        plan_changed = self.plan is not self._committed_plan
        if plan_changed:
            self.plan_count += 1
            # Update plan's ID to match the finalized counter
            if self.plan is not None:
                self.plan.plan_id = self.plan_count
                # Record new plan to memory
                self.memory.record_plan(
                    plan_id=self.plan.plan_id,
                    specification=self.plan.specification,
                    actions=[str(a) for a in self.plan.actions],
                    timestamp=timestamp,
                    env_step=env_step
                )
            print(f"  [{self.agent_id}] Plan #{self.plan_count} ready: {self.plan.specification if self.plan else 'unknown'}")
        else:
            print(f"  [{self.agent_id}] Resuming Plan #{self.plan_count}")
        
        # Record when this plan became active (for visualization)
        self.plan_timeline.append((timestamp, env_step, self.plan_count))
        
        self.ready = True
        self._set_state(AgentState.W, timestamp, env_step)
        
        # Record committed plan at the end
        self._committed_plan = self.plan
    
    def set_unready(self, reason: str, timestamp: float = None, env_step: int = None):
        """
        Mark agent as not ready and transition state based on reason.
        
        Args:
            reason: Why agent became unready:
                - 'plan_terminated': Plan finished (success/failure/interrupted) -> R state
                - 'message_received': Message arrived during W/X -> I state
            timestamp: Wall clock time for state tracking
            env_step: Environment step number for state tracking
        """
        self.ready = False
        
        if reason == 'plan_terminated':
            # Plan finished, need new plan: X -> R
            if self._state != AgentState.X:
                print(f"WARNING [{self.agent_id}]: plan_terminated from state {self._state.value}, expected X")
            self._set_state(AgentState.R, timestamp, env_step)
        
        elif reason == 'message_received':
            # Message arrived during W or X: -> I
            if self._state not in [AgentState.W, AgentState.X]:
                print(f"WARNING [{self.agent_id}]: message_received from invalid state {self._state.value}")
            self._set_state(AgentState.I, timestamp, env_step)
        
        else:
            raise ValueError(f"Unknown reason for set_unready: {reason}")
    
    def resume_execution(self, timestamp: float = None, env_step: int = None):
        """
        Resume plan execution after interrupt (without replanning).
        
        Transition: I -> W (agent becomes ready with existing plan)
        Does NOT increment plan counter.
        
        Args:
            timestamp: Wall clock time
            env_step: Environment step number
        """
        if self._state != AgentState.I:
            print(f"WARNING [{self.agent_id}]: resume_execution from state {self._state.value}, expected I")
        
        if self.plan is None:
            print(f"WARNING [{self.agent_id}]: resume_execution called but no plan exists")
        
        self.ready = True
        self._set_state(AgentState.W, timestamp, env_step)
    
    @abstractmethod
    def observe(self, observation: Any, env_step: int):
        """
        Process observation from the environment.
        
        This method is called to provide the agent with the current
        observation and environment step. The agent should update its
        internal state based on this information.
        
        Args:
            observation: Current observation from the environment
            env_step: Current environment step number
        """
        pass
    
    @abstractmethod
    def generate_plan(self) -> SymbolicPlan:
        """
        Generate a new plan based on current state.
        
        This method should create and return a SymbolicPlan that the
        agent will execute. The plan should be based on the agent's
        current observation and internal state.
        
        Implementations should set self.ready appropriately during
        plan generation (False while generating, True when complete).
        State should transition from R → W when plan is ready.
        
        Returns:
            SymbolicPlan: A new plan to execute
        """
        pass
    
    def create_agent_thread(self):
        """
        Agent thread entry point - make the agent ready.
        
        This method handles all state transitions to become ready:
        - W (waiting): Already ready
        - I (interrupted): Call handle_interrupt() to decide resume/replan
        - R (reasoning): Call handle_reasoning() for communication + plan generation
        
        After this method completes, agent should be in W state (ready)
        only if a plan was successfully generated. Otherwise stays in R.
        """
        # Already waiting/ready, nothing to do
        if self._state == AgentState.W:
            return
        
        # Handle state transitions
        if self._state == AgentState.I:
            self.handle_interrupt()
        elif self._state == AgentState.R and self.needs_new_plan():
            self.handle_reasoning()
        else:
            return
        
        # Transition to W - set_ready will check plan vs _committed_plan
        self.set_ready()
    
    def needs_new_plan(self) -> bool:
        """
        Check if the agent needs a new plan.
        
        Sets ready=False and state=R when a new plan is needed.
        
        Returns True if:
        - Agent has no plan
        - Current plan is finished (success or failure)
        
        Returns:
            bool: True if agent needs a new plan
        """
        needs_plan = False
        
        if self.plan is None:
            needs_plan = True
        else:
            # Check if plan is in a terminal state
            from ..plan.plan import SymbolicPlanStatus
            needs_plan = self.plan.status in [
                SymbolicPlanStatus.SUCCESS,
                SymbolicPlanStatus.FAILED,
                SymbolicPlanStatus.INTERRUPTED
            ]
        
        # Mark as not ready and enter reasoning state when new plan is needed
        # Only transition if not already in reasoning state (wrapper may have already done X→R)
        if needs_plan:
            self.ready = False
            if self._state != AgentState.R:
                self.set_unready(reason='plan_terminated')
        
        return needs_plan
    
    def set_plan(self, plan: SymbolicPlan):
        """
        Set a new plan for this agent (without state transition).
        
        Args:
            plan: The plan to execute
        """
        self.plan = plan
    
    def start_execution(self, timestamp: float = None, env_step: int = None):
        """
        Transition agent to executing state.
        
        Args:
            timestamp: Wall clock time for state tracking (from time.time())
            env_step: Environment step number for state tracking
        
        Should be called when all agents are ready and execution begins.
        """
        if self.ready and self.state == AgentState.W:
            self._set_state(AgentState.X, timestamp, env_step)
    
    def reset(self):
        """
        Reset the agent's internal state.
        
        Called at the start of a new episode. Subclasses can override
        to perform additional reset logic.
        """
        self.observation = None
        self.env_step = 0
        self.plan = None
        self.plan_count = 0
        self.ready = False
        self._state = AgentState.R  # Start in reasoning state
        self._start_time = time.time()
        self.state_history = [(time.time(), 0, AgentState.R)]  # Reset history
        self.message_buffer = []
        self.message_history = []
        self.buffer_senders = {}
        self._committed_plan = None
        self.plan_timeline = []
        self.memory.clear()
    
    def send_message(self, recipients: Union[str, List[str]], content: Any, metadata: Optional[Dict] = None):
        """
        Send a message to other agents via the message broker.
        
        Args:
            recipients: Single agent ID, list of agent IDs, or 'all' for broadcast
            content: Message content (any type)
            metadata: Optional metadata dict
        
        Returns:
            dict: The message record, or None if no broker is set
        """
        if self.message_broker is None:
            raise RuntimeError(f"Agent {self.agent_id} has no message broker set. Cannot send messages.")
        
        # Normalize recipients to list for memory recording
        if isinstance(recipients, str):
            recipient_list = [recipients] if recipients != 'all' else ['all']
        else:
            recipient_list = list(recipients)
        
        # Record outgoing message to memory
        self.memory.record_message_out(
            sender=self.agent_id,
            recipients=recipient_list,
            content=content,
            env_step=self.env_step
        )
        
        return self.message_broker.send_message(
            sender_id=self.agent_id,
            recipients=recipients,
            content=content,
            metadata=metadata
        )
    
    def get_messages(self, clear_buffer: bool = False):
        """
        Get messages from the buffer.
        
        Args:
            clear_buffer: If True, clear the buffer after reading
        
        Returns:
            list: List of messages in buffer
        """
        messages = self.message_buffer.copy()
        if clear_buffer:
            self.message_buffer = []
            self.buffer_senders = {}  # Clear sender tracking
        return messages
    
    def has_message_from(self, sender_id: str) -> bool:
        """
        Check if there's a message from a specific sender in the buffer.
        
        Args:
            sender_id: The agent ID to check for
        
        Returns:
            bool: True if there's a message from this sender
        """
        return self.buffer_senders.get(sender_id, False)
    
    def get_buffer_senders(self) -> list:
        """
        Get list of all senders who have messages in the buffer.
        
        Returns:
            list: List of sender IDs
        """
        return list(self.buffer_senders.keys())
    
    def has_messages(self):
        """Check if there are unread messages in the buffer."""
        return len(self.message_buffer) > 0

    def has_coop2_repair_request(self, messages: Optional[List[Dict[str, Any]]] = None) -> bool:
        """Return True when buffered messages include a COOP2 repair request."""
        messages = self.message_buffer if messages is None else messages
        return has_coop2_repair_message(messages)

    def describe_repair_intention(
        self,
        repair_context: Dict[str, Any],
        previous_statements: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Describe the agent's current intent for one ordered COOP2 repair round.

        LLM agents can override this to generate a natural-language statement.
        The base implementation is deterministic and summarizes the current plan.
        """
        previous_count = len(previous_statements or [])
        if self.plan is None:
            plan_summary = "I do not currently have a committed plan."
        else:
            remaining_actions = self.plan.actions[self.plan.current_action_index:]
            action_text = ", ".join(str(action) for action in remaining_actions)
            if not action_text:
                action_text = "no remaining actions"
            plan_summary = (
                f"I currently intend to follow plan #{self.plan.plan_id}: "
                f"{self.plan.specification}. Remaining actions: {action_text}."
            )
        failure_count = len(repair_context.get("failures", []))
        return (
            f"{plan_summary} I see {failure_count} predicted cooperative "
            f"constraint failure(s) and {previous_count} prior repair statement(s)."
        )
    
    def interrupt(self, timestamp: float = None, env_step: int = None):
        """
        Interrupt the agent (called when receiving messages during W or X state).
        
        Transitions to I state and marks agent as not ready.
        Agent must then decide to resume current plan or generate new plan.
        
        Args:
            timestamp: Wall clock time
            env_step: Environment step number
        """
        if self._state in [AgentState.W, AgentState.X]:
            self.set_unready(reason='message_received', timestamp=timestamp, env_step=env_step)
    
    @abstractmethod
    def handle_interrupt(self):
        """
        Handle interrupt state - decide whether to resume or replan.
        
        Called when agent is in I state with messages in buffer.
        Implementation should:
        1. Process messages via self.get_messages()
        2. Decide to resume current plan or generate new plan
        3. If resuming: do nothing (plan unchanged, set_ready will detect)
        4. If replanning: execute communication flow + call generate_plan()
        
        After this method, create_agent_thread calls set_ready().
        """
        pass
    
    @abstractmethod
    def handle_reasoning(self):
        """
        Handle reasoning state - execute communication flow and generate plan.
        
        Called when agent is in R state and needs a new plan.
        Implementation should:
        1. Execute any communication flow (wait_for, send_to)
        2. Generate a new plan via generate_plan()
        
        For simple agents: just call generate_plan()
        For topology agents: execute flow (wait → send → generate_plan)
        
        After this method, create_agent_thread calls set_ready().
        """
        pass
    
    def get_state_timeline(self):
        """
        Get the complete state transition timeline with relative times.
        
        Returns:
            list: List of (relative_time, env_step, state) tuples tracking all state changes.
                  relative_time is seconds since agent initialization.
        """
        return [(t - self._start_time, step, state) for t, step, state in self.state_history]


class SimpleAgent(Agent):
    """
    Simple agent implementation for testing and demonstration.
    
    This agent generates fixed plans based on hardcoded logic.
    It serves as a reference implementation and can be used for testing.
    """
    
    def __init__(self, agent_id: str, plan_spec: Optional[dict] = None):
        """
        Initialize simple agent.
        
        Args:
            agent_id: Unique identifier for this agent
            plan_spec: Optional plan specification dict with 'actions' and 'description'
        """
        super().__init__(agent_id)
        self.plan_spec = plan_spec or self._get_default_plan_spec()
    
    def _get_default_plan_spec(self) -> dict:
        """Get default plan specification for this agent."""
        from ..plan.plan import SymbolicAction
        
        # Default: move around and collect wood
        return {
            'description': 'Explore and collect wood',
            'actions': [
                SymbolicAction("move", {"direction": "left", "num_steps": 2}),
                SymbolicAction("collect", {"target": "wood"}),
                SymbolicAction("move", {"direction": "right", "num_steps": 3}),
                SymbolicAction("collect", {"target": "wood"}),
            ]
        }
    
    def observe(self, observation: Any, env_step: int):
        """
        Process observation.
        
        For the simple agent, just store the observation and step.
        """
        self.observation = observation
        self.env_step = env_step
    
    def generate_plan(self) -> SymbolicPlan:
        """
        Generate a plan based on the plan specification.
        
        Returns:
            SymbolicPlan: A new plan with actions from the plan spec
        """
        import copy
        # Deep copy actions to avoid mutating the original plan_spec
        # Each SymbolicAction tracks start_step/end_step during execution
        actions = copy.deepcopy(self.plan_spec['actions'])
        # Note: plan_count is incremented in set_ready(), not here
        self.plan = SymbolicPlan(
            specification=self.plan_spec['description'],
            actions=actions,
            plan_id=self.plan_count + 1,  # Preview next plan ID (will be finalized in set_ready)
            agent_id=self.agent_id,
            created_at_step=self.env_step
        )
        return self.plan
    
    def reset(self):
        """Reset agent state."""
        super().reset()
    
    def handle_interrupt(self):
        """
        Handle interrupt - SimpleAgent always resumes current plan.
        """
        messages = self.get_messages(clear_buffer=True)
        if self.has_coop2_repair_request(messages):
            self.generate_plan()
    
    def handle_reasoning(self):
        """
        Handle reasoning - SimpleAgent just generates a plan (no communication).
        """
        self.generate_plan()
