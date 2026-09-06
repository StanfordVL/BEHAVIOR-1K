"""
Memory management for agents in MA-Crafter.

Provides a unified event memory that tracks messages and plans in chronological order.
"""

import time
from typing import Any, Optional, List, Dict


class AgentMemory:
    """
    Memory that stores recent events (messages and plans) in chronological order.
    
    Each event is a dict with:
        "type": "message_in" | "message_out" | "plan"
        "timestamp": float (wall clock time)
        "env_step": int
        
    For messages:
        "sender": str
        "recipients": list
        "content": any
        
    For plans:
        "plan_id": int
        "specification": str
        "actions": list of action strings

    For plan failures:
        "plan_id": int
        "specification": str
        "failed_action": str
        "failure_reason": str
    """
    
    def __init__(self, max_size: int = 10):
        """
        Initialize memory.
        
        Args:
            max_size: Maximum number of events to keep in memory
        """
        self.max_size = max_size
        self.events: List[Dict] = []
    
    def _add(self, event: Dict):
        """
        Add an event to memory, maintaining size limit.
        
        Args:
            event: Event dict to add
        """
        self.events.append(event)
        if len(self.events) > self.max_size:
            self.events = self.events[-self.max_size:]
    
    def record_message_in(
        self,
        sender: str,
        recipients: List[str],
        content: Any,
        timestamp: float = None,
        env_step: int = 0
    ):
        """
        Record an incoming message.
        
        Args:
            sender: Sender agent ID
            recipients: List of recipient agent IDs
            content: Message content
            timestamp: Wall clock time (default: now)
            env_step: Environment step
        """
        self._add({
            "type": "message_in",
            "timestamp": timestamp or time.time(),
            "env_step": env_step,
            "sender": sender,
            "recipients": recipients,
            "content": content,
        })
    
    def record_message_out(
        self,
        sender: str,
        recipients: List[str],
        content: Any,
        timestamp: float = None,
        env_step: int = 0
    ):
        """
        Record an outgoing message.
        
        Args:
            sender: Sender agent ID (self)
            recipients: List of recipient agent IDs
            content: Message content
            timestamp: Wall clock time (default: now)
            env_step: Environment step
        """
        self._add({
            "type": "message_out",
            "timestamp": timestamp or time.time(),
            "env_step": env_step,
            "sender": sender,
            "recipients": recipients,
            "content": content,
        })
    
    def record_plan(
        self,
        plan_id: int,
        specification: str,
        actions: List[str],
        timestamp: float = None,
        env_step: int = 0
    ):
        """
        Record a new plan.
        
        Args:
            plan_id: Plan ID number
            specification: Task/goal description
            actions: List of action strings
            timestamp: Wall clock time (default: now)
            env_step: Environment step
        """
        self._add({
            "type": "plan",
            "timestamp": timestamp or time.time(),
            "env_step": env_step,
            "plan_id": plan_id,
            "specification": specification,
            "actions": actions,
        })

    def record_plan_failure(
        self,
        plan_id: int,
        specification: str,
        failed_action: str,
        failure_reason: str,
        timestamp: float = None,
        env_step: int = 0
    ):
        """
        Record why the most recent plan failed.

        Args:
            plan_id: Plan ID number
            specification: Task/goal description
            failed_action: Action string that failed
            failure_reason: Environment-backed failure reason
            timestamp: Wall clock time (default: now)
            env_step: Environment step
        """
        self._add({
            "type": "plan_failure",
            "timestamp": timestamp or time.time(),
            "env_step": env_step,
            "plan_id": plan_id,
            "specification": specification,
            "failed_action": failed_action,
            "failure_reason": failure_reason,
        })
    
    def get_events(self) -> List[Dict]:
        """
        Get all events in memory.
        
        Returns:
            List of event dicts in chronological order
        """
        return self.events.copy()
    
    def get_messages(self, direction: str = None) -> List[Dict]:
        """
        Get message events from memory.
        
        Args:
            direction: "in", "out", or None for both
            
        Returns:
            List of message event dicts
        """
        if direction is None:
            return [e for e in self.events if e["type"] in ("message_in", "message_out")]
        elif direction == "in":
            return [e for e in self.events if e["type"] == "message_in"]
        elif direction == "out":
            return [e for e in self.events if e["type"] == "message_out"]
        else:
            raise ValueError(f"Invalid direction: {direction}")
    
    def get_plans(self) -> List[Dict]:
        """
        Get plan events from memory.
        
        Returns:
            List of plan event dicts
        """
        return [e for e in self.events if e["type"] == "plan"]
    
    def clear(self):
        """Clear all events from memory."""
        self.events = []
    
    def __len__(self) -> int:
        """Return number of events in memory."""
        return len(self.events)
    
    def summary(self) -> str:
        """
        Get a human-readable summary of memory.
        
        Returns:
            String summary of recent events
        """
        if not self.events:
            return "No events in memory."
        
        lines = []
        for event in self.events:
            step = event['env_step']
            if event['type'] == 'message_in':
                lines.append(f"[Step {step}] IN: {event['sender']} -> {event['recipients']}: {event['content']}")
            elif event['type'] == 'message_out':
                lines.append(f"[Step {step}] OUT: {event['sender']} -> {event['recipients']}: {event['content']}")
            elif event['type'] == 'plan':
                lines.append(f"[Step {step}] PLAN #{event['plan_id']}: {event['specification']}")
                for i, action in enumerate(event['actions'], 1):
                    lines.append(f"  {i}. {action}")
            elif event['type'] == 'plan_failure':
                lines.append(
                    f"[Step {step}] PLAN #{event['plan_id']} FAILED: "
                    f"{event.get('failed_action', '?')} because {event.get('failure_reason', '?')}"
                )
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"AgentMemory({len(self.events)} events, max_size={self.max_size})"
