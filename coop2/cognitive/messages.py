"""
Message broker for inter-agent communication in MA-Crafter.

Routes messages between agents by adding to their message buffers.
"""

from typing import Dict, List, Optional, Any, Union
import time

from .coop2_messages import is_coop2_repair_message


class MessageBroker:
    """
    Message broker for routing messages between agents.
    
    The broker routes messages by adding them to recipient agent buffers.
    It also maintains a global log of all messages.
    """
    
    def __init__(self, agents: Dict[str, Any]):
        """
        Initialize the message broker.
        
        Args:
            agents: Dict mapping agent_id to Agent instances
        """
        self.agents = agents
        
        # Track all messages for analysis
        self.message_log = []
        
        # Optional wrapper reference for state change notifications
        self.wrapper = None
    
    def send_message(self, sender_id: str, recipients: Union[str, List[str]], 
                    content: Any, metadata: Optional[Dict] = None,
                    timestamp: Optional[float] = None, env_step: Optional[int] = None):
        """
        Send a message from one agent to one or more recipients.
        
        Args:
            sender_id: ID of the sending agent
            recipients: Single agent ID, list of agent IDs, or 'all' for broadcast
            content: Message content (any type)
            metadata: Optional metadata dict
            timestamp: Wall clock time. If None, uses current time.
            env_step: Environment step number. If None, uses sender's current step.
        
        Returns:
            dict: The message record
        """
        if timestamp is None:
            timestamp = time.time()
        if env_step is None and sender_id in self.agents and self.agents[sender_id] is not None:
            env_step = self.agents[sender_id].env_step
        
        # Normalize recipients to a list
        if recipients == 'all':
            recipient_list = [aid for aid in self.agents.keys() if aid != sender_id]
        elif isinstance(recipients, str):
            recipient_list = [recipients]
        else:
            recipient_list = recipients
        
        # Create message record with relative timestamp
        sender_agent = self.agents.get(sender_id)
        relative_timestamp = timestamp - sender_agent._start_time if sender_agent else timestamp
        
        message_record = {
            'timestamp': relative_timestamp,
            'env_step': env_step,
            'sender': sender_id,
            'recipients': recipient_list,
            'content': content,
            'metadata': metadata or {}
        }
        
        # Deliver to each recipient's buffer and history
        for recipient_id in recipient_list:
            if recipient_id in self.agents and self.agents[recipient_id] is not None:
                recipient = self.agents[recipient_id]
                msg_copy = {
                    'timestamp': relative_timestamp,
                    'env_step': env_step,
                    'sender': sender_id,
                    'content': content,
                    'metadata': metadata or {}
                }
                recipient.message_buffer.append(msg_copy)
                recipient.message_history.append(msg_copy)
                
                # Record incoming message to agent's memory
                recipient.memory.record_message_in(
                    sender=sender_id,
                    recipients=recipient_list,
                    content=content,
                    timestamp=timestamp,
                    env_step=env_step
                )
                
                # Update buffer sender indicator
                recipient.buffer_senders[sender_id] = True
                
                # Interrupt waiting agents for ordinary communication so topology
                # protocols can still coordinate before execution. Once an agent
                # is executing, ordinary messages are buffered for the next
                # reasoning point unless the sender marks the message as an
                # execution interrupt, such as COOP2 repair or centralized
                # leader planning broadcasts.
                from .agent import AgentState
                is_repair_message = is_coop2_repair_message(msg_copy)
                message_type = (metadata or {}).get("type") or (metadata or {}).get("message_type")
                is_execution_interrupt = bool((metadata or {}).get("interrupts_execution")) or message_type == "leader_broadcast"
                should_interrupt = (
                    recipient.state == AgentState.W
                    or (recipient.state == AgentState.X and (is_repair_message or is_execution_interrupt))
                )
                if should_interrupt:
                    recipient.interrupt(timestamp, env_step)
                    # Notify wrapper of state change if available
                    if self.wrapper is not None:
                        self.wrapper.notify_state_change()
        
        # Log the message
        self.message_log.append(message_record)
        
        return message_record

    def get_interrupted_agents(self) -> List[str]:
        """
        Get list of agent IDs that are currently interrupted with pending messages.
        
        Returns:
            List of agent IDs in interrupted state with messages in buffer
        """
        from .agent import AgentState
        
        interrupted = []
        for agent_id, agent in self.agents.items():
            if agent is not None and agent.state == AgentState.I and agent.has_messages():
                interrupted.append(agent_id)
        return interrupted
    
    def get_message_log(self):
        """Get all messages sent through the broker."""
        return self.message_log.copy()
    
    def reset(self):
        """Clear message log. Agent buffers/history are reset by agents themselves."""
        self.message_log = []
