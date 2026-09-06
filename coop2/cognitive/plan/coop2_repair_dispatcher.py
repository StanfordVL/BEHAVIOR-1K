"""Planning-wrapper delivery path for COOP2 repair requests."""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Dict, Iterable, Optional

from coop2._repair_shim.message_protocol import COOP2_REPAIR_SENDER_ID, coop2_repair_metadata


class Coop2RepairDispatcher:
    """Deliver COOP2 repair contexts to this repo's Agent/message-broker model."""

    def __init__(
        self,
        agents: Dict[str, Any],
        message_broker_getter: Callable[[], Optional[Any]],
    ):
        self.agents = agents
        self._message_broker_getter = message_broker_getter

    def dispatch(self, affected_agents: Iterable[str], repair_context: Dict[str, Any], env_step: int) -> None:
        """Attach repair-channel transcript and interrupt affected agents."""
        affected_agents = list(affected_agents)
        self._attach_repair_channel(affected_agents, repair_context, env_step)
        metadata = coop2_repair_metadata()
        message_broker = self._message_broker_getter()
        if message_broker is not None:
            message_broker.send_message(
                sender_id=COOP2_REPAIR_SENDER_ID,
                recipients=affected_agents,
                content=repair_context,
                metadata=metadata,
                env_step=env_step,
            )
            return

        timestamp = time.time()
        for agent_id in affected_agents:
            agent = self.agents.get(agent_id)
            if agent is None:
                continue
            message = {
                "timestamp": timestamp - agent._start_time,
                "env_step": env_step,
                "sender": COOP2_REPAIR_SENDER_ID,
                "content": repair_context,
                "metadata": metadata,
            }
            agent.message_buffer.append(message)
            agent.message_history.append(message)
            agent.memory.record_message_in(
                sender=COOP2_REPAIR_SENDER_ID,
                recipients=affected_agents,
                content=repair_context,
                timestamp=timestamp,
                env_step=env_step,
            )
            agent.buffer_senders[COOP2_REPAIR_SENDER_ID] = True
            agent.interrupt(timestamp=timestamp, env_step=env_step)

    def _attach_repair_channel(
        self,
        affected_agents: Iterable[str],
        repair_context: Dict[str, Any],
        env_step: int,
    ) -> None:
        """Run one ordered COOP2 repair-intention round and attach the transcript."""
        order = sorted(affected_agents, key=self._agent_id_sort_key)
        statements = []
        for agent_id in order:
            agent = self.agents.get(agent_id)
            if agent is None:
                continue
            try:
                statement = agent.describe_repair_intention(
                    repair_context=repair_context,
                    previous_statements=statements,
                )
            except Exception as exc:
                statement = f"Repair intention unavailable: {exc}"
            statements.append({
                "turn_index": len(statements),
                "agent_id": agent_id,
                "env_step": env_step,
                "statement": str(statement),
            })

        repair_context["repair_channel"] = {
            "protocol": "ordered_one_round_intention_then_global_revision",
            "order": order,
            "statements": statements,
            "revision_instruction": (
                "Revise your plan using the repair failures, all committed plan "
                "views, and the full ordered repair-channel transcript."
            ),
        }

    @staticmethod
    def _agent_id_sort_key(agent_id: Any):
        text = str(agent_id)
        match = re.search(r"(\d+)$", text)
        if match:
            return (text[:match.start()], int(match.group(1)), text)
        return (text, -1, text)
