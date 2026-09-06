"""
Shared centralized topology protocol.

This module owns the centralized decision order:
leader sends a planning request, followers respond, then the leader plans.
Rule-based and LLM agents provide only the message and planning hooks.
"""

import time
from typing import Any, Dict, List, Tuple

from coop2.cognitive.plan import SymbolicPlan


class CentralizedLeaderFlow:
    """Leader-side flow for the centralized topology."""

    centralized_response_timeout_seconds: float = 5.0

    def _execute_flow(self) -> SymbolicPlan:
        """
        Execute the leader decision flow:
        1. Send a planning request to followers.
        2. Wait for follower responses.
        3. Generate the leader plan using those responses.
        """
        self.expected_responses = set()
        self.follower_responses = []

        if self.send_to and self.message_broker is not None:
            content, metadata = self._build_leader_broadcast()
            self.send_message(
                recipients=self.send_to,
                content=content,
                metadata=metadata,
            )
            self.expected_responses = set(self.wait_for_response)
            self._on_leader_broadcast_sent(content)

        timeout_seconds = self._centralized_response_timeout_seconds()
        start_time = time.time()
        while self.expected_responses and (time.time() - start_time) < timeout_seconds:
            for message in self.get_messages(clear_buffer=True):
                if message.get("metadata", {}).get("type") != "follower_response":
                    continue
                sender = str(message.get("sender"))
                if sender not in self.expected_responses:
                    continue
                self.expected_responses.remove(sender)
                self._record_follower_response(sender, message)
                self._on_follower_response_received(sender)

            if self.expected_responses:
                time.sleep(0.05)

        if self.expected_responses:
            self._handle_leader_response_timeout(set(self.expected_responses))
        else:
            self._on_all_follower_responses_received()

        self.plan = self._generate_leader_plan_after_responses()
        return self.plan

    def _build_leader_broadcast(self) -> Tuple[Any, Dict[str, Any]]:
        raise NotImplementedError

    def _generate_leader_plan_after_responses(self) -> SymbolicPlan:
        raise NotImplementedError

    def _centralized_response_timeout_seconds(self) -> float:
        return float(getattr(self, "centralized_response_timeout_seconds", 5.0))

    def _record_follower_response(self, sender: str, message: Dict[str, Any]) -> None:
        responses = getattr(self, "follower_responses", None)
        if responses is None:
            self.follower_responses = []
            responses = self.follower_responses
        responses.append(f"[{sender}]: {message.get('content')}")

    def _handle_leader_response_timeout(self, expected_responses: set[str]) -> None:
        raise TimeoutError(f"Leader timeout waiting for {expected_responses}")

    def _flow_verbose(self) -> bool:
        return bool(getattr(self, "verbose", True))

    def _on_leader_broadcast_sent(self, content: Any) -> None:
        if self._flow_verbose():
            print(f"  [{self.agent_id}] Sent to {self.send_to}")

    def _on_follower_response_received(self, sender: str) -> None:
        if self._flow_verbose():
            remaining = len(getattr(self, "expected_responses", []))
            print(f"  [{self.agent_id}] Response from {sender} ({remaining} remaining)")

    def _on_all_follower_responses_received(self) -> None:
        if self._flow_verbose():
            print(f"  [{self.agent_id}] All responses received")


class CentralizedFollowerFlow:
    """Follower-side flow for the centralized topology."""

    def _execute_flow(self) -> SymbolicPlan:
        """
        Execute the follower decision flow:
        1. Wait for leader planning request when the leader is still planning.
        2. Send a response to the leader.
        3. Generate a local executable plan.
        """
        if self.wait_for and self._leader_is_not_ready():
            while not self.wait_for_messages_from(self.wait_for):
                if not self._leader_is_not_ready():
                    break
                time.sleep(0.05)

        leader_messages = self.get_messages(clear_buffer=True)
        self._handle_leader_messages(leader_messages)

        if self.send_to and self.message_broker is not None and self._leader_is_not_ready():
            content, metadata = self._build_follower_response()
            self.send_message(
                recipients=self.send_to,
                content=content,
                metadata=metadata,
            )
            self._on_follower_response_sent(content)

        self.plan = self._generate_follower_plan_after_response()
        return self.plan

    def _build_follower_response(self) -> Tuple[Any, Dict[str, Any]]:
        raise NotImplementedError

    def _generate_follower_plan_after_response(self) -> SymbolicPlan:
        raise NotImplementedError

    def _leader_is_not_ready(self) -> bool:
        leader_agent = getattr(self, "leader_agent", None)
        if leader_agent is None:
            return False
        return not leader_agent.ready

    def _handle_leader_messages(self, messages: List[Dict[str, Any]]) -> None:
        pass

    def _flow_verbose(self) -> bool:
        return bool(getattr(self, "verbose", True))

    def _on_follower_response_sent(self, content: Any) -> None:
        if self._flow_verbose():
            print(f"  [{self.agent_id}] Sent response to {self.send_to}")
