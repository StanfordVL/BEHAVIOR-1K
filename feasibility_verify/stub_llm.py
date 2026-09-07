"""A scripted stand-in for LLMClient, so the runner can be exercised offline.

M7's real risk is not the language model -- it is whether L4's agent FSM, L5's
topology and L6's runner actually drive *this* environment. Those need no API
key to test, and testing them separately means a later failure with real
credentials is unambiguous: the model, not the plumbing.

The plans are not canned. The stub reads ``Current Reachable Targets`` out of
the prompt it is handed and plans against those ids, so it exercises the same
path a real model does: an id it invents is rejected by L2 exactly as the
model's would be.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from coop2.cognitive.agent.llm_client import (
    GraspAction,
    InterruptDecision,
    LLMInterruptResponse,
    LLMMessageResponse,
    LLMPlanResponse,
    NavigateToAction,
    Task,
    TaskSpecification,
    WaitAction,
)

#: BDDL instance ids: synset (dotted, with a .n. part) plus _index.
_ID = re.compile(r"\b([a-z_]+(?:\.[a-z])?\.n\.\d+_\d+)\b")


class StubLLMClient:
    """Same surface as LLMClient, no network."""

    def __init__(self, model: str = "stub", verbose: bool = False):
        self.model = model
        self.verbose = verbose
        self.calls = 0
        self.plans_emitted: List[List[str]] = []

    @classmethod
    def from_env(cls, backend=None, model=None, verbose=False) -> "StubLLMClient":
        return cls(model=model or "stub", verbose=verbose)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _targets(messages: List[Dict]) -> List[str]:
        """Ids offered under Current Reachable Targets, in order."""
        text = "\n".join(str(m.get("content", "")) for m in messages)
        section = text.split("Current Reachable Targets", 1)
        hay = section[1] if len(section) > 1 else text
        seen, out = set(), []
        for match in _ID.finditer(hay):
            if match.group(1) not in seen:
                seen.add(match.group(1))
                out.append(match.group(1))
        return out

    def _usage(self) -> Dict[str, Any]:
        """Every key the real client returns.

        llm_usage aggregates these by name; a missing one is a KeyError deep in
        the summary, after the episode has already run.
        """
        return {
            "model": self.model,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_seconds": 0.0,
            "retry_count": 0,
            "rate_limit_retry_count": 0,
            "api_error_retry_count": 0,
            "guard_filter_count": 0,
            "calls": 1,
        }

    # -- LLMClient surface -------------------------------------------------

    def generate(self, messages, response_format=None, temperature: float = 0.7, **kwargs) -> Tuple[Any, Dict]:
        self.calls += 1
        if response_format is LLMPlanResponse or response_format is None:
            return self._plan(messages), self._usage()
        if response_format is LLMInterruptResponse:
            return (
                LLMInterruptResponse(
                    decision=InterruptDecision.REPLAN,
                    reasoning="stub always replans, to exercise the reasoning path",
                    new_plan=self._plan(messages),
                ),
                self._usage(),
            )
        if response_format is LLMMessageResponse:
            return (
                LLMMessageResponse(
                    recipients=[], content="stub has nothing to say", reasoning="stub"
                ),
                self._usage(),
            )
        raise AssertionError(f"stub does not know response_format {response_format!r}")

    def generate_plan(self, messages, temperature: float = 0.7) -> Tuple[Any, Dict]:
        return self.generate(messages, response_format=LLMPlanResponse, temperature=temperature)

    def generate_message(self, messages, temperature: float = 0.7) -> Tuple[Any, Dict]:
        return self.generate(messages, response_format=LLMMessageResponse, temperature=temperature)

    def generate_interrupt_decision(self, messages, temperature: float = 0.7) -> Tuple[Any, Dict]:
        return self.generate(messages, response_format=LLMInterruptResponse, temperature=temperature)

    # -- the "policy" ------------------------------------------------------

    def _plan(self, messages: List[Dict]) -> LLMPlanResponse:
        """Go to the first offered target and pick it up; wait if none."""
        targets = self._targets(messages)
        if not targets:
            self.plans_emitted.append(["wait"])
            return LLMPlanResponse(
                task=TaskSpecification(task=Task.HOLDING, object_type="none"),
                actions=[WaitAction()],
                reasoning="stub saw no reachable targets",
            )
        target = targets[0]
        actions = [NavigateToAction(target=target), GraspAction(target=target)]
        self.plans_emitted.append([a.action_type for a in actions])
        return LLMPlanResponse(
            task=TaskSpecification(task=Task.HOLDING, object_type=target),
            actions=actions,
            reasoning=f"stub picked the first reachable target, {target}",
        )
