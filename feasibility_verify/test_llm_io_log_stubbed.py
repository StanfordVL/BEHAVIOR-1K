"""CPU-only test for the per-run LLM call log.

No Isaac, no GPU, no model: a stub client returns canned completions and the
recorder is read back off disk. What is pinned is the reason the format was
chosen -- a run that dies mid-episode keeps every call it had already made --
and the fact that a logging failure cannot end an episode.

Run:
    python feasibility_verify/test_llm_io_log_stubbed.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coop2.cognitive.agent.llm_io_log import LLMIORecorder, read_llm_calls


def ok(message: str) -> None:
    print(f"  ok: {message}")


class Structured:
    """Stands in for the parsed objects generate_plan / the decision return."""

    def __init__(self) -> None:
        self.reasoning = "because the apple is nearer"
        self.action = {"action_type": "grasp", "args": {"target": "apple.n.01_1"}}


class Pydanticish:
    def model_dump(self):
        return {"decision": "resume", "confidence": 0.8}


def main() -> int:
    directory = tempfile.mkdtemp(prefix="llm_io_")
    path = os.path.join(directory, "llm_calls.jsonl")
    recorder = LLMIORecorder(path)

    print("test: a call records its prompt, its completion and its usage")
    messages = [
        {"role": "system", "content": "You are agent_0."},
        {"role": "user", "content": "GLOBAL OBJECTIVE: Put all nine apples on coffee_table.n.01_1."},
    ]
    recorder.record(
        agent_id="agent_0", env_step=0, label="Plan Generation",
        messages=messages, response="ontop(apple.n.01_1, coffee_table.n.01_1)",
        usage={"total_tokens": 8500, "prompt_tokens": 8200, "completion_tokens": 300},
    )
    records = list(read_llm_calls(path))
    assert len(records) == 1
    record = records[0]
    assert record["agent_id"] == "agent_0" and record["env_step"] == 0
    assert record["label"] == "Plan Generation"
    assert record["messages"] == messages, "the prompt must round-trip verbatim"
    assert "nine apples" in record["messages"][1]["content"], (
        "the goal text reaches the run folder only through the prompt"
    )
    assert record["response"] == "ontop(apple.n.01_1, coffee_table.n.01_1)"
    assert record["usage"]["total_tokens"] == 8500
    assert record["seq"] == 1
    ok("prompt, completion and usage all land, attributed to the agent and step")

    print("\ntest: a completion that is not a string is still readable")
    recorder.record(agent_id="agent_1", env_step=120, label="Plan Generation",
                    messages=[], response=Structured(), usage={})
    recorder.record(agent_id="agent_2", env_step=120, label="Interrupt Decision",
                    messages=[], response=Pydanticish(), usage={})
    records = list(read_llm_calls(path))
    assert records[1]["response"]["reasoning"].startswith("because"), (
        "a plain object must be flattened, not stringified away"
    )
    assert records[1]["response"]["action"]["action_type"] == "grasp"
    assert records[2]["response"] == {"decision": "resume", "confidence": 0.8}, (
        "model_dump() is preferred over __dict__ for a pydantic-shaped response"
    )
    ok("objects, models and plain text all serialise")

    print("\ntest: a truncated final line -- a killed run -- loses only that line")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"seq": 4, "agent_id": "agent_3", "messa')   # no newline
    records = list(read_llm_calls(path))
    assert len(records) == 3, f"expected the 3 complete records, got {len(records)}"
    assert [r["seq"] for r in records] == [1, 2, 3]
    ok("the three completed calls survive the interrupted fourth")

    print("\ntest: concurrent agents do not interleave inside a line")
    busy = os.path.join(directory, "concurrent.jsonl")
    parallel = LLMIORecorder(busy)

    def hammer(agent_index: int) -> None:
        for step in range(25):
            parallel.record(
                agent_id=f"agent_{agent_index}", env_step=step, label="Plan Generation",
                messages=[{"role": "user", "content": "x" * 500}],
                response="y" * 200, usage={"total_tokens": 1},
            )

    threads = [threading.Thread(target=hammer, args=(i,)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    with open(busy, encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip()]
    assert len(lines) == 150, f"expected 150 lines, got {len(lines)}"
    parsed = [json.loads(line) for line in lines]          # raises on interleaving
    assert sorted(r["seq"] for r in parsed) == list(range(1, 151)), "sequence gaps"
    ok("150 calls from 6 threads, every line whole and numbered once")

    print("\ntest: a broken log does not take the episode with it")
    doomed = LLMIORecorder(os.path.join(directory, "nested", "calls.jsonl"))
    os.chmod(os.path.dirname(doomed.path), 0o500)           # read-only directory
    try:
        doomed.record(agent_id="agent_0", env_step=0, label="Plan Generation",
                      messages=[], response="x", usage={})
        doomed.record(agent_id="agent_0", env_step=1, label="Plan Generation",
                      messages=[], response="x", usage={})
    finally:
        os.chmod(os.path.dirname(doomed.path), 0o700)
    assert doomed._failed, "it should have given up rather than raise"
    ok("an unwritable log warns once and is then inert")

    print("\ntest: reading a run that never wrote one yields nothing, not an error")
    assert list(read_llm_calls(os.path.join(directory, "absent.jsonl"))) == []
    ok("a missing log reads as empty")

    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
