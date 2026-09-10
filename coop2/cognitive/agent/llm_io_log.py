"""Persist every LLM call of a run: the prompt in, the completion out.

Both already went to stdout when ``--llm-quiet`` was off, and stdout is exactly
where they stopped being useful -- several runs in this project were killed
mid-episode and left nothing behind but their videos, and a run that finishes
still leaves no record of what its agents were actually told (the goal text, for
instance, appears only in the prompt).

**JSON Lines, flushed per call, on purpose.** One object per line means a run
that is interrupted keeps every call it had already made, which a single JSON
array written at teardown does not. It also means the file can be read back with
a streaming pass instead of loading tens of MB at once.

Size: a call is roughly 8.5 k tokens of prompt, so budget ~35 kB per call --
about 2 MB for a 6-agent chain episode and ~15 MB for a 9-agent one, which puts
it in the same class as ``coop2_process_log.json``.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional

__all__ = ["LLMIORecorder", "read_llm_calls"]


def _jsonable(value: Any) -> Any:
    """Best-effort JSON form. Responses are not all strings.

    ``generate_plan`` hands back a parsed object and the interrupt decision a
    structured one (``response.reasoning`` is read at one call site), so this
    has to cope with pydantic models, dataclasses and plain text alike -- and
    must never raise, because a logging failure must not end an episode.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    for attribute in ("model_dump", "dict"):
        method = getattr(value, attribute, None)
        if callable(method):
            try:
                return _jsonable(method())
            except Exception:  # noqa: BLE001
                pass
    data = getattr(value, "__dict__", None)
    if isinstance(data, dict) and data:
        return {str(k): _jsonable(v) for k, v in data.items()}
    return str(value)


class LLMIORecorder:
    """Append one JSON object per LLM call. Shared by every agent in a run.

    Agents reason on their own threads, so the write is under a lock; the
    sequence number is assigned there too, which makes the file's own order the
    order the calls were issued in.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._sequence = 0
        self._started = time.time()
        self._failed = False
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)

    def record(
        self,
        agent_id: str,
        env_step: Optional[int],
        label: str,
        messages: Optional[List[Dict[str, Any]]],
        response: Any,
        usage: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write one call. Never raises: a lost log line is not a lost episode."""
        if self._failed:
            return
        try:
            with self._lock:
                self._sequence += 1
                record = {
                    "seq": self._sequence,
                    "wall_clock": round(time.time() - self._started, 3),
                    "agent_id": agent_id,
                    "env_step": env_step,
                    "label": label,
                    "messages": _jsonable(messages) or [],
                    "response": _jsonable(response),
                    "usage": _jsonable(usage) or {},
                }
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
        except Exception as error:  # noqa: BLE001
            # Say so once, then stay quiet rather than one line per call.
            self._failed = True
            print(f"[llm-io] disabling LLM call log after {error!r}")


def read_llm_calls(path: str):
    """Yield the records in an llm_calls.jsonl, skipping any truncated tail.

    A killed run can leave a partial final line; that is the format earning its
    keep, not a defect to repair.
    """
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue
