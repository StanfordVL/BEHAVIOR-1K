"""One-off generator for the 2026 challenge demo-gallery task data.

The 2026 BEHAVIOR Challenge covers 100 tasks: the 50 tasks from the 2025 challenge
(carried over verbatim, with their existing Vimeo media) plus 50 brand-new tasks whose
instances live in ``datasets/2026-challenge-task-instances/``.

This script merges those two sources into ``docs/challenge/task_data.json``, which is
committed and consumed at build time by ``docs/gen_task_pages.py`` (exactly like the 2025
``docs/challenge/archive/2025/task_data.json``). It is intentionally a manual, idempotent one-off: the
mkdocs build does NOT depend on the ``datasets/`` repo, which is a separate git repository
not guaranteed to be present in the docs-build environment.

Run from the repository root::

    python docs/gen_2026_task_data.py

Re-running reproduces the committed JSON (sources permitting). New tasks have no
video/duration/instruction/thumbnail yet, so the gallery shows "coming soon" placeholders
until those are added (videos are Vimeo URLs added by hand to the JSON).
"""

import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# Sources
TASK_DATA_2025 = REPO_ROOT / "docs" / "challenge" / "archive" / "2025" / "task_data.json"
DATASET_2026 = REPO_ROOT / "datasets" / "2026-challenge-task-instances" / "metadata"
AVAILABLE_TASKS = DATASET_2026 / "available_tasks.yaml"
CUSTOM_LISTS = DATASET_2026 / "task_custom_lists.json"

# Output
OUT_FILE = REPO_ROOT / "docs" / "challenge" / "task_data.json"


def title_case(task_id: str) -> str:
    """Derive a human-readable display name from a task id.

    Auto-generated; may benefit from light manual polish in the committed JSON.
    """
    return task_id.replace("_", " ").title()


def main() -> None:
    # 50 carryover tasks: copy 2025 entries verbatim (full media preserved).
    carryover = json.loads(TASK_DATA_2025.read_text())["tasks"]
    carryover_ids = {t["id"] for t in carryover}

    # 50 new tasks: authoritative list = keys of available_tasks.yaml.
    new_ids = list(yaml.safe_load(AVAILABLE_TASKS.read_text()).keys())
    custom_lists = json.loads(CUSTOM_LISTS.read_text())

    new_tasks = []
    for task_id in new_ids:
        if task_id in carryover_ids:
            # Defensive: skip anything already covered by the 2025 carryover.
            continue
        rooms = custom_lists.get(task_id, {}).get("room_types", [])
        new_tasks.append(
            {
                "id": task_id,
                "name": title_case(task_id),
                "rooms": rooms,
                # No video / duration / instruction / thumbnail yet -> placeholders.
            }
        )

    tasks = carryover + new_tasks

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps({"tasks": tasks}, indent=2) + "\n")

    print(f"Wrote {len(tasks)} tasks ({len(carryover)} carryover + {len(new_tasks)} new) -> {OUT_FILE}")


if __name__ == "__main__":
    main()
