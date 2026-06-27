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

Re-running reproduces the committed JSON (sources permitting). For 2026-only tasks, manually
curated fields already present in the committed JSON (videos, durations, thumbnails, etc.) are
preserved while source-derived room and scene metadata are refreshed.
"""

import csv
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# Sources
TASK_DATA_2025 = REPO_ROOT / "docs" / "challenge" / "archive" / "2025" / "task_data.json"
DATASET_2026 = REPO_ROOT / "datasets" / "2026-challenge-task-instances" / "metadata"
AVAILABLE_TASKS = DATASET_2026 / "available_tasks.yaml"
CUSTOM_LISTS = DATASET_2026 / "task_custom_lists.json"
# Canonical B100 task ordering: column "Task ID" is a numeric index, column "Task" is the id.
B100_TASK_LIST = DATASET_2026 / "B100_task_misc.csv"

# Output
OUT_FILE = REPO_ROOT / "docs" / "challenge" / "task_data.json"


def title_case(task_id: str) -> str:
    """Derive a human-readable display name from a task id.

    Auto-generated; may benefit from light manual polish in the committed JSON.
    """
    return task_id.replace("_", " ").title()


def load_b100_order() -> dict:
    """Map each task id to its index in the canonical B100 task list."""
    order = {}
    with open(B100_TASK_LIST, newline="") as f:
        for row in csv.DictReader(f):
            task_id = row["Task"].strip()
            if task_id:
                order[task_id] = int(row["Task ID"])
    return order


def load_scene_models(available_tasks: dict) -> dict:
    """Map each task id to its configured scene model."""
    scene_models = {}
    for task_id, instances in available_tasks.items():
        scenes = sorted(
            {
                instance.get("scene_model")
                for instance in instances.values()
                if isinstance(instance, dict) and instance.get("scene_model")
            }
        )
        if scenes:
            scene_models[task_id] = scenes[0]
    return scene_models


def main() -> None:
    existing_by_id = {}
    if OUT_FILE.exists():
        existing_by_id = {task["id"]: task for task in json.loads(OUT_FILE.read_text())["tasks"]}

    # 50 carryover tasks: copy 2025 entries verbatim (full media preserved).
    carryover = json.loads(TASK_DATA_2025.read_text())["tasks"]
    carryover_ids = {t["id"] for t in carryover}
    available_tasks = yaml.safe_load(AVAILABLE_TASKS.read_text())
    scene_models = load_scene_models(available_tasks)
    for task in carryover:
        scene_model = scene_models.get(task["id"])
        if scene_model:
            task["scene_model"] = scene_model

    # 50 new tasks: authoritative list = keys of available_tasks.yaml, ordered by their index
    # in the canonical B100 task list. Tasks absent from B100 keep their available_tasks order
    # and are appended after the indexed ones (stable sort + sentinel index).
    b100_order = load_b100_order()
    new_ids = list(available_tasks.keys())
    new_ids.sort(key=lambda task_id: b100_order.get(task_id, len(b100_order) + 1))
    custom_lists = json.loads(CUSTOM_LISTS.read_text())

    new_tasks = []
    for task_id in new_ids:
        if task_id in carryover_ids:
            # Defensive: skip anything already covered by the 2025 carryover.
            continue
        rooms = custom_lists.get(task_id, {}).get("room_types", [])
        task = dict(existing_by_id.get(task_id, {}))
        task["id"] = task_id
        task.setdefault("name", title_case(task_id))
        task["rooms"] = rooms
        scene_model = scene_models.get(task_id)
        if scene_model:
            task["scene_model"] = scene_model
        new_tasks.append(task)

    tasks = carryover + new_tasks

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps({"tasks": tasks}, indent=2) + "\n")

    print(f"Wrote {len(tasks)} tasks ({len(carryover)} carryover + {len(new_tasks)} new) -> {OUT_FILE}")


if __name__ == "__main__":
    main()
