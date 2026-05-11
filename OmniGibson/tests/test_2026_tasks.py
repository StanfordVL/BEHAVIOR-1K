import os
import sys

import pytest
import yaml

from omnigibson.macros import gm

_DATASET_PATH = os.path.join(gm.DATA_PATH, "2026-challenge-task-instances")
_AVAILABLE_TASKS_PATH = os.path.join(_DATASET_PATH, "metadata", "available_tasks.yaml")

pytestmark = pytest.mark.skipif(
    not os.path.exists(_AVAILABLE_TASKS_PATH),
    reason="2026-challenge-task-instances dataset not present",
)


def _load_tasks_one_per_scene():
    """Return one (task_name, scene_model) tuple per scene from the 2026 dataset."""
    with open(_AVAILABLE_TASKS_PATH) as f:
        available_tasks = yaml.safe_load(f)
    seen_scenes = {}
    for task_name, instances in sorted(available_tasks.items()):
        scene_model = instances[0]["scene_model"]
        if scene_model not in seen_scenes:
            seen_scenes[scene_model] = task_name
    return [(task, scene) for scene, task in sorted(seen_scenes.items())]


# Evaluated at collection time; module-level skip above guards against missing file.
TASKS_2026 = _load_tasks_one_per_scene() if os.path.exists(_AVAILABLE_TASKS_PATH) else []


@pytest.mark.parametrize("task_name,scene_model", TASKS_2026, ids=[t[0] for t in TASKS_2026])
def test_2026_task_loads(task_name, scene_model):
    import omnigibson as og

    gm.ENABLE_OBJECT_STATES = True
    gm.ENABLE_TRANSITION_RULES = True
    gm.HEADLESS = True

    config = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": scene_model,
            "include_robots": False,
        },
        "task": {
            "type": "BehaviorTask",
            "activity_name": task_name,
            "activity_definition_id": 0,
            "activity_instance_id": 0,
            "online_object_sampling": False,
            "use_presampled_robot_pose": True,
            "include_obs": False,
        },
        "robots": [
            {
                "model": "r1pro",
                "obs_modalities": [],
            }
        ],
    }
    try:
        env = og.Environment(configs=config)
        env.reset()
        env.step(env.action_space.sample())
        print(
            f"Task {task_name!r} in {scene_model!r} loaded successfully. "
            f"Goal state options: {len(env.task.ground_goal_state_options)}"
        )
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        og.clear()


if __name__ == "__main__":
    for task_name, scene_model in TASKS_2026:
        test_2026_task_loads(task_name, scene_model)
