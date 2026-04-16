import omnigibson as og
from omnigibson.macros import gm
import sys


def test_behavior_task():
    gm.ENABLE_OBJECT_STATES = True
    gm.HEADLESS = True
    config = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
        },
        "task": {
            "type": "BehaviorTask",
            "activity_name": "putting_away_Halloween_decorations",
            "activity_definition_id": 0,
            "online_object_sampling": True,
            "use_presampled_robot_pose": False,
        },
        "robots": [
            {
                "type": "Fetch",
                "obs_modalities": ["rgb"],
            }
        ],
    }
    try:
        env = og.Environment(configs=config)
        print(
            "BehaviorTask instantiated successfully! Ground goal state options:",
            len(env.task.ground_goal_state_options),
        )
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    test_behavior_task()
