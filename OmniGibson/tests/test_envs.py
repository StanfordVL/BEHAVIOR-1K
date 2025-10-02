import omnigibson as og
from omnigibson.macros import gm


def task_tester(task_type):
    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "load_object_categories": ["floors", "breakfast_table"],
        },
        "robots": [
            {
                "type": "Fetch",
                "obs_modalities": [],
            }
        ],
        # Task kwargs
        "task": {
            "type": task_type,
            # BehaviorTask-specific
            "activity_name": "laying_wood_floors",
            "online_object_sampling": True,
            "use_presampled_robot_pose": False,
        },
    }

    if og.sim is None:
        # Make sure GPU dynamics are enabled (GPU dynamics needed for cloth)
        gm.ENABLE_OBJECT_STATES = True
        gm.USE_GPU_DYNAMICS = True
        gm.ENABLE_TRANSITION_RULES = False
    else:
        # Make sure sim is stopped
        og.sim.stop()

    # Create the environment
    env = og.Environment(configs=cfg)

    env.reset()
    for _ in range(5):
        env.step(env.robots[0].action_space.sample())

    # Clear the sim
    og.clear()


def test_dummy_task():
    task_tester("DummyTask")


def test_point_reaching_task():
    task_tester("PointReachingTask")


def test_point_navigation_task():
    task_tester("PointNavigationTask")


def test_behavior_task():
    task_tester("BehaviorTask")


def test_rs_int_full_load():
    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
        },
        "robots": [
            {
                "type": "Fetch",
                "obs_modalities": [],
            }
        ],
        # Task kwargs
        "task": {
            "type": "DummyTask",
        },
    }

    # Make sure sim is stopped
    if og.sim:
        og.sim.stop()

    # Create the environment
    env = og.Environment(configs=cfg)

    env.reset()
    for _ in range(5):
        env.step(env.robots[0].action_space.sample())

    # Clear the sim
    og.clear()
