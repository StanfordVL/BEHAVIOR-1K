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
            "activity_name": "assembling_gift_baskets",
            "online_object_sampling": True
        },
    }

    # Make sure sim is stopped
    og.sim.stop()

    # Make sure GPU dynamics are enabled (GPU dynamics needed for cloth) and no flatcache
    gm.ENABLE_OBJECT_STATES = True
    gm.USE_GPU_DYNAMICS = True
    gm.ENABLE_FLATCACHE = False

    # Create the environment
    env = og.Environment(configs=cfg, action_timestep=1 / 60., physics_timestep=1 / 60.)

    env.reset()
    for _ in range(5):
        env.step(env.robots[0].action_space.sample())

    # Clear the sim
    og.sim.clear()


def test_dummy_task():
    task_tester("DummyTask")


def test_point_reaching_task():
    task_tester("PointReachingTask")


def test_point_navigation_task():
    task_tester("PointNavigationTask")


def test_behavior_task():
    task_tester("BehaviorTask")
