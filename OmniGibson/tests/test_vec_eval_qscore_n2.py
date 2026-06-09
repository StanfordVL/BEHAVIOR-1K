"""
Level-B (N=2, GPU) test for the env-aware per-option goal satisfaction used by the vectorized
Q-score metric: ``BehaviorTask.get_goal_option_satisfaction(env_idx)``.

The bug this guards against: the legacy ``TaskMetric`` computes partial credit via
``ground_goal_state_options[*].evaluate()``, which binds the single shared ``compiled_task`` scope
and therefore reports env 0's result for every env. The fix routes evaluation through
``_evaluate_predicate(env_idx, ...)`` so each env reports its own state.

Key assertion: perturbing env 0's objects must NOT change env 1's reported satisfaction (isolation).
"""

import copy

import torch as th

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.eval.utils.vec_eval_scheduler import compute_q_score
from omnigibson.metrics import AgentMetric, TaskMetric
from omnigibson.utils.bddl_utils import is_system_bddl_inst

NUM_ENVS = 2
ACTIVITY_NAME = "picking_up_trash"
SCENE_MODEL = "house_double_floor_lower"
FAR_AWAY = th.tensor([200.0, 200.0, 200.0], dtype=th.float32)


def _setup_env(num_envs=NUM_ENVS):
    if og.sim is None:
        gm.RENDER_VIEWER_CAMERA = False
        gm.ENABLE_OBJECT_STATES = True
        gm.USE_GPU_DYNAMICS = False
        gm.ENABLE_FLATCACHE = False
        gm.ENABLE_TRANSITION_RULES = False
    else:
        og.sim.stop()
    cfg = {
        "env": {"num_envs": num_envs},
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": SCENE_MODEL,
            "load_room_types": ["living_room", "kitchen"],
        },
        "robots": [{"model": "r1pro", "obs_modalities": ["proprio"]}],
        "task": {
            "type": "BehaviorTask",
            "activity_name": ACTIVITY_NAME,
            "activity_definition_id": 0,
            "activity_instance_id": 0,
            "online_object_sampling": False,
            "use_presampled_robot_pose": True,
            "termination_config": {"max_steps": 500},
            "reward_config": {"r_potential": 1.0},
        },
    }
    return og.Environment(configs=cfg)


def test_goal_option_satisfaction_is_env_aware_and_isolated():
    env = _setup_env()
    try:
        env.reset()
        task = env.task

        masks0 = task.get_goal_option_satisfaction(0)
        masks1 = task.get_goal_option_satisfaction(1)

        # Shape sanity: one entry per grounded goal-state option, with the option's predicate count.
        assert len(masks0) == len(task.ground_goal_state_options)
        for option_mask, option in zip(masks0, task.ground_goal_state_options):
            assert len(option_mask) == len(option)
            assert all(isinstance(b, bool) for b in option_mask)

        # Both envs loaded the same instance => identical satisfaction. (If this fails, evaluation is
        # not actually reading per-env scope.)
        assert masks0 == masks1, "Freshly-loaded identical instances should report identical goal satisfaction"

        # compute_q_score composes with these masks (no success, no progress vs itself => 0.0).
        assert compute_q_score(False, masks0, masks0) == 0.0

        # --- Isolation: perturb env 0 only, env 1 must be unaffected ---
        masks1_before = copy.deepcopy(masks1)
        moved_any = False
        for inst, entity in task.object_scope[0].items():
            if entity is not None and not is_system_bddl_inst(inst) and inst != "agent.n.01_1":
                entity.set_position_orientation(position=FAR_AWAY)
                moved_any = True
        assert moved_any, "Expected at least one task-relevant object in env 0 to perturb"
        for _ in range(5):
            og.sim.step()

        masks1_after = task.get_goal_option_satisfaction(1)
        assert (
            masks1_after == masks1_before
        ), "env 1 goal satisfaction changed after perturbing only env 0 -> scope is leaking across envs"

        # --- per-env metrics compute independently and emit the score_utils schema ---
        env.reset()  # restore (we perturbed env 0 above)
        human = {
            "length": 100,
            "distance_traveled": 1.0,
            "left_eef_displacement": 1.0,
            "right_eef_displacement": 1.0,
        }
        task_metrics = [TaskMetric(human, env_idx=i) for i in range(NUM_ENVS)]
        agent_metrics = [AgentMetric(human, env_idx=i) for i in range(NUM_ENVS)]
        for m in task_metrics + agent_metrics:
            m.reset(env)

        action = th.zeros((NUM_ENVS, env.scenes[0].robots[0].action_dim), dtype=th.float32)
        obs, _, term, trunc, info = env.step(action)
        for i in range(NUM_ENVS):
            task_metrics[i].step(env, action[i], obs[i], 0.0, bool(term[i]), bool(trunc[i]), info[i])
            agent_metrics[i].step(env, action[i], obs[i], 0.0, bool(term[i]), bool(trunc[i]), info[i])

        for i in range(NUM_ENVS):
            res = {**task_metrics[i].aggregate(env), **agent_metrics[i].aggregate(env)}
            # Schema consumed by learning/utils/score_utils.compute_final_q_score:
            assert 0.0 <= res["q_score"]["final"] <= 1.0
            assert "normalized_time" in res["time"]
            assert {"base", "left", "right"}.issubset(res["normalized_agent_distance"].keys())
    finally:
        og.clear()
