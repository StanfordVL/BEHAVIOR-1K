from types import SimpleNamespace

import pytest
import torch as th

from omnigibson.eval.evaluator import (
    BatchedEvaluator,
    InstanceEnvAccessor,
    InstanceEvaluationState,
    evaluate_instances_batched,
)
from omnigibson.metrics import TaskMetric


def test_evaluate_instances_batched_tracks_environments_and_requires_equal_batch_size():
    loaded_batches = []
    active_env_history = []

    def load_fn(env_idx_to_instance):
        loaded_batches.append(env_idx_to_instance)

    def step_fn(active_env_indices):
        active_env_history.append(active_env_indices)
        if len(active_env_history) == 1:
            return [True, False], [False, False]
        return [False, True], [False, False]

    def record_fn(**record):
        return record

    results = evaluate_instances_batched(
        instances=[101, 202],
        num_envs=2,
        load_fn=load_fn,
        step_fn=step_fn,
        record_fn=record_fn,
    )

    assert loaded_batches == [{0: 101, 1: 202}]
    assert active_env_history == [[0, 1], [1]]
    assert results[101] == {"env_idx": 0, "instance": 101, "step": 1, "terminated": True, "truncated": False}
    assert results[202] == {"env_idx": 1, "instance": 202, "step": 2, "terminated": True, "truncated": False}

    with pytest.raises(ValueError, match="exactly one instance per logical environment"):
        evaluate_instances_batched(
            instances=[101],
            num_envs=2,
            load_fn=load_fn,
            step_fn=step_fn,
            record_fn=record_fn,
        )


def test_instance_evaluation_state_owns_one_logical_environment():
    robots = [object(), object()]
    scenes = [SimpleNamespace(robots=[robot]) for robot in robots]
    task = SimpleNamespace(
        object_scope=[{"object": 0}, {"object": 1}],
        success=th.tensor([False, True]),
        get_goal_option_satisfaction=lambda env_idx: [[env_idx == 1]],
    )
    shared_env = SimpleNamespace(scenes=scenes, task=task)

    states = [
        InstanceEvaluationState(InstanceEnvAccessor(shared_env=shared_env, env_idx=env_idx)) for env_idx in range(2)
    ]
    states[0].instance_id = 101
    states[0].obs = {"value": 1}
    states[0].active = True

    assert states[0].env_accessor.scene is scenes[0]
    assert states[1].env_accessor.robot is robots[1]
    assert states[1].env_accessor.object_scope == {"object": 1}
    assert states[1].env_accessor.success
    assert states[1].env_accessor.get_goal_option_satisfaction() == [[True]]
    assert states[1].instance_id is None
    assert states[1].obs is None
    assert not states[1].active
    assert states[0].metrics is not states[1].metrics


def test_batched_evaluator_steps_shared_resources_once_and_freezes_finished_environments():
    class FakeMetric:
        def __init__(self):
            self.steps = []

        def step(self, **kwargs):
            self.steps.append(kwargs)

    class FakePolicy:
        def __init__(self):
            self.observations = []

        def forward(self, obs):
            self.observations.append(obs)
            return th.tensor([[1.0, 2.0], [3.0, 4.0]])

    class FakeEnv:
        def __init__(self):
            self.step_calls = []
            self.scenes = [
                SimpleNamespace(robots=[SimpleNamespace(action_dim=2)]),
                SimpleNamespace(robots=[SimpleNamespace(action_dim=2)]),
            ]
            self.task = SimpleNamespace(activity_name="not_a_light_task")

        def step(self, actions, n_render_iterations):
            self.step_calls.append((actions.clone(), n_render_iterations))
            obs = [{"value": th.tensor([10.0])}, {"value": th.tensor([20.0])}]
            return obs, None, th.tensor([False, False]), th.tensor([False, False]), [{}, {}]

    shared_env = FakeEnv()
    metrics = [FakeMetric(), FakeMetric()]
    evaluator = BatchedEvaluator.__new__(BatchedEvaluator)
    evaluator.num_envs = 2
    evaluator.env = shared_env
    evaluator.policy = FakePolicy()
    evaluator.instance_eval_states = [
        InstanceEvaluationState(
            env_accessor=InstanceEnvAccessor(shared_env=shared_env, env_idx=env_idx),
            metrics=[metrics[env_idx]],
            obs={"value": th.tensor([float(env_idx)])},
            active=env_idx == 0,
        )
        for env_idx in range(2)
    ]
    evaluator._preprocess_obs = lambda obs, instance_eval_state: obs

    evaluator._step_fn(active_env_indices=[0])

    assert len(evaluator.policy.observations) == 1
    assert evaluator.policy.observations[0]["value"].shape == (2, 1)
    assert len(shared_env.step_calls) == 1
    assert th.equal(shared_env.step_calls[0][0], th.tensor([[1.0, 2.0], [0.0, 0.0]]))
    assert shared_env.step_calls[0][1] == 1
    assert th.equal(evaluator.instance_eval_states[0].obs["value"], th.tensor([10.0]))
    assert th.equal(evaluator.instance_eval_states[1].obs["value"], th.tensor([1.0]))
    assert len(metrics[0].steps) == 1
    assert metrics[1].steps == []


def test_task_metrics_are_isolated_by_instance_environment_accessor(monkeypatch):
    scenes = [object(), object()]

    class FakeTask:
        def __init__(self):
            self.success = th.tensor([False, True])
            self.goal_options = [[[False, False]], [[False, False]]]

        def get_goal_option_satisfaction(self, env_idx):
            return self.goal_options[env_idx]

    task = FakeTask()
    shared_env = SimpleNamespace(scenes=scenes, task=task)
    monkeypatch.setattr("omnigibson.metrics.task_metric.og.sim", SimpleNamespace(get_rendering_dt=lambda: 0.1))
    metrics = [
        TaskMetric(
            {"length": 10},
            env_accessor=InstanceEnvAccessor(shared_env=shared_env, env_idx=env_idx),
        )
        for env_idx in range(2)
    ]

    for metric in metrics:
        metric.reset()
        metric.step(action=None, obs={}, reward=0.0, terminated=False, truncated=False, info={})
    task.goal_options = [[[True, False]], [[False, False]]]

    assert metrics[0].aggregate()["q_score"]["final"] == 0.5
    assert metrics[1].aggregate()["q_score"]["final"] == 1.0

    task.goal_options[0] = [[False, False]]
    legacy_metric = TaskMetric({"length": 10}, env_idx=0)
    legacy_metric.reset(shared_env)
    legacy_metric.step(shared_env, None, {}, 0.0, False, False, {})
    task.goal_options[0] = [[True, False]]
    assert legacy_metric.aggregate(shared_env)["q_score"]["final"] == 0.5
