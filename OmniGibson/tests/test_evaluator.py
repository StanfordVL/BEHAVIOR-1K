import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from omnigibson.eval.evaluator import (
    BatchedEvaluator,
    InstanceEnvAccessor,
    InstanceEvaluationState,
    evaluate_instances_batched,
)
from omnigibson.eval.utils.network_utils import PolicyConnectionError, WebsocketClientPolicy, packb
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


def test_evaluate_instances_batched_records_connection_failure_without_advancing_step():
    calls = []

    def step_fn(active_env_indices):
        calls.append(active_env_indices)
        if len(calls) == 1:
            return [True, False], [False, False]
        raise PolicyConnectionError("reconnect limit reached")

    results = evaluate_instances_batched(
        instances=[101, 202],
        num_envs=2,
        load_fn=lambda instances: None,
        step_fn=step_fn,
        record_fn=lambda **record: record,
    )

    assert calls == [[0, 1], [1]]
    assert results[101]["step"] == 1
    assert results[202]["step"] == 1
    assert results[202]["connection_failure"] == "reconnect limit reached"
    assert results[202]["truncated"]


def test_evaluate_instances_batched_records_connection_failure_during_load():
    def load_fn(instances):
        raise PolicyConnectionError("reconnect limit reached")

    results = evaluate_instances_batched(
        instances=[101],
        num_envs=1,
        load_fn=load_fn,
        step_fn=lambda active_env_indices: pytest.fail("simulation should not step"),
        record_fn=lambda **record: record,
    )

    assert results[101]["step"] == 0
    assert results[101]["connection_failure"] == "reconnect limit reached"


@pytest.mark.parametrize("disconnect", [ConnectionResetError("lost connection"), TimeoutError("response timed out")])
def test_websocket_policy_reconnects_with_the_same_observation(monkeypatch, disconnect):
    class FakeSocket:
        def __init__(self, response):
            self.response = response
            self.sent = []

        def send(self, data):
            self.sent.append(data)

        def recv(self, timeout=None):
            if isinstance(self.response, Exception):
                raise self.response
            return self.response

    first_socket = FakeSocket(disconnect)
    second_socket = FakeSocket(packb({"action": np.array([0.5], dtype=np.float32)}))
    policy = WebsocketClientPolicy(allow_reconnect=True)
    policy._ws = first_socket
    monkeypatch.setattr(policy, "_wait_for_server", lambda max_attempts=None: (second_socket, {}))

    action = policy.act({"observation": th.tensor([1.0])})

    assert th.equal(action, th.tensor([0.5]))
    assert first_socket.sent == second_socket.sent
    assert policy._reconnect_attempts == 1


def test_websocket_policy_fails_after_three_reconnects(monkeypatch):
    class BrokenSocket:
        def send(self, data):
            raise ConnectionResetError("lost connection")

    policy = WebsocketClientPolicy(allow_reconnect=True)
    policy._ws = BrokenSocket()
    reconnects = []

    def reconnect(max_attempts=None):
        reconnects.append(max_attempts)
        return BrokenSocket(), {}

    monkeypatch.setattr(policy, "_wait_for_server", reconnect)
    with pytest.raises(PolicyConnectionError, match="after 3 reconnect attempts"):
        policy.act({"observation": th.tensor([1.0])})

    assert reconnects == [1, 1, 1]


def test_websocket_policy_fails_rollout_when_server_stays_unavailable(monkeypatch):
    policy = WebsocketClientPolicy(allow_reconnect=True)
    attempts = []

    def unavailable(max_attempts=None):
        attempts.append(max_attempts)
        raise PolicyConnectionError("server unavailable")

    monkeypatch.setattr(policy, "_wait_for_server", unavailable)
    monkeypatch.setattr("omnigibson.eval.utils.network_utils.time.sleep", lambda seconds: None)

    with pytest.raises(PolicyConnectionError, match="after 3 reconnect attempts"):
        policy.reset()

    assert attempts == [1, 1, 1, 1]


def test_websocket_policy_reconnect_budget_spans_steps(monkeypatch):
    class ActionThenDisconnectSocket:
        def __init__(self):
            self.sent = False

        def send(self, data):
            if self.sent:
                raise ConnectionResetError("lost connection")
            self.sent = True

        def recv(self, timeout=None):
            return packb({"action": np.array([0.5], dtype=np.float32)})

    policy = WebsocketClientPolicy(allow_reconnect=True)
    policy._ws = ActionThenDisconnectSocket()
    monkeypatch.setattr(policy, "_wait_for_server", lambda max_attempts=None: (ActionThenDisconnectSocket(), {}))

    for _ in range(4):
        assert th.equal(policy.act({"observation": th.tensor([1.0])}), th.tensor([0.5]))
    with pytest.raises(PolicyConnectionError, match="after 3 reconnect attempts"):
        policy.act({"observation": th.tensor([1.0])})

    policy.reset()
    assert policy._reconnect_attempts == 1


def test_batched_evaluator_writes_failed_result_after_connection_loss(tmp_path, monkeypatch):
    evaluator = BatchedEvaluator.__new__(BatchedEvaluator)
    evaluator.cfg = SimpleNamespace(task=SimpleNamespace(name="turning_on_radio"))
    evaluator.num_envs = 1
    evaluator.n_trials = 0
    evaluator.n_success_trials = 0
    evaluator.instance_eval_states = [
        SimpleNamespace(active=True, env_accessor=SimpleNamespace(success=True), metrics=[], video_writer=None)
    ]
    evaluator.load_batch = lambda *args, **kwargs: None

    def step_fn(active_env_indices):
        raise PolicyConnectionError("reconnect limit reached")

    evaluator._step_fn = step_fn
    monkeypatch.setattr("omnigibson.eval.evaluator.og.sim", SimpleNamespace(get_rendering_dt=lambda: 1 / 30))

    result = evaluator.run([101], metrics_dir=str(tmp_path))[101]
    saved = json.loads((tmp_path / "turning_on_radio_101_0.json").read_text())

    assert result == saved
    assert result["steps"] == 0
    assert result["success"] is False
    assert result["failure_reason"] == "policy_connection_lost"
    assert result["q_score"]["final"] == 0.0
    assert result["time"]["normalized_time"] == pytest.approx(2 / 3)
    assert result["normalized_agent_distance"] == {"base": 0.0, "left": 0.0, "right": 0.0}
    assert evaluator.n_trials == 1


def test_instance_evaluation_state_owns_one_logical_environment():
    robots = [object(), object()]
    scenes = [SimpleNamespace(robots=[robot]) for robot in robots]
    task = SimpleNamespace(
        object_scopes=[{"object": 0}, {"object": 1}],
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
