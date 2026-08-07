import pytest

from omnigibson.eval.evaluator import evaluate_instances_batched


def test_evaluate_instances_batched_tracks_slots_and_requires_one_instance_per_slot():
    loaded_batches = []
    active_slot_history = []

    def load_fn(slot_to_instance):
        loaded_batches.append(slot_to_instance)

    def step_fn(active_slots):
        active_slot_history.append(active_slots)
        if len(active_slot_history) == 1:
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
    assert active_slot_history == [[0, 1], [1]]
    assert results[101] == {"slot": 0, "instance": 101, "step": 1, "terminated": True, "truncated": False}
    assert results[202] == {"slot": 1, "instance": 202, "step": 2, "terminated": True, "truncated": False}

    with pytest.raises(ValueError, match="exactly one instance per environment slot"):
        evaluate_instances_batched(
            instances=[101],
            num_envs=2,
            load_fn=load_fn,
            step_fn=step_fn,
            record_fn=record_fn,
        )
