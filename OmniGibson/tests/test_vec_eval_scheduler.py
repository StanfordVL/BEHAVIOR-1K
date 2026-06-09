"""
Level-A unit tests for the pure (sim-free) vectorized-eval orchestration logic in
``omnigibson.eval.utils.vec_eval_scheduler``. These run without a GPU or a launched simulator.

They cover:
  * compute_q_score -- the partial-success math (mirrors TaskMetric).
  * evaluate_instances_batched -- batch/barrier scheduling, freeze-on-done, staggered finish order,
    and the max_steps cap.
"""

import pytest

from omnigibson.eval.utils.vec_eval_scheduler import compute_q_score, evaluate_instances_batched


# --------------------------------------------------------------------------------------
# compute_q_score
# --------------------------------------------------------------------------------------


def test_qscore_success_short_circuits_to_one():
    # Success => 1.0 regardless of predicate masks.
    assert compute_q_score(True, [[False, False]], [[False, False]]) == 1.0


def test_qscore_partial_newly_satisfied_fraction():
    # One option, 3 predicates. Initial: [F, F, T]; now: [T, F, T].
    # Newly satisfied = pred0 only (pred2 was already satisfied at start => no credit). => 1/3.
    score = compute_q_score(False, [[True, False, True]], [[False, False, True]])
    assert score == pytest.approx(1.0 / 3.0)


def test_qscore_takes_max_over_options():
    # Option A: 1/2 newly satisfied; Option B: 2/2 newly satisfied => max = 1.0... but success is False,
    # so verify it picks the better option's fraction (1.0 here) without the success short-circuit path.
    now = [[True, False], [True, True]]
    init = [[False, False], [False, False]]
    assert compute_q_score(False, now, init) == pytest.approx(1.0)


def test_qscore_no_options_is_zero():
    assert compute_q_score(False, [], []) == 0.0


def test_qscore_empty_option_does_not_divide_by_zero():
    # An empty option contributes 0; a sibling option with progress still scores.
    now = [[], [True]]
    init = [[], [False]]
    assert compute_q_score(False, now, init) == pytest.approx(1.0)


def test_qscore_no_progress_is_zero():
    now = [[False, False]]
    init = [[False, False]]
    assert compute_q_score(False, now, init) == 0.0


# --------------------------------------------------------------------------------------
# evaluate_instances_batched -- fake sim harness
# --------------------------------------------------------------------------------------


class FakeSim:
    """
    Test double for the injected sim callables. Each instance finishes (terminates) after a fixed
    number of steps. Records load order, per-step active-slot lists, and per-instance end records.
    """

    def __init__(self, durations, truncate_instances=None):
        # durations: dict instance_id -> #steps until that instance terminates
        self.durations = durations
        # instances that should report `truncated` instead of `terminated` when they hit their duration
        self.truncate_instances = set(truncate_instances or ())
        self.slot_instance = {}
        self.slot_steps = {}
        self.load_calls = []  # list of slot_to_instance dicts, in load order
        self.active_history = []  # list of active_slots lists, one per step_fn call

    def load(self, slot_to_instance):
        self.load_calls.append(dict(slot_to_instance))
        self.slot_instance = dict(slot_to_instance)
        self.slot_steps = {slot: 0 for slot in slot_to_instance}

    def step(self, active_slots):
        self.active_history.append(list(active_slots))
        n = max(self.slot_instance) + 1
        terminated = [False] * n
        truncated = [False] * n
        for slot in active_slots:
            self.slot_steps[slot] += 1
            inst = self.slot_instance[slot]
            if self.slot_steps[slot] >= self.durations[inst]:
                if inst in self.truncate_instances:
                    truncated[slot] = True
                else:
                    terminated[slot] = True
        return terminated, truncated

    @staticmethod
    def record(slot, instance, step, terminated, truncated):
        return {"slot": slot, "instance": instance, "step": step, "terminated": terminated, "truncated": truncated}


def test_scheduler_batches_barrier_and_count():
    # 3 instances, N=2 => batch1=[10,11] (slots 0,1), batch2=[12] (slot 0).
    sim = FakeSim(durations={10: 2, 11: 3, 12: 1})
    results = evaluate_instances_batched(
        [10, 11, 12], num_envs=2, load_fn=sim.load, step_fn=sim.step, record_fn=sim.record
    )

    # Each instance recorded exactly once, keyed correctly.
    assert set(results.keys()) == {10, 11, 12}
    assert results[10]["step"] == 2 and results[10]["instance"] == 10
    assert results[11]["step"] == 3 and results[11]["instance"] == 11
    assert results[12]["step"] == 1 and results[12]["instance"] == 12

    # Two batches loaded, in order; second batch only has instance 12 (barrier => loaded after batch1 done).
    assert sim.load_calls == [{0: 10, 1: 11}, {0: 12}]

    # Freeze + barrier: batch1 runs 3 steps (slot 0 frozen after step 2), then batch2 runs 1 step.
    assert sim.active_history == [[0, 1], [0, 1], [1], [0]]


def test_scheduler_freeze_one_slot_idles_until_barrier():
    # Single batch [A=2 steps, B=5 steps]; slot 0 must freeze after step 2 while slot 1 runs to step 5.
    sim = FakeSim(durations={"A": 2, "B": 5})
    results = evaluate_instances_batched(
        ["A", "B"], num_envs=2, load_fn=sim.load, step_fn=sim.step, record_fn=sim.record
    )

    assert results["A"]["step"] == 2
    assert results["B"]["step"] == 5
    # slot 0 dropped from active list from step 3 onward.
    assert sim.active_history == [[0, 1], [0, 1], [1], [1], [1]]
    assert len(sim.load_calls) == 1


def test_scheduler_staggered_finish_order_no_cross_slot_bleed():
    # Finish order (101 first) differs from slot order (100 in slot 0). Verify per-slot keying is correct.
    sim = FakeSim(durations={100: 5, 101: 1})
    results = evaluate_instances_batched(
        [100, 101], num_envs=2, load_fn=sim.load, step_fn=sim.step, record_fn=sim.record
    )

    assert results[101]["step"] == 1 and results[101]["slot"] == 1
    assert results[100]["step"] == 5 and results[100]["slot"] == 0
    # slot 1 frozen after its early finish; slot 0 keeps running.
    assert sim.active_history == [[0, 1], [0], [0], [0], [0]]


def test_scheduler_max_steps_cap_truncates():
    # Instance would run 100 steps but the cap truncates it at step 3.
    sim = FakeSim(durations={200: 100})
    results = evaluate_instances_batched(
        [200], num_envs=1, load_fn=sim.load, step_fn=sim.step, record_fn=sim.record, max_steps=3
    )

    assert results[200]["step"] == 3
    assert results[200]["truncated"] is True
    assert results[200]["terminated"] is False


def test_scheduler_truncated_flag_propagates():
    sim = FakeSim(durations={"T": 2}, truncate_instances={"T"})
    results = evaluate_instances_batched(["T"], num_envs=1, load_fn=sim.load, step_fn=sim.step, record_fn=sim.record)
    assert results["T"]["truncated"] is True and results["T"]["terminated"] is False


def test_scheduler_rejects_bad_num_envs():
    with pytest.raises(ValueError):
        evaluate_instances_batched(
            [1], num_envs=0, load_fn=lambda x: None, step_fn=lambda a: ([], []), record_fn=lambda **k: None
        )


# --------------------------------------------------------------------------------------
# edge cases
# --------------------------------------------------------------------------------------


def test_scheduler_num_envs_exceeds_instances():
    # Fewer instances than slots: a single short batch using only slot 0; extra slots stay unused.
    sim = FakeSim(durations={10: 2})
    results = evaluate_instances_batched([10], num_envs=3, load_fn=sim.load, step_fn=sim.step, record_fn=sim.record)
    assert set(results.keys()) == {10}
    assert results[10]["step"] == 2 and results[10]["slot"] == 0
    assert sim.load_calls == [{0: 10}]  # only slot 0 loaded
    assert sim.active_history == [[0], [0]]  # never references slots 1/2


def test_scheduler_terminated_and_truncated_same_step():
    # A slot may report terminated AND truncated on the same step; it must be recorded exactly once.
    record_calls = []

    def step(active_slots):
        n = max(active_slots) + 1
        return [True] * n, [True] * n  # both flags true for every active slot

    def record(slot, instance, step, terminated, truncated):
        record_calls.append(instance)
        return {"terminated": terminated, "truncated": truncated}

    results = evaluate_instances_batched(["X"], num_envs=1, load_fn=lambda s2i: None, step_fn=step, record_fn=record)
    assert record_calls == ["X"]  # recorded exactly once, not twice
    assert results["X"]["terminated"] is True and results["X"]["truncated"] is True


def test_scheduler_empty_instances_returns_empty():
    sim = FakeSim(durations={})
    results = evaluate_instances_batched([], num_envs=2, load_fn=sim.load, step_fn=sim.step, record_fn=sim.record)
    assert results == {}
    assert sim.load_calls == []  # nothing loaded, no steps
