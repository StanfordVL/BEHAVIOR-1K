"""
Pure (sim-free) orchestration logic for vectorized BEHAVIOR evaluation.

This module is deliberately free of any ``omnigibson`` / ``torch`` / Isaac imports so it can be
unit-tested without a GPU or a running simulator. It holds two pieces of *production* logic that
the vectorized ``Evaluator`` calls:

* :func:`evaluate_instances_batched` -- the batch/barrier driver loop. Instances are evaluated in
  batches of ``num_envs``; there is a barrier between batches (a slot that finishes early *freezes*
  until the whole batch is done, then the next batch is loaded). Streaming refill is intentionally
  NOT done here -- loading a new instance mid-run requires settling physics with global steps that
  would corrupt the other still-running envs.
* :func:`compute_q_score` -- the per-episode partial-success (Q-score) math, mirroring exactly the
  formula in ``omnigibson/metrics/task_metric.py``. It is fed by env-aware predicate evaluation in
  the real harness; here it operates on plain boolean masks so it is trivially testable.

End-of-project note: per the plan, these functions are folded back into the eval script once the
work is complete; they live here during development purely to stay GPU-free testable.
"""

from typing import Callable, Dict, List, Optional, Sequence

__all__ = ["compute_q_score", "evaluate_instances_batched"]


def compute_q_score(
    success: bool,
    now_satisfied_options: Sequence[Sequence[bool]],
    initial_satisfied_options: Sequence[Sequence[bool]],
) -> float:
    """
    Partial-success (Q-score) for a single episode/env.

    Mirrors ``TaskMetric._compute_episode_metrics`` exactly: a fully successful episode scores 1.0;
    otherwise the score is the fraction of goal predicates that were NOT satisfied at episode start
    but ARE satisfied now, maximized over the alternative goal-state options.

    Args:
        success: Whether the task's overall goal was met (``task.success[env_idx]``).
        now_satisfied_options: Per goal-state option, per predicate, whether it is satisfied NOW
            (at episode end), evaluated against THIS env's object scope.
        initial_satisfied_options: Same shape, evaluated at episode START.

    Returns:
        float in [0.0, 1.0].
    """
    if success:
        return 1.0
    if not now_satisfied_options:
        return 0.0
    option_scores = []
    for now_opt, init_opt in zip(now_satisfied_options, initial_satisfied_options):
        if len(now_opt) == 0:
            # An empty option contributes no progress; treat as 0 to avoid division by zero.
            option_scores.append(0.0)
            continue
        newly_satisfied = sum(int((not init) and now) for now, init in zip(now_opt, init_opt))
        option_scores.append(newly_satisfied / len(now_opt))
    return max(option_scores) if option_scores else 0.0


def evaluate_instances_batched(
    instances: Sequence,
    num_envs: int,
    load_fn: Callable[[Dict[int, object]], None],
    step_fn: Callable[[List[int]], "tuple[Sequence[bool], Sequence[bool]]"],
    record_fn: Callable[..., object],
    max_steps: Optional[int] = None,
) -> "Dict[object, object]":
    """
    Batch/barrier driver loop for vectorized evaluation (no refill).

    Instances are processed in batches of at most ``num_envs`` (each batch fills the N env slots
    with N instances of the same activity). For each batch:

      1. ``load_fn`` loads the batch's instances into env slots (called once, at the barrier, while
         nothing is running -- so its global physics settle is safe).
      2. The batch is stepped via ``step_fn`` until every slot in the batch has terminated/truncated.
         A slot that finishes early is *frozen*: it is dropped from the active-slot list passed to
         ``step_fn`` (so the caller can feed it a no-op action and stop accumulating its metrics),
         and ``record_fn`` is invoked exactly once for it.
      3. Only after the whole batch is done does the next batch load (the barrier).

    All sim interaction is delegated to the injected callables, so this function is pure and
    unit-testable.

    Args:
        instances: Ordered instance ids to evaluate.
        num_envs: Number of env slots (batch size).
        load_fn: ``load_fn(slot_to_instance)`` where ``slot_to_instance`` maps env slot index ->
            instance id for the current batch. Loads those instances into those slots.
        step_fn: ``step_fn(active_slots)`` advances the simulator one step. ``active_slots`` is the
            sorted list of slot indices still running this batch. Returns ``(terminated, truncated)``,
            each indexable by slot index (length >= max active slot + 1). Flags for inactive/frozen
            slots are ignored.
        record_fn: ``record_fn(slot=, instance=, step=, terminated=, truncated=)`` called once when a
            slot's episode ends. Its return value (if any) is collected and returned keyed by instance.
        max_steps: Optional hard cap on steps per batch; if reached, all still-active slots are recorded
            as truncated. ``None`` means rely on ``step_fn``'s own termination/truncation.

    Returns:
        Dict mapping instance id -> whatever ``record_fn`` returned for it.
    """
    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}")

    results: Dict[object, object] = {}
    pending = list(instances)

    while pending:
        batch = pending[:num_envs]
        pending = pending[num_envs:]

        slot_to_instance: Dict[int, object] = {slot: inst for slot, inst in enumerate(batch)}
        load_fn(dict(slot_to_instance))

        active = {slot: True for slot in slot_to_instance}
        step = 0
        while any(active.values()):
            active_slots = sorted(slot for slot, is_active in active.items() if is_active)
            terminated, truncated = step_fn(active_slots)
            step += 1

            hit_cap = max_steps is not None and step >= max_steps
            for slot in active_slots:
                term = bool(terminated[slot])
                trunc = bool(truncated[slot]) or hit_cap
                if term or trunc:
                    results[slot_to_instance[slot]] = record_fn(
                        slot=slot,
                        instance=slot_to_instance[slot],
                        step=step,
                        terminated=term,
                        truncated=trunc,
                    )
                    active[slot] = False

    return results
