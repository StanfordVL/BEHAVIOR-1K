from dataclasses import dataclass

import pytest

from omnigibson.envs.action_guard_wrapper import ActionGuardWrapper, ActionRejectedError


class FakeEnv:
    def __init__(self):
        self.steps = 0

    def step(self, action, n_render_iterations=1):
        self.steps += 1
        return action, n_render_iterations


@dataclass
class StructuralDecision:
    allowed: bool
    reason: str
    metadata: dict


def make_wrapper(guard):
    # Avoid launching Isaac Sim; the behavior under test only needs the wrapped
    # env and guard attributes initialized by ActionGuardWrapper.__init__.
    wrapper = object.__new__(ActionGuardWrapper)
    wrapper.env = FakeEnv()
    wrapper._guard = guard
    wrapper.last_decision = None
    return wrapper


def test_guarded_action_steps_environment():
    wrapper = make_wrapper(lambda env, action: True)

    result = wrapper.step("move", n_render_iterations=2)

    assert result == ("move", 2)
    assert wrapper.env.steps == 1
    assert wrapper.last_decision.allowed


def test_denial_does_not_step_environment_and_preserves_metadata():
    wrapper = make_wrapper(lambda env, action: StructuralDecision(False, "outside_workspace", {"zone": "restricted"}))

    with pytest.raises(ActionRejectedError) as exc_info:
        wrapper.step("move")

    assert wrapper.env.steps == 0
    assert exc_info.value.reason == "outside_workspace"
    assert exc_info.value.metadata == {"zone": "restricted"}


def test_guard_exception_fails_closed_without_leaking_exception_text():
    def broken_guard(env, action):
        raise RuntimeError("private backend detail")

    wrapper = make_wrapper(broken_guard)

    with pytest.raises(ActionRejectedError) as exc_info:
        wrapper.step("move")

    assert wrapper.env.steps == 0
    assert exc_info.value.reason == "action_guard_error"
    assert exc_info.value.metadata == {"error_type": "RuntimeError"}


def test_malformed_guard_result_fails_closed():
    wrapper = make_wrapper(lambda env, action: object())

    with pytest.raises(ActionRejectedError):
        wrapper.step("move")

    assert wrapper.env.steps == 0
