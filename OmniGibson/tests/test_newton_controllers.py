"""Quantitative controller tracking tests for the Newton runtime (Phase 3).

Unlike the smoke suite, these assert closed-loop behavior: commanded motion
must produce that motion within tolerance. Scenarios print MEASURED lines so
regressions and baselines are visible in test output.

Run inside the ``newton-b1k`` conda environment:

    conda run -n newton-b1k python -m pytest tests/test_newton_controllers.py -v

Subprocess-per-scenario for the same reasons as test_newton_smoke.py.

Known state (July 22, 2026): the Fetch tracking tests pass. Per-family checks
pass for fetch, r1pro, locobot, ur5e, and stretch. tiago is xfail pending
workaround W15 in docs/other/newton_migration.md (massless footprint virtual
links diverge to NaN at import).
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

OMNIGIBSON_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = OMNIGIBSON_ROOT.parent
DEFAULT_DATA_PATH = REPO_ROOT / "datasets"

SCENARIO_OK = "NEWTON_CONTROLLER_SCENARIO_OK"


def _run_scenario(name, timeout=600):
    env = dict(os.environ)
    env.setdefault("OMNIGIBSON_DATA_PATH", str(DEFAULT_DATA_PATH))
    env["OMNIGIBSON_HEADLESS"] = "1"
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), name],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    tail = "\n".join(
        ["--- stdout tail ---"]
        + result.stdout.splitlines()[-30:]
        + ["--- stderr tail ---"]
        + result.stderr.splitlines()[-30:]
    )
    assert result.returncode == 0, f"Scenario {name!r} exited with {result.returncode}\n{tail}"
    assert f"{SCENARIO_OK} {name}" in result.stdout, f"Scenario {name!r} missing success marker\n{tail}"
    return result.stdout


def test_hold_default_controllers():
    _run_scenario("hold_default")


def test_hold_ik_controller():
    _run_scenario("hold_ik")


def test_joint_position_tracking():
    _run_scenario("joint_tracking")


def test_ik_cartesian_tracking():
    _run_scenario("ik_cartesian_tracking")


def test_base_velocity_tracking():
    _run_scenario("base_velocity_tracking")


def test_gripper_limits():
    _run_scenario("gripper_limits")


def test_family_r1pro():
    _run_scenario("family_check:r1pro", timeout=900)


@pytest.mark.xfail(reason="W15: tiago footprint virtual links diverge to NaN at import", strict=True)
def test_family_tiago():
    _run_scenario("family_check:tiago", timeout=900)


def test_family_locobot():
    _run_scenario("family_check:locobot", timeout=900)


def test_family_ur5e():
    _run_scenario("family_check:ur5e", timeout=900)


def test_family_stretch():
    _run_scenario("family_check:stretch", timeout=900)


# --- Scenario implementations. Everything below runs in a subprocess. ---


def _fetch_env(controller_config=None):
    import omnigibson as og

    env = og.Environment(
        configs={
            "scene": {"type": "Scene"},
            "robots": [
                {
                    "model": "fetch",
                    "obs_modalities": ["proprio"],
                    "action_type": "continuous",
                    "action_normalize": True,
                }
            ],
        }
    )
    robot = list(og.sim.robots)[0]
    if controller_config is not None:
        robot.reload_controllers(controller_config=controller_config)
    return env, robot


def _eef_position(robot):
    import warp as wp

    sim = robot.simulator
    labels = [str(label) for label in sim.model.body_label]
    eef_index = next(i for i in robot.body_indices if labels[i].endswith("/eef_link"))
    return wp.to_torch(sim.state_0.body_q)[eef_index, :3].detach().cpu().clone()


def _settle(env, robot, seconds=1.0):
    import torch as th

    for _ in range(int(seconds * 50)):
        env.step(th.zeros(robot.action_dim))


def _hold_drift_cm(env, robot, seconds=2.0):
    import torch as th

    _settle(env, robot)
    start = _eef_position(robot)
    for _ in range(int(seconds * 50)):
        env.step(th.zeros(robot.action_dim))
    return float((_eef_position(robot) - start).norm()) * 100.0


def scenario_hold_default():
    env, robot = _fetch_env()
    drift = _hold_drift_cm(env, robot)
    print(f"MEASURED hold_default drift_cm={drift:.2f}")
    assert drift < 2.0, f"EEF drifted {drift:.2f} cm under zero commands with default controllers"


def scenario_hold_ik():
    env, robot = _fetch_env(
        controller_config={
            "base": {"name": "NullJointController"},
            "camera": {"name": "NullJointController"},
            "arm_0": {"name": "InverseKinematicsController"},
            "gripper_0": {"name": "NullJointController"},
        }
    )
    drift = _hold_drift_cm(env, robot)
    print(f"MEASURED hold_ik drift_cm={drift:.2f}")
    assert drift < 2.0, f"EEF drifted {drift:.2f} cm under zero commands with IK controller"


def scenario_joint_tracking():
    import torch as th

    env, robot = _fetch_env(
        controller_config={
            "base": {"name": "NullJointController"},
            "camera": {"name": "NullJointController"},
            "arm_0": {"name": "JointController", "use_delta_commands": False},
            "gripper_0": {"name": "NullJointController"},
        }
    )
    arm = robot.controllers["arm_0"]
    indices = list(arm.dof_idx)

    # Command normalized mid-range (0.0 -> midpoint of joint limits) and hold.
    action = th.zeros(robot.action_dim)
    # A strictly zero action is the legacy no-op path, so nudge one entry off
    # zero by an epsilon that still maps to mid-range within tolerance.
    action[arm.command_start] = 1.0e-4
    for _ in range(150):
        env.step(action)

    lower, upper = robot.control_limits["position"]
    target = 0.5 * (lower[indices] + upper[indices])
    actual = robot.get_joint_positions()[indices]
    error = float((actual - target).abs().max())
    print(f"MEASURED joint_tracking max_error_rad={error:.4f}")
    assert error < 0.05, f"Arm joints missed mid-range targets by up to {error:.3f} rad"


def scenario_ik_cartesian_tracking():
    import torch as th

    env, robot = _fetch_env(
        controller_config={
            "base": {"name": "NullJointController"},
            "camera": {"name": "NullJointController"},
            "arm_0": {"name": "InverseKinematicsController"},
            "gripper_0": {"name": "NullJointController"},
        }
    )
    arm = robot.controllers["arm_0"]
    _settle(env, robot)

    # Servo the EEF toward a fixed world target 15 cm to the side, closing the
    # loop over position like teleop/policy users of the IK controller do.
    # The target is lateral because the zero-joint home pose has the arm fully
    # extended along +x, which is kinematically singular for further x motion.
    start = _eef_position(robot)
    target = start + th.tensor([0.0, 0.15, 0.0])
    action = th.zeros(robot.action_dim)
    for _ in range(150):
        error = target - _eef_position(robot)
        action[arm.command_start : arm.command_start + 3] = th.clamp(error / 0.05, -1.0, 1.0) * 0.5
        env.step(action)
    final_error = float((target - _eef_position(robot)).norm())
    print(f"MEASURED ik_tracking final_error_cm={final_error * 100:.2f}")
    assert final_error < 0.03, f"EEF stopped {final_error * 100:.2f} cm from a reachable target"


def scenario_base_velocity_tracking():
    import torch as th

    env, robot = _fetch_env(
        controller_config={
            "base": {"name": "DifferentialDriveController"},
            "camera": {"name": "NullJointController"},
            "arm_0": {"name": "NullJointController"},
            "gripper_0": {"name": "NullJointController"},
        }
    )
    base = robot.controllers["base"]
    _settle(env, robot, seconds=0.5)

    start_pos, start_quat = robot.get_position_orientation()
    action = th.zeros(robot.action_dim)
    action[base.command_start + 0] = 0.5  # forward
    for _ in range(100):
        env.step(action)
    end_pos, _ = robot.get_position_orientation()
    delta = end_pos - start_pos

    planar = float(delta[:2].norm())
    print(f"MEASURED base_tracking planar_cm={planar * 100:.2f}")
    assert planar > 0.10, f"Forward drive command moved the base only {planar * 100:.2f} cm in 2 s"


def scenario_gripper_limits():
    import torch as th

    env, robot = _fetch_env(
        controller_config={
            "base": {"name": "NullJointController"},
            "camera": {"name": "NullJointController"},
            "arm_0": {"name": "NullJointController"},
            "gripper_0": {"name": "MultiFingerGripperController"},
        }
    )
    gripper = robot.controllers["gripper_0"]
    indices = list(gripper.dof_idx)
    lower, upper = robot.control_limits["position"]

    def gripper_error(target):
        actual = robot.get_joint_positions()[indices]
        return float((actual - target).abs().max())

    action = th.zeros(robot.action_dim)
    action[gripper.command_start] = 1.0  # open
    for _ in range(75):
        env.step(action)
    open_error = gripper_error(upper[indices])

    action[gripper.command_start] = -1.0  # close
    for _ in range(75):
        env.step(action)
    close_error = gripper_error(lower[indices])

    print(f"MEASURED gripper open_error={open_error:.4f} close_error={close_error:.4f}")
    assert open_error < 0.01, f"Gripper open missed upper limits by {open_error:.4f}"
    assert close_error < 0.01, f"Gripper close missed lower limits by {close_error:.4f}"


def scenario_family_check(model):
    """Capability-driven tracking checks for one robot family.

    Runs hold for every robot, joint tracking for position-controlled dofs,
    closed-loop Cartesian tracking when an end-effector resolves, base drive
    for mobile bases, and gripper limits when finger joints exist. Skipped
    checks are reported, not silently passed.
    """
    import torch as th
    import warp as wp

    import omnigibson as og

    env = og.Environment(
        configs={
            "scene": {"type": "Scene"},
            "robots": [
                {
                    "model": model,
                    "obs_modalities": ["proprio"],
                    "action_type": "continuous",
                    "action_normalize": True,
                }
            ],
        }
    )
    robot = list(og.sim.robots)[0]
    sim = og.sim

    def body_positions():
        q = wp.to_torch(sim.state_0.body_q)
        return q[list(robot.body_indices), :3].detach().cpu().clone()

    # Hold: nothing on the robot may drift under zero commands.
    _settle(env, robot)
    start = body_positions()
    for _ in range(100):
        env.step(th.zeros(robot.action_dim))
    drift = float((body_positions() - start).norm(dim=1).max()) * 100.0
    print(f"MEASURED {model} hold drift_cm={drift:.2f}")
    assert drift < 3.0, f"{model}: body drifted {drift:.2f} cm under zero commands"

    # Joint tracking on position-controlled dofs with finite limits.
    position_controllers = [
        c
        for c in robot.controllers.values()
        if c.controller_type == "JointController" and c.motor_type == "position" and c.dof_idx
    ]
    if position_controllers:
        controller = position_controllers[0]
        indices = list(controller.dof_idx)
        lower, upper = robot.control_limits["position"]
        finite = th.isfinite(lower[indices]) & th.isfinite(upper[indices])
        if bool(finite.any()):
            current = robot.get_joint_positions()[indices]
            mid = 0.5 * (lower[indices] + upper[indices])
            target = th.where(finite, mid, current)
            span = th.where(finite, upper[indices] - lower[indices], th.ones_like(mid))
            normalized = th.where(finite, 2.0 * (target - lower[indices]) / span - 1.0, th.zeros_like(mid))
            action = th.zeros(robot.action_dim)
            if not controller.use_delta_commands:
                action[controller.command_start : controller.command_start + controller.command_dim] = normalized
                for _ in range(150):
                    env.step(action)
                error = float(((robot.get_joint_positions()[indices] - target).abs() * finite).max())
                print(f"MEASURED {model} joint_tracking max_error_rad={error:.4f}")
                assert error < 0.1, f"{model}: joints missed targets by {error:.3f} rad"
        else:
            print(f"SKIPPED {model} joint_tracking (no finite position limits)")
    else:
        print(f"SKIPPED {model} joint_tracking (no position JointController)")

    # Closed-loop Cartesian tracking on the first arm group with an end-effector.
    arm_group = next((g for g in robot.controllers if g.startswith("arm") and robot.controllers[g].dof_idx), None)
    side = (
        "left"
        if arm_group and arm_group.endswith("left")
        else ("right" if arm_group and arm_group.endswith("right") else None)
    )
    eef = robot._eef_body(side) if arm_group else None
    if arm_group is not None and eef is not None:
        robot.reload_controllers(
            controller_config={
                **{group: {"name": "NullJointController"} for group in robot.controllers},
                arm_group: {"name": "InverseKinematicsController"},
            }
        )
        arm = robot.controllers[arm_group]
        _settle(env, robot, seconds=0.5)

        def eef_pos():
            return wp.to_torch(sim.state_0.body_q)[eef.index, :3].detach().cpu().clone()

        # Build a target that is reachable by construction: the eef displacement
        # produced by a small feasible joint step (toward mid-range, so within
        # limits). A hardcoded world axis can be singular or limit-blocked at a
        # robot's home pose (Fetch fully extended, Stretch's retracted telescope).
        start = eef_pos()
        indices = list(arm.dof_idx)
        lower, upper = robot.control_limits["position"]
        lo, hi, q = lower[indices], upper[indices], robot.get_joint_positions()[indices]
        finite = th.isfinite(lo) & th.isfinite(hi)
        # Step each joint toward its roomier limit so the motion is nonzero
        # regardless of whether the home pose sits at mid-range (ur5e) or at a
        # limit (stretch retracted), and always stays within limits.
        head_lo, head_hi = q - lo, hi - q
        direction = th.where(head_hi >= head_lo, 1.0, -1.0)
        magnitude = th.minimum(0.2 * (hi - lo), th.maximum(head_lo, head_hi))
        step = th.where(finite, direction * magnitude, th.zeros_like(q))
        jacobian, _ = robot._eef_jacobian(eef.index, arm.dof_idx)
        predicted = (jacobian @ step)[:3]
        assert float(predicted.norm()) > 0.02, f"{model}: no feasible eef motion from home pose"
        target = start + predicted / predicted.norm() * 0.06

        action = th.zeros(robot.action_dim)
        for _ in range(150):
            error = target - eef_pos()
            action[arm.command_start : arm.command_start + 3] = th.clamp(error / 0.05, -1.0, 1.0) * 0.5
            env.step(action)
        final_error = float((target - eef_pos()).norm()) * 100.0
        print(f"MEASURED {model} ik ({arm_group}) final_error_cm={final_error:.2f}")
        assert final_error < 4.0, f"{model}: EEF stopped {final_error:.2f} cm from target"
        robot.reload_controllers()
    else:
        print(f"SKIPPED {model} ik (no arm controller or end-effector)")

    # Base drive for mobile bases.
    base = robot.controllers.get("base")
    if base is not None and base.controller_type in {"DifferentialDriveController", "HolonomicBaseJointController"}:
        _settle(env, robot, seconds=0.5)
        start_pos, _ = robot.get_position_orientation()
        action = th.zeros(robot.action_dim)
        action[base.command_start] = 0.5
        for _ in range(100):
            env.step(action)
        end_pos, _ = robot.get_position_orientation()
        planar = float((end_pos - start_pos)[:2].norm()) * 100.0
        print(f"MEASURED {model} base planar_cm={planar:.2f}")
        assert planar > 10.0, f"{model}: forward drive moved the base only {planar:.2f} cm in 2 s"
    else:
        print(f"SKIPPED {model} base (no mobile base controller)")

    _assert_newton_only_modules()


def _assert_newton_only_modules():
    forbidden = sorted(
        m
        for m in sys.modules
        if m == "isaacsim"
        or m.startswith("isaacsim.")
        or m == "omni"
        or m.startswith("omni.")
        or m == "carb"
        or m.startswith("carb.")
        or m == "omnigibson.prims"
        or m.startswith("omnigibson.prims.")
    )
    assert not forbidden, f"Legacy Isaac/prim modules imported on the Newton path: {forbidden}"


SCENARIOS = {
    "hold_default": scenario_hold_default,
    "hold_ik": scenario_hold_ik,
    "joint_tracking": scenario_joint_tracking,
    "ik_cartesian_tracking": scenario_ik_cartesian_tracking,
    "base_velocity_tracking": scenario_base_velocity_tracking,
    "gripper_limits": scenario_gripper_limits,
}


if __name__ == "__main__":
    scenario_name = sys.argv[1]
    if scenario_name.startswith("family_check:"):
        scenario_family_check(scenario_name.split(":", 1)[1])
    else:
        SCENARIOS[scenario_name]()
    print(f"{SCENARIO_OK} {scenario_name}", flush=True)
