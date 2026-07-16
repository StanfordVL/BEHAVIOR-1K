"""Automated smoke tests for the Newton runtime (migration Phase 1).

Run inside the ``newton-b1k`` conda environment (see
docs/other/newton_migration.md, Environment Setup):

    conda run -n newton-b1k python -m pytest tests/test_newton_smoke.py -v

Each scenario runs in a fresh subprocess because ``og.shutdown()`` exits the
process, the Newton model is immutable after finalization, and scenarios must
not share native USD/mesh state (workaround W4 in the migration record). The
pytest process itself never imports omnigibson, so these tests also run under
the legacy Isaac conftest without launching a simulator.
"""

import os
import subprocess
import sys
from pathlib import Path

OMNIGIBSON_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = OMNIGIBSON_ROOT.parent
DEFAULT_DATA_PATH = REPO_ROOT / "datasets"

SCENARIO_OK = "NEWTON_SMOKE_SCENARIO_OK"

POSITION_BOUND = 50.0
LINEAR_VELOCITY_BOUND = 100.0


def _run_scenario(name, timeout):
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


def test_import_without_isaac():
    _run_scenario("import_without_isaac", timeout=300)


def test_empty_scene_object_registry_and_stepping():
    _run_scenario("empty_scene_object", timeout=600)


def test_robot_controller_commands():
    _run_scenario("robot_controller_commands", timeout=600)


def test_repeated_build_close():
    _run_scenario("repeated_build_close", timeout=600)


def test_state_reset_determinism():
    _run_scenario("state_reset_determinism", timeout=600)


def test_gym_api_conformance():
    _run_scenario("gym_api_conformance", timeout=600)


def test_rs_int_stability_with_disturbance():
    _run_scenario("rs_int_stability", timeout=1200)


# --- Scenario implementations. Everything below runs in a subprocess. ---


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


def _assert_body_states_finite_and_bounded(sim):
    import numpy as np

    q = sim.state_0.body_q.numpy()
    qd = sim.state_0.body_qd.numpy()
    assert np.isfinite(q).all(), "Non-finite body pose detected"
    assert np.isfinite(qd).all(), "Non-finite body velocity detected"
    max_pos = float(np.abs(q[:, :3]).max()) if len(q) else 0.0
    max_lin = float(np.linalg.norm(qd[:, 3:6], axis=1).max()) if len(qd) else 0.0
    assert max_pos < POSITION_BOUND, f"Body escaped scene bounds: max|pos|={max_pos:.2f}"
    assert max_lin < LINEAR_VELOCITY_BOUND, f"Body velocity exploded: max|v|={max_lin:.2f}"


def _step(env, sim, steps, check_every=25):
    for i in range(1, steps + 1):
        env.step([])
        if i % check_every == 0 or i == steps:
            _assert_body_states_finite_and_bounded(sim)


def _bowl_cfg():
    return {
        "type": "DatasetObject",
        "name": "bowl",
        "category": "bowl",
        "model": "ajzltc",
        "position": [0.0, 0.0, 0.1],
    }


def scenario_import_without_isaac():
    import omnigibson as og

    assert og.sim is None
    assert og.simulator.BACKEND == "newton"
    _assert_newton_only_modules()


def scenario_empty_scene_object():
    import torch as th

    import omnigibson as og

    env = og.Environment(configs={"scene": {"type": "Scene"}, "objects": [_bowl_cfg()]})
    sim = og.sim
    assert sim is not None

    summary = sim.summary()
    assert summary["body_count"] > 0
    assert summary["joint_count"] > 0
    assert summary["shape_count"] > 0

    assert len(sim.objects) == 1
    assert len(sim.robots) == 0
    entity = sim.entity_registry.get("bowl")
    assert entity is not None
    assert entity is sim.objects[0]

    pos, quat = entity.get_position_orientation()
    assert th.isfinite(pos).all() and th.isfinite(quat).all()

    _step(env, sim, 50)

    # Pose update APIs: lift the bowl, freeze it, then let it settle again.
    pos, _ = entity.get_position_orientation()
    entity.set_position_orientation(position=pos + th.tensor([0.0, 0.0, 0.2]))
    entity.keep_still()
    _step(env, sim, 25)
    pos_after, _ = entity.get_position_orientation()
    assert pos_after[2] < 5.0, "Bowl left the workspace after pose update"

    _assert_newton_only_modules()


def scenario_robot_controller_commands():
    import torch as th

    import omnigibson as og

    env = og.Environment(
        configs={
            "scene": {"type": "Scene"},
            "robots": [
                {
                    "model": "fetch",
                    "obs_modalities": ["rgb"],
                    "action_type": "continuous",
                    "action_normalize": True,
                }
            ],
        }
    )
    sim = og.sim
    robots = list(sim.robots)
    assert len(robots) == 1
    robot = robots[0]

    dim = robot.action_dim
    assert dim > 0
    assert robot.n_dof > 0
    assert th.isfinite(robot.get_joint_positions()).all()

    # Wrong-sized commands must be rejected by the controller surface.
    try:
        robot.apply_action(th.zeros(dim + 1))
    except ValueError:
        pass
    else:
        raise AssertionError("apply_action accepted a wrong-sized action")

    generator = th.Generator().manual_seed(0)
    for _ in range(100):
        action = (th.rand(dim, generator=generator) * 2.0 - 1.0) * 0.2
        env.step(action)
        _assert_body_states_finite_and_bounded(sim)

    assert th.isfinite(robot.get_joint_positions()).all()
    assert th.isfinite(robot.get_joint_velocities()).all()
    _assert_newton_only_modules()


def scenario_repeated_build_close():
    import omnigibson as og

    cfg = {"scene": {"type": "Scene"}, "objects": [_bowl_cfg()]}
    previous_sim = None
    for _ in range(2):
        env = og.Environment(configs=cfg)
        sim = og.sim
        assert sim is not None and sim is not previous_sim
        _step(env, sim, 10)
        env.close()
        previous_sim = sim


def scenario_state_reset_determinism():
    import torch as th

    import omnigibson as og

    env = og.Environment(
        configs={
            "scene": {"type": "Scene"},
            "objects": [_bowl_cfg()],
            "robots": [
                {
                    "model": "fetch",
                    "obs_modalities": ["rgb"],
                    "action_type": "continuous",
                    "action_normalize": True,
                }
            ],
        }
    )
    sim = og.sim
    robot = list(sim.robots)[0]
    bowl = sim.entity_registry.get("bowl")
    dim = robot.action_dim

    def rollout(steps=40):
        generator = th.Generator().manual_seed(7)
        trajectory = []
        for _ in range(steps):
            env.step((th.rand(dim, generator=generator) * 2.0 - 1.0) * 0.2)
            trajectory.append(th.cat([robot.get_joint_positions(), bowl.get_position_orientation()[0]]))
        return th.stack(trajectory)

    snapshot = sim.dump_state()
    q_initial = robot.get_joint_positions()

    trajectory_a = rollout()

    # Restoring the snapshot must return joint state exactly.
    sim.load_state(snapshot)
    assert th.allclose(robot.get_joint_positions(), q_initial, atol=1e-6), "load_state did not restore joint state"

    # An identical action sequence from the restored state must reproduce the
    # trajectory. MuJoCo-Warp accumulates constraint forces with GPU atomics,
    # so replay matches to solver noise (~2e-5 observed over 40 steps), not
    # bitwise; the tolerance is set to catch genuine state leakage instead.
    trajectory_b = rollout()
    deviation = float((trajectory_a - trajectory_b).abs().max())
    assert th.allclose(trajectory_a, trajectory_b, atol=1e-3), f"Post-restore trajectory deviates by {deviation}"

    # Serialized roundtrip restores the state it was dumped from.
    flat = sim.dump_state(serialized=True)
    assert isinstance(flat, th.Tensor) and flat.dim() == 1
    q_before = robot.get_joint_positions()
    sim.load_state(flat, serialized=True)
    assert th.allclose(robot.get_joint_positions(), q_before, atol=1e-6), "Serialized roundtrip changed joint state"

    # env.reset() returns to the freshly built environment state, and seeding
    # is accepted.
    env.reset(seed=0)
    assert th.allclose(robot.get_joint_positions(), q_initial, atol=1e-6), "env.reset did not restore initial state"

    _assert_newton_only_modules()


def scenario_gym_api_conformance():
    import gymnasium as gym
    import torch as th

    import omnigibson as og

    env = og.Environment(
        configs={
            "scene": {"type": "Scene"},
            "objects": [_bowl_cfg()],
            "robots": [
                {
                    "model": "fetch",
                    # rgb is not supported on the Newton path yet and must be
                    # skipped with a warning rather than crash.
                    "obs_modalities": ["rgb", "proprio"],
                    "action_type": "continuous",
                    "action_normalize": True,
                }
            ],
        }
    )
    robot = list(og.sim.robots)[0]

    # Action space mirrors the legacy shape: Dict of per-robot Boxes.
    assert isinstance(env.action_space, gym.spaces.Dict)
    robot_action_space = env.action_space.spaces[robot.name]
    assert isinstance(robot_action_space, gym.spaces.Box)
    assert robot_action_space.shape == (robot.action_dim,)
    assert env.action_space.contains(env.action_space.sample())

    # Observation space contains proprio only; rgb is skipped.
    assert isinstance(env.observation_space, gym.spaces.Dict)
    robot_obs_space = env.observation_space.spaces[robot.name]
    assert set(robot_obs_space.spaces) == {"proprio"}

    def check_obs(obs):
        compat = {name: {m: v.numpy() for m, v in modalities.items()} for name, modalities in obs.items()}
        assert env.observation_space.contains(compat), "Observation does not match observation space"

    obs, info = env.reset(seed=0)
    check_obs(obs)
    assert robot.name in info

    # Flat-tensor action path; obs stay torch tensors at the API boundary.
    action = th.from_numpy(env.action_space.sample()[robot.name])
    obs, reward, terminated, truncated, _ = env.step(action)
    check_obs(obs)
    assert isinstance(obs[robot.name]["proprio"], th.Tensor)
    assert reward == 0.0 and terminated is False and truncated is False

    # Dict action path keyed by robot name (legacy contract).
    obs, _, _, _, _ = env.step({robot.name: action})
    check_obs(obs)

    _assert_newton_only_modules()


def scenario_rs_int_stability():
    import torch as th

    import omnigibson as og

    env = og.Environment(configs={"scene": {"type": "InteractiveTraversableScene", "scene_model": "Rs_int"}})
    sim = og.sim
    objects = list(sim.objects)
    assert len(objects) > 10, f"Rs_int loaded only {len(objects)} objects"
    assert sim.summary()["body_count"] > 50

    # Settle, then disturb: drop a chair from 20 cm and open an articulated
    # cabinet joint to mid-range (mirrors the disturbance validation in
    # docs/other/newton_migration.md).
    _step(env, sim, 100)

    chair = next((e for e in objects if e.name.startswith("straight_chair")), None)
    assert chair is not None, "Rs_int is expected to contain a straight_chair instance"
    pos, _ = chair.get_position_orientation()
    chair.set_position_orientation(position=pos + th.tensor([0.0, 0.0, 0.2]))

    cabinet = next((e for e in objects if e.name.startswith("bottom_cabinet") and e.n_dof > 0), None)
    assert cabinet is not None, "Rs_int is expected to contain an articulated bottom_cabinet instance"
    cabinet.set_joint_positions(th.full((cabinet.n_dof,), 0.5), normalized=True)

    _step(env, sim, 200)
    _assert_newton_only_modules()


SCENARIOS = {
    "import_without_isaac": scenario_import_without_isaac,
    "empty_scene_object": scenario_empty_scene_object,
    "robot_controller_commands": scenario_robot_controller_commands,
    "repeated_build_close": scenario_repeated_build_close,
    "state_reset_determinism": scenario_state_reset_determinism,
    "gym_api_conformance": scenario_gym_api_conformance,
    "rs_int_stability": scenario_rs_int_stability,
}


if __name__ == "__main__":
    scenario_name = sys.argv[1]
    SCENARIOS[scenario_name]()
    print(f"{SCENARIO_OK} {scenario_name}", flush=True)
