"""Multi-env (num_envs=2) transition-rule tests.

Regression coverage for the multi-scene transition-rule bug where recipe rules
(e.g. CookingPhysicalParticleRule) effectively fired only in the canonical scene
(env 0): macro physical particle systems dumped/loaded their particle poses in
the WORLD frame, so any state authored in the canonical scene (dataset task
instances, scene-file init states) loaded cloned scenes' particles at env 0's
world coordinates -- outside the cloned scene's own containers -- and the
rules' Contains gate could then never pass outside env 0.

Fixed by storing particle poses scene-relative in BaseSystem._dump_state /
_load_state (mirroring PhysxParticleInstancer, which was already scene-aware).

Run standalone (one Environment per process):
  OMNIGIBSON_HEADLESS=1 pytest tests/test_multiple_envs_transition_rules.py -v -s
"""

import pytest
import torch as th

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.object_states import Contains, Heated

N_ENVS = 2


@pytest.fixture(scope="module")
def multi_env():
    assert og.sim is None, "This module must run in a fresh process (one Environment per process)."
    gm.RENDER_VIEWER_CAMERA = False
    gm.ENABLE_OBJECT_STATES = True
    gm.USE_GPU_DYNAMICS = True
    gm.ENABLE_FLATCACHE = False
    gm.ENABLE_TRANSITION_RULES = True

    cfg = {
        "env": {"num_envs": N_ENVS},
        "scene": {"type": "Scene"},
        "robots": [{"type": "Fetch", "obs_modalities": ["rgb"], "position": [20.0, 20.0, 0.1]}],
        "objects": [
            {
                "type": "DatasetObject",
                "name": "stockpot",
                "category": "stockpot",
                "model": "dcleem",
                "abilities": {"fillable": {}, "heatable": {}},
                "position": [0.0, 0.0, 0.15],
            },
        ],
        "task": {"type": "DummyTask"},
    }
    env = og.Environment(configs=cfg)
    for _ in range(10):
        og.sim.step()
    yield env
    og.clear()


def _pots(env):
    return [env.scenes[s].object_registry("name", "stockpot") for s in range(N_ENVS)]


def test_macro_physical_particle_state_is_scene_relative(multi_env):
    """Macro physical particle poses (e.g. popcorn, the system from the make_microwave_popcorn
    eval task) must be dumped scene-relative and loaded back through the OWNING scene's pose,
    so a state authored in the canonical scene places a cloned scene's particles inside the
    cloned scene's own geometry."""
    env = multi_env
    pots = _pots(env)
    popcorn = [env.scenes[s].get_system("popcorn") for s in range(N_ENVS)]
    og.sim.step()

    for s in range(N_ENVS):
        popcorn[s].generate_particles(positions=[(pots[s].aabb_center + th.tensor([0.0, 0.03, 0.05])).tolist()])
    og.sim.step()
    for s in range(N_ENVS):
        assert pots[s].states[Contains].get_value(popcorn[s]), f"setup failed: popcorn not contained in scene {s}"

    states = [popcorn[s].dump_state(serialized=False) for s in range(N_ENVS)]

    # 1) Dumped positions are scene-relative: no scene world-offset baked into the state.
    identity_quat = th.tensor([0.0, 0.0, 0.0, 1.0])
    for s in range(N_ENVS):
        world_pos = popcorn[s].get_particles_position_orientation()[0][0]
        rel_expected = env.scenes[s].convert_world_pose_to_scene_relative(world_pos, identity_quat)[0]
        delta = th.norm(states[s]["positions"][0] - rel_expected)
        assert delta < 1e-4, (
            f"scene {s}: dumped particle position {states[s]['positions'][0].tolist()} is not scene-relative "
            f"(expected {rel_expected.tolist()})"
        )

    # 2) Cross-scene load: a canonical-scene-authored state (env 0's dump) loaded into env 1's
    # system must land inside env 1's own container, not env 0's.
    popcorn[1].load_state(states[0], serialized=False)
    og.sim.step()
    pos_after = popcorn[1].get_particles_position_orientation()[0][0]
    lo, hi = pots[1].aabb
    assert th.all(pos_after > lo - 0.2) and th.all(pos_after < hi + 0.2), (
        f"canonical-authored particle state loaded into scene 1 landed at {pos_after.tolist()}, "
        f"outside scene 1's stockpot aabb {lo.tolist()}..{hi.tolist()} (world-frame leak into scene 0)"
    )
    assert pots[1].states[Contains].get_value(popcorn[1]), "scene 1's pot does not contain the cross-loaded particle"

    # Clean up so the (heatable) pots don't cook popcorn during the next test
    for s in range(N_ENVS):
        popcorn[s].remove_all_particles()
    og.sim.step()


def test_cooking_physical_particle_rule_fires_per_scene_with_isolation(multi_env):
    """CookingPhysicalParticleRule must fire in whichever scene meets its conditions --
    exclusively in a non-canonical scene when only that scene's container is heated, and in
    the canonical scene when its container is heated too."""
    env = multi_env
    pots = _pots(env)
    rice = [env.scenes[s].get_system("brown_rice") for s in range(N_ENVS)]
    water = [env.scenes[s].get_system("water") for s in range(N_ENVS)]
    og.sim.step()

    for s in range(N_ENVS):
        rice[s].generate_particles(positions=[(pots[s].aabb_center + th.tensor([0.03, 0.0, 0.0])).tolist()])
        water[s].generate_particles(positions=[(pots[s].aabb_center + th.tensor([-0.03, 0.0, 0.0])).tolist()])
    og.sim.step()
    for s in range(N_ENVS):
        assert pots[s].states[Contains].get_value(rice[s]), f"setup failed: rice not contained in scene {s}"
        assert pots[s].states[Contains].get_value(water[s]), f"setup failed: water not contained in scene {s}"

    def cooked(s):
        scene = env.scenes[s]
        return scene.is_system_active("cooked__brown_rice") and scene.get_system("cooked__brown_rice").n_particles > 0

    # Heat ONLY scene 1's pot: the rule must fire in scene 1 and ONLY in scene 1.
    assert pots[1].states[Heated].set_value(True)
    for _ in range(8):
        og.sim.step()
    assert cooked(1), "rule did not fire in scene 1 despite its container being heated with recipe inputs inside"
    assert not cooked(0), "rule fired in scene 0 even though only scene 1's container was heated"
    assert rice[0].n_particles > 0, "scene 0's raw particles were consumed without its container being heated"

    # Now heat scene 0's pot as well: the rule must fire there too.
    assert pots[0].states[Heated].set_value(True)
    for _ in range(8):
        og.sim.step()
    assert cooked(0), "rule did not fire in scene 0 after its container was heated"
