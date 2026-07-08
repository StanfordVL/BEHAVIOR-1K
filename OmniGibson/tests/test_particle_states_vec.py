"""
Multi-env / vectorization regression tests for particle object states
(Contains, Filled, Covered, ContainedParticles).

NON-OVERLAPPING with the existing N=1 tests in test_object_states.py / test_systems.py
(those are the N=1 golden oracle). Here we cover what they DON'T:
  - multi-env scene independence (a change in scene i must not affect scene j)
  - runtime particle-count change (grow/shrink) read back through the states

These assert CORRECT behavior, so they should already pass on the current (non-vectorized)
code (which is correct-but-slow per scene) and must KEEP passing after vectorization.
Per-env ParticleModifier throttling (T5) is deferred to Phase 4 and marked skip.
"""

import pytest

import omnigibson as og
from omnigibson.object_states import ContainedParticles, Contains, Covered, Filled
from omnigibson.systems import VisualParticleSystem

from utils import SYSTEM_EXAMPLES, setup_multi_environment


def _stockpot_cfg(pos):
    return dict(
        type="DatasetObject",
        name="vec_stockpot",
        category="stockpot",
        model="dcleem",
        abilities={"fillable": {}},
        position=pos,
    )


def _bowl_cfg(pos):
    return dict(type="DatasetObject", name="vec_bowl", category="bowl", model="ajzltc", position=pos)


def _physical_systems(scene):
    return [
        scene.get_system(name) for name, cls in SYSTEM_EXAMPLES.items() if not issubclass(cls, VisualParticleSystem)
    ]


def test_filled_scene_independence():
    """Filling the container in scene 0 must not fill it in scene 1."""
    env = setup_multi_environment(2, additional_objects_cfg=[_stockpot_cfg([0.0, -1.5, 0.5])])
    for _ in range(10):
        og.sim.step()

    pot0 = env.scenes[0].object_registry("name", "vec_stockpot")
    pot1 = env.scenes[1].object_registry("name", "vec_stockpot")

    for sysname in ("water", "white_rice"):
        w0 = env.scenes[0].get_system(sysname)
        w1 = env.scenes[1].get_system(sysname)

        assert pot0.states[Filled].set_value(w0, True), f"could not fill scene 0 with {sysname}"
        og.sim.step()

        assert pot0.states[Filled].get_value(w0), f"scene 0 {sysname} should be Filled"
        assert not pot1.states[Filled].get_value(w1), f"scene 1 {sysname} leaked from scene 0"
        assert pot1.states[ContainedParticles].get_value(w1).n_in_volume == 0

        w0.remove_all_particles()
        og.sim.step()
        assert not pot0.states[Filled].get_value(w0)

    og.clear()


def test_covered_scene_independence():
    """Covering the object in scene 0 must not cover it in scene 1."""
    env = setup_multi_environment(2, additional_objects_cfg=[_bowl_cfg([0.0, -1.5, 0.3])])
    for _ in range(10):
        og.sim.step()

    bowl0 = env.scenes[0].object_registry("name", "vec_bowl")
    bowl1 = env.scenes[1].object_registry("name", "vec_bowl")

    for sysname in SYSTEM_EXAMPLES:  # all families
        s0 = env.scenes[0].get_system(sysname)
        s1 = env.scenes[1].get_system(sysname)

        assert bowl0.states[Covered].set_value(s0, True), f"could not cover scene 0 with {sysname}"
        og.sim.step()

        assert bowl0.states[Covered].get_value(s0), f"scene 0 {sysname} should be Covered"
        assert not bowl1.states[Covered].get_value(s1), f"scene 1 {sysname} leaked from scene 0"

        bowl0.states[Covered].set_value(s0, False)
        bowl0.states[Covered].clear_cache()
        s0.remove_all_particles()

    og.clear()


def test_contains_reflects_fill_and_empty():
    """ContainedParticles / Contains must track a runtime count change: empty -> filled -> empty.
    (Reader-level grow/shrink of the Fabric buffer is already covered by probe6; this is the
    state-level guard.) Uses Filled.set_value to populate reliably per-scene (no manual placement)."""
    env = setup_multi_environment(1, additional_objects_cfg=[_stockpot_cfg([0.0, -1.5, 0.5])])
    for _ in range(10):
        og.sim.step()

    pot = env.scenes[0].object_registry("name", "vec_stockpot")
    water = env.scenes[0].get_system("water")

    # empty
    assert pot.states[ContainedParticles].get_value(water).n_in_volume == 0
    assert not pot.states[Contains].get_value(water)

    # grow: fill it (reliable per-scene sampling)
    assert pot.states[Filled].set_value(water, True)
    og.sim.step()
    n_full = pot.states[ContainedParticles].get_value(water).n_in_volume
    assert n_full > 0, "filled container should contain particles"
    assert pot.states[Contains].get_value(water)

    # shrink: back to empty
    water.remove_all_particles()
    og.sim.step()
    assert pot.states[ContainedParticles].get_value(water).n_in_volume == 0
    assert not pot.states[Contains].get_value(water)

    og.clear()


@pytest.mark.skip(reason="Per-env ParticleModifier throttling is Phase 4 (team-gated actuator split).")
def test_particle_modifier_per_env_throttling():
    raise NotImplementedError
