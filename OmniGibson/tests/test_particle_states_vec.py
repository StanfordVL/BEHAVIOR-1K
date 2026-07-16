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
from omnigibson.object_states import ContactParticles, ContainedParticles, Contains, Covered, Filled
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
        assert bowl0.states[Covered].cache == {}
        assert bowl1.states[Covered].cache == {}

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


def test_particle_metadata_mutation_refreshes_values_without_legacy_cache():
    """Particle topology changes must invalidate VALUES_CPU immediately, without a simulation step.

    Tensorized states and their cheap boolean wrappers use VALUES_CPU as their only cache. Keeping a
    second BaseObjectState cache would make an empty -> non-empty -> empty sequence return stale data.
    """
    env = setup_multi_environment(1, additional_objects_cfg=[_stockpot_cfg([0.0, -1.5, 0.5])])
    try:
        for _ in range(10):
            og.sim.step()

        pot = env.scenes[0].object_registry("name", "vec_stockpot")
        water = env.scenes[0].get_system("water")
        contained = pot.states[ContainedParticles]
        contains = pot.states[Contains]
        filled = pot.states[Filled]

        assert contained.get_value(water).n_in_volume == 0
        assert not contains.get_value(water)
        assert not filled.get_value(water)
        assert contained.cache == {}
        assert contains.cache == {}
        assert filled.cache == {}

        water.generate_particles(positions=[pot.aabb_center.tolist()])

        assert contained.get_value(water).n_in_volume == 1
        assert contains.get_value(water)
        assert not filled.get_value(water)
        assert contained.cache == {}
        assert contains.cache == {}
        assert filled.cache == {}

        water.remove_all_particles()

        assert contained.get_value(water).n_in_volume == 0
        assert not contains.get_value(water)
        assert contained.cache == {}
        assert contains.cache == {}
    finally:
        og.clear()


def test_contained_particles_kernel_matches_golden():
    """Sub-step 2 (kernel isolation): drive ParticleViewAPI + the tensorized ContainedParticles
    directly and assert the batched count matches the per-object golden (system getter +
    link.check_points_in_volume), per (scene, container, system). This bypasses get_value / the sim
    graph; the end-to-end get_value paths are validated after the sim wiring in sub-step 3."""
    import warp as wp

    from omnigibson.utils.particle_view_utils import ParticleViewAPI

    env = setup_multi_environment(
        2, additional_objects_cfg=[_stockpot_cfg([0.0, -1.5, 0.5]), _bowl_cfg([1.0, -1.5, 0.3])]
    )
    for _ in range(10):
        og.sim.step()

    # Fill water into the pot in both scenes; cover the bowl with stain in scene 0 (exercises the
    # visual-offset branch, which should correctly count 0 against the far-away pot).
    for s in range(2):
        pot = env.scenes[s].object_registry("name", "vec_stockpot")
        assert pot.states[Filled].set_value(env.scenes[s].get_system("water"), True)
    bowl0 = env.scenes[0].object_registry("name", "vec_bowl")
    bowl0.states[Covered].set_value(env.scenes[0].get_system("stain"), True)
    for _ in range(3):
        og.sim.step()

    # Drive the reader + the tensorized state directly (no sim graph, no get_value).
    ParticleViewAPI.initialize_view()
    ContainedParticles.initialize_view()
    ParticleViewAPI.prepare_step_host()
    ParticleViewAPI.update_positions_gpu()  # graph-phase macro-visual kernel; manual drive must run both phases
    ContainedParticles.global_update()
    wp.synchronize()

    checked_nonzero = 0
    for scene_idx, system_name in ParticleViewAPI.entries():
        system = env.scenes[scene_idx].get_system(system_name)
        sys_idx = ContainedParticles.SYS_IDXS[system_name]
        for relpath, obj_idx in ContainedParticles.OBJ_IDXS.items():
            container = ContainedParticles.IDX_OBJS[scene_idx][obj_idx]
            if container is None:
                continue
            golden = container.states[ContainedParticles]._compute_positions_in_volume(system)[1]
            golden_count = int(golden.sum()) if golden.numel() > 0 else 0
            tensor_count = int(ContainedParticles.VALUES_CPU[scene_idx, obj_idx, sys_idx])
            assert tensor_count == golden_count, (
                f"scene {scene_idx} container {relpath} system {system_name}: "
                f"kernel {tensor_count} != golden {golden_count}"
            )
            checked_nonzero += int(golden_count > 0)
    assert checked_nonzero > 0, "expected at least one non-empty (container, system) count"

    og.clear()


def test_contained_particles_data_supports_transition_rule_consumers():
    """Guard the ContainedParticlesData contract that cooking transition rules depend on. _execute_recipe
    selects the in-volume particles with th.where(state.in_volume) and then indexes state.positions by
    them, so the count, the per-particle mask, and the positions array must all describe the same state:
    the in_volume mask must have exactly n_in_volume set entries and positions must be indexable by it."""
    env = setup_multi_environment(1, additional_objects_cfg=[_stockpot_cfg([0.0, -1.5, 0.5])])
    try:
        for _ in range(10):
            og.sim.step()

        pot = env.scenes[0].object_registry("name", "vec_stockpot")
        water = env.scenes[0].get_system("water")
        assert pot.states[Filled].set_value(water, True)
        og.sim.step()

        data = pot.states[ContainedParticles].get_value(water)
        n_in_volume = data.n_in_volume
        assert n_in_volume > 0, "filled container should report contained particles"

        # Mirror _execute_recipe's consumption of the result.
        in_volume_idx = data.in_volume.nonzero().flatten()
        assert in_volume_idx.numel() == n_in_volume, "in_volume mask count must match n_in_volume"
        assert int(data.in_volume.sum()) == n_in_volume, "mask and count must describe the same state"
        assert data.positions[in_volume_idx].shape[0] == n_in_volume, "positions must be indexable by in_volume"
    finally:
        og.clear()


def _overlap_contact_ids(obj, system, link=None):
    """Golden reference: the ORIGINAL per-particle overlap_sphere contact set (ids into the system's
    own particle order), replicated here since ContactParticles no longer keeps the loop."""
    from omnigibson.object_states import AABB

    contacts = set()
    cur = {}

    def report_hit(hit):
        link_name = None if link is None else link.prim_path.split("/")[-1]
        base, body = "/".join(hit.rigid_body.split("/")[:-1]), hit.rigid_body.split("/")[-1]
        if (link is None and base == obj.prim_path) or (link is not None and link_name == body):
            contacts.add(cur["i"])
            return False
        return True

    lower, upper = obj.states[AABB].get_value() if link is None else link.visual_aabb
    lower = lower - (system.particle_radius + 2.5e-2)
    upper = upper + (system.particle_radius + 2.5e-2)
    positions = system.get_particles_position_orientation()[0]
    inbound = ((lower < positions) & (positions < upper)).all(dim=-1).nonzero()
    dist = system.particle_contact_radius + 5e-3
    for idx in inbound:
        cur["i"] = int(idx.item())
        og.sim.psqi.overlap_sphere(dist, positions[idx.item()].cpu().numpy(), report_hit, False)
    return contacts


def test_contact_particles_matches_golden():
    """Phase 3: end-to-end via get_value, assert the tensorized ContactParticles `.count` AND the exact
    `.particle_indices` set match the golden overlap_sphere, per (scene, object, physical-system), at N=2. The signed
    mesh_query_point kernel reconstructs overlap_sphere's solid-overlap, so this holds for both the
    thin-walled bowl and the thick/structural floor."""
    env = setup_multi_environment(2, additional_objects_cfg=[_bowl_cfg([0.0, -1.5, 0.3])])
    for _ in range(10):
        og.sim.step()

    # Cover the bowl with a physical system in both scenes -> particles rest on it (in contact).
    for s in range(2):
        bowl = env.scenes[s].object_registry("name", "vec_bowl")
        assert bowl.states[Covered].set_value(env.scenes[s].get_system("white_rice"), True)
    for _ in range(3):
        og.sim.step()

    checked_nonzero = 0
    for s in range(2):
        scene = env.scenes[s]
        for system_name in ("white_rice", "water"):
            if not scene.is_physical_particle_system(system_name=system_name):
                continue
            system = scene.get_system(system_name)
            for obj in scene.objects:
                if ContactParticles not in obj.states:
                    continue
                golden = _overlap_contact_ids(obj, system)  # inline overlap_sphere reference
                data = obj.states[ContactParticles].get_value(system)  # tensor count + GPU particle_indices
                assert data.count == len(
                    golden
                ), f"scene {s} {obj.name} {system_name}: count {data.count} != golden {len(golden)}"
                assert data.particle_indices == golden, f"scene {s} {obj.name} {system_name}: index set mismatch"
                checked_nonzero += int(len(golden) > 0)
    assert checked_nonzero > 0, "expected at least one non-empty (object, system) contact set"

    og.clear()


def test_contact_particles_data_is_a_coherent_snapshot():
    """count and particle_indices must never disagree. Reading count alone is a cheap tensor read that
    does NOT build the index set; the moment the indices are built they define count (len), computed
    from the current positions over the same contact test. So even on a result retained across a world
    change, materializing its indices yields a count that matches them, rather than pairing an old
    tensor count with freshly recomputed indices (the original bug)."""
    env = setup_multi_environment(1, additional_objects_cfg=[_bowl_cfg([0.0, -1.5, 0.3])])
    try:
        for _ in range(10):
            og.sim.step()

        bowl = env.scenes[0].object_registry("name", "vec_bowl")
        rice = env.scenes[0].get_system("white_rice")
        assert bowl.states[Covered].set_value(rice, True)
        for _ in range(3):
            og.sim.step()

        # Reading count alone is cheap: it must not build the index set.
        data = bowl.states[ContactParticles].get_value(rice)
        assert bowl.states[ContactParticles].cache == {}
        assert data.count > 0, "test setup expected contacting particles"
        assert not data._particle_indices_computed, "reading count must not build the index set"
        # Once the set is built, count is derived from it (same contact test, same positions) -> agree.
        assert data.count == len(data.particle_indices)

        # A new step refreshes VALUES_CPU. Fetch a new lazy result, snapshot only its count (indices
        # NOT built yet), then empty the system underneath it.
        og.sim.step()
        retained = bowl.states[ContactParticles].get_value(rice)
        assert retained.count > 0
        assert not retained._particle_indices_computed
        rice.remove_all_particles()
        for _ in range(3):
            og.sim.step()

        # The shared particle view now reflects the emptied system.
        assert bowl.states[ContactParticles].get_value(rice).count == 0
        # Materializing the retained result's indices reflects that same emptied state, and its count
        # follows the indices -> the pair stays internally consistent (old code kept the stale count>0).
        assert len(retained.particle_indices) == 0
        assert retained.count == 0
    finally:
        og.clear()


@pytest.mark.skip(reason="Per-env ParticleModifier throttling is Phase 4 (team-gated actuator split).")
def test_particle_modifier_per_env_throttling():
    raise NotImplementedError
