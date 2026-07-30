"""
Phase-1 parity tests for ParticleViewAPI (omnigibson/utils/particle_view_utils.py).

Validates that the unified reader exposes, per (scene, system), GPU-resident particle
positions that match each system's own get_particles_position_orientation() getter,
across all three families and across multiple scenes (correct per-scene keying).

This is the guard used when the micro backend is swapped to the Fabric fast-path.
"""

import pytest
import torch as th
import warp as wp

import omnigibson as og
from omnigibson.object_states import Covered, Filled
from omnigibson.utils.particle_view_utils import ParticleViewAPI

from utils import setup_multi_environment


def _stockpot_cfg(pos):
    return dict(
        type="DatasetObject",
        name="pv_stockpot",
        category="stockpot",
        model="dcleem",
        abilities={"fillable": {}},
        position=pos,
    )


def _bowl_cfg(pos):
    return dict(type="DatasetObject", name="pv_bowl", category="bowl", model="ajzltc", position=pos)


def _assert_parity(env):
    """ParticleViewAPI positions must match each (scene, system)'s own getter, on GPU."""
    ParticleViewAPI.initialize_view()
    ParticleViewAPI.prepare_step_host()
    ParticleViewAPI.update_positions_gpu()

    keys = ParticleViewAPI.entries()
    assert len(keys) > 0, "ParticleViewAPI registered no systems"

    checked = 0
    for scene_idx, system_name in keys:
        system = env.scenes[scene_idx].get_system(system_name)
        gpu = ParticleViewAPI.get_particle_positions(scene_idx, system_name)
        assert gpu is not None, f"no positions for ({scene_idx}, {system_name})"
        assert gpu.device.is_cuda, f"positions for ({scene_idx}, {system_name}) not on GPU"

        got = wp.to_torch(gpu).cpu()
        ref = system.get_particles_position_orientation()[0].cpu() if system.n_particles > 0 else th.zeros((0, 3))
        assert got.shape == ref.shape, f"shape mismatch ({scene_idx},{system_name}): {got.shape} vs {ref.shape}"
        if ref.numel() > 0:
            assert th.allclose(got, ref, atol=1e-4), f"position mismatch for ({scene_idx}, {system_name})"
            checked += 1
    assert checked > 0, "no non-empty systems were actually compared"


def test_particle_view_parity_multi_scene():
    """N=2: populate all 3 families (water=micro, stain=macro-visual, diced__apple=macro-physical),
    with per-scene asymmetry, and assert ParticleViewAPI matches each scene's getter."""
    env = setup_multi_environment(
        2, additional_objects_cfg=[_stockpot_cfg([0.0, -1.5, 0.5]), _bowl_cfg([1.0, -1.5, 0.3])]
    )
    for _ in range(10):
        og.sim.step()

    # scene 0: all three families; scene 1: only water (asymmetry -> checks per-scene keying)
    pot0 = env.scenes[0].object_registry("name", "pv_stockpot")
    bowl0 = env.scenes[0].object_registry("name", "pv_bowl")
    pot1 = env.scenes[1].object_registry("name", "pv_stockpot")

    # micro-physical (water) in BOTH scenes
    pot0.states[Filled].set_value(env.scenes[0].get_system("water"), True)
    pot1.states[Filled].set_value(env.scenes[1].get_system("water"), True)
    # macro-visual (stain) in scene 0 only
    bowl0.states[Covered].set_value(env.scenes[0].get_system("stain"), True)
    # macro-physical (diced__apple) in scene 0 only
    diced0 = env.scenes[0].get_system("diced__apple")
    diced0.generate_particles(positions=[[0.0, -1.5, 1.0], [0.05, -1.5, 1.0], [0.0, -1.45, 1.0]])

    for _ in range(3):
        og.sim.step()

    _assert_parity(env)

    # macro-visual (stain) must have gone through the single combined attached-body KERNEL: it is
    # NOT a getter fallback, and the combined table covers > 0 particles.
    assert (0, "stain") not in ParticleViewAPI._macro_visual_fallback_keys, "stain should use the kernel path"
    assert ParticleViewAPI._macro_visual_num_particles > 0, "combined macro-visual table should be non-empty"

    # macro-physical (diced__apple) must have gone through the single cross-scene rigid-body VIEW
    # + scatter kernel (not the per-system getter loop), covering all 3 diced particles.
    assert ParticleViewAPI._macro_physical_view is not None, "macro-physical should use the cross-scene view"
    assert ParticleViewAPI._macro_physical_num_particles == 3, "cross-scene view should cover all 3 diced particles"

    # explicit per-scene keying: scene 1 must have water but NOT stain/diced
    assert (1, "water") in ParticleViewAPI.entries()
    assert (0, "stain") in ParticleViewAPI.entries()
    assert (1, "stain") not in ParticleViewAPI.entries()

    # Sub-step 1: device particle count matches the total across all entries.
    total = sum(count for (_, count) in ParticleViewAPI._entry_ranges.values())
    assert wp.to_torch(ParticleViewAPI.PARTICLE_COUNT)[0].item() == total

    # Sub-step 1: VISUAL_PARTICLE_ORIENTATION is populated (unit quats) for the visual (stain) slice.
    start, count = ParticleViewAPI._entry_ranges[(0, "stain")]
    stain_quats = wp.to_torch(ParticleViewAPI.VISUAL_PARTICLE_ORIENTATION)[start : start + count].cpu()
    assert count > 0
    assert th.allclose(stain_quats.norm(dim=1), th.ones(count), atol=1e-3), "stain orientations should be unit quats"

    og.clear()


def _setup_stain_particle():
    env = setup_multi_environment(1, additional_objects_cfg=[_bowl_cfg([0.0, -1.5, 0.3])])
    for _ in range(10):
        og.sim.step()

    bowl = env.scenes[0].object_registry("name", "pv_bowl")
    stain = env.scenes[0].get_system("stain")
    assert bowl.states[Covered].set_value(stain, True)
    for _ in range(3):
        og.sim.step()
    assert stain.n_particles > 0

    ParticleViewAPI.initialize_view()
    ParticleViewAPI.prepare_step_host()
    ParticleViewAPI.update_positions_gpu()
    wp.synchronize()
    return env, bowl, stain


def _particle_view_positions(system):
    positions = ParticleViewAPI.get_particle_positions(system.scene.idx, system.name)
    assert positions is not None
    return wp.to_torch(positions).cpu().clone()


def test_macro_visual_parent_motion_visible_in_same_refresh():
    """Moving an attached particle's parent must be visible in the first cache refresh."""
    from omnigibson.object_states.tensorized_state import TensorizedState

    env, bowl, stain = _setup_stain_particle()
    try:
        position, orientation = bowl.get_position_orientation()
        bowl.set_position_orientation(position=position + th.tensor([0.4, 0.0, 0.0]), orientation=orientation)

        TensorizedState.caches_dirty = True
        og.sim._refresh_state_caches()

        got = _particle_view_positions(stain)
        expected = stain.get_particles_position_orientation()[0]
        assert th.allclose(got, expected, atol=1e-4), "visual particles lagged their moved parent by one refresh"
    finally:
        og.clear()


def test_macro_physical_motion_visible_in_same_refresh():
    """The host PhysX read and captured H2D/scatter must expose a rigid particle move immediately."""
    from omnigibson.object_states.tensorized_state import TensorizedState

    env, _, _ = _setup_stain_particle()
    try:
        diced = env.scenes[0].get_system("diced__apple")
        diced.generate_particles(positions=[[0.0, -1.5, 1.0]])
        for _ in range(2):
            og.sim.step()

        positions, orientations = diced.get_particles_position_orientation()
        diced.set_particles_position_orientation(
            positions=positions + th.tensor([[0.3, 0.0, 0.0]]), orientations=orientations
        )

        TensorizedState.caches_dirty = True
        og.sim._refresh_state_caches()

        got = _particle_view_positions(diced)
        expected = diced.get_particles_position_orientation()[0]
        assert th.allclose(got, expected, atol=1e-4), "macro-physical particles lagged the PhysX host staging read"
    finally:
        og.clear()


def test_macro_physical_count_changes_reuse_capacity_without_global_handle_rebuild(monkeypatch):
    """In-capacity rigid-particle add/remove refreshes only particle-owned PhysX views."""
    from omnigibson.object_states.tensorized_state import TensorizedState

    env, _, _ = _setup_stain_particle()
    try:
        diced = env.scenes[0].get_system("diced__apple")
        # System activation changes the entry registry once. The add/remove operations below are
        # count-only changes within that stable registry and must not rebuild global handles.
        og.sim.update_handles()
        ParticleViewAPI.prepare_step_host()

        old_positions = ParticleViewAPI.PARTICLE_POSITIONS
        old_transforms = ParticleViewAPI._macro_physical_transforms
        old_entry_index = ParticleViewAPI._macro_physical_particle_entry_index
        assert ParticleViewAPI._macro_physical_capacity >= 2

        def fail_global_handle_rebuild():
            raise AssertionError("macro-physical membership change called Simulator.update_handles()")

        monkeypatch.setattr(og.sim, "update_handles", fail_global_handle_rebuild)
        TensorizedState.graph_dirty = False

        diced.generate_particles(positions=[[0.0, -1.5, 1.0], [0.2, -1.5, 1.0]])
        assert diced.particle_metadata_dirty
        ParticleViewAPI.prepare_step_host()
        ParticleViewAPI.update_positions_gpu()
        wp.synchronize()

        assert ParticleViewAPI.PARTICLE_POSITIONS is old_positions
        assert ParticleViewAPI._macro_physical_transforms is old_transforms
        assert ParticleViewAPI._macro_physical_particle_entry_index is old_entry_index
        assert not TensorizedState.graph_dirty
        assert th.allclose(_particle_view_positions(diced), diced.get_particles_position_orientation()[0], atol=1e-4)

        diced.remove_particles(idxs=th.tensor([0]))
        ParticleViewAPI.prepare_step_host()
        ParticleViewAPI.update_positions_gpu()
        wp.synchronize()

        assert ParticleViewAPI.PARTICLE_POSITIONS is old_positions
        assert ParticleViewAPI._macro_physical_transforms is old_transforms
        assert ParticleViewAPI._macro_physical_particle_entry_index is old_entry_index
        assert not TensorizedState.graph_dirty
        assert th.allclose(_particle_view_positions(diced), diced.get_particles_position_orientation()[0], atol=1e-4)
    finally:
        monkeypatch.undo()
        og.clear()


def test_macro_visual_local_pose_update_invalidates_particle_view_table():
    """Changing existing visual particles' poses must refresh cached local matrices without add/remove."""
    env, _, stain = _setup_stain_particle()
    try:
        assert not stain.particle_metadata_dirty
        old_local_matrix = ParticleViewAPI._macro_visual_local_matrix

        positions, orientations = stain.get_particles_position_orientation()
        moved_positions = positions.clone()
        moved_positions[:, 0] += 0.25
        stain.set_particles_position_orientation(positions=moved_positions, orientations=orientations)
        assert stain.particle_metadata_dirty

        from omnigibson.object_states.tensorized_state import TensorizedState

        TensorizedState.graph_dirty = False
        ParticleViewAPI.prepare_step_host()
        ParticleViewAPI.update_positions_gpu()
        wp.synchronize()

        assert not stain.particle_metadata_dirty
        assert ParticleViewAPI._macro_visual_local_matrix is old_local_matrix
        assert not TensorizedState.graph_dirty, "in-capacity metadata updates should not force graph recapture"
        got = _particle_view_positions(stain)
        expected = stain.get_particles_position_orientation()[0]
        assert th.allclose(got, expected, atol=1e-4), "visual-particle local-pose cache was not invalidated"

        # Removing every visual particle still has to clear the old table. In particular, the reader
        # must not return early merely because the new total particle count is zero. Prim removal may
        # synchronously rebuild topology, so the flag can already be clean when this call returns.
        stain.remove_all_particles()
        ParticleViewAPI.prepare_step_host()
        assert not stain.particle_metadata_dirty
        assert ParticleViewAPI._macro_visual_num_particles == 0
    finally:
        og.clear()


def test_particle_metadata_dirty_is_family_scoped():
    """Micro metadata changes must not rebuild macro-visual tables, live physical state is not
    metadata, and rebuilding ParticleViewAPI must eagerly produce a complete clean view."""
    env, _, stain = _setup_stain_particle()
    try:
        scene = env.scenes[0]
        water = scene.get_system("water")
        diced = scene.get_system("diced__apple")
        diced.generate_particles(positions=[[0.0, -1.5, 1.0]])

        ParticleViewAPI.initialize_view()
        ParticleViewAPI.prepare_step_host()
        ParticleViewAPI.update_positions_gpu()
        wp.synchronize()
        assert not stain.particle_metadata_dirty
        assert not water.particle_metadata_dirty
        assert not diced.particle_metadata_dirty

        # Deserializing an identical group layout calls _sync_particle_groups(), but the sync is a
        # no-op and therefore must not trigger a table rebuild by itself.
        stain.deserialize(stain.serialize(stain._dump_state()))
        assert not stain.particle_metadata_dirty

        # Micro topology dirties only the micro system. Its family metadata/layout refresh must not
        # rebuild the unchanged macro-visual arrays.
        from omnigibson.object_states.tensorized_state import TensorizedState

        old_visual_local_matrix = ParticleViewAPI._macro_visual_local_matrix
        old_entry_start = ParticleViewAPI._entry_start
        TensorizedState.graph_dirty = False
        water.generate_particles(positions=_water_positions(3))
        assert water.particle_metadata_dirty
        assert not stain.particle_metadata_dirty
        ParticleViewAPI.prepare_step_host()
        ParticleViewAPI.update_positions_gpu()
        wp.synchronize()
        assert not water.particle_metadata_dirty
        assert ParticleViewAPI._macro_visual_local_matrix is old_visual_local_matrix
        assert ParticleViewAPI._entry_start is old_entry_start
        assert not TensorizedState.graph_dirty, "in-capacity layout updates should not force graph recapture"

        # Count changes are metadata changes too: they update the packed entry offsets.
        water.generate_particles(positions=_water_positions(1, base=(3.5, 0.0, 2.0)))
        assert water.particle_metadata_dirty
        ParticleViewAPI.prepare_step_host()
        assert not water.particle_metadata_dirty
        assert ParticleViewAPI._entry_start is old_entry_start
        assert not TensorizedState.graph_dirty
        assert ParticleViewAPI._entry_ranges[(0, "water")][1] == water.n_particles
        assert ParticleViewAPI._macro_visual_local_matrix is old_visual_local_matrix

        water.remove_particles(idxs=th.tensor([water.n_particles - 1]))
        assert water.particle_metadata_dirty
        ParticleViewAPI.prepare_step_host()
        assert ParticleViewAPI._entry_ranges[(0, "water")][1] == water.n_particles

        # Physical positions and velocities are fetched live, so changing them must not invalidate
        # structural/static metadata for either physical family.
        water_positions, water_orientations = water.get_particles_position_orientation()
        water.set_particles_position_orientation(
            positions=water_positions + th.tensor([0.01, 0.0, 0.0]), orientations=water_orientations
        )
        assert not water.particle_metadata_dirty

        diced_positions, diced_orientations = diced.get_particles_position_orientation()
        diced.set_particles_position_orientation(
            positions=diced_positions + th.tensor([0.01, 0.0, 0.0]), orientations=diced_orientations
        )
        diced.set_particles_velocities(lin_vels=th.zeros((1, 3)), ang_vels=th.zeros((1, 3)))
        assert not diced.particle_metadata_dirty

        water.remove_all_particles()
        ParticleViewAPI.prepare_step_host()
        assert not water.particle_metadata_dirty

        # Rebuilding topology is eager: no later position read is needed to create entry ranges or
        # family metadata, and all tracked system flags are clean when it returns.
        ParticleViewAPI.clear()
        assert not stain.particle_metadata_dirty
        assert not water.particle_metadata_dirty
        ParticleViewAPI.initialize_view()
        assert not stain.particle_metadata_dirty
        assert not water.particle_metadata_dirty
        assert not diced.particle_metadata_dirty
        assert ParticleViewAPI._entry_ranges[(0, "water")][1] == water.n_particles
        assert ParticleViewAPI._macro_visual_num_particles == stain.n_particles
        ParticleViewAPI.prepare_step_host()
        ParticleViewAPI.update_positions_gpu()
        wp.synchronize()
        assert not stain.particle_metadata_dirty
        assert not water.particle_metadata_dirty
        assert ParticleViewAPI._entry_ranges[(0, "water")][1] == water.n_particles
    finally:
        og.clear()


def test_micro_particle_view_supports_nondefault_instancer():
    """ParticleView must gather a supported non-default-first instancer without creating a default one."""
    env = setup_multi_environment(1)
    try:
        for _ in range(10):
            og.sim.step()

        water = env.scenes[0].get_system("water")
        nondefault = water.generate_particle_instancer(
            n_particles=3,
            positions=th.tensor([[4.0, 0.0, 2.0], [4.1, 0.0, 2.0], [4.2, 0.0, 2.0]]),
            idn=17,
            prototype_indices=[0, 0, 0],
        )
        og.sim.step()

        ParticleViewAPI.initialize_view()
        ParticleViewAPI.prepare_step_host()
        ParticleViewAPI.update_positions_gpu()  # graph-phase macro-visual kernel
        wp.synchronize()

        got = _particle_view_positions(water)
        expected = nondefault.particle_positions
        assert got.shape == expected.shape
        assert th.allclose(got, expected, atol=1e-4), "the non-default micro instancer was not gathered"
        assert water.n_instancers == 1, "ParticleView created a default instancer as a read side effect"
    finally:
        og.clear()


# ---------------------------------------------------------------------------
# Micro-physical instancer lifecycle: a system owns 0 or 1 instancer, access is
# non-mutating, state round-trips, and scenes stay independent.
# ---------------------------------------------------------------------------


def _water_positions(n, base=(3.0, 0.0, 2.0)):
    x, y, z = base
    return th.tensor([[x + 0.1 * i, y, z] for i in range(n)], dtype=th.float32)


def test_micro_instancer_lifecycle():
    """A micro system owns 0 or 1 instancer: absent access is non-mutating, first generation creates
    exactly one, later generation reuses it, and creating a second one fails clearly."""
    env = setup_multi_environment(1)
    try:
        for _ in range(5):
            og.sim.step()
        water = env.scenes[0].get_system("water")

        # Absent: non-mutating access creates nothing, even when reading positions of an empty system.
        assert water.n_instancers == 0
        assert water.particle_instancer is None
        pos, ori = water.get_particles_position_orientation()
        assert pos.shape == (0, 3) and ori.shape == (0, 4)
        assert water.n_instancers == 0, "reading an empty system created an instancer"

        # Setting empty poses on an empty system is a valid no-op; setting non-empty poses is a
        # caller error (there are no particles to move) and must fail loudly, not silently no-op.
        water.set_particles_position_orientation()
        with pytest.raises(AssertionError, match="no particle instancer"):
            water.set_particles_position_orientation(positions=th.zeros((2, 3)))
        assert water.n_instancers == 0

        # First generation creates exactly one instancer.
        water.generate_particles(positions=_water_positions(3))
        og.sim.step()
        assert water.n_instancers == 1
        first = water.particle_instancer
        assert first is not None
        first_path, first_count = first.prim_path, water.n_particles

        # Later generation reuses the same instancer (no second instancer).
        water.generate_particles(positions=_water_positions(2, base=(3.5, 0.0, 2.0)))
        og.sim.step()
        assert water.n_instancers == 1, "second generation created a new instancer"
        assert water.particle_instancer.prim_path == first_path
        assert water.n_particles == first_count + 2

        # Explicitly creating a second instancer fails clearly.
        with pytest.raises(AssertionError, match="only one is supported"):
            water.generate_particle_instancer(
                n_particles=1, idn=5, positions=_water_positions(1), prototype_indices=[0]
            )
    finally:
        og.clear()


def test_micro_instancer_state_roundtrip():
    """Zero-, default-ID, and non-default-ID (legacy) instancer states dump/load; a multi-instancer
    state is rejected clearly with no silent reinterpretation."""
    env = setup_multi_environment(1)
    try:
        for _ in range(5):
            og.sim.step()
        water = env.scenes[0].get_system("water")

        # Zero-instancer state loads (stays empty).
        empty_state = water._dump_state()
        assert int(empty_state["n_instancers"]) == 0
        water._load_state(empty_state)
        assert water.n_instancers == 0

        # Default-ID (0) state: dump, wipe, reload -> restored.
        water.generate_particles(positions=_water_positions(4))
        og.sim.step()
        assert water.particle_instancer.idn == 0
        saved, saved_count = water._dump_state(), water.n_particles
        water.remove_all_particles()
        og.sim.step()
        assert water.n_instancers == 0
        water._load_state(saved)
        og.sim.step()
        assert water.n_instancers == 1 and water.n_particles == saved_count
        assert water.particle_instancer.idn == 0

        # Non-default-ID (legacy) state loads and the sole instancer keeps its ID.
        water.remove_all_particles()
        og.sim.step()
        water.generate_particle_instancer(
            n_particles=3, idn=17, positions=_water_positions(3), prototype_indices=[0, 0, 0]
        )
        og.sim.step()
        legacy = water._dump_state()
        assert list(legacy["instancer_idns"]) == [17]
        water.remove_all_particles()
        og.sim.step()
        water._load_state(legacy)
        og.sim.step()
        assert water.n_instancers == 1 and water.particle_instancer.idn == 17

        # A multi-instancer state is rejected (no silent reinterpretation).
        with pytest.raises(AssertionError, match="at most one particle instancer"):
            water._sync_particle_instancers(idns=[0, 1], particle_groups=[0, 0], particle_counts=[1, 1])
    finally:
        og.clear()


def test_micro_instancer_serialize_roundtrip():
    """The flat serialize()/deserialize() path handles zero and one (non-default ID) instancer, and
    rejects a serialized state that declares more than one instancer."""
    env = setup_multi_environment(1)
    try:
        for _ in range(5):
            og.sim.step()
        water = env.scenes[0].get_system("water")

        # Zero instancers: serialize -> deserialize stays empty.
        water.deserialize(water.serialize(water._dump_state()))
        assert water.n_instancers == 0

        # One non-default-ID instancer: full flat round trip restores its ID and particle count.
        water.generate_particle_instancer(
            n_particles=3, idn=17, positions=_water_positions(3), prototype_indices=[0, 0, 0]
        )
        og.sim.step()
        flat, saved_count = water.serialize(water._dump_state()), water.n_particles
        water.remove_all_particles()
        og.sim.step()
        assert water.n_instancers == 0
        water.deserialize(flat)
        og.sim.step()
        assert water.n_instancers == 1 and water.particle_instancer.idn == 17
        assert water.n_particles == saved_count

        # A serialized state declaring >1 instancer is rejected (n_instancers=2, idns=[0,1]).
        with pytest.raises(AssertionError, match="at most one particle instancer"):
            water.deserialize(th.tensor([2, 0, 1, 0, 0, 0, 0]))
    finally:
        og.clear()


def test_micro_generate_large_particle_count():
    """Regression for the bulk prototype-index conversion (physx_utils): generating many particles at
    once must succeed. The fix converts the prototype-index tensor to a host list in one transfer; a
    per-element int() loop would be O(n) Python work (and one device sync per element for a CUDA
    tensor), pathological at large particle counts."""
    env = setup_multi_environment(1)
    try:
        for _ in range(5):
            og.sim.step()
        water = env.scenes[0].get_system("water")
        n = 2000
        positions = th.rand((n, 3)) * 0.5 + th.tensor([0.0, -1.5, 1.0])
        # Also pass an explicit CUDA prototype-index tensor to exercise the device->host bulk path.
        proto = th.zeros(n, dtype=th.int32, device="cuda") if th.cuda.is_available() else th.zeros(n, dtype=th.int32)
        water.generate_particle_instancer(n_particles=n, positions=positions, prototype_indices=proto)
        og.sim.step()
        assert water.n_particles == n
    finally:
        og.clear()


def test_micro_two_env_independent_instancers():
    """Two environments own different system objects and different scene-local instancer paths;
    mutating one does not affect the other; ParticleViewAPI reports correct per-scene positions."""
    env = setup_multi_environment(2)
    try:
        for _ in range(5):
            og.sim.step()
        w0 = env.scenes[0].get_system("water")
        w1 = env.scenes[1].get_system("water")
        assert w0 is not w1, "each scene must own its own system object"

        w0.generate_particles(positions=_water_positions(4, base=(0.0, -1.5, 1.0)))
        w1.generate_particles(positions=_water_positions(2, base=(0.0, -1.5, 1.0)))
        og.sim.step()

        # Distinct scene-local instancers, independent counts.
        assert w0.particle_instancer.prim_path != w1.particle_instancer.prim_path
        assert w0.n_particles == 4 and w1.n_particles == 2

        # Removing from env 0 does not affect env 1.
        w0.remove_all_particles()
        og.sim.step()
        assert w0.n_particles == 0
        assert w1.n_particles == 2, "removing particles in scene 0 changed scene 1"

        # ParticleViewAPI reports correct per-scene positions for both environments.
        ParticleViewAPI.initialize_view()
        ParticleViewAPI.prepare_step_host()
        ParticleViewAPI.update_positions_gpu()  # graph-phase macro-visual kernel
        wp.synchronize()
        for scene_idx, system in ((0, w0), (1, w1)):
            gpu = ParticleViewAPI.get_particle_positions(scene_idx, "water")
            got = wp.to_torch(gpu).cpu() if gpu is not None else th.zeros((0, 3))
            ref = system.get_particles_position_orientation()[0].cpu() if system.n_particles > 0 else th.zeros((0, 3))
            assert got.shape == ref.shape, f"scene {scene_idx} shape {got.shape} vs {ref.shape}"
            if ref.numel() > 0:
                assert th.allclose(got, ref, atol=1e-4), f"scene {scene_idx} positions mismatch"
    finally:
        og.clear()
