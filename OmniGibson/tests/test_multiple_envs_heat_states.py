"""Multi-env (num_envs=3) versions of the heat / temperature object-state tests.

Mirrors the heat-related coverage of ``test_object_states.py`` (test_temperature,
test_heat_source_or_sink, test_heated) across 3 cloned scenes, plus diagnostic
tests that read the vectorized heat pipeline's internal index maps to explain the
two failures observed in eval (2026-08-10, see eval-vector-check-20260810/REPORT.md):

  1. An OFF oven (requires_toggled_on + requires_inside, both unsatisfied) heated a
     popcorn bag sitting on a countertop to ~its 250 C source temperature.
  2. A toggled-ON burner/stove did not heat an object placed directly on it.

Transition rules are intentionally DISABLED here (the clone-scene transition-rule
bug is tracked separately).

Run standalone (one Environment per process):
  OMNIGIBSON_HEADLESS=1 pytest tests/test_multiple_envs_heat_states.py -v -s
"""

import pytest
import torch as th
import warp as wp

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.macros import macros as m
from omnigibson.object_states import HeatSourceOrSink, Inside, MaxTemperature, Open, Temperature, ToggledOn
from omnigibson.utils.usd_utils import RigidBodyViewAPI

N_ENVS = 3
DEFAULT_TEMP = m.object_states.temperature.DEFAULT_TEMPERATURE

# Heat sources under test. stove = point source (requires_toggled_on only);
# microwave / oven = containment sources (requires_toggled_on + closed + inside);
# fridge = always-active containment cold source.
OBJECTS_CFG = [
    {"type": "DatasetObject", "name": "stove", "category": "stove", "model": "yhjzwg", "position": [0.0, 0.0, 0.5]},
    {
        "type": "DatasetObject",
        "name": "microwave",
        "category": "microwave",
        "model": "hjjxmi",
        "position": [3.0, 0.0, 0.3],
    },
    {"type": "DatasetObject", "name": "fridge", "category": "fridge", "model": "xyejdx", "position": [6.0, 0.0, 1.0]},
    {"type": "DatasetObject", "name": "oven", "category": "oven", "model": "amblrk", "position": [9.0, 0.0, 0.6]},
    # Fillable decoy + an object inside it: creates a true Inside pair that a
    # misaligned requires_inside lookup could confuse with "inside the oven".
    {"type": "DatasetObject", "name": "bowl", "category": "bowl", "model": "ajzltc", "position": [12.0, 0.0, 0.1]},
    {
        "type": "DatasetObject",
        "name": "bagel_in_bowl",
        "category": "bagel",
        "model": "zlxkry",
        "abilities": {"cookable": {}, "freezable": {}, "burnable": {}, "heatable": {}},
        "position": [12.0, 0.0, 0.25],
    },
    # Flammable object: exercises the OnFire dump/load path (eval bug #1).
    {
        "type": "DatasetObject",
        "name": "plywood",
        "category": "plywood",
        "model": "fkmkqa",
        "abilities": {"flammable": {}},
        "position": [18.0, 0.0, 0.1],
    },
    # Far-away heatable targets (mirror bagel/cookable_dishtowel of the n=1 tests).
    {
        "type": "DatasetObject",
        "name": "bagel_a",
        "category": "bagel",
        "model": "zlxkry",
        "abilities": {"cookable": {}, "freezable": {}, "burnable": {}, "heatable": {}},
        "position": [15.0, 0.0, 0.1],
    },
    {
        "type": "DatasetObject",
        "name": "bagel_b",
        "category": "bagel",
        "model": "zlxkry",
        "abilities": {"cookable": {}, "freezable": {}, "burnable": {}, "heatable": {}},
        "position": [15.0, 1.0, 0.1],
    },
]

TARGET_NAMES = ("bagel_a", "bagel_b", "bagel_in_bowl")
SOURCE_NAMES = ("stove", "microwave", "fridge", "oven")


@pytest.fixture(scope="module")
def multi_env():
    assert og.sim is None, "This module must run in a fresh process (one Environment per process)."
    gm.RENDER_VIEWER_CAMERA = False
    gm.ENABLE_OBJECT_STATES = True
    gm.USE_GPU_DYNAMICS = True
    gm.ENABLE_FLATCACHE = False
    gm.ENABLE_TRANSITION_RULES = False  # clone-scene transition-rule bug tracked separately

    cfg = {
        "env": {"num_envs": N_ENVS},
        "scene": {"type": "Scene"},
        "robots": [{"model": "fetch", "obs_modalities": ["rgb"], "position": [20.0, 20.0, 0.1]}],
        "objects": OBJECTS_CFG,
        "task": {"type": "DummyTask"},
    }
    env = og.Environment(configs=cfg)
    for _ in range(10):
        og.sim.step()
    yield env
    og.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def scene_objs(env, name):
    """The per-scene clones of object @name, index-aligned with env indices."""
    return [env.scenes[s].object_registry("name", name) for s in range(N_ENVS)]


def temps(env, name):
    return [round(o.states[Temperature].get_value(), 2) for o in scene_objs(env, name)]


def reset_thermals(env):
    """All sources off / closed, all targets back to DEFAULT_TEMP."""
    for s in range(N_ENVS):
        for name in ("stove", "microwave", "oven"):
            obj = env.scenes[s].object_registry("name", name)
            if ToggledOn in obj.states:
                obj.states[ToggledOn].set_value(False)
        for name in ("microwave", "fridge", "oven"):
            obj = env.scenes[s].object_registry("name", name)
            if Open in obj.states:
                obj.states[Open].set_value(False, fully=True)
    og.sim.step()
    for s in range(N_ENVS):
        for name in TARGET_NAMES:
            obj = env.scenes[s].object_registry("name", name)
            assert obj.states[Temperature].set_value(DEFAULT_TEMP)
            assert obj.states[MaxTemperature].set_value(DEFAULT_TEMP)
    og.sim.step()


def influence_pairs(env):
    """Decode Temperature.INFLUENCE_MASK into human-readable (scene, source, target) triples."""
    mask = Temperature.INFLUENCE_MASK
    if mask is None:
        return []
    mask_t = wp.to_torch(mask).to("cpu")
    pairs = []
    for s, h, n in th.nonzero(mask_t, as_tuple=False).tolist():
        src = HeatSourceOrSink.IDX_OBJS[s][h] if s < len(HeatSourceOrSink.IDX_OBJS) else None
        tgt = Temperature.IDX_OBJS[s][n] if s < len(Temperature.IDX_OBJS) else None
        pairs.append((s, src.name if src is not None else f"h={h}", tgt.name if tgt is not None else f"n={n}"))
    return pairs


def heat_link_world_pos(obj):
    """World position of a point heat source's heating element link."""
    hss = obj.states[HeatSourceOrSink]
    link = hss.link if hss._links else obj.root_link
    return link.get_position_orientation()[0]


def dump_heat_maps(env):
    """Print the internal index maps the heat kernel consumes (diagnostic aid)."""
    print("\n--- heat pipeline maps")
    print("HSS OBJ_IDXS:", dict(HeatSourceOrSink.OBJ_IDXS or {}))
    print("Temperature OBJ_IDXS:", dict(Temperature.OBJ_IDXS or {}))
    print("Inside OBJ_IDXS:", dict(Inside.OBJ_IDXS or {}))
    if HeatSourceOrSink._link_flat_idx is not None:
        print("link_flat_idx:", wp.to_torch(HeatSourceOrSink._link_flat_idx).to("cpu").tolist())
    if Temperature._hss_self_inside_idx is not None:
        print("hss_self_inside_idx:", wp.to_torch(Temperature._hss_self_inside_idx).to("cpu").tolist())
    if Temperature._temp_to_inside_idx is not None:
        print("temp_to_inside_idx:", wp.to_torch(Temperature._temp_to_inside_idx).to("cpu").tolist())
    if Temperature._temp_to_aabb_idx is not None:
        print("temp_to_aabb_idx:", wp.to_torch(Temperature._temp_to_aabb_idx).to("cpu").tolist())


# ---------------------------------------------------------------------------
# 1) Leak detector — mirrors eval bug #1 (OFF oven heated the popcorn bag)
# ---------------------------------------------------------------------------


def test_all_sources_off_no_heating_n3(multi_env):
    """With every heat source off/closed and all targets far away, no target's
    temperature may change in ANY env. In eval, an OFF oven (both gates
    unsatisfied) heated a countertop popcorn bag to ~250 C; `bagel_in_bowl`
    reproduces the true-Inside decoy pair that a misaligned requires_inside
    lookup could confuse with "inside the oven"."""
    env = multi_env
    reset_thermals(env)

    for _ in range(30):
        og.sim.step()

    failures = []
    for name in TARGET_NAMES:
        for s, t in enumerate(temps(env, name)):
            if abs(t - DEFAULT_TEMP) > 1e-3:
                failures.append(f"{name} env{s}: {t} != {DEFAULT_TEMP}")
    if failures:
        print("\nLEAK: temperatures changed with all sources off:", failures)
        print("influence pairs this step:", influence_pairs(env))
        dump_heat_maps(env)
    assert not failures, f"Heat leak with all sources off: {failures}"


# ---------------------------------------------------------------------------
# 2) Point-source heating — mirrors eval bug #2 (ON burner didn't heat hotdog)
# ---------------------------------------------------------------------------


def test_toggled_stove_heats_targets_in_every_env_n3(multi_env):
    """A toggled-on stove must heat an object placed directly at its heating
    element in EVERY env. Targets are placed at each scene's OWN heat-element
    link position, so if the kernel resolves the element pose through a single
    per-column link index (first scene wins) instead of a per-scene one, envs
    beyond that scene compute a ~scene-offset distance and never heat."""
    env = multi_env
    reset_thermals(env)

    stoves = scene_objs(env, "stove")
    bagels = scene_objs(env, "bagel_a")
    for stove, bagel in zip(stoves, bagels):
        elem = heat_link_world_pos(stove)
        bagel.keep_still()
        bagel.set_position_orientation(position=elem + th.tensor([0.0, 0.0, 0.05]))
    og.sim.step()
    for stove in stoves:
        assert stove.states[ToggledOn].set_value(True)
    for _ in range(10):
        og.sim.step()

    gate_values = [stove.states[HeatSourceOrSink].get_value() for stove in stoves]
    print("\nstove HSS active per env:", gate_values)
    print("bagel_a temps per env:", temps(env, "bagel_a"))
    print("influence pairs:", influence_pairs(env))

    assert all(gate_values), f"Toggled-on stove not active as heat source in all envs: {gate_values}"

    failures = [f"env{s}: {t}" for s, t in enumerate(temps(env, "bagel_a")) if t <= DEFAULT_TEMP + 1e-3]
    if failures:
        # Print exactly what the kernel used vs what each scene needs.
        dump_heat_maps(env)
        h = (HeatSourceOrSink.OBJ_IDXS or {}).get(stoves[0].relative_prim_path)
        for s, stove in enumerate(stoves):
            link = stove.states[HeatSourceOrSink].link
            print(
                f"env{s}: heat element world pos {heat_link_world_pos(stove).tolist()} "
                f"flat_idx(per-scene link)={RigidBodyViewAPI.get_flat_idx(link.prim_path)}"
            )
        if h is not None and HeatSourceOrSink._link_flat_idx is not None:
            lfi = wp.to_torch(HeatSourceOrSink._link_flat_idx).to("cpu")
            per_scene = lfi[:, h].tolist() if lfi.dim() == 2 else lfi[h].item()
            print(f"kernel link_flat_idx for stove column {h}:", per_scene)
    assert not failures, f"Toggled-on stove failed to heat bagel placed on its element: {failures}"

    reset_thermals(env)


def test_stove_heats_only_its_own_env_n3(multi_env):
    """Toggling only env1's stove must heat only env1's target (targets remain
    at each scene's heat element from the previous placement)."""
    env = multi_env
    reset_thermals(env)

    stoves = scene_objs(env, "stove")
    bagels = scene_objs(env, "bagel_a")
    for stove, bagel in zip(stoves, bagels):
        bagel.keep_still()
        bagel.set_position_orientation(position=heat_link_world_pos(stove) + th.tensor([0.0, 0.0, 0.05]))
    og.sim.step()
    assert stoves[1].states[ToggledOn].set_value(True)
    for _ in range(10):
        og.sim.step()

    t = temps(env, "bagel_a")
    print("\nbagel_a temps (only env1 stove on):", t, "| influence:", influence_pairs(env))
    assert abs(t[0] - DEFAULT_TEMP) < 1e-3, f"env0 heated by env1's stove: {t}"
    assert abs(t[2] - DEFAULT_TEMP) < 1e-3, f"env2 heated by env1's stove: {t}"
    assert t[1] > DEFAULT_TEMP + 1e-3, f"env1's own stove did not heat its target: {t}"

    reset_thermals(env)


# ---------------------------------------------------------------------------
# 3) Containment sources per env (microwave heat, fridge cold)
# ---------------------------------------------------------------------------


def test_microwave_requires_inside_per_env_n3(multi_env):
    """Objects inside a closed, toggled-on microwave heat up in every env;
    the same objects do NOT heat while the microwave is off."""
    env = multi_env
    reset_thermals(env)

    microwaves = scene_objs(env, "microwave")
    bagels = scene_objs(env, "bagel_b")
    for mw, bagel in zip(microwaves, bagels):
        bagel.keep_still()
        bagel.set_position_orientation(position=mw.aabb_center + th.tensor([0.0, 0.03, 0.0]))
    for _ in range(3):
        og.sim.step()

    inside_flags = [b.states[Inside].get_value(mw) for b, mw in zip(bagels, microwaves)]
    print("\nbagel_b inside microwave per env:", inside_flags)
    assert all(inside_flags), f"Placement failed, bagel not inside microwave in all envs: {inside_flags}"

    # Off: no heating.
    for _ in range(5):
        og.sim.step()
    t_off = temps(env, "bagel_b")
    assert all(abs(t - DEFAULT_TEMP) < 1e-3 for t in t_off), f"Heated by OFF microwave: {t_off}"

    # Closed + on: heats in every env.
    for mw in microwaves:
        assert mw.states[ToggledOn].set_value(True)
    for _ in range(10):
        og.sim.step()
    t_on = temps(env, "bagel_b")
    print("bagel_b temps with microwave on:", t_on, "| influence:", influence_pairs(env))
    failures = [f"env{s}: {t}" for s, t in enumerate(t_on) if t <= DEFAULT_TEMP + 1e-3]
    if failures:
        dump_heat_maps(env)
    assert not failures, f"Closed+on microwave failed to heat inside object: {failures}"

    reset_thermals(env)


def test_fridge_cools_per_env_n3(multi_env):
    """The fridge (always-active cold source) must cool objects inside it in every env."""
    env = multi_env
    reset_thermals(env)

    fridges = scene_objs(env, "fridge")
    bagels = scene_objs(env, "bagel_b")
    for fridge, bagel in zip(fridges, bagels):
        bagel.keep_still()
        bagel.set_position_orientation(position=fridge.aabb_center + th.tensor([0.0, 0.0, 0.1]))
        assert fridge.states[Open].set_value(False, fully=True)
    for _ in range(3):
        og.sim.step()

    inside_flags = [b.states[Inside].get_value(f) for b, f in zip(bagels, fridges)]
    print("\nbagel_b inside fridge per env:", inside_flags)
    assert all(inside_flags), f"Placement failed, bagel not inside fridge in all envs: {inside_flags}"

    for _ in range(10):
        og.sim.step()
    t = temps(env, "bagel_b")
    print("bagel_b temps in closed fridge:", t)
    failures = [f"env{s}: {v}" for s, v in enumerate(t) if v >= DEFAULT_TEMP - 1e-3]
    if failures:
        dump_heat_maps(env)
    assert not failures, f"Fridge failed to cool inside object: {failures}"

    reset_thermals(env)


# ---------------------------------------------------------------------------
# 4) Gate logic per env — mirrors test_heat_source_or_sink at n=3
# ---------------------------------------------------------------------------


def test_heat_source_gates_per_env_n3(multi_env):
    """HeatSourceOrSink activation gates must be evaluated per env independently."""
    env = multi_env
    reset_thermals(env)

    stoves = scene_objs(env, "stove")
    microwaves = scene_objs(env, "microwave")

    # All off -> inactive everywhere.
    for s in range(N_ENVS):
        assert not stoves[s].states[HeatSourceOrSink].get_value(), f"stove env{s} active while off"
        assert not microwaves[s].states[HeatSourceOrSink].get_value(), f"microwave env{s} active while off"

    # Toggle only env2's stove.
    assert stoves[2].states[ToggledOn].set_value(True)
    og.sim.step()
    values = [st.states[HeatSourceOrSink].get_value() for st in stoves]
    assert values == [False, False, True], f"stove gate per env wrong: {values}"

    # Microwave: open door blocks activation per env.
    for mw in microwaves:
        assert mw.states[ToggledOn].set_value(True)
    assert microwaves[0].states[Open].set_value(True)
    og.sim.step()
    values = [mw.states[HeatSourceOrSink].get_value() for mw in microwaves]
    assert values == [False, True, True], f"microwave open-door gate per env wrong: {values}"

    reset_thermals(env)


# ---------------------------------------------------------------------------
# 5) Index-map alignment diagnostics — the "why" for both eval bugs
# ---------------------------------------------------------------------------


def test_point_source_link_index_is_per_scene_n3(multi_env):
    """THE smoking-gun check for eval bug #2: for point sources the kernel
    resolves the heating-element pose via HeatSourceOrSink._link_flat_idx,
    which is (N_hss,) — one flat RigidBodyView index per source COLUMN. With
    S cloned scenes the element link is a different rigid body in each scene,
    so a single per-column index cannot be right for all envs. This test
    asserts the stored index matches every scene's own element link and
    prints the full mismatch table when it does not."""
    env = multi_env
    stoves = scene_objs(env, "stove")
    col = (HeatSourceOrSink.OBJ_IDXS or {}).get(stoves[0].relative_prim_path)
    assert col is not None, "stove not tracked by HeatSourceOrSink"
    assert HeatSourceOrSink._link_flat_idx is not None
    stored = wp.to_torch(HeatSourceOrSink._link_flat_idx).to("cpu")

    mismatches = []
    for s, stove in enumerate(stoves):
        link = stove.states[HeatSourceOrSink].link
        expected = RigidBodyViewAPI.get_flat_idx(link.prim_path)
        kernel_uses = stored[s, col].item() if stored.dim() == 2 else stored[col].item()
        print(f"env{s}: stove element link {link.prim_path} flat_idx={expected} (kernel uses {kernel_uses})")
        if expected != kernel_uses:
            mismatches.append((s, expected, kernel_uses))
    assert not mismatches, (
        f"Point-source link index is not per-scene (shape={tuple(stored.shape)}): scenes "
        f"{mismatches} (scene, expected, kernel_uses) resolve the wrong element pose in "
        f"_incoming_heat_kernel's distance check."
    )


def test_onfire_state_roundtrip_preserves_temperature_n3(multi_env):
    """THE root-cause regression test for eval bug #1 (popcorn bag at ~249 C after
    every reset): OnFire is a derived threshold state over Temperature, but it
    inherits the generic TensorizedAbsoluteState._load_state, which routes
    through _set_value. OnFire._set_value(False) writes
    Temperature = ignition_temperature - 1 (= 249 with the 250 default) — correct
    for a user-level "extinguish" call, catastrophic when replayed by
    scene dump/load: restoring a COLD flammable object heats it to 249 C.
    scene.reset() -> restore() -> load_state() does exactly this, so every
    flammable object in every scene starts every episode Heated. This test
    round-trips each env's flammable object through dump_state/load_state and
    requires its temperature to be unchanged."""
    env = multi_env
    reset_thermals(env)

    from omnigibson.object_states import OnFire

    failures = []
    for s, plywood in enumerate(scene_objs(env, "plywood")):
        assert OnFire in plywood.states, "plywood should be flammable (OnFire state)"
        assert plywood.states[Temperature].set_value(DEFAULT_TEMP)
        og.sim.step()
        assert not plywood.states[OnFire].get_value()
        t_before = plywood.states[Temperature].get_value()

        state = plywood.dump_state(serialized=False)
        plywood.load_state(state, serialized=False)

        t_after = plywood.states[Temperature].get_value()
        print(f"env{s}: plywood T before dump/load={t_before:.1f} after={t_after:.1f}")
        if abs(t_after - t_before) > 1e-3:
            failures.append(f"env{s}: {t_before:.1f} -> {t_after:.1f}")
    assert not failures, (
        "OnFire state round-trip changed a cold flammable object's temperature "
        f"(load_state routes through _set_value which clamps Temperature to ignition-1): {failures}"
    )

    reset_thermals(env)


def test_requires_inside_index_alignment_n3(multi_env):
    """Diagnostic for eval bug #1 (OFF oven heats far-away bag): validates the
    requires_inside lookup chain end to end. For every (containment source,
    target) pair, the kernel's inside_values[s, temp_to_inside_idx[n],
    hss_self_inside_idx[h]] read must equal the ground-truth per-env
    Inside.get_value(). Any mismatch is printed with names."""
    env = multi_env
    reset_thermals(env)
    for _ in range(3):
        og.sim.step()

    inside_vals = wp.to_torch(Inside.VALUES_WP).to("cpu") if Inside.VALUES_WP is not None else None
    assert inside_vals is not None, "Inside state has no tensorized values"
    hss_inside = wp.to_torch(Temperature._hss_self_inside_idx).to("cpu")
    temp_inside = wp.to_torch(Temperature._temp_to_inside_idx).to("cpu")

    mismatches = []
    for src_name in ("microwave", "fridge", "oven"):
        sources = scene_objs(env, src_name)
        h = (HeatSourceOrSink.OBJ_IDXS or {}).get(sources[0].relative_prim_path)
        if h is None or not sources[0].states[HeatSourceOrSink].requires_inside:
            continue
        for tgt_name in TARGET_NAMES:
            targets = scene_objs(env, tgt_name)
            n = (Temperature.OBJ_IDXS or {}).get(targets[0].relative_prim_path)
            if n is None:
                continue
            src_ii, tgt_ii = hss_inside[h].item(), temp_inside[n].item()
            for s in range(N_ENVS):
                truth = bool(targets[s].states[Inside].get_value(sources[s]))
                if src_ii < 0 or tgt_ii < 0:
                    kernel_view = False
                else:
                    kernel_view = bool(inside_vals[s, tgt_ii, src_ii].item())
                if truth != kernel_view:
                    mismatches.append(
                        f"env{s}: Inside({tgt_name}, {src_name}) truth={truth} but kernel reads "
                        f"inside_values[{s},{tgt_ii},{src_ii}]={kernel_view}"
                    )
    if mismatches:
        dump_heat_maps(env)
    assert not mismatches, "requires_inside index chain misaligned:\n" + "\n".join(mismatches)
