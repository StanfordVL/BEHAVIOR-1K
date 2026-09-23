"""Point heat sources must measure to the target's GEOMETRY, not to its AABB.

Pre-vectorization, `HeatSourceOrSink` decided which objects a point source affects with
`og.sim.psqi.overlap_sphere(radius=distance_threshold, ...)`, a PhysX query against each
target's actual COLLISION GEOMETRY. The vectorized kernel cannot call that (CPU-side query,
one call per source, delivered by callback), so it reproduces it in two stages: a cheap
AABB distance test as a broad phase, then `wp.mesh_query_point` against the target's collision
meshes for whatever survives.

The broad phase alone is not equivalent. An AABB always contains its geometry, so
distance-to-AABB <= distance-to-geometry, and the AABB test therefore ACCEPTS targets whose
real geometry is out of range. The gap is largest for long, thin or diagonally-oriented
objects: measured 0.69 m of slack for a bar lying along a box diagonal, against a default
threshold of 0.2 m.

None of the tests in test_multiple_envs_heat_states.py cover this — they all place targets
directly on the heat element, where AABB and mesh agree. These do.

Run standalone (one Environment per process):
  OMNIGIBSON_HEADLESS=1 pytest tests/test_multiple_envs_heat_narrow_phase.py -v -s
"""

import pytest
import torch as th

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.macros import macros as m
from omnigibson.object_states import HeatSourceOrSink, Temperature, ToggledOn

N_ENVS = 3
DEFAULT_TEMP = m.object_states.temperature.DEFAULT_TEMPERATURE

OBJECTS_CFG = [
    {"type": "DatasetObject", "name": "stove", "category": "stove", "model": "yhjzwg", "position": [0.0, 0.0, 0.5]},
    # Flat sheet: rotated off-axis its AABB balloons well beyond the geometry, which is exactly
    # the configuration where an AABB-only test over-reaches.
    #
    # Scaled up and made kinematic deliberately. At native size the sheet is 0.47 m long and the
    # furthest an in-AABB point can sit from it is 0.30 m -- exactly the margin this test wants,
    # with no room for error. Scaling 3x makes the gap comfortable instead of borderline.
    #
    # kinematic_only, NOT fixed_base: the board has to stay exactly where the search puts it.
    # keep_still() only zeroes velocity, so gravity still moves it; and fixed_base is worse than
    # nothing here, because it anchors a joint at the spawn pose and physics then drags the board
    # back toward that anchor after every teleport. Measured drift with fixed_base was
    # 0.682 -> 0.175 m over ten steps, i.e. into the threshold, which made the stove heat it
    # correctly and the test blame the kernel for the test's own instability.
    {
        "type": "DatasetObject",
        "name": "board",
        "category": "plywood",
        "model": "fkmkqa",
        "abilities": {"cookable": {}, "heatable": {}},
        "scale": [3.0, 3.0, 3.0],
        "kinematic_only": True,
        "position": [15.0, 0.0, 0.1],
    },
    # Compact control target: AABB and geometry nearly coincide, so it must still heat.
    {
        "type": "DatasetObject",
        "name": "bagel",
        "category": "bagel",
        "model": "zlxkry",
        "abilities": {"cookable": {}, "freezable": {}, "burnable": {}, "heatable": {}},
        "position": [15.0, 2.0, 0.1],
    },
]


@pytest.fixture(scope="module")
def multi_env():
    assert og.sim is None, "This module must run in a fresh process (one Environment per process)."
    gm.RENDER_VIEWER_CAMERA = False
    gm.ENABLE_OBJECT_STATES = True
    gm.USE_GPU_DYNAMICS = True
    gm.ENABLE_FLATCACHE = False
    gm.ENABLE_TRANSITION_RULES = False

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


def scene_objs(env, name):
    return [env.scenes[s].object_registry("name", name) for s in range(N_ENVS)]


def heat_link_world_pos(source):
    link = source.states[HeatSourceOrSink].link
    return link.get_position_orientation()[0]


def threshold_of(source):
    return source.states[HeatSourceOrSink].distance_threshold


def aabb_distance(obj, point):
    """Distance from @point to the closest point of @obj's world AABB (what the broad phase uses)."""
    lo, hi = obj.aabb
    return float(th.linalg.norm(point - th.clamp(point, lo, hi)))


def geometry_distance(obj, point):
    """Distance from @point to the nearest collision-hull vertex of @obj.

    An upper bound on the true surface distance (a face interior can be nearer than any vertex),
    so the tests below only rely on it together with a generous margin.
    """
    best = float("inf")
    for link in obj.links.values():
        pts = link.collision_boundary_points_world
        if pts is None or len(pts) == 0:
            continue
        best = min(best, float(th.linalg.norm(pts - point.reshape(1, 3), dim=-1).min()))
    return best


def reset_thermals(env):
    for s in range(N_ENVS):
        stove = env.scenes[s].object_registry("name", "stove")
        if ToggledOn in stove.states:
            stove.states[ToggledOn].set_value(False)
        for name in ("board", "bagel"):
            env.scenes[s].object_registry("name", name).states[Temperature].set_value(DEFAULT_TEMP)
    og.sim.step()


def place_board_so_aabb_reaches_but_geometry_does_not(board, element, threshold):
    """Find a pose where the element is inside the board's AABB but far from the board itself.

    A rotated flat sheet leaves most of its bounding box empty, so the interesting spots are the
    AABB's own CORNERS: inside the box by construction, yet far from the sheet. For each candidate
    orientation the board is shifted so a chosen corner (pulled slightly inward) lands on the heat
    element, then the achieved distances are measured rather than assumed.

    Searching real corners instead of guessing offsets also keeps the test asset-agnostic: it does
    not depend on the sheet's dimensions, only on its bounding box being much larger than itself.

    Returns the achieved (aabb_dist, geom_dist), or None if no pose qualified.
    """
    orientations = [
        th.tensor([0.3826834, 0.0, 0.0, 0.9238795]),  # 45 deg about X
        th.tensor([0.0, 0.3826834, 0.0, 0.9238795]),  # 45 deg about Y
        th.tensor([0.5, 0.5, 0.5, 0.5]),  # 120 deg about (1,1,1)
    ]
    for quat in orientations:
        board.keep_still()
        board.set_position_orientation(position=element, orientation=quat)
        og.sim.step()
        # AABB is only meaningful once the new orientation has been applied.
        lo, hi = board.aabb
        centre = 0.5 * (lo + hi)
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    sign = th.tensor([sx, sy, sz])
                    # 0.8 keeps the target point inside the box rather than exactly on its surface.
                    corner = centre + 0.8 * sign * (hi - lo) * 0.5
                    board.keep_still()
                    board.set_position_orientation(
                        position=board.get_position_orientation()[0] + (element - corner), orientation=quat
                    )
                    og.sim.step()
                    a, g = aabb_distance(board, element), geometry_distance(board, element)
                    if a <= 0.5 * threshold and g >= 1.5 * threshold:
                        return a, g
    return None


def test_point_source_ignores_target_whose_aabb_reaches_but_geometry_does_not_n3(multi_env):
    """A toggled-on stove must NOT heat a board that only its AABB brings into range.

    This is the case the AABB-only broad phase gets wrong: the element sits inside the rotated
    board's bounding box while the board itself is several thresholds away. overlap_sphere would
    report no contact, so the kernel must not heat it.
    """
    env = multi_env
    reset_thermals(env)

    stoves, boards = scene_objs(env, "stove"), scene_objs(env, "board")
    achieved = []
    for stove, board in zip(stoves, boards):
        element = heat_link_world_pos(stove)
        got = place_board_so_aabb_reaches_but_geometry_does_not(board, element, threshold_of(stove))
        assert got is not None, (
            "Could not place the board so that its AABB reaches the heat element while its "
            "geometry stays out of range — the asset's dimensions may have changed. Adjust the "
            "sweep in place_board_so_aabb_reaches_but_geometry_does_not; this test is meaningless "
            "without that configuration."
        )
        achieved.append(got)

    for stove in stoves:
        assert stove.states[ToggledOn].set_value(True)
    for _ in range(10):
        og.sim.step()

    # Re-measure AFTER the heating steps. The distances above were taken before the sim ran, so a
    # board that drifted (or was nudged by a collision) would make the verdict unreadable — the
    # kernel would be judged against a geometry that no longer matches what it saw.
    for s, (a, g) in enumerate(achieved):
        element = heat_link_world_pos(stoves[s])
        a2, g2 = aabb_distance(boards[s], element), geometry_distance(boards[s], element)
        print(
            f"\nenv{s}: threshold {threshold_of(stoves[s]):.3f} m"
            f"\n   before heating: element->AABB {a:.3f}, element->geometry {g:.3f}"
            f"\n   after  heating: element->AABB {a2:.3f}, element->geometry {g2:.3f}"
        )
        assert g2 >= 1.5 * threshold_of(stoves[s]), (
            f"env{s}: the board moved during the heating steps (geometry distance {g:.3f} -> "
            f"{g2:.3f} m), so this episode no longer tests the AABB-vs-geometry gap. Keep the "
            f"board static instead of tightening the assertion below."
        )
    print("board temps:", [round(b.states[Temperature].get_value(), 2) for b in boards])

    heated = [
        f"env{s}: {b.states[Temperature].get_value():.2f} C"
        for s, b in enumerate(boards)
        if b.states[Temperature].get_value() > DEFAULT_TEMP + 1e-3
    ]
    assert not heated, (
        "Stove heated a board whose collision geometry is out of range and only whose AABB is "
        f"within the threshold — the narrow phase is not running: {heated}"
    )
    reset_thermals(env)


def test_point_source_still_heats_a_target_on_the_element_n3(multi_env):
    """Control: the narrow phase must not reject a target that genuinely IS in range.

    Guards the opposite failure — a too-strict mesh test would stop food cooking on a lit
    burner, which is the regression the AABB-center bug originally caused.
    """
    env = multi_env
    reset_thermals(env)

    stoves, bagels = scene_objs(env, "stove"), scene_objs(env, "bagel")
    for stove, bagel in zip(stoves, bagels):
        bagel.keep_still()
        bagel.set_position_orientation(position=heat_link_world_pos(stove) + th.tensor([0.0, 0.0, 0.05]))
    og.sim.step()
    for stove in stoves:
        assert stove.states[ToggledOn].set_value(True)
    for _ in range(10):
        og.sim.step()

    temps = [round(b.states[Temperature].get_value(), 2) for b in bagels]
    print("\nbagel temps on the element:", temps)
    cold = [f"env{s}: {t}" for s, t in enumerate(temps) if t <= DEFAULT_TEMP + 1e-3]
    assert not cold, f"Narrow phase rejected a target sitting on the heat element: {cold}"
    reset_thermals(env)
