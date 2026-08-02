"""VBD cloth self-contact viability tests for the Newton runtime.

BEHAVIOR cloth activities (folding towels, making beds) require cloth that does
not pass through itself. This is the gating evidence for the deformable solver
decision recorded in docs/other/newton_solver_strategy.md: VBD is the only
solver in the pinned build that supports cloth self-contact (Style3D exposes
none, XPBD is not used for deformables upstream), and IPC is unavailable, so
these tests establish whether VBD's mollified-barrier self-contact is adequate
before any IPC-class integration is considered.

Each scenario runs twice, with ``particle_enable_self_contact`` on and off. The
off run is the control: if self-contact is doing real work, the layer
separation must collapse without it.

Run inside the ``newton-b1k`` conda environment:

    conda run -n newton-b1k python -m pytest tests/test_newton_cloth.py -v -s

Subprocess-per-scenario for the same reasons as test_newton_smoke.py.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

SCENARIO_OK = "NEWTON_CLOTH_SCENARIO_OK"

# Cloth roughly the size and resolution of a BEHAVIOR dishtowel.
CLOTH_SIZE = 0.4
CLOTH_DIM = 32
PARTICLE_RADIUS = 0.002
SELF_CONTACT_RADIUS = 0.002
SELF_CONTACT_MARGIN = 0.003

# Particle pairs this far apart in the rest shape are on different layers when
# they come close at runtime, so their live distance measures layer separation.
REST_FAR_FACTOR = 6.0


def _run_scenario(name, timeout=900):
    env = dict(os.environ)
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
        + result.stdout.splitlines()[-25:]
        + ["--- stderr tail ---"]
        + result.stderr.splitlines()[-25:]
    )
    assert result.returncode == 0, f"Scenario {name!r} exited with {result.returncode}\n{tail}"
    assert f"{SCENARIO_OK} {name}" in result.stdout, f"Scenario {name!r} missing success marker\n{tail}"
    payload = next(line for line in result.stdout.splitlines() if line.startswith("RESULT_JSON "))
    return json.loads(payload[len("RESULT_JSON ") :])


def _assert_self_contact_works(report, scenario):
    on, off = report["on"], report["off"]

    assert on["finite"], f"{scenario}: cloth state went non-finite with self-contact enabled"

    # Control first: the scenario must actually drive layers into each other,
    # otherwise a clean self-contact result proves nothing. Without self-contact
    # the layers should interpenetrate freely.
    assert off["peak_violations"] > 0, (
        f"{scenario}: no interpenetration even with self-contact disabled - "
        "the scenario is not exercising self-contact and the result is vacuous"
    )

    # With self-contact on, layers must not interpenetrate.
    assert on["peak_violations"] == 0, (
        f"{scenario}: {on['peak_violations']} interpenetrating particle pairs with self-contact on "
        f"(closer than {VIOLATION_FRACTION * SELF_CONTACT_RADIUS * 1000:.1f} mm); "
        f"minimum layer gap reached {on['min_layer_gap_m'] * 1000:.2f} mm"
    )


def test_cloth_drape_self_contact():
    """Cloth folded over a rod: the two hanging halves must not pass through each other."""
    report = _run_scenario("drape")
    _assert_self_contact_works(report, "drape")


def test_cloth_crumple_self_contact():
    """Cloth dropped onto a pedestal: dense multi-layer self-contact must hold."""
    report = _run_scenario("crumple")
    _assert_self_contact_works(report, "crumple")


# --- Scenario implementations. Everything below runs in a subprocess. ---


def _build(scenario, self_contact):
    import newton
    import warp as wp

    builder = newton.ModelBuilder()
    builder.default_particle_radius = PARTICLE_RADIUS

    cell = CLOTH_SIZE / CLOTH_DIM
    if scenario == "drape":
        # Cloth centered above a thin horizontal rod so gravity folds it in half
        # and the two halves hang down against each other.
        cloth_height = 0.30
        builder.add_cloth_grid(
            pos=wp.vec3(-CLOTH_SIZE / 2, -CLOTH_SIZE / 2, cloth_height),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=CLOTH_DIM,
            dim_y=CLOTH_DIM,
            cell_x=cell,
            cell_y=cell,
            mass=0.1 / (CLOTH_DIM * CLOTH_DIM),
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=1.0e-1,
            edge_ke=1.0e-1,  # low bending stiffness: fabric, not card stock
            particle_radius=PARTICLE_RADIUS,
        )
        # Rod along y, thin enough that the cloth folds sharply over it.
        builder.add_shape_capsule(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 0.0, cloth_height - 0.01), wp.quat_identity()),
            radius=0.008,
            half_height=CLOTH_SIZE,
        )
    else:
        # Cloth dropped centered onto a thin vertical post. All four quadrants
        # fold down around a 2 cm post and press against each other, which is a
        # much denser multi-layer self-contact case than the two-layer drape.
        cloth_height = 0.32
        builder.add_cloth_grid(
            pos=wp.vec3(-CLOTH_SIZE / 2, -CLOTH_SIZE / 2, cloth_height),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=CLOTH_DIM,
            dim_y=CLOTH_DIM,
            cell_x=cell,
            cell_y=cell,
            mass=0.1 / (CLOTH_DIM * CLOTH_DIM),
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=1.0e-1,
            edge_ke=1.0e-1,
            particle_radius=PARTICLE_RADIUS,
        )
        builder.add_shape_capsule(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.14), wp.quat_identity()),
            radius=0.01,
            half_height=0.14,
            cfg=newton.ModelBuilder.ShapeConfig(),
        )

    builder.add_ground_plane()
    # VBD parallelizes by graph coloring, which must be computed before finalize.
    builder.color(include_bending=True)
    model = builder.finalize()
    model.soft_contact_mu = 0.3

    solver = newton.solvers.SolverVBD(
        model=model,
        iterations=10,
        particle_enable_self_contact=self_contact,
        particle_self_contact_radius=SELF_CONTACT_RADIUS,
        particle_self_contact_margin=SELF_CONTACT_MARGIN,
        particle_topological_contact_filter_threshold=2,
        rigid_body_particle_contact_buffer_size=1024,
    )
    return model, solver


# Layers closer than this fraction of the contact radius are interpenetrating:
# working self-contact should hold them apart at roughly the radius.
VIOLATION_FRACTION = 0.5


def _layer_stats(live_q, rest_far):
    """Measure interpenetration between cloth layers at one instant.

    Particle pairs far apart in the rest shape can only be close at runtime by
    lying on different layers of the folded cloth, so their live distance is
    layer separation. Counting pairs closer than a fraction of the contact
    radius is far more robust than a single minimum distance: layers that pass
    fully through each other separate again afterwards, so a final-frame
    minimum misses the penetration entirely, while the violation count spikes
    while it is happening.
    """
    import torch as th

    dist = th.cdist(live_q, live_q).masked_fill(~rest_far, float("inf"))
    violations = int((dist < VIOLATION_FRACTION * SELF_CONTACT_RADIUS).sum())
    return float(dist.min()), violations


def _run(scenario, self_contact, frames=180, substeps=20):
    import statistics
    import time

    import newton
    import torch as th
    import warp as wp

    model, solver = _build(scenario, self_contact)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(model, broad_phase="nxn", soft_contact_margin=0.005)
    contacts = collision_pipeline.contacts()

    rest_q = wp.to_torch(state_0.particle_q).clone()
    rest_far = th.cdist(rest_q, rest_q) > REST_FAR_FACTOR * (CLOTH_SIZE / CLOTH_DIM)

    frame_dt = 1.0 / 60.0
    sim_dt = frame_dt / substeps
    times = []
    min_gap_ever = float("inf")
    peak_violations = 0
    finite = True
    for _ in range(frames):
        t0 = time.perf_counter()
        for _ in range(substeps):
            state_0.clear_forces()
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0
        wp.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

        live_q = wp.to_torch(state_0.particle_q)
        if not bool(th.isfinite(live_q).all()):
            finite = False
            break
        gap, violations = _layer_stats(live_q, rest_far)
        min_gap_ever = min(min_gap_ever, gap)
        peak_violations = max(peak_violations, violations)

    return {
        "self_contact": self_contact,
        "min_layer_gap_m": min_gap_ever if finite else 0.0,
        "peak_violations": peak_violations,
        "median_ms": statistics.median(times),
        "finite": finite,
        "particles": int(model.particle_count),
    }


def scenario(name):
    report = {}
    for key, enabled in (("on", True), ("off", False)):
        r = _run(name, enabled)
        report[key] = r
        print(
            f"MEASURED {name} self_contact={'ON ' if enabled else 'OFF'} "
            f"min_gap={r['min_layer_gap_m'] * 1000:7.3f} mm  "
            f"interpenetrating_pairs={r['peak_violations']:6d}  "
            f"{r['median_ms']:7.2f} ms/frame  finite={r['finite']}  particles={r['particles']}",
            flush=True,
        )
    overhead = report["on"]["median_ms"] / report["off"]["median_ms"]
    print(f"MEASURED {name} self_contact_overhead={overhead:.2f}x", flush=True)
    print("RESULT_JSON " + json.dumps(report), flush=True)
    return report


if __name__ == "__main__":
    scenario_name = sys.argv[1]
    scenario(scenario_name)
    print(f"{SCENARIO_OK} {scenario_name}", flush=True)
