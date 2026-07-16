# Particle-System Vectorization — Future Work

**Branch context:** `vec/particle-system`

This document contains proposals and follow-up work that are not part of the
implemented particle-state vectorization described in
`PARTICLE_VECTORIZATION_DESIGN_REVIEW.md`. Keeping this material separate makes
the design review an accurate description of the current PR.

## 1. Correctness baseline before modifier vectorization

Particle modifiers are the remaining particle states that mutate the world:

- `ParticleApplier`: spray adds stain; shaker adds salt.
- `ParticleRemover`: sponge or vacuum removes particles.
- `ParticleSource`: faucet spout streams particles.
- `ParticleSink`: drain removes particles.

Before changing their execution model, preserve and test these existing semantics:

1. One object can source and sink the same system. Modifier identity must therefore
   include the concrete state type; `(object, system)` is not sufficient.
2. `_current_step` belongs to each state instance. Instances advance from the same
   simulator clock, but their serialized counters are independent.
3. The skipped issue-2066 modifier cases need to be split into
   `{projection, adjacency} × {applier, remover}` and characterized individually.

Known correctness issues to address independently:

- Visual-applier group accounting currently overwrites earlier group counts with the
  final group's count instead of accumulating all groups.
- Mutation budgeting can use a modifier's default limit instead of the effective
  `Saturated` limit.
- `ParticleSink` does not have a per-step removal cap.

Tests proposed for this baseline:

- Serialize and restore at least two modifier instances with different
  `_current_step` values.
- Strengthen the multi-scene source test: enable one scene, assert only that scene
  changes, enable a second scene later, and verify independent system instances and
  cleanup.
- Turn the monolithic skipped modifier tests into focused geometry/timing tests.

### Separate modifier budgets from object saturation

This is a future semantics decision and is not part of the current vectorization
implementation.

One object can have two independent modifiers for the same system. A kitchen sink,
for example, can both create and remove water:

```text
kitchen_sink
    ParticleSource(water)  -- faucet creates water
    ParticleSink(water)    -- drain removes water
```

The existing `ModifiedParticles` / `Saturated` state is keyed only by
`(object, system)`, so both modifiers address the same `(kitchen_sink, water)` value.
That single value cannot unambiguously answer both of these questions:

1. How many particles has each modifier processed toward its own limit?
2. Is the object semantically saturated with this material?

For example, after the faucet creates 20 particles and the drain removes 5, a shared
count could mean 25 total operations, 15 net additions, or 5 absorbed particles.
Each interpretation changes behavior.

The proposed representation separates the two meanings:

```text
Operational modifier budgets
    (kitchen_sink.ParticleSource, water) -> created 20 of its limit
    (kitchen_sink.ParticleSink, water)   -> removed 5 of its limit

Semantic object state
    (kitchen_sink, water) -> ModifiedParticles / Saturated
```

Operational counts prevent a source from consuming a sink's budget, or vice versa.
The object-level state remains available for task semantics, but the team must decide
which operations affect it. A likely rule is that source-created particles update
only the source's budget, while particles absorbed by a remover may contribute to
the object's saturation. Other interpretations are possible.

This requires team sign-off because it can change modifier limits, `Saturated`
results, saved state, transition rules, and task outcomes. It should therefore be a
separate behavior change rather than being hidden inside performance/vectorization
work.

## 2. Proposed ParticleModifierManager

Retain modifier object states as compatibility façades, but remove
`UpdateStateMixin` from the individual modifier states. Leaving the mixin on each
state would preserve the current simulator loop over every modifier object and would
therefore preserve the main Python bottleneck.

The simulator should instead invoke one global manager after the current tensorized
particle/state snapshot has been refreshed and before transition rules run:

```text
Simulator._non_physics_step()
    -> _refresh_state_caches()
    -> ParticleModifierManager.update()   one call for all scenes/modifiers
    -> object visual updates
    -> transition rules
```

```text
obj.states[Source / Sink / Applier / Remover]
    declaration, static configuration, public API, serialization
    NO per-object UpdateStateMixin._update()
                         |
                         v
ParticleModifierManager
    registry + clocks + conditions + planning + conflict resolution
                         |
                         v
batched GPU planning kernels
                         |
                         v
grouped host mutation commit
    one bulk call per affected (scene, system / visual attachment group)
                         |
                         v
one dirty mark per affected system, one global PhysX flush,
one handle refresh per affected macro system
at most one ParticleViewAPI/state refresh after the complete commit
```

The state façade would retain:

- presence in `obj.states` and compatibility with factories/actions/demos;
- modifier method, condition specification, meta-link, and limits;
- public query methods such as `supports_system()`,
  `check_conditions_for_system()`, `n_steps_per_modification`, and
  `projection_is_active`;
- serialization of its manager-owned runtime row.
- no `UpdateStateMixin` inheritance and no per-step `_update()` implementation.

The manager would own:

- the single simulator update hook and throttle advancement for all modifier rows;
- iteration over active modifier/system jobs;
- batched built-in condition evaluation;
- ray generation and geometry queries;
- add/remove request construction;
- deterministic conflict resolution and grouped mutation commit;
- progress accounting from successful mutations;
- metadata invalidation and post-mutation refresh coordination.

### Proposed data layout

Use two flattened tables because objects do not support identical systems:

- Modifier table: scene, concrete state type, object/link index, method, period,
  counter, present/enabled mask.
- Job table: modifier index, ParticleViewAPI entry index, operation, progress,
  limit, and condition range.

Identity must include at least
`(scene, object relative path, concrete state type)`, allowing one object to own
both a source and sink for the same system.

### Proposed per-step execution

Split execution into planning and commit so every modifier reads the same
pre-mutation snapshot:

```text
PHASE A: batched planning, no mutations
    read ParticleViewAPI snapshot
    advance all clocks -> due mask
    evaluate built-in conditions
    run geometry kernels
    compact add/remove commands

PHASE B: conflict resolution
    deduplicate removal indices
    enforce per-modifier budgets
    group commands by destination system / visual attachment group
    reserve output and ParticleViewAPI capacities

PHASE C: grouped host commit
    bulk remove/add once per affected destination
    update counters from successful operations
    mark metadata dirty once per affected system
    flush all committed USD changes into PhysX once
    refresh each affected macro system's particle handles once
    perform at most one same-step ParticleViewAPI/state refresh
```

Snapshot semantics should be explicit: every modifier plans from the same
pre-mutation snapshot; particles added in the commit cannot be removed until the
next modifier stage.

### What can be batched with current system APIs

| Family | Current mutation API | Manager commit |
|---|---|---|
| micro-physical | `generate_particles(positions=N×3)` and `remove_particles(idxs=N)` already operate on arrays | Merge all requests for one `(scene, system)`, deduplicate removals, then issue one remove and one generate call. |
| macro-physical | Public APIs accept arrays, but internally loop over rigid-body prim creation/removal | Group into one call per `(scene, system)`, then add a true bulk/transaction path that defers dirty marking and `particles_sim_view` refresh until all prim operations finish; the manager flushes PhysX once after all affected systems commit. |
| macro-visual | `generate_group_particles()` accepts arrays, but internally loops over attached prims | Group by `(scene, system, attachment group)` and add a bulk/transaction path that defers bookkeeping and topology refresh until the group/system commit finishes. |

Separate scenes own separate system/instancer instances, so one backend call across
all scenes is not currently available. The practical target is O(affected systems
and visual attachment groups) host calls, rather than O(modifier objects × candidate
particles) calls.

For macro systems, use either explicit `generate_particles_batch()` /
`remove_particles_batch()` methods or a mutation transaction with this contract:

```text
begin bulk mutation
    create/remove all requested prims
    update internal particle/group dictionaries
end bulk mutation
    mark metadata dirty once
after every affected system finishes its stage edits
    manager flushes USD changes into PhysX once globally
    recreate each affected system's dedicated particle views once
```

This is particularly important for macro-physical generation: the current
`generate_particles()` loops through `add_particle()`, and each `add_particle()` can
refresh the dedicated PhysX view. A bulk path must defer that refresh until the full
array has been added.

### Conditions and unavoidable host work

Compile built-in conditions such as toggled-on, saturated, gravity, system-nonempty,
and overlap into the fast path. Keep custom `FUNCTION` conditions as a warned slow
fallback; current runtime configuration does not depend on them.

The planning and selection work can be cross-scene GPU work, but the mutation commit
cannot currently be fully GPU-only:

- micro systems expose per-scene instancers;
- macro systems create and remove USD/rigid-body prims;
- macro-visual systems create attachment-group prims;
- `raytest_batch` still performs a Python loop per ray.

The initial design should therefore use GPU-batched planning plus a grouped host
commit. Internal macro prim loops may remain initially, but repeated per-modifier
condition work, invalidation, PhysX flushes, and handle refreshes should not. Measure
planning, commit, and ray-cast costs separately before assuming which part provides
the dominant speedup. Replacing PhysX Python ray loops with Warp mesh-ray queries may
be the larger applier improvement.

Reserve mutation buffers from the maximum additions due in a step and grow them
geometrically; do not introduce a fixed particle bound.

## 3. Shared geometry and containment cleanup

Independent follow-up work:

- Extract the duplicated fillable-face table and volume kernels in `inside.py`,
  `contains.py`, and `contact_particles.py` into a shared container-hull module.
  They share a lot of common log now.
- Add the AABB prefilter used by `inside.py` to the containment kernel.
- Vectorize the lazy `ContainedParticles.positions` / `.in_volume` path so rare
  detailed consumers do not recompute per object on CPU.
- Add N=1/2/4/8 benchmarks against the pre-vectorization baseline.

## 4. Recommended sequence

1. Land the correctness baseline and characterize issue 2066.
2. Land shared-geometry cleanup and benchmarks.
3. Use those measurements as the ROI gate for ParticleModifierManager.
4. If approved, implement the registry/scheduler while delegating to existing
   mutation methods, then projection sources/sinks, projection/adjacency removers,
   and finally ray-cast appliers plus visual attachment grouping.

Open design questions:

- global versus per-scene ownership of modifier rows;
- exact placement of the single mutation/flush stage in the simulator step;
- interaction with graph capture and same-step state visibility;
- determinism when schedules diverge across environments;
- final ownership rules for modifier progress and semantic saturation.
