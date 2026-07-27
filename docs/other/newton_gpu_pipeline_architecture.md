# Newton GPU Pipeline Architecture

This document defines the execution architecture for GPU-vectorized
BEHAVIOR-1K environments on Newton. The goal is to keep the steady-state
simulation, control, task, observation, and rendering path on the GPU while
providing an explicitly reproducible execution mode.

Last updated: July 22, 2026.

## Decision

The Newton runtime will be GPU-first:

```text
batched actions on GPU
        -> batched controllers
        -> batched Newton physics and coupling
        -> batched object states and predicates
        -> batched rewards and termination
        -> batched observations and RGB
        -> policy tensors remain on GPU
```

CPU execution remains useful for debugging, validation, scene parsing, USD
import, initial model construction, logging, and dataset output. It is not the
target steady-state runtime for vector environments.

GPU execution provides parallelism, not automatic reproducibility. The runtime
must separately control contact ordering, reductions, random number generation,
and solver settings.

## Current State

Legacy OmniGibson can run physics on a GPU, but its surrounding architecture is
largely scalar and CPU-orchestrated:

- `omnigibson/envs/vec_env_base.py` constructs a Python list of environments
  and loops over actions, observations, rewards, termination, and reset.
- `omnigibson/envs/sb3_vec_env.py` similarly loops over environments and notes
  that task classes are not vectorized.
- Higher-level entity and controller code still contains `.cpu()`, `.numpy()`,
  `.tolist()`, scalar extraction, and per-entity Python loops.

The Newton migration already has a useful foundation: simulation state is held
in Warp arrays, and `wp.to_torch()` can expose compatible arrays to Torch
without a CPU copy. The remaining work is to make high-level runtime APIs
batched rather than wrapping device physics in scalar Python execution.

## Runtime Data Model

Use one world-indexed, structure-of-arrays representation rather than one
Python `Environment` object per simulated world. Exact layouts may be flattened
when environments contain different numbers of entities, but ownership must be
explicit and stable.

Representative contracts:

```python
actions          # [num_envs, action_dim]
observations     # [num_envs, observation_dim]
rewards          # [num_envs]
terminated       # [num_envs]
truncated        # [num_envs]

body_world       # [num_bodies]
joint_world      # [num_joints]
particle_world   # [particle_capacity]
particle_active  # [particle_capacity]
```

Every environment-local runtime operation must support a device-resident world
mask or environment index array:

```python
environment.reset(env_ids)
controller.compute(env_ids)
task.evaluate(env_ids)
renderer.render(env_ids)
solver.clear_persistent_state(env_ids)
```

Contacts, MPM grids, particles, random streams, coupling constraints, and
renderer history must never cross world boundaries.

## Hot-Path Rules

The normal step path must avoid:

- per-environment Python loops
- required `.cpu()`, `.numpy()`, or `.tolist()` conversions
- scalar reads that synchronize the GPU
- per-step topology changes or unbounded allocation
- CPU evaluation of common rewards, termination conditions, or predicates
- rebuilding model or solver objects during partial reset
- copying observations through host memory before policy inference

Allocate state and observation buffers once and reuse them. Keep topology fixed
inside an episode. Slicing, dicing, recipe outputs, and object replacement
should use preallocated activation pools. Particle systems should use fixed
capacities, active masks, and bounded emission.

CUDA graph capture is a desired optimization, not the first correctness target.
The data model must nevertheless avoid choices that make later capture
impossible, such as unbounded allocation and host-controlled per-world work.

## Reproducibility Modes

Expose reproducibility as an explicit execution policy:

```yaml
newton:
  execution:
    mode: throughput       # throughput or reproducible
```

### Throughput mode

- Use normal parallel atomics and reductions.
- Permit fast math where validated.
- Prioritize batching, graph capture, and total simulated steps per second.
- Require physical and statistical equivalence, not bitwise replay.

### Reproducible mode

- Enable deterministic Warp lowering where supported.
- Enable deterministic Newton contact ordering.
- Use stable environment-local sorting and reduction order.
- Use fixed time steps, substeps, iteration counts, and solver settings.
- Use counter-based random streams keyed by global seed, environment identity,
  episode count, and operation identity.
- Reset all persistent solver, coupling, contact, renderer, and RNG state for
  selected environments.
- Document and measure the performance cost.

Changing batch size or resetting one world must not change another world's
random sequence. Cross-device bitwise equality is not required. Regression
tests should use tolerances, invariants, and distributions because contact-rich
trajectories amplify small floating-point differences.

## Vectorized Subsystems

The following must eventually operate on batches:

1. Robot actions and controller calculations
2. Rigid, deformable, MPM, and particle physics state
3. Cross-solver contacts and coupling state
4. Partial reset and state serialization
5. Common object-state predicates and spatial queries
6. BDDL condition evaluation inputs
7. Rewards, termination, and transition triggers
8. Proprioceptive and camera observations
9. Object activation, visibility, and preallocated transition pools

Not every rare predicate needs a custom GPU kernel initially. A capability
fallback may exist during migration, but it must be measurable and must not
silently introduce a per-step device synchronization into otherwise vectorized
training.

## Implementation Order

1. Introduce stable world indexing in model, state, entities, and contacts.
2. Define device-native action, observation, reward, termination, and info
   contracts.
3. Implement masked partial reset, including solver and coupling history.
4. Replace the list-of-environments execution loop with a batched environment
   runtime while preserving the single-environment public API.
5. Batch robot state access and controllers; remove finite-difference and CPU
   control paths from the training hot path.
6. Vectorize common object-state predicates, task evaluation, rewards, and
   termination.
7. Integrate the replaceable batched renderer described in
   [Newton Renderer Architecture Handoff](newton_renderer_architecture.md).
8. Integrate deformable, MPM, particle, and coupled solvers using the same world
   ownership and reset contracts.
9. Port less common task features and optimize with graph capture after
   correctness and profiling.

World indexing, masked reset, and device contracts should precede broad feature
migration. Otherwise scalar task and object-state implementations will need to
be rewritten after vectorization.

## Acceptance Criteria

The initial GPU-vector runtime is complete when:

- A single batched runtime owns multiple Newton worlds; it does not create one
  complete Python environment runtime per world.
- Actions enter and observations, rewards, and done flags leave as
  device-resident tensors.
- The steady-state step has no required CPU readback or per-world Python loop.
- Selected worlds can reset without modifying or synchronizing unaffected
  worlds.
- Reset clears all public and persistent physics state for the selected worlds.
- Random streams are independent of reset order and batch size.
- Contacts and constraints cannot cross world boundaries.
- At least proprioception, common rigid predicates, rewards, and termination
  are batched.
- The renderer can return batched RGB without a CPU bridge.
- Throughput and reproducible modes are both covered by regression tests.
- Repeated runs in reproducible mode satisfy documented same-device guarantees.
- Performance is reported at representative batch sizes with physics,
  rendering, observation, and policy-transfer costs measured separately.

## Validation Policy

Do not use exact long-horizon state equality as the primary cross-device test.
Validate:

- absence of NaNs, divergence, and cross-world contamination
- bounded penetration and constraint error
- joint-limit and activation correctness
- conservation metrics appropriate to each material
- predicate, reward, and task-success agreement within defined tolerances
- statistical stability across seeds
- same-device reproducibility guarantees in reproducible mode
- throughput, reset throughput, GPU memory, and synchronization count

The production target is GPU-native execution with deterministic behavior when
requested. CPU execution is a reference and debugging path, not an architectural
dependency of the environment step.

