# Newton Solver and Coupling Strategy

This document records the solver conclusions and open questions for the
BEHAVIOR-1K Newton migration. It intentionally does not compare simulation
backends or revisit the decision to use Newton. Its scope is the physics
architecture inside Newton: which solver families are needed, how they can be
coupled, and how GPU-vectorized execution changes the design.

Last updated: July 16, 2026.

## Executive Summary

BEHAVIOR-1K cannot be served well by one universal solver. The workload spans
articulated robots, rigid household objects, cloth, ropes, volumetric soft
bodies, liquids, granular materials, and large numbers of particles. These
materials have different state representations, integration schemes, contact
requirements, and performance characteristics.

The working Newton architecture should therefore be a collection of
specialized solvers over a shared model and state, with explicit ownership and
coupling:

| Material or subsystem | Starting solver direction |
| --- | --- |
| Robots, articulations, and rigid objects | MuJoCo-Warp initially; benchmark Kamino as it matures |
| Cloth, rope, and deformable surfaces | VBD; keep XPBD as the faster approximation |
| Volumetric soft bodies | VBD for mesh-based elasticity; Implicit MPM for very large deformation or plasticity |
| Liquids and viscous substances | Implicit MPM, initially at task-dependent resolution |
| Granular materials and continuum-like particles | Implicit MPM; use simpler particles when continuum accuracy is unnecessary |
| Simple discrete particles | XPBD or Semi-Implicit, depending on the required constraints |
| Temperature, cooking, and phase state | Lumped thermal/state model first; spatial thermal solves only where tasks require them |
| Surface coatings, dirt, and wetness | Surface occupancy or attached particles; avoid full fluid simulation unless transport matters |

For bring-up and throughput, use proxy coupling between the smallest possible
set of solver pairs. For stronger two-way consistency, evaluate ADMM coupling.
Both coupling implementations in the pinned Newton build are experimental and
must be validated at BEHAVIOR scale before they become runtime dependencies.

IPC is not a universal multiphysics solver. It is a high-quality contact method
that is especially valuable for cloth and deformable contact. Newton does not
currently provide an IPC solver or coupler. Adding one remains a possible
future project, but it does not replace the need for rigid, deformable, and
continuum solvers or for coupling among them.

## Why BEHAVIOR-1K Needs Multiple Solvers

The BDDL data makes multiphysics a core requirement rather than an edge case:

- There are 1,016 activity definition files.
- The propagated taxonomy contains 97 cloth synsets, 97 soft-body synsets, and
  6 rope synsets.
- It also contains 944 substance synsets, including 891 physical substances
  and 273 liquids.
- The substance parameter table contains 996 modeled substances: 308 fluids,
  142 granular substances, 507 macro physical particle substances, and 39
  macro visual particle substances.
- Tasks also rely on non-mechanical state transitions such as cooking,
  melting, mixing, saturation, slicing, dicing, and particle application or
  removal.

These counts come from `bddl3/bddl/activity_definitions/`,
`bddl3/bddl/generated_data/propagated_annots_canonical.json`, and
`bddl3/bddl/generated_data/substance_hyperparams.csv`.

The final category is important: no numerical physics solver should own the
meaning of `Cooked`, `Saturated`, `Covered`, or a slicing recipe. BDDL predicates,
object states, and transition rules remain a semantic layer above physics. A
solver supplies quantities such as pose, deformation, contact, temperature,
or particle occupancy; OmniGibson decides how those quantities change symbolic
state and when an object is replaced or activated.

## Solver Families and Their Roles

### Rigid and articulated dynamics

Robots, doors, drawers, tools, containers, and most household objects require
stable articulated dynamics, joint limits, drives, frictional contact, and
reliable control behavior. MuJoCo-Warp is the current migration baseline and
should remain the rigid reference until another Newton rigid solver passes the
same import, control, stability, and contact tests.

Rigid-solver quality must be evaluated on manipulation, not just free-fall or
locomotion. Important cases include grasp closure, high-friction sliding,
cabinet articulation under contact, stacks of small objects, and rigid contact
against deformable or particle materials.

### XPBD

Extended Position-Based Dynamics is useful for fast constraint-based cloth,
ropes, soft objects, and discrete particles. It is attractive for GPU batching
because the state is simple and the iteration budget can be bounded.

Its tradeoff is that compliance, damping, and iteration count are part of the
observed material behavior. Low iteration counts can produce excessive stretch
or volume loss, while high counts and small substeps erode the speed advantage.
XPBD is therefore the performance-oriented approximation, not an automatic
maximum-fidelity choice.

### VBD

Vertex Block Descent solves deformable dynamics through an implicit
optimization. It is the leading Newton candidate for cloth, cables, and
mesh-based soft bodies when stiffness, contact stability, and reduced stretch
matter more than the lowest step cost.

VBD should be the default deformable evaluation path, with XPBD retained as a
speed baseline. Self-contact, thin-shell collision, friction, grasping, and
highly folded cloth need dedicated validation; a stable unconstrained cloth
drop is not sufficient evidence for manipulation tasks.

Self-contact validation, August 2, 2026 (`OmniGibson/tests/test_newton_cloth.py`,
RTX 5090, 0.4 m cloth at 32x32, 1089 particles, 2 mm contact radius, 20
substeps x 10 iterations, 3 s settle):

| scenario | self-contact | min layer gap | interpenetrating pairs | ms/frame |
| --- | --- | --- | --- | --- |
| drape over rod | on | 1.95 mm | 0 | 79.8 |
| drape over rod | off | 0.31 mm | 4 | 55.2 |
| fold around post | on | 1.99 mm | 0 | 78.0 |
| fold around post | off | 0.15 mm | 16 | 54.6 |

VBD self-contact holds: layers settle at essentially the authored contact
radius (1.95-1.99 mm against a 2 mm radius) with zero interpenetrating pairs,
in both a two-layer drape and a denser four-quadrant fold. The self-contact
disabled rows are the control; they interpenetrate, which is what makes the
enabled result meaningful rather than vacuous.

Cost: self-contact adds about 1.45x. The absolute figure is the open concern —
about 79 ms/frame for a single 1089-particle cloth is roughly 12 fps for one
cloth in one environment, which does not yet fit a vectorized training budget.
Substep count, solver iterations, and the O(n^2) `nxn` broad phase are the
untested levers; the measurement above deliberately reuses the stiff
poker-card configuration from Newton's own example and is not tuned for
fabric. Grasping with an actuated gripper remains unvalidated.

This result is what makes IPC unnecessary for now: the reason to want IPC was
guaranteed non-penetration under folding, and VBD's mollified barrier already
delivers it in these cases. See the decision log entry on deformable solver
selection.

### Implicit MPM

The Material Point Method is the broadest current Newton representation for
continuum materials. It is relevant to viscous fluids, granular materials,
plastic materials, and soft bodies undergoing deformations that are awkward
for a persistent mesh.

MPM cost and behavior depend heavily on particle resolution, grid resolution,
transfer scheme, constitutive model, boundary handling, and solve tolerance.
It can cover several B1K material categories, but one parameter set cannot
represent water, dough, sand, and foam. Each material class needs calibrated
constitutive parameters and task-level validation.

### Semi-implicit particles

A semi-implicit particle solver is useful when B1K needs many discrete
particles but does not need a continuum material model. It is a possible
fast path for visual or weakly interacting substances. It should not be used
to claim liquid or granular fidelity without validation of the task-relevant
behavior.

### Thermal state, phase changes, and surface substances

Most B1K cooking and temperature predicates do not require solving a spatial
heat equation. A lumped thermal model can exchange heat among objects,
particles, appliances, and the environment while transition rules interpret
temperature and exposure over time. This is substantially cheaper and easier
to vectorize than thermomechanical FEM or MPM.

Spatial conduction, convection, or phase-change mechanics should be introduced
only for tasks whose outcome depends on temperature gradients or material flow.
Likewise, wetness, dirt, stains, and coatings can often be represented by
surface occupancy fields or attached particles. They require a full liquid
solver only when pouring, runoff, absorption, or displacement is relevant to
the task. This hybrid modeling choice is part of the fidelity definition, not
merely an optimization.

### IPC

Incremental Potential Contact is a barrier-based contact formulation designed
to avoid intersections during optimization. Its strongest value for B1K would
be robust cloth self-contact and rigid-deformable contact under severe folding,
grasping, and compression.

IPC does not by itself provide articulated rigid dynamics, a fluid model, or a
granular constitutive model. It normally augments a deformable solver and must
still exchange forces and motion with the other subsystems. It can also be
expensive: continuous collision detection, barrier evaluation, and global
nonlinear solves are difficult to scale across hundreds or thousands of rich
vector environments.

The pinned Newton build has no IPC implementation. Newton's VBD contact work
may cover some of the same use cases, but it should not be described as IPC or
assumed to have identical guarantees.

## Coupling in the Pinned Newton Build

This section describes Newton `1.5.0.dev0` at commit
`8447545335aa264ef79bc7cd386bcdbb4f26eec3`. The API lives under
`newton.solvers.experimental.coupled`, so these details can change during a
Newton upgrade.

### Common coupled-solver model

`SolverCoupled` partitions one Newton model into named solver entries. Each
entry receives explicit ownership of bodies, particles, joints, and shapes, as
well as its own model view, state, controls, substep count, and solver factory.
The base class distributes global state to the entries and reconciles their
owned results after stepping.

Independent state ownership is necessary but is not physical coupling. Bodies
owned by different entries affect each other only when an algorithm such as
proxy coupling or ADMM creates the appropriate interaction.

In the pinned build, the following solvers implement Newton's coupling
interface:

- `SolverMuJoCo`
- `SolverKamino`
- `SolverFeatherstone`
- `SolverXPBD`
- `SolverVBD`
- `SolverSemiImplicit`
- `SolverImplicitMPM`

`SolverStyle3D` does not currently implement the interface. Interface support
means a solver exposes the hooks needed by the framework; it does not prove
that every solver pair, contact type, or vectorized B1K workload is correct.

### Proxy coupling

`SolverCoupledProxy` represents selected bodies or particles from one solver
as virtual proxies in another solver. The destination solver computes contact
against those proxies, and the resulting reaction is harvested and fed back to
the source.

The implementation offers two transfer modes:

- `lagged` transfers begin-step pose information and feeds back a delayed
  response.
- `staggered` transfers end-state information before the destination solve.

Multiple coupling passes and fixed or Aitken relaxation can reduce splitting
error. The current implementation supports at most two solver entries in one
proxy coupler.

Proxy coupling is the simplest practical bring-up path and often the better
throughput choice. Its weaknesses are order dependence, delayed feedback, and
possible instability when the two subsystems have very different masses or
stiffnesses. It must be tested specifically for light cloth against heavy robot
links, particle pressure on containers, and stiff deformable grasping.

### ADMM coupling

`SolverCoupledADMM` performs iterative coupling over constraints that cross
solver ownership boundaries. The pinned implementation includes cross-solver
joints, rigid-body-to-particle attachments, and contact rows for rigid-rigid,
rigid-particle, and particle-particle interactions.

ADMM is the more promising direction when strong two-way interaction and
consistent cross-solver constraints matter. It is also more expensive and adds
iteration count, penalty, proximal, stabilization, contact refresh, and
warm-start choices. More iterations do not automatically guarantee a better
result if the subsolvers, collision geometry, material models, or penalty
scales are inconsistent.

Newton ships examples for combinations including MuJoCo with VBD, XPBD, and
Implicit MPM, as well as VBD-XPBD and VBD/XPBD-MPM combinations. These examples
demonstrate that the plumbing exists. They are not evidence of stability,
accuracy, or throughput for full BEHAVIOR scenes.

## Coupling Requirements by Interaction

Coupling should be enabled for an actual physical interaction, not simply
because two solvers are present in a scene.

| Interaction | Required direction | Initial coupling approach |
| --- | --- | --- |
| Robot manipulates cloth or rope | Two-way rigid-deformable contact | Proxy for bring-up; compare with ADMM |
| Robot squeezes a soft object | Two-way rigid-deformable contact | ADMM candidate; proxy performance baseline |
| Robot stirs, scoops, or displaces liquid | Two-way rigid-MPM contact | Proxy initially; validate reaction forces carefully |
| Liquid or granular material loads a container | Two-way rigid-MPM contact | Proxy or ADMM, depending on stability and cost |
| Soft object rests on cloth | Two-way deformable-deformable contact | Proxy initially; ADMM if contact consistency is inadequate |
| Particles merely produce a visual effect | None or one-way | Avoid coupled dynamics |
| Thermal or symbolic state changes | Not mechanical coupling | OmniGibson state and transition systems |

One-way or uncoupled approximations are valid performance options only when
they preserve the task outcome. For example, purely visual steam need not push
the robot. Water being poured into a light movable cup probably must.

## Two Recommended Operating Points

These are targets for evaluation, not claims that the combinations are already
implemented or validated in OmniGibson.

### Performance-oriented vector simulation

- Keep robots and ordinary scene objects in one batched rigid solver.
- Use XPBD for cloth, ropes, and simple deformables when it passes the task's
  material and contact tests; use VBD only for cases that need it.
- Use MPM only for tasks requiring liquid, granular, plastic, or other
  continuum behavior. Use task-dependent particle resolution rather than the
  highest global resolution.
- Prefer proxy coupling with a small, fixed iteration count. Enable two-way
  coupling only for solver pairs that materially affect the task.
- Group environments by material and solver configuration. Do not make every
  environment pay for every solver.
- Keep collision detection, coupling, reset, observations, and rewards on the
  GPU, avoiding per-step host synchronization.

This operating point is intended for large-scale policy training. Its quality
threshold is not visual plausibility alone: it must preserve manipulation
success, failure modes, and the physical quantities used by task predicates.

### Maximum-fidelity simulation

- Use the best-validated articulated rigid solver with smaller time steps,
  stronger convergence tolerances, and high-quality collision geometry.
- Prefer VBD for mesh-based cloth and soft bodies; investigate an IPC-quality
  contact path if VBD contact cannot robustly handle folding and grasping.
- Prefer calibrated Implicit MPM materials at higher particle and grid
  resolution for liquids, granular materials, plasticity, and extreme
  deformation.
- Prefer iterative two-way coupling such as ADMM, with convergence and
  conservation diagnostics rather than a fixed iteration count chosen only by
  appearance.
- Accept fewer parallel environments when a global deformable/contact or MPM
  solve dominates runtime.

There is no single "maximum fidelity" setting across all B1K tasks. Solver and
resolution must be selected by material class, and the result must be checked
against measurements or trusted reference scenarios. A more expensive solver
can still be less realistic if its constitutive model or parameters are wrong.

## GPU Vector Environment Implications

Vectorization changes solver selection because throughput depends on much more
than the cost of one simulation step.

### Required architecture

- Environment state must be represented by batched device arrays with stable
  environment indexing.
- Contacts and coupling constraints must never cross environment boundaries.
- MPM grids must be isolated per environment, either structurally or through a
  proven spatial/indexing scheme.
- Reset must restore both public state and solver-persistent state such as
  contact history, warm starts, multipliers, proxy feedback, and MPM grids.
- Topology and allocation should remain fixed inside an episode. Slicing,
  dicing, melting, and recipe outputs should use preallocated activation pools
  where possible.
- Particle systems need fixed capacities, active masks, and bounded emission;
  unconstrained dynamic allocation is hostile to CUDA graph capture and stable
  memory use.
- Environments should be bucketed by solver graph and approximate problem
  size. A batch of rigid-only tasks should not carry dormant high-resolution
  MPM and cloth allocations.

### Expected scaling limits

Rigid-only environments can scale to much larger batch counts than environments
containing cloth, MPM, or strongly iterative cross-solver coupling. The main
limits will likely be particle and grid memory, contact pair count, solver
iteration divergence, and synchronization among coupled kernels.

The right target is therefore not one advertised environment count. Scaling
must be reported separately for representative workload classes and include
GPU memory, simulated steps per second, wall-clock steps per second, and the
fraction of environments that become unstable.

## Validation Plan

Before committing B1K to a solver combination, build a small suite that isolates
the behaviors the benchmark actually uses:

1. Rigid manipulation: grasp, stack, slide, open a door, and operate a drawer.
2. Cloth: drape, fold, self-contact, pinch grasp, and pull across an edge.
3. Rope: bend, knot-like self-contact, grasp, and cut/activate replacement.
4. Soft body: squeeze, recover, place in a container, and contact a sharp edge.
5. Liquid: pour, fill, spill, retain in a moving container, and displace with a
   tool.
6. Granular material: pour, heap, scoop, and load a movable container.
7. Coupled stress cases: rigid-cloth, rigid-soft, rigid-liquid,
   rigid-granular, and deformable-MPM contact.
8. Vector execution: run each applicable case at batch sizes such as 1, 16,
   64, and 256, subject to memory.

Record at least:

- penetration depth and contact jitter
- stretch, volume, or mass conservation as applicable
- energy growth and long-horizon drift
- constraint and coupling residuals
- task-level success and failure mode
- determinism or bounded nondeterminism across repeated runs
- step throughput, reset throughput, and peak GPU memory
- instability rate as batch size and material resolution increase

Solver selection should be based on these measurements. Example scenes and
visual inspection are useful diagnostics but are not acceptance criteria.

## Current Decision and Open Questions

The current rigid migration remains on `SolverMuJoCo`. The multiphysics stack
has not yet been selected or integrated. The leading evaluation stack is:

- MuJoCo-Warp for rigid and articulated dynamics
- VBD, with XPBD as a speed baseline, for cloth, rope, and mesh soft bodies
- Implicit MPM for fluids, granular materials, and extreme deformation
- Proxy coupling for initial integration
- ADMM coupling as the stronger two-way candidate

The following questions must be answered experimentally:

1. Can VBD contact handle B1K cloth folding and robot grasping without an IPC
   implementation? Folding: answered yes, see the self-contact validation under
   VBD above (zero interpenetration in a two-layer drape and a four-quadrant
   fold). Grasping with an actuated gripper, and the throughput needed for
   vectorized environments, are still open. Newton's own
   `example_cloth_franka.py` is the reference for the grasping half.
2. Which MPM material models and resolutions preserve pouring, filling,
   scooping, and container loading at useful vector throughput?
3. Does proxy coupling remain stable across the mass ratios and stiffnesses in
   household manipulation?
4. Does ADMM improve task-relevant behavior enough to justify its iteration and
   synchronization cost?
5. Can all persistent coupling and continuum state be reset independently per
   environment?
6. What solver-graph buckets cover the task distribution without excessive
   compilation, memory, or scheduling complexity?

Until those questions are answered, solver names in this document are
evaluation priorities rather than promises of production support.
