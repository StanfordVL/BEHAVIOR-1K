# coop2 — COOP² multi-agent cooperation on BEHAVIOR-1K

Port of `coop2-llm-mas/ma_crafter`'s LLM multi-agent cooperation stack onto
BEHAVIOR-1K. Goal: run **individual / broadcast_chain / centralized** topologies
here with minimal changes to the COOP² code, then add **decentralized**.
Repair is explicitly **not** ported.

**Read `coop2/PORTING_PLAN.md` before designing anything.** It is the canonical
design doc: seven-layer structure, the full environment-contract checklist that
COOP²'s upper layers require, the BDDL multi-agent audit, and the trap list.
This file is only the operational summary.

## Layers

| layer | package | status |
|---|---|---|
| L6 experiment (runners, grid, metrics) | `coop2/experiment/` | **copied, imports clean** (M3) |
| L5 comm_topology (individual/chain/centralized) | `coop2/comm_topology/` | **copied verbatim** (M3) |
| L4 cognitive/agent (FSM, memory, broker, prompts, LLM) | `coop2/cognitive/agent/` | **copied**, 3 edits pending (M5) |
| L3 cognitive/plan (plan lifecycle, PlanningEnvWrapper) | `coop2/cognitive/plan/` | **copied**, `_plan_goal_failure_reason` pending |
| L2 cognitive/action (symbolic action → primitive) | `coop2/cognitive/action/behavior_action.py` | **rewritten, GPU-verified** (M5) |
| L1a world model | `coop2/behavior_env/world_state.py` | **done**, CPU-tested (M4) |
| L1b text observation + target_hints | `coop2/behavior_env/symbolic_view.py` | **done**, CPU-tested (M4) |
| L1 placement (scene-derived poses) | `coop2/behavior_env/placement.py` | **done**, GPU-verified |
| L1 facade | `coop2/behavior_env/coop_env.py` | **done, GPU-verified** (M5) |
| L1d task tracker | `coop2/behavior_env/cooperative_tasks.py` | **import stub only** (M6) |
| — repair shim, viz stub | `coop2/_repair_shim/`, `coop2/cognitive/viz/` | **done** (no-op by design) |
| L1c primitive execution engine | `coop2/behavior_env/primitive_engine.py` | **done, unverified on GPU** |
| L1 symbolic nav fix + contention | `coop2/behavior_env/symbolic_{navigation,contention}.py` | **done, unverified on GPU** |
| L1 env facade, world model, text obs, task tracker | `coop2/behavior_env/` | only L0+L1c exist |
| L0 N-robot env config + startup ritual | `coop2/behavior_env/env_setup.py` | **done, unverified on GPU** |

**M1 / M2 / M2.5 passed on GPU 2026-09-05** (RTX 5070 Ti, symbolic mode): env
loads in 20.4 s, two symbolic controllers build in 0.1 s with cuRobo skipped
entirely, `--mode concurrent` gives overlap_ratio 0.99 against 0.00 for
`--mode exclusive`, and the radius gate / travel cost / `OBJECT_CLAIMED` all
fire. **M3/M4/M5 passed** (M5 on 2026-09-06): `import coop2.experiment.run_individual` and every other copied module imports clean. Next is **M4** (L1a/L1b world model + text observation). Acceptance criteria are in PORTING_PLAN.md §7.

## Known defects (found during M1–M2.5, not yet fixed)

0. **Measured 2026-09-05: upstream's collision check is affordable but not
   sufficient.** One `CuRoboMotionGenerator(robot, use_default_embodiment_only=True,
   batch_size=2)` per R1 costs **7.7 s / 4.0 s and ~2.2 GB each** (6.23 GB of
   15.4 GB for two), and one `check_collisions` batch costs **96 ms with
   `update_obstacles`, 5 ms without**. It **does** reject poses on top of the
   other robot (0/4 collision-free), fixing the `holding=agent_1` corruption.
   It does **not** reject standing on a 5 cm apple: 8/8 collision-free at every
   distance from 0.05 m to 0.45 m, because the robot base genuinely clears it.
   So a target-clearance floor is still needed on top. NB `batch_size=16` OOMs
   a 16 GB card during `mg.warmup()` (13.25 GiB allocated) -- keep the default 2.

1. ~~**`distance_range` lower bound is 0.0**~~ **FIXED 2026-09-05.** The sampler
   now derives its range per object as
   `[clearance(obj), clearance(obj) + reach]` where
   `clearance = obj_half_diagonal_xy + robot_radius + margin`, and filters every
   candidate on traversability (eroded trav map) and separation from other
   robots (default `2 * robot_radius`, just-touching -- symbolic navigation
   teleports, so there is no path to keep clear). `interaction_radius` is
   likewise per object now, derived from the same range, because a fixed radius
   cannot serve both an apple (0.72 m) and a table (3.38 m). Original text:
   **`distance_range` lower bound is 0.0**, inherited from upstream
   `BASE_POSE_SAMPLING_LOWER_BOUND`. Safe upstream only because cuRobo
   validates the sampled pose; our cuRobo-free sampler does not, so the robot
   can teleport *onto* its target. Observed: agent_0 landed 0.105 m from
   apple_0 and the subsequent grasp failed `POST_CONDITION`; with contention
   off, both robots landed on each other and agent_0's **physical** assisted
   grasp latched onto robot agent_1 (`holding=agent_1`). Raise the lower bound
   and reject candidates near another robot.
2. ~~**apple_0 at (-1.694, -3.578) has `room=None`**~~ **FIXED 2026-09-05**:
   moved to (-0.5, -1.25), which is in `living_room_0`, inside the viewer camera
   frustum, has the best standable-pose fraction of any visible in-room point in
   this scene (23%), and is near-equidistant from both robots (1.99 vs 1.95 m)
   so `--contend` is a real race. Original text: **apple_0 has `room=None`** so
   `_target_rooms` returns `[None]` and the same-room filter degenerates to
   `None in [None]`, accepting any candidate. Will break L1b's room-level
   observation.
3. **`_settle_robot` makes tick counts nondeterministic**: it loops to
   `MAX_STEPS_FOR_SETTLING=500` until velocity < 0.01, so a robot teleported
   into geometry never settles. The same seeded NAVIGATE_TO cost 359 / 421 /
   858 ticks across three runs.
4. A rejected precondition still costs **50 ticks**, not 0: `apply_ref` runs
   its "settle before returning" block after catching the error and before
   raising the group.

## Commands

Use the `behavior` conda env (see the repo root `AGENTS.md`), and
`OMNIGIBSON_HEADLESS=1` when there is no display.

```bash
# GPU demos (need Isaac + an RTX GPU)
python feasibility_verify/multiagent_concurrent_primitives.py --plan navigate
python feasibility_verify/multiagent_concurrent_primitives.py --mode exclusive
python feasibility_verify/multiagent_concurrent_symbolic_primitives.py --plan navigate_to

# CPU-only regression tests -- no Isaac, no GPU. RUN THESE FIRST after
# touching anything in coop2/behavior_env/.
python feasibility_verify/test_primitive_engine_stubbed.py
python feasibility_verify/test_symbolic_navigation_stubbed.py
python feasibility_verify/test_symbolic_contention_stubbed.py
python feasibility_verify/test_world_state_stubbed.py
```

## The engine (L1c) in one paragraph

`MultiAgentPrimitiveEngine` is a **stepper, not a driver**: the caller owns the
main loop. `assign(agent_id, primitive, target)` starts a primitive without
advancing anything; `tick()` performs exactly one `env.step`, feeding each
active agent the next value from its `apply_ref` generator and every other
robot a hold-position action; `has_active(agent_id)` says whether an agent's
previous primitive is still in flight. `tick()` returns only the primitives
that terminated on that tick, so the normal return value is `{}`.

This shape exists so the barrier can sit at the **plan** boundary, matching
COOP²'s `PlanningEnvWrapper.step`:

```python
while not done:
    if not all_ready:                                  # plan-level barrier
        yield idle_step_return(...); continue          # tick() NOT called: physics frozen
    for agent_id in executing:
        if not engine.has_active(agent_id):
            engine.assign(agent_id, *next_primitive_of(agent_id))
    for agent_id, outcome in engine.tick().items():
        ...plan.advance_action() / complete_failed() / set_unready('plan_terminated')
```

`macro_step()` exists for scripted demos only. It aligns agents at every
primitive boundary, which COOP² does **not** do — never build the runner on it.

## Hard constraints (each of these fails silently or crashes)

- `scene.include_robots: false`. `Environment._load_robots` is guarded by
  `if len(self.scene.robots) == 0`, so otherwise your `robots:` list is ignored.
- Every robot needs an explicit `name` — it is the action/obs dict key.
- Use `model: r1`, not the deprecated `type: R1`. Prefer **R1 over R1Pro**:
  cuRobo drops the DEFAULT embodiment at cuda capability (12,0) (RTX-50) while
  `update_obstacles` indexes it unconditionally → `KeyError`.
- `enable_head_tracking=False` always. `_overwrite_head_action` asserts
  `robot.model == "tiago"` and `_grasp` sets `_tracking_object`.
- `apply_ref(attempts=1)`. The default 5× retry is **not idempotent** and burns
  thousands of ticks per attempt.
- Call `tune_primitive_macros()` **before** constructing any controller: reading
  a macro locks it against writes.
- Construct controllers only after the robots are at their reset pose —
  `_arm_targets` / `_reset_eef_pose` are frozen in `__init__`.
- Idle action is `robot.q_to_action(robot.get_joint_positions())`, **not**
  `controller._empty_action()` (which servos the arm to the frozen targets).
- `og.sim` is a process singleton. One env per process; parallel runs must
  fan out via subprocess.

## Primitive sets: what actually works

- **Physical** (`StarterSemanticActionPrimitives`): only GRASP, PLACE_ON_TOP,
  PLACE_INSIDE, NAVIGATE_TO, RELEASE. OPEN / CLOSE / TOGGLE_ON / TOGGLE_OFF
  `raise NotImplementedError`. One primitive costs 10³–10⁴ ticks.
- **Symbolic** (`SymbolicSemanticActionPrimitives`): OPEN/CLOSE/TOGGLE do work,
  but `NAVIGATE_TO` is **broken as shipped** — its inherited sampler
  dereferences the cuRobo motion generator the symbolic constructor never
  builds, and then passes a keyword the symbolic `_navigate_to_pose` rejects.
  Use `coop2.behavior_env.symbolic_navigation.NavigableSymbolicActionPrimitives`
  instead.
- Symbolic `_grasp` teleports the object to the end-effector **at any
  distance** and `_navigate_to_pose` is a pure teleport, so symbolic mode has
  essentially no resource contention as shipped. Use
  `symbolic_contention.ContentiousSymbolicActionPrimitives` (subclass of
  `NavigableSymbolicActionPrimitives`) to put it back — see below.

## Pose-filter comparison (measured 2026-09-05, N=9 planning)

`feasibility_verify/measure_pose_filters.py`, 120 candidate poses around apple_0,
one cuRobo generator as ground truth. R1 base radius measured **0.62 m**;
trav_map erosion radius **0.82 m**.

| filter | VRAM | scales to N=9 | holes vs cuRobo |
|---|---|---|---|
| per-robot cuRobo | 2.2 GB *each* -> 21.5 GB for 9 | **no** (card is 15.4 GB) | 0 by definition |
| shared cuRobo (`update_obstacles(ignore_objects=...)`) | ~2.2 GB total | yes | unverified: obstacles are expressed in the generator's own robot root frame |
| trav_map AND geometry | 0 | yes | **3 / 120** |

- `cuRobo accepts but geometry rejects = 76` is **not** a regression: it is almost
  entirely d <= 0.8 m, i.e. exactly the "standing on the apple" poses we want to
  reject and cuRobo does not.
- `cuRobo accepts but trav_map rejects = 84`: the baked `floor_trav_0.png` covers
  **all** of Rs_int's furniture, while the env loads only
  `["floors", "walls", "coffee_table"]`. The map is therefore more conservative
  than the actual scene. Loading full furniture would shrink this.
- The residual 3 holes survive raising robot separation from 0.8 to 1.3 m
  (`geometry accepts but cuRobo rejects` fell 15 -> 11, union stayed 3), so they
  are trav_map holes, not robot overlap. Cause not isolated.
- trav_map rejects 0/12 within 0.8 m of apple_0 -- that spot is genuinely cramped
  (it is also the `room=None` point). ~25% of candidates pass at d >= 1.0 m, so
  200 sampling attempts still succeed; the robot just stands further back.

## Scene choice: Rs_int is unusable (measured 2026-09-05)

`feasibility_verify/measure_teleport_risk.py` and `survey_scene_capacity.py`
read the baked `floor_trav_0.png` maps directly (0.01 m/px) -- CPU only, no
Isaac. R1's circumscribed radius is **0.62 m**
(`norm(reset_joint_pos_aabb_extent[:2]) / 2`, arms included).

| scene | bad-pose rate | dead targets | 9 robots fit? |
|---|---|---|---|
| **Rs_int** | **96.2%** | **46.3%** | no -- largest connected free region is **0.9 m2** |
| house_single_floor | 29.5% | 1.7% | yes (1709 m2) |
| Beechwood_0_int | 81.9% | 19.7% | marginal |
| Merom_1_int | 90.5% | 35.3% | no |
| office_large | 65.2% | 18.7% | marginal |

* **bad-pose rate** = sampled base poses landing where an R1 does not fit. A
  validity filter turns these into retries, so they are survivable.
* **dead targets** = targets with *no* valid pose anywhere in the 0-1.5 m
  annulus. A filter cannot help: NAVIGATE_TO just raises PLANNING_ERROR.

So **the scene must change before the filter matters**. Rs_int stays broken at
any radius (86.5% / 13.3% even at an unrealistically small 0.42 m). This is why
robots were visibly teleporting into walls and toppling in the demo videos.

Prefer a multi-room house over the big halls (`hall_arch_wood` has 4560 m2 but
is one undivided space): L1b's room-level world graph and COOP2's spatial
constraint both need real room separation. `house_single_floor` is the
candidate. 38 / 51 scenes fit 9 robots at 1.24 m separation.

## N=3 end-to-end, verified on GPU 2026-09-05

`house_single_floor`, `--n-robots 3 --plan navigate_to_then_grasp --contend`:
one winner, two losers, each with a legible reason.

```
agent_1  NAVIGATE_TO  153  ->  GRASP  success
agent_0  NAVIGATE_TO  272  ->  GRASP  OBJECT_CLAIMED (held by agent_1)
agent_2  NAVIGATE_TO  379  ->  GRASP  OBJECT_CLAIMED (held by agent_1)
overlap 0.75   env.step 431   wall clock 21 s
```

Two fixes got it there, and their effect was much larger than expected:

| | before | after |
|---|---|---|
| entities in the prompt | 218 | 47 |
| legal (primitive, target) pairs | 132 | 49 |
| env.step ticks | 2830 | 431 |
| wall clock | 214 s | 21 s |
| overlap ratio | 0.37 | 0.75 |

1. **Prompt filtering** (`symbolic_view.STRUCTURAL_CATEGORIES` /
   `RECEPTACLE_CATEGORIES`). One corridor produced 78 walls, 24 shelves, 20
   switches, 16 downlights and 14 paintings, plus nonsense hints like
   `place_on_top(downlight#22)` and `place_inside(door#3)`. Filtering lives in
   L1b, **not** L1a: the world model stays complete for task evaluation and only
   the prompt is pruned.
2. **Objects clustered near the team** (`place_objects(near_robots=8.0)`).
   `place_robots` already clustered the robots, but objects were still sampled
   from the whole room -- a 20 m corridor put the contested apple 11 m away.

**The `_settle_robot` blow-up was a symptom, not a separate defect.** Primitives
that cost 1100-1728 ticks now cost 118-379. `MAX_STEPS_FOR_SETTLING=500` is only
reached when a robot never comes to rest, and that was caused by the long-range
placement, not by the settle logic. No macro tuning was needed.

## M5 acceptance, GPU-verified 2026-09-06

One agent through the plan channel, no LLM
(`feasibility_verify/verify_plan_channel.py`):

```
navigate_to  success  551 ticks
grasp        success  100 ticks
navigate_to  success  523 ticks
place_on_top success  150 ticks
final held_objects: {}      decision_count=4   env_step=1328
apple relations: OnTop(apple#1, bookcase#2)
```

⚠️ **The first run of this printed PLAN COMPLETE while the apple was still in
the gripper.** `action_outcome` is true for exactly the tick its primitive
terminated on, but `step()` cached the whole `info` dict between refreshes and
served the stale outcome with it -- so every action reported its predecessor's
success the instant it was issued, and three of the four primitives never ran
(`decision_count` was 2, not 4). Fixed by always overwriting `action_outcome`
when reusing cached info. The status column could not catch this; only the
independent facts could -- `held_objects` and the scene graph's `OnTop`. Keep
verifying against physical state, not against the status field.

## Observation scope and refresh (decided 2026-09-06)

* `observation_for()` shows the agent's **current room only**;
  `include_seen_rooms=True` is opt-in. An observation that accumulates every
  room ever visited grows without bound over an episode and stops describing
  where the agent is, and the current room is the scope COOP2's spatial
  constraint is defined on anyway.
* The world model is rebuilt **only when a primitive terminates** — i.e. when an
  agent returns to the reasoning stage and actually has a reason to look. There
  is no timer refresh (`observation_every` defaults to 0): a primitive spans
  10^2–10^3 ticks, so a periodic rebuild would recompute the scene graph
  hundreds of times inside one primitive for nobody to read.

## Concurrent destination race (fixed 2026-09-06)

Separation was checked against other robots' **current** positions, which under
concurrency is a time-of-check/time-of-use bug. All N agents get NAVIGATE_TO on
the same tick; each samples its destination on its generator's first `next()`,
while every other robot still stands at its start pose metres away. Every check
passes, then all of them teleport beside the same object.

Measured on a 3-agent contend run: agent_0 and agent_1 ended **0.64 m** apart
against a 1.24 m requirement, and agent_1's assisted grasp latched onto
**agent_0** — `holding=agent_0`. The same `holding=<robot>` corruption the
separation filter was supposed to have removed.

Fix: `symbolic_navigation.DestinationRegistry`, **one per scene**, shared by
every controller. An agent reserves the pose it is about to occupy; every other
sampler avoids reservations as well as bodies. Reservations are overwritten,
never released — "this agent intends to be here" holds until it decides
otherwise, and once it arrives the reservation and its body coincide, so an
aborted primitive self-corrects instead of leaking a blocked spot.

After: closest pair 1.93 m, no `holding=<robot>`, and `OBJECT_CLAIMED` is back
as the contention signal instead of physics-induced `POST_CONDITION`.

The regression test is statistical on purpose: without the registry ~100/200
trials overlap, with it 0/200. A single-draw version of that assertion is flaky
(three random poses around one object are sometimes well separated) and would
eventually get deleted rather than fixed.

## L3 verified end-to-end, 2026-09-06

`feasibility_verify/verify_l3_plan_loop.py` drives the real
`PlanningEnvWrapper` (its ready barrier, plan lifecycle and logging) with a
scripted agent in place of the LLM:

```
Plan #1 navigate_to -> grasp -> release   all OK, SUCCEEDED at step 693
Plan #2 grasp(ghost#99)                   failed -> "terminating plan" -> reasoning
Plan #3 navigate_to                       OK -> complete -> reasoning
decision_count 4, env_step 888, barrier closed for exactly 3 ticks
```

Confirms the intended model: an arbitrary-length plan runs to completion
without the driver advancing it, only completion or failure returns the agent
to reasoning, and physics is frozen while it reasons.

⚠️ **Two executor sets is the trap here.** The facade builds
`CooperativeBehaviorEnv.executors` and calls `execute()` on them, while L3 reads
plan progress from `get_action_records()` on the *wrapper's*
`agent_actions`. When those were separate objects the wrapper's history stayed
empty, `action_status` came back None, and L3 re-issued action 1 forever. From
outside it is indistinguishable from a slow primitive -- it burned a 30-minute
timeout before being caught. `BehaviorSymbolicEnvWrapper._adopt_facade_executors`
now shares one executor per agent.

The verify script has a wall-clock cap and a stall detector (>12 primitives
issued without `current_action_index` moving) precisely because a timeout
cannot tell "slow" from "not progressing".

## Metrics: two different counters

`engine.env_step` counts ticks (what `Timeout(max_steps)` counts).
`engine.decision_count` counts primitives issued — **this is the denominator
for COOP²'s metrics**. One primitive is 10³–10⁴ ticks, so per-tick rates are
meaningless.

## Symbolic contention (L1)

`ContentiousSymbolicActionPrimitives` restores resource competition to the
distance-blind, holder-blind symbolic set with three coupled rules:

1. **Interaction radius** — GRASP / PLACE / OPEN / TOGGLE require the base
   within `interaction_radius` m of the target, else `TOO_FAR`.
2. **Travel cost** — `_navigate_to_pose` yields hold-position ticks
   proportional to distance **before** teleporting. Padding after the teleport
   would be wrong: the robot would arrive instantly and then idle, so a
   teammate reading the world during those ticks sees it already there.
3. **Claims** — acting on an object another robot holds raises
   `OBJECT_CLAIMED`. Upstream `_grasp` checks only `self.robot._ag_obj_in_hand`
   and `_establish_grasp` puts its joint under the *grasping* robot's eef, so
   without this the object is yanked out of the holder's hand, carries **two**
   FixedJoints, and both robots' post-conditions pass. Silent corruption.

⚠️ `interaction_radius` must be ≥ `distance_range[1]` (the nav sampler's upper
bound), or a successful navigate still sometimes lands out of range and the
agent loops navigate → TOO_FAR forever. The constructor rejects that outright;
the default derives the radius from `distance_range`.

Both new codes are raised as `PRE_CONDITION_ERROR` with
`metadata["reason_code"]` set; `ReasonCode.from_primitive_error` prefers that
over the five-member enum.

**Both are in `TERMINATES_PLAN`** (decided 2026-09-06). They are individually
recoverable — a teammate may release the object, walking closer fixes the
distance — but the plan that produced them was written against a world that has
since contradicted it, so its next action is a stale intention. Terminating
returns the agent to the **reasoning stage**, which is the only place it can
negotiate for the contested object or retarget. Grinding the plan on instead
would turn contention into silent wasted motion rather than a decision the
topology layer is measured on.

## Deliberately not implemented

**Target arbitration as a lock.** Contention is enforced as a *precondition
failure*, never as a refusal at `assign()`. The loser still burns the full
navigate and still issues its GRASP, so the wasted decision stays visible to
the cognitive layer — that waste is exactly what the centralized leader's
allocation and the broadcast chain's proposals are measured on. A pre-
assignment lock would hide the signal. `engine.held_objects()` exposes the
cross-agent "who holds what" view for L1b's text observation to surface.
