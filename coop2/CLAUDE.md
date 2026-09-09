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
| L6 experiment (runners, grid, metrics) | `coop2/experiment/` | **GPU-verified with a stub LLM** (M7 steps 1–2) |
| L5 comm_topology (individual/chain/centralized) | `coop2/comm_topology/` | **copied verbatim, exercised by M7** (individual) |
| L4 cognitive/agent (FSM, memory, broker, prompts, LLM) | `coop2/cognitive/agent/` | **BDDL vocabulary aligned, FSM GPU-verified** (M7) |
| L3 cognitive/plan (plan lifecycle, PlanningEnvWrapper) | `coop2/cognitive/plan/` | **GPU-verified driving the facade** (2026-09-06) |
| L2 cognitive/action (symbolic action → primitive) | `coop2/cognitive/action/behavior_action.py` | **rewritten, GPU-verified** (M5) |
| L1a world model | `coop2/behavior_env/world_state.py` | **done, CPU-tested + GPU-verified** (M4) |
| L1b text observation + target_hints | `coop2/behavior_env/symbolic_view.py` | **done, CPU-tested + GPU-verified** (M4) |
| L1c primitive execution engine | `coop2/behavior_env/primitive_engine.py` | **done, GPU-verified** 2026-09-05 |
| L1d task tracker | `coop2/behavior_env/cooperative_tasks.py` | **done, M6 PASSED** 2026-09-07 |
| L1 symbolic nav fix + contention | `coop2/behavior_env/symbolic_{navigation,contention}.py` | **done, GPU-verified** 2026-09-05 |
| L1 placement (scene-derived poses) | `coop2/behavior_env/placement.py` | **done, GPU-verified** |
| L1 facade | `coop2/behavior_env/coop_env.py` | **done, GPU-verified** (M5) |
| L0 N-robot env config + startup ritual | `coop2/behavior_env/env_setup.py` | **done, GPU-verified** 2026-09-05 |
| — repair shim, viz stub | `coop2/_repair_shim/`, `coop2/cognitive/viz/` | **done** (no-op by design) |
| BDDL task + cached instance | `bddl3/.../coop_two_apples_pomaria/`, `feasibility_verify/{sample,verify}_coop_task_instance.py` | **done, GPU-verified** 2026-09-07 |
| M9 wiring (BehaviorTask + `check_goal` termination) | `coop2/behavior_env/coop_env.py` | **done**, commits `e4d28a98`…`5fda3d28` |

## Where we are (2026-09-08, evening)

**All three topologies solve the BDDL activity.** `individual`,
`broadcast_chain` and `centralized` each reached `coop_two_apples_pomaria`'s goal
at seed 0 with 2 agents and `--steps 4000`: `check_goal` fired at env_step 1303,
1303 and 1278 respectively, `{'satisfied': [0], 'unsatisfied': []}`. Runs take
about two minutes each. Reproduce with the command in "Commands" below.

| topology | goal @ step | plans | succeeded | primitives | primitive ticks | messages | LLM calls | tokens |
|---|---|---|---|---|---|---|---|---|
| individual | 1303 | 4 | 1 | 6 | 2278 | 0 | 4 | 13731 |
| broadcast_chain | 1303 | 4 | 1 | 6 | 2278 | 3 | 6 | 22328 |
| centralized | 1278 | 4 | 1 | 6 | 1924 | 6 | 7 | 19669 |

`centralized` finishes the manipulation in 1924 primitive ticks against 2278 for
the other two, and pays 6 messages and the highest API latency for it -- which is
the trade COOP2 exists to measure. One seed proves nothing about the ordering;
that is what M7 step 3's remaining seeds are for.

**M1-M6 passed. M7 steps 1-2 passed.** M9 is wired: `coop_env` loads
OmniGibson's `BehaviorTask` from the cached instance when `bddl_activity` is set,
and `compiled_task.check_goal` is the *only* authority over `terminated`.
Acceptance criteria are in PORTING_PLAN.md section 7.

What is left:

1. **M7 step 3** -- more seeds per topology for the metrics table. Note
   `build_results_table.py` cannot read these runs: it skips any folder without
   `team_score.json`, which the runners do not write while `team_score.enabled`
   is False. Same defect class as the rest of that file (see "The recurring
   defect class" in the memory notes) -- post-episode code that only runs after
   a full GPU episode.
2. **M8** -- decentralized topology.

**Scene note:** the earlier target was `house_single_floor` (Rs_int measured
unusable, 96.2 % of sampled base poses reject). The BDDL task is on
`Pomaria_1_int`/`living_room_0` because that is where the two-armchair +
coffee-table layout exists; revisit if N=9 needs more floor area than that room
has.

## The bug that made the activity look unsolvable (2026-09-08)

`OnTop` is `Touching` and `Touching` is a contact-report query, and **a sleeping
PhysX actor emits no contact reports**. Every OmniGibson object is constructed
with `DEFAULT_SLEEP_THRESHOLD = 5e-05` (`entity_prim.py`). An apple placed on the
coffee table falls the sampler's 2 cm z-offset, comes to rest, and is slept --
after which it is still sitting on the table and `OnTop` reads False forever.

Our own placement path is what makes it arrive so fast: `_place_with_predicate`
does release -> `set_position_orientation` -> `keep_still()` -> settle, and
`keep_still()` zeroes both velocities. `keep_still()` was added by the *previous*
fix, to stop the object being flung out of the house; it cured the fling and
brought the sleep forward. Both were real.

Two readers were being lied to, which is why the fix is central
(`coop_env._keep_task_objects_awake`, applied at build and after every reset) and
not at either call site:

- `_place_with_predicate` raised EXECUTION_ERROR "it did not come to rest there"
  for a third to a half of all placements -- which reads as a physics failure and
  is not one;
- `check_goal` cannot see a delivered apple that has gone to sleep, so **no run
  could have reported success no matter how well the agents played**. That is why
  "one of two apples delivered" was the ceiling for days.

Measured, not argued -- `feasibility_verify/measure_placement_rest.py`:

| | before | after |
|---|---|---|
| placements that come to rest, empty table | 10/20 | 20/20 |
| placements that come to rest, one apple already there | 11/20 | 20/20 |
| single-step `OnTop` reads of a settled apple | 0/20 | 20/20 |

The discriminating evidence: success and failure are geometrically identical --
same sampled and final z (0.415 -> 0.390), same `VerticalAdjacency` below/above
lists, and the radius-from-centre medians order *oppositely* in the two
conditions, i.e. noise. The neighbouring apple makes no difference (63 % vs
67 %), which killed the first hypothesis. The failing pose reports no contact
with **anything**, at velocity exactly `[0,0,0]`; `is_asleep` is the only thing
that differs, and `wake()` plus one step flips `OnTop` to True with the object
not having moved. Three geometric hypotheses were proposed and all three were
measured wrong before this one was measured right.

## Seed 3, one run per topology (2026-09-09)

| topology | end_step | goal | plans | ok / fail / cut | Y_plan | msgs | API | tokens |
|---|---|---|---|---|---|---|---|---|
| broadcast_chain | 1956 | **solved** | 3 | 1 / 0 / 2 | 1.00 | 1 | 3 | 10 819 |
| centralized | 2094 | **solved** | 3 | 1 / 0 / 2 | 1.00 | 2 | 4 | 11 360 |
| individual | 3174 | not solved | 5 | 1 / 2 / 2 | 0.33 | 0 | 5 | 17 847 |

Four seeds now: individual 2/4, centralized 4/4, broadcast_chain 4/4. Still not
enough to separate topology from seed noise, but individual is the only one that
has ever failed, and both of its failures are physics, not coordination --
`place_on_top[EXECUTION]` at seeds 1 and 3.

### `--steps` is not the budget: `--time-limit-seconds` defaults to 120

`run_*.py` takes a wall-clock deadline (`run_individual.py:326`) that defaults to
**120 s** and stops the episode independently of `--steps`. At ~26 env_step/s
that caps a run near 3100 steps, so `--steps 4000` has never once been reachable
and every run "to 4000" was really a run to 120 s. Individual seed 3 stopped at
3174 for exactly this reason.

Re-run with `--time-limit-seconds 0` to check: it still did not solve, so this
particular result stands. But the deadline is invisible in the run folder --
nothing records which limit fired -- and it silently makes `--steps` a
lower-bound-only knob. Pass `--time-limit-seconds 0` when the step count is
meant to be the experiment variable.

### The coffee table can fly away (seen once, not yet diagnosed)

In the 4000-step individual re-run, `coffee_table_gpkbiw_0` left the living room
under constant velocity: the `NO_SPACE_AROUND_TARGET` attribution logs it at
(-10.5, -10.1), then (-14.5, -17.9), (-17.6, -24.1) ... (-46.4, -80.8), roughly
equal steps in a fixed direction -- coasting, not accelerating, i.e. it carries a
velocity nothing damps. 97 rejections, all `{'room': 200, 'trav': 0, 'robots': 0}`:
every candidate pose was rejected for being outside the room, because the target
had left it. 18 of that run's 21 plans died this way (Y_plan 0.05).

Same shape as blocker #6 but on the *receptacle*, which is supposed to be
furniture. Not present in any of the three 120 s seed-3 runs, so it is an event
during the episode rather than a step-count threshold; the obvious suspect is a
placement impulse, and `_keep_task_objects_awake` keeps the table from ever
sleeping it off. Evidence kept in
`coop2/runs/individual_agents2_repair_off_seed3_20260909_025858_415866/`.

## Open defects

Fixed ones are not listed here -- the fix and its reasoning live in the commit
and in the code comment at the site. What is still true:

1. **A rejected precondition costs 50 ticks, not 0.** ``apply_ref`` runs its
   "settle before returning" block after catching the error and before raising
   the group, so ``TOO_FAR`` and friends are not free. Visible in the plan log
   as ``ticks=50`` on a failed action.
2. **cuRobo, if it is ever reintroduced, rejects poses on the other robot but
   not poses on a small object.** Measured 2026-09-05: 8/8 candidate poses
   collision-free from 0.05 m to 0.45 m from a 5 cm apple, because the base
   genuinely clears it -- a target-clearance floor is needed on top of any
   collision check. One generator per R1 costs ~2.2 GB and 4-8 s, and
   ``batch_size=16`` OOMs a 16 GB card in ``mg.warmup()``; the default is 2.
   cuRobo is currently **not used at all** (user decision): pose validity comes
   from the trav map plus geometry.

See also "Open, not yet diagnosed" near the end of this file for behaviour that
is understood but not yet explained.

## Commands

Use the `behavior` conda env (see the repo root `AGENTS.md`), and
`OMNIGIBSON_HEADLESS=1` when there is no display.

```bash
# CPU-only regression checks -- no Isaac, no GPU, a few seconds each. RUN THESE
# FIRST after touching anything in coop2/. They are main() scripts, not pytest
# cases: `pytest` collects nothing from them.
for f in feasibility_verify/test_*.py; do python "$f"; done

# The real thing: an LLM-driven episode against the BDDL activity.
python -m coop2.experiment.run_individual --agents 2 --steps 4000 --seed 0 \
  --scene Pomaria_1_int --room living_room_0 \
  --bddl-activity coop_two_apples_pomaria \
  --goal "Put both apples on coffee_table.n.01_1." \
  --model gpt-5.6-luna --llm-quiet
# run_centralized / run_broadcast_chain take the same flags.
# Output lands in coop2/runs/<topology>_agents<N>_..._<timestamp>/.

# Watch it in a window. --gui is the ONLY way: --show is a no-op (the
# visualisation wrapper is a stub by design) and OMNIGIBSON_HEADLESS is not
# enough on its own, because CooperativeBehaviorEnv defaults headless=True and
# _build assigns gm.HEADLESS itself. Needs a DISPLAY. Verified: an
# "OmniGibson 3.9.2" X window at 1468x966, with carb.windowing.plugins and
# omni.kit.mainwindow started -- neither loads headless.
python -m coop2.experiment.run_individual --gui --agents 2 --steps 60 --seed 0 \
  --scene Pomaria_1_int --room living_room_0 \
  --bddl-activity coop_two_apples_pomaria --model gpt-5.6-luna --llm-quiet
# Headless runs still record: per-robot episode_agent_<i>.mp4 land in the run
# folder, because RENDER_VIEWER_CAMERA and HEADLESS are separate switches.

# Inspect the scene by hand once the episode is over. --keep-viewer implies
# --gui, and runs *after* every log, plot and metric file has been written, so
# Ctrl+C out of it costs nothing. It keeps stepping the sim and prints each
# BDDL task object's position and speed every 2 s -- which is how to watch for
# the drifting coffee table without trying to catch it by eye.
python -m coop2.experiment.run_individual --keep-viewer --agents 2 --steps 4000 \
  --seed 3 --time-limit-seconds 0 --scene Pomaria_1_int --room living_room_0 \
  --bddl-activity coop_two_apples_pomaria --model gpt-5.6-luna --llm-quiet

# Same stack with no credentials: substitutes StubLLMClient for the model, so a
# later failure with a real one is unambiguously the model and not the plumbing.
python feasibility_verify/verify_runner_offline.py

# BDDL: sample an activity instance (~45 s, exits non-zero without saving if the
# layout is unstable), then check it, then check that check_goal ends an episode.
python feasibility_verify/sample_coop_task_instance.py
python -u feasibility_verify/verify_coop_task_instance.py
python -u feasibility_verify/verify_bddl_terminates_episode.py

# Diagnostics. Reach for these before forming a hypothesis.
python -u feasibility_verify/preview_cameras.py --out /tmp/cams
python -u feasibility_verify/measure_target_capacity.py
COOP2_ENGINE_VERBOSE=1 python -m coop2.experiment.run_individual ...   # per-primitive ticks

# Older GPU demos, pre-BDDL: contention and concurrency in isolation.
python feasibility_verify/multiagent_concurrent_symbolic_primitives.py --plan navigate_to
python feasibility_verify/multiagent_concurrent_primitives.py --mode exclusive
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

- `scene.include_robots: false`, **and that is not sufficient with a BDDL
  activity.** `Environment._load_robots` is guarded by
  `if len(self.scene.robots) == 0`, so your `robots:` list is ignored whenever
  the scene has already imported robots -- which it has, because
  `BehaviorTask.verify_scene_and_task_config` points `scene_instance` at the
  cached template and the template contains the robots it was sampled with.
  The robots then come up with R1's defaults (IK arms, delta trunk) while
  `robot._controller_config` still reports yours, and nothing raises until
  `q_to_action` asserts on the first tick. `coop_env._enforce_controller_config`
  compares the live `ControllerView` registry against the config we built and
  calls `reload_controllers` when they disagree.
- `task.use_presampled_robot_pose: false`. The class default is **True** despite
  its docstring saying False, and a template sampled without presampled poses
  has no `robot_poses` metadata, so `BehaviorTask.reset` dereferences None.
  This facade places robots itself anyway.
- Every robot needs an explicit `name` — it is the action/obs dict key.
- Use `model: r1`, not the deprecated `type: R1`. Prefer **R1 over R1Pro**:
  cuRobo drops the DEFAULT embodiment at cuda capability (12,0) (RTX-50) while
  `update_obstacles` indexes it unconditionally → `KeyError`.
- `enable_head_tracking=False` always. `_overwrite_head_action` asserts
  `robot.model == "tiago"` and `_grasp` sets `_tracking_object`.
- `apply_ref(attempts=1)`. The default 5× retry is **not idempotent** and burns
  thousands of ticks per attempt.
- Call `tune_primitive_macros()` **before** constructing any controller: reading
  a macro locks it against writes. `coop_env._build` does this now -- it did not
  for most of the port, which left `MAX_STEPS_FOR_SETTLING` at upstream's 500
  and made a single PLACE_ON_TOP cost over 1000 ticks (`_release` and
  `_settle_robot` each burn the budget in full).
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

1. **Prompt filtering** (`symbolic_view.STRUCTURAL_SYNSETS` /
   `RECEPTACLE_ABILITIES`, since renamed from the original category lists). One corridor produced 78 walls, 24 shelves, 20
   switches, 16 downlights and 14 paintings, plus nonsense hints like
   `place_on_top(downlight#22)` and `place_inside(door#3)`. Filtering lives in
   L1b, **not** L1a: the world model stays complete for task evaluation and only
   the prompt is pruned.
2. **Objects clustered near the team** (`place_objects(near_robots=8.0)`).
   `place_robots` already clustered the robots, but objects were still sampled
   from the whole room -- a 20 m corridor put the contested apple 11 m away.

**Corrected 2026-09-08.** This section originally concluded: "the
`_settle_robot` blow-up was a symptom, not a separate defect ... no macro tuning
was needed", on the grounds that primitives went from 1100-1728 ticks to 118-379
once objects were clustered near the team. That conclusion is why
`tune_primitive_macros()` was written and then **never called**, and the defect
survived until 2026-09-08: with `MAX_STEPS_FOR_SETTLING` at upstream's 500, a
single PLACE_ON_TOP was measured at 1063 ticks and still rising
(`COOP2_ENGINE_VERBOSE=1` shows the count climbing linearly with env_step),
because `_release` and `_settle_robot` each burn the full budget and an R1's
holonomic base does not reach `velocity < 0.01`. Clustering objects helped, but
it did not remove the need for the macro -- both were required.

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

## Reuse audit (2026-09-06)

Systematic pass for hand-rolled code that OmniGibson or BDDL already provides.
Replaced:

| was | now |
|---|---|
| private `robot._ag_obj_in_hand` (3 sites) | `robot.is_grasping(arm, candidate_obj)` |
| `RECEPTACLE_CATEGORIES`, ~25 category names | `obj.abilities` (`fillable` / `openable`) from the taxonomy |
| `STRUCTURAL_CATEGORIES` name list | synset ancestry via `ObjectTaxonomy.is_descendant` |
| `imageio.get_writer` | `eval.utils.obs_utils.create_video_writer` / `write_video` |
| ids `apple#1` | BDDL instance naming `apple.n.01_1` |
| scene-graph edge labels | BDDL tokens via `bddl_utils.PREDICATE_TO_STATE` |
| hand-rolled erosion / connectivity / free-space sampling | `scene.get_random_point(floor, reference_point, robot)` |
| objects dropped on random floor cells | `obj.states[OnTop].set_value(surface, True)` |

A rendered fact is now `ontop(apple.n.01_1, breakfast_table.n.01_1)` -- the same
strings an activity definition and its goal predicates use, so M9's
`check_goal` needs no translation layer.

The state->token mapping matters more than it looks: `Hot` is
`object_states.Heated` and `Attached` is `AttachedTo`, so a name-equality check
would work for most predicates and fail silently on exactly those.

**BDDL cannot build the room world graph.** It is purely symbolic: its
predicate classes are empty (`class OnTop(BinaryPredicate): pass`), truth comes
from a callback into OmniGibson, `InRoom` is not even in `PREDICATE_TO_STATE`
so it cannot be evaluated at runtime, and `knowledge_base` is an offline
catalogue of what a scene's rooms contain *by design*, not what is in them now.
The live graph has to come from `SceneGraphBuilder` + `seg_map`, which is what
`world_state.py` does -- that part is not duplicated work.

Genuinely no upstream equivalent, and kept: mutual separation between N robots,
destination reservation under concurrency, room-scoped free space, the
concurrent primitive engine, and the symbolic contention layer.

Found while doing this: `is_fixed` was **always False**. `scene.fixed_objects`
is a `{name: obj}` dict, so `set(...)` of it is a set of names and `obj in` it
never matches. Nothing depended on it until `is_receptacle` did.

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

## M9: BDDL reconnected — first custom task (2026-09-07)

A real BDDL activity now exists and is cached as a task instance, and `coop_env`
already loads it (`bddl_activity=...` -> `BehaviorTask` with
`online_object_sampling=False`) with `compiled_task.check_goal` deciding `terminated`.
This section is the authoring record: how the activity was written and how to
regenerate the instance.

**The task**: `bddl3/bddl/activity_definitions/coop_two_apples_pomaria/problem0.bddl`
— `Pomaria_1_int` / `living_room_0`, one apple on each of the two armchairs, goal is
`(forall (?apple.n.01 - apple.n.01) (ontop ?apple.n.01 ?coffee_table.n.01_1))`.

```bash
# produce the instance (~45 s, writes the template json). NOT in git -- re-run after a machine change.
OMNIGIBSON_HEADLESS=1 python -u feasibility_verify/sample_coop_task_instance.py
# acceptance test: loads the cached template, forces the goal, checks check_goal flips
OMNIGIBSON_HEADLESS=1 python -u feasibility_verify/verify_coop_task_instance.py
```

Template lands at
`$OMNIGIBSON_DATASET/2026-challenge-task-instances/scenes/Pomaria_1_int/json/Pomaria_1_int_task_coop_two_apples_pomaria_0_0_template.json`.
`BehaviorTask` finds it on its own — with `online_object_sampling: false` and no
`scene_file`/`scene_instance`, `verify_scene_and_task_config` rebuilds that exact
filename from `{scene}_task_{activity}_{def_id}_{inst_id}_template`.

### Authoring facts (all verified, not read off docstrings)

- **A new activity directory is auto-discovered.** `get_all_activities()` is `os.listdir`
  on `activity_definitions/`; `activity_manifest.txt` and
  `activity_to_preselected_scenes.json` are read by no code at all.
  `kb.add_task(name, definition=<string>)` registers one at runtime with no files, but then
  `room_requirements` stays empty and `task.matching_scene(scene)` falsely returns "pass" —
  only the file route gets a real pre-flight check. Use `matching_scene` before burning a
  GPU run; it names the missing furniture per room instance.
- **`sampling_whitelist` is `{synset: {category: {model: None-or-bbox}}}`** — a dict, not
  the "list of valid models" the `BehaviorTask` docstring claims (`bddl_utils.py:894` calls
  `.keys()` on it). It is the only way to pin one of several same-synset scene objects:
  `living_room_0` has two coffee tables and BDDL only knows the synset.
- **Declare exactly one agent.** Omitting it entirely crashes `BDDLSampler.__init__` at
  `bddl_utils.py:486` with `KeyError: 'agent.n.01_1'`, because `update_activity` puts that
  key in `object_scope` unconditionally while `_object_instance_to_synset` comes from
  `parsed_objects`. Declaring a *second* agent also crashes (the sampler binds only
  `agent.n.01_1`, leaving `None` for the rest, and `_filter_object_scope` then dereferences
  `None.prim_type`); it works only with a patch to `bddl_utils.py:553`, which was written,
  verified, and then **reverted on purpose** — OmniGibson is unmodified. Robots past
  `robots[0]` are simply invisible to BDDL, which costs nothing while no goal mentions an
  agent. The *cached* path never needed the patch: `behavior_task.py:548` already maps
  `agent.n.01_N -> env.robots[N-1]`.
- **`inroom` cannot be evaluated** (`PREDICATE_TO_STATE` has no `InRoom`), so
  `compiled_task.check_initial_conditions()` raises `KeyError` on any task using it. Check
  the kinematic facts directly instead. Same trap for `broken` and `grasped`; `grasped` is
  one dict line away from working (`object_states.IsGrasping` already exists).
- **Sampling is unseeded and apples roll.** Different apple models get drawn per instance
  and one rolled off the armchair during the 300-step settle. Both apples are pinned to
  model `omzprq` via the whitelist, and the script exits non-zero rather than save a layout
  whose own initial conditions are already violated. Always cache a template; never sample
  per run. `--instance_id N` produces alternative layouts.

### What the template does and does not carry

- Robot **world poses are in there**, but as `joint_pos` of the holonomic base joints, not
  `root_link.pos` — the root stays at the spawn/park anchor (`agent_0` reads
  `[300, 300, 300]`). Real pose = anchor + first three joint values.
- Robot **controller configs are NOT in there.** `init_info.args` holds only
  `name / model / obs_modalities / default_reset_mode / scale`; the saved
  `controller_groups` is goal *state*, and its `arm_left` goal is `target_pos` +
  `target_ori_mat`, i.e. the R1 **default task-space controller**, not the
  `JointController` stack `r1_primitives.yaml` requires.
- ⇒ **Load it with `include_robots: False`** and supply coop2's own robot list. The template
  then contributes only the object layout (apples, armchairs, coffee table), which is all we
  want from it; `build_multi_robot_config` already sets that flag.
- `env.reset()` restores the initial file, so poses set after loading need either
  re-applying each reset or a `scene.update_initial_file()` (`prepare_robots` does this).

## The seven blockers, in the order they were found (2026-09-08)

All fixed, and all seven had to go before any topology could finish the activity.
The seventh -- the sleeping apple, section above -- is the one that made the other
six look insufficient, because it hid success even when the agents played well.

| # | symptom | cause | fixed |
|---|---|---|---|
| 1 | every plan died at grounding, `decisions` stayed 0 | the runners inherited crafter's `CooperativeEnv(...)` call and placed no objects | `77d175296` |
| 2 | ~3.4 s per env_step | two all-pairs scans per macro-step: `SceneGraphBuilder.step()` (14.9 s) and `CoopTaskTracker._fact_set()` (9.4 s) | `77d175296` |
| 3 | agent placed apples on the wrong coffee table, `check_goal` never fired | `entity_id_for` numbered instances in scene order, independently of `task.object_scope`; `living_room_0` has two coffee tables | `5fda3d283` |
| 4 | a single `PLACE_ON_TOP` cost >1000 ticks, so 2500 steps bought ~2 primitives per agent | `tune_primitive_macros()` existed, was measured (1100-1728 -> 118-379 ticks) and **was never called**; `MAX_STEPS_FOR_SETTLING` stayed at upstream's 500, and `_release` + `_settle_robot` each burn it in full | `efe4c0159` |
| 5 | `wait` was an instant no-op, so an agent yielding the floor **stopped the world** (the plan loop does not step the env while any agent reasons) | `wait` was in `COMMUNICATION_ACTIONS` | `5fda3d283` |
| 6 | 51 x `NO_SPACE_AROUND_TARGET`, all `{room: 200, trav: 0, robots: 0}`, target logged at 12 m -> 27 m -> 36 m -> 38 m from the room | upstream's `_place_with_predicate` does release -> `set_position_orientation` -> settle, and **`set_position_orientation` does not zero velocity**: the object arrives carrying the fall it accumulated while being released, and the settle integrates it out of the house | `efe4c0159` |
| 7 | `place_on_top` failed "it did not come to rest there" on a third to a half of placements, and `check_goal` never fired even after an apple was correctly delivered | `OnTop` is `Touching`, `Touching` is a contact-report query, and a slept PhysX actor reports no contacts; the default sleep threshold is 5e-05 and `keep_still()` zeroes the velocity on the way in | `e205c0e0b` |

Method note, because it cost most of the day: for #6 I proposed three geometric
explanations (the annulus round the table is full; the target is being carried by a
teammate; `DestinationRegistry` reservations accumulate) and **measured all three to
be wrong** -- `feasibility_verify/measure_target_capacity.py` shows 28-48 % of
candidate poses accepted in every reproducible state, so 200 consecutive rejections
were impossible. What settled it was making the failure report its own attribution
(`rejected_by` per filter, plus `target_xy`) rather than reproducing states by hand.
Same shape as #2, where four rounds of guessing lost to one cProfile run.

## Diagnostics that exist now, use them first

- `feasibility_verify/preview_cameras.py` -- one frame per camera view, then stops.
  Framing cannot be checked by reading pose numbers; four wrong poses got through
  that way. Also renders control shots straight at each robot, which separates "the
  framing is wrong" from "nothing renders".
- `feasibility_verify/measure_target_capacity.py` -- per-filter rejection histogram
  around any target, with and without a teammate parked next to it.
- `COOP2_ENGINE_VERBOSE=1` -- per-primitive progress lines (`agent_0:PLACE_ON_TOP@1063`).
  This is what exposed #4: tick counts rising linearly with env_step and never ending.
- `[nav]` lines -- sampled pose, distance and travel ticks charged, one per navigate.
- `NO_SPACE_AROUND_TARGET` metadata carries `rejected_by` and `target_xy`.

## Open, not yet diagnosed

- **Models mangle instance suffixes.** `apple.n.01_01` for `_1` appeared in three
  separate runs; `resolve_target` now normalises zero padding. But putting the id in
  the goal text produced `coffee_table.n.01_01_1` -- a *doubled* suffix, which the
  normaliser does not handle. Prompt wording was tried first and did not hold.
- **`TOO_FAR` after a successful `navigate_to` to the same object** (16 in one run).
  The prompt promises navigation puts you in range. The object was being carried by a
  teammate, so it moved; and the code returned `TOO_FAR` rather than `OBJECT_CLAIMED`,
  which means `holder_of` saw it as unheld -- consistent with the window inside
  `_place_with_predicate` where the object has been released but not yet placed.
- **`progress 0/4`** reads as "nothing started" when it means "action 1 is still
  running". Cosmetic, but it misled a diagnosis once.

## Deliberately not implemented

**Target arbitration as a lock.** Contention is enforced as a *precondition
failure*, never as a refusal at `assign()`. The loser still burns the full
navigate and still issues its GRASP, so the wasted decision stays visible to
the cognitive layer — that waste is exactly what the centralized leader's
allocation and the broadcast chain's proposals are measured on. A pre-
assignment lock would hide the signal. `engine.held_objects()` exposes the
cross-agent "who holds what" view for L1b's text observation to surface.
