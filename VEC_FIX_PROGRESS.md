# vec/fix — Progress & Handoff

Context for an agent/developer picking this up on another machine. This branch (`vec/fix`)
holds **bug fixes for the vectorized OmniGibson env**, deliberately separated from the eval
work (which lives on `vec/eval-v2`) so they can go up as **two independent PRs**.

## Environment / workflow gotchas (READ FIRST)
- **Conda env is `behav`** (NOT `behavior`). Run Python via `~/miniconda3/condabin/conda run -n behav ...`.
- Run OmniGibson from the `OmniGibson/` directory. Set `OMNIGIBSON_HEADLESS=1` for no-display runs
  — **but keyboard teleop needs a real display** (see debug tool below).
- **OG hard-exits and swallows stdout** (incl. the pytest summary). Use `pytest --junit-xml=...`
  or have scripts write results to a file, then read the file.
- Full-scene loads are slow (`house_*` ≈ 5+ min). In constrained sandboxes, long background
  processes may get killed; prefer foreground or a powerful machine. **Partial room loading**
  (`load_room_types`) is the key speedup — see debug tool.
- **Disk filled up once** (98%): OmniGibson `appdata/global/cache/texturecache` grew to ~33 GB and
  caused `errno=28 / No space left on device` (inotify "change watch" spam + failures). Fix:
  `rm -rf OmniGibson/appdata/global/cache/texturecache` (regenerable). If inotify errors persist
  after freeing disk, raise limits: `sudo sysctl fs.inotify.max_user_watches=524288 max_user_instances=1024`.
- **2026 challenge task instances** are installed at `datasets/2026-challenge-task-instances/`
  (downloaded from HF `behavior-1k/zipped-datasets`). `metadata/available_tasks.yaml` maps
  task → scene + robot start pose. Re-download via
  `python -m omnigibson.utils.asset_utils --download_2026_challenge_task_instances`.

## Branch layout
```
main
 └─ vector                       (vec-env infra)
     └─ 7d584aef8  "base for vectorized eval"   ← vec/fix branches from HERE (pre-eval)
         ├─ vec/fix               = 7d584aef8 + 4 bug-fix commits + debug script   [THIS BRANCH]
         └─ vec/eval-v2           = 7d584aef8 + eval framework + the same 4 fixes
```
`vec/fix` contains ONLY the bug fixes + one debug example (no eval code). Verified the diff vs
`7d584aef8` touches only: `usd_utils.py`, `toggle.py`, `simulator.py`, `joylo/gello/robots/og_robot.py`.

## DONE — fixes on this branch (all validated)
JoyLo `launch_og.py` (r1pro) now runs against the vectorized env, and the assisted-grasp crash is fixed.

1. **JoyLo vec-env migration** (`joylo/gello/robots/og_robot.py`): the vectorized `Environment`
   changed accessors — `env.robots` is now `list[list[robot]]`, `env.external_sensors` is per-scene,
   `env.task.object_scope` is per-env, `env.step` returns per-env lists. Migrated all call sites to
   index scene/env 0 (`robots[0][0]`, `object_scope[0]`, `external_sensors[0]`, unwrap `info[0]`),
   guarded the reset stabilization loop with `isinstance(task, BehaviorTask)`, and renamed
   `ToggledOn.visual_marker` → `.marker`.

2. **Teleop "ghost" robot vs the batched articulation view** (`usd_utils.py`,
   `ArticulatedObjectViewAPI.initialize_view`): the ghost is a `visual_only`, `register=False`
   articulated `USDObject`; it matches the `/World/scene_*/articulated__*/*` USD pattern but isn't a
   registered scene object, so the old exact-equality assert failed. Relaxed to a **subset check**
   (every *registered* object must be covered; unregistered visual helpers are tolerated).

3. **Mid-step Fabric-read crash during assisted grasp** (`simulator.py`) — the important one:
   - Root cause: assisted grasp creates a joint inside `robot.post_step()` (i.e. while
     `currently_stepping=True`). `create_joint()` → `og.sim.update_handles()` → rebuilds all
     tensorized object-state views, and several of those (`ToggledOn`, `Inside`, ...) read **world
     poses from Fabric** in their `initialize_view()`, which is forbidden mid-physics-step →
     `AssertionError: Do not read poses from Fabric during a physics step`. Patching one state just
     moved the crash to the next → it's a **class** of bug.
   - Fix (systemic): in `update_handles()`, when `currently_stepping`, **defer** the tensorized-state
     `initialize_view()` loop (set `self._deferred_tensorized_view_init = True`) and run it in
     `_on_post_physics_step` after `currently_stepping` clears — mirroring the existing
     `_deferred_joint_breaks` mechanism. Step-safe physics-view refreshes still run immediately.
   - Also **reverted** an earlier `ToggledOn` "bake marker offset in `_initialize`" workaround
     (`toggle.py`) since the systemic defer supersedes it (single mechanism, no per-state whack-a-mole).
   - Validated: forcing `currently_stepping=True` + `update_handles()` with a microwave present
     (whose `Inside` reads Fabric) no longer crashes; the deferred rebuild runs on the next step.
     User confirmed the real assisted-grasp-on-oven crash is fixed.

## OPEN — toggle "button" not toggling / marker not turning green
User report: with r1pro in task `thawing_frozen_food` (scene `house_double_floor_lower`), touching a
button on the microwave/countertop does not toggle it on and the marker is **always red** (never green).

How `ToggledOn` works now (tensorized/warp, `omnigibson/object_states/toggle.py`):
- Per step, `_update_values` builds a tri-state mask: 0=no, 1=finger in *contact* with the toggle
  object (`RigidContactAPI.is_in_contact_batch_warp`), 2=contact AND finger mesh *overlaps* the marker
  sphere (`_check_overlap_kernel`: `wp.mesh_query_point_no_sign(finger_mesh, marker_center, radius)`).
  While mask==2 a per-object counter increments; at `CAN_TOGGLE_STEPS` (=5) the value flips.
- `post_update` sets `marker.color = COLOR_ON` (green `[0,1,0]`) / `COLOR_OFF` (red `[1,0,0]`) when the
  value changes. Marker center in the kernel = parent link pose @ static local offset;
  `radius = min(marker.extent * marker.scale)`.

Findings so far (see scratchpad scripts referenced below):
- The existing test `tests/test_object_states.py::test_toggled_on` **only passes because it inflates
  the button** (`stove.states[ToggledOn].link.scale = 3.0`, "to add tolerance"). At **normal scale**
  with the same arm motion → does NOT toggle. That's why the test can't expose the user's bug.
- Detection *does* fire when a finger is genuinely within the marker radius (forced fetch finger to
  d=0.019 < radius=0.043 → counter incremented). **BUT** that used unrealistic penetration — a real
  finger can only *touch the surface*, it can't go inside the object/marker.
- **Leading hypothesis (reframed with the user):** the overlap test needs the finger mesh within
  `radius` (~4 cm) of the marker **center**. If the marker center sits at/behind the collision
  surface, a realistic *surface touch* may never get within `radius` of the center → never toggles.
  The 3× inflation in the test hides exactly this.
- Color update itself works (in the 3× case the marker turned green), so "always red" is most likely
  "never toggles" rather than a color bug — **but confirm with the `Y` key test below.**
- r1pro's fingers ARE registered (`_marker_finger_pair` had 4 pairs) — NOT a "fingers not collected"
  bug. r1pro precise placement was untested (its holonomic base didn't respond to the base-shift trick).

### Debug tool (use this to bisect)
`OmniGibson/omnigibson/examples/robots/toggle_button_debug.py` (committed on this branch).
Run WITH a display:
```bash
python -m omnigibson.examples.robots.toggle_button_debug --robot r1pro --task thawing_frozen_food
```
- Loads the task's scene with **partial room loading** (only task rooms, not the whole house/yard) →
  fast. `--rooms kitchen,dining_room` to override, `--rooms all` for the full scene.
- Sets arms to **IK** (arrow keys move the hand); parks the robot next to `--target microwave`.
- Diagnostic keys: **`Y`** = force `set_value(True)` on all buttons → if the marker turns GREEN,
  render works and the bug is **detection**; if it stays RED, it's a **render** bug. **`H`** = force
  off. **`G`** = print marker center, **radius**, and current finger distance (tests whether a
  surface touch can reach within the radius). Status line prints `[on, color, d]` every 10 steps.

### Suggested next steps
1. Run the debug tool; press **`Y`** first (no driving needed) to settle render-vs-detection.
2. Press **`G`** while touching the button: compare `radius` to the smallest achievable `d`. If a real
   touch can't get within `radius`, that confirms the reframed hypothesis.
3. If it's detection: inspect where the toggle meta-link / marker center sits relative to the collision
   surface, and whether `radius = min(extent*scale)` is the sphere radius vs diameter. Consider
   comparing the new warp `_check_overlap_kernel` (point-in-mesh within radius, using
   `RigidBodyViewAPI.LINK_MESH_IDS` — check if that's collision or visual mesh) against the deprecated
   CPU path `_check_overlap` (used `og.sim.psqi.overlap_sphere(radius, pos=marker_center)` — a PhysX
   collider overlap). A behavioral difference there is a prime suspect. Possible fixes: enlarge the
   detection radius, move the marker center to the surface, or switch to a collider/volume overlap.
4. If it's render: check `GeomPrim.color` setter actually propagates to the rendered material under
   Fabric in the vec branch.

Scratchpad diagnostic scripts (not committed; recreate as needed) lived under the session scratchpad:
`toggle_normalscale.py` (3× vs normal A/B), `toggle_diag2.py` / `toggle_robot_cmp.py` (precise
placement + counter watch), `repro_light.py` (mid-step update_handles defer validation).
