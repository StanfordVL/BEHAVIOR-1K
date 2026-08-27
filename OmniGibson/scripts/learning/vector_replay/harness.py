"""Standalone multi-env (num_envs > 1) replay harness for recorded BEHAVIOR demos.

``VectorReplayHarness`` mirrors the single-env playback logic of
``omnigibson.envs.data_wrapper.DataPlaybackWrapper`` (config surgery from
``create_from_hdf5`` with ``include_contacts=True``, and the per-episode
lifecycle of ``playback_episode``), generalized to N scenes stepped in
lockstep. It deliberately does NOT inherit ``DataWrapper`` /
``DataPlaybackWrapper``: on this branch both assert ``num_envs == 1`` by
design, and this harness records no observation data, so there is nothing to
inherit.

Key deviations from the single-env upstream, each intentional:

- Per-scene state injection: a single-scene recording's serialized state blob
  is exactly one scene's serialized state (``og.sim.serialize`` is a pure
  concatenation of per-scene states), so each demo's state row is injected via
  ``scene_i.load_state(blob, serialized=True)`` instead of
  ``og.sim.load_state``.
- Transition placement frame: upstream parks objects added by a transition at
  world-frame ``100 + 5j``. In a multi-scene stage that would park every env's
  added objects at the same world location (inside some other env's cell), so
  this harness parks them at the same offset in the *scene* frame
  (``frame="scene"``). The parking pose is transient either way: the recorded
  state stream re-poses the object on the next iteration's ``load_state``.
- Transitions recorded at a demo's final step are DEFERRED to a terminal pass
  after the batch's step loop, not skipped. Upstream captures a step's info
  before applying that step's transitions, so a goal produced by the final
  step's transition can only be observed on a step the recording does not
  contain -- which is why such a demo used to be scored a fail with nothing in
  its row explaining why (task-0040 demo 400100: transition at 4852 of 4853
  steps, popcorn never created, both predicates unsatisfied). Applying it
  in-loop is what the old skip avoided, because the deterministic hold would
  then keep re-injecting a final recorded row whose serialized layout predates
  the added objects. The terminal pass sidesteps both problems: it runs once
  the loop is over, re-injects the final recorded row (topology still matches),
  applies the transition, settles, and takes ONE extra batched step to read the
  post-transition verdict. See _apply_terminal_transitions.
- Finished demos hold: once demo i's steps are exhausted, its final recorded
  state row keeps being re-injected and its final action re-applied until the
  longest demo in the batch finishes (deterministic hold).

Designed error behavior (the ONLY designed continue in this module): a demo
whose HDF5 file / demo group / datasets are malformed is recorded as a result
row with ``status="replay_error"`` plus the exception string, excluded from
the batch, and the rest of the batch proceeds. Every other failure raises and
propagates. A batch-membership violation (a demo whose recording is
incompatible with the batch scene -- see BatchObjectSetMismatchError for the
exact invariant) always raises -- the sharder must split such batches, never
silently continue. So does a scene-separation violation (see
SceneSeparationError): the batch's scenes stopped being geometrically disjoint,
which voids every demo in it.
"""

import json
import logging
import os
import re
import time

import h5py
import torch as th

import omnigibson as og
from omnigibson.controllers import ControllerView, ControlType
from omnigibson.envs.data_wrapper import (
    _align_scene_object_states_with_recorded_schema,
    _is_system_particle_template_info,
    _is_system_particle_template_name,
    _recorded_non_kin_state_name_union,
)
from omnigibson.envs.env_base import Environment
from omnigibson.macros import gm, macros
from omnigibson.robots import REGISTERED_ROBOTS
from omnigibson.utils.data_utils import merge_scene_files
from omnigibson.utils.python_utils import create_object_from_init_info, h5py_group_to_torch
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)
log.setLevel(logging.INFO)

# --- Continuous scene-separation invariant (see _init_separation_baseline) ---------------
# Steps between separation checks. Default 250: at the measured multi-scene step
# costs (~0.1 s/step at num_envs=2, ~0.9 s at 10) one check per 250 steps is far
# under 1% of the batch wall (measured: see the "Separation invariant" line each
# batch logs), while still sampling every demo's replay ~25-30 times.
DEFAULT_SEPARATION_CHECK_EVERY = 250
# Objects sampled per scene by the periodic check. The baseline uses every shared
# object; the periodic sample keeps the six cell-defining extremes plus an even
# spread over the sorted names, so its box tracks the full cloud's box closely.
DEFAULT_SEPARATION_SAMPLE_OBJECTS = 24
# Fraction of the batch's own derived clearance that must remain at every check.
# The clearance is (tile geometry - content extents), i.e. the empty space the
# loader's tiling leaves between two scenes' object clouds -- 39.6 m on task-0001
# at num_envs=2. Objects legitimately move within their scene during a replay
# (a robot carries trash across a room), so the invariant cannot demand the full
# clearance; it demands a quarter of it, which no room-scale motion can consume
# (an object would have to travel ~30 m out of its own cell on task-0001) while a
# collapse -- the failure this exists to catch -- makes the gap negative at once.
SEPARATION_MARGIN_FRACTION = 0.25


def points_aabb(points):
    """Axis-aligned bounds ``[[lo, hi], [lo, hi], [lo, hi]]`` of an iterable of 3-vectors."""
    pts = list(points)
    assert pts, "cannot take the AABB of an empty point set"
    return [[min(p[axis] for p in pts), max(p[axis] for p in pts)] for axis in range(3)]


def aabb_gap(box_a, box_b):
    """Largest per-axis separation between two boxes: > 0 iff they are disjoint.

    Negative values are overlap depth along the least-separated axis.
    """
    return max(max(box_a[axis][0] - box_b[axis][1], box_b[axis][0] - box_a[axis][1]) for axis in range(3))


def union_box(boxes):
    """Smallest box containing every box in @boxes."""
    return [[min(b[axis][0] for b in boxes), max(b[axis][1] for b in boxes)] for axis in range(3)]


def shift_box(box, offset):
    """@box translated by the 3-vector @offset."""
    return [[box[axis][0] + offset[axis], box[axis][1] + offset[axis]] for axis in range(3)]


class SceneSeparationError(RuntimeError):
    """The batch's scenes stopped being geometrically disjoint mid-replay.

    Raised, never converted into per-demo rows inside this module: spatial
    tiling is the only cross-scene isolation that exists (there is no
    cross-scene collision filtering), so overlapping cells mean one demo's
    objects can physically push another demo's, and EVERY verdict in the batch
    is void -- not just one demo's. Worked example: before the ``Scene._load_state``
    fix in this PR, a collapse into scene 0's cell flipped four of twenty
    task-0001 demos to fail and moved a fifth's success step, deterministically,
    with nothing in the result rows hinting at it.

    Callers are expected to treat this as batch-fatal -- record it against every
    demo of the failing batch rather than leaving the output silently short, and
    account for it separately from demo-level failures.

    Attributes:
        scenes (tuple of int): the two scene indices whose cells came too close.
        measured_gap_m (float): the measured gap between their object AABBs.
        required_gap_m (float): the minimum this batch derived from its own
            tile geometry and content extents.
        step (int or None): replay step of the violation; None at batch start.
        batch (int): index of the replay_batch call.
    """

    def __init__(self, message, scenes, measured_gap_m, required_gap_m, step, batch):
        super().__init__(message)
        self.scenes = scenes
        self.measured_gap_m = measured_gap_m
        self.required_gap_m = required_gap_m
        self.step = step
        self.batch = batch


class BatchObjectSetMismatchError(ValueError):
    """A demo's recorded scene is incompatible with the batch scene topology.

    Raised (never converted into a replay_error row) so that the caller /
    sharder splits the batch instead of silently dropping the demo.

    Recorded scene files of one task's demos are NOT byte-identical (observed
    on task-0001: two collection-code vintages saved 102- vs 129-object
    partial scenes, nested, plus per-session ``expected_file_hash`` drift on
    every shared object), so the batch invariant is deliberately NOT recorded
    object-set equality. What batching genuinely requires -- and what this
    error reports the violation of -- is that every demo replays correctly in
    the ONE scene built from the first demo's merged scene file:

    1. every object the demo recorded exists in the batch scene (its state
       rows' uuid lookups must resolve),
    2. the demo's recording-only objects (recorded but absent from the full
       scene file) equal the reference's, so merge(full, recorded_i) yields
       the same object set as the batch scene,
    3. the recorded robot object names match the reference's, and
    4. every recorded object's init_info matches what the batch scene was
       built with, structurally: compared with ``expected_file_hash`` removed
       (asset-version metadata; the object loader itself only warns on a hash
       mismatch) and robots excluded (identity checked by name in 3; the
       batch keeps the reference robot entry via ``keep_robot_from="b"``),
    5. the recorded initial system registry names match the reference's (the
       batch scene is built with the reference's systems).
    """


class VectorReplayHarness:
    """Replays batches of recorded demos in one multi-env OmniGibson environment.

    Usage:
        harness = VectorReplayHarness(task_id=..., task_name=...)
        env = harness.build_batch_env(all_demo_paths, num_envs, full_scene_file, load_room_instances)
        # (optional caller-side env tweaks, e.g. robot base mass)
        rows = harness.replay_batch(demo_paths_chunk)  # len(chunk) <= num_envs

    **Slots are not interchangeable.** Demo k of a batch plays in
    ``env.scenes[k]``, and the scenes differ in ways that reach the numbers:

    - Slot 0 sits at the world origin, so its recorded scene-pose header equals
      its current pose and ``Scene._load_state`` takes the exact-equality fast
      path -- slot 0's injection is bit-identical to a num_envs=1 replay. Every
      slot > 0 is re-based by ``rel = cur o rec^-1`` instead.
    - Slots are tiled along +x, and kinematic predicates (Inside, OnTop, ...)
      evaluate in world coordinates, so the float32 noise floor grows with tile
      distance: one ulp is ~3.8 um at x = 57 m and ~61 um at x = 515 m.

    Every result row therefore carries its ``slot`` (and ``batch``). Without
    them, "do failures correlate with slot?" -- the sharpest available detector
    for a residual multi-scene defect -- is not answerable from the campaign's
    output at all.

    **The separation invariant is re-tested continuously, not once.** Scene
    isolation is a mechanism the replay depends on and cannot observe in its own
    verdicts (a collapsed batch produces plausible-looking rows -- which is how
    the collapse fixed in this PR survived several rounds of parity testing: an
    early test passed *because of* it, since stacked scenes trivially agree).
    This harness therefore re-measures separation at every batch start and every
    ``separation_check_every`` steps, against a threshold derived from each
    batch's own tile geometry and content extents, and raises
    SceneSeparationError on violation. Every result row also carries the smallest
    gap measured against its own slot (``min_scene_gap_m``), so the campaign ends
    with a distribution to inspect rather than one bit.

    Per-demo result row schema (all rows, in input order):
        demo_id (int, or file basename if the filename does not match
            ``episode_<N>.hdf5`` -- identifier only, documented here),
        task_id (int), task_name (str),
        demo_group (str or None): HDF5 group replayed, e.g. "demo_3",
        n_steps (int or None): replayed steps in that group -- the minimum
            dataset length across action/state/state_size/reward/terminated/
            truncated, matching upstream playback's zip iteration semantics,
        status (str): "pass" (fresh success at final step), "fail", or
            "replay_error" (designed malformed-demo row, see module docstring),
        fresh_success_final (bool or None): env.task.success[i] at the demo's
            final recorded step,
        first_fresh_success_step (int or None): first step where the fresh
            (non-timeout) terminated signal was True,
        recorded_terminated_final (bool or None): last recorded terminated,
        per_step_terminated_mismatches (int or None): count of steps where
            fresh terminated != recorded terminated,
        goal_status_final (dict or None): {"satisfied": [...], "unsatisfied":
            [...]} predicate index lists at the final step,
        wall_s (float): seconds from replay_batch entry until this demo's
            final step completed (or until its load error),
        num_envs (int),
        batch (int): 0-based index of the replay_batch call within this
            harness instance (one shard = one instance),
        slot (int or None): scene index this demo played in --
            ``env.scenes[slot]``. None only for a demo that failed to load and
            was therefore never assigned a scene. Slots are inhomogeneous BY
            CONSTRUCTION (see class docstring's "Slots are not interchangeable"),
            so this is the field that makes slot-correlated failures
            computable after the fact,
        terminal_transition_step (int or None): the demo's final recorded step,
            when a transition recorded there was applied by the terminal pass
            and the verdict below was read from the extra step that followed.
            None for every ordinary demo. Makes the class computable in the
            campaign's output instead of looking like an unexplained fail,
        unreplayable_recording_step (int or None): set when the demo matches the
            recording-boundary rule (see unreplayable_recording_step): its only
            recorded success is produced by a transition on its final replayed
            step, so no replay can score it. Such a demo is still REPLAYED and
            its partial goal_status kept -- the row's status becomes
            "unreplayable_recording" rather than "fail". A row carrying this field
            with status "pass" means the rule mis-classified the demo and is
            surfaced as an anomaly, which is the rule's own self-check,
        min_scene_gap_m (float or None): smallest world-space gap measured
            between this demo's scene and any other scene of the batch, over
            every separation check of the batch (batch start + every
            ``separation_check_every`` steps + the final step). None at
            num_envs=1, where no scene pair exists, and on replay_error rows.
            Recorded so the campaign yields a distribution of realized
            isolation, not just the pass/fail of the invariant,
        error (None or "ExcType: message").
    """

    _DEMO_FNAME_RE = re.compile(r"episode_(\d+)\.hdf5$")

    def __init__(
        self,
        task_id,
        task_name,
        separation_check_every=DEFAULT_SEPARATION_CHECK_EVERY,
        separation_sample_objects=DEFAULT_SEPARATION_SAMPLE_OBJECTS,
    ):
        """
        Args:
            task_id (int): 2026-challenge task id, recorded into every result row
            task_name (str): task (activity) name, recorded into every result row
            separation_check_every (int): steps between in-run separation checks
                (>= 1). The batch's first and last steps are always checked.
            separation_sample_objects (int): objects sampled per scene by the
                periodic check (>= 2). The batch-start baseline always uses every
                shared object regardless of this value.
        """
        self.task_id = int(task_id)
        self.task_name = task_name
        assert int(separation_check_every) >= 1, f"separation_check_every must be >= 1, got {separation_check_every}"
        assert (
            int(separation_sample_objects) >= 2
        ), f"separation_sample_objects must be >= 2, got {separation_sample_objects}"
        self.separation_check_every = int(separation_check_every)
        self.separation_sample_objects = int(separation_sample_objects)
        self.env = None
        self.num_envs = None
        self._batch_index = 0  # 0-based replay_batch call counter, stamped into every row
        # Per-batch separation state, (re)initialized by _init_separation_baseline
        self._sep_batch = None
        self._sep_pairs = []
        self._sep_sample_names = None
        self._sep_required_gap_m = None
        self._sep_min_gap_by_slot = {}
        self._sep_check_count = 0
        self._sep_check_s = 0.0
        self._sep_baseline_s = 0.0
        self._batch_t0 = None  # monotonic clock of the current replay_batch call
        self._merged_scene_file = None
        self._restore_files = None
        self._reference_demo_path = None
        self._reference_object_names = None
        self._reference_recording_only_names = None
        self._reference_robot_names = None
        self._reference_system_names = None
        self._full_scene_object_names = None
        self._merged_object_names = None
        self._merged_init_info_stripped = None

    @staticmethod
    def _read_recorded_scene_file(demo_h5_path):
        """Reads the recorded (partial) scene file dict from a demo HDF5's attrs."""
        with h5py.File(demo_h5_path, "r") as f:
            return json.loads(f["data"].attrs["scene_file"])

    @staticmethod
    def _recorded_object_names(recorded_scene_file):
        return set(recorded_scene_file["objects_info"]["init_info"].keys())

    @staticmethod
    def _recorded_robot_names(recorded_scene_file):
        """Names of robot objects in a recorded scene file (merge_scene_files' detection rule)."""
        return {
            name
            for name, info in recorded_scene_file["objects_info"]["init_info"].items()
            if info["class_name"] == "Robot" or info["class_name"].lower() in REGISTERED_ROBOTS
        }

    @staticmethod
    def _stripped_init_info(info):
        """Canonical structural form of one object's init_info.

        ``expected_file_hash`` is removed before comparison: it is per-session
        asset-version metadata (recorded collection sessions of one task
        demonstrably drift on it), and the object loader itself only warns on
        a hash mismatch. Everything else (category, model, scale, rooms, ...)
        is structural and must match for a shared batch scene to be valid.
        """
        args = {key: value for key, value in info.get("args", {}).items() if key != "expected_file_hash"}
        return json.dumps({**info, "args": args}, sort_keys=True)

    def _check_batch_compat(self, path, recorded_scene_file):
        """Raises BatchObjectSetMismatchError unless @recorded_scene_file can replay in the batch scene.

        The five checks are documented on BatchObjectSetMismatchError. Called
        for every demo at build time and again from _read_demo as a backstop
        for direct replay_batch callers.
        """
        problems = []
        names = self._recorded_object_names(recorded_scene_file)
        missing = sorted(names - self._merged_object_names)
        if missing:
            problems.append(f"objects recorded by this demo are absent from the batch scene: {missing}")
        recording_only = names - self._full_scene_object_names
        if recording_only != self._reference_recording_only_names:
            gained = sorted(recording_only - self._reference_recording_only_names)
            lost = sorted(self._reference_recording_only_names - recording_only)
            problems.append(
                f"recording-only objects differ from the batch reference (extra here: {gained}, "
                f"missing here: {lost}), so this demo's merged scene topology differs from the batch scene"
            )
        robots = self._recorded_robot_names(recorded_scene_file)
        if robots != self._reference_robot_names:
            problems.append(
                f"recorded robot objects {sorted(robots)} differ from the batch reference "
                f"{sorted(self._reference_robot_names)}"
            )
        systems = set(recorded_scene_file["state"]["registry"]["system_registry"].keys())
        if systems != self._reference_system_names:
            problems.append(
                f"recorded initial systems {sorted(systems)} differ from the batch reference "
                f"{sorted(self._reference_system_names)}"
            )
        drifted = [
            name
            for name in sorted(names - robots)
            if name in self._merged_init_info_stripped
            and self._stripped_init_info(recorded_scene_file["objects_info"]["init_info"][name])
            != self._merged_init_info_stripped[name]
        ]
        if drifted:
            shown = drifted[:10]
            suffix = f" (+{len(drifted) - len(shown)} more)" if len(drifted) > len(shown) else ""
            problems.append(
                f"init_info differs structurally (beyond expected_file_hash) from the batch scene "
                f"for objects: {shown}{suffix}"
            )
        if problems:
            raise BatchObjectSetMismatchError(
                f"Demo file {path} cannot replay in the batch scene built from "
                f"{self._reference_demo_path}: " + "; ".join(problems) + ". Split the batch so "
                "incompatible demos get their own env build."
            )

    def build_batch_env(self, demo_h5_paths, num_envs, full_scene_file, load_room_instances):
        """Builds the multi-env playback Environment shared by all batches.

        Applies the same config surgery ``DataPlaybackWrapper.create_from_hdf5``
        performs on the ``include_contacts=True`` path, reading the base config
        from the FIRST demo file's attrs, plus ``num_envs=N``. Checks that
        every given demo file's recording is compatible with the batch scene
        built from the first file (the five checks documented on
        BatchObjectSetMismatchError -- deliberately NOT recorded-object-set
        equality, which real same-task demos violate); raises
        BatchObjectSetMismatchError naming the violations otherwise.

        Mirroring ``DataPlaybackWrapper.__init__``: when ``load_room_instances``
        is specified the per-scene restore target for each batch is a snapshot
        of the actually-loaded scene (``scene.save(as_dict=True)``) rather than
        the merged full-scene file -- the env loads only the requested rooms,
        so restoring from the merged file would re-add every unloaded room's
        objects.

        Args:
            demo_h5_paths (list of str): ALL demo files this harness will
                replay (across all subsequent replay_batch calls), so the
                object-set assert covers the full shard.
            num_envs (int): number of parallel scenes (batch capacity)
            full_scene_file (str): path to the full task scene .json to merge
                with the recorded (partial) scene file (keep_robot_from="b")
            load_room_instances (None or list of str): room instances to load

        Returns:
            Environment: the constructed multi-env environment (also stored on
                self.env)
        """
        assert self.env is None, "build_batch_env() may only be called once per harness instance"
        assert len(demo_h5_paths) > 0, "At least one demo file is required to build the batch env"
        assert int(num_envs) >= 1, f"num_envs must be >= 1, got {num_envs}"
        # Mirrors DataPlaybackWrapper.__init__'s fence: transitions are propagated manually here
        assert not gm.ENABLE_TRANSITION_RULES, "Transition rules must be disabled for vector replay!"

        # Read the base config and recorded scene file from the first demo file
        # (mirrors create_from_hdf5, which reads them from its single input file)
        with h5py.File(demo_h5_paths[0], "r") as f:
            config = json.loads(f["data"].attrs["config"])
            recorded_scene_file = json.loads(f["data"].attrs["scene_file"])

        # Batch-compat reference invariants come from the first demo's recording and the full
        # scene file; the compat check itself runs after the merge below (it needs the merged
        # object set). NOTE: merge_scene_files mutates robot init_info class_names in its inputs,
        # so everything reference-side is computed from the recording BEFORE merging.
        with open(full_scene_file, "r") as json_file:
            full_scene_json = json.load(json_file)
        self._reference_demo_path = demo_h5_paths[0]
        self._reference_object_names = self._recorded_object_names(recorded_scene_file)
        self._full_scene_object_names = set(full_scene_json["objects_info"]["init_info"].keys())
        self._reference_recording_only_names = self._reference_object_names - self._full_scene_object_names
        self._reference_robot_names = self._recorded_robot_names(recorded_scene_file)
        self._reference_system_names = set(recorded_scene_file["state"]["registry"]["system_registry"].keys())

        # --- Config surgery, mirroring create_from_hdf5's include_contacts=True path ---
        # Minimize physics leakage during playback (we need to take an env step when loading state)
        config["env"]["action_frequency"] = 1000.0
        config["env"]["rendering_frequency"] = 1000.0
        config["env"]["physics_frequency"] = 1000.0
        # Make sure obs space is flattened (matches upstream playback config)
        config["env"]["flatten_obs_space"] = True
        # The one multi-env addition: N scenes (env_base reads env_config["num_envs"])
        config["env"]["num_envs"] = int(num_envs)

        # Merge the full scene file with the recorded (partial) scene file, keeping the robot
        # from the recording
        merged_scene_file = merge_scene_files(scene_a=full_scene_json, scene_b=recorded_scene_file, keep_robot_from="b")
        config["scene"]["scene_file"] = merged_scene_file

        # Batch-compat invariants derived from the merged (= actually built) scene, then check
        # every other demo against them (the reference demo defines them, so it trivially passes)
        self._merged_object_names = set(merged_scene_file["objects_info"]["init_info"].keys())
        self._merged_init_info_stripped = {
            name: self._stripped_init_info(info)
            for name, info in merged_scene_file["objects_info"]["init_info"].items()
            if not (info["class_name"] == "Robot" or info["class_name"].lower() in REGISTERED_ROBOTS)
        }
        for path in demo_h5_paths[1:]:
            self._check_batch_compat(path, self._read_recorded_scene_file(path))
        # Overwrite room types to avoid loading room types from the recorded config
        config["scene"]["load_room_types"] = None
        config["scene"]["load_room_instances"] = load_room_instances

        # Load from the cached scene file: no online sampling, no presampled robot pose
        if config["task"]["type"] == "BehaviorTask":
            config["task"]["online_object_sampling"] = False
            config["task"]["use_presampled_robot_pose"] = False
        # No task observations are recorded by this harness
        config["task"]["include_obs"] = False

        # Additional config objects are already baked into the recorded scene file
        config["objects"] = []

        # Proprio-only robot observations; no sensor name filtering (upstream defaults)
        for robot_cfg in config["robots"]:
            robot_cfg["obs_modalities"] = ["proprio"]
            robot_cfg["include_sensor_names"] = None
            robot_cfg["exclude_sensor_names"] = None

        env = Environment(configs=config)

        # Automatic reset would re-initialize a slot the moment its demo terminates (and zero
        # task.success before we can read it): the recorded config must not enable it.
        assert not env.env_config["automatic_reset"], (
            "Recorded config has automatic_reset enabled; vector replay requires it disabled "
            "(a terminating demo would be silently reset mid-batch)."
        )

        # Stabilize skipped objects (mirrors DataPlaybackWrapper.__init__): whatever load_state
        # skips must have been asleep during collection, so keeping it still is safe.
        with macros.unlocked():
            macros.utils.registry_utils.STABILIZE_SKIPPED_OBJECTS = True

        # Per-scene restore target for each batch (mirrors DataPlaybackWrapper.__init__):
        # with load_room_instances we loaded more rooms than the recorded scene file but fewer
        # than the full scene, so snapshot the actually-loaded scenes; otherwise the merged file
        # itself is the correct target.
        if load_room_instances is not None:
            self._restore_files = [scene.save(as_dict=True) for scene in env.scenes]
        else:
            self._restore_files = [merged_scene_file for _ in env.scenes]

        self.env = env
        self.num_envs = int(num_envs)
        self._merged_scene_file = merged_scene_file
        return env

    def _read_demo(self, path, demo_id):
        """Loads one demo's replay data from HDF5 into torch tensors.

        Chooses the LAST ``demo_N`` group (numerically sorted by N), matching
        the production replay default in scripts/learning/replay_obs.py.
        Unequal dataset lengths are tolerated by iterating the shortest
        (upstream zip semantics; see inline comment). Raises on any malformed
        content (missing groups/keys, zero steps, non-2D action/state); the
        caller converts those raises into replay_error rows. A batch-scene
        incompatibility raises BatchObjectSetMismatchError, which the caller
        re-raises.
        """
        with h5py.File(path, "r") as f:
            data_grp = f["data"]
            recorded_scene_file = json.loads(data_grp.attrs["scene_file"])
            self._check_batch_compat(path, recorded_scene_file)

            demo_nums = sorted(int(key.split("_", 1)[1]) for key in data_grp.keys() if key.startswith("demo_"))
            if not demo_nums:
                raise KeyError(f"No demo_N groups found in {path}")
            # Last demo group (retakes overwrite earlier attempts) -- production replay default
            episode_id = demo_nums[-1]
            traj_grp = data_grp[f"demo_{episode_id}"]
            transitions = json.loads(traj_grp.attrs["transitions"])
            tensors = h5py_group_to_torch(traj_grp)

        init_metadata = tensors["init_metadata"]
        action = tensors["action"]
        state = tensors["state"]
        state_size = tensors["state_size"]
        reward = tensors["reward"]
        terminated = tensors["terminated"]
        truncated = tensors["truncated"]

        if action.ndim != 2 or state.ndim != 2:
            raise ValueError(
                f"Demo group demo_{episode_id} in {path} has malformed action/state datasets: "
                f"expected 2-D (n_steps, dim) arrays, got action ndim={action.ndim}, "
                f"state ndim={state.ndim}"
            )
        # Upstream playback iterates zip(action, state, state_size, reward, terminated,
        # truncated), so the SHORTEST dataset governs the step count. Raw demos really do
        # differ (action sometimes has n-1 rows vs state's n); mirror the zip semantics and
        # record the iterated length as n_steps.
        lengths = {
            "action": int(action.shape[0]),
            "state": int(state.shape[0]),
            "state_size": int(state_size.shape[0]),
            "reward": int(reward.shape[0]),
            "terminated": int(terminated.shape[0]),
            "truncated": int(truncated.shape[0]),
        }
        n_steps = min(lengths.values())
        if n_steps == 0:
            raise ValueError(f"Demo group demo_{episode_id} in {path} has zero recorded steps")
        if len(set(lengths.values())) > 1:
            log.info(
                f"Demo group demo_{episode_id} in {path} has unequal dataset lengths {lengths}; "
                f"iterating the shortest ({n_steps} steps), matching upstream zip semantics"
            )
        # Transitions recorded past the replayed range can never be applied (there is no
        # such step to apply them at). Say so: a dropped transition changes what the demo
        # can achieve, so it is reported, never silently ignored.
        unreachable_transitions = sorted(step for step in (int(k) for k in transitions) if step > n_steps - 1)
        if unreachable_transitions:
            log.warning(
                f"Demo group demo_{episode_id} in {path} records transition(s) at step(s) "
                f"{unreachable_transitions}, beyond the {n_steps} replayed steps (n_steps is the "
                f"min over the dataset lengths {lengths}); they cannot be applied and this demo's "
                f"verdict is evaluated without them"
            )

        return {
            "demo_id": demo_id,
            "path": path,
            "episode_id": episode_id,
            "n_steps": n_steps,
            "transitions": transitions,
            "recorded_scene_file": recorded_scene_file,
            "init_metadata": init_metadata,
            "action": action,
            "state": state,
            "state_size": state_size,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        }

    @staticmethod
    def terminal_transition_step(demo):
        """The demo's final replayed step if a transition is recorded there, else None.

        A transition recorded on the final step produces state the recording
        cannot contain (there is no later row), so its effect on the goal can only
        be seen by stepping past the recording -- what _apply_terminal_transitions
        does. Transitions recorded BEYOND the replayed range (possible because
        n_steps is the min over unequal dataset lengths) are never applied; they
        are reported by _read_demo at WARNING rather than dropped silently.

        Args:
            demo (dict): loaded demo from _read_demo.

        Returns:
            int or None: the final step index when it carries a transition.
        """
        final_step = demo["n_steps"] - 1
        return final_step if str(final_step) in demo["transitions"] else None

    @staticmethod
    def unreplayable_recording_step(demo):
        """The final replayed step when this demo's only recorded success is unreachable.

        The recording-boundary bucket, deliberately CONJUNCTIVE:

        (A) a transition is recorded AT or PAST the final replayed step
            (``max(transition steps) >= n_steps - 1``), so its products can never
            appear in a recorded state row -- there is no later row to carry them.
            Measured on task-0040: 400010's state_size grows +25 floats one step
            after its transition, 400100's never grows at all, so its popcorn
            exists in no row and a system-add can only produce an EMPTY system; and
        (B) the recording's ``terminated`` first turns True only on that final step,
            so the campaign's "fresh success one step later" convention has no step
            to observe it on either.

        (A) alone is NOT sufficient and must not be used alone: a demo may carry a
        final-step transition and have satisfied its goal earlier, in which case
        replay scores it normally. (B) alone is the physics-mediated variant of the
        same boundary. Only both together mean no replay -- ours or upstream's --
        can score the demo, at any num_envs, in any slot.

        Distinct from ``terminal_transition_step``, which is an EXACT final-step
        match and drives whether the terminal pass has a transition to apply; a
        transition recorded PAST the replayed range is never applied (it is warned
        about in _read_demo) but still counts for (A).

        Args:
            demo (dict): loaded demo from _read_demo.

        Returns:
            int or None: the final replayed step index when the demo is in the bucket.
        """
        final_step = demo["n_steps"] - 1
        steps = [int(key) for key in demo["transitions"]]
        if not steps or max(steps) < final_step:
            return None
        terminated = demo["terminated"]
        first_true = next((i for i in range(demo["n_steps"]) if bool(terminated[i])), None)
        if first_true is None or first_true < final_step:
            return None
        return final_step

    def _disable_robot_control(self):
        """Disables joint control on every scene's robots.

        Mirrors DataPlaybackWrapper.playback_episode's include_robot_control=False
        branch: control_enabled=False plus every controller dof forced into
        effort mode with zero gains, which keeps the robots still between state
        injections.
        """
        for scene in self.env.scenes:
            for robot in scene.robots:
                robot.control_enabled = False
                # robot.controllers maps controller name -> (group_key, controller_idx)
                for controller in robot.controllers.values():
                    for dof in ControllerView.get_dof_idx(controller[0]).tolist():
                        dof_joint = robot.joints[robot.dof_names_ordered[dof]]
                        dof_joint.set_control_type(
                            control_type=ControlType.EFFORT,
                            kp=None,
                            kd=None,
                        )

    def _apply_transitions(self, scene, cur_transitions, recorded_scene_file):
        """Applies one recorded transition event to @scene.

        Mirrors the transition block in DataPlaybackWrapper.playback_episode,
        parametrized by scene (upstream hardcodes og.sim.scenes[0]) and with
        added objects parked in the SCENE frame instead of the world frame
        (see module docstring). Added objects are stamped with the recording's
        non-kin state vocabulary from @recorded_scene_file (the owning demo's,
        NOT the batch's -- recordings differ within a task), matching upstream:
        they have no per-name entry in the recorded scene file, so the
        _align_scene_object_states_with_recorded_schema pass at batch prepare
        time can never reach them. The caller is responsible for the trailing
        og.sim.step().
        """
        added_systems = set(cur_transitions["systems"]["add"])
        removed_systems = set(cur_transitions["systems"]["remove"])
        for add_sys_name in cur_transitions["systems"]["add"]:
            scene.get_system(add_sys_name, force_init=True)
        for remove_sys_name in cur_transitions["systems"]["remove"]:
            scene.clear_system(remove_sys_name)
        for remove_obj_name in cur_transitions["objects"]["remove"]:
            if _is_system_particle_template_name(remove_obj_name, removed_systems):
                continue
            obj = scene.object_registry("name", remove_obj_name)
            scene.remove_object(obj)
        recorded_non_kin_names = _recorded_non_kin_state_name_union(recorded_scene_file)
        for j, add_obj_info in enumerate(cur_transitions["objects"]["add"]):
            if _is_system_particle_template_info(add_obj_info, added_systems):
                continue
            obj = create_object_from_init_info(add_obj_info)
            if recorded_non_kin_names is not None:
                obj._recorded_non_kin_state_names = set(recorded_non_kin_names)
            scene.add_object(obj)
            # Scene frame, NOT world frame: world-frame parking (upstream) would place every
            # env's added objects at the same world location, i.e. inside another env's cell.
            obj.set_position_orientation(position=th.ones(3) * 100.0 + th.ones(3) * 5 * j, frame="scene")

    def _make_error_row(self, demo_id, error, wall_s, batch, slot):
        """Builds the designed replay_error result row (see class docstring).

        Args:
            demo_id (int or str): identifier of the demo that failed.
            error (str): "ExcType: message".
            wall_s (float): seconds from replay_batch entry until the failure.
            batch (int): index of the replay_batch call this row belongs to.
            slot (int or None): scene the demo played in; None when it never
                got one (a load failure happens before slot assignment).
        """
        return {
            "demo_id": demo_id,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "demo_group": None,
            "n_steps": None,
            "status": "replay_error",
            "fresh_success_final": None,
            "first_fresh_success_step": None,
            "recorded_terminated_final": None,
            "per_step_terminated_mismatches": None,
            "goal_status_final": None,
            "wall_s": round(wall_s, 3),
            "num_envs": self.num_envs,
            "batch": batch,
            "slot": slot,
            "terminal_transition_step": None,
            "unreplayable_recording_step": None,
            # A demo that never got a scene has no inter-scene gap of its own.
            "min_scene_gap_m": None,
            "error": error,
        }

    @classmethod
    def demo_id_for_path(cls, path):
        """Row identifier for a demo file: its ``episode_<N>.hdf5`` number, else the basename."""
        basename = os.path.basename(path)
        match = cls._DEMO_FNAME_RE.search(basename)
        return int(match.group(1)) if match else basename

    def make_separation_rows(self, demo_h5_paths, error):
        """One ``status="scene_separation_error"`` row per demo of a voided batch.

        A SceneSeparationError says the batch's scenes were not isolated, so no
        demo in it has a meaningful verdict -- these rows record that fact (with
        the gap that tripped, per slot) instead of leaving the output silently
        short. The caller is expected to write them and then fail the batch's
        unit of work; they are never a substitute for a verdict.

        Args:
            demo_h5_paths (list of str): the failing batch's demo files, in the
                order they were passed to replay_batch (index == scene slot).
            error (str): "ExcType: message" of the SceneSeparationError.

        Returns:
            list of dict: rows in the same schema as replay_batch's, with every
                verdict field None.
        """
        wall_s = 0.0 if self._batch_t0 is None else time.monotonic() - self._batch_t0
        rows = []
        for slot, path in enumerate(demo_h5_paths):
            row = self._make_error_row(
                demo_id=self.demo_id_for_path(path),
                error=error,
                wall_s=wall_s,
                batch=self._sep_batch,
                slot=slot,
            )
            row["status"] = "scene_separation_error"
            row["min_scene_gap_m"] = (
                round(self._sep_min_gap_by_slot[slot], 4) if slot in self._sep_min_gap_by_slot else None
            )
            rows.append(row)
        return rows

    @staticmethod
    def _jsonable_goal_status(goal_status):
        """Converts a goal_status info dict into plain-int JSON-serializable lists."""
        return {
            "satisfied": [int(idx) for idx in goal_status["satisfied"]],
            "unsatisfied": [int(idx) for idx in goal_status["unsatisfied"]],
        }

    def _prepare_batch(self, demos):
        """Prepares every scene for a batch of loaded demos (slot k plays in scenes[k]).

        The per-scene mirror of playback_episode's pre-injection lifecycle:
        every scene restores its batch restore target -> ONE og.sim.stop() ->
        per-demo init_metadata applied to its scene's objects -> ONE
        og.sim.play() -> env.reset() -> per-demo recorded non-kin schema
        alignment -> robot control disabled on every scene. Factored out of
        replay_batch so diagnostic probes exercise the exact production
        lifecycle.

        Args:
            demos (list of dict): loaded demos from _read_demo, at most num_envs
        """
        scenes = self.env.scenes
        # Restore every scene (including inert slots, so leftovers from the previous batch are
        # cleared) and update its initial file so env.reset() below resets to the restored state.
        for env_idx, scene in enumerate(scenes):
            scene.restore(self._restore_files[env_idx], update_initial_file=True)

        og.sim.stop()
        for slot, demo in enumerate(demos):
            scene = scenes[slot]
            init_metadata = demo["init_metadata"]
            for attr, vals in init_metadata.items():
                assert len(vals) == scene.n_objects, (
                    f"Demo {demo['demo_id']} init_metadata attr '{attr}' has {len(vals)} values "
                    f"but scene {slot} has {scene.n_objects} objects"
                )
            for i, obj in enumerate(scene.objects):
                for attr, vals in init_metadata.items():
                    val = vals[i]
                    setattr(obj, attr, val.item() if val.ndim == 0 else val)
        og.sim.play()

        self.env.reset()

        for slot, demo in enumerate(demos):
            _align_scene_object_states_with_recorded_schema(
                scene=scenes[slot],
                recorded_scene_file=demo["recorded_scene_file"],
            )

        self._disable_robot_control()

    # ------------------------------------------------------------------ separation invariant
    @staticmethod
    def _object_positions(scene, names):
        """World-frame positions ``{name: [x, y, z]}`` of @names present in @scene.

        Names absent from the registry are omitted: a recorded transition may
        legitimately ``remove_object`` one mid-replay, and the caller asserts the
        per-scene sample is non-empty rather than letting an empty set through.
        """
        out = {}
        for name in names:
            obj = scene.object_registry("name", name, None)
            if obj is None:
                continue
            pos, _ = obj.get_position_orientation(frame="world")
            out[name] = [float(v) for v in pos]
        return out

    def _raise_separation(self, pair, measured_gap, step, detail):
        """Raises SceneSeparationError naming the pair, the numbers, and the step."""
        i, j = pair
        origins = [[round(float(v), 4) for v in self.env.scenes[slot].get_position_orientation()[0]] for slot in (i, j)]
        where = "batch start (after state[0] injection)" if step is None else f"replay step {step}"
        raise SceneSeparationError(
            f"Scene separation violated at {where} of batch {self._sep_batch} "
            f"(task {self.task_id} {self.task_name}, num_envs={self.num_envs}): scenes {i} and {j} "
            f"(prim origins {origins[0]} and {origins[1]}) are {measured_gap:.4f} m apart, "
            f"below the {self._sep_required_gap_m:.4f} m minimum this batch derived from its own tile "
            f"geometry and object extents ({SEPARATION_MARGIN_FRACTION:g} x the batch-start clearance). "
            f"{detail} Spatial tiling is the only cross-scene isolation there is, so the whole batch's "
            f"verdicts are void.",
            scenes=(i, j),
            measured_gap_m=measured_gap,
            required_gap_m=self._sep_required_gap_m,
            step=step,
            batch=self._sep_batch,
        )

    def _init_separation_baseline(self, batch_index):
        """Derives this batch's separation threshold from the batch's own geometry.

        Called once per batch, immediately after the ``state[0]`` injection and
        its settle step, i.e. on exactly the geometry the replay is about to
        step. Everything it needs comes from the loaded scenes:

        1. each scene's prim pose (the isolation the loader set up),
        2. each scene's post-injection object cloud, taken relative to that
           scene's own prim pose -- the "content cell" each scene should hold.

        The union of those cells, replicated at each scene's prim pose, gives the
        pairwise *clearance* the tiling actually provides for this task, this
        room set and this recording. The threshold every later check must clear
        is ``SEPARATION_MARGIN_FRACTION`` of the smallest such clearance. Nothing
        is hardcoded: a task whose loaded rooms nearly fill their tile gets a
        proportionally smaller threshold, and if the tiling ever changes spacing
        the derived numbers follow.

        The derivation is self-validating against the failure it exists to catch:
        a collapsed scene's objects sit a whole tile away from its own origin, so
        its content cell is inflated by the tile offset, the union cell exceeds
        the tile spacing, and the predicted clearance goes negative -- caught
        here, at batch start, before a single step is replayed.

        Args:
            batch_index (int): index of the replay_batch call, for the log lines
                and any failure message.

        Raises:
            SceneSeparationError: if the tiling provides no clearance for these
                content extents, or if the measured post-injection gap is
                already below the derived threshold.
        """
        self._sep_batch = batch_index
        self._sep_sample_names = None
        self._sep_required_gap_m = None
        self._sep_min_gap_by_slot = {}
        self._sep_check_count = 0
        self._sep_check_s = 0.0
        self._sep_baseline_s = 0.0
        self._sep_pairs = [(i, j) for i in range(self.num_envs) for j in range(i + 1, self.num_envs)]
        if not self._sep_pairs:
            # One scene has no pair: the invariant's expectation is empty, not assumed.
            log.info("num_envs=1: no scene pair exists, so the continuous separation invariant is inactive")
            return

        t0 = time.monotonic()
        scenes = self.env.scenes
        origins = [[float(v) for v in scene.get_position_orientation()[0]] for scene in scenes]
        shared = set.intersection(*[set(scene.object_registry.get_dict("name").keys()) for scene in scenes])
        assert shared, (
            f"Batched scenes share no object names, so inter-scene separation cannot be measured "
            f"(per-scene object counts: {[scene.n_objects for scene in scenes]})"
        )
        names = sorted(shared)
        world = [self._object_positions(scenes[slot], names) for slot in range(self.num_envs)]
        for slot, positions in enumerate(world):
            assert positions, f"scene {slot} reported no positions for the {len(names)} shared objects"
        world_boxes = [points_aabb(positions.values()) for positions in world]
        own_boxes = [
            points_aabb([[p[axis] - origins[slot][axis] for axis in range(3)] for p in world[slot].values()])
            for slot in range(self.num_envs)
        ]
        cell = union_box(own_boxes)
        predicted = {
            pair: aabb_gap(shift_box(cell, origins[pair[0]]), shift_box(cell, origins[pair[1]]))
            for pair in self._sep_pairs
        }
        measured = {pair: aabb_gap(world_boxes[pair[0]], world_boxes[pair[1]]) for pair in self._sep_pairs}
        predicted_min = min(predicted.values())
        self._sep_required_gap_m = SEPARATION_MARGIN_FRACTION * predicted_min
        # The pair that is actually closest is the diagnosable one to name, in both
        # failure branches: with a single scene collapsed onto scene 0, the worst
        # PREDICTED pair is a pair of correctly-tiled scenes (the inflated union cell
        # spoils every prediction), while the worst MEASURED pair is the overlap itself.
        worst_pair = min(measured, key=measured.get)
        self._record_gaps(measured)
        if predicted_min <= 0.0:
            self._raise_separation(
                pair=worst_pair,
                measured_gap=measured[worst_pair],
                step=None,
                detail=(
                    f"The scenes' own content extents already exceed their tile spacing: the union content "
                    f"cell {[[round(v, 3) for v in ax] for ax in cell]} replicated at each scene's prim pose "
                    f"overlaps by {-predicted_min:.4f} m, which is what a collapsed injection looks like "
                    f"(a scene holding its objects a tile away from its own origin inflates the cell). "
                    f"Measured over all {len(names)} shared objects of each scene."
                ),
            )
        self._sep_baseline_s = time.monotonic() - t0
        if measured[worst_pair] < self._sep_required_gap_m:
            self._raise_separation(
                pair=worst_pair,
                measured_gap=measured[worst_pair],
                step=None,
                detail=(
                    f"Measured over all {len(names)} shared objects of each scene; the tiling's derived "
                    f"clearance is {predicted_min:.4f} m."
                ),
            )

        # Periodic sample: the six objects defining the content cell's faces (so the
        # sample's box tracks the full cloud's box) plus an even spread over the rest.
        # Transition-added objects are deliberately never sampled: they exist only after
        # a transition and are parked far outside the cell by construction (see
        # _apply_transitions), which would make the sample's box meaningless.
        extremes = set()
        reference = world[0]
        for axis in range(3):
            extremes.add(min(reference, key=lambda name: reference[name][axis]))
            extremes.add(max(reference, key=lambda name: reference[name][axis]))
        sample = set(extremes)
        stride = max(1, len(names) // max(1, self.separation_sample_objects - len(extremes)))
        sample.update(names[::stride])
        self._sep_sample_names = sorted(sample)
        log.info(
            f"Separation baseline for batch {batch_index}: derived clearance "
            f"{predicted_min:.3f} m, required minimum "
            f"{self._sep_required_gap_m:.3f} m, measured post-injection gap {measured[worst_pair]:.3f} m "
            f"(closest pair {worst_pair}) over {len(names)} shared objects; periodic checks sample "
            f"{len(self._sep_sample_names)} objects per scene every {self.separation_check_every} steps"
        )

    def _record_gaps(self, gaps):
        """Folds a check's per-pair gaps into each slot's running minimum."""
        for (i, j), gap in gaps.items():
            for slot in (i, j):
                current = self._sep_min_gap_by_slot.get(slot)
                if current is None or gap < current:
                    self._sep_min_gap_by_slot[slot] = gap

    def _check_separation(self, step):
        """Re-measures inter-scene separation on the sampled objects; raises on violation.

        Cheap by construction (``separation_sample_objects`` pose reads per scene)
        and compared against the threshold the batch derived at its start from
        every shared object -- the sample's box is a subset of the full cloud's,
        so its gap is never smaller than the full-cloud gap and the comparison
        cannot false-positive from sampling.

        Args:
            step (int): replay step, for the failure message.
        """
        if self._sep_required_gap_m is None:
            return  # num_envs=1: no pair to measure (logged once at the baseline)
        t0 = time.monotonic()
        boxes = []
        for slot, scene in enumerate(self.env.scenes):
            positions = self._object_positions(scene, self._sep_sample_names)
            assert positions, (
                f"scene {slot} holds none of the {len(self._sep_sample_names)} sampled objects at step {step}; "
                f"the separation sample was invalidated (object removals?) and the invariant cannot be measured"
            )
            boxes.append(points_aabb(positions.values()))
        gaps = {pair: aabb_gap(boxes[pair[0]], boxes[pair[1]]) for pair in self._sep_pairs}
        self._record_gaps(gaps)
        self._sep_check_count += 1
        self._sep_check_s += time.monotonic() - t0
        worst_pair = min(gaps, key=gaps.get)
        if gaps[worst_pair] < self._sep_required_gap_m:
            self._raise_separation(
                pair=worst_pair,
                measured_gap=gaps[worst_pair],
                step=step,
                detail=f"Measured over {len(self._sep_sample_names)} sampled objects per scene.",
            )

    def _apply_terminal_transitions(self, demos, slots, t_batch_start):
        """Applies final-step transitions and re-reads those slots' verdicts, once per batch.

        Runs after the step loop, so nothing here can perturb another demo's
        recorded numbers: every demo has already reached its final step and had
        its verdict captured. For each slot in @slots:

        1. re-inject that demo's final recorded state row -- the topology still
           matches it (the transition has not been applied yet), and injecting
           makes the starting state exact rather than "whatever the hold left",
        2. apply the recorded transition, then ONE global ``og.sim.step()`` to
           initialize added objects (the same settle the in-loop path takes),
        3. take ONE batched ``env.step`` with every slot's last recorded action.
           This is the step whose ``terminated``/``success``/``goal_status`` can
           see the transition's products, and it is the step the recording itself
           does not contain.

        The other slots take that one extra step from their held state. Their
        verdicts are already final and their mismatch counters are not touched,
        so the only effect is one step of unobserved physics in their own cells --
        the same trade the in-loop post-transition settle step already makes.

        ``first_fresh_success_step`` for a repaired demo is ``n_steps`` -- one past
        the recording's last index, deliberately: it keeps the campaign's
        ``first_fresh_success == recorded_first_terminated + 1`` signature true for
        these demos too.

        Args:
            demos (list of dict): the batch's loaded demos, slot-aligned.
            slots (list of int): slots whose final step carries a transition.
            t_batch_start (float): the batch's monotonic clock origin, for wall_s.

        Returns:
            dict: {slot: (fresh_terminated, success, goal_status, wall_s)}.
        """
        env = self.env
        scenes = env.scenes
        for slot in slots:
            demo = demos[slot]
            final_step = demo["n_steps"] - 1
            scenes[slot].load_state(demo["state"][final_step, : int(demo["state_size"][final_step])], serialized=True)
            cur_transitions = demo["transitions"][str(final_step)]
            t_apply = time.monotonic()
            self._apply_transitions(
                scene=scenes[slot],
                cur_transitions=cur_transitions,
                recorded_scene_file=demo["recorded_scene_file"],
            )
            log.info(
                f"Terminal pass: applied demo {demo['demo_id']} (slot {slot}) final-step "
                f"transition at t={final_step} in {time.monotonic() - t_apply:.3f}s: "
                f"sys_add={cur_transitions['systems']['add']}, "
                f"sys_rm={cur_transitions['systems']['remove']}, "
                f"obj_add={[info['args']['name'] for info in cur_transitions['objects']['add']]}, "
                f"obj_rm={cur_transitions['objects']['remove']}"
            )
        t_settle = time.monotonic()
        og.sim.step()
        log.info(f"Terminal pass: global settle step took {time.monotonic() - t_settle:.3f}s")

        action_dim = int(demos[0]["action"].shape[1])
        action_rows = []
        for env_idx in range(self.num_envs):
            if env_idx < len(demos):
                demo = demos[env_idx]
                action_rows.append(demo["action"][demo["n_steps"] - 1].to(th.float32))
            else:
                action_rows.append(th.zeros(action_dim, dtype=th.float32))
        _, _, terminateds, _, infos = env.step(th.stack(action_rows, dim=0))

        results = {}
        for slot in slots:
            demo = demos[slot]
            results[slot] = (
                bool(terminateds[slot]),
                bool(env.task.success[slot].item()),
                self._jsonable_goal_status(infos[slot]["done"]["goal_status"]),
                time.monotonic() - t_batch_start,
            )
            log.info(
                f"Terminal pass: demo {demo['demo_id']} (slot {slot}) post-transition verdict at "
                f"step {demo['n_steps']} (one past its last recorded step): "
                f"terminated={results[slot][0]} success={results[slot][1]} goal={results[slot][2]}"
            )
        return results

    def replay_batch(self, demo_h5_paths, step_callback=None):
        """Replays up to num_envs demos in lockstep and returns per-demo result rows.

        Batch lifecycle (per-scene mirror of playback_episode):
        every scene restores its batch restore target -> ONE og.sim.stop() ->
        per-demo init_metadata applied to its scene's objects -> ONE
        og.sim.play() -> env.reset() -> per-demo recorded non-kin schema
        alignment -> robot control disabled on every scene -> per-demo state[0]
        injection -> one og.sim.step().

        Step loop (t = 0 .. max demo length - 1): active demos (t < n_steps)
        inject state[t]; finished demos re-inject their final state
        (deterministic hold). One batched env.step with active demos' recorded
        actions (finished demos re-apply their last action; env slots beyond
        len(demo_h5_paths) get zero actions and are inert because robot
        control is disabled). Fresh terminated/success/goal_status are read
        from the env.step results; recorded transitions are applied after the
        step against the owning scene, followed by one global og.sim.step()
        (matches upstream; this advances every other scene one extra physics
        step, which is harmless because their state is re-injected on the next
        iteration).

        Terminal pass: a demo whose final recorded step carries a transition gets
        its verdict re-read after that transition is applied (see
        _apply_terminal_transitions), which costs the batch one extra step. The
        in-loop verdict for such a demo is provisional; for every other demo the
        loop's verdict is final and nothing about its replay changes.

        Separation invariant: the batch derives its own minimum inter-scene gap
        from the post-injection geometry (``_init_separation_baseline``) and
        re-measures it every ``separation_check_every`` steps and on the final
        step. A violation raises SceneSeparationError -- the batch's verdicts are
        void, so it is never downgraded to a per-demo row here. Every returned
        row carries the smallest gap measured against its own slot.

        Malformed demos become replay_error rows (module docstring); rows are
        returned in the same order as @demo_h5_paths.

        Args:
            demo_h5_paths (list of str): demo files for this batch,
                1 <= len <= num_envs
            step_callback (None or Callable): diagnostic-only hook, called at
                the END of every step-loop iteration (after env.step and after
                any transition application + its global settle step) as
                ``step_callback(t=t, demos=demos, scenes=scenes,
                terminateds=terminateds, infos=infos)``. Exceptions propagate
                (they fail the batch); production replay passes None.

        Returns:
            list of dict: one result row per input demo (schema in the class
                docstring)
        """
        assert self.env is not None, "build_batch_env() must be called before replay_batch()"
        n_input = len(demo_h5_paths)
        assert (
            0 < n_input <= self.num_envs
        ), f"replay_batch() takes between 1 and num_envs={self.num_envs} demos, got {n_input}"

        t_batch_start = time.monotonic()
        self._batch_t0 = t_batch_start
        batch_index = self._batch_index
        self._sep_batch = batch_index
        self._batch_index += 1
        rows_by_input_idx = {}
        demos = []  # loaded demos, slot-aligned: demos[k] plays in env.scenes[k]
        demo_input_idx = []  # demos[k]'s index into demo_h5_paths

        for input_idx, path in enumerate(demo_h5_paths):
            # Identifier only: falls back to the file basename when the filename does not
            # match episode_<N>.hdf5 (documented in the class docstring row schema).
            demo_id = self.demo_id_for_path(path)
            try:
                demos.append(self._read_demo(path, demo_id))
                demo_input_idx.append(input_idx)
            except BatchObjectSetMismatchError:
                # Batch-membership violation: never a replay_error row (see module docstring)
                raise
            except Exception as e:
                # Designed continue: record the malformed demo and replay the rest
                error = f"{type(e).__name__}: {e}"
                log.error(f"Failed to load demo {demo_id} from {path}: {error}")
                rows_by_input_idx[input_idx] = self._make_error_row(
                    demo_id=demo_id,
                    error=error,
                    wall_s=time.monotonic() - t_batch_start,
                    batch=batch_index,
                    # The demo never entered `demos`, so it was never assigned a scene.
                    slot=None,
                )

        if not demos:
            return [rows_by_input_idx[i] for i in range(n_input)]

        env = self.env
        scenes = env.scenes
        max_len = max(demo["n_steps"] for demo in demos)
        log.info(
            f"Replaying batch of {len(demos)} demo(s) "
            f"({len(rows_by_input_idx)} load error(s)) over {max_len} steps at num_envs={self.num_envs}"
        )

        self._prepare_batch(demos)

        for slot, demo in enumerate(demos):
            scenes[slot].load_state(demo["state"][0, : int(demo["state_size"][0])], serialized=True)
        # Take one sim step to propagate the injected initial states
        og.sim.step()

        # Scene isolation is a precondition of every verdict below, and the verdicts
        # themselves cannot reveal its absence: derive this batch's separation threshold
        # from the geometry the replay is about to step, then re-check it as it runs.
        self._init_separation_baseline(batch_index)

        # --- Step loop ---
        action_dim = int(demos[0]["action"].shape[1])
        terminal_slots = []  # slots whose final step carries a transition (see the terminal pass)
        first_success_step = [None] * len(demos)
        mismatches = [0] * len(demos)
        final_success = [None] * len(demos)
        final_goal_status = [None] * len(demos)
        wall_s = [None] * len(demos)

        for t in range(max_len):
            if t % 1000 == 0:
                log.info(f"Replaying batch step {t}/{max_len}")

            # State injection: active demos load state[t]; finished demos re-inject their final
            # recorded state row (deterministic hold)
            for slot, demo in enumerate(demos):
                idx = t if t < demo["n_steps"] else demo["n_steps"] - 1
                scenes[slot].load_state(demo["state"][idx, : int(demo["state_size"][idx])], serialized=True)

            # Batched actions: recorded action for active demos, last action for finished demos,
            # zeros for inert slots (their robots have control disabled, so this is a no-op)
            action_rows = []
            for env_idx in range(self.num_envs):
                if env_idx < len(demos):
                    demo = demos[env_idx]
                    idx = t if t < demo["n_steps"] else demo["n_steps"] - 1
                    action_rows.append(demo["action"][idx].to(th.float32))
                else:
                    action_rows.append(th.zeros(action_dim, dtype=th.float32))
            actions = th.stack(action_rows, dim=0)

            _, _, terminateds, _, infos = env.step(actions)

            any_transition = False
            for slot, demo in enumerate(demos):
                if t >= demo["n_steps"]:
                    continue
                fresh_terminated = bool(terminateds[slot])
                if fresh_terminated and first_success_step[slot] is None:
                    first_success_step[slot] = t
                if fresh_terminated != bool(demo["terminated"][t]):
                    mismatches[slot] += 1
                if t == demo["n_steps"] - 1:
                    # Final recorded step: capture fresh success and goal status from this step.
                    # BehaviorTask._step_termination adds goal_status to the termination infos,
                    # which BaseTask.step nests under the "done" key of each env's info dict.
                    final_success[slot] = bool(env.task.success[slot].item())
                    final_goal_status[slot] = self._jsonable_goal_status(infos[slot]["done"]["goal_status"])
                    wall_s[slot] = time.monotonic() - t_batch_start
                    log.info(
                        f"Demo {demo['demo_id']} (slot {slot}) finished at step {t}: "
                        f"fresh_success={final_success[slot]}"
                    )
                    if self.terminal_transition_step(demo) is not None:
                        # This verdict is PROVISIONAL: the goal this demo records is produced by
                        # the transition on this very step, which upstream applies after the info
                        # capture. The terminal pass below re-reads it after applying that
                        # transition; without it the demo is scored a fail it cannot avoid.
                        terminal_slots.append(slot)
                        log.info(
                            f"Demo {demo['demo_id']} (slot {slot}) records a transition on its "
                            f"final step {t}; its verdict is provisional until the terminal pass"
                        )
                # Transitions apply after the step (upstream ordering). One recorded ON the
                # demo's final step is deferred to the terminal pass instead: applying it here
                # would break the deterministic hold's re-injection of the final recorded state
                # row (whose serialized layout predates the added objects), and the verdict it
                # produces cannot be seen without stepping past the recording.
                if t < demo["n_steps"] - 1 and str(t) in demo["transitions"]:
                    cur_transitions = demo["transitions"][str(t)]
                    t_apply = time.monotonic()
                    self._apply_transitions(
                        scene=scenes[slot],
                        cur_transitions=cur_transitions,
                        recorded_scene_file=demo["recorded_scene_file"],
                    )
                    log.info(
                        f"Applied transition for demo {demo['demo_id']} (slot {slot}) at t={t} in "
                        f"{time.monotonic() - t_apply:.3f}s: "
                        f"sys_add={cur_transitions['systems']['add']}, "
                        f"sys_rm={cur_transitions['systems']['remove']}, "
                        f"obj_add={[info['args']['name'] for info in cur_transitions['objects']['add']]}, "
                        f"obj_rm={cur_transitions['objects']['remove']}"
                    )
                    any_transition = True

            if any_transition:
                # One global step to initialize newly added objects (matches upstream). This
                # advances every other scene one extra physics step, which is harmless because
                # their state is re-injected at the top of the next iteration.
                t_settle = time.monotonic()
                og.sim.step()
                log.info(f"Post-transition global settle step at t={t} took {time.monotonic() - t_settle:.3f}s")

            # Periodic re-check of the separation invariant, plus the batch's last step so
            # the tail is covered whatever max_len is modulo the period.
            if (t + 1) % self.separation_check_every == 0 or t == max_len - 1:
                self._check_separation(step=t)

            if step_callback is not None:
                step_callback(t=t, demos=demos, scenes=scenes, terminateds=terminateds, infos=infos)

        # --- Terminal pass: final-step transitions, and the verdicts that need them ---
        terminal_steps = {}
        if terminal_slots:
            for slot, (fresh, success, goal_status, wall) in self._apply_terminal_transitions(
                demos=demos, slots=terminal_slots, t_batch_start=t_batch_start
            ).items():
                demo = demos[slot]
                terminal_steps[slot] = demo["n_steps"] - 1
                final_success[slot] = success
                final_goal_status[slot] = goal_status
                wall_s[slot] = wall
                if fresh and first_success_step[slot] is None:
                    # One past the recording's last index, keeping the campaign's
                    # first_fresh_success == recorded_first_terminated + 1 signature.
                    first_success_step[slot] = demo["n_steps"]

        # --- Separation-check cost, measured rather than asserted ---
        if self._sep_check_count:
            batch_wall = time.monotonic() - t_batch_start
            total_s = self._sep_check_s + self._sep_baseline_s
            log.info(
                f"Separation invariant for batch {batch_index}: baseline (all shared objects x "
                f"{self.num_envs} scenes) {self._sep_baseline_s * 1000.0:.0f} ms + "
                f"{self._sep_check_count} periodic check(s) over {max_len} steps "
                f"{self._sep_check_s:.2f}s ({1000.0 * self._sep_check_s / self._sep_check_count:.1f} ms/check, "
                f"{len(self._sep_sample_names)} sampled objects x {self.num_envs} scenes, every "
                f"{self.separation_check_every} steps) = {total_s:.2f}s total = "
                f"{100.0 * total_s / batch_wall:.4f}% of the batch's {batch_wall:.0f}s wall; smallest gap seen "
                f"{min(self._sep_min_gap_by_slot.values()):.3f} m vs the "
                f"{self._sep_required_gap_m:.3f} m minimum"
            )

        # --- Assemble result rows ---
        for slot, demo in enumerate(demos):
            assert (
                final_success[slot] is not None
            ), f"Demo {demo['demo_id']} (slot {slot}) never reached its final step -- step loop bug"
            # Recording-boundary bucket: replayed like any other demo, then LABELLED
            # instead of being called a failure. Replaying rather than skipping is what
            # keeps the classification honest -- each such demo's partial goal_status is
            # the per-demo evidence for the diagnosis, and a demo that passes anyway
            # proves the rule wrong instead of hiding behind a skip.
            unreplayable_step = self.unreplayable_recording_step(demo)
            status = "pass" if final_success[slot] else "fail"
            if unreplayable_step is not None:
                if final_success[slot]:
                    log.warning(
                        f"Demo {demo['demo_id']} (slot {slot}) matches the unreplayable-recording rule "
                        f"(transition at/past its final step {unreplayable_step}, recorded terminated "
                        f"first True only there) yet REACHED SUCCESS at step "
                        f"{first_success_step[slot]}: the rule mis-classified this demo. Row keeps "
                        f"status=pass and the mismatch is reported for follow-up."
                    )
                else:
                    status = "unreplayable_recording"
                    log.info(
                        f"Demo {demo['demo_id']} (slot {slot}) labelled unreplayable_recording: its only "
                        f"recorded success is produced by the transition on its final step "
                        f"{unreplayable_step}, whose products appear in no recorded state row. Partial "
                        f"goal status kept: {final_goal_status[slot]}"
                    )
            rows_by_input_idx[demo_input_idx[slot]] = {
                "demo_id": demo["demo_id"],
                "task_id": self.task_id,
                "task_name": self.task_name,
                "demo_group": f"demo_{demo['episode_id']}",
                "n_steps": demo["n_steps"],
                "status": status,
                "fresh_success_final": final_success[slot],
                "first_fresh_success_step": first_success_step[slot],
                "recorded_terminated_final": bool(demo["terminated"][demo["n_steps"] - 1]),
                "per_step_terminated_mismatches": mismatches[slot],
                "goal_status_final": final_goal_status[slot],
                "wall_s": round(wall_s[slot], 3),
                "num_envs": self.num_envs,
                "batch": batch_index,
                "slot": slot,
                "terminal_transition_step": terminal_steps.get(slot),
                "unreplayable_recording_step": unreplayable_step,
                "min_scene_gap_m": (
                    round(self._sep_min_gap_by_slot[slot], 4) if slot in self._sep_min_gap_by_slot else None
                ),
                "error": None,
            }

        return [rows_by_input_idx[i] for i in range(n_input)]
