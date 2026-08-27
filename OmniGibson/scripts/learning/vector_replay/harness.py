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
  its row explaining why. A transition on the final action step needs no
  special handling: it is applied in-loop like any other, and on an offset-1
  recording the following iteration injects the trailing state row -- the
  authoritative post-transition state, serialized against exactly the topology
  the transition just created. Only an offset-0 recording (no trailing row)
  whose success appears solely at its last verdict is unscoreable; see
  unreplayable_recording_step.
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
        n_steps (int or None): the demo's RECORDED length -- its action /
            reward / terminated / truncated row count, which the shape contract
            requires to be equal (see validate_demo_shape). NOT the number of
            replay iterations: an offset-1 recording is replayed for one step
            more, to inject its trailing post-final-action state row,
        status (str): the comparison's outcome, a function of BOTH sides --
            "pass" (fresh and recorded both report success), "fail" (no fresh
            success), "fresh_only_success" (fresh reached the goal on a
            recording that never claimed it -- a disagreement, quarantined, and
            never counted as a pass), "unreplayable_recording" (see
            unreplayable_recording_step), or "replay_error" (designed
            malformed-demo row, see module docstring). The reporter may further
            reclassify "fail" as "agrees_not_successful" when the recording
            never claimed success AND the two agreed at every step,
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
        terminal_transition_step (int or None): the demo's final recorded
            action step when a transition is recorded there, else None.
            Descriptive: such a transition needs no special handling, because
            an offset-1 recording's trailing state row is the authoritative
            post-transition state and is injected on the following iteration.
            Kept so the class stays computable in the campaign's output,
        unreplayable_recording_step (int or None): set when the demo matches the
            recording-boundary rule (see unreplayable_recording_step): the
            recording has no trailing post-final-action state row and its only
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

    @staticmethod
    def validate_demo_shape(*, path, episode_id, action, reward, terminated, truncated, state, state_size, transitions):
        """Validates one demo's dataset shapes and returns (n_verdicts, n_state_rows, state_offset).

        Split out of _read_demo so the contract is a pure function of the shapes and can
        be exercised directly: every rejection below is a way a recording can be
        truncated or malformed, and each one used to be absorbed silently.

        Raises:
            ValueError: on any unsupported shape. The caller turns this into a
                replay_error row rather than replaying a recording it cannot trust.
        """
        # ---- Dataset-shape contract ----
        # Collection stores S0 at reset and S(i+1) after action i, so a recording has
        # n_verdicts actions and n_verdicts + 1 state rows. Some recordings carry no
        # trailing row (state == action); those two are the ONLY shapes the 2026 dataset
        # exhibits (censused over the local cache: 54 files at offset 1, 13 at offset 0,
        # nothing else). Anything else is a truncated or malformed file.
        #
        # The previous behavior -- min() over all six lengths, matching upstream's zip --
        # silently absorbed truncation AND, on every offset-1 recording, discarded the
        # trailing state row: the authoritative post-final-action state, the one the
        # recording's own last verdict was computed on. A demo whose success first appears
        # there was then unscoreable for a reason that did not exist.
        lengths = {
            name: int(tensor.shape[0])
            for name, tensor in (
                ("action", action),
                ("reward", reward),
                ("terminated", terminated),
                ("truncated", truncated),
                ("state", state),
                ("state_size", state_size),
            )
        }
        verdict_lengths = {name: lengths[name] for name in ("action", "reward", "terminated", "truncated")}
        state_lengths = {name: lengths[name] for name in ("state", "state_size")}
        if len(set(verdict_lengths.values())) != 1:
            raise ValueError(
                f"Demo group demo_{episode_id} in {path}: the per-action datasets must all have the "
                f"same length, got {verdict_lengths}. This recording is truncated or malformed; "
                f"replaying the shortest would silently drop recorded actions and verdicts."
            )
        if len(set(state_lengths.values())) != 1:
            raise ValueError(
                f"Demo group demo_{episode_id} in {path}: state and state_size must have the same "
                f"length, got {state_lengths}."
            )
        n_verdicts = lengths["action"]
        n_state_rows = lengths["state"]
        if n_verdicts == 0:
            raise ValueError(f"Demo group demo_{episode_id} in {path} has zero recorded steps")
        state_offset = n_state_rows - n_verdicts
        if state_offset not in (0, 1):
            raise ValueError(
                f"Demo group demo_{episode_id} in {path}: {n_state_rows} state rows for "
                f"{n_verdicts} recorded actions (offset {state_offset}). The supported formats are "
                f"offset 1 (a trailing post-final-action state row) and offset 0 (no trailing row); "
                f"any other shape means the file is truncated."
            )
        # Every state row must declare a usable serialized length. A zero, negative,
        # non-integral or over-wide entry would slice the padded row wrongly and
        # deserialize garbage -- silently, since deserialize walks whatever it is handed.
        state_width = int(state.shape[1])
        sizes = state_size.to(th.int64) if state_size.dtype.is_floating_point else state_size
        if state_size.dtype.is_floating_point and not bool(th.equal(state_size, sizes.to(state_size.dtype))):
            raise ValueError(f"Demo group demo_{episode_id} in {path}: state_size holds non-integral values.")
        bad = [(int(i), int(sizes[i])) for i in range(n_state_rows) if not (0 < int(sizes[i]) <= state_width)]
        if bad:
            raise ValueError(
                f"Demo group demo_{episode_id} in {path}: {len(bad)} state_size entries are not in "
                f"(0, {state_width}] (row, value) {bad[:10]}{' ...' if len(bad) > 10 else ''}."
            )
        # A transition is indexed by the action step it follows, so it is reachable only at
        # an index the replay actually steps. An unreachable one changes what the demo can
        # achieve, so it is a malformed recording, not a warning.
        unreachable_transitions = sorted(step for step in (int(k) for k in transitions) if step > n_verdicts - 1)
        if unreachable_transitions:
            raise ValueError(
                f"Demo group demo_{episode_id} in {path} records transition(s) at step(s) "
                f"{unreachable_transitions}, beyond its {n_verdicts} recorded actions; they can "
                f"never be applied, so the demo's verdict cannot be reproduced."
            )
        return n_verdicts, n_state_rows, state_offset

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
        n_verdicts, n_state_rows, state_offset = self.validate_demo_shape(
            path=path,
            episode_id=episode_id,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            state=state,
            state_size=state_size,
            transitions=transitions,
        )

        return {
            "demo_id": demo_id,
            "path": path,
            "episode_id": episode_id,
            # Recorded actions / verdicts. Kept named n_steps because it is the demo's
            # recorded length in every report and row the campaign emits.
            "n_steps": n_verdicts,
            # Replay iterations for this demo: one per state row, so the trailing
            # post-final-action row is injected and evaluated when the recording has one.
            "n_replay_steps": n_state_rows,
            "state_offset": state_offset,
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
        """The demo's final recorded action step if a transition is recorded there, else None.

        Descriptive only -- it drives no special handling. A transition on the final
        action step is applied by the ordinary in-loop path, and on an offset-1
        recording the following iteration then injects the trailing state row, which
        is the authoritative post-transition state. Recorded in the result row so the
        class stays computable in the campaign's output.

        This used to select demos for a separate terminal pass that reconstructed the
        post-transition state from transition metadata. That pass existed because the
        trailing state row was being discarded by the loader; with the row preserved,
        reconstructing anything is both unnecessary and less accurate.

        Args:
            demo (dict): loaded demo from _read_demo.

        Returns:
            int or None: the final recorded action step when it carries a transition.
        """
        final_step = demo["n_steps"] - 1
        return final_step if str(final_step) in demo["transitions"] else None

    @staticmethod
    def unreplayable_recording_step(demo):
        """The final verdict index when this demo's only recorded success cannot be reached.

        The rule is about the RECORDING'S SHAPE, not about transitions:

        (A) ``state_offset == 0`` -- the recording has no trailing post-final-action
            state row, so the state its own last verdict was computed on, S(N), was
            never written; and
        (B) the recording's ``terminated`` first turns True only at that last verdict
            index, so the success exists solely in the state that is missing.

        Both together mean no replay -- ours or upstream's -- can score the demo, at
        any num_envs, in any slot: the only state that satisfies the goal is absent
        from the file.

        Neither half is sufficient. Offset 0 alone is ordinary (most such demos
        succeed well before the end, in rows that are present). A final-only success
        alone is ordinary too when the trailing row exists: offset-1 recordings ARE
        replayed through S(N), which is why this rule says nothing about transitions.
        A final-step transition used to appear here, on the premise that its products
        "can never appear in a recorded state row". That premise was false -- they
        appear in the trailing row, which the loader was discarding -- and it made six
        replayable task-40 demos unscoreable. The mechanism is the missing row, and
        whether the last state change was symbolic or physics-mediated is irrelevant.

        Args:
            demo (dict): loaded demo from _read_demo.

        Returns:
            int or None: the last verdict index when the demo is in the bucket.
        """
        if demo["state_offset"] != 0:
            return None
        last_verdict = demo["n_steps"] - 1
        terminated = demo["terminated"]
        first_true = next((i for i in range(demo["n_steps"]) if bool(terminated[i])), None)
        if first_true is None or first_true < last_verdict:
            return None
        return last_verdict

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
        the gap that tripped, per slot) instead of leaving the shard's output
        silently short. The caller is expected to write them and then fail the
        batch's unit of work; they are never a substitute for a verdict.

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

        Step count: one iteration per STATE ROW, not per recorded action. An
        offset-1 recording (the common shape: N actions, N+1 states) therefore
        gets one iteration more than it has actions, whose job is to inject the
        trailing post-final-action row -- the state the recording's own last
        verdict was computed on -- and read the verdict there. Its action is the
        last recorded one, re-applied. An offset-0 recording has no such row and
        gets exactly N iterations.

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
        # One iteration per STATE ROW, so an offset-1 recording's trailing
        # post-final-action row is injected and evaluated like any other.
        max_len = max(demo["n_replay_steps"] for demo in demos)
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
        first_success_step = [None] * len(demos)
        mismatches = [0] * len(demos)
        final_success = [None] * len(demos)
        final_goal_status = [None] * len(demos)
        wall_s = [None] * len(demos)

        for t in range(max_len):
            if t % 1000 == 0:
                log.info(f"Replaying batch step {t}/{max_len}")

            # State injection: active demos load state[t]; finished demos re-inject their final
            # recorded state row (deterministic hold). Indexed over state rows, so on the
            # last iteration of an offset-1 demo this is the trailing authoritative row --
            # and by then any transition on the final action step has already been applied,
            # so the topology that row was serialized against exists.
            for slot, demo in enumerate(demos):
                idx = min(t, demo["n_replay_steps"] - 1)
                scenes[slot].load_state(demo["state"][idx, : int(demo["state_size"][idx])], serialized=True)

            # Batched actions: recorded action for active demos, last action for finished demos,
            # zeros for inert slots (their robots have control disabled, so this is a no-op).
            # Clamped to the last RECORDED action: an offset-1 demo's extra iteration has a
            # state row but no action of its own, and re-applying the last action is the
            # convention the recording's own final verdict was produced under.
            action_rows = []
            for env_idx in range(self.num_envs):
                if env_idx < len(demos):
                    demo = demos[env_idx]
                    idx = min(t, demo["n_steps"] - 1)
                    action_rows.append(demo["action"][idx].to(th.float32))
                else:
                    action_rows.append(th.zeros(action_dim, dtype=th.float32))
            actions = th.stack(action_rows, dim=0)

            _, _, terminateds, _, infos = env.step(actions)

            any_transition = False
            for slot, demo in enumerate(demos):
                if t >= demo["n_replay_steps"]:
                    continue
                fresh_terminated = bool(terminateds[slot])
                if fresh_terminated and first_success_step[slot] is None:
                    first_success_step[slot] = t
                # Per-step agreement is only defined where the recording has a verdict.
                # An offset-1 demo's extra iteration has a state row but no verdict of its
                # own -- the recording's last verdict was computed ON that row, and it is
                # already accounted for by the first_fresh_success == recorded + 1 convention.
                if t < demo["n_steps"] and fresh_terminated != bool(demo["terminated"][t]):
                    mismatches[slot] += 1
                if t == demo["n_replay_steps"] - 1:
                    # Last state row: capture fresh success and goal status from this step.
                    # BehaviorTask._step_termination adds goal_status to the termination infos,
                    # which BaseTask.step nests under the "done" key of each env's info dict.
                    final_success[slot] = bool(env.task.success[slot].item())
                    final_goal_status[slot] = self._jsonable_goal_status(infos[slot]["done"]["goal_status"])
                    wall_s[slot] = time.monotonic() - t_batch_start
                    log.info(
                        f"Demo {demo['demo_id']} (slot {slot}) finished at step {t} "
                        f"(state row {t} of {demo['n_replay_steps']}, offset "
                        f"{demo['state_offset']}): fresh_success={final_success[slot]}"
                    )
                # Transitions apply after the step, indexed by the action step they follow
                # (upstream ordering). One on the final action step is applied here like any
                # other: on an offset-1 recording the next iteration then injects the trailing
                # state row, which is the authoritative post-transition state and needs this
                # topology to exist before it can be deserialized.
                if t < demo["n_steps"] and str(t) in demo["transitions"]:
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
            recorded_final = bool(demo["terminated"][demo["n_steps"] - 1])
            # A verdict is a COMPARISON, so the status is a function of both sides.
            # "pass" means the two agree that the demo succeeded -- nothing else. Deriving
            # it from the fresh side alone let a fresh-only success (the recording never
            # claimed success; this replay says it did) be reported as a pass, which is the
            # exact shape a multi-scene contamination bug produces: a goal predicate
            # spuriously true at the final step. That is the campaign's most interesting
            # possible finding, so it gets its own status, is quarantined, and never enters
            # the pass total.
            if not final_success[slot]:
                # No fresh success. Whether this is a real shortfall or agreement with a
                # recording that never claimed success is decided by the reporter, which
                # also requires zero per-step mismatches before calling it agreement.
                status = "fail"
            elif recorded_final:
                status = "pass"
            else:
                status = "fresh_only_success"
            if status == "fresh_only_success":
                log.warning(
                    f"Demo {demo['demo_id']} (slot {slot}) reached success at step "
                    f"{first_success_step[slot]} but its recording never reports success "
                    f"(recorded_terminated_final=False). This is a DISAGREEMENT, not a pass: it is "
                    f"what a spurious goal predicate looks like. Quarantined for triage. "
                    f"Goal status: {final_goal_status[slot]}"
                )
            if unreplayable_step is not None:
                if final_success[slot]:
                    log.warning(
                        f"Demo {demo['demo_id']} (slot {slot}) matches the unreplayable-recording rule "
                        f"(no trailing state row, recorded success only at verdict "
                        f"{unreplayable_step}) yet REACHED SUCCESS at step "
                        f"{first_success_step[slot]}: the rule mis-classified this demo. Row keeps "
                        f"status={status} and the mismatch is reported for follow-up."
                    )
                else:
                    status = "unreplayable_recording"
                    log.info(
                        f"Demo {demo['demo_id']} (slot {slot}) labelled unreplayable_recording: the "
                        f"recording has no trailing post-final-action state row (state_offset=0) and "
                        f"its success appears only at verdict {unreplayable_step}, so the only state "
                        f"satisfying the goal was never written. Partial goal status kept: "
                        f"{final_goal_status[slot]}"
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
                "terminal_transition_step": self.terminal_transition_step(demo),
                "unreplayable_recording_step": unreplayable_step,
                "min_scene_gap_m": (
                    round(self._sep_min_gap_by_slot[slot], 4) if slot in self._sep_min_gap_by_slot else None
                ),
                "error": None,
            }

        return [rows_by_input_idx[i] for i in range(n_input)]
