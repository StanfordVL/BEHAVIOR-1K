"""L0 — N-robot OmniGibson environment construction and startup ritual.

Everything here is configuration only; no OmniGibson source is modified.

Three things are easy to get wrong and all of them fail *silently*:

1. ``scene.include_robots`` must be **false**. ``Environment._load_robots``
   is guarded by ``if len(self.scene.robots) == 0``, so if the scene USD
   already carries robots your entire ``robots:`` list is ignored.
2. Every robot needs an explicit ``name``. The name becomes the key of the
   action dict and the observation dict; without one you get
   ``robot_<6 random letters>``.
3. Use ``model: r1``, not the deprecated ``type: R1`` (which the shipped
   configs still use and which only survives via a lowercasing warning).
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch as th
import yaml

import omnigibson as og
from omnigibson.macros import macros

__all__ = [
    "tune_primitive_macros",
    "build_multi_robot_config",
    "prepare_robots",
    "assert_multi_robot_sanity",
]


def tune_primitive_macros(
    max_steps_for_joint_motion: int = 30,
    joint_pos_diff_threshold: float = 0.02,
    max_steps_for_settling: int = 200,
    collision_activation_distance: float = 0.03,
    base_pose_sampling_upper_bound: Optional[float] = None,
    holonomic_base_prismatic_joint_limit: Optional[float] = None,
) -> None:
    """Loosen the primitive tolerances that concurrency makes untenable.

    Must be called **before** any controller is constructed: ``MacroDict``
    locks a macro once it has been *read*, and a later write raises unless
    wrapped in ``macros.unlocked()`` (which we do here anyway).

    Defaults rationale:
        - ``MAX_STEPS_FOR_JOINT_MOTION`` (stock 10): the per-waypoint tracking
          budget in ``_execute_motion_plan``. With teammates perturbing the
          physics, 10 ticks routinely misses the waypoint and raises a
          spurious ``EXECUTION_ERROR``.
        - ``JOINT_POS_DIFF_THRESHOLD`` (stock 0.005): the "arrived" tolerance
          for the same check.
        - ``MAX_STEPS_FOR_SETTLING`` (stock 500): every ``_move_hand`` begins
          with a ``_settle_robot``, so this is paid many times per primitive.
        - ``DEFAULT_COLLISION_ACTIVATION_DISTANCE`` (stock 0.02): inflates the
          robot spheres against the world; extra margin for teammates.
        - ``HOLONOMIC_BASE_PRISMATIC_JOINT_LIMIT`` (stock 5.0 m): the
          planner's virtual base joint limit. Cross-room transport silently
          fails to plan beyond it. Pass a larger value for big scenes.
    """
    from omnigibson.action_primitives import curobo as curobo_module
    from omnigibson.action_primitives import starter_semantic_action_primitives as ssap

    with macros.unlocked():
        m = ssap.m
        m.MAX_STEPS_FOR_JOINT_MOTION = max_steps_for_joint_motion
        m.JOINT_POS_DIFF_THRESHOLD = joint_pos_diff_threshold
        m.MAX_STEPS_FOR_SETTLING = max_steps_for_settling
        m.DEFAULT_COLLISION_ACTIVATION_DISTANCE = collision_activation_distance
        if base_pose_sampling_upper_bound is not None:
            m.BASE_POSE_SAMPLING_UPPER_BOUND = base_pose_sampling_upper_bound
        if holonomic_base_prismatic_joint_limit is not None:
            curobo_module.m.HOLONOMIC_BASE_PRISMATIC_JOINT_LIMIT = holonomic_base_prismatic_joint_limit


def build_multi_robot_config(
    robot_poses: Sequence[Tuple[Sequence[float], Sequence[float]]],
    robot_model: str = "R1",
    scene_model: str = "Rs_int",
    load_object_categories: Optional[Sequence[str]] = ("floors", "walls", "coffee_table"),
    objects: Optional[Sequence[Dict[str, Any]]] = None,
    agent_names: Optional[Sequence[str]] = None,
    symbolic_only: bool = True,
    task: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build an ``og.Environment`` config with one robot per entry of ``robot_poses``.

    Starts from the shipped ``<robot>_primitives.yaml`` because
    ``StarterSemanticActionPrimitives`` depends on that exact controller
    setup: a ``HolonomicBaseJointController`` for the base and
    ``JointController`` for trunk/arms/grippers, all with
    ``motor_type: position``, ``command_input_limits: null``,
    ``use_delta_commands: false``, ``use_impedances: false``. That is what
    makes ``robot.q_to_action(q)`` a pass-through of absolute joint targets,
    which the whole motion-execution loop assumes.

    Args:
        robot_poses: ``[(position_xyz, orientation_xyzw), ...]``, one per robot.
        robot_model: "R1" recommended. **Avoid R1Pro on RTX-50 class GPUs**:
            cuRobo drops the DEFAULT embodiment at cuda capability (12,0)
            while ``update_obstacles`` / ``check_collisions`` index
            ``self.mg[DEFAULT]`` unconditionally, giving a ``KeyError``.
        scene_model: must be an ``InteractiveTraversableScene`` model --
            ``_sample_pose_near_object`` needs ``scene._seg_map``.
        load_object_categories: keep the collision world small so cuRobo
            planning stays responsive. ``None`` loads the whole house.
        objects: extra ``DatasetObject`` config dicts.
        agent_names: defaults to ``agent_0 ... agent_{n-1}``, matching COOP2's
            naming so ``SymbolicEnvWrapper``'s name_map is the identity.
        symbolic_only: set ``obs_modalities: []`` so no camera is created and
            each robot drops out of the observation space entirely
            (``Environment`` gates robot obs on ``maxdim(...) > 0``).
        task: defaults to ``DummyTask``.
    """
    config_path = os.path.join(og.example_config_path, f"{robot_model.lower()}_primitives.yaml")
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    template = copy.deepcopy(config["robots"][0])
    # "type" is deprecated in favour of "model"; drop it so we don't rely on
    # the lowercasing fallback.
    template.pop("type", None)
    template["model"] = robot_model.lower()
    if symbolic_only:
        template["obs_modalities"] = []

    names = list(agent_names) if agent_names is not None else [f"agent_{i}" for i in range(len(robot_poses))]
    if len(names) != len(robot_poses):
        raise ValueError(f"Got {len(names)} agent names for {len(robot_poses)} robot poses.")

    robots: List[Dict[str, Any]] = []
    for name, (position, orientation) in zip(names, robot_poses):
        robot_config = copy.deepcopy(template)
        robot_config["name"] = name
        robot_config["position"] = list(position)
        robot_config["orientation"] = list(orientation)
        robots.append(robot_config)
    config["robots"] = robots

    config["scene"].update(
        {
            "scene_model": scene_model,
            "load_object_categories": list(load_object_categories) if load_object_categories else None,
            "not_load_object_categories": None,
            # Non-negotiable: otherwise _load_robots skips our list entirely.
            "include_robots": False,
        }
    )
    config["env"]["external_sensors"] = None
    config["objects"] = [copy.deepcopy(obj) for obj in (objects or [])]
    config["task"] = dict(task) if task else {"type": "DummyTask"}
    return config


def prepare_robots(env, settle_steps: int = 5, object_settle_steps: int = 30) -> None:
    """Put every robot into the state cuRobo's retract config assumes.

    Skipping this causes immediate planning failures: the locked-joint /
    retract configuration built in ``CuRoboMotionGenerator.__init__``
    assumes fully open grippers. Mirrors
    ``examples/wip/rs_int_primitives_example.py`` but loops over all robots.
    """
    for robot in env.robots:
        for gripper_control_idx in robot.gripper_control_idx.values():
            robot.set_joint_positions(
                th.ones_like(gripper_control_idx), indices=gripper_control_idx, normalized=True
            )
        robot.keep_still()

    for _ in range(settle_steps):
        og.sim.step()

    env.scene.update_initial_file()
    # NOTE: reset() unconditionally runs 1 sim step + 3 renders, but only when
    # get_obs=True. With a purely symbolic observation you can pass
    # get_obs=False to skip them -- worth it for long training runs, kept
    # simple (and viewer-friendly) here.
    env.reset()

    for _ in range(object_settle_steps):
        og.sim.step()


def assert_multi_robot_sanity(env, expected_robots: int) -> None:
    """Fail loudly on the silent multi-robot misconfigurations.

    In particular, verifies that the robots really are in
    ``scene.objects`` -- that registry is what
    ``CuRoboMotionGenerator.update_obstacles`` walks (skipping only
    ``self.robot``), so it decides whether teammates exist as obstacles for
    each other's planner at all.
    """
    robots = env.robots
    assert len(robots) == expected_robots, (
        f"Expected {expected_robots} robots but the scene has {len(robots)}: {[r.name for r in robots]}. "
        "The usual cause is scene.include_robots=true, which makes _load_robots ignore the robots config."
    )

    names = [r.name for r in robots]
    assert len(set(names)) == len(names), f"Duplicate robot names {names}; names are the action/obs dict keys."

    scene_objects = set(env.scene.objects)
    for robot in robots:
        assert robot in scene_objects, (
            f"{robot.name} is not in scene.objects, so it will be INVISIBLE to every teammate's cuRobo "
            "collision world (update_obstacles iterates robot.scene.objects). Teammate meshes would have to "
            "be injected manually."
        )

    action_space = env.action_space
    for name in names:
        assert name in action_space.spaces, (
            f"{name} missing from the action space {sorted(action_space.spaces)}. "
            "env.step requires a key for EVERY robot on every tick."
        )

    seg_map = getattr(env.scene, "_seg_map", None)
    assert seg_map is not None, (
        "scene._seg_map is None. StarterSemanticActionPrimitives._sample_pose_near_object filters candidate "
        "base poses by room, so an InteractiveTraversableScene model (e.g. Rs_int) is required."
    )

    print(f"[sanity] {len(robots)} robots: {names}")
    print("[sanity] all robots present in scene.objects (visible to each other's cuRobo collision world)")
    print(f"[sanity] action space keys: {sorted(action_space.spaces)}")
    for robot in robots:
        position, _ = robot.get_position_orientation()
        room = seg_map.get_room_instance_by_point(position[:2])
        print(f"[sanity]   {robot.name}: action_dim={robot.action_dim} pos={[round(v, 3) for v in position.tolist()]} room={room}")
