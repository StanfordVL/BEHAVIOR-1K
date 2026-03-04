import logging
import math
import random
import traceback
from pathlib import Path
from typing import Callable

import numpy as np
import torch as th
import yaml

import omnigibson as og
import omnigibson.lazy as lazy
from .config import DataCollectionConfig, get_object_filters
from .data_collector import DataCollector
from .omnigibson_lerobot_wrapper import OmniGibsonLeRobotWrapper, OmniGibsonLeRobotConfig
from .scene_management import spawn_and_place_object, safe_remove_object
from .scene_sampling import (
    find_graspable_objects_on_support,
    get_support_position_2d,
    get_valid_supports,
    map_supports_to_rooms,
    sample_robot_start_near_support,
    select_best_graspable_object,
)

logger = logging.getLogger(__name__)

DEBUG_EPISODE = False
SCENE_UNSUITABLE_SENTINEL = "__SCENE_UNSUITABLE__"
RETRY_ATTEMPT_SENTINEL = "__RETRY_ATTEMPT__"


def _build_pick_instruction(episode_metadata: dict) -> str:
    target_category = episode_metadata.get("target_object_category")
    target_name = episode_metadata.get("target_object_name", "object")
    support_name = episode_metadata.get("support_object_name", "support")
    target_label = target_category if target_category else target_name
    target_label = str(target_label).replace("_", " ")
    return f"Pick up {target_label} on the {support_name}."


def _save_trav_map_raw(scene, room_supports: dict):
    from .debug_travmap import save_trav_map_raw

    save_trav_map_raw(scene, room_supports)


def _save_trav_map_visualization(scene, robot, room_supports: dict, source_support, target_support):
    from .debug_travmap import save_trav_map_visualization

    save_trav_map_visualization(scene, robot, room_supports, source_support, target_support)


def _step_sim(num_steps: int) -> None:
    for _ in range(num_steps):
        og.sim.step()




def collect_episode(
    env, scene, robot, collector: DataCollector,
    is_graspable_fn: Callable[[str], bool],
    is_support_fn: Callable[[str], bool],
    wrapper: OmniGibsonLeRobotWrapper = None,
    failed_objects: dict[str, int] = None,
    max_object_failures: int = 3,
    dataset_name: str = "behavior-1k-assets",
    requested_scene_name: str | None = None,
    cached_pairs: list = None,
    max_support_height: float = 1.0,
    max_arm_reach_m: float = 0.9,
    support_search_radius_m: float = 2.5,
    support_erosion_extra_margin_m: float = 0.15,
    ignored_nav_obstacle_categories: tuple[str, ...] = (),
    remove_other_movable_objects: bool = False,
) -> tuple[bool, list, list, dict, str | None, list]:
    """Returns (success, observations, actions, failed_obj_name, cached_pairs)"""
    if failed_objects is None:
        failed_objects = {}

    robot.set_position_orientation(position=[0, 0, -10], orientation=[0, 0, 0, 1])
    robot.keep_still()
    _step_sim(5)

    floors = list(scene.object_registry("category", "floors"))
    if floors:
        floor_top_z = max([obj.aabb[1][2].item() for obj in floors])
        floor_z = floor_top_z  # Robot at floor surface level
        print(f"[Episode] Floor top Z: {floor_top_z:.3f}, robot Z: {floor_z:.3f}", flush=True)
    else:
        floor_z = 0.0
        print(f"[Episode] No floors found, using floor_z={floor_z:.3f}", flush=True)

    # Each candidate is (room_id, support, robot_start).
    room_supports = None
    if cached_pairs is not None:
        support_candidates = cached_pairs
    else:
        valid_supports = get_valid_supports(scene, is_support_fn, max_support_height)
        if len(valid_supports) < 1:
            print(f"[Episode] Not enough valid supports: {len(valid_supports)}", flush=True)
            return False, [], [], {}, None, None

        room_supports = map_supports_to_rooms(scene, valid_supports)

        num_rooms = len(room_supports)
        print(f"[Episode] {num_rooms} rooms, supports by room: {[(r, len(s)) for r, s in room_supports.items()]}", flush=True)

        if num_rooms == 0:
            print(f"[Episode] No rooms found with supports - scene unsuitable", flush=True)
            return False, [], [], {}, SCENE_UNSUITABLE_SENTINEL, None

        support_candidates = []
        for room_ins_id, supports in room_supports.items():
            for support in supports:
                support_pos_2d = get_support_position_2d(support)
                robot_start = sample_robot_start_near_support(
                    scene,
                    robot,
                    support,
                    floor_z,
                    max_arm_reach_m,
                    search_radius_m=support_search_radius_m,
                    erosion_extra_margin_m=support_erosion_extra_margin_m,
                    ignored_obstacle_categories=ignored_nav_obstacle_categories,
                )
                if robot_start is None:
                    print(f"[Episode] Support {support.name}: no sampled start within arm reach", flush=True)
                    continue
                support_candidates.append((room_ins_id, support, robot_start))
                sx, sy, _, _ = robot_start
                dist = math.sqrt(
                    (float(support_pos_2d[0].item()) - sx) ** 2 + (float(support_pos_2d[1].item()) - sy) ** 2
                )
                print(f"[Episode] Support {support.name}: candidate accepted at distance {dist:.2f}m", flush=True)

        if not support_candidates:
            print(f"[Episode] No valid support candidates this attempt", flush=True)
            if DEBUG_EPISODE:
                room_supports_viz = {f"room_{k}": v for k, v in room_supports.items()}
                _save_trav_map_raw(scene, room_supports_viz)
            return False, [], [], {}, RETRY_ATTEMPT_SENTINEL, None

    selected_candidate = random.choice(support_candidates)
    chosen_room_id, source_support, robot_start = selected_candidate

    print(f"[Episode] Selected support for pick-only task:", flush=True)
    print(f"[Episode]   Room: {chosen_room_id}", flush=True)
    print(f"[Episode]   Support: {source_support.name}", flush=True)
    print(f"[Episode]   Robot/base start is within arm reach of support", flush=True)

    if DEBUG_EPISODE and cached_pairs is None and room_supports is not None:
        room_supports_viz = {f"room_{k}": v for k, v in room_supports.items()}
        _save_trav_map_visualization(scene, robot, room_supports_viz, source_support, source_support)

    start_x, start_y, start_z, facing_yaw = robot_start
    print(f"[Episode] Robot placed near support: ({start_x:.2f}, {start_y:.2f}, z={start_z:.2f})", flush=True)

    start_quat = [0, 0, math.sin(facing_yaw / 2), math.cos(facing_yaw / 2)]
    robot.set_position_orientation(position=[start_x, start_y, start_z], orientation=start_quat)
    robot.keep_still()

    _step_sim(10)

    spawned_obj = False

    existing_objects = find_graspable_objects_on_support(scene, source_support, is_graspable_fn)

    valid_objects = [
        obj for obj in existing_objects
        if failed_objects.get(obj.name, 0) < max_object_failures
    ]

    skipped = len(existing_objects) - len(valid_objects)
    if skipped > 0:
        print(f"[Episode] Skipped {skipped} failed objects", flush=True)

    if valid_objects:
        target_obj = select_best_graspable_object(valid_objects, scene, robot)
        print(f"[Episode] Using {target_obj.name} on {source_support.name}", flush=True)
    else:
        robot_pos, _ = robot.get_position_orientation()
        target_obj = None
        for _ in range(20):
            spawned = spawn_and_place_object(scene, source_support, robot_pos=robot_pos)
            if spawned is not None:
                target_obj = spawned
                break

        if target_obj is None:
            print(f"[Episode] Failed to spawn object on {source_support.name}", flush=True)
            return False, [], [], {}, None, support_candidates
        spawned_obj = True
        print(f"[Episode] Spawned {target_obj.name} on {source_support.name}", flush=True)

    target_obj_initial_position, target_obj_initial_orientation = target_obj.get_position_orientation()

    if remove_other_movable_objects:
        for j, obj in enumerate(scene.objects):
            if not obj.fixed_base and obj not in (target_obj, source_support, robot):
                obj.set_position_orientation(position=th.as_tensor([100 + j, 0, 10.]))

    if wrapper is not None:
        wrapper.set_target_objects(target_obj, source_support)

    all_observations = []
    all_actions = []

    try:
        success, obs, acts = collector.pick_object(target_obj)
    except Exception as exc:
        print(f"[Episode] Exception during pick_object: {exc}", flush=True)
        traceback.print_exc()
        if spawned_obj:
            safe_remove_object(scene, target_obj, robot)
        return False, all_observations, all_actions, {}, RETRY_ATTEMPT_SENTINEL, support_candidates
    print(f"[Episode] Pick: success={success}, steps={len(acts)}", flush=True)
    all_observations.extend(obs)
    all_actions.extend(acts)

    if not success:
        failed_obj_name = target_obj.name
        if spawned_obj:
            safe_remove_object(scene, target_obj, robot)
        return False, all_observations, all_actions, {}, failed_obj_name, support_candidates

    print(f"[Episode] Complete pick-only episode: {len(all_actions)} steps, success={success}", flush=True)
    target_obj_category = getattr(target_obj, "category", None)
    target_obj_model = getattr(target_obj, "model", None)
    target_obj_dataset_name = getattr(target_obj, "dataset_name", None)
    support_object_position, support_object_orientation = source_support.get_position_orientation()
    metadata = {
        "dataset_name": dataset_name,
        "scene_name": requested_scene_name or scene.scene_model,
        "room_instance_id": chosen_room_id,
        "support_object_name": source_support.name,
        "support_object_position": support_object_position.detach().cpu().numpy().tolist(),
        "support_object_orientation": support_object_orientation.detach().cpu().numpy().tolist(),
        "robot_start_x_y_z_theta": list(robot_start),
        "spawned_target_object": spawned_obj,
        "target_object_name": target_obj.name,
        "target_object_category": target_obj_category,
        "target_object_model": target_obj_model,
        "target_object_dataset_name": target_obj_dataset_name,
        "target_object_setup_position": target_obj_initial_position.detach().cpu().numpy().tolist(),
        "target_object_setup_orientation": target_obj_initial_orientation.detach().cpu().numpy().tolist(),
    }
    return success, all_observations, all_actions, metadata, None if success else target_obj.name, support_candidates


class DataCollectionRunner:
    """Run multi-episode data collection."""

    MAX_CONSECUTIVE_FAILURES = 20
    MAX_OBJECT_FAILURES = 1
    MAX_POSITION_FAILURES = 3

    def __init__(self, config: DataCollectionConfig):
        self.config = config

        self.is_support_fn, self.is_graspable_fn = get_object_filters(config)
        logger.info("Using object filter method: %s", config.object_filter_method)

        self.env = None
        self.scene = None
        self.robot = None
        self.wrapper = None
        self.collector = None

        self.max_support_height = 1.0
        self.max_arm_reach_m = self.config.max_arm_reach_m
        self.support_search_radius_m = self.config.support_search_radius_m
        self.support_erosion_extra_margin_m = self.config.support_erosion_extra_margin_m

        self.successful_episodes = 0
        self.attempt = 0
        self.consecutive_failures = 0
        self.failed_objects: dict[str, int] = {}
        self.cached_pairs = None
        self.position_failures = 0

    def _build_og_config(self) -> dict:
        og_config_path = Path(og.example_config_path) / "fetch_vid2scene.yaml"
        with open(og_config_path, "r", encoding="utf-8") as f:
            og_config = yaml.safe_load(f)
        og_config["scene"]["scene_model"] = self.config.scene_model
        og_config["scene"]["dataset_name"] = self.config.dataset_name
        og_config["scene"]["not_load_object_categories"] = ["ceilings"]


        og_config["scene"]["load_room_types"] = None
        og_config["scene"]["load_room_instances"] = None

        if self.config.dataset_name != "behavior-1k-assets":
            og_config["scene"]["scene_instance"] = f"{self.config.scene_model}_best"

        if self.config.dataset_name == "spoc":
            og_config["scene"]["use_floor_plane"] = True
            og_config["scene"]["floor_plane_visible"] = True

        if "robots" in og_config and og_config["robots"]:
            og_config["robots"][0]["grasping_mode"] = "sticky"

        # Disable command scaling for consistent action dimensions.
        controller_cfg = og_config.get("robots", [{}])[0].get("controller_config", {})
        for key in (
            "base",
            "trunk",
            "arm_left",
            "arm_right",
            "gripper_left",
            "gripper_right",
            "arm_0",
            "gripper_0",
            "camera",
        ):
            if key in controller_cfg and isinstance(controller_cfg[key], dict):
                controller_cfg[key]["command_input_limits"] = None
                controller_cfg[key]["command_output_limits"] = None

        return og_config

    def _create_environment(self):
        og_config = self._build_og_config()
        self.env = og.Environment(configs=og_config)
        self.scene = self.env.scene
        self.robot = self.env.robots[0]
        logger.info("Environment created: scene=%s", self.config.scene_model)

        self.scene._seg_map.load_map()
        logger.info(
            "Segmentation map loaded: %d rooms",
            len(self.scene._seg_map.room_ins_name_to_ins_id),
        )

    def _configure_friction_and_mass(self):
        for obj in self.scene.objects:
            if getattr(obj, "category", "") == "floors":
                for link in obj.links.values():
                    for mesh in link.collision_meshes.values():
                        mat_name = f"{obj.name}_floor_physics_mat"
                        physics_mat = (
                            lazy.isaacsim.core.api.materials.physics_material.PhysicsMaterial(
                                prim_path=f"{obj.prim_path}/Looks/{mat_name}",
                                name=mat_name,
                                static_friction=1.0,
                                dynamic_friction=1.0,
                                restitution=0.0,
                            )
                        )
                        mesh.apply_physics_material(physics_mat)
        logger.info("Floor friction applied")

        with og.sim.stopped():
            original_mass = self.robot.base_footprint_link.mass
            self.robot.base_footprint_link.mass = original_mass * 2.0
            logger.info(
                "Robot base mass: %.1f -> %.1f kg",
                original_mass,
                self.robot.base_footprint_link.mass,
            )

    def _setup_wrapper_and_collector(self):
        wrapper_config = OmniGibsonLeRobotConfig(
            repo_id=self.config.repo_id,
            root=self.config.output_dir,
            task_description=f"pick_objects_in_{self.config.scene_model}",
            fps=self.config.fps,
            num_episodes=self.config.num_episodes,
            max_steps=self.config.max_steps_per_episode,
            use_videos=True,
            include_depth=True,
            include_segmentation=True,
        )
        self.wrapper = OmniGibsonLeRobotWrapper(self.env, wrapper_config)

        for _ in range(30):
            og.sim.step()

        self.collector = DataCollector(self.env, self.robot, self.config)

        for _ in range(10):
            og.sim.step()

        self.scene.update_initial_file()
        self.scene.reset()

        for _ in range(30):
            og.sim.step()

        self.robot.reset()
        for _ in range(10):
            og.sim.step()

        self.max_support_height = 1.0
        logger.info("max support height: %.3f", self.max_support_height)
        logger.info("max arm reach: %.3f m", self.max_arm_reach_m)
        logger.info("support search radius: %.3f m", self.support_search_radius_m)
        logger.info("support erosion extra margin: %.3f m", self.support_erosion_extra_margin_m)

        self.wrapper.reset_env()
        self.wrapper.start_recording()

    def _maybe_adjust_spoc_floor(self):
        if self.config.dataset_name != "spoc":
            return
        floors = list(self.env.scene.object_registry("category", "floors"))
        for floor in floors:
            top_surface = floor.aabb[1][2].item()
            logger.info("SPOC: Floor %s top surface at Z=%.3f", floor.name, top_surface)
            if np.isclose(top_surface, 0, atol=0.1):
                continue
            floor_pos, floor_ori = floor.get_position_orientation()
            offset = -top_surface
            floor_pos[2] += offset
            logger.info(
                "SPOC: Moving scene down by %.3f to align floor top at Z=0",
                -offset,
            )
            floor.set_position_orientation(floor_pos, floor_ori)

    def _run_single_attempt(self):
        self.attempt += 1
        print(
            f"[Episode] Attempt {self.attempt}, success: {self.successful_episodes}/{self.config.num_episodes}",
            flush=True,
        )

        if self.consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
            print(f"[Episode] {self.MAX_CONSECUTIVE_FAILURES} consecutive failures", flush=True)
            return False

        self._maybe_adjust_spoc_floor()

        try:
            success, observations, actions, episode_metadata, failed_obj_name, self.cached_pairs = collect_episode(
                self.env,
                self.scene,
                self.robot,
                self.collector,
                self.is_graspable_fn,
                self.is_support_fn,
                wrapper=self.wrapper,
                failed_objects=self.failed_objects,
                max_object_failures=self.MAX_OBJECT_FAILURES,
                dataset_name=self.config.dataset_name,
                requested_scene_name=self.config.scene_model,
                cached_pairs=self.cached_pairs,
                max_support_height=self.max_support_height,
                max_arm_reach_m=self.max_arm_reach_m,
                support_search_radius_m=self.support_search_radius_m,
                support_erosion_extra_margin_m=self.support_erosion_extra_margin_m,
                ignored_nav_obstacle_categories=self.config.ignored_nav_obstacle_categories,
                remove_other_movable_objects=self.config.remove_other_movable_objects,
            )
        except Exception as exc:
            print(f"[Episode] Exception in collect_episode: {exc}", flush=True)
            traceback.print_exc()
            self.cached_pairs = None
            self.consecutive_failures += 1
            return True

        if not success and failed_obj_name:
            if failed_obj_name == SCENE_UNSUITABLE_SENTINEL:
                self.cached_pairs = None
                self.failed_objects.clear()
                self.position_failures = 0
                self.consecutive_failures += 1
                print("[Episode] Scene unsuitable this attempt; retrying", flush=True)
                success = False
            elif failed_obj_name == RETRY_ATTEMPT_SENTINEL:
                self.cached_pairs = None
                self.position_failures = 0
                self.consecutive_failures += 1
                print("[Episode] Retrying with new support sampling", flush=True)
                success = False
            else:
                self.failed_objects[failed_obj_name] = self.failed_objects.get(failed_obj_name, 0) + 1
                self.position_failures += 1
                print(
                    f"[Episode] {failed_obj_name} failed {self.failed_objects[failed_obj_name]}/"
                    f"{self.MAX_OBJECT_FAILURES}, position failures: "
                    f"{self.position_failures}/{self.MAX_POSITION_FAILURES}",
                    flush=True,
                )

                if self.position_failures >= self.MAX_POSITION_FAILURES:
                    print(
                        f"[Episode] {self.MAX_POSITION_FAILURES} failures at current position, trying new robot position",
                        flush=True,
                    )
                    self.failed_objects.clear()
                    self.position_failures = 0

        if success:
            try:
                n_obs = len(observations)
                n_actions = len(actions)
                if n_obs == 0 or n_actions == 0:
                    print(
                        f"[Episode] WARN: success=True but empty trajectory (obs={n_obs}, actions={n_actions})",
                        flush=True,
                    )
                    self.consecutive_failures += 1
                else:
                    task_instruction = _build_pick_instruction(episode_metadata)
                    episode_metadata["task_instruction"] = task_instruction
                    self.wrapper.set_task_description(task_instruction)
                    n_frames = min(n_obs, n_actions)
                    for i in range(n_frames):
                        obs = observations[i]
                        action = actions[i]
                        lerobot_obs = self.wrapper._convert_observation(obs)
                        is_last = i == n_frames - 1
                        self.wrapper.record_frame(
                            lerobot_obs,
                            np.array(action, dtype=np.float32),
                            1.0 if is_last else 0.0,
                            is_last,
                        )
                    self.wrapper.save_episode(episode_metadata=episode_metadata)
                    self.successful_episodes += 1
                    self.consecutive_failures = 0
                    self.position_failures = 0
                    print(
                        f"[Episode] Saved episode {self.successful_episodes} with {n_frames} frames "
                        f"(obs={n_obs}, actions={n_actions})",
                        flush=True,
                    )
            except Exception as exc:
                print(f"[Episode] Exception while saving successful episode: {exc}", flush=True)
                traceback.print_exc()
                self.consecutive_failures += 1
        else:
            self.consecutive_failures += 1

        for arm in self.robot.arm_names:
            if self.robot._ag_obj_in_hand.get(arm) is not None:
                try:
                    self.robot.release_grasp_immediately(arm=arm)
                except Exception:
                    logger.debug("Failed to release grasp for arm %s during reset", arm, exc_info=True)

        self.robot.reset()
        self.scene.reset()

        _step_sim(50)

        return True

    def run(self):
        self._create_environment()
        self._configure_friction_and_mass()
        self._setup_wrapper_and_collector()

        try:
            while self.successful_episodes < self.config.num_episodes:
                if not self._run_single_attempt():
                    break
        finally:
            if self.wrapper is not None:
                self.wrapper.stop_recording()
            print(
                f"[Episode] Done: {self.successful_episodes}/{self.config.num_episodes} "
                f"episodes in {self.attempt} attempts",
                flush=True,
            )
            og.shutdown()


def run_data_collection(config: DataCollectionConfig):
    runner = DataCollectionRunner(config)
    runner.run()
