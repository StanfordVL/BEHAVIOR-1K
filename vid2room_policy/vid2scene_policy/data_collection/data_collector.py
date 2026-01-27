import math
import logging
import random

import torch as th

from .config import DataCollectionConfig
from .robot_context import RobotContext, MODE_MANIPULATION
from .navigation import NavigationController
from .arm_control import ArmController
from .gripper_control import GripperController

logger = logging.getLogger(__name__)


class DataCollector:
    """Composes navigation, arm, and gripper controllers for pick-and-place."""

    def __init__(self, env, robot, config: DataCollectionConfig):
        self.ctx = RobotContext(env, robot, config)
        self.nav = NavigationController(self.ctx)
        self.arm = ArmController(self.ctx)
        self.gripper = GripperController(self.ctx)

    def _check_gripper_alignment(self, object_pos: th.Tensor, tolerance: float = 0.1) -> bool:
        eef_pos, _ = self.ctx.robot.get_eef_pose(self.ctx.arm)
        robot_pos, _ = self.ctx.robot.get_position_orientation()

        dx = object_pos[0].item() - eef_pos[0].item()
        dy = object_pos[1].item() - eef_pos[1].item()
        dist_to_obj = math.sqrt(dx*dx + dy*dy)

        robot_yaw = self.ctx.get_robot_yaw()
        arm_direction = robot_yaw + math.pi / 2

        perp_direction = arm_direction + math.pi / 2
        perp_error = dx * math.cos(perp_direction) + dy * math.sin(perp_direction)

        print(f"[Pick] Alignment check: robot=({robot_pos[0].item():.2f}, {robot_pos[1].item():.2f}), "
              f"eef=({eef_pos[0].item():.2f}, {eef_pos[1].item():.2f}), "
              f"obj=({object_pos[0].item():.2f}, {object_pos[1].item():.2f})", flush=True)
        print(f"[Pick] Alignment: robot_yaw={math.degrees(robot_yaw):.0f}°, arm_dir={math.degrees(arm_direction):.0f}°, "
              f"dist={dist_to_obj:.2f}m, perp_err={perp_error:.3f}m (tol={tolerance})", flush=True)

        return abs(perp_error) < tolerance

    def pick_object(self, target_obj, source_support=None) -> tuple[bool, list, list]:
        """Execute full pick sequence."""
        observations = []
        actions = []

        obj_pos, _ = target_obj.get_position_orientation()
        logger.info("Picking object at (%.2f, %.2f, %.2f)",
                    obj_pos[0].item(), obj_pos[1].item(), obj_pos[2].item())

        # Find approach point
        APPROACH_DIST = 0.32
        approach_point, use_eroded = self.nav.find_approach_point(obj_pos[:2], approach_dist=APPROACH_DIST, support=source_support)
        if approach_point is None:
            print("[Pick] FAIL: Could not find approach point", flush=True)
            return False, observations, actions

        # Navigate to approach point
        success, obs, acts = self.nav.navigate_to_target(approach_point, approach_dist=0.20, use_eroded_map=use_eroded)
        observations.extend(obs)
        actions.extend(acts)
        if not success:
            print("[Pick] FAIL: Navigation failed", flush=True)
            return False, observations, actions
        print("[Pick] Navigation OK", flush=True)

        # Orient toward object
        obj_pos, _ = target_obj.get_position_orientation()
        success, obs, acts = self.nav.orient_to_object(obj_pos)
        observations.extend(obs)
        actions.extend(acts)
        print("[Pick] Orient OK", flush=True)

        # Rotate to align arm with object
        success, obs, acts = self.nav.rotate_to_align_wrist_with_object(target_obj)
        observations.extend(obs)
        actions.extend(acts)
        final_yaw = self.ctx.get_robot_yaw()
        print(f"[Pick] Wrist rotation OK, final_yaw={math.degrees(final_yaw):.0f}°, steps={len(acts)}", flush=True)

        # Check alignment
        obj_pos, _ = target_obj.get_position_orientation()
        if not self._check_gripper_alignment(obj_pos, tolerance=0.10):
            print("[Pick] FAIL: Gripper alignment check failed", flush=True)
            return False, observations, actions
        print("[Pick] Alignment OK, extending arm...", flush=True)
        obj_pos, _ = target_obj.get_position_orientation()
        aabb = target_obj.aabb
        bottom_z = aabb[0][2].item()
        bbox_height = aabb[1][2].item() - bottom_z
        grasp_z = bottom_z + bbox_height * 0.30
        print(f"[Pick] Grasp height: bottom_z={bottom_z:.3f}, bbox_h={bbox_height:.3f}, grasp_z={grasp_z:.3f}", flush=True)
        grasp_pos = th.tensor([obj_pos[0].item(), obj_pos[1].item(), grasp_z])
        success, obs, acts = self.arm.align_to_object(grasp_pos, keep_gripper_open=True)
        observations.extend(obs)
        actions.extend(acts)

        # Close gripper
        success, obs, acts = self.gripper.close(max_steps=40)
        observations.extend(obs)
        actions.extend(acts)

        # Brief settle after grasp
        for _ in range(2):
            action = self.ctx.empty_action(mode=MODE_MANIPULATION)
            action[self.ctx.gripper_idx] = -1.0
            obs, _, _, _, _ = self.ctx.env.step(action.numpy())
            observations.append(obs)
            actions.append(action.numpy())

        obj_pos_before_lift, _ = target_obj.get_position_orientation()

        # Initial lift
        obs, acts = self.arm.lift_up(amount=0.15)
        observations.extend(obs)
        actions.extend(acts)

        # Maximize lift for transport
        obs, acts = self.arm.maximize_lift(max_steps=80)
        observations.extend(obs)
        actions.extend(acts)

        # Contract arm for transport
        obs, acts = self.arm.contract_arm()
        observations.extend(obs)
        actions.extend(acts)

        # Verify grasp
        obj_pos_after_lift, _ = target_obj.get_position_orientation()
        height_change = obj_pos_after_lift[2].item() - obj_pos_before_lift[2].item()

        eef_pos, _ = self.ctx.robot.get_eef_pose(self.ctx.arm)
        dist_to_obj = math.sqrt(
            (eef_pos[0].item() - obj_pos_after_lift[0].item())**2 +
            (eef_pos[1].item() - obj_pos_after_lift[1].item())**2 +
            (eef_pos[2].item() - obj_pos_after_lift[2].item())**2
        )

        obj_in_hand = self.ctx.robot._ag_obj_in_hand.get(self.ctx.arm)

        # Check if we grasped the correct target object
        correct_object_grasped = False
        if obj_in_hand is not None:
            if obj_in_hand.name == target_obj.name:
                correct_object_grasped = True
                print(f"[Pick] Correct object grasped: {obj_in_hand.name}", flush=True)
            else:
                print(f"[Pick] FAIL: Wrong object grasped: {obj_in_hand.name} (expected {target_obj.name})", flush=True)
                return False, observations, actions

        grasp_success = correct_object_grasped or (height_change > 0.05) or (dist_to_obj < 0.15)

        if not grasp_success:
            print(f"[Pick] FAIL: Grasp verification failed (height_change={height_change:.2f}, dist={dist_to_obj:.2f})", flush=True)
            return False, observations, actions

        print("[Pick] SUCCESS - object lifted", flush=True)
        return True, observations, actions

    def place_object(self, target_support) -> tuple[bool, list, list]:
        """Execute full place sequence."""
        observations = []
        actions = []

        # Randomize place position within table bounds
        EDGE_MARGIN = 0.15
        aabb = target_support.aabb
        min_x = aabb[0][0].item() + EDGE_MARGIN
        max_x = aabb[1][0].item() - EDGE_MARGIN
        min_y = aabb[0][1].item() + EDGE_MARGIN
        max_y = aabb[1][1].item() - EDGE_MARGIN
        surface_z = aabb[1][2].item()  # Top of bbox = surface height
        place_x = random.uniform(min_x, max_x)
        place_y = random.uniform(min_y, max_y)
        target_pos = th.tensor([place_x, place_y, surface_z])
        logger.info("Placing at (%.2f, %.2f, %.2f) on %s",
                    target_pos[0].item(), target_pos[1].item(), target_pos[2].item(),
                    target_support.name)

        # Find approach point toward place position (prefer long sides of support)
        APPROACH_DIST = 0.55
        approach_point, use_eroded = self.nav.find_approach_point(target_pos[:2], approach_dist=APPROACH_DIST, support=target_support)
        if approach_point is None:
            logger.info("Could not find valid approach point for placing")
            return False, observations, actions

        # Navigate to approach point
        success, obs, acts = self.nav.navigate_to_target(
            approach_point, approach_dist=0.25, keep_gripper_closed=True, use_eroded_map=use_eroded
        )
        observations.extend(obs)
        actions.extend(acts)
        if not success:
            return False, observations, actions

        # Rotate to align arm with target
        success, obs, acts = self.nav.rotate_to_align_wrist_with_position(target_pos, keep_gripper_closed=True)
        observations.extend(obs)
        actions.extend(acts)

        # Lower lift
        place_height = target_pos[2].item() + 0.40
        obs, acts = self.arm.lower_lift(place_height)
        observations.extend(obs)
        actions.extend(acts)

        # Extend arm toward target
        obs, acts = self.arm.extend_arm_for_place(target_pos)
        observations.extend(obs)
        actions.extend(acts)

        # Open gripper to release
        obs, acts = self.gripper.open(num_steps=40)
        observations.extend(obs)
        actions.extend(acts)

        # Let object settle
        for _ in range(10):
            action = self.ctx.empty_action(mode=MODE_MANIPULATION)
            obs, _, _, _, _ = self.ctx.env.step(action.numpy())
            observations.append(obs)
            actions.append(action.numpy())

        logger.info("Place complete")
        return True, observations, actions
