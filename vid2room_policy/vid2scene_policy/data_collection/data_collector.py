import logging
from typing import Any

import torch as th

from .config import DataCollectionConfig
from .robot_context import RobotContext
from .arm_control import ArmController

logger = logging.getLogger(__name__)


class DataCollector:
    """Composes arm control for pick-only data collection."""

    def __init__(self, env, robot, config: DataCollectionConfig):
        self.ctx = RobotContext(env, robot, config)
        self.arm = ArmController(self.ctx)

    @staticmethod
    def _extend_trajectory(
        observations: list[Any],
        actions: list[Any],
        new_observations: list[Any],
        new_actions: list[Any],
    ) -> None:
        observations.extend(new_observations)
        actions.extend(new_actions)

    def pick_object(self, target_obj) -> tuple[bool, list, list]:
        """Execute full pick sequence."""
        observations = []
        actions = []

        obj_pos, _ = target_obj.get_position_orientation()
        logger.info("Picking object at (%.2f, %.2f, %.2f)",
                    obj_pos[0].item(), obj_pos[1].item(), obj_pos[2].item())

        # Base pose is pre-sampled near the support.
        print("[Pick] Frozen-base mode: manipulation-only pick", flush=True)

        # Use base-aligned bbox center in world XY.
        bbox_center_world, _, bbox_extent_in_aligned_frame, _ = target_obj.get_base_aligned_bbox(xy_aligned=True)
        top_z = bbox_center_world[2].item() + bbox_extent_in_aligned_frame[2].item() / 2.0
        grasp_pos = th.tensor([bbox_center_world[0].item(), bbox_center_world[1].item(), top_z])
        print(f"[Pick] Grasp target: ({grasp_pos[0].item():.3f}, {grasp_pos[1].item():.3f}, {grasp_pos[2].item():.3f})", flush=True)
        success, obs, acts = self.arm.align_to_object(grasp_pos)
        self._extend_trajectory(observations, actions, obs, acts)
        if not success:
            print("[Pick] FAIL: Hover alignment failed", flush=True)
            return False, observations, actions

        success, obs, acts = self.arm.descend_until_grasp_contact(target_obj)
        self._extend_trajectory(observations, actions, obs, acts)
        if not success:
            print("[Pick] FAIL: No valid target contact during descend", flush=True)
            return False, observations, actions

        print("[Pick] SUCCESS - target contacted and grasped", flush=True)
        return True, observations, actions
