import math
import logging

import torch as th

from .robot_context import RobotContext, MODE_MANIPULATION

logger = logging.getLogger(__name__)

GRIPPER_OFFSET = -math.radians(105)
EXTEND_STEP = 0.006
MAX_JOINT_EXTEND = 0.18
YAW_THRESHOLD = 0.005
FINGER_ATTACHMENT_Z = 0.058
Z_OFFSET_ADJUSTMENT = -0.02
CLOSER_ADJUSTMENT = 0.11


class ArmController:
    def __init__(self, ctx: RobotContext):
        self.ctx = ctx

    def _calculate_grasp_point_offset(self) -> tuple[float, float]:
        joint_positions = self.ctx.robot.get_joint_positions()
        joint_names = list(self.ctx.robot.joints.keys())

        wrist_pitch_idx = None
        for i, name in enumerate(joint_names):
            if 'wrist_pitch' in name:
                wrist_pitch_idx = i
                break

        if wrist_pitch_idx is not None:
            wrist_pitch = joint_positions[wrist_pitch_idx].item()
        else:
            wrist_pitch = -0.75

        horizontal_offset = FINGER_ATTACHMENT_Z * math.cos(abs(wrist_pitch))
        vertical_offset = -FINGER_ATTACHMENT_Z * math.sin(abs(wrist_pitch))
        return horizontal_offset, vertical_offset

    def align_to_object(self, object_pos: th.Tensor, keep_gripper_open: bool = True) -> tuple[bool, list, list]:
        """Position gripper at object: first align z, then extend arm."""
        observations = []
        actions = []

        horiz_offset, vert_offset = self._calculate_grasp_point_offset()

        robot_pos, _ = self.ctx.robot.get_position_orientation()
        obj_dx = object_pos[0].item() - robot_pos[0].item()
        obj_dy = object_pos[1].item() - robot_pos[1].item()
        obj_dist = math.sqrt(obj_dx * obj_dx + obj_dy * obj_dy)

        if obj_dist > 0.01:
            dir_x = obj_dx / obj_dist
            dir_y = obj_dy / obj_dist
        else:
            dir_x, dir_y = 1.0, 0.0

        effective_horiz = max(0.0, horiz_offset - CLOSER_ADJUSTMENT)
        target_eef_x = object_pos[0].item() - dir_x * effective_horiz
        target_eef_y = object_pos[1].item() - dir_y * effective_horiz
        target_eef_z = object_pos[2].item()
        print(f"[Arm] Target EEF: ({target_eef_x:.3f}, {target_eef_y:.3f}, {target_eef_z:.3f}), obj_z={object_pos[2].item():.3f}", flush=True)

        # Phase 1: Align Z (lift) first
        print("[Arm] Phase 1: Aligning Z...", flush=True)
        for step in range(100):
            eef_pos, _ = self.ctx.robot.get_eef_pose(self.ctx.arm)
            height_error = target_eef_z - eef_pos[2].item()

            if abs(height_error) < 0.02:
                print(f"[Arm] Z aligned at step {step}, error={height_error:.3f}", flush=True)
                break

            action = self.ctx.empty_action(mode=MODE_MANIPULATION)
            action[self.ctx.gripper_idx] = 1.0 if keep_gripper_open else -1.0
            lift_delta = max(-0.008, min(0.008, height_error * 0.3))
            action[self.ctx.arm_idx[0]] = lift_delta

            obs, _, terminated, truncated, _ = self.ctx.env.step(action.numpy())
            observations.append(obs)
            actions.append(action.numpy())

            if terminated or truncated:
                return False, observations, actions

        # Phase 2: Extend arm horizontally
        print("[Arm] Phase 2: Extending arm...", flush=True)
        joint_extensions = [0.0, 0.0, 0.0, 0.0]
        last_dist = float('inf')
        min_dist = float('inf')
        stuck_count = 0
        STUCK_THRESHOLD = 50

        for step in range(200):
            eef_pos, _ = self.ctx.robot.get_eef_pose(self.ctx.arm)
            robot_pos, _ = self.ctx.robot.get_position_orientation()

            dx = target_eef_x - eef_pos[0].item()
            dy = target_eef_y - eef_pos[1].item()
            horiz_dist = math.sqrt(dx * dx + dy * dy)
            height_error = target_eef_z - eef_pos[2].item()

            obj_dx = object_pos[0].item() - robot_pos[0].item()
            obj_dy = object_pos[1].item() - robot_pos[1].item()
            angle_to_object = math.atan2(obj_dy, obj_dx)
            target_yaw = self.ctx.normalize_angle(angle_to_object - GRIPPER_OFFSET)
            current_yaw = self.ctx.get_robot_yaw()
            yaw_error = self.ctx.normalize_angle(target_yaw - current_yaw)

            if horiz_dist < 0.03 and abs(height_error) < 0.02:
                print(f"[Arm] EEF at target at step {step}, horiz_dist={horiz_dist:.3f}, h_err={height_error:.3f}", flush=True)
                return True, observations, actions

            if horiz_dist < min_dist:
                min_dist = horiz_dist

            if horiz_dist >= last_dist - 0.002:
                stuck_count += 1
            else:
                stuck_count = 0
                last_dist = horiz_dist

            if min_dist < 0.35 and horiz_dist > min_dist + 0.05:
                print(f"[Arm] Drifted from {min_dist:.3f} to {horiz_dist:.3f} at step {step} - starting grasp", flush=True)
                return True, observations, actions

            if stuck_count >= STUCK_THRESHOLD:
                if horiz_dist < 0.15:
                    print(f"[Arm] Stuck but close enough at {horiz_dist:.3f}, step {step}", flush=True)
                    return True, observations, actions
                else:
                    print(f"[Arm] Stuck at {horiz_dist:.3f} (step {step}), resetting stuck counter", flush=True)
                    stuck_count = 0
                    last_dist = horiz_dist

            action = self.ctx.empty_action(mode=MODE_MANIPULATION)
            action[self.ctx.gripper_idx] = 1.0 if keep_gripper_open else -1.0

            # Perpendicular error
            base_to_obj_dist = math.sqrt(obj_dx * obj_dx + obj_dy * obj_dy)
            if base_to_obj_dist > 0.001:
                eef_dx = eef_pos[0].item() - robot_pos[0].item()
                eef_dy = eef_pos[1].item() - robot_pos[1].item()
                h_err = (obj_dx * eef_dy - obj_dy * eef_dx) / base_to_obj_dist
            else:
                h_err = 0.0

            # Maintain Z while extending
            if abs(height_error) > 0.01:
                lift_delta = max(-0.004, min(0.004, height_error * 0.2))
                action[self.ctx.arm_idx[0]] = lift_delta

            can_extend = horiz_dist > 0.02
            if can_extend:
                joint_order = [3, 2, 1, 0]
                arm_indices = [4, 3, 2, 1]
                for je_idx, arm_idx in zip(joint_order, arm_indices):
                    if joint_extensions[je_idx] < MAX_JOINT_EXTEND:
                        action[self.ctx.arm_idx[arm_idx]] = EXTEND_STEP
                        joint_extensions[je_idx] += EXTEND_STEP
                        break
                if step % 20 == 0:
                    total_ext = sum(joint_extensions)
                    print(f"[Arm] Step {step}: Extending - horiz_dist={horiz_dist:.3f}, total_ext={total_ext:.3f}", flush=True)

            action[self.ctx.base_idx[0]] = 0.0
            if abs(yaw_error) > YAW_THRESHOLD or abs(h_err) > 0.02:
                rot_gain = 3.0
                ang_vel = max(-1.0, min(1.0, yaw_error * rot_gain))
                action[self.ctx.base_idx[1]] = ang_vel

            obs, _, terminated, truncated, _ = self.ctx.env.step(action.numpy())
            observations.append(obs)
            actions.append(action.numpy())

            if terminated or truncated:
                return False, observations, actions

        total_ext = sum(joint_extensions)
        eef_pos, _ = self.ctx.robot.get_eef_pose(self.ctx.arm)
        final_dx = target_eef_x - eef_pos[0].item()
        final_dy = target_eef_y - eef_pos[1].item()
        final_dist = math.sqrt(final_dx * final_dx + final_dy * final_dy)
        print(f"[Arm] Phase 2 complete: total_extension={total_ext:.3f}, final_dist={final_dist:.3f}", flush=True)
        return True, observations, actions

    def lift_up(self, amount: float = 0.2) -> tuple[list, list]:
        observations = []
        actions = []

        lift_joint_idx = list(self.ctx.robot.joints.keys()).index("joint_lift")
        start_pos = self.ctx.robot.get_joint_positions()[lift_joint_idx].item()
        target_pos = start_pos + amount

        WAYPOINT_INCREMENT = 0.05
        num_waypoints = int(amount / WAYPOINT_INCREMENT) + 1
        waypoints = [start_pos + i * WAYPOINT_INCREMENT for i in range(1, num_waypoints + 1)]
        waypoints[-1] = min(waypoints[-1], target_pos)

        current_waypoint_idx = 0
        for step in range(200):
            current_pos = self.ctx.robot.get_joint_positions()[lift_joint_idx].item()

            if current_waypoint_idx < len(waypoints) and current_pos >= waypoints[current_waypoint_idx]:
                current_waypoint_idx += 1

            if current_pos >= target_pos - 0.01:
                break

            action = self.ctx.empty_action(mode=MODE_MANIPULATION)
            action[self.ctx.arm_idx[0]] = 0.015
            action[self.ctx.gripper_idx] = -1.0

            obs, _, terminated, truncated, _ = self.ctx.env.step(action.numpy())
            observations.append(obs)
            actions.append(action.numpy())

            if terminated or truncated:
                break

        return observations, actions

    def maximize_lift(self, max_steps: int = 150) -> tuple[list, list]:
        observations = []
        actions = []

        LIFT_UPPER_LIMIT = 1.05
        lift_joint_idx = list(self.ctx.robot.joints.keys()).index("joint_lift")
        start_pos = self.ctx.robot.get_joint_positions()[lift_joint_idx].item()

        WAYPOINT_INCREMENT = 0.05
        waypoints = []
        wp = start_pos + WAYPOINT_INCREMENT
        while wp < LIFT_UPPER_LIMIT:
            waypoints.append(wp)
            wp += WAYPOINT_INCREMENT
        waypoints.append(LIFT_UPPER_LIMIT)

        current_waypoint_idx = 0
        for step in range(max_steps):
            lift_pos = self.ctx.robot.get_joint_positions()[lift_joint_idx].item()

            if current_waypoint_idx < len(waypoints) and lift_pos >= waypoints[current_waypoint_idx]:
                current_waypoint_idx += 1

            if lift_pos >= LIFT_UPPER_LIMIT:
                break

            action = self.ctx.empty_action(mode=MODE_MANIPULATION)
            action[self.ctx.arm_idx[0]] = 0.015
            action[self.ctx.gripper_idx] = -1.0

            obs, _, terminated, truncated, _ = self.ctx.env.step(action.numpy())
            observations.append(obs)
            actions.append(action.numpy())

            if terminated or truncated:
                break

        return observations, actions

    def lower_lift(self, target_height: float) -> tuple[list, list]:
        observations = []
        actions = []

        for step in range(100):
            eef_pos, _ = self.ctx.robot.get_eef_pose(self.ctx.arm)
            eef_height = eef_pos[2].item()

            if eef_height <= target_height + 0.02:
                break

            action = self.ctx.empty_action(mode=MODE_MANIPULATION)
            action[self.ctx.arm_idx[0]] = -0.01
            action[self.ctx.gripper_idx] = -1.0

            obs, _, terminated, truncated, _ = self.ctx.env.step(action.numpy())
            observations.append(obs)
            actions.append(action.numpy())

            if terminated or truncated:
                break

        return observations, actions

    def contract_arm(self) -> tuple[list, list]:
        observations = []
        actions = []

        CONTRACTION_STEPS = 60
        DELTA = -0.005

        for step in range(CONTRACTION_STEPS):
            action = self.ctx.empty_action(mode=MODE_MANIPULATION)
            action[self.ctx.arm_idx[1]] = DELTA
            action[self.ctx.arm_idx[2]] = DELTA
            action[self.ctx.arm_idx[3]] = DELTA
            action[self.ctx.arm_idx[4]] = DELTA
            action[self.ctx.gripper_idx] = -1.0

            obs, _, terminated, truncated, _ = self.ctx.env.step(action.numpy())
            observations.append(obs)
            actions.append(action.numpy())

            if terminated or truncated:
                break

        return observations, actions

    def lift_and_contract(self, max_steps: int = 80) -> tuple[list, list]:
        observations = []
        actions = []

        LIFT_UPPER_LIMIT = 1.05
        lift_joint_idx = list(self.ctx.robot.joints.keys()).index("joint_lift")
        CONTRACT_DELTA = -0.006

        for step in range(max_steps):
            lift_pos = self.ctx.robot.get_joint_positions()[lift_joint_idx].item()
            lift_done = lift_pos >= LIFT_UPPER_LIMIT

            action = self.ctx.empty_action(mode=MODE_MANIPULATION)
            if not lift_done:
                action[self.ctx.arm_idx[0]] = 0.015
            action[self.ctx.arm_idx[1]] = CONTRACT_DELTA
            action[self.ctx.arm_idx[2]] = CONTRACT_DELTA
            action[self.ctx.arm_idx[3]] = CONTRACT_DELTA
            action[self.ctx.arm_idx[4]] = CONTRACT_DELTA
            action[self.ctx.gripper_idx] = -1.0

            obs, _, terminated, truncated, _ = self.ctx.env.step(action.numpy())
            observations.append(obs)
            actions.append(action.numpy())

            if terminated or truncated:
                break

        return observations, actions

    def extend_arm_for_place(self, target_pos: th.Tensor) -> tuple[list, list]:
        observations = []
        actions = []

        MAX_STEPS = 120
        DELTA = 0.005
        DIST_THRESHOLD = 0.15

        for step in range(MAX_STEPS):
            eef_pos, _ = self.ctx.robot.eef_links[self.ctx.arm].get_position_orientation()
            dist = math.sqrt((eef_pos[0].item() - target_pos[0].item())**2 +
                           (eef_pos[1].item() - target_pos[1].item())**2)

            if dist < DIST_THRESHOLD:
                break

            action = self.ctx.empty_action(mode=MODE_MANIPULATION)
            action[self.ctx.arm_idx[1]] = DELTA
            action[self.ctx.arm_idx[2]] = DELTA
            action[self.ctx.arm_idx[3]] = DELTA
            action[self.ctx.arm_idx[4]] = DELTA
            action[self.ctx.gripper_idx] = -1.0

            obs, _, terminated, truncated, _ = self.ctx.env.step(action.numpy())
            observations.append(obs)
            actions.append(action.numpy())

            if terminated or truncated:
                break

        return observations, actions
