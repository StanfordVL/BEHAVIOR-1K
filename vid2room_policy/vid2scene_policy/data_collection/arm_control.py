import logging
import math

import omnigibson.utils.transform_utils as T
import torch as th

from .robot_context import RobotContext

logger = logging.getLogger(__name__)

FINGER_ATTACHMENT_Z = 0.058
TOP_GRASP_CONTACT_OFFSET_M = 0.008
TOP_GRASP_HOVER_OFFSET_M = 0.04


class ArmStuckError(RuntimeError):
    pass


class ArmController:
    HOVER_GAIN_XY = 0.30
    HOVER_GAIN_Z = 0.32
    HOVER_MAX_DX_DY_CMD = 0.010
    HOVER_MAX_DZ_CMD = 0.006
    HOVER_XY_DEADBAND_M = 0.003
    HOVER_XY_FINAL_TOL_M = 0.008
    HOVER_XY_LOCK_STEPS = 120
    HOVER_XY_LOCK_GAIN = 0.35
    CONTACT_PROBE_STEPS = 1000
    CONTACT_Z_GAIN = 0.30
    CONTACT_XY_GAIN = 0.16
    CONTACT_XY_TOL_M = 0.008
    CONTACT_MIN_DZ_CMD = -0.007
    CONTACT_MAX_DZ_CMD = 0.0
    CONTACT_SEARCH_EXTRA_DEPTH_M = 0.07
    CONTACT_SEARCH_MIN_DESCEND_CMD = -0.004
    HOVER_STUCK_PROGRESS_EPS = 5e-4
    HOVER_STUCK_STEPS = 80
    XY_LOCK_STUCK_PROGRESS_EPS = 2e-4
    XY_LOCK_STUCK_STEPS = 50
    CONTACT_STUCK_PROGRESS_EPS = 2e-4
    CONTACT_STUCK_STEPS = 120

    def __init__(self, ctx: RobotContext):
        self.ctx = ctx
        self._last_grasp_target: tuple[float, float, float] | None = None

    @staticmethod
    def _clip(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def _apply_ik_delta(self, action: th.Tensor, dx: float, dy: float, dz: float) -> None:
        """Write Cartesian delta commands for IK arm controllers."""
        if len(self.ctx.arm_idx) < 3:
            return
        action[self.ctx.arm_idx[0]] = self._clip(dx, -0.03, 0.03)
        action[self.ctx.arm_idx[1]] = self._clip(dy, -0.03, 0.03)
        action[self.ctx.arm_idx[2]] = self._clip(dz, -0.015, 0.015)

    def _calculate_grasp_point_offset(self) -> tuple[float, float]:
        joint_positions = self.ctx.robot.get_joint_positions()
        joint_names = list(self.ctx.robot.joints.keys())

        wrist_pitch_idx = None
        for i, name in enumerate(joint_names):
            if "wrist_pitch" in name:
                wrist_pitch_idx = i
                break

        if wrist_pitch_idx is not None:
            wrist_pitch = joint_positions[wrist_pitch_idx].item()
        else:
            wrist_pitch = -0.75

        horizontal_offset = FINGER_ATTACHMENT_Z * math.cos(abs(wrist_pitch))
        vertical_offset = -FINGER_ATTACHMENT_Z * math.sin(abs(wrist_pitch))
        return horizontal_offset, vertical_offset

    def _compute_base_frame_pos_error(self, tx: float, ty: float, tz: float) -> tuple[float, float, float]:
        """Compute Cartesian position error in robot base frame."""
        base_pos, base_quat = self.ctx.robot.get_position_orientation()
        target_rel, _ = T.relative_pose_transform(
            th.tensor([tx, ty, tz], dtype=th.float32),
            th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
            th.as_tensor(base_pos, dtype=th.float32),
            th.as_tensor(base_quat, dtype=th.float32),
        )
        eef_pos_world, _ = self.ctx.robot.get_eef_pose(self.ctx.arm)
        eef_rel, _ = T.relative_pose_transform(
            th.as_tensor(eef_pos_world, dtype=th.float32),
            th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
            th.as_tensor(base_pos, dtype=th.float32),
            th.as_tensor(base_quat, dtype=th.float32),
        )
        pos_err_rel = th.as_tensor(target_rel, dtype=th.float32) - th.as_tensor(eef_rel, dtype=th.float32)
        return float(pos_err_rel[0].item()), float(pos_err_rel[1].item()), float(pos_err_rel[2].item())

    @staticmethod
    def _is_target_contact(contact_prim_path: str, target_obj) -> bool:
        target_root = getattr(target_obj, "prim_path", "")
        target_name = getattr(target_obj, "name", "")
        return (bool(target_root) and contact_prim_path.startswith(target_root)) or (
            bool(target_name) and target_name in contact_prim_path
        )

    def _eef_touches_target_bbox(self, target_obj, margin_m: float = 0.01) -> bool:
        """Fallback touch check when simulator contact paths miss target matching."""
        try:
            eef_pos, _ = self.ctx.robot.get_eef_pose(self.ctx.arm)
            aabb = target_obj.aabb
            min_xyz = th.as_tensor(aabb[0], dtype=th.float32) - margin_m
            max_xyz = th.as_tensor(aabb[1], dtype=th.float32) + margin_m
            eef = th.as_tensor(eef_pos, dtype=th.float32)
            return bool(((eef >= min_xyz) & (eef <= max_xyz)).all().item())
        except Exception:
            return False

    def _eef_to_fingertip_z_offset(self) -> float:
        """Estimate EEF-to-fingertip clearance for top grasps."""
        lengths = getattr(self.ctx.robot, "eef_to_fingertip_lengths", None)
        if isinstance(lengths, dict):
            arm_lengths = lengths.get(self.ctx.arm, None)
            if isinstance(arm_lengths, dict) and len(arm_lengths) > 0:
                avg = sum(float(v) for v in arm_lengths.values()) / len(arm_lengths)
                return self._clip(avg, 0.04, 0.18)
        return 0.10

    def _ik_precheck_reachable(self, target_pos_world: th.Tensor) -> tuple[bool, str]:
        """Quick IK feasibility check using the controller Jacobian."""
        try:
            arm_controller = self.ctx.robot.controllers.get(f"arm_{self.ctx.arm}")
            if arm_controller is None:
                return False, "missing arm controller"

            control_dict = self.ctx.robot.get_control_dict()
            task_name = arm_controller.task_name

            j_eef = th.as_tensor(control_dict[f"{task_name}_jacobian_relative"], dtype=th.float32)[
                :, arm_controller.dof_idx
            ]
            ee_pos_rel = th.as_tensor(control_dict[f"{task_name}_pos_relative"], dtype=th.float32)

            base_pos, base_quat = self.ctx.robot.get_position_orientation()
            target_pos_rel, _ = T.relative_pose_transform(
                th.as_tensor(target_pos_world, dtype=th.float32),
                th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32),
                th.as_tensor(base_pos, dtype=th.float32),
                th.as_tensor(base_quat, dtype=th.float32),
            )
            target_pos_rel = th.as_tensor(target_pos_rel, dtype=th.float32)

            pos_err = target_pos_rel - ee_pos_rel
            err = th.cat([pos_err, th.zeros(3, dtype=th.float32)])

            j_pinv = th.linalg.pinv(j_eef)
            delta_j = j_pinv @ err
            residual = err - (j_eef @ delta_j)

            pos_residual_norm = float(th.norm(residual[:3]).item())
            delta_norm = float(th.norm(delta_j).item())
            rank_xyz = int(th.linalg.matrix_rank(j_eef[:3, :]).item())

            if rank_xyz < 3:
                return False, f"low translational jacobian rank={rank_xyz}"
            if not th.isfinite(delta_j).all():
                return False, "non-finite IK delta"
            if pos_residual_norm > 0.06:
                return False, f"high IK residual={pos_residual_norm:.3f}"
            if delta_norm > 2.5:
                return False, f"large joint delta norm={delta_norm:.3f}"

            return True, (
                f"ok residual={pos_residual_norm:.3f}, "
                f"delta_norm={delta_norm:.3f}, rank_xyz={rank_xyz}"
            )
        except Exception as exc:
            return False, f"precheck exception: {exc}"

    def align_to_object(self, object_pos: th.Tensor) -> tuple[bool, list, list]:
        observations = []
        actions = []

        if not self.ctx.is_ik_arm:
            print("[Arm] align_to_object requires IK arm controller", flush=True)
            return False, observations, actions

        target_x = object_pos[0].item()
        target_y = object_pos[1].item()
        fingertip_z = self._eef_to_fingertip_z_offset()
        target_z = object_pos[2].item() + fingertip_z + TOP_GRASP_CONTACT_OFFSET_M
        hover_z = target_z + TOP_GRASP_HOVER_OFFSET_M
        self._last_grasp_target = (target_x, target_y, target_z)

        reachable, reason = self._ik_precheck_reachable(th.tensor([target_x, target_y, hover_z]))
        if not reachable:
            print(f"[Arm] IK precheck failed: {reason}", flush=True)
            return False, observations, actions
        print(f"[Arm] IK precheck: {reason}", flush=True)

        print(
            f"[Arm] IK targets: object_xy=({object_pos[0].item():.3f}, {object_pos[1].item():.3f}), "
            f"target_xy=({target_x:.3f}, {target_y:.3f}), "
            f"grasp_z={target_z:.3f}, hover_z={hover_z:.3f}",
            flush=True,
        )

        def _ik_stage(
            stage_name: str,
            tx: float,
            ty: float,
            tz: float,
            max_steps: int,
            xy_tol: float,
            z_tol: float,
            gain_xy: float,
            gain_z: float,
        ) -> bool:
            best_metric = float("inf")
            stalled_steps = 0
            for step in range(max_steps):
                dx, dy, dz = self._compute_base_frame_pos_error(tx, ty, tz)
                horiz_dist = math.sqrt(dx * dx + dy * dy)
                metric = horiz_dist + abs(dz)

                if horiz_dist < xy_tol and abs(dz) < z_tol:
                    print(
                        f"[Arm] {stage_name} reached at step {step}, "
                        f"horiz={horiz_dist:.3f}, z_err={dz:.3f}",
                        flush=True,
                    )
                    return True

                if best_metric - metric > self.HOVER_STUCK_PROGRESS_EPS:
                    best_metric = metric
                    stalled_steps = 0
                else:
                    stalled_steps += 1
                    if stalled_steps >= self.HOVER_STUCK_STEPS:
                        raise ArmStuckError(
                            f"{stage_name} stuck: metric={metric:.4f}, horiz={horiz_dist:.4f}, z_err={dz:.4f}"
                        )

                action = self.ctx.empty_action()
                action[self.ctx.gripper_idx] = -1.0
                dx_cmd = self._clip(dx * gain_xy, -self.HOVER_MAX_DX_DY_CMD, self.HOVER_MAX_DX_DY_CMD)
                dy_cmd = self._clip(dy * gain_xy, -self.HOVER_MAX_DX_DY_CMD, self.HOVER_MAX_DX_DY_CMD)
                if abs(dx) < self.HOVER_XY_DEADBAND_M:
                    dx_cmd = 0.0
                if abs(dy) < self.HOVER_XY_DEADBAND_M:
                    dy_cmd = 0.0
                dz_cmd = self._clip(dz * gain_z, -self.HOVER_MAX_DZ_CMD, self.HOVER_MAX_DZ_CMD)
                self._apply_ik_delta(action, dx_cmd, dy_cmd, dz_cmd)

                obs, _, terminated, truncated, _ = self.ctx.env.step(action.numpy())
                observations.append(obs)
                actions.append(action.numpy())
                if terminated or truncated:
                    return False
            return False

        print("[Arm] Stage 1/1: move to hover", flush=True)
        if not _ik_stage(
            stage_name="hover",
            tx=target_x,
            ty=target_y,
            tz=hover_z,
            max_steps=200,
            xy_tol=0.03,
            z_tol=0.025,
            gain_xy=self.HOVER_GAIN_XY,
            gain_z=self.HOVER_GAIN_Z,
        ):
            return False, observations, actions

        best_horiz_dist = float("inf")
        stalled_steps = 0
        for step in range(self.HOVER_XY_LOCK_STEPS):
            dx, dy, _ = self._compute_base_frame_pos_error(target_x, target_y, hover_z)
            horiz_dist = math.sqrt(dx * dx + dy * dy)

            if horiz_dist <= self.HOVER_XY_FINAL_TOL_M:
                print(f"[Arm] XY lock reached at step {step}, err={horiz_dist:.4f}m", flush=True)
                return True, observations, actions

            if best_horiz_dist - horiz_dist > self.XY_LOCK_STUCK_PROGRESS_EPS:
                best_horiz_dist = horiz_dist
                stalled_steps = 0
            else:
                stalled_steps += 1
                if stalled_steps >= self.XY_LOCK_STUCK_STEPS:
                    raise ArmStuckError(f"xy-lock stuck: err={horiz_dist:.4f}m")

            action = self.ctx.empty_action()
            action[self.ctx.gripper_idx] = -1.0
            dx_cmd = self._clip(dx * self.HOVER_XY_LOCK_GAIN, -self.HOVER_MAX_DX_DY_CMD, self.HOVER_MAX_DX_DY_CMD)
            dy_cmd = self._clip(dy * self.HOVER_XY_LOCK_GAIN, -self.HOVER_MAX_DX_DY_CMD, self.HOVER_MAX_DX_DY_CMD)
            if abs(dx) < self.HOVER_XY_DEADBAND_M:
                dx_cmd = 0.0
            if abs(dy) < self.HOVER_XY_DEADBAND_M:
                dy_cmd = 0.0
            self._apply_ik_delta(action, dx_cmd, dy_cmd, 0.0)

            obs, _, terminated, truncated, _ = self.ctx.env.step(action.numpy())
            observations.append(obs)
            actions.append(action.numpy())

            if terminated or truncated:
                return False, observations, actions

        dx, dy, _ = self._compute_base_frame_pos_error(target_x, target_y, hover_z)
        raise ArmStuckError(f"xy-lock timeout: remaining err={math.sqrt(dx * dx + dy * dy):.4f}m")

    def descend_until_grasp_contact(self, target_obj) -> tuple[bool, list, list]:
        """
        Descend from hover until sticky grasp attaches to an object.

        Success condition:
            - Robot reports an object in hand AND the object is the target.
        Failure conditions:
            - Wrong object gets attached.
            - Probe depth exceeded with no contact.
            - Episode terminates/truncates.
        """
        observations: list = []
        actions: list = []

        if not self.ctx.is_ik_arm:
            print("[Arm] descend_until_grasp_contact requires IK arm controller", flush=True)
            return False, observations, actions
        if self._last_grasp_target is None:
            print("[Arm] No grasp target cached from align_to_object", flush=True)
            return False, observations, actions

        target_x, target_y, target_z = self._last_grasp_target
        search_floor_z = target_z - self.CONTACT_SEARCH_EXTRA_DEPTH_M
        best_metric = float("inf")
        stalled_steps = 0
        print("[Arm] Descend for contact/grasp", flush=True)
        for step in range(self.CONTACT_PROBE_STEPS):
            contacts, _ = self.ctx.robot._find_gripper_contacts(arm=self.ctx.arm)
            if contacts:
                if any(self._is_target_contact(contact_path, target_obj) for contact_path in contacts):
                    print(f"[Arm] Contact success with target at step {step}", flush=True)
                    return True, observations, actions
                first_contact = sorted(contacts)[0]
                print(
                    f"[Arm] Contact on non-target object at step {step}: {first_contact}",
                    flush=True,
                )
                return False, observations, actions
            if self._eef_touches_target_bbox(target_obj):
                print(f"[Arm] Contact success via bbox touch at step {step}", flush=True)
                return True, observations, actions

            eef_pos, _ = self.ctx.robot.get_eef_pose(self.ctx.arm)
            dx, dy, dz = self._compute_base_frame_pos_error(target_x, target_y, target_z)
            horiz_dist = math.sqrt(dx * dx + dy * dy)
            metric = abs(dz) + 0.5 * horiz_dist

            obj_in_hand = self.ctx.robot._ag_obj_in_hand.get(self.ctx.arm)
            if obj_in_hand is not None:
                if obj_in_hand.name == target_obj.name:
                    print(f"[Arm] Contact success with target: {obj_in_hand.name}", flush=True)
                    return True, observations, actions
                print(
                    f"[Arm] Contact on wrong object: {obj_in_hand.name} (expected {target_obj.name})",
                    flush=True,
                )
                return False, observations, actions

            action = self.ctx.empty_action()
            action[self.ctx.gripper_idx] = -1.0

            xy_gain = self.CONTACT_XY_GAIN if horiz_dist > self.CONTACT_XY_TOL_M else 0.0
            dz_cmd = self._clip(dz * self.CONTACT_Z_GAIN, self.CONTACT_MIN_DZ_CMD, self.CONTACT_MAX_DZ_CMD)
            # Continue a bounded descent until contact.
            eef_z_world = eef_pos[2].item()
            if eef_z_world > search_floor_z:
                dz_cmd = min(dz_cmd, self.CONTACT_SEARCH_MIN_DESCEND_CMD)
            self._apply_ik_delta(action, dx * xy_gain, dy * xy_gain, dz_cmd)

            obs, _, terminated, truncated, _ = self.ctx.env.step(action.numpy())
            observations.append(obs)
            actions.append(action.numpy())

            contacts, _ = self.ctx.robot._find_gripper_contacts(arm=self.ctx.arm)
            if contacts:
                if any(self._is_target_contact(contact_path, target_obj) for contact_path in contacts):
                    print(f"[Arm] Contact success with target at step {step} (post-step)", flush=True)
                    return True, observations, actions
                first_contact = sorted(contacts)[0]
                print(
                    f"[Arm] Contact on non-target object at step {step} (post-step): {first_contact}",
                    flush=True,
                )
                return False, observations, actions
            if self._eef_touches_target_bbox(target_obj):
                print(f"[Arm] Contact success via bbox touch at step {step} (post-step)", flush=True)
                return True, observations, actions

            obj_in_hand = self.ctx.robot._ag_obj_in_hand.get(self.ctx.arm)
            if obj_in_hand is not None:
                if obj_in_hand.name == target_obj.name:
                    print(f"[Arm] Contact success with target: {obj_in_hand.name} (post-step)", flush=True)
                    return True, observations, actions
                print(
                    f"[Arm] Contact on wrong object: {obj_in_hand.name} (expected {target_obj.name}) (post-step)",
                    flush=True,
                )
                return False, observations, actions

            if best_metric - metric > self.CONTACT_STUCK_PROGRESS_EPS:
                best_metric = metric
                stalled_steps = 0
            else:
                stalled_steps += 1
                if stalled_steps >= self.CONTACT_STUCK_STEPS:
                    raise ArmStuckError(
                        f"descend stuck: metric={metric:.4f}, horiz={horiz_dist:.4f}, z_err={dz:.4f}"
                    )
            if terminated or truncated:
                return False, observations, actions

        return False, observations, actions

    def lift_slightly(self, max_steps: int = 20, dz: float = 0.01) -> tuple[list, list]:
        """Small post-grasp lift to secure contact before verification."""
        observations = []
        actions = []
        for _ in range(max_steps):
            action = self.ctx.empty_action()
            if self.ctx.is_ik_arm:
                self._apply_ik_delta(action, 0.0, 0.0, dz)
            else:
                action[self.ctx.arm_idx[0]] = dz
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

            action = self.ctx.empty_action()
            action[self.ctx.arm_idx[0]] = -0.01
            action[self.ctx.gripper_idx] = -1.0

            obs, _, terminated, truncated, _ = self.ctx.env.step(action.numpy())
            observations.append(obs)
            actions.append(action.numpy())

            if terminated or truncated:
                break

        return observations, action