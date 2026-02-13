from omnigibson.utils.backend_utils import _compute_backend as cb
from omnigibson.controllers.joint_controller import JointController
from omnigibson.utils.geometry_utils import wrap_angle
from copy import deepcopy

class HolonomicBaseJointController(JointController):
    """
    Singleton controller for holonomic base joint control.
    """

    @classmethod
    def _process_config(cls, controller_id: str, input_config: dict):
        config = deepcopy(input_config)
        assert len(config["dof_idx"]) == 3, f"Expected 3 DOFs for holonomic base control, got {len(config['dof_idx'])}"
        config["use_delta_commands"] = False
        config["compute_delta_in_quat_space"] = None
        return super()._process_config(controller_id, config)

    @classmethod
    def _update_goal(cls, controller_id: str, command, control_dict):
        base_pose = cb.T.pose2mat((control_dict["root_pos"], control_dict["root_quat"]))
        canonical_pose = cb.T.pose2mat((control_dict["canonical_pos"], control_dict["canonical_quat"]))
        canonical_to_base_pose = cb.T.pose_inv(canonical_pose) @ base_pose

        if cls.motor_type(controller_id) == "position":
            command_in_base_frame = cb.as_float32(cb.eye(4))
            command_in_base_frame[:2, 3] = command[:2]
            command_in_canonical_frame = canonical_to_base_pose @ command_in_base_frame
            position = command_in_canonical_frame[:2, 3]
            rz_joint_pos = control_dict["joint_position"][cls.dof_idx(controller_id)][2:3]
            delta_joint_pos = wrap_angle(command[2])
            new_joint_pos = rz_joint_pos + delta_joint_pos
            command = cb.cat([position, new_joint_pos])
        else:
            command_in_base_frame = cb.as_float32(cb.eye(4))
            command_in_base_frame[:2, 3] = command[:2]
            canonical_to_base_pose_rotation = cb.as_float32(cb.eye(4))
            canonical_to_base_pose_rotation[:3, :3] = canonical_to_base_pose[:3, :3]
            command_in_canonical_frame = canonical_to_base_pose_rotation @ command_in_base_frame
            linear_velocity = command_in_canonical_frame[:2, 3]
            angular_velocity = command[2:3]
            command = cb.cat([linear_velocity, angular_velocity])

        return super()._update_goal(controller_id, command=command, control_dict=control_dict)

    @classmethod
    def _compute_no_op_command(cls, controller_id: str, control_dict):
        return cb.zeros(cls.command_dim(controller_id))
