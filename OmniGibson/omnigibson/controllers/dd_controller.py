from copy import deepcopy
from omnigibson.controllers.controller_base import BaseController, ControlType
from omnigibson.utils.backend_utils import _compute_backend as cb


class DifferentialDriveController(BaseController):
    """
    Differential drive (DD) controller for controlling two independently controlled wheeled joints.

    Each controller step consists of the following:
        1. Clip + Scale inputted command according to @command_input_limits and @command_output_limits
        2. Convert desired (lin_vel, ang_vel) command into (left, right) wheel joint velocity control signals
        3. Clips the resulting command by the joint velocity limits
    """

    @classmethod
    def _process_config(cls, controller_id: str, input_config: dict):
        config = deepcopy(input_config)
        config["wheel_radius"] = config["wheel_radius"]
        config["wheel_axle_halflength"] = config["wheel_axle_length"] / 2.0

        command_output_limits = config.get("command_output_limits", "default")
        if type(command_output_limits) is str and command_output_limits == "default":
            control_limits = config["control_limits"]
            dof_idx = config["dof_idx"]
            min_vels = control_limits["velocity"][0][dof_idx]
            assert (
                min_vels[0] == min_vels[1]
            ), "Differential drive requires both wheel joints to have same min velocities!"
            max_vels = control_limits["velocity"][1][dof_idx]
            assert (
                max_vels[0] == max_vels[1]
            ), "Differential drive requires both wheel joints to have same max velocities!"
            assert abs(min_vels[0]) == abs(
                max_vels[0]
            ), "Differential drive requires both wheel joints to have same min and max absolute velocities!"
            max_lin_vel = max_vels[0] * config["wheel_radius"]
            max_ang_vel = max_lin_vel * 2.0 / config["wheel_axle_halflength"]
            config["command_output_limits"] = ((-max_lin_vel, -max_ang_vel), (max_lin_vel, max_ang_vel))

        return super()._process_config(controller_id, config)

    @classmethod
    def _update_goal(cls, controller_id: str, command, control_dict):
        return dict(vel=command)

    @classmethod
    def compute_control(cls, controller_id: str, control_dict):
        """
        Converts the (already preprocessed) inputted @command into deployable (non-clipped!) joint control signal.
        This processes converts the desired (lin_vel, ang_vel) command into (left, right) wheel joint velocity control
        signals.

        Args:
            goal_dict (Dict[str, Any]): dictionary that should include any relevant keyword-mapped
                goals necessary for controller computation. Must include the following keys:
                    vel: desired (lin_vel, ang_vel) of the controlled body
            control_dict (Dict[str, Any]): dictionary that should include any relevant keyword-mapped
                states necessary for controller computation. Must include the following keys:

        Returns:
            Array[float]: outputted (non-clipped!) velocity control signal to deploy
                to the [left, right] wheel joints
        """
        goal_dict = cls._goals[controller_id]
        config = cls._configs[controller_id]
        lin_vel, ang_vel = goal_dict["vel"]

        # Convert to wheel velocities
        left_wheel_joint_vel = (lin_vel - ang_vel * config["wheel_axle_halflength"]) / config["wheel_radius"]
        right_wheel_joint_vel = (lin_vel + ang_vel * config["wheel_axle_halflength"]) / config["wheel_radius"]
        
        # Return desired velocities
        return cb.array([left_wheel_joint_vel, right_wheel_joint_vel])

    @classmethod
    def compute_no_op_goal(cls, controller_id: str, control_dict):
        return dict(vel=cb.zeros(2))

    @classmethod
    def _compute_no_op_command(cls, controller_id: str, control_dict):
        return cb.zeros(2)

    @classmethod
    def _get_goal_shapes(cls, controller_id: str):
        return dict(vel=(2,))

    @classmethod
    def control_type(cls, controller_id: str):
        return ControlType.VELOCITY

    @classmethod
    def command_dim(cls, controller_id: str):
        # [lin_vel, ang_vel]
        return 2
