import numpy as np

from igibson.controllers import IsGraspingState, ControlType, GripperController

VALID_MODES = {
    "binary",
    "smooth",
    "independent",
}


class NullGripperController(GripperController):
    """
    Dummy Controller class for non-prehensile gripper control (i.e.: no control).
    This class has a zero-size command space, and returns an empty array for control
    """

    def __init__(
        self,
        control_freq,
        control_limits,
        command_input_limits="default",
        command_output_limits="default",
    ):
        """
        :param control_freq: int, controller loop frequency
        :param control_limits: Dict[str, Tuple[Array[float], Array[float]]]: The min/max limits to the outputted
            control signal. Should specify per-actuator type limits, i.e.:

            "position": [[min], [max]]
            "velocity": [[min], [max]]
            "torque": [[min], [max]]
            "has_limit": [...bool...]

            Values outside of this range will be clipped, if the corresponding joint index in has_limit is True.
        :param command_input_limits: None or "default" or Tuple[float, float] or Tuple[Array[float], Array[float]],
            if set, is the min/max acceptable inputted command. Values outside of this range will be clipped.
            If None, no clipping will be used. If "default", range will be set to (-1, 1)
        :param command_output_limits: None or "default" or Tuple[float, float] or Tuple[Array[float], Array[float]], if set,
            is the min/max scaled command. If both this value and @command_input_limits is not None,
            then all inputted command values will be scaled from the input range to the output range.
            If either is None, no scaling will be used. If "default", then this range will automatically be set
            to the @control_limits entry corresponding to self.control_type
        """
        # Immediately run super init

        # Run super init
        super().__init__(
            control_freq=control_freq,
            control_limits=control_limits,
            dof_idx=np.array([], dtype=np.int),  # no joints controlled
            command_input_limits=command_input_limits,
            command_output_limits=command_output_limits,
        )

    def reset(self):
        # No-op
        pass

    def _preprocess_command(self, command):
        # No action
        return command

    def clip_control(self, control):
        # No action
        return control

    def _command_to_control(self, command, control_dict):
        """
        Converts the (already preprocessed) inputted @command into deployable (non-clipped!) gripper
        joint control signal. Since this is a null controller, command should be an empty numpy array
        and this function will equivalently return an empty array

        :param command: Array[float], desired (already preprocessed) command to convert into control signals.
            This should always be 2D command for each gripper joint (Empty in this case)
        :param control_dict: Dict[str, Any], dictionary that should include any relevant keyword-mapped
            states necessary for controller computation.

        :return: Array[float], outputted (non-clipped!) control signal to deploy. This is an empty np.array
        """
        return np.array([])

    def is_grasping(self):
        # Null gripper cannot grasp anything
        return IsGraspingState.FALSE

    @property
    def control_type(self):
        return ControlType.POSITION

    @property
    def command_dim(self):
        return 0
