import numpy as np

from omnigibson.controllers import JointController


class NullJointController(JointController):
    """
    Dummy Controller class for a null-type of joint control (i.e.: no control or constant pass-through control).
    This class has a zero-size command space, and returns either an empty array for control if dof_idx is None
    else constant values as specified by @default_command (if not specified, uses zeros)
    """

    def __init__(
        self,
        control_freq,
        motor_type,
        control_limits,
        dof_idx,
        command_input_limits="default",
        command_output_limits="default",
        default_command=None,
    ):
        """
        Args:
            control_freq (int): controller loop frequency
            control_limits (Dict[str, Tuple[Array[float], Array[float]]]): The min/max limits to the outputted
                control signal. Should specify per-dof type limits, i.e.:

                "position": [[min], [max]]
                "velocity": [[min], [max]]
                "effort": [[min], [max]]
                "has_limit": [...bool...]

                Values outside of this range will be clipped, if the corresponding joint index in has_limit is True.
            dof_idx (Array[int]): specific dof indices controlled by this robot. Used for inferring
                controller-relevant values during control computations
            command_input_limits (None or "default" or Tuple[float, float] or Tuple[Array[float], Array[float]]):
                if set, is the min/max acceptable inputted command. Values outside this range will be clipped.
                If None, no clipping will be used. If "default", range will be set to (-1, 1)
            command_output_limits (None or "default" or Tuple[float, float] or Tuple[Array[float], Array[float]]):
                if set, is the min/max scaled command. If both this value and @command_input_limits is not None,
                then all inputted command values will be scaled from the input range to the output range.
                If either is None, no scaling will be used. If "default", then this range will automatically be set
                to the @control_limits entry corresponding to self.control_type
            default_command (None or n-array): if specified, should be the same length as @dof_idx, specifying
                the default control for this controller to output
        """
        # Store values
        self._default_command = np.zeros(len(dof_idx)) if default_command is None else np.array(default_command)

        # Run super init
        super().__init__(
            control_freq=control_freq,
            motor_type=motor_type,
            control_limits=control_limits,
            dof_idx=dof_idx,
            command_input_limits=command_input_limits,
            command_output_limits=command_output_limits,
        )

    def _preprocess_command(self, command):
        # Set the command to be internal stored default value
        return np.array(self._default_command)

    def update_default_command(self, command):
        """
        Updates the internal default command value.

        Args:
            command (n-array): New default command values to set for this controller.
                Should be of dimension @command_dim
        """
        assert len(command) == self.command_dim, \
            f"Default control must be length: {self.command_dim}, got length: {len(command)}"

        self._default_command = np.array(command)
