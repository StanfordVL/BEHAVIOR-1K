from gello.robots.sim_robot.og_teleop_cfg import *

class BaseDevice:
    """
    Base device class that provides common functionality for all teleop devices.
    """

    def __init__(self, robot):
        self.robot = robot
        self.obs = {}
        self.grasp_action = {arm: 1 for arm in self.robot.arm_names}
        self.current_trunk_translate = DEFAULT_TRUNK_TRANSLATE
        self.current_trunk_tilt_offset = 0.0

        self._joint_state = None
        self._joint_cmd = None

    def get_base_cmd(self):
        """
        Get the base command for the robot.
        This method should be overridden by subclasses to provide specific base commands.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def get_button_input_cmd(self):
        """
        Get the button input command for the robot.
        This method should be overridden by subclasses to provide specific button input commands.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def reset(self):
        self.grasp_action = {arm: 1 for arm in self.robot.arm_names}
        self.current_trunk_translate = DEFAULT_TRUNK_TRANSLATE
        self.current_trunk_tilt_offset = 0.0