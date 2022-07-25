"""
A set of utilities for working with Omniverse cameras
"""
import numpy as np
import igibson.utils.transform_utils as T
import omni
import carb


class CameraMover:
    """
    A helper class for manipulating a camera via the keyboard. Utilizes carb keyboard callbacks to move
    the camera around.

    Args:
        cam (VisionSensor): The camera vision sensor to manipulate via the keyboard
        sim (Simulator): Active simulator instance being used
        delta (float): Change (m) per keypress when moving the camera
    """
    def __init__(self, cam, sim, delta=0.25):
        self.cam = cam
        self.sim = sim
        self.delta = delta

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)

    def print_info(self):
        """
        Prints keyboard command info out to the user
        """
        print("*" * 40)
        print("CameraMover! Commands:")
        print()
        print(f"\t Right Click + Drag: Rotate camera")
        print(f"\t W / S : Move camera forward / backward")
        print(f"\t A / D : Move camera left / right")
        print(f"\t T / G : Move camera up / down")
        print(f"\t P : Print current camera pose")

    def print_cam_pose(self):
        """
        Prints out the camera pose as (position, quaternion) in the world frame
        """
        print(f"cam pose: {self.cam.get_position_orientation()}")

    def set_delta(self, delta):
        """
        Sets the delta value (how much the camera moves with each keypress) for this CameraMover

        Args:
            delta (float): Change (m) per keypress when moving the camera
        """
        self.delta = delta

    def set_cam(self, cam):
        """
        Sets the active camera sensor for this CameraMover

        Args:
            cam (VisionSensor): The camera vision sensor to manipulate via the keyboard
        """
        self.cam = cam

    @property
    def input_to_function(self):
        """
        Returns:
            dict: Mapping from relevant keypresses to corresponding function call to use
        """
        return {
            carb.input.KeyboardInput.P: lambda: self.print_cam_pose(),
        }

    @property
    def input_to_command(self):
        """
        Returns:
            dict: Mapping from relevant keypresses to corresponding delta command to apply to the camera pose
        """
        return {
            carb.input.KeyboardInput.D: np.array([self.delta, 0, 0]),
            carb.input.KeyboardInput.A: np.array([-self.delta, 0, 0]),
            carb.input.KeyboardInput.W: np.array([0, 0, -self.delta]),
            carb.input.KeyboardInput.S: np.array([0, 0, self.delta]),
            carb.input.KeyboardInput.T: np.array([0, self.delta, 0]),
            carb.input.KeyboardInput.G: np.array([0, -self.delta, 0]),
        }

    def _sub_keyboard_event(self, event, *args, **kwargs):
        """
        Handle keyboard events. Note: The signature is pulled directly from omni.

        Args:
            event (int): keyboard event type
        """
        if event.type == carb.input.KeyboardEventType.KEY_PRESS \
                or event.type == carb.input.KeyboardEventType.KEY_REPEAT:

            if event.type == carb.input.KeyboardEventType.KEY_PRESS and event.input in self.input_to_function:
                self.input_to_function[event.input]()

            else:
                command = self.input_to_command.get(event.input, None)

                if command is not None:
                    # Convert to world frame to move the camera
                    transform = T.quat2mat(self.cam.get_orientation())
                    delta_pos_global = transform @ command
                    self.cam.set_position(self.cam.get_position() + delta_pos_global)

        return True
