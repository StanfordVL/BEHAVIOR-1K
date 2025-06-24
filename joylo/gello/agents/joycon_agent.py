from gello.agents.agent import Agent
from gello.joycon.rumble import RumbleJoyCon, RumbleData
import numpy as np
from pyjoycon import get_L_id, get_R_id
from typing import Optional, Dict
import yaml
import torch as th

import omnigibson.utils.backend_utils as _backend_utils
from omnigibson.macros import gm

_backend_utils._compute_backend.set_methods_from_backend(
    _backend_utils._ComputeNumpyBackend if gm.USE_NUMPY_CONTROLLER_BACKEND else _backend_utils._ComputeTorchBackend
)

from omnigibson.utils.processing_utils import MovingAverageFilter, ExponentialAverageFilter


class JoyconAgent(Agent):
    """
    Agent for controlling base + additional joints
    """
    def __init__(
            self,
            calibration_dir: str,
            deadzone_threshold: float = 0.1,
            max_translation: float = 0.05,
            max_rotation: float = 0.1,
            max_trunk_translate: float = 0.05,
            max_trunk_tilt: float = 0.001,
            enable_rumble: bool = True,
            # default_trunk_translate: float = 0.0,
            # default_trunk_tilt: float = 0.0,
    ):
        self.deadzone_threshold = deadzone_threshold
        self.max_translation = max_translation
        self.max_rotation = max_rotation
        self.max_trunk_translate = max_trunk_translate
        self.max_trunk_tilt = max_trunk_tilt
        self.joystick_filters = {
            "left": MovingAverageFilter(obs_dim=2, filter_width=3),
            "right": MovingAverageFilter(obs_dim=2, filter_width=3),
        }
        self.max_gripper_cooldown = 5
        self.gripper_info = {
            "left": {
                "cooldown": 0,
                "pressed": False,
                "status": 1,                # 'status' can be either 1 or -1, working as a toggle rather than directly mapping to gripper actions
            },
            "right": {
                "cooldown": 0,
                "pressed": False,
                "status": 1,                # 'status' can be either 1 or -1, working as a toggle rather than directly mapping to gripper actions
            },
            "-": {
                "cooldown": 0,
                "pressed": False,
                "status": 1,                # 'status' can be either 1 or -1, working as a toggle rather than directly mapping to external actions
            },
            "+": {
                "cooldown": 0,
                "pressed": False,
                "status": 1,                # 'status' can be either 1 or -1, working as a toggle rather than directly mapping to external actions
            },
        }
        self.enable_rumble = enable_rumble
        self.rumble_info = {
            "left": None,
            "right": None,
        }
        # self.current_tilt = default_tilt
        
        self.velocity_filters = {
            "left": ExponentialAverageFilter(obs_dim=2, alpha=0.2),
            "right": ExponentialAverageFilter(obs_dim=2, alpha=0.2),
        }

        # Connect the joycons
        jc_id_left = get_L_id()
        jc_id_right = get_R_id()
        assert jc_id_left[0] is not None, "Failed to connect to Left JoyCon!"
        assert jc_id_right[0] is not None, "Failed to connect to Right JoyCon!"
        self.jc_left = RumbleJoyCon(*jc_id_left)
        self.jc_right = RumbleJoyCon(*jc_id_right)

        # Load calibration data
        self.calibration_data = {"joystick": {}}
        left_serial = jc_id_left[2]
        right_serial = jc_id_right[2]

        for serial, side in zip((left_serial, right_serial), ("left", "right")):
            try:
                path_serial = serial.replace(':', '-')
                with open(f"{calibration_dir}/joycon_calibration_{path_serial}.yaml", "r") as f:
                    self.calibration_data["joystick"][side] = yaml.load(f, Loader=yaml.FullLoader)["joystick"]

            except FileNotFoundError as e:
                raise FileNotFoundError(f"""No calibration data found for {side} joycon (serial={serial})! \nTry running scripts/calibrate_joycons.py""") from e

    def act(self, obs: Dict[str, np.ndarray]) -> th.tensor:
        # Vibrate motors if we're in contact
        finger_freq = 880
        arm_freq = 320
        trunk_freq = 160
        base_freq = 40
        finger_amp = 0.01
        arm_amp = 0.2 #0.3
        trunk_amp = 0.2 #0.3
        base_amp = 0.3 #0.37
        finger_b = RumbleData(finger_freq / 2, finger_freq, finger_amp).GetData()
        arm_b = RumbleData(arm_freq / 2, arm_freq, arm_amp).GetData()

        if self.enable_rumble:
            cur_rumble_info = {
                "left": None,
                "right": None,
            }
    
            # Parse fingers, then arms, then trunk, then base
            # This defines the rumble priority (in increasing order)
    
            for arm in ["left", "right"]:
                if obs[f"arm_{arm}_finger_max_contact"] > 0:
                    # print(obs[f"arm_{arm}_finger_max_contact"])
                    # Handle finger grasping haptic feedback
                    # finger_amp = #min(np.sqrt(obs[f"arm_{arm}_finger_max_contact"]) * 0.5, 0.6)
                    cur_rumble_info[arm] = finger_b
    
                # Handle arm grasping haptic feedback
                if obs[f"arm_{arm}_contact"]:
                    cur_rumble_info[arm] = arm_b
    
            # Handle trunk collision haptic feedback
            if obs[f"trunk_contact"]:
                trunk_b = RumbleData(trunk_freq / 2, trunk_freq, trunk_amp).GetData()
                cur_rumble_info["left"] = trunk_b
                cur_rumble_info["right"] = trunk_b
    
            # Handle base collision haptic feedback
            if obs[f"base_contact"]:
                base_b = RumbleData(base_freq / 2, base_freq, base_amp).GetData()
                cur_rumble_info["left"] = base_b
                cur_rumble_info["right"] = base_b
    
            # Send values if active
            for arm, joycon in zip(["left", "right"], [self.jc_left, self.jc_right]):
                cur_val, previous_val = cur_rumble_info[arm], self.rumble_info[arm]
                if cur_val:
                    # We should be active -- enable rumble if not already done
                    if previous_val is None:
                        joycon.enable_vibration(True)
                    # Send rumble
                    joycon._send_rumble(cur_val)
                # Otherwise -- should be off -- disable if we're previously on
                elif previous_val is not None:
                    joycon.enable_vibration(False)
    
            # Update rumble info
            self.rumble_info = cur_rumble_info

        vals = []

        # Read raw joycon values
        joystick_values_raw = {
            "left": {
                "horizontal": self.jc_left.get_stick_left_horizontal(),
                "vertical": self.jc_left.get_stick_left_vertical(),
            },
            "right": {
                "horizontal": self.jc_right.get_stick_right_horizontal(),
                "vertical": self.jc_right.get_stick_right_vertical(),
            }
        }

        # Apply calibration and update moving average filters
        joystick_values = {}

        for side in ("left", "right"):
            this_joystick_vals = np.zeros(2)

            for i, direction in enumerate(("vertical", "horizontal")):
                limits = self.calibration_data["joystick"][side][direction]["limits"]
                val_raw = joystick_values_raw[side][direction]

                # Apply limits and deadzone
                half_range = (limits[1] - limits[0]) / 2
                center_value = (limits[1] + limits[0]) / 2
                val = (np.clip(val_raw, *limits) - center_value) / half_range
                if np.abs(val) < self.deadzone_threshold:
                    val = 0.0

                this_joystick_vals[i] = val

            # Update moving average filters
            joystick_values[side] = self.joystick_filters[side].estimate(observation=this_joystick_vals)

        # Use joysticks to genreate array of motions for the base
        # Values are:  (base_dx, base_dy, base_drz, trunk_dz, trunk_dry)
        base_trunk_vals = np.zeros(5)
        base_speed = self.max_translation if not self.jc_left.get_button_l_stick() else self.max_translation * 2.0
        
        # Get filtered joystick values with smooth acceleration
        left_filtered = self.velocity_filters["left"].estimate(joystick_values["left"])
        right_filtered = self.velocity_filters["right"].estimate(joystick_values["right"])

        # Left stick is (base_dx, base_dy)
        base_trunk_vals[:2] = left_filtered * base_speed * np.array([1.0, -1.0])
        # Right stick is (trunk_dry, base_drz); only apply trunk_dry if the right stick is pressed
        trunk_tilt_value = -self.max_trunk_tilt if not self.jc_right.get_button_r_stick() else 0
        base_yaw_value = -self.max_rotation if not self.jc_right.get_button_r_stick() else 0
        base_trunk_vals[[4, 2]] = right_filtered * np.array([trunk_tilt_value, base_yaw_value])
 
            
        # Left joycon up/down buttons control (trunk_dz or combined trunk dz and tilt)
        trunk_translate = 0.0
        if self.jc_left.get_button_up():
            trunk_translate = self.max_trunk_translate
        elif self.jc_left.get_button_down():
            trunk_translate = -self.max_trunk_translate
        base_trunk_vals[3] = trunk_translate

        # Add all values to the values array
        vals += base_trunk_vals.tolist()

        # Get L / R gripper action
        for arm, button_pressed in zip(("left", "right", "-", "+"), (self.jc_left.get_button_zl(), self.jc_right.get_button_zr(), self.jc_left.get_button_minus(), self.jc_right.get_button_plus())):
            gripper_info = self.gripper_info[arm]
            # Toggle grasping state if cooldown is 0 and transition from F -> T
            if gripper_info["cooldown"] == 0 and not gripper_info["pressed"] and button_pressed:
                gripper_info["status"] = -gripper_info["status"]
                gripper_info["cooldown"] = self.max_gripper_cooldown
            # Update internal gripper info
            gripper_info["cooldown"] = max(gripper_info["cooldown"] - 1, 0)
            gripper_info["pressed"] = button_pressed
            vals.append(gripper_info["status"])

        # Get X, Y, B, A, capture, home, left arrow, right arrow buttons
        vals.append(self.jc_right.get_button_x())
        vals.append(self.jc_right.get_button_y())
        vals.append(self.jc_right.get_button_b())
        vals.append(self.jc_right.get_button_a())
        vals.append(self.jc_left.get_button_capture())
        vals.append(self.jc_right.get_button_home())
        vals.append(self.jc_left.get_button_left())
        vals.append(self.jc_left.get_button_right())

        # Compose values and return
        # [base_x, base_y, base_r, trunk_translate, trunk_tilt, gripper_l, gripper_r, -, +, X, Y, B, A, capture, home, left arrow, right arrow buttons]
        return th.Tensor(vals)
