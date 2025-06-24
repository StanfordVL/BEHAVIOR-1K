from pyjoycon import JoyCon, ButtonEventJoyCon, get_R_id, get_L_id
import sys, select, os
import time
import numpy as np
import yaml
from gello import REPO_DIR

SAVE_DIR = f"{REPO_DIR}/configs"

jc_id_left = get_L_id()
jc_id_right = get_R_id()
assert jc_id_left[0] is not None
assert jc_id_right[0] is not None

jc_left = JoyCon(*jc_id_left)
jc_right = JoyCon(*jc_id_right)

time.sleep(1)

init_left_horizontal_value = jc_left.get_stick_left_horizontal()
init_left_vertical_value = jc_left.get_stick_left_vertical()
init_right_horizontal_value = jc_right.get_stick_right_horizontal()
init_right_vertical_value = jc_right.get_stick_right_vertical()
joystick_limits = {
    "left": {
        "horizontal": {
            "center": init_left_horizontal_value,
            "limits": [init_left_horizontal_value, init_left_horizontal_value],
        },
        "vertical": {
            "center": init_left_vertical_value,
            "limits": [init_left_vertical_value, init_left_vertical_value],
        },
    },
    "right": {
        "horizontal": {
            "center": init_right_horizontal_value,
            "limits": [init_right_horizontal_value, init_right_horizontal_value],
        },
        "vertical": {
            "center": init_right_vertical_value,
            "limits": [init_right_vertical_value, init_right_vertical_value],
        },
    },
}

print("Move joycon joysticks in a circular motion. Press ENTER to quit calibration.")
while True:
    # Update limits
    left_horizontal_value = jc_left.get_stick_left_horizontal()
    left_vertical_value = jc_left.get_stick_left_vertical()
    right_horizontal_value = jc_right.get_stick_right_horizontal()
    right_vertical_value = jc_right.get_stick_right_vertical()
    for val, side, direction in zip(
            (left_horizontal_value, left_vertical_value, right_horizontal_value, right_vertical_value),
            ("left", "left", "right", "right"),
            ("horizontal", "vertical", "horizontal", "vertical"),
    ):
        limits = joystick_limits[side][direction]["limits"]
        limits[0] = min(limits[0], val)
        limits[1] = max(limits[1], val)

    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        break

# Save output
left_serial = jc_id_left[2]
right_serial = jc_id_right[2]

# Save each joycon's calibration in a file with its serial number
for serial, side in zip((left_serial, right_serial), ("left", "right")):
    path_serial = serial.replace(":", "-")
    with open(f"{SAVE_DIR}/joycon_calibration_{path_serial}.yaml", "w+") as f:
        yaml.dump({"joystick": joystick_limits[side], "side": side}, f)

print("\nSuccessfully wrote calibration data!")