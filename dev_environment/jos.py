import yaml
import numpy as np

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.utils.motion_planning_utils import detect_self_collision
    

def pause(time):
    for _ in range(int(time*100)):
        og.sim.step()

def main():
    # Load the config
    config_filename = "test.yaml"
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls", "coffee_table"]

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    # Allow user to move camera more easily
    og.sim.enable_viewer_camera_teleoperation()

    robot.tuck()
    og.sim.step()

    while True:
        print(detect_self_collision(robot))
        pause(1)

    # from IPython import embed; embed()
    # og.sim.step()
    # og.sim.stop()
    # og.sim.play()
    pause(10000)

if __name__ == "__main__":
    main()