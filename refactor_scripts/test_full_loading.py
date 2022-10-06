import igibson as ig
from igibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
from igibson.robots.turtlebot import Turtlebot
import numpy as np


##### SET THIS ######
SCENE_ID = "Rs_int"
USD_TEMPLATE_FILE = f"{ig.ig_dataset_path}/scenes/{SCENE_ID}/usd/{SCENE_ID}_best_template.usd"
#### YOU DONT NEED TO TOUCH ANYTHING BELOW HERE IDEALLY :) #####

# Load scene
scene = InteractiveTraversableScene(
    scene_model=SCENE_ID,
    usd_path=USD_TEMPLATE_FILE,
)

# Import scene
ig.sim.import_scene(scene=scene)

# Import robot
robot = Turtlebot(prim_path=f"/World/robot", name="robot", obs_modalities=["proprio", "rgb", "scan", "occupancy_grid"])
ig.sim.import_object(obj=robot)

# The simulator must always be playing and have a single step taken in order to initialize any added objects
ig.sim.play()
ig.sim.step()

# Grab the robot's observations and verify the received modalities
obs = robot.get_obs()
for obs_name, ob in obs.items():
    print(f"obs modality {obs_name} -- shape {ob.shape}")

# Set the camera to a well-known, good position for viewing the robot
ig.sim.viewer_camera.set_position_orientation(
    position=np.array([-0.300727, -3.7592,  2.03752]),
    orientation=np.array([0.53647195, -0.02424788, -0.03808954, 0.84270936]),
)

# We also enable keyboard teleoperation of the simulator's viewer camera for convenience
ig.sim.enable_viewer_camera_teleoperation()

# Loop indefinitely to allow the user to move around
while True:
    robot.apply_action(np.array([0.5, 0]))
    ig.sim.step()

# Always close the app at the end
ig.app.close()
