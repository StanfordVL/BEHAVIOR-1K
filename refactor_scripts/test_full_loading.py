from igibson import app, ig_dataset_path, Simulator
from igibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
from igibson.robots.turtlebot import Turtlebot
import numpy as np


##### SET THIS ######
SCENE_ID = "Rs_int"
USD_TEMPLATE_FILE = f"{ig_dataset_path}/scenes/{SCENE_ID}/usd/{SCENE_ID}_best_template.usd"
#### YOU DONT NEED TO TOUCH ANYTHING BELOW HERE IDEALLY :) #####


# Create simulator
sim = Simulator()

# Load scene
scene = InteractiveTraversableScene(
    scene_model=SCENE_ID,
    usd_path=USD_TEMPLATE_FILE,
)

# Import scene
sim.import_scene(scene=scene)

# Import robot
robot = Turtlebot(prim_path=f"/World/robot", name="robot", obs_modalities=["proprio", "rgb", "scan", "occupancy_grid"])
sim.import_object(obj=robot)

# The simulator must always be playing and have a single step taken in order to initialize any added objects
sim.play()
sim.step()

# Grab the robot's observations and verify the received modalities
obs = robot.get_obs()
for obs_name, ob in obs.items():
    print(f"obs modality {obs_name} -- shape {ob.shape}")

# Set the camera to a well-known, good position for viewing the robot
sim.viewer_camera.set_position_orientation(
    position=np.array([-0.300727, -3.7592,  2.03752]),
    orientation=np.array([0.53647195, -0.02424788, -0.03808954, 0.84270936]),
)

# We also enable keyboard teleoperation of the simulator's viewer camera for convenience
sim.enable_viewer_camera_teleoperation()

# Loop indefinitely to allow the user to move around
while True:
    robot.apply_action(np.array([0.5, 0]))
    sim.step()

# Always close the app at the end
app.close()
