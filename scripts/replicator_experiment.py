import numpy as np

import omnigibson as og
from omnigibson.macros import gm
import omnigibson.lazy as lazy

from omnigibson.sensors import VisionSensor

from omnigibson.object_states import IsGrasping, ObjectsInFOVOfRobot
from omnigibson.utils.constants import PrimType
from omnigibson.object_states import *
from omnigibson.systems import get_system
from omnigibson.utils.constants import semantic_class_name_to_id, semantic_class_id_to_name
import time


# config = {
#     "scene": {
#         "type": "Scene",
#     },
#     "objects": [
#             # {
#             #     "type": "DatasetObject",
#             #     "fit_avg_dim_volume": True,
#             #     "name": "bagel_dough",
#             #     "category": "bagel_dough",
#             #     "model": "iuembm",
#             #     "prim_type": PrimType.RIGID,
#             #     "position": [1, 0, 1],
#             #     "scale": None,
#             #     "bounding_box": None,
#             #     "abilities": None,
#             #     "visual_only": False,
#             # },
#             # {
#             #     "type": "DatasetObject",
#             #     "fit_avg_dim_volume": True,
#             #     "name": "dishtowel",
#             #     "category": "dishtowel",
#             #     "model": "dtfspn",
#             #     "prim_type": PrimType.CLOTH,
#             #     "position": [1, 0, 1],
#             #     "scale": None,
#             #     "bounding_box": None,
#             #     "abilities": {"cloth": {}},
#             #     "visual_only": False,
#             # },
#     ],
#     "robots": [
#         {
#             "type": "Fetch",
#             "obs_modalities": 'all',
#             "controller_config": {
#                 "arm_0": {
#                     "name": "NullJointController",
#                     "motor_type": "position",
#                 },
#             },
#         }
#     ]
# }


config = {
    "scene": {
        "type": "InteractiveTraversableScene",
        "scene_model": "Rs_int",
    },
    "objects": [
        # {
        #     "type": "DatasetObject",
        #     "fit_avg_dim_volume": True,
        #     "name": "bagel_dough",
        #     "category": "bagel_dough",
        #     "model": "iuembm",
        #     "prim_type": PrimType.RIGID,
        #     "position": [1, 0, 1],
        #     "scale": None,
        #     "bounding_box": None,
        #     "abilities": None,
        #     "visual_only": False,
        # },
        # {
        #     "type": "DatasetObject",
        #     "fit_avg_dim_volume": True,
        #     "name": "dishtowel",
        #     "category": "dishtowel",
        #     "model": "dtfspn",
        #     "prim_type": PrimType.CLOTH,
        #     "position": [1, 0, 1],
        #     "scale": None,
        #     "bounding_box": None,
        #     "abilities": {"cloth": {}},
        #     "visual_only": False,
        # },
    ],
    "robots": [
        {
            "type": "Fetch",
            "obs_modalities": "all",
        }
    ],
}

# Make sure sim is stopped
if og.sim is not None:
    og.sim.stop()

gm.USE_GPU_DYNAMICS = True
gm.ENABLE_OBJECT_STATES = True

# wait for 10 seconds
# time.sleep(10)

env = og.Environment(configs=config)
robot = env.robots[0]

sensors = [s for s in robot.sensors.values() if isinstance(s, VisionSensor)]
assert len(sensors) > 0
vision_sensor = sensors[0]

env.reset()

# # For water:
# water_system = get_system("water")
# water_system.generate_particles(positions=[np.array([1, -0.1, 1.0]), np.array([1, -0.05, 1.0]), np.array([1, 0, 1.0]), np.array([1, 0.05, 1.0]), np.array([1, 0.1, 1.0])])
# # water_system.generate_particles(positions=[np.array([1, -0.1, 1.0])])
# # water_system.generate_particles(positions=[np.array([1, -0.05, 1.0])])
# # water_system.generate_particles(positions=[np.array([1, 0, 1.0])])
# # water_system.generate_particles(positions=[np.array([1, 0.05, 1.0])])
# # water_system.generate_particles(positions=[np.array([1, 0.1, 1.0])])
# og.sim.step()
# og.sim.step()
# og.sim.step()

# # For sesame seed:
# bagel_dough = og.sim.scene.object_registry("name", "bagel_dough")
# sesame_seed = get_system("sesame_seed")
# bagel_dough.states[Covered].set_value(sesame_seed, True)
# for _ in range(3):
#     og.sim.step()


# For diced__carrot:
# diced_carrot = get_system("diced__carrot")
# diced_carrot.generate_particles(positions=[np.array([1, -0.1, 1.0]), np.array([1, -0.05, 1.0]), np.array([1, 0, 1.0]), np.array([1, 0.05, 1.0]), np.array([1, 0.1, 1.0])])
# for _ in range(6):
#     og.sim.step()

# # For stain:
# stain = get_system("stain")
# breakfast_table = og.sim.scene.object_registry("name", "breakfast_table_skczfi_0")
# breakfast_table.states[Covered].set_value(stain, True)
# chair2 = og.sim.scene.object_registry("name", "straight_chair_amgwaw_2")
# chair2.states[Covered].set_value(stain, True)
# chair1 = og.sim.scene.object_registry("name", "straight_chair_amgwaw_1")
# chair1.states[Covered].set_value(stain, True)
# og.sim.step()

# For cloth:
# og.sim.step()
# og.sim.step()
# og.sim.step()

import time

print("GET PID NOW")
print("GET PID NOW")
print("GET PID NOW")
time.sleep(10)


for _ in range(2000):
    action_dict = dict()
    idx = 0
    for robot in env.robots:
        action_dim = robot.action_dim
        action_dict[robot.name] = np.random.uniform(-1, 1, action_dim)
        idx += action_dim
    for robot in env.robots:
        robot.apply_action(action_dict[robot.name])
    og.sim.step()
    all_observation = env.get_obs()
    # if len(og.sim._objects_to_initialize) > 0:
    #     og.sim.render()
    # super(type(og.sim), og.sim).step(render=True)
    # og.sim._non_physics_step()
    # all_observation = env.get_obs()


# all_observation, all_info = vision_sensor.get_obs()
# rgb = all_observation['rgb']
# seg_semantic = all_observation['seg_semantic']
# seg_instance = all_observation['seg_instance']
# seg_instance_id = all_observation['seg_instance_id']
# seg_semantic_info = all_info['seg_semantic']
# seg_instance_info = all_info['seg_instance']
# seg_instance_id_info = all_info['seg_instance_id']


# depth = all_observation['depth']
# depth_linear = all_observation['depth_linear']
# normal = all_observation['normal']

# flow = all_observation['flow']
# bbox_2d_tight = all_observation['bbox_2d_tight']
# bbox_2d_loose = all_observation['bbox_2d_loose']
# bbox_3d = all_observation['bbox_3d']

# all_observation = env.get_obs()


# # semantic_mapping = og.utils.usd_utils.SemanticsAPI.get_semantic_mapping()

# def visualize(observation, name):
#     import matplotlib.pyplot as plt
#     plt.imshow(observation)
#     plt.axis('off')
#     plt.savefig(f'{name}.png', bbox_inches='tight', pad_inches=0)
#     # plt.colorbar(label='Color value')
#     plt.show()

# from PIL import Image, ImageDraw

# camera_params = vision_sensor.camera_parameters

# from omnigibson.utils.vision_utils import colorize_bboxes_3d

# breakpoint()

# rgb = Image.fromarray(rgb)


# def world_to_image_pinhole(world_points, camera_params):
# 	# Project corners to image space (assumes pinhole camera model)
# 	proj_mat = camera_params["cameraProjection"].reshape(4, 4)
# 	view_mat = camera_params["cameraViewTransform"].reshape(4, 4)
# 	view_proj_mat = np.dot(view_mat, proj_mat)
# 	world_points_homo = np.pad(world_points, ((0, 0), (0, 1)), constant_values=1.0)
# 	tf_points = np.dot(world_points_homo, view_proj_mat)
# 	tf_points = tf_points / (tf_points[..., -1:])
# 	return 0.5 * (tf_points[..., :2] + 1)

# def get_bbox_3d_corners(extents):
#     """Return transformed points in the following order: [LDB, RDB, LUB, RUB, LDF, RDF, LUF, RUF]
#     where R=Right, L=Left, D=Down, U=Up, B=Back, F=Front and LR: x-axis, UD: y-axis, FB: z-axis.

#     Args:
#         extents (numpy.ndarray): A structured numpy array containing the fields: [`x_min`, `y_min`,
#             `x_max`, `y_max`, `transform`.

#     Returns:
#         (numpy.ndarray): Transformed corner coordinates with shape `(N, 8, 3)`.
#     """
#     occulusion_ratio = extents["occlusionRatio"]
#     truncated_extents = extents[np.logical_and(occulusion_ratio < 1.0, occulusion_ratio >= 0.0)]

#     rdb = [truncated_extents["x_max"], truncated_extents["y_min"], truncated_extents["z_min"]]
#     ldb = [truncated_extents["x_min"], truncated_extents["y_min"], truncated_extents["z_min"]]
#     lub = [truncated_extents["x_min"], truncated_extents["y_max"], truncated_extents["z_min"]]
#     rub = [truncated_extents["x_max"], truncated_extents["y_max"], truncated_extents["z_min"]]
#     ldf = [truncated_extents["x_min"], truncated_extents["y_min"], truncated_extents["z_max"]]
#     rdf = [truncated_extents["x_max"], truncated_extents["y_min"], truncated_extents["z_max"]]
#     luf = [truncated_extents["x_min"], truncated_extents["y_max"], truncated_extents["z_max"]]
#     ruf = [truncated_extents["x_max"], truncated_extents["y_max"], truncated_extents["z_max"]]
#     tfs = truncated_extents["transform"]

#     corners = np.stack((ldb, rdb, lub, rub, ldf, rdf, luf, ruf), 0)
#     corners_homo = np.pad(corners, ((0, 0), (0, 1), (0, 0)), constant_values=1.0)

#     return np.einsum("jki,ikl->ijl", corners_homo, tfs)[..., :3]

# corners_3d = get_bbox_3d_corners(bbox_3d)
# corners_3d = corners_3d.reshape(-1, 3)

# # Project to image space
# corners_2d = world_to_image_pinhole(corners_3d, camera_params)
# width, height = rgb.size
# corners_2d *= np.array([[width, height]])

# # Draw corners on image
# # draw_points(rgb, corners_2d)

# from omni.replicator.core import random_colours

# def draw_lines_and_points_for_boxes(img, all_image_points):
#     width, height = img.size
#     draw = ImageDraw.Draw(img)

#     # Define connections between the corners of the bounding box
#     connections = [
#         (0, 1), (1, 3), (3, 2), (2, 0),  # Front face
#         (4, 5), (5, 7), (7, 6), (6, 4),  # Back face
#         (0, 4), (1, 5), (2, 6), (3, 7)   # Side edges connecting front and back faces
#     ]

#     # Calculate the number of bounding boxes
#     num_boxes = len(all_image_points) // 8

#     # Generate random colors for each bounding box
#     box_colors = random_colours(num_boxes, enable_random=True, num_channels=3)

#     # Ensure colors are in the correct format for drawing (255 scale)
#     box_colors = [(int(r), int(g), int(b)) for r, g, b in box_colors]

#     # Iterate over each set of 8 points (each bounding box)
#     for i in range(0, len(all_image_points), 8):
#         image_points = all_image_points[i:i+8]
#         image_points[:, 1] = height - image_points[:, 1]  # Flip Y-axis to match image coordinates

#         # Use a distinct color for each bounding box
#         line_color = box_colors[i // 8]

#         # Draw lines for each connection
#         for start, end in connections:
#             draw.line((image_points[start][0], image_points[start][1],
#                        image_points[end][0], image_points[end][1]),
#                       fill=line_color, width=2)


# # Now, use this function to draw all bounding boxes
# draw_lines_and_points_for_boxes(rgb, corners_2d)


# lidar = all_observation[0]['robot0']['robot0:laser_link:Lidar:0']['scan']
# occupancy_grid = all_observation[0]['robot0']['robot0:laser_link:Lidar:0']['occupancy_grid']

# import numpy as np
# import matplotlib.pyplot as plt

# def visualize_lidar_data(lidar_data):
#     """
#     Visualizes 2D LiDAR data.

#     Args:
#         lidar_data (numpy.ndarray): 2D LiDAR data with shape (360, 1),
#                                     containing distance measurements for each degree of a 360-degree sweep.
#     """
#     # Angles for each measurement (in radians)
#     angles = np.radians(np.arange(0, 360, 1))  # Convert degrees to radians
#     distances = lidar_data.flatten()  # Flatten the (360, 1) array to (360,)

#     # Convert polar coordinates (angle, distance) to Cartesian coordinates (x, y)
#     x = distances * np.cos(angles)
#     y = distances * np.sin(angles)

#     plt.figure(figsize=(8, 8))
#     plt.plot(x, y, 'o', markersize=2)
#     plt.axis('equal')  # Ensure equal scaling for both axes to maintain aspect ratio
#     plt.grid(True)
#     # plt.axis('off')
#     plt.savefig('lidar.png', bbox_inches='tight', pad_inches=0)
#     plt.show()


# def visualize_occupancy_grid(occupancy_grid):
#     """
#     Visualizes a 2D occupancy grid using a heatmap.

#     Args:
#         occupancy_grid (numpy.ndarray): 2D occupancy grid with shape (128, 128, 1),
#                                         where each cell's value indicates occupancy.
#     """
#     # Remove the last dimension if it's singular to fit imshow requirements
#     if occupancy_grid.shape[-1] == 1:
#         occupancy_grid = occupancy_grid.squeeze()

#     plt.figure(figsize=(8, 8))
#     # Display the grid as a heatmap, you can choose different colormaps as needed, e.g., 'gray', 'viridis'
#     plt.imshow(occupancy_grid, cmap='gray', interpolation='nearest')
#     plt.axis('off')  # Turn off axis labels and ticks for a cleaner visualization
#     plt.savefig('occupancy_grid.png', bbox_inches='tight', pad_inches=0)
#     plt.show()

# breakpoint()

# for _ in range(30):
#     action = np.random.uniform(-1, 1, robot.action_dim)
#     action[robot.base_control_idx] = 0
#     action[robot.camera_control_idx] = 0
#     for _ in range(10):
#         env.step(action)
#     all_observation, all_info = vision_sensor.get_obs()
#     flow = all_observation['flow']
#     breakpoint()


# # laptop = og.sim.scene.object_registry('prim_path', '/World/laptop_nvulcs_0')
# # base_link = laptop.links['base_link']

# # fov_objs = robot.states[ObjectsInFOVOfRobot].get_value()
# breakpoint()

"""
_RAW_SENSOR_TYPES = dict(
    rgb="rgb",
    depth="distance_to_camera",
    depth_linear="distance_to_image_plane",
    normal="normals",
    seg_semantic="semantic_segmentation",
    seg_instance="instance_segmentation",
    seg_instance_id="instance_id_segmentation",
    flow="motion_vectors",
    bbox_2d_tight="bounding_box_2d_tight",
    bbox_2d_loose="bounding_box_2d_loose",
    bbox_3d="bounding_box_3d",
    camera_params="camera_params",
)

annotators = None
resolution = (1024, 1024)
render_product = lazy.omni.replicator.core.create.render_product('/World/robot0/head_camera_link', resolution)

breakpoint()

camera = lazy.omni.replicator.core.create.camera(position=(10, 0, 0), look_at=(0, 0, 0))
render_product = lazy.omni.replicator.core.create.render_product(camera, (1024, 1024))
rgb = lazy.omni.replicator.core.AnnotatorRegistry.get_annotator("rgb")
bbox_2d_tight = lazy.omni.replicator.core.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
instance_seg = lazy.omni.replicator.core.AnnotatorRegistry.get_annotator("instance_segmentation")
rgb.attach(render_product)
bbox_2d_tight.attach(render_product)
instance_seg.attach(render_product)

og.app.update()

camera_data = {
    "rgb": rgb.get_data(device="cuda"),
    "boundingBox2DTight": bbox_2d_tight.get_data(device="cuda"),
    "instanceSegmentation": instance_seg.get_data(device="cuda"),
}

image = camera_data['rgb']

# visualize the warp image


"""
og.sim.clear()
