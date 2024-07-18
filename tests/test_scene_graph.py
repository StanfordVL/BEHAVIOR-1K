import numpy as np
import pytest
from utils import place_obj_on_floor_plane

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.scene_graphs.graph_builder import SceneGraphBuilder, visualize_scene_graph
from omnigibson.utils.constants import PrimType


def test_scene_graph():

    if og.sim is None:
        # Set global flags
        gm.ENABLE_OBJECT_STATES = True
    else:
        # Make sure sim is stopped
        og.sim.stop()

    # Define the environment configuration
    config = {
        "scene": {
            "type": "Scene",
        },
        "robots": [
            {
                "type": "Fetch",
                "obs_modalities": "all",
                "position": [0, 0, 0],
                "orientation": [0, 0, 0, 1],
                "controller_config": {
                    "arm_0": {
                        "name": "NullJointController",
                        "motor_type": "position",
                    },
                },
            }
        ],
        "objects": [
            {
                "type": "DatasetObject",
                "fit_avg_dim_volume": True,
                "name": "breakfast_table",
                "category": "breakfast_table",
                "model": "skczfi",
                "prim_type": PrimType.RIGID,
                "position": [150, 150, 150],
                "scale": None,
                "bounding_box": None,
                "abilities": None,
                "visual_only": False,
            },
            {
                "type": "DatasetObject",
                "fit_avg_dim_volume": True,
                "name": "bowl",
                "category": "bowl",
                "model": "ajzltc",
                "prim_type": PrimType.RIGID,
                "position": [150, 150, 150],
                "scale": None,
                "bounding_box": None,
                "abilities": None,
                "visual_only": False,
            },
        ],
    }

    env = og.Environment(configs=config)

    breakfast_table = og.sim.scene.object_registry("name", "breakfast_table")
    bowl = og.sim.scene.object_registry("name", "bowl")
    robot = og.sim.scene.robots[0]
    place_obj_on_floor_plane(breakfast_table)
    bowl.set_position_orientation([0.0, -0.8, 0.1], [0, 0, 0, 1])
    robot.set_position_orientation([0, 0.8, 0.0], T.euler2quat([0, 0, -np.pi / 2]))
    robot.reset()

    scene_graph_builder = SceneGraphBuilder(
        robot_name=None, egocentric=False, full_obs=True, only_true=True, merge_parallel_edges=True
    )
    scene_graph_builder.start(og.sim.scene)
    for _ in range(3):
        og.sim.step()
        scene_graph_builder.step(og.sim.scene)

    assert visualize_scene_graph(
        og.sim.scene, scene_graph_builder.get_scene_graph(), show_window=False, cartesian_positioning=True
    )
