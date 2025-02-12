import math

import torch as th
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

    def create_robot_config(name, position):
        return {
            "name": name,
            "type": "Fetch",
            "obs_modalities": ["rgb", "seg_instance"],
            "position": position,
            "orientation": T.euler2quat(th.tensor([0, 0, -math.pi / 2], dtype=th.float32)),
            "controller_config": {
                "arm_0": {
                    "name": "NullJointController",
                    "motor_type": "position",
                },
            },
        }

    robot_names = ["fetch_1", "fetch_2", "fetch_3"]
    robot_positions = [[0, 0.8, 0], [1, 0.8, 0], [2, 0.8, 0]]

    config = {
        "scene": {
            "type": "Scene",
        },
        "robots": [create_robot_config(name, position) for name, position in zip(robot_names, robot_positions)],
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
    scene = env.scene

    breakfast_table = scene.object_registry("name", "breakfast_table")
    bowl = scene.object_registry("name", "bowl")
    place_obj_on_floor_plane(breakfast_table)
    bowl.set_position_orientation(position=[0.0, -0.8, 0.1], orientation=[0, 0, 0, 1])

    # Test single robot scene graph
    scene_graph_builder_single = SceneGraphBuilder(
        robot_names=robot_names[:1], egocentric=False, full_obs=True, only_true=True, merge_parallel_edges=True
    )
    scene_graph_builder_single.start(scene)
    for _ in range(3):
        og.sim.step()
        scene_graph_builder_single.step(scene)

    assert isinstance(
        visualize_scene_graph(
            scene, scene_graph_builder_single.get_scene_graph(), show_window=False, cartesian_positioning=True
        ),
        th.Tensor,
    )

    # Test multi robot scene graph
    scene_graph_builder_multi = SceneGraphBuilder(
        robot_names=robot_names, egocentric=False, full_obs=True, only_true=True, merge_parallel_edges=True
    )
    scene_graph_builder_multi.start(scene)
    for _ in range(3):
        og.sim.step()
        scene_graph_builder_multi.step(scene)

    assert isinstance(
        visualize_scene_graph(
            scene, scene_graph_builder_multi.get_scene_graph(), show_window=False, cartesian_positioning=True
        ),
        th.Tensor,
    )
