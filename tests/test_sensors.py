import math

import torch as th
from utils import SYSTEM_EXAMPLES, og_test, place_obj_on_floor_plane

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.sensors import VisionSensor
from omnigibson.utils.constants import semantic_class_id_to_name


@og_test
def test_segmentation_modalities(env):
    breakfast_table = env.scene.object_registry("name", "breakfast_table")
    dishtowel = env.scene.object_registry("name", "dishtowel")
    robot = env.scene.robots[0]
    place_obj_on_floor_plane(breakfast_table)
    dishtowel.set_position_orientation(position=[-0.4, 0.0, 0.55], orientation=[0, 0, 0, 1])
    robot.set_position_orientation(
        position=[0.0, 0.8, 0.0], orientation=T.euler2quat(th.tensor([0, 0, -math.pi / 2], dtype=th.float32))
    )
    robot.reset()

    modalities_required = ["seg_semantic", "seg_instance", "seg_instance_id"]
    for modality in modalities_required:
        robot.add_obs_modality(modality)

    systems = [env.scene.get_system(system_name) for system_name in SYSTEM_EXAMPLES.keys()]
    for i, system in enumerate(systems):
        # Sample two particles for each system
        pos = th.tensor([-0.2 + i * 0.2, 0, 0.55])
        if env.scene.is_physical_particle_system(system_name=system.name):
            system.generate_particles(positions=[pos.tolist(), (pos + th.tensor([0.1, 0.0, 0.0])).tolist()])
        else:
            if system.get_group_name(breakfast_table) not in system.groups:
                system.create_attachment_group(breakfast_table)
            system.generate_group_particles(
                group=system.get_group_name(breakfast_table),
                positions=[pos, pos + th.tensor([0.1, 0.0, 0.0])],
                link_prim_paths=[breakfast_table.root_link.prim_path],
            )

    og.sim.step()
    for _ in range(3):
        og.sim.render()

    sensors = [s for s in robot.sensors.values() if isinstance(s, VisionSensor)]
    assert len(sensors) > 0
    vision_sensor = sensors[0]
    all_observation, all_info = vision_sensor.get_obs()

    seg_semantic = all_observation["seg_semantic"]
    seg_semantic_info = all_info["seg_semantic"]
    assert set(int(x.item()) for x in th.unique(seg_semantic)) == set(seg_semantic_info.keys())
    expected_dict = {
        335706086: "diced__apple",
        825831922: "floors",
        884110082: "stain",
        1949122937: "breakfast_table",
        2814990211: "agent",
        3051938632: "white_rice",
        3330677804: "water",
        4207839377: "dishtowel",
    }
    assert set(seg_semantic_info.values()) == set(expected_dict.values())

    seg_instance = all_observation["seg_instance"]
    seg_instance_info = all_info["seg_instance"]
    assert set(int(x.item()) for x in th.unique(seg_instance)) == set(seg_instance_info.keys())
    expected_dict = {
        1: "unlabelled",
        2: "robot0",
        3: "groundPlane",
        4: "dishtowel",
        5: "breakfast_table",
        6: "stain",
        # 7: "water",
        # 8: "white_rice",
        9: "diced__apple",
    }
    assert set(seg_instance_info.values()) == set(expected_dict.values())

    seg_instance_id = all_observation["seg_instance_id"]
    seg_instance_id_info = all_info["seg_instance_id"]
    assert set(int(x.item()) for x in th.unique(seg_instance_id)) == set(seg_instance_id_info.keys())
    expected_dict = {
        3: "/World/robot0/gripper_link/visuals",
        4: "/World/robot0/wrist_roll_link/visuals",
        5: "/World/robot0/forearm_roll_link/visuals",
        6: "/World/robot0/wrist_flex_link/visuals",
        8: "/World/groundPlane/geom",
        9: "/World/dishtowel/base_link_cloth",
        10: "/World/robot0/r_gripper_finger_link/visuals",
        11: "/World/robot0/l_gripper_finger_link/visuals",
        12: "/World/breakfast_table/base_link/visuals",
        13: "stain",
        14: "white_rice",
        15: "diced__apple",
        16: "water",
    }
    # Temporarily disable this test because og_assets are outdated on CI machines
    # assert set(seg_instance_id_info.values()) == set(expected_dict.values())

    for system in systems:
        env.scene.clear_system(system.name)


@og_test
def test_bbox_modalities(env):
    breakfast_table = env.scene.object_registry("name", "breakfast_table")
    dishtowel = env.scene.object_registry("name", "dishtowel")
    robot = env.scene.robots[0]
    place_obj_on_floor_plane(breakfast_table)
    dishtowel.set_position_orientation(position=[-0.4, 0.0, 0.55], orientation=[0, 0, 0, 1])
    robot.set_position_orientation(
        position=[0, 0.8, 0.0], orientation=T.euler2quat(th.tensor([0, 0, -math.pi / 2], dtype=th.float32))
    )
    robot.reset()

    modalities_required = ["bbox_2d_tight", "bbox_2d_loose", "bbox_3d"]
    for modality in modalities_required:
        robot.add_obs_modality(modality)

    og.sim.step()
    for _ in range(3):
        og.sim.render()

    sensors = [s for s in robot.sensors.values() if isinstance(s, VisionSensor)]
    assert len(sensors) > 0
    vision_sensor = sensors[0]
    all_observation, all_info = vision_sensor.get_obs()

    bbox_2d_tight = all_observation["bbox_2d_tight"]
    bbox_2d_loose = all_observation["bbox_2d_loose"]
    bbox_3d = all_observation["bbox_3d"]

    assert len(bbox_2d_tight) == 4
    assert len(bbox_2d_loose) == 4
    assert len(bbox_3d) == 3

    bbox_2d_expected_objs = set(["floors", "agent", "breakfast_table", "dishtowel"])
    bbox_3d_expected_objs = set(["agent", "breakfast_table", "dishtowel"])

    bbox_2d_objs = set([semantic_class_id_to_name()[bbox[0]] for bbox in bbox_2d_tight])
    bbox_3d_objs = set([semantic_class_id_to_name()[bbox[0]] for bbox in bbox_3d])

    assert bbox_2d_objs == bbox_2d_expected_objs
    assert bbox_3d_objs == bbox_3d_expected_objs
