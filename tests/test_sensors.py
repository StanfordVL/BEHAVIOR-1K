from omnigibson.systems import get_system, is_physical_particle_system, is_visual_particle_system
import omnigibson.utils.transform_utils as T
import omnigibson as og
from omnigibson.sensors import VisionSensor

from utils import og_test, place_obj_on_floor_plane, SYSTEM_EXAMPLES

import pytest
import numpy as np

@og_test
def test_seg():
    breakfast_table = og.sim.scene.object_registry("name", "breakfast_table")
    dishtowel = og.sim.scene.object_registry("name", "dishtowel")
    robot = og.sim.scene.robots[0]
    place_obj_on_floor_plane(breakfast_table)
    dishtowel.set_position_orientation([-0.4, 0.0, 0.55], [0, 0, 0, 1])
    robot.set_position_orientation([0, 0.8, 0.0], T.euler2quat([0, 0, -np.pi/2]))
    robot.reset()

    systems = [get_system(system_name) for system_name, system_class in SYSTEM_EXAMPLES.items()]
    for i, system in enumerate(systems):
        # Sample two particles for each system
        pos = np.array([-0.2 + i * 0.2, 0, 0.55])
        if is_physical_particle_system(system_name=system.name):
            system.generate_particles(positions=[pos, pos + np.array([0.1, 0.0, 0.0])])
        else:
            if system.get_group_name(breakfast_table) not in system.groups:
                system.create_attachment_group(breakfast_table)
            system.generate_group_particles(
                group=system.get_group_name(breakfast_table),
                positions=np.array([pos, pos + np.array([0.1, 0.0, 0.0])]),
                link_prim_paths=[breakfast_table.root_link.prim_path],
            )

    og.sim.step()
    og.sim.render()

    sensors = [s for s in robot.sensors.values() if isinstance(s, VisionSensor)]
    assert len(sensors) > 0
    vision_sensor = sensors[0]
    all_observation, all_info = vision_sensor.get_obs()

    seg_semantic = all_observation['seg_semantic']
    seg_semantic_info = all_info['seg_semantic']
    assert set(np.unique(seg_semantic)) == set(seg_semantic_info.keys())
    expected_dict = {
        335706086: 'diced__apple',
        825831922: 'floors',
        884110082: 'stain',
        1949122937: 'breakfast_table',
        2814990211: 'agent',
        3051938632: 'white_rice',
        3330677804: 'water',
        4207839377: 'dishtowel'
    }
    assert set(seg_semantic_info.values()) == set(expected_dict.values())

    seg_instance = all_observation['seg_instance']
    seg_instance_info = all_info['seg_instance']
    assert set(np.unique(seg_instance)) == set(seg_instance_info.keys())
    expected_dict = {
        2: 'robot0',
        3: 'groundPlane',
        4: 'dishtowel',
        5: 'breakfast_table',
        6: 'stain',
        7: 'water',
        8: 'white_rice',
        9: 'diced__apple'
    }
    assert set(seg_instance_info.values()) == set(expected_dict.values())

    seg_instance_id = all_observation['seg_instance_id']
    seg_instance_id_info = all_info['seg_instance_id']
    assert set(np.unique(seg_instance_id)) == set(seg_instance_id_info.keys())
    expected_dict = {
        3: '/World/robot0/gripper_link/visuals',
        4: '/World/robot0/wrist_roll_link/visuals',
        5: '/World/robot0/forearm_roll_link/visuals',
        6: '/World/robot0/wrist_flex_link/visuals',
        8: '/World/groundPlane/geom',
        9: '/World/dishtowel/base_link_cloth',
        10: '/World/robot0/r_gripper_finger_link/visuals',
        11: '/World/robot0/l_gripper_finger_link/visuals',
        12: '/World/breakfast_table/base_link/visuals',
        13: 'stain',
        14: 'white_rice',
        15: 'diced__apple',
        16: 'water'
    }
    # Temporarily disable this test because og_assets are outdated on CI machines
    # assert set(seg_instance_id_info.values()) == set(expected_dict.values())

def test_clear_sim():
    og.sim.clear()
