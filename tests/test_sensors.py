import torch as th
from utils import SYSTEM_EXAMPLES, og_test, place_obj_on_floor_plane

import omnigibson as og
from omnigibson.utils.constants import semantic_class_id_to_name


@og_test
def test_segmentation_modalities(env):
    breakfast_table = env.scene.object_registry("name", "breakfast_table")
    dishtowel = env.scene.object_registry("name", "dishtowel")
    place_obj_on_floor_plane(breakfast_table)
    dishtowel.set_position_orientation(position=[-0.4, 0.0, 0.55], orientation=[0, 0, 0, 1])

    og.sim.viewer_camera.set_position_orientation(position=[-0.0017, -0.1072, 1.4969], orientation=[0.0, 0.0, 0.0, 1.0])

    modalities_required = ["seg_semantic", "seg_instance", "seg_instance_id"]
    for modality in modalities_required:
        og.sim.viewer_camera.add_modality(modality)

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
                link_prim_paths=[breakfast_table.root_link.prim_path] * 2,
            )

    og.sim.step()
    for _ in range(3):
        og.sim.render()

    all_observation, all_info = og.sim.viewer_camera.get_obs()

    seg_semantic = all_observation["seg_semantic"]
    seg_semantic_info = all_info["seg_semantic"]
    assert set(int(x.item()) for x in th.unique(seg_semantic)) == set(seg_semantic_info.keys())
    expected_dict = {
        335706086: "diced__apple",
        825831922: "floors",
        884110082: "stain",
        1949122937: "breakfast_table",
        3051938632: "white_rice",
        3330677804: "water",
        4207839377: "dishtowel",
    }
    assert set(seg_semantic_info.values()) == set(expected_dict.values())

    seg_instance = all_observation["seg_instance"]
    seg_instance_info = all_info["seg_instance"]
    assert set(int(x.item()) for x in th.unique(seg_instance)) == set(seg_instance_info.keys())
    expected_dict = {
        2: "groundPlane",
        3: "water",
        4: "diced__apple",
        5: "stain",
        6: "white_rice",
        7: "breakfast_table",
        8: "dishtowel",
    }
    assert set(seg_instance_info.values()) == set(expected_dict.values())

    seg_instance_id = all_observation["seg_instance_id"]
    seg_instance_id_info = all_info["seg_instance_id"]
    assert set(int(x.item()) for x in th.unique(seg_instance_id)) == set(seg_instance_id_info.keys())
    expected_dict = {
        1: "/World/ground_plane/geom",
        2: "/World/scene_0/breakfast_table/base_link/visuals",
        3: "/World/scene_0/dishtowel/base_link_cloth",
        4: "/World/scene_0/water/waterInstancer0/prototype0",
        5: "/World/scene_0/white_rice/white_riceInstancer0/prototype0",
        6: "/World/scene_0/diced__apple/particles/diced__appleParticle1",
        7: "/World/scene_0/breakfast_table/base_link/stainParticle1",
        8: "/World/scene_0/breakfast_table/base_link/stainParticle0",
        9: "/World/scene_0/diced__apple/particles/diced__appleParticle0",
    }
    assert set(seg_instance_id_info.values()) == set(expected_dict.values())

    for system in systems:
        env.scene.clear_system(system.name)


@og_test
def test_bbox_modalities(env):
    breakfast_table = env.scene.object_registry("name", "breakfast_table")
    dishtowel = env.scene.object_registry("name", "dishtowel")
    place_obj_on_floor_plane(breakfast_table)
    dishtowel.set_position_orientation(position=[-0.4, 0.0, 0.55], orientation=[0, 0, 0, 1])

    og.sim.viewer_camera.set_position_orientation(position=[-0.0017, -0.1072, 1.4969], orientation=[0.0, 0.0, 0.0, 1.0])

    modalities_required = ["bbox_2d_tight", "bbox_2d_loose", "bbox_3d"]
    for modality in modalities_required:
        og.sim.viewer_camera.add_modality(modality)

    og.sim.step()
    for _ in range(3):
        og.sim.render()

    all_observation, all_info = og.sim.viewer_camera.get_obs()

    bbox_2d_tight = all_observation["bbox_2d_tight"]
    bbox_2d_loose = all_observation["bbox_2d_loose"]
    bbox_3d = all_observation["bbox_3d"]

    assert len(bbox_2d_tight) == 3
    assert len(bbox_2d_loose) == 3
    assert len(bbox_3d) == 2

    bbox_2d_expected_objs = set(["floors", "breakfast_table", "dishtowel"])
    bbox_3d_expected_objs = set(["breakfast_table", "dishtowel"])

    bbox_2d_objs = set([semantic_class_id_to_name()[bbox[0]] for bbox in bbox_2d_tight])
    bbox_3d_objs = set([semantic_class_id_to_name()[bbox[0]] for bbox in bbox_3d])

    assert bbox_2d_objs == bbox_2d_expected_objs
    assert bbox_3d_objs == bbox_3d_expected_objs
