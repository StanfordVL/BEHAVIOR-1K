import torch as th
from utils import SYSTEM_EXAMPLES, place_obj_on_floor_plane

import omnigibson as og
from omnigibson.systems import MacroParticleSystem, MicroParticleSystem
from omnigibson.utils.constants import semantic_class_id_to_name


def test_segmentation_modalities(env, breakfast_table, dishtowel):
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
    expected_semantic_names = {"floors", "breakfast_table", "dishtowel", *SYSTEM_EXAMPLES.keys()}
    assert set(seg_semantic_info.values()) == expected_semantic_names

    seg_instance = all_observation["seg_instance"]
    seg_instance_info = all_info["seg_instance"]
    assert set(int(x.item()) for x in th.unique(seg_instance)) == set(seg_instance_info.keys())
    expected_instance_names = {"groundPlane", "breakfast_table", "dishtowel", *SYSTEM_EXAMPLES.keys()}
    assert set(seg_instance_info.values()) == expected_instance_names

    seg_instance_id = all_observation["seg_instance_id"]
    seg_instance_id_info = all_info["seg_instance_id"]
    assert set(int(x.item()) for x in th.unique(seg_instance_id)) == set(seg_instance_id_info.keys())
    expected_instance_id_paths = {
        "/World/ground_plane/geom",
        *[visual_mesh.prim_path for visual_mesh in breakfast_table.root_link.visual_meshes.values()],
        *[visual_mesh.prim_path for visual_mesh in dishtowel.root_link.visual_meshes.values()],
    }
    for system in systems:
        if isinstance(system, MicroParticleSystem):
            for instancer in system.particle_instancers.values():
                instancer_scope_path = instancer.prim_path.rsplit("/instancer", 1)[0]
                expected_instance_id_paths.update(
                    f"{instancer_scope_path}/prototype{int(prototype_id.item())}"
                    for prototype_id in th.unique(instancer.particle_prototype_ids)
                )
        elif isinstance(system, MacroParticleSystem):
            expected_instance_id_paths.update(particle.prim_path for particle in system.particles.values())
    assert set(seg_instance_id_info.values()) == expected_instance_id_paths

    for system in systems:
        env.scene.clear_system(system.name)


def test_bbox_modalities(env, breakfast_table, dishtowel):
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
