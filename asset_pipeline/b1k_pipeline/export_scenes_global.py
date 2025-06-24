import io
import json
import os
import sys
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from fs.osfs import OSFS
from scipy.spatial.transform import Rotation as R
from concurrent import futures
import tqdm
import traceback

from b1k_pipeline.mesh_tree import build_mesh_tree
import b1k_pipeline.utils

ALLOWED_PART_TAGS = {
    "subpart",
    "extrapart",
    "connectedpart",
}

OBJECT_CORRECTIONS = {
    # "office_large": {
    #     "bottom_cabinet_bamfsz_52": ([0.0, -0.02, 0.0], [0.0, 0.0, 0.0], None, "world"),
    #     "bottom_cabinet_bamfsz_53": ([0.0, -0.02, 0.], [0.0, 0.0, 0.0], None, "world"),
    #     "bottom_cabinet_bamfsz_56": ([0.02, 0.0, 0.0], [0.0, 0.0, 0.0], None, "world"),
    #     "bottom_cabinet_bamfsz_57": ([0.02, 0.0, 0.0], [0.0, 0.0, 0.0], None, "world"),
    #     "bottom_cabinet_bamfsz_58": ([-0.02, 0.0, 0.0], [0.0, 0.0, 0.0], None, "world"),
    #     "bottom_cabinet_bamfsz_59": ([-0.02, 0.0, 0.0], [0.0, 0.0, 0.0], None, "world"),
    #     "fridge_juwaoh_0": ([0.0, 0.0, 0.0], [-np.pi / 2.0 , 0.0, 0.0], None, "world"),
    #     "fridge_juwaoh_1": ([0.0, 0.0, 0.0], [-np.pi / 2.0, 0.0, 0.0], None, "world"),
    # },
    # "Beechwood_0_garden": {
    #     "coffee_table_qplrhv_0": ([0.0, 0.0, 0.03], [0.0, 0.0, 0.0], None, "world"),
    #     "coffee_table_wwgaym_0": ([0.0, 0.0, 0.03], [0.0, 0.0, 0.0], None, "world"),
    #     "coffee_table_hbgcej_0": ([0.0, 0.0, 0.03], [0.0, 0.0, 0.0], None, "world"),
    #     "coffee_table_rkdwxe_1": ([0.0, 0.0, 0.03], [0.0, 0.0, 0.0], None, "world"),
    #     "coffee_table_rkdwxe_0": ([0.0, 0.0, 0.03], [0.0, 0.0, 0.0], None, "world"),
    #     "sofa_rqipbs_0": ([0.0, 0.0, 0.03], [0.0, 0.0, 0.0], None, "world"),
    #     "coffee_table_mjyitq_0": ([0.0, 0.0, 0.03], [0.0, 0.0, 0.0], None, "world"),
    #     "armchair_nibzys_0": ([0.0, 0.0, 0.03], [0.0, 0.0, 0.0], None, "world"),
    #     "sofa_ahgkci_0": ([0.0, 0.0, 0.03], [0.0, 0.0, 0.0], True, "world"),
    #     "sofa_tlsxpc_0": ([0.0, 0.0, 0.03], [0.0, 0.0, 0.0], True, "world"),
    #     "coffee_table_lpwbgm_0": ([0.0, 0.0, 0.03], [0.0, 0.0, 0.0], True, "world"),
    #     "pot_plant_ibvdwh_0": ([0.03, 0.0, 0.03], [0.0, 0.0, 0.0], None, "world"),
    #     "pot_plant_ibvdwh_1": ([0.03, 0.0, 0.03], [0.0, 0.0, 0.0], None, "world"),
    #     "pot_plant_ibvdwh_2": ([0.03, 0.0, 0.03], [0.0, 0.0, 0.0], None, "world"),
    # },
    # "house_single_floor": {
    #     "stove_mjvqii_0": ([0.0, 0.0, 0.03], [0.0, 0.0, 0.0], None, "world"),
    #     "sink_awvzkn_0": ([0.0, 0.0, 0.03], [0.0, 0.0, 0.0], None, "world"),
    #     "countertop_rkgjer_0": ([0.0, 0.0, 0.03], [0.0, 0.0, 0.0], None, "world"),
    #     "countertop_gjeoer_0": ([0.0, 0.0, 0.03], [0.0, 0.0, 0.0], None, "world"),
    #     "fridge_dszchb_0": ([0.0, 0.0, 0.0], [0.0, 0.0, np.pi], None, "world"),
    #     "clothes_dryer_xsuyua_0": ([0.0, 0.0, 0.0], [0.0, 0.0, -np.pi / 2.0], None, "world"),
    #     "water_glass_bbpraa_0": ([0.0, 0.0, 0.03], [0.0, 0.0, 0.0], None, "world"),
    #     "sink_vekaaf_0": ([0.0, 0.0, 0.03], [0.0, 0.0, 0.0], None, "world"),
    #     "countertop_ikwqer_0": ([0.0, 0.0, 0.03], [0.0, 0.0, 0.0], None, "world"),
    #     "countertop_sjxber_0": ([0.0, 0.0, 0.03], [0.0, 0.0, 0.0], None, "world"),
    # },
    # "Wainscott_0_garden": {
    #     "garden_umbrella_bkfuvq_0": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], True, "world"),
    #     "garden_umbrella_bkfuvq_1": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], True, "world"),
    #     "pedestal_table_ifmhpn_0": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], True, "world"),
    #     "pedestal_table_ifmhpn_1": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], True, "world"),
    # },
    # "office_cubicles_right": {
    #     "conference_table_fpkmyw_0": ([0.0, 0.0, 0.0], [np.pi, 0.0, 0.0], None, "world"),
    #     "swivel_chair_ucawqa_0": ([0.0, 0.0, 0.0], [-np.pi / 2, 0.0, -np.pi / 2], None, "local"),
    #     "swivel_chair_ucawqa_1": ([0.0, 0.0, 0.0], [-np.pi / 2, 0.0, -np.pi / 2], None, "local"),
    #     "swivel_chair_ucawqa_2": ([0.0, 0.0, 0.0], [-np.pi / 2, 0.0, -np.pi / 2], None, "local"),
    #     "swivel_chair_ucawqa_3": ([0.0, 0.0, 0.0], [-np.pi / 2, 0.0, -np.pi / 2], None, "local"),
    #     "swivel_chair_ucawqa_4": ([0.0, 0.0, 0.0], [-np.pi / 2, 0.0, -np.pi / 2], None, "local"),
    #     "swivel_chair_ucawqa_5": ([0.0, 0.0, 0.0], [-np.pi / 2, 0.0, -np.pi / 2], None, "local"),
    #     "swivel_chair_ucawqa_6": ([0.0, 0.0, 0.0], [-np.pi / 2, 0.0, -np.pi / 2], None, "local"),
    #     "swivel_chair_ucawqa_7": ([0.0, 0.0, 0.0], [-np.pi / 2, 0.0, -np.pi / 2], None, "local"),
    #     "swivel_chair_ucawqa_8": ([0.0, 0.0, 0.0], [-np.pi / 2, 0.0, -np.pi / 2], None, "local"),
    #     "swivel_chair_ucawqa_9": ([0.0, 0.0, 0.0], [-np.pi / 2, 0.0, -np.pi / 2], None, "local"),
    # },
    # "office_vendor_machine": {
    #     "bottom_cabinet_bamfsz_0": ([0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2], None, "local"),
    #     "bottom_cabinet_bamfsz_1": ([0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2], None, "local"),
    #     "bottom_cabinet_bamfsz_2": ([0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2], None, "local"),
    #     "bottom_cabinet_bamfsz_3": ([0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2], None, "local"),
    #     "bottom_cabinet_bamfsz_4": ([0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2], None, "local"),
    #     "bottom_cabinet_bamfsz_5": ([0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2], None, "local"),
    #     "bottom_cabinet_bamfsz_6": ([0.0, -0.2, 0.0], [0.0, 0.0, np.pi / 2], None, "local"),
    #     "bottom_cabinet_bamfsz_7": ([0.0, -0.2, 0.0], [0.0, 0.0, np.pi / 2], None, "local"),
    #     "bottom_cabinet_bamfsz_8": ([0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2], None, "local"),
    #     "bottom_cabinet_bamfsz_9": ([0.0, -0.2, 0.0], [0.0, 0.0, np.pi / 2], None, "local"),
    #     "bottom_cabinet_bamfsz_10": ([0.0, -0.2, 0.0], [0.0, 0.0, np.pi / 2], None, "local"),
    # },
    # "restaurant_diner": {
    #     "straight_chair_nntxvr_0":  ([0.33, 0.0, 0.0], [0.0, 0.0, 0.0], None, "world"),
    #     "straight_chair_nntxvr_1":  ([0.33, 0.0, 0.0], [0.0, 0.0, 0.0], None, "world"),
    #     "bottom_cabinet_olgoza_0":  ([0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2], None, "world"),
    #     "fridge_juwaoh_0":  ([0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2], None, "world"),
    # },
}


def process_target(target, scenes_dir):
    scene_name = os.path.split(target)[-1]
    pipeline_fs = b1k_pipeline.utils.PipelineFS()

    scene_tree_root = ET.Element("robot")
    scene_tree_root.attrib = {"name": "igibson_scene"}
    world_link = ET.SubElement(scene_tree_root, "link")
    world_link.attrib = {"name": "world"}

    # Load info about the scene parts.
    target_output_fs = pipeline_fs.target_output(target)
    scene_parts = {scene_name: None}
    with target_output_fs.open("object_list.json") as f:
        room_object_list = json.load(f)
    for partial_scene_name, (portal_pos, portal_quat, _) in room_object_list[
        "outgoing_portals"
    ].items():
        transform = np.eye(4)
        transform[:3, :3] = R.from_quat(portal_quat).as_matrix()
        transform[:3, 3] = np.array(portal_pos) / 1000.0
        scene_parts[partial_scene_name] = transform

    # Get the mesh tree of each part.
    for partial_scene_name, portal_pose_in_parent in scene_parts.items():
        partial_scene_target_name = "scenes/" + partial_scene_name
        partial_scene_output_fs = pipeline_fs.target_output(partial_scene_target_name)

        # Compute the incoming transform
        rel_transform = np.eye(4)
        if portal_pose_in_parent is not None:
            with partial_scene_output_fs.open("object_list.json", "r") as f:
                room_object_list = json.load(f)
            assert room_object_list["incoming_portal"] is not None
            portal_pos, portal_quat, _ = room_object_list["incoming_portal"]
            incoming_portal_transform = np.eye(4)
            incoming_portal_transform[:3, :3] = R.from_quat(portal_quat).as_matrix()
            incoming_portal_transform[:3, 3] = np.array(portal_pos) / 1000.0

            # Each object needs to be reorigined at the incoming portal and then moved to the outgoing
            rel_transform = portal_pose_in_parent @ np.linalg.inv(
                incoming_portal_transform
            )

        # Assert that the rel transform represents only a Z rotation
        rel_rot = R.from_matrix(rel_transform[:3, :3])
        if rel_rot.magnitude() > 1e-4:
            rot_axis = np.abs(rel_rot.as_rotvec() / rel_rot.magnitude())
            assert np.allclose(
                rot_axis, [0, 0, 1]
            ), f"Relative transform should only be a Z rotation. Current axis: {rot_axis}"

        # Build the mesh tree using our mesh tree library.
        # We don't need the upper side joints since we will only use these objects for bboxes.
        G = build_mesh_tree(partial_scene_target_name, load_upper=False)

        # Go through each object.
        roots = [
            node
            for node, in_degree in G.in_degree()
            if in_degree == 0 and not (set(G.nodes[node]["tags"]) & ALLOWED_PART_TAGS)
        ]

        for root_node in roots:
            obj_cat, obj_model, obj_inst_id, _ = root_node

            # For now, skip clutter objects
            if G.nodes[root_node]["is_loose"] == "C-":
                continue

            obj_name_in_scene = "-".join([obj_cat, obj_model, obj_inst_id])
            obj_rooms = G.nodes[root_node]["object_bounding_box"]["layer"]
            # TODO: Verify rooms.
            if obj_rooms == "0":
                obj_rooms = ""

            # Get the relevant bbox info.
            bbox_size = G.nodes[root_node]["object_bounding_box"]["extent"]
            bbox_world_center = G.nodes[root_node]["object_bounding_box"]["position"]
            bbox_world_rot = R.from_quat(G.nodes[root_node]["object_bounding_box"]["rotation"]) 

            # Apply any manual corrections
            force_fixed = None
            pos_correction = np.array([0, 0, 0])
            correction_key = obj_name_in_scene.replace("-", "_")
            if (
                scene_name in OBJECT_CORRECTIONS
                and correction_key in OBJECT_CORRECTIONS[scene_name]
            ):
                print("Correcting object", correction_key, "in", scene_name)
                pos_correction, orn_correction, force_fixed, rot_frame = (
                    OBJECT_CORRECTIONS[scene_name][correction_key]
                )
                if rot_frame == "world":
                    bbox_world_rot = R.from_euler(
                        "xyz", orn_correction
                    ) * bbox_world_rot
                elif rot_frame == "local":
                    bbox_world_rot = bbox_world_rot * R.from_euler("xyz", orn_correction)
                else:
                    raise ValueError(f"Unknown rot_frame {rot_frame}")

            # Apply the relevant transformation
            bbox_transform = np.eye(4)
            bbox_transform[:3, :3] = bbox_world_rot.as_matrix()
            bbox_transform[:3, 3] = bbox_world_center
            corrected_bbox_transform = rel_transform @ bbox_transform
            assert (
                corrected_bbox_transform[3, 3] == 1
            ), "Homogeneous coordinate should be 1"
            corrected_bbox_center = corrected_bbox_transform[:3, 3]
            corrected_bbox_rot = R.from_matrix(corrected_bbox_transform[:3, :3])

            # Apply the position correction
            corrected_bbox_center += pos_correction

            # Save pose to scene URDF
            scene_link = ET.SubElement(scene_tree_root, "link")
            scene_link.attrib = {
                "category": obj_cat,
                "model": obj_model,
                "name": obj_name_in_scene,
                "rooms": obj_rooms,
            }
            scene_link.attrib["bounding_box"] = " ".join(
                ["%.4f" % item for item in bbox_size]
            )
            joint = ET.SubElement(scene_tree_root, "joint")
            joint_fixed = (
                force_fixed
                if force_fixed is not None
                else (G.nodes[root_node]["is_loose"] is None)
            )
            joint.attrib = {
                "name": f"j_{obj_name_in_scene}",
                "type": "fixed" if joint_fixed else "floating",
            }
            joint_origin = ET.SubElement(joint, "origin")
            joint_origin_xyz = corrected_bbox_center.tolist()
            joint_origin_rpy = corrected_bbox_rot.as_euler("xyz").tolist()
            joint_origin.attrib = {
                "xyz": " ".join([str(item) for item in joint_origin_xyz]),
                "rpy": " ".join([str(item) for item in joint_origin_rpy]),
            }
            joint_parent = ET.SubElement(joint, "parent")
            joint_parent.attrib = {"link": "world"}
            joint_child = ET.SubElement(joint, "child")
            joint_child.attrib = {"link": obj_name_in_scene}

    # Write, reparse, and write with header, using the XML library,
    xmlstr = minidom.parseString(ET.tostring(scene_tree_root)).toprettyxml(indent="   ")
    xmlio = io.StringIO(xmlstr)
    tree = ET.parse(xmlio)
    with OSFS(scenes_dir).makedir(scene_name).makedir("urdf").open(
        f"{scene_name}_best.urdf", "wb"
    ) as f:
        tree.write(f, xml_declaration=True)


def main():
    # If this variable is set, we will only export the scenes in this list. This is useful for quickly
    # iterating on scenes (usdify otherwise takes 8+ hours) to resolve stability etc. issues.
    SCENES_TO_INCLUDE = []  # ["house_single_floor", "house_double_floor_lower", "house_double_floor_upper"]

    with b1k_pipeline.utils.ParallelZipFS("scenes.zip", write=True) as archive_fs:
        scenes_dir = archive_fs.makedir("scenes").getsyspath("/")
        errors = {}
        target_futures = {}

        with futures.ProcessPoolExecutor(max_workers=16) as target_executor:
            targets = b1k_pipeline.utils.get_targets("final_scenes")
            if SCENES_TO_INCLUDE:
                targets = [
                    target
                    for target in targets
                    if target.replace("scenes/", "") in SCENES_TO_INCLUDE
                ]
            for target in tqdm.tqdm(targets):
                target_futures[
                    target_executor.submit(process_target, target, scenes_dir)
                ] = target

            with tqdm.tqdm(total=len(target_futures)) as pbar:
                for future in futures.as_completed(target_futures.keys()):
                    try:
                        result = future.result()
                    except:
                        name = target_futures[future]
                        errors[name] = traceback.format_exc()

                    pbar.update(1)

                    remaining_targets = [
                        v for k, v in target_futures.items() if not k.done()
                    ]
                    if len(remaining_targets) < 10:
                        print("Remaining:", remaining_targets)

            print("Time for executor shutdown")

        print("Finished processing")

    pipeline_fs = b1k_pipeline.utils.PipelineFS()
    with pipeline_fs.pipeline_output().open("export_scenes.json", "w") as f:
        json.dump({"success": not errors, "errors": errors}, f)


if __name__ == "__main__":
    main()
