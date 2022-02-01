import itertools
import json
import logging
import math
import os
import random
import time
import xml.etree.ElementTree as ET

import cv2
import networkx as nx
import numpy as np

import trimesh
from scipy.spatial.transform import Rotation

import igibson
from igibson.external.pybullet_tools.utils import (
    get_all_links,
    get_center_extent,
    get_link_name,
    get_link_state,
    link_from_name,
    matrix_from_quat,
    quat_from_matrix,
)
from igibson.object_states.texture_change_state_mixin import TextureChangeStateMixin
from igibson.object_states.utils import clear_cached_states
from igibson.objects.stateful_object import StatefulObject
from igibson.render.mesh_renderer.materials import ProceduralMaterial, RandomizedMaterial
from igibson.utils import utils
from igibson.utils.urdf_utils import add_fixed_link, get_base_link_name, round_up, save_urdfs_without_floating_joints
from igibson.utils.utils import mat_to_quat_pos, rotate_vector_3d

from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path

class USDObject(StatefulObject):
    """
    # TODO: Update
    USDObjects are instantiated from a USD file. They can be composed of one
    or more links and joints. They should be passive. We use this class to
    parse our modified link tag for URDFs that embed objects into scenes
    """

    def __init__(
        self,
        prim_path,
        usd_path=None,
        name=None,
        category="object",
        class_id=None,
        scale=None,
        rendering_params=None,
        visible=True,
        fixed_base=False,
        abilities=None,
        model_path=None,
        bounding_box=None,
        fit_avg_dim_volume=False,
        avg_obj_dims=None,
        joint_friction=None,
        in_rooms=None,
        texture_randomization=False,
        # overwrite_inertial=True,
        scene_instance_folder=None,
        bddl_object_scope=None,
        # visualize_primitives=False,
        # merge_fixed_links=True,
    ):
        """
        # TODO: Update
        :param filename: urdf file path of that object model
        :param name: object name, unique for each object instance, e.g. door_3
        :param category: object category, e.g. door
        :param abilities: a list of abilities to instantiate the corresponding object states
        :param model_path: folder path of that object model
        :param bounding_box: bounding box of this object
        :param scale: scaling factor of this object
        :param fit_avg_dim_volume: whether to fit the object to have the same volume as the average dimension while keeping the aspect ratio
        :param fixed_base: whether this object has a fixed base to the world
        :param avg_obj_dims: average object dimension of this object
        :param joint_friction: joint friction for joints in this object
        :param in_rooms: which room(s) this object is in. It can be in more than one rooms if it sits at room boundary (e.g. doors)
        :param texture_randomization: whether to enable texture randomization
        :param overwrite_inertial: whether to overwrite the inertial frame of the original URDF using trimesh + density estimate
        :param scene_instance_folder: scene instance folder to split and save sub-URDFs
        :param bddl_object_scope: bddl object scope name, e.g. chip.n.04_2
        :param visualize_primitives: whether to render geometric primitives
        :param joint_states: joint positions and velocities, keyed by body index and joint name, in the form of
            List[Dict[name, Tuple(position, velocity)]]
        :param merge_fixed_links: whether to merge fixed links when importing to pybullet
        """
        super().__init__(
            prim_path=prim_path,
            name=name,
            category=category,
            class_id=class_id,
            scale=scale,
            rendering_params=rendering_params,
            visible=visible,
            fixed_base=fixed_base,
            abilities=abilities,
        )
        self._usd_path = usd_path
        self.in_rooms = in_rooms
        self.texture_randomization = texture_randomization
        # self.overwrite_inertial = overwrite_inertial
        self.scene_instance_folder = scene_instance_folder
        self.bddl_object_scope = bddl_object_scope
        # self.merge_fixed_links = merge_fixed_links
        self.room_floor = None

        # Friction for all prismatic and revolute joints
        # TODO: Shouldn't these be baked into the USD directly?
        if joint_friction is not None:
            self.joint_friction = joint_friction
        else:
            if self.category in ["oven", "dishwasher"]:
                self.joint_friction = 30.0
            elif self.category in ["toilet"]:
                self.joint_friction = 3.0
            else:
                self.joint_friction = 10.0

        # # These following fields have exactly the same length (i.e. the number
        # # of sub URDFs in this object)
        # # urdf_paths, string
        # self.urdf_paths = []
        # # local transformations from sub URDFs to the main body
        # self.local_transforms = []
        # # whether these sub URDFs are fixed
        # self.is_fixed = []
        #
        # self.main_body = -1

        logging.info("Category " + self.category)
        # self.filename = filename
        # dirname = os.path.dirname(filename)
        # urdf = os.path.basename(filename)
        # urdf_name, _ = os.path.splitext(urdf)
        # simplified_urdf = os.path.join(dirname, urdf_name + "_simplified.urdf")
        # if os.path.exists(simplified_urdf):
        #     self.filename = simplified_urdf
        #     filename = simplified_urdf
        logging.info("Loading the following USD template " + self._usd_path)
        # self.object_tree = ET.parse(filename)  # Parse the URDF

        # TODO: May need to come back to this if extra boxes are being visually loaded (e.g.: with the "simplified" objects)
        # if not visualize_primitives:
        #     for link in self.object_tree.findall("link"):
        #         for element in link:
        #             if element.tag == "visual" and len(element.findall(".//box")) > 0:
        #                 link.remove(element)

        self.model_path = model_path
        if self.model_path is None:
            self.model_path = os.path.dirname(self._usd_path)

        # # Change the mesh filenames to include the entire path
        # for mesh in self.object_tree.iter("mesh"):
        #     mesh.attrib["filename"] = os.path.join(self.model_path, mesh.attrib["filename"])

        # Apply the desired bounding box size / scale
        # First obtain the scaling factor
        if bounding_box is not None and scale is not None:
            raise Exception("You cannot define both scale and bounding box size for a URDFObject")

        meta_json = os.path.join(self.model_path, "misc", "metadata.json")
        bbox_json = os.path.join(self.model_path, "misc", "bbox.json")
        # In the format of {link_name: [linkX, linkY, linkZ]}
        self.metadata = {}
        meta_links = dict()
        if os.path.isfile(meta_json):
            with open(meta_json, "r") as f:
                self.metadata = json.load(f)
                bbox_size = np.array(self.metadata["bbox_size"])
                base_link_offset = np.array(self.metadata["base_link_offset"])

                if "orientations" in self.metadata and len(self.metadata["orientations"]) > 0:
                    self.orientations = self.metadata["orientations"]
                else:
                    self.orientations = None

                if "links" in self.metadata:
                    meta_links = self.metadata["links"]

        elif os.path.isfile(bbox_json):
            with open(bbox_json, "r") as bbox_file:
                bbox_data = json.load(bbox_file)
                bbox_max = np.array(bbox_data["max"])
                bbox_min = np.array(bbox_data["min"])
                bbox_size = bbox_max - bbox_min
                base_link_offset = (bbox_min + bbox_max) / 2.0
        else:
            bbox_size = None
            base_link_offset = np.zeros(3)

        if bbox_size is not None:
            if fit_avg_dim_volume:
                if avg_obj_dims is None:
                    scale = np.ones(3)
                else:
                    spec_vol = avg_obj_dims["size"][0] * avg_obj_dims["size"][1] * avg_obj_dims["size"][2]
                    cur_vol = bbox_size[0] * bbox_size[1] * bbox_size[2]
                    volume_ratio = spec_vol / cur_vol
                    size_ratio = np.cbrt(volume_ratio)
                    scale = np.array([size_ratio] * 3)
                bounding_box = bbox_size * scale
            elif bounding_box is not None:
                # Obtain the scale as the ratio between the desired bounding box size
                # and the original bounding box size of the object at scale (1, 1, 1)
                scale = bounding_box / bbox_size
            else:
                if scale is None:
                    scale = np.ones(3)
                bounding_box = bbox_size * scale

        self.scale = scale
        self.bounding_box = bounding_box

        # If no bounding box, cannot compute dynamic properties from density
        if self.bounding_box is None:
            self.overwrite_inertial = False

        logging.info("Scale: " + np.array2string(scale))

        # We need to know where the base_link origin is wrt. the bounding box
        # center. That allows us to place the model correctly since the joint
        # transformations given in the scene urdf are wrt. the bounding box
        # center. We need to scale this offset as well.
        self.scaled_bbxc_in_blf = -self.scale * base_link_offset

        self.avg_obj_dims = avg_obj_dims

        self.base_link_name = get_base_link_name(self.object_tree)
        self.rename_urdf()

        self.meta_links = {}
        self.add_meta_links(meta_links)

        self.scale_object()
        self.remove_floating_joints(self.scene_instance_folder)
        self.prepare_link_based_bounding_boxes()

        self.prepare_visual_mesh_to_material()

    def load_supporting_surfaces(self):
        self.supporting_surfaces = {}

        # Supporting surfaces can potentially refer to the names of the fixed links that are
        # merged into the world. These links will become inaccessible after the merge, e.g.
        # link_from_name will raise an error and we won't have any correspounding link id to
        # invoke get_link_state later.
        if self.merge_fixed_links:
            return

        heights_file = os.path.join(self.model_path, "misc", "heights_per_link.json")
        if not os.path.isfile(heights_file):
            return

        with open(heights_file, "r") as f:
            heights = json.load(f)

        original_object_tree = ET.parse(self.filename)
        sub_urdfs = [ET.parse(urdf_path) for urdf_path in self.urdf_paths]
        for predicate in heights:
            height_maps_dir = os.path.join(self.model_path, "misc", "height_maps_per_link", "{}".format(predicate))

            height_maps = {}
            for link_name in heights[predicate]:
                link_dir = os.path.join(height_maps_dir, link_name)

                # Get collision mesh of the link in the original urdf
                link = original_object_tree.find(".//link[@name='{}']".format(link_name))
                link_col_mesh = link.find("collision/geometry/mesh")
                col_mesh_path = os.path.join(self.model_path, link_col_mesh.attrib["filename"])

                # Try to find the body_id (after splitting) and the new link name (after renaming)
                # by matching the collision mesh file path
                new_link = None
                new_body_id = None
                assert len(sub_urdfs) == len(self.get_body_ids())
                for sub_urdf, body_id in zip(sub_urdfs, self.get_body_ids()):
                    for link in sub_urdf.findall("link"):
                        link_col_mesh = link.find("collision/geometry/mesh")
                        if link_col_mesh is None:
                            continue
                        if link_col_mesh.attrib["filename"] == col_mesh_path:
                            new_link = link.attrib["name"]
                            new_body_id = body_id
                            break
                    if new_link is not None:
                        break

                assert new_link is not None
                new_link_id = link_from_name(new_body_id, new_link)

                height_maps[(new_body_id, new_link_id)] = []

                for i, z_value in enumerate(heights[predicate][link_name]):
                    img_fname = os.path.join(link_dir, link_dir, "{}.png".format(i))
                    xy_map = cv2.imread(img_fname, 0)
                    height_maps[(new_body_id, new_link_id)].append((z_value, xy_map))
            self.supporting_surfaces[predicate] = height_maps

    def sample_orientation(self):
        if self.orientations is None:
            raise ValueError("No orientation probabilities set")
        indices = list(range(len(self.orientations)))
        orientations = [np.array(o["rotation"]) for o in self.orientations]
        probabilities = [o["prob"] for o in self.orientations]
        probabilities = np.array(probabilities) / np.sum(probabilities)
        chosen_orientation_idx = np.random.choice(indices, p=probabilities)
        chosen_orientation = orientations[chosen_orientation_idx]
        # Randomize yaw based on the variation annotation
        # variation = [o['variation'] for o in self.orientations]
        # min_rotation = 0.05
        # rotation_variance = max(
        #     variation[chosen_orientation_idx], min_rotation)
        # rot_num = np.random.random() * rotation_variance

        # Randomize yaw from -pi to pi
        rot_num = np.random.uniform(-1, 1)
        rot_matrix = np.array(
            [
                [math.cos(math.pi * rot_num), -math.sin(math.pi * rot_num), 0.0],
                [math.sin(math.pi * rot_num), math.cos(math.pi * rot_num), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rotated_quat = quat_from_matrix(np.dot(rot_matrix, matrix_from_quat(chosen_orientation)))
        return rotated_quat

    def get_prefixed_joint_name(self, name):
        return self.name + "_" + name

    def get_prefixed_link_name(self, name):
        if name == "world":
            return name
        elif name == self.base_link_name:
            # The base_link get renamed as the link tag indicates
            # Just change the name of the base link in the embedded urdf
            return self.name
        else:
            # The other links get also renamed to add the name of the link tag as prefix
            # This allows us to load several instances of the same object
            return self.name + "_" + name

    def rename_urdf(self):
        """
        Helper function that renames the file paths in the object urdf
        from relative paths to absolute paths
        """

        # Change the links of the added object to adapt to the given name
        for link_emb in self.object_tree.iter("link"):
            link_emb.attrib["name"] = self.get_prefixed_link_name(link_emb.attrib["name"])

        # Change the joints of the added object to adapt them to the given name
        for joint_emb in self.object_tree.iter("joint"):
            # We change the joint name
            joint_emb.attrib["name"] = self.get_prefixed_joint_name(joint_emb.attrib["name"])

            # We change the child link names
            for child_emb in joint_emb.findall("child"):
                child_emb.attrib["link"] = self.get_prefixed_link_name(child_emb.attrib["link"])

            # and the parent link names
            for parent_emb in joint_emb.findall("parent"):
                parent_emb.attrib["link"] = self.get_prefixed_link_name(parent_emb.attrib["link"])

    def scale_object(self):
        """
        Scale the object according to the given bounding box
        """
        # We need to scale 1) the meshes, 2) the position of meshes, 3) the position of joints, 4) the orientation
        # axis of joints. The problem is that those quantities are given wrt. its parent link frame, and this can be
        # rotated wrt. the frame the scale was given in Solution: parse the kin tree joint by joint, extract the
        # rotation, rotate the scale, apply rotated scale to 1, 2, 3, 4 in the child link frame

        # First, define the scale in each link reference frame
        # and apply it to the joint values
        base_link_name = self.get_prefixed_link_name(self.base_link_name)
        self.scales_in_link_frame = {base_link_name: self.scale}
        all_processed = False
        while not all_processed:
            all_processed = True
            for joint in self.object_tree.iter("joint"):
                parent_link_name = joint.find("parent").attrib["link"]
                child_link_name = joint.find("child").attrib["link"]
                if parent_link_name in self.scales_in_link_frame and child_link_name not in self.scales_in_link_frame:
                    scale_in_parent_lf = self.scales_in_link_frame[parent_link_name]
                    # The location of the joint frame is scaled using the scale in the parent frame
                    for origin in joint.iter("origin"):
                        current_origin_xyz = np.array([float(val) for val in origin.attrib["xyz"].split(" ")])
                        new_origin_xyz = np.multiply(current_origin_xyz, scale_in_parent_lf)
                        new_origin_xyz = np.array([round_up(val, 10) for val in new_origin_xyz])
                        origin.attrib["xyz"] = " ".join(map(str, new_origin_xyz))

                    # scale the prismatic joint
                    if joint.attrib["type"] == "prismatic":
                        limits = joint.findall("limit")
                        assert len(limits) == 1
                        limit = limits[0]
                        axes = joint.findall("axis")
                        assert len(axes) == 1
                        axis = axes[0]
                        axis_np = np.array([float(elem) for elem in axis.attrib["xyz"].split()])
                        major_axis = np.argmax(np.abs(axis_np))
                        # assume the prismatic joint is roughly axis-aligned
                        limit.attrib["upper"] = str(float(limit.attrib["upper"]) * scale_in_parent_lf[major_axis])
                        limit.attrib["lower"] = str(float(limit.attrib["lower"]) * scale_in_parent_lf[major_axis])

                    # Get the rotation of the joint frame and apply it to the scale
                    if "rpy" in joint.keys():
                        joint_frame_rot = np.array([float(val) for val in joint.attrib["rpy"].split(" ")])
                        # Rotate the scale
                        scale_in_child_lf = rotate_vector_3d(scale_in_parent_lf, *joint_frame_rot, cck=True)
                        scale_in_child_lf = np.absolute(scale_in_child_lf)
                    else:
                        scale_in_child_lf = scale_in_parent_lf

                    # print("Adding: ", joint.find("child").attrib["link"])

                    self.scales_in_link_frame[joint.find("child").attrib["link"]] = scale_in_child_lf

                    # The axis of the joint is defined in the joint frame, we scale it after applying the rotation
                    for axis in joint.iter("axis"):
                        current_axis_xyz = np.array([float(val) for val in axis.attrib["xyz"].split(" ")])
                        new_axis_xyz = np.multiply(current_axis_xyz, scale_in_child_lf)
                        new_axis_xyz /= np.linalg.norm(new_axis_xyz)
                        new_axis_xyz = np.array([round_up(val, 10) for val in new_axis_xyz])
                        axis.attrib["xyz"] = " ".join(map(str, new_axis_xyz))

                    # Iterate again the for loop since we added new elements to the dictionary
                    all_processed = False

        all_links = self.object_tree.findall("link")
        # compute dynamics properties
        if self.overwrite_inertial and self.category not in ["walls", "floors", "ceilings"]:
            all_links_trimesh = []
            total_volume = 0.0
            for link in all_links:
                meshes = link.findall("collision/geometry/mesh")
                if len(meshes) == 0:
                    all_links_trimesh.append(None)
                    continue
                # assume one collision mesh per link
                assert len(meshes) == 1, (self.filename, link.attrib["name"])
                # check collision mesh path
                collision_mesh_path = os.path.join(meshes[0].attrib["filename"])
                trimesh_obj = trimesh.load(file_obj=collision_mesh_path, force="mesh")
                all_links_trimesh.append(trimesh_obj)
                volume = trimesh_obj.volume
                # a hack to artificially increase the density of the lamp base
                if link.attrib["name"] == base_link_name:
                    if self.category in ["lamp"]:
                        volume *= 10.0
                total_volume += volume

            # avg L x W x H and Weight is given for this object category
            if self.avg_obj_dims is not None:
                avg_density = self.avg_obj_dims["density"]

            # otherwise, use the median density across all existing object categories
            else:
                avg_density = 67.0

            # Scale the mass based on bounding box size
            # TODO: how to scale moment of inertia?
            total_mass = avg_density * self.bounding_box[0] * self.bounding_box[1] * self.bounding_box[2]
            # print('total_mass', total_mass)

            density = total_mass / total_volume
            # print('avg density', density)
            for trimesh_obj in all_links_trimesh:
                if trimesh_obj is not None:
                    trimesh_obj.density = density

            assert len(all_links_trimesh) == len(all_links)

        # Now iterate over all links and scale the meshes and positions
        for i, link in enumerate(all_links):
            if self.overwrite_inertial and self.category not in ["walls", "floors", "ceilings"]:
                link_trimesh = all_links_trimesh[i]
                # assign dynamics properties
                inertials = link.findall("inertial")
                if len(inertials) == 0:
                    inertial = ET.SubElement(link, "inertial")
                else:
                    assert len(inertials) == 1
                    inertial = inertials[0]

                masses = inertial.findall("mass")
                if len(masses) == 0:
                    mass = ET.SubElement(inertial, "mass")
                else:
                    assert len(masses) == 1
                    mass = masses[0]

                inertias = inertial.findall("inertia")
                if len(inertias) == 0:
                    inertia = ET.SubElement(inertial, "inertia")
                else:
                    assert len(inertias) == 1
                    inertia = inertias[0]

                origins = inertial.findall("origin")
                if len(origins) == 0:
                    origin = ET.SubElement(inertial, "origin")
                else:
                    assert len(origins) == 1
                    origin = origins[0]

                if link_trimesh is not None:
                    # a hack to artificially increase the density of the lamp base
                    if link.attrib["name"] == base_link_name:
                        if self.category in ["lamp"]:
                            link_trimesh.density *= 10.0

                    if link_trimesh.is_watertight:
                        center = np.copy(link_trimesh.center_mass)
                    else:
                        center = np.copy(link_trimesh.centroid)

                    collision_mesh = [col for col in link.findall("collision") if col.find("geometry/mesh") is not None]
                    assert len(collision_mesh) == 1, "more than one collision mesh in one link"
                    collision_mesh = collision_mesh[0]
                    collision_mesh_origin = collision_mesh.find("origin")
                    if collision_mesh_origin is not None:
                        offset = np.array([float(val) for val in collision_mesh_origin.attrib["xyz"].split(" ")])
                        center += offset

                    # The inertial frame origin will be scaled down below.
                    # Here, it has the value BEFORE scaling
                    origin.attrib["xyz"] = " ".join(map(str, center))
                    origin.attrib["rpy"] = " ".join(map(str, [0.0, 0.0, 0.0]))

                    mass.attrib["value"] = str(round_up(link_trimesh.mass, 10))
                    moment_of_inertia = link_trimesh.moment_inertia
                    inertia.attrib["ixx"] = str(moment_of_inertia[0][0])
                    inertia.attrib["ixy"] = str(moment_of_inertia[0][1])
                    inertia.attrib["ixz"] = str(moment_of_inertia[0][2])
                    inertia.attrib["iyy"] = str(moment_of_inertia[1][1])
                    inertia.attrib["iyz"] = str(moment_of_inertia[1][2])
                    inertia.attrib["izz"] = str(moment_of_inertia[2][2])
                else:
                    # empty link that does not have any mesh
                    origin.attrib["xyz"] = " ".join(map(str, [0.0, 0.0, 0.0]))
                    origin.attrib["rpy"] = " ".join(map(str, [0.0, 0.0, 0.0]))
                    mass.attrib["value"] = str(0.0)
                    inertia.attrib["ixx"] = str(0.0)
                    inertia.attrib["ixy"] = str(0.0)
                    inertia.attrib["ixz"] = str(0.0)
                    inertia.attrib["iyy"] = str(0.0)
                    inertia.attrib["iyz"] = str(0.0)
                    inertia.attrib["izz"] = str(0.0)

            scale_in_lf = self.scales_in_link_frame[link.attrib["name"]]
            # Apply the scale to all mesh elements within the link (original scale and origin)
            for mesh in link.iter("mesh"):
                if "scale" in mesh.attrib:
                    mesh_scale = np.array([float(val) for val in mesh.attrib["scale"].split(" ")])
                    new_scale = np.multiply(mesh_scale, scale_in_lf)
                    new_scale = np.array([round_up(val, 10) for val in new_scale])
                    mesh.attrib["scale"] = " ".join(map(str, new_scale))
                else:
                    new_scale = np.array([round_up(val, 10) for val in scale_in_lf])
                    mesh.set("scale", " ".join(map(str, new_scale)))

            for box in link.iter("box"):
                if "size" in box.attrib:
                    box_scale = np.array([float(val) for val in box.attrib["size"].split(" ")])
                    new_scale = np.multiply(box_scale, scale_in_lf)
                    new_scale = np.array([round_up(val, 10) for val in new_scale])
                    box.attrib["size"] = " ".join(map(str, new_scale))

            for origin in link.iter("origin"):
                origin_xyz = np.array([float(val) for val in origin.attrib["xyz"].split(" ")])
                new_origin_xyz = np.multiply(origin_xyz, scale_in_lf)
                new_origin_xyz = np.array([round_up(val, 10) for val in new_origin_xyz])
                origin.attrib["xyz"] = " ".join(map(str, new_origin_xyz))

    # def remove_floating_joints(self, folder=None):
    #     """
    #     Split a single urdf to multiple urdfs if there exist floating joints
    #     """
    #     if folder is None:
    #         timestr = time.strftime("%Y%m%d-%H%M%S")
    #         folder = os.path.join(
    #             igibson.ig_dataset_path,
    #             "scene_instances",
    #             "{}_{}_{}".format(timestr, random.getrandbits(64), os.getpid()),
    #         )
    #         os.makedirs(folder, exist_ok=True)
    #
    #     # Deal with floating joints inside the embedded urdf
    #     file_prefix = os.path.join(folder, self.name)
    #     urdfs_no_floating = save_urdfs_without_floating_joints(self.object_tree, file_prefix)
    #
    #     # append a new tuple of file name of the instantiated embedded urdf
    #     # and the transformation (!= identity if its connection was floating)
    #     for i, urdf in enumerate(urdfs_no_floating):
    #         self.urdf_paths.append(urdfs_no_floating[urdf][0])
    #         self.local_transforms.append(mat_to_quat_pos(urdfs_no_floating[urdf][1]))
    #         if urdfs_no_floating[urdf][2]:
    #             self.main_body = i
    #             self.is_fixed.append(self.fixed_base)
    #         else:
    #             self.is_fixed.append(False)

    # def prepare_visual_mesh_to_material(self):
    #     # mapping between visual objects and possible textures
    #     # multiple visual objects can share the same material
    #     # if some sub URDF does not have visual object or this URDF is part of
    #     # the building structure, it will have an empty dict
    #     # [
    #     #     {                                             # 1st sub URDF
    #     #         'visual_1.obj': randomized_material_1
    #     #         'visual_2.obj': randomized_material_1
    #     #     },
    #     #     {},                                            # 2nd sub URDF
    #     #     {                                              # 3rd sub URDF
    #     #         'visual_3.obj': randomized_material_2
    #     #     }
    #     # ]
    #
    #     self.visual_mesh_to_material = [{} for _ in self.urdf_paths]
    #
    #     # a list of all materials used for RandomizedMaterial
    #     self.randomized_materials = []
    #     # mapping from material class to friction coefficient
    #     self.material_to_friction = None
    #
    #     # procedural material that can change based on state changes
    #     self.procedural_material = None
    #
    #     self.texture_procedural_generation = False
    #     for state in self.states:
    #         if issubclass(state, TextureChangeStateMixin):
    #             self.texture_procedural_generation = True
    #             break
    #
    #     if self.texture_randomization and self.texture_procedural_generation:
    #         logging.warn("Cannot support both randomized and procedural texture. Dropping texture randomization.")
    #         self.texture_randomization = False
    #
    #     if self.texture_randomization:
    #         self.prepare_randomized_texture()
    #     if self.texture_procedural_generation:
    #         self.prepare_procedural_texture()
    #
    #     self.create_link_name_vm_mapping()

    # def create_link_name_vm_mapping(self):
    #     self.link_name_to_vm = []
    #
    #     for i in range(len(self.urdf_paths)):
    #         link_name_to_vm_urdf = {}
    #         sub_urdf_tree = ET.parse(self.urdf_paths[i])
    #
    #         links = sub_urdf_tree.findall("link")
    #         for link in links:
    #             name = link.attrib["name"]
    #             if name in link_name_to_vm_urdf:
    #                 raise ValueError("link name collision")
    #             link_name_to_vm_urdf[name] = []
    #             for visual_mesh in link.findall("visual/geometry/mesh"):
    #                 link_name_to_vm_urdf[name].append(visual_mesh.attrib["filename"])
    #
    #         if self.merge_fixed_links:
    #             # Add visual meshes of the child link to the parent link for fixed joints because they will be merged
    #             # by pybullet after loading
    #             vms_before_merging = set([item for _, vms in link_name_to_vm_urdf.items() for item in vms])
    #             directed_graph = nx.DiGraph()
    #             child_to_parent = dict()
    #             for joint in sub_urdf_tree.findall("joint"):
    #                 if joint.attrib["type"] == "fixed":
    #                     child_link_name = joint.find("child").attrib["link"]
    #                     parent_link_name = joint.find("parent").attrib["link"]
    #                     directed_graph.add_edge(child_link_name, parent_link_name)
    #                     child_to_parent[child_link_name] = parent_link_name
    #             for child_link_name in list(nx.algorithms.topological_sort(directed_graph)):
    #                 if child_link_name in child_to_parent:
    #                     parent_link_name = child_to_parent[child_link_name]
    #                     link_name_to_vm_urdf[parent_link_name].extend(link_name_to_vm_urdf[child_link_name])
    #                     del link_name_to_vm_urdf[child_link_name]
    #             vms_after_merging = set([item for _, vms in link_name_to_vm_urdf.items() for item in vms])
    #             assert vms_before_merging == vms_after_merging
    #
    #         self.link_name_to_vm.append(link_name_to_vm_urdf)
    #
    # def randomize_texture(self):
    #     """
    #     Randomize texture and material for each link / visual shape
    #     """
    #     for material in self.randomized_materials:
    #         material.randomize()
    #     self.update_friction()
    #
    # def update_friction(self):
    #     """
    #     Update the surface lateral friction for each link based on its material
    #     """
    #     if self.material_to_friction is None:
    #         return
    #     for i in range(len(self.urdf_paths)):
    #         # if the sub URDF does not have visual meshes
    #         if len(self.visual_mesh_to_material[i]) == 0:
    #             continue
    #         body_id = self.get_body_ids()[i]
    #         sub_urdf_tree = ET.parse(self.urdf_paths[i])
    #
    #         for j in np.arange(-1, p.getNumJoints(body_id)):
    #             # base_link
    #             if j == -1:
    #                 link_name = p.getBodyInfo(body_id)[0].decode("UTF-8")
    #             else:
    #                 link_name = p.getJointInfo(body_id, j)[12].decode("UTF-8")
    #             link = sub_urdf_tree.find(".//link[@name='{}']".format(link_name))
    #             link_materials = []
    #             for visual_mesh in link.findall("visual/geometry/mesh"):
    #                 link_materials.append(self.visual_mesh_to_material[i][visual_mesh.attrib["filename"]])
    #             link_frictions = []
    #             for link_material in link_materials:
    #                 if link_material.random_class is None:
    #                     friction = 0.5
    #                 elif link_material.random_class not in self.material_to_friction:
    #                     friction = 0.5
    #                 else:
    #                     friction = self.material_to_friction.get(link_material.random_class, 0.5)
    #                 link_frictions.append(friction)
    #             link_friction = np.mean(link_frictions)
    #             p.changeDynamics(body_id, j, lateralFriction=link_friction)
    #
    # def prepare_randomized_texture(self):
    #     """
    #     Set up mapping from visual meshes to randomizable materials
    #     """
    #     if self.category in ["walls", "floors", "ceilings"]:
    #         material_groups_file = os.path.join(
    #             self.model_path, "misc", "{}_material_groups.json".format(self.category)
    #         )
    #     else:
    #         material_groups_file = os.path.join(self.model_path, "misc", "material_groups.json")
    #
    #     assert os.path.isfile(material_groups_file), "cannot find material group: {}".format(material_groups_file)
    #     with open(material_groups_file) as f:
    #         material_groups = json.load(f)
    #
    #     # create randomized material for each material group
    #     all_material_categories = material_groups[0]
    #     all_materials = {}
    #     for key in all_material_categories:
    #         all_materials[int(key)] = RandomizedMaterial(all_material_categories[key])
    #
    #     # make visual mesh file path absolute
    #     visual_mesh_to_idx = material_groups[1]
    #     for old_path in list(visual_mesh_to_idx.keys()):
    #         new_path = os.path.join(self.model_path, "shape", "visual", old_path)
    #         visual_mesh_to_idx[new_path] = visual_mesh_to_idx[old_path]
    #         del visual_mesh_to_idx[old_path]
    #
    #     # check each visual object belongs to which sub URDF in case of splitting
    #     for i, urdf_path in enumerate(self.urdf_paths):
    #         sub_urdf_tree = ET.parse(urdf_path)
    #         for visual_mesh_path in visual_mesh_to_idx:
    #             # check if this visual object belongs to this URDF
    #             if sub_urdf_tree.find(".//mesh[@filename='{}']".format(visual_mesh_path)) is not None:
    #                 self.visual_mesh_to_material[i][visual_mesh_path] = all_materials[
    #                     visual_mesh_to_idx[visual_mesh_path]
    #                 ]
    #
    #     self.randomized_materials = list(all_materials.values())
    #
    #     friction_json = os.path.join(igibson.ig_dataset_path, "materials", "material_friction.json")
    #     if os.path.isfile(friction_json):
    #         with open(friction_json) as f:
    #             self.material_to_friction = json.load(f)
    #
    # def prepare_procedural_texture(self):
    #     """
    #     Set up mapping from visual meshes to procedural materials
    #     Assign all visual meshes to the same ProceduralMaterial
    #     """
    #     procedural_material = ProceduralMaterial(material_folder=os.path.join(self.model_path, "material"))
    #
    #     for i, urdf_path in enumerate(self.urdf_paths):
    #         sub_urdf_tree = ET.parse(urdf_path)
    #         for visual_mesh in sub_urdf_tree.findall("link/visual/geometry/mesh"):
    #             filename = visual_mesh.attrib["filename"]
    #             self.visual_mesh_to_material[i][filename] = procedural_material
    #
    #     for state in self.states:
    #         if issubclass(state, TextureChangeStateMixin):
    #             procedural_material.add_state(state)
    #             self.states[state].material = procedural_material
    #
    #     self.procedural_material = procedural_material

    def _load(self, simulator):
        """
        Load the object into pybullet and set it to the correct pose
        """
        # Add reference to stage and grab prim
        add_reference_to_stage(usd_path=self._usd_path, prim_path=self._prim_path)
        prim = get_prim_at_path(self._prim_path)

        # body_ids = []

        # flags = p.URDF_ENABLE_SLEEPING
        # if self.merge_fixed_links:
        #     flags |= p.URDF_MERGE_FIXED_LINKS
        #
        # if igibson.ignore_visual_shape:
        #     flags |= p.URDF_IGNORE_VISUAL_SHAPES

        # for idx in range(len(self.urdf_paths)):
        #     logging.info("Loading " + self.urdf_paths[idx])
        #     is_fixed = self.is_fixed[idx]
        #     body_id = p.loadURDF(self.urdf_paths[idx], flags=flags, useFixedBase=is_fixed)
        #     p.changeDynamics(body_id, -1, activationState=p.ACTIVATION_STATE_ENABLE_SLEEPING)
        #
        #     # Set joint friction
        #     for j in range(p.getNumJoints(body_id)):
        #         info = p.getJointInfo(body_id, j)
        #         joint_type = info[2]
        #         if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
        #             p.setJointMotorControl2(
        #                 body_id, j, p.VELOCITY_CONTROL, targetVelocity=0.0, force=self.joint_friction
        #             )
        #
        #     simulator.load_object_in_renderer(
        #         self,
        #         body_id,
        #         self.class_id,
        #         visual_mesh_to_material=self.visual_mesh_to_material[idx],
        #         link_name_to_vm=self.link_name_to_vm[idx],
        #         **self._rendering_params
        #     )
        #
        #     body_ids.append(body_id)

        # self.load_supporting_surfaces()

        return prim

    def initialize(self):
        # Run super method first
        super().initialize()

        # Also set the joint friction values
        # TODO: Move all joint info to separate class
        for i in range(self._dc_interface.get_articulation_joint_count(self._handle)):
            joint_handle = self._dc_interface.get_articulation_joint(self._handle, i)
            joint_path = self._dc_interface.get_joint_path(joint_handle)
            joint_prim = get_prim_at_path(joint_path)
            joint_prim.GetAttribute("physxJoint:jointFriction").Set(self.joint_friction)

    def set_bbox_center_position_orientation(self, pos, orn):
        # TODO
        rotated_offset = p.multiplyTransforms([0, 0, 0], orn, self.scaled_bbxc_in_blf, [0, 0, 0, 1])[0]
        self.set_base_link_position_orientation(pos + rotated_offset, orn)

    # def set_position_orientation(self, pos, orn):
    #     # TODO: replace this with super() call once URDFObject no longer works with multiple body ids
    #     p.resetBasePositionAndOrientation(self.get_body_ids()[self.main_body], pos, orn)
    #
    #     # Get pose of the main body base link frame.
    #     dynamics_info = p.getDynamicsInfo(self.get_body_ids()[self.main_body], -1)
    #     inertial_pos, inertial_orn = dynamics_info[3], dynamics_info[4]
    #     inv_inertial_pos, inv_inertial_orn = p.invertTransform(inertial_pos, inertial_orn)
    #     link_frame_pos, link_frame_orn = p.multiplyTransforms(pos, orn, inv_inertial_pos, inv_inertial_orn)
    #
    #     # Set the non-main bodies to their original local poses
    #     for i, body_id in enumerate(self.get_body_ids()):
    #         if body_id == self.get_body_ids()[self.main_body]:
    #             continue
    #
    #         # Get pose of the sub URDF base link frame
    #         sub_urdf_pos, sub_urdf_orn = p.multiplyTransforms(link_frame_pos, link_frame_orn, *self.local_transforms[i])
    #
    #         # Convert to CoM frame
    #         dynamics_info = p.getDynamicsInfo(body_id, -1)
    #         inertial_pos, inertial_orn = dynamics_info[3], dynamics_info[4]
    #         sub_urdf_pos, sub_urdf_orn = p.multiplyTransforms(sub_urdf_pos, sub_urdf_orn, inertial_pos, inertial_orn)
    #
    #         p.resetBasePositionAndOrientation(body_id, sub_urdf_pos, sub_urdf_orn)
    #
    #     clear_cached_states(self)

    # def set_base_link_position_orientation(self, pos, orn):
    #     """Set object base link position and orientation in the format of Tuple[Array[x, y, z], Array[x, y, z, w]]"""
    #     dynamics_info = p.getDynamicsInfo(self.get_body_ids()[self.main_body], -1)
    #     inertial_pos, inertial_orn = dynamics_info[3], dynamics_info[4]
    #     pos, orn = p.multiplyTransforms(pos, orn, inertial_pos, inertial_orn)
    #     self.set_position_orientation(pos, orn)

    # def add_meta_links(self, meta_links):
    #     """
    #     Adds the meta links (e.g. heating element position, water source position) from the metadata file
    #     into the URDF tree for this object prior to loading.
    #
    #     :param meta_links: Dictionary of meta links in the form of {link_name: [linkX, linkY, linkZ]}
    #     :return: None.
    #     """
    #     for meta_link_name, link_info in meta_links.items():
    #         if link_info["geometry"] is not None:
    #             # Objects with geometry actually need to be added into the URDF for collision purposes.
    #             # These objects cannot be imported with fixed links.
    #             self.merge_fixed_links = False
    #             add_fixed_link(self.object_tree, meta_link_name, link_info)
    #         else:
    #             # Otherwise, the "link" is just an offset, so we save its position.
    #             self.meta_links[meta_link_name] = np.array(link_info["xyz"])

    # TODO: remove after split floors
    def set_room_floor(self, room_floor):
        assert self.category == "floors"
        self.room_floor = room_floor

    def prepare_link_based_bounding_boxes(self):
        """This function simply converts the link bounding box metadata to match prefixed link names."""
        self.unscaled_link_bounding_boxes = {}
        if "link_bounding_boxes" in self.metadata:
            for name, bb_data in self.metadata["link_bounding_boxes"].items():
                converted_name = self.get_prefixed_link_name(name)
                self.unscaled_link_bounding_boxes[converted_name] = bb_data

    def get_base_aligned_bounding_box(self, visual=False, xy_aligned=False, fallback_to_aabb=False):
        """
        Get a bounding box for this object that's axis-aligned in the object's base frame.
        """
        bbox_type = "visual" if visual else "collision"

        # Get the base position transform.
        pos, orn = self.get_position_orientation()
        base_com_to_world = utils.quat_pos_to_mat(pos, orn)

        # Compute the world-to-base frame transform.
        world_to_base_com = trimesh.transformations.inverse_matrix(base_com_to_world)

        # Grab the corners of all the different links' bounding boxes. We will later fit a bounding box to
        # this set of points to get our final, base-frame bounding box.
        points = []

        links = [link_id] if link_id is not None else get_all_links(body_id)
        for link in links:
            link_name = get_link_name(body_id, link)
            # If the link has a bounding box annotation.
            if link_name in self.unscaled_link_bounding_boxes:
                # If a visual bounding box does not exist in the dictionary, try switching to collision.
                # We expect that every link has its collision bb annotated (or set to None if none exists).
                if bbox_type == "visual" and "visual" not in self.unscaled_link_bounding_boxes[link_name]:
                    logging.debug(
                        "Falling back to collision bbox for object %s link %s since no visual bbox exists.",
                        self.name,
                        link_name,
                    )
                    bbox_type = "collision"

                # Check if the annotation is still missing.
                if bbox_type not in self.unscaled_link_bounding_boxes[link_name]:
                    raise ValueError(
                        "Could not find %s bounding box for object %s link %s" % (bbox_type, self.name, link_name)
                    )

                # Check if a mesh exists for this link. If None, the link is meshless, so we continue to the next link.
                if self.unscaled_link_bounding_boxes[link_name][bbox_type] is None:
                    continue

                # Get the extent and transform.
                bb_data = self.unscaled_link_bounding_boxes[link_name][bbox_type]["oriented"]
                extent_in_bbox_frame = np.array(bb_data["extent"])
                bbox_to_link_origin = np.array(bb_data["transform"])

                # Get the link's pose in the base frame.
                if link == -1:
                    link_com_to_base_com = np.eye(4)
                else:
                    link_state = get_link_state(body_id, link, velocity=False)
                    link_com_to_world = utils.quat_pos_to_mat(
                        link_state.linkWorldPosition, link_state.linkWorldOrientation
                    )
                    link_com_to_base_com = np.dot(world_to_base_com, link_com_to_world)

                # Scale the bounding box in link origin frame. Here we create a transform that first puts the bounding
                # box's vertices into the link frame, and then scales them to match the scale applied to this object.
                # Note that once scaled, the vertices of the bounding box do not necessarily form a cuboid anymore but
                # instead a parallelepiped. This is not a problem because we later fit a bounding box to the points,
                # this time in the object's base link frame.
                scale_in_link_frame = np.diag(np.concatenate([self.scales_in_link_frame[link_name], [1]]))
                bbox_to_scaled_link_origin = np.dot(scale_in_link_frame, bbox_to_link_origin)

                # Account for the link vs. center-of-mass.
                dynamics_info = p.getDynamicsInfo(body_id, link)
                inertial_pos, inertial_orn = p.invertTransform(dynamics_info[3], dynamics_info[4])
                link_origin_to_link_com = utils.quat_pos_to_mat(inertial_pos, inertial_orn)

                # Compute the bounding box vertices in the base frame.
                bbox_to_link_com = np.dot(link_origin_to_link_com, bbox_to_scaled_link_origin)
                bbox_center_in_base_com = np.dot(link_com_to_base_com, bbox_to_link_com)
                vertices_in_base_com = np.array(list(itertools.product((1, -1), repeat=3))) * (extent_in_bbox_frame / 2)

                # Add the points to our collection of points.
                points.extend(trimesh.transformations.transform_points(vertices_in_base_com, bbox_center_in_base_com))
            elif fallback_to_aabb:
                # If no BB annotation is available, get the AABB for this link.
                aabb_center, aabb_extent = get_center_extent(body_id, link=link)
                aabb_vertices_in_world = aabb_center + np.array(list(itertools.product((1, -1), repeat=3))) * (
                    aabb_extent / 2
                )
                aabb_vertices_in_base_com = trimesh.transformations.transform_points(
                    aabb_vertices_in_world, world_to_base_com
                )
                points.extend(aabb_vertices_in_base_com)
            else:
                raise ValueError(
                    "Bounding box annotation missing for link: %s. Use fallback_to_aabb=True if you're okay with using "
                    "AABB as fallback." % link_name
                )

        if xy_aligned:
            # If the user requested an XY-plane aligned bbox, convert everything to that frame.
            # The desired frame is same as the base_com frame with its X/Y rotations removed.
            translate = trimesh.transformations.translation_from_matrix(base_com_to_world)

            # To find the rotation that this transform does around the Z axis, we rotate the [1, 0, 0] vector by it
            # and then take the arctangent of its projection onto the XY plane.
            rotated_X_axis = base_com_to_world[:3, 0]
            rotation_around_Z_axis = np.arctan2(rotated_X_axis[1], rotated_X_axis[0])
            xy_aligned_base_com_to_world = trimesh.transformations.compose_matrix(
                translate=translate, angles=[0, 0, rotation_around_Z_axis]
            )

            # We want to move our points to this frame as well.
            world_to_xy_aligned_base_com = trimesh.transformations.inverse_matrix(xy_aligned_base_com_to_world)
            base_com_to_xy_aligned_base_com = np.dot(world_to_xy_aligned_base_com, base_com_to_world)
            points = trimesh.transformations.transform_points(points, base_com_to_xy_aligned_base_com)

            # Finally update our desired frame.
            desired_frame_to_world = xy_aligned_base_com_to_world
        else:
            # Default desired frame is base CoM frame.
            desired_frame_to_world = base_com_to_world

        # TODO: Implement logic to allow tight bounding boxes that don't necessarily have to match the base frame.
        # All points are now in the desired frame: either the base CoM or the xy-plane-aligned base CoM.
        # Now fit a bounding box to all the points by taking the minimum/maximum in the desired frame.
        aabb_min_in_desired_frame = np.amin(points, axis=0)
        aabb_max_in_desired_frame = np.amax(points, axis=0)
        bbox_center_in_desired_frame = (aabb_min_in_desired_frame + aabb_max_in_desired_frame) / 2
        bbox_extent_in_desired_frame = aabb_max_in_desired_frame - aabb_min_in_desired_frame

        # Transform the center to the world frame.
        bbox_center_in_world = trimesh.transformations.transform_points(
            [bbox_center_in_desired_frame], desired_frame_to_world
        )[0]
        bbox_orn_in_world = Rotation.from_matrix(desired_frame_to_world[:3, :3]).as_quat()

        return bbox_center_in_world, bbox_orn_in_world, bbox_extent_in_desired_frame, bbox_center_in_desired_frame
