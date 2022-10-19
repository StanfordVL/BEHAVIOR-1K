import itertools
import json
import logging
import math
import os
import random
import time
import tempfile
import xml.etree.ElementTree as ET

import cv2
import networkx as nx
import numpy as np

import trimesh
from scipy.spatial.transform import Rotation

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.rotations import gf_quat_to_np_array

import igibson as ig
from igibson.objects.usd_object import USDObject
from igibson.utils.constants import AVERAGE_CATEGORY_SPECS, DEFAULT_JOINT_FRICTION, SPECIAL_JOINT_FRICTIONS, JointType
import igibson.utils.transform_utils as T
from igibson.utils.usd_utils import BoundingBoxAPI
from igibson.utils.asset_utils import decrypt_file
from igibson.utils.constants import PrimType
from igibson.macros import gm


class DatasetObject(USDObject):
    """
    DatasetObjects are instantiated from a USD file. It is an object that is assumed to come from an iG-supported
    dataset. These objects should contain additional metadata, including aggregate statistics across the
    object's category, e.g., avg dims, bounding boxes, masses, etc.
    """

    def __init__(
        self,
        prim_path,
        usd_path=None,
        name=None,
        category="object",
        model=None,
        class_id=None,
        uuid=None,
        scale=None,
        rendering_params=None,
        visible=True,
        fixed_base=False,
        visual_only=False,
        self_collisions=False,
        prim_type=PrimType.RIGID,
        load_config=None,
        abilities=None,

        bounding_box=None,
        fit_avg_dim_volume=False,
        in_rooms=None,
        texture_randomization=False,
        # scene_instance_folder=None,
        bddl_object_scope=None,
        # avg_obj_dims=None,
        # model_path=None,
        # joint_friction=None,
        # overwrite_inertial=True,
        # visualize_primitives=False,
        # merge_fixed_links=True,
        **kwargs,
    ):
        """
        @param prim_path: str, global path in the stage to this object
        @param usd_path: str, global path to the USD file to load. Either this or a combination of @category + @model
            will be used to infer the raw USD file location from which to load.
        @param name: Name for the object. Names need to be unique per scene. If no name is set, a name will be generated
            at the time the object is added to the scene, using the object's category.
        @param category: Category for the object. Defaults to "object".
        @param model: str or None, if @usd_path is not specified and @model is specified in conjunction with @category,
            the usd filepath will be inferred based on the set ig_dataset_path global variable, from the following
            location:

                {ig_dataset_path}/objects/{category}/{model}/usd/{model}.usd

        @param class_id: What class ID the object should be assigned in semantic segmentation rendering mode.
        @param uuid: Unique unsigned-integer identifier to assign to this object (max 8-numbers).
            If None is specified, then it will be auto-generated
        @param scale: float or 3-array, sets the scale for this object. A single number corresponds to uniform scaling
            along the x,y,z axes, whereas a 3-array specifies per-axis scaling.
        @param rendering_params: Any relevant rendering settings for this object.
        @param visible: bool, whether to render this object or not in the stage
        @param fixed_base: bool, whether to fix the base of this object or not
        visual_only (bool): Whether this object should be visual only (and not collide with any other objects)
        self_collisions (bool): Whether to enable self collisions for this object
        prim_type (PrimType): Which type of prim the object is, Valid options are: {PrimType.RIGID, PrimType.CLOTH}
        load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
            loading this prim at runtime.
        @param abilities: dict in the form of {ability: {param: value}} containing
            object abilities and parameters.
        :param fit_avg_dim_volume: whether to fit the object to have the same volume as the average dimension while keeping the aspect ratio
        :param fixed_base: whether this object has a fixed base to the world
        :param avg_obj_dims: average object dimension of this object
        :param joint_friction: joint friction for joints in this object
        :param in_rooms: which room(s) this object is in. It can be in more than one rooms if it sits at room boundary (e.g. doors)
        :param texture_randomization: whether to enable texture randomization
        :param overwrite_inertial: whether to overwrite the inertial frame of the original URDF using trimesh + density estimate
        # :param scene_instance_folder: scene instance folder to split and save sub-URDFs
        :param bddl_object_scope: bddl object scope name, e.g. chip.n.04_2
        :param visualize_primitives: whether to render geometric primitives
        :param joint_states: joint positions and velocities, keyed by body index and joint name, in the form of
            List[Dict[name, Tuple(position, velocity)]]
        :param merge_fixed_links: whether to merge fixed links when importing to pybullet
        kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
            for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
        """
        self.in_rooms = in_rooms
        self.texture_randomization = texture_randomization
        # self.overwrite_inertial = overwrite_inertial
        # self.scene_instance_folder = scene_instance_folder
        self.bddl_object_scope = bddl_object_scope
        # self.merge_fixed_links = merge_fixed_links

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

        # TODO: May need to come back to this if extra boxes are being visually loaded (e.g.: with the "simplified" objects)
        # if not visualize_primitives:
        #     for link in self.object_tree.findall("link"):
        #         for element in link:
        #             if element.tag == "visual" and len(element.findall(".//box")) > 0:
        #                 link.remove(element)

        # self.model_path = model_path
        # if self.model_path is None:
        #     self.model_path = os.path.dirname(self._usd_path)

        # # Change the mesh filenames to include the entire path
        # for mesh in self.object_tree.iter("mesh"):
        #     mesh.attrib["filename"] = os.path.join(self.model_path, mesh.attrib["filename"])

        # Info that will be filled in at runtime
        self.room_floor = None
        self.supporting_surfaces = None             # Dictionary mapping link names to surfaces represented by links

        # Make sure only one of bounding_box and scale are specified
        if bounding_box is not None and scale is not None:
            raise Exception("You cannot define both scale and bounding box size for an DatasetObject")

        # Add info to load config
        load_config = dict() if load_config is None else load_config
        load_config["bounding_box"] = bounding_box
        load_config["fit_avg_dim_volume"] = fit_avg_dim_volume

        # # If no bounding box, cannot compute dynamic properties from density
        # if self.bounding_box is None:
        #     self.overwrite_inertial = False

        # logging.info("Scale: " + np.array2string(scale))

        # # We need to know where the base_link origin is wrt. the bounding box
        # # center. That allows us to place the model correctly since the joint
        # # transformations given in the scene urdf are wrt. the bounding box
        # # center. We need to scale this offset as well.
        # self.scaled_bbxc_in_blf = -self.scale * base_link_offset

        # self.base_link_name = get_base_link_name(self.object_tree)
        # self.rename_urdf()

        # self.meta_links = {}
        # self.add_meta_links(meta_links)

        # self.scale_object()
        # self.remove_floating_joints(self.scene_instance_folder)
        # self.prepare_link_based_bounding_boxes()
        #
        # self.prepare_visual_mesh_to_material()

        # Infer the correct usd path to use
        if usd_path is None:
            assert model is not None, f"Either usd_path or model and category must be specified in order to create a" \
                                      f"DatasetObject!"
            usd_path = f"{ig.ig_dataset_path}/objects/{category}/{model}/usd/{model}.usd"

        # Post-process the usd path if we're generating a cloth object
        if prim_type == PrimType.CLOTH:
            assert usd_path.endswith(".usd"), f"usd_path [{usd_path}] is invalid."
            usd_path = usd_path[:-4] + "_cloth.usd"

        # Run super init
        super().__init__(
            prim_path=prim_path,
            usd_path=usd_path,
            name=name,
            category=category,
            class_id=class_id,
            uuid=uuid,
            scale=scale,
            rendering_params=rendering_params,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            self_collisions=self_collisions,
            prim_type=prim_type,
            load_config=load_config,
            abilities=abilities,
            **kwargs,
        )

    def load_supporting_surfaces(self):
        # Initialize dict of supporting surface info
        self.supporting_surfaces = {}

        # See if we have any height info -- if not, we can immediately return
        heights_info = self.heights_per_link
        if heights_info is None:
            return

        # TODO: Integrate images directly into usd file?
        # We loop over all the predicates and corresponding supported links in our heights info
        usd_dir = os.path.dirname(self._usd_path)
        for predicate, links in heights_info.items():
            height_maps = {}
            for link_name, heights in links.items():
                height_maps[link_name] = []
                for i, z_value in enumerate(heights):
                    # Get boolean birds-eye view xy-mask image for this surface
                    img_fname = os.path.join(usd_dir, "height_maps_per_link", predicate, link_name, f"{i}.png")
                    xy_map = cv2.imread(img_fname, 0)
                    # Add this map to the supporting surfaces for this link and predicate combination
                    height_maps[link_name].append((z_value, xy_map))
            # Add this heights map to the overall supporting surfaces
            self.supporting_surfaces[predicate] = height_maps

    def sample_orientation(self):
        """
        Samples an orientation in quaternion (x,y,z,w) form

        Returns:
            4-array: (x,y,z,w) sampled quaternion orientation for this object, based on self.orientations
        """
        if self.orientations is None:
            raise ValueError("No orientation probabilities set")
        if len(self.orientations) == 0:
            # Set default value
            chosen_orientation = np.array([0, 0, 0, 1.0])
        else:
            probabilities = [o["prob"] for o in self.orientations.values()]
            probabilities = np.array(probabilities) / np.sum(probabilities)
            chosen_orientation = np.array(np.random.choice(list(self.orientations.values()), p=probabilities)["rotation"])

        # Randomize yaw from -pi to pi
        rot_num = np.random.uniform(-1, 1)
        rot_matrix = np.array(
            [
                [math.cos(math.pi * rot_num), -math.sin(math.pi * rot_num), 0.0],
                [math.sin(math.pi * rot_num), math.cos(math.pi * rot_num), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rotated_quat = T.mat2quat(rot_matrix @ T.quat2mat(chosen_orientation))
        return rotated_quat

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

    def _initialize(self):
        # Run super method first
        super()._initialize()

        # Set the joint frictions based on category
        friction = SPECIAL_JOINT_FRICTIONS.get(self.category, DEFAULT_JOINT_FRICTION)
        for joint in self._joints.values():
            if joint.joint_type != JointType.JOINT_FIXED:
                joint.friction = friction

    def _load(self, simulator=None):
        if gm.USE_ENCRYPTED_ASSETS:
            # Create a temporary file to store the decrytped asset, load it, and then delete it.
            with tempfile.NamedTemporaryFile(suffix=".usd") as fp:
                original_usd_path = self._usd_path
                encrypted_filename = original_usd_path.replace(".usd", ".encrypted.usd")
                decrypt_file(encrypted_filename, decrypted_file=fp)
                self._usd_path = fp.name
                prim = super()._load(simulator=simulator)
                self._usd_path = original_usd_path
                return prim
        else:
            return super()._load(simulator=simulator)

    def _post_load(self):
        # We run this post loading first before any others because we're modifying the load config that will be used
        # downstream
        # Set the scale of this prim according to its bounding box
        if self._load_config["fit_avg_dim_volume"]:
            # By default, we assume scale does not change if no avg obj specs are given, otherwise, scale accordingly
            scale = np.ones(3)
            if self.avg_obj_dims is not None:
                # Find the average volume, and scale accordingly relative to the native volume based on the bbox
                volume_ratio = np.product(self.avg_obj_dims["size"]) / np.product(self.native_bbox)
                size_ratio = np.cbrt(volume_ratio)
                scale *= size_ratio
        # Otherwise, if manual bounding box is specified, scale based on ratio between that and the native bbox
        elif self._load_config["bounding_box"] is not None:
            scale = self._load_config["bounding_box"] / self.native_bbox
        else:
            scale = np.ones(3) if self._load_config["scale"] is None else self._load_config["scale"]

        # Set this scale in the load config -- it will automatically scale the object during self.initialize()
        self._load_config["scale"] = scale

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

        self.load_supporting_surfaces()

        # Run super last
        super()._post_load()

        if gm.USE_ENCRYPTED_ASSETS:
            # The loaded USD is from an already-deleted temporary file, so the asset paths for texture maps are wrong.
            # We explicitly provide the root_path to update all the asset paths: the asset paths are relative to the
            # original USD folder, i.e. <category>/<model>/usd.
            root_path = os.path.dirname(self._usd_path)
            for material in self.materials:
                material.shader_update_asset_paths_with_root_path(root_path)

        # Assign realistic density and mass based on average object category spec
        if self.avg_obj_dims is not None:
            v_ratio = (np.product(self.native_bbox) * np.product(self.scale)) / np.product(self.avg_obj_dims["size"])
            mass = self.avg_obj_dims["mass"] * v_ratio
            if self._prim_type == PrimType.RIGID:
                # Assume each link has the same density
                density = mass / self.volume
                for link in self._links.values():
                    # Overwrite the original, inaccurate mass value
                    link.mass = 0.0
                    link.density = density

            elif self._prim_type == PrimType.CLOTH:
                # Cloth cannot set density. Internally omni evenly distributes the mass to each particle
                self._links["base_link"].mass = mass

        # Lastly, after post loading (which includes loading / registering the links internally)
        # check for any metalinks. If there are any, we disable gravity and collisions for them
        for link in self._links.values():
            is_metalink = link.prim.GetAttribute("ig:is_metalink").Get() or False
            if is_metalink:
                # Make sure this link is only visual (i.e.: no collisions or gravity enabled)
                link.visual_only = True

    def _update_texture_change(self, object_state):
        """
        Update the texture based on the given object_state. E.g. if object_state is Frozen, update the diffuse color
        to match the frozen state. If object_state is None, update the diffuse color to the default value. It attempts
        to load the cached texture map named DIFFUSE/albedo_[STATE_NAME].png. If the cached texture map does not exist,
        it modifies the current albedo map by adding and scaling the values. See @self._update_albedo_value for details.

        Args:
            object_state (BooleanState or None): the object state that the diffuse color should match to
        """
        DEFAULT_ALBEDO_MAP_SUFFIX = frozenset({"DIFFUSE", "COMBINED", "albedo"})
        state_name = object_state.__class__.__name__ if object_state is not None else None
        for material in self.materials:
            texture_path = material.diffuse_texture
            assert texture_path is not None, f"DatasetObject [{self.prim_path}] has invalid diffuse texture map."

            # Get updated texture file path for state.
            texture_path_split = texture_path.split("/")
            filedir, filename = "/".join(texture_path_split[:-1]), texture_path_split[-1]
            assert filename[-4:] == ".png", f"Texture file {filename} does not end with .png"

            filename_split = filename[:-4].split("_")
            # Check all three file names for backward compatibility.
            if len(filename_split) > 0 and filename_split[-1] not in DEFAULT_ALBEDO_MAP_SUFFIX:
                filename_split.pop()
            target_texture_path = f"{filedir}/{'_'.join(filename_split)}"
            target_texture_path += f"_{state_name}.png" if state_name is not None else ".png"

            if os.path.exists(target_texture_path):
                # Since we are loading a pre-cached texture map, we need to reset the albedo value to the default
                self._update_albedo_value(None, material)
                if material.diffuse_texture != target_texture_path:
                    material.diffuse_texture = target_texture_path
            else:
                print(f"Warning: DatasetObject [{self.prim_path}] does not have texture map: "
                      f"[{target_texture_path}]. Falling back to directly updating albedo value.")
                self._update_albedo_value(object_state, material)

    def set_bbox_center_position_orientation(self, pos, orn):
        # TODO - check to make sure works
        rotated_offset = T.pose_transform([0, 0, 0], orn, self.scaled_bbox_center_in_base_frame, [0, 0, 0, 1])[0]
        # self.set_local_pose(pos + rotated_offset, orn)
        self.set_position_orientation(pos + rotated_offset, orn)

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

    # TODO: remove after split floors
    def set_room_floor(self, room_floor):
        assert self.category == "floors"
        self.room_floor = room_floor

    @property
    def native_bbox(self):
        """
        Get this object's native bounding box

        Returns:
            None or 3-array: (x,y,z) bounding box
        """
        assert "ig:nativeBB" in self.property_names, \
            f"This dataset object '{self.name}' is expected to have native_bbox specified, but found none!"
        return np.array(self.get_attribute(attr="ig:nativeBB"))

    @property
    def base_link_offset(self):
        """
        Get this object's native base link offset

        Returns:
            3-array: (x,y,z) base link offset
        """
        return np.array(self.get_attribute(attr="ig:offsetBaseLink"))

    @property
    def metadata(self):
        """
        Gets this object's metadata, if it exists

        Returns:
            None or dict: Nested dictionary of object's metadata if it exists, else None
        """
        return self.get_custom_data().get("metadata", None)

    @property
    def heights_per_link(self):
        """
        Gets this object's heights per link information, if it exists

        Returns:
            None or dict: Nested dictionary of object's height per link information if it exists, else None
        """
        return self.get_custom_data().get("heights_per_link", None)

    @property
    def orientations(self):
        """
        Returns:
            None or dict: Possible orientation information for this object, if it exists. Otherwise, returns None
        """
        metadata = self.metadata
        return None if metadata is None else metadata.get("orientations", None)

    @property
    def scaled_bbox_center_in_base_frame(self):
        """
        where the base_link origin is wrt. the bounding box center. This allows us to place the model correctly
        since the joint transformations given in the scene USD are wrt. the bounding box center.
        We need to scale this offset as well.

        Returns:
            3-array: (x,y,z) location of bounding box, with respet to the base link's coordinate frame
        """
        return -self.scale * self.base_link_offset

    @property
    def native_link_bboxes(self):
        """
        Returns:
             dict: Keyword-mapped native bounding boxes for each link of this object
        """
        return None if self.metadata is None else self.metadata.get("link_bounding_boxes", None)

    @property
    def scales_in_link_frame(self):
        """
        Returns:
        dict: Keyword-mapped relative scales for each link of this object
        """
        scales = {self.root_link.body_name: self.scale}

        # We iterate through all links in this object, and check for any joint prims that exist
        # We traverse manually this way instead of accessing the self._joints dictionary, because
        # the dictionary only includes articulated joints and not fixed joints!
        for link in self._links.values():
            for prim in link.prim.GetChildren():
                if "joint" in prim.GetTypeName().lower():
                    # Grab relevant joint information
                    parent_name = prim.GetProperty("physics:body0").GetTargets()[0].pathString.split("/")[-1]
                    child_name = prim.GetProperty("physics:body1").GetTargets()[0].pathString.split("/")[-1]
                    if parent_name in scales and child_name not in scales:
                        scale_in_parent_lf = scales[parent_name]
                        # The location of the joint frame is scaled using the scale in the parent frame
                        quat0 = gf_quat_to_np_array(prim.GetAttribute("physics:localRot0").Get())[[1, 2, 3, 0]]
                        quat1 = gf_quat_to_np_array(prim.GetAttribute("physics:localRot1").Get())[[1, 2, 3, 0]]

                        # Invert the child link relationship, and multiply the two rotations together to get the final rotation
                        local_ori = T.quat_multiply(quaternion1=T.quat_inverse(quat1), quaternion0=quat0)
                        jnt_frame_rot = T.quat2mat(local_ori)
                        scale_in_child_lf = np.absolute(jnt_frame_rot.T @ np.array(scale_in_parent_lf))
                        scales[child_name] = scale_in_child_lf

        return scales

    def get_base_aligned_bbox(self, link_name=None, visual=False, xy_aligned=False, fallback_to_aabb=False):
        """
        Get a bounding box for this object that's axis-aligned in the object's base frame.

        Returns:
            tuple:
                -
        :return:
        """
        bbox_type = "visual" if visual else "collision"

        # Get the base position transform.
        pos, orn = self.get_position_orientation()
        base_frame_to_world = T.pose2mat((pos, orn))

        # Compute the world-to-base frame transform.
        world_to_base_frame = trimesh.transformations.inverse_matrix(base_frame_to_world)

        # Grab the corners of all the different links' bounding boxes. We will later fit a bounding box to
        # this set of points to get our final, base-frame bounding box.
        points = []

        links = {link_name: self._links[link_name]} if link_name is not None else self._links
        for link_name, link in links.items():
            # If the link has no visual or collision meshes, we skip over it (based on the @visual flag)
            meshes = link.visual_meshes if visual else link.collision_meshes
            if len(meshes) == 0:
                continue

            # If the link has a bounding box annotation.
            if self.native_link_bboxes is not None and link_name in self.native_link_bboxes:
                # If a visual bounding box does not exist in the dictionary, try switching to collision.
                # We expect that every link has its collision bb annotated (or set to None if none exists).
                if bbox_type == "visual" and "visual" not in self.native_link_bboxes[link_name]:
                    logging.debug(
                        "Falling back to collision bbox for object %s link %s since no visual bbox exists.",
                        self.name,
                        link_name,
                    )
                    bbox_type = "collision"

                # Check if the annotation is still missing.
                if bbox_type not in self.native_link_bboxes[link_name]:
                    raise ValueError(
                        "Could not find %s bounding box for object %s link %s" % (bbox_type, self.name, link_name)
                    )

                # Check if a mesh exists for this link. If None, the link is meshless, so we continue to the next link.
                # TODO: Because of encoding, may need to be UsdTokens.none, not None
                if self.native_link_bboxes[link_name][bbox_type] is None:
                    continue

                # Get the extent and transform.
                bb_data = self.native_link_bboxes[link_name][bbox_type]["oriented"]
                extent_in_bbox_frame = np.array(bb_data["extent"])
                bbox_to_link_origin = np.array(bb_data["transform"])

                # # Get the link's pose in the base frame.
                # if link.name == self.root_link.name:
                #     link_com_to_base_com = np.eye(4)
                # else:
                #     link_com_to_world = T.pose2mat(link.get_position_orientation())
                #     link_com_to_base_com = world_to_base_com @ link_com_to_world
                link_frame_to_world = T.pose2mat(link.get_position_orientation())
                link_frame_to_base_frame = world_to_base_frame @ link_frame_to_world

                # Scale the bounding box in link origin frame. Here we create a transform that first puts the bounding
                # box's vertices into the link frame, and then scales them to match the scale applied to this object.
                # Note that once scaled, the vertices of the bounding box do not necessarily form a cuboid anymore but
                # instead a parallelepiped. This is not a problem because we later fit a bounding box to the points,
                # this time in the object's base link frame.
                scale_in_link_frame = np.diag(np.concatenate([self.scales_in_link_frame[link_name], [1]]))
                bbox_to_scaled_link_origin = np.dot(scale_in_link_frame, bbox_to_link_origin)

                # # Account for the link vs. center-of-mass.
                # # TODO: is this the same as local pose?
                # local_pos, local_ori = self.get_local_pose()
                # inertial_pos, inertial_ori = T.mat2pose(T.pose_inv(T.pose2mat((local_pos, local_ori))))
                # # dynamics_info = p.getDynamicsInfo(body_id, link)
                # # inertial_pos, inertial_orn = p.invertTransform(dynamics_info[3], dynamics_info[4])
                # link_origin_to_link_com = utils.quat_pos_to_mat(inertial_pos, inertial_ori)

                # Compute the bounding box vertices in the base frame.
                # bbox_to_link_com = np.dot(link_origin_to_link_com, bbox_to_scaled_link_origin)
                bbox_center_in_base_frame = np.dot(link_frame_to_base_frame, bbox_to_scaled_link_origin)
                vertices_in_base_frame = np.array(list(itertools.product((1, -1), repeat=3))) * (extent_in_bbox_frame / 2)

                # Add the points to our collection of points.
                points.extend(trimesh.transformations.transform_points(vertices_in_base_frame, bbox_center_in_base_frame))
            elif fallback_to_aabb:
                # If no BB annotation is available, get the AABB for this link.
                aabb_center, aabb_extent = BoundingBoxAPI.compute_center_extent(prim_path=link.prim_path)
                aabb_vertices_in_world = aabb_center + np.array(list(itertools.product((1, -1), repeat=3))) * (
                        aabb_extent / 2
                )
                aabb_vertices_in_base_frame = trimesh.transformations.transform_points(
                    aabb_vertices_in_world, world_to_base_frame
                )
                points.extend(aabb_vertices_in_base_frame)
            else:
                raise ValueError(
                    "Bounding box annotation missing for link: %s. Use fallback_to_aabb=True if you're okay with using "
                    "AABB as fallback." % link_name
                )

        if xy_aligned:
            # If the user requested an XY-plane aligned bbox, convert everything to that frame.
            # The desired frame is same as the base_com frame with its X/Y rotations removed.
            translate = trimesh.transformations.translation_from_matrix(base_frame_to_world)

            # To find the rotation that this transform does around the Z axis, we rotate the [1, 0, 0] vector by it
            # and then take the arctangent of its projection onto the XY plane.
            rotated_X_axis = base_frame_to_world[:3, 0]
            rotation_around_Z_axis = np.arctan2(rotated_X_axis[1], rotated_X_axis[0])
            xy_aligned_base_com_to_world = trimesh.transformations.compose_matrix(
                translate=translate, angles=[0, 0, rotation_around_Z_axis]
            )

            # We want to move our points to this frame as well.
            world_to_xy_aligned_base_com = trimesh.transformations.inverse_matrix(xy_aligned_base_com_to_world)
            base_com_to_xy_aligned_base_com = np.dot(world_to_xy_aligned_base_com, base_frame_to_world)
            points = trimesh.transformations.transform_points(points, base_com_to_xy_aligned_base_com)

            # Finally update our desired frame.
            desired_frame_to_world = xy_aligned_base_com_to_world
        else:
            # Default desired frame is base CoM frame.
            desired_frame_to_world = base_frame_to_world

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

    @property
    def avg_obj_dims(self):
        """
        Get the average object dimensions for this object, based on its category

        Returns:
            None or dict: Average object information based on its category
        """
        return AVERAGE_CATEGORY_SPECS.get(self.category, None)

    def _create_prim_with_same_kwargs(self, prim_path, name, load_config):
        # Add additional kwargs (fit_avg_dim_volume and bounding_box are already captured in load_config)
        return self.__class__(
            prim_path=prim_path,
            usd_path=self._usd_path,
            name=name,
            category=self.category,
            class_id=self.class_id,
            scale=self.scale,
            rendering_params=self.rendering_params,
            visible=self.visible,
            fixed_base=self.fixed_base,
            visual_only=self._visual_only,
            prim_type=self._prim_type,
            load_config=load_config,
            abilities=self._abilities,
            in_rooms=self.in_rooms,
            texture_randomization=self.texture_randomization,
            bddl_object_scope=self.bddl_object_scope,
        )

    # @property
    # def meta_links(self):
    #     """
    #     Any meta links that belong to this object, e.g., links to a heating or cooling source element position.
    #
    #     Returns:
    #         dict: Keyword-mapped meta link information, mapping meta link name to the (x,y,z) position of the link
    #     """
    #     meta_link_data = self.metadata.get("links", None)
    #     if meta_link_data is not None:
    #         for meta_link_name, link_info in meta_link_data.items():
    #             if link_info["geometry"] is not None:
    #                 # Objects with geometry actually need to be added into the URDF for collision purposes.
    #                 # These objects cannot be imported with fixed links.
    #                 self.merge_fixed_links = False
    #                 add_fixed_link(self.object_tree, meta_link_name, link_info)
    #             else:
    #                 # Otherwise, the "link" is just an offset, so we save its position.
    #                 self.meta_links[meta_link_name] = np.array(link_info["xyz"])
