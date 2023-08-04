import itertools
import math
import os
import tempfile
import stat
import cv2
import numpy as np

import trimesh
from scipy.spatial.transform import Rotation
from omni.isaac.core.utils.rotations import gf_quat_to_np_array

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.objects.usd_object import USDObject
from omnigibson.utils.constants import AVERAGE_CATEGORY_SPECS, DEFAULT_JOINT_FRICTION, SPECIAL_JOINT_FRICTIONS, JointType
import omnigibson.utils.transform_utils as T
from omnigibson.utils.usd_utils import BoundingBoxAPI
from omnigibson.utils.asset_utils import decrypt_file, get_all_object_category_models
from omnigibson.utils.constants import PrimType
from omnigibson.macros import gm, create_module_macros
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)


# Create settings for this module
m = create_module_macros(module_path=__file__)

# A lower bound is needed in order to consistently trigger contacts
m.MIN_OBJ_MASS = 0.4


class DatasetObject(USDObject):
    """
    DatasetObjects are instantiated from a USD file. It is an object that is assumed to come from an iG-supported
    dataset. These objects should contain additional metadata, including aggregate statistics across the
    object's category, e.g., avg dims, bounding boxes, masses, etc.
    """

    def __init__(
        self,
        name,
        prim_path=None,
        category="object",
        model=None,
        class_id=None,
        uuid=None,
        scale=None,
        visible=True,
        fixed_base=False,
        visual_only=False,
        self_collisions=False,
        prim_type=PrimType.RIGID,
        load_config=None,
        abilities=None,
        include_default_states=True,
        bounding_box=None,
        fit_avg_dim_volume=False,
        in_rooms=None,
        **kwargs,
    ):
        """
        Args:
            name (str): Name for the object. Names need to be unique per scene
            prim_path (None or str): global path in the stage to this object. If not specified, will automatically be
                created at /World/<name>
            category (str): Category for the object. Defaults to "object".
            model (None or str): If specified, this is used in conjunction with
                @category to infer the usd filepath to load for this object, which evaluates to the following:

                    {og_dataset_path}/objects/{category}/{model}/usd/{model}.usd

                Otherwise, will randomly sample a model given @category

            class_id (None or int): What class ID the object should be assigned in semantic segmentation rendering mode.
                If None, the ID will be inferred from this object's category.
            uuid (None or int): Unique unsigned-integer identifier to assign to this object (max 8-numbers).
                If None is specified, then it will be auto-generated
            scale (None or float or 3-array): if specified, sets either the uniform (float) or x,y,z (3-array) scale
                for this object. A single number corresponds to uniform scaling along the x,y,z axes, whereas a
                3-array specifies per-axis scaling.
            visible (bool): whether to render this object or not in the stage
            fixed_base (bool): whether to fix the base of this object or not
            visual_only (bool): Whether this object should be visual only (and not collide with any other objects)
            self_collisions (bool): Whether to enable self collisions for this object
            prim_type (PrimType): Which type of prim the object is, Valid options are: {PrimType.RIGID, PrimType.CLOTH}
            load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
                loading this prim at runtime.
            abilities (None or dict): If specified, manually adds specific object states to this object. It should be
                a dict in the form of {ability: {param: value}} containing object abilities and parameters to pass to
                the object state instance constructor.
            include_default_states (bool): whether to include the default object states from @get_default_states
            bounding_box (None or 3-array): If specified, will scale this object such that it fits in the desired
                (x,y,z) object-aligned bounding box. Note that EITHER @bounding_box or @scale may be specified
                -- not both!
            fit_avg_dim_volume (bool): whether to fit the object to have the same volume as the average dimension
                while keeping the aspect ratio. Note that if this is set, it will override both @scale and @bounding_box
            in_rooms (None or str or list): If specified, sets the room(s) that this object should belong to. Either
                a list of room type(s) or a single room type
            kwargs (dict): Additional keyword arguments that are used for other super() calls from subclasses, allowing
                for flexible compositions of various object subclasses (e.g.: Robot is USDObject + ControllableObject).
        """
        # Store variables
        if isinstance(in_rooms, str):
            assert "," not in in_rooms
        self._in_rooms = [in_rooms] if isinstance(in_rooms, str) else in_rooms

        # Make sure only one of bounding_box and scale are specified
        if bounding_box is not None and scale is not None:
            raise Exception("You cannot define both scale and bounding box size for an DatasetObject")

        # Add info to load config
        load_config = dict() if load_config is None else load_config
        load_config["bounding_box"] = bounding_box
        load_config["fit_avg_dim_volume"] = fit_avg_dim_volume

        # Infer the correct usd path to use
        if model is None:
            available_models = get_all_object_category_models(category=category)
            assert len(available_models) > 0, f"No available models found for category {category}!"
            model = np.random.choice(available_models)
        self._model = model
        usd_path = self.get_usd_path(category=category, model=model)

        # Run super init
        super().__init__(
            prim_path=prim_path,
            usd_path=usd_path,
            name=name,
            category=category,
            class_id=class_id,
            uuid=uuid,
            scale=scale,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            self_collisions=self_collisions,
            prim_type=prim_type,
            include_default_states=include_default_states,
            load_config=load_config,
            abilities=abilities,
            **kwargs,
        )

    @classmethod
    def get_usd_path(cls, category, model):
        """
        Grabs the USD path for a DatasetObject corresponding to @category and @model.

        NOTE: This is the unencrypted path, NOT the encrypted path

        Args:
            category (str): Category for the object
            model (str): Specific model ID of the object

        Returns:
            str: Absolute filepath to the corresponding USD asset file
        """
        return os.path.join(gm.DATASET_PATH, "objects", category, model, "usd", f"{model}.usd")

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

    def _initialize(self):
        # Run super method first
        super()._initialize()

        # Apply any forced light intensity updates.
        if gm.FORCE_LIGHT_INTENSITY is not None:
            def recursive_light_update(child_prim):
                if "Light" in child_prim.GetPrimTypeInfo().GetTypeName():
                    child_prim.GetAttribute("intensity").Set(gm.FORCE_LIGHT_INTENSITY)

                for child_child_prim in child_prim.GetChildren():
                    recursive_light_update(child_child_prim)

            recursive_light_update(self._prim)

        # Apply any forced roughness updates
        for material in self.materials:
            if ("reflection_roughness_texture_influence" in material.shader_input_names and
                "reflection_roughness_constant" in material.shader_input_names):
                material.reflection_roughness_texture_influence = 0.0
                material.reflection_roughness_constant = gm.FORCE_ROUGHNESS

        # Set the joint frictions based on category
        friction = SPECIAL_JOINT_FRICTIONS.get(self.category, DEFAULT_JOINT_FRICTION)
        for joint in self._joints.values():
            if joint.joint_type != JointType.JOINT_FIXED:
                joint.friction = friction

    def _load(self):
        # Create a temporary file to store the decrytped asset, load it, and then delete it.
        original_usd_path = self._usd_path
        encrypted_filename = original_usd_path.replace(".usd", ".encrypted.usd")
        decrypted_fd, decrypted_filename = tempfile.mkstemp(os.path.basename(original_usd_path), dir=og.tempdir)
        decrypt_file(encrypted_filename, decrypted_filename)
        self._usd_path = decrypted_filename
        prim = super()._load()
        os.close(decrypted_fd)
        # On Windows, Isaac Sim won't let go of the file until the prim is removed, so we can't delete it.
        if os.name == "posix":
            os.remove(decrypted_filename)
        self._usd_path = original_usd_path
        return prim

    def _post_load(self):
        # We run this post loading first before any others because we're modifying the load config that will be used
        # downstream
        # Set the scale of this prim according to its bounding box
        if self._load_config["fit_avg_dim_volume"]:
            # By default, we assume scale does not change if no avg obj specs are given, otherwise, scale accordingly
            scale = np.ones(3)
            if self.avg_obj_dims is not None and self.avg_obj_dims["size"] is not None:
                # Find the average volume, and scale accordingly relative to the native volume based on the bbox
                volume_ratio = np.product(self.avg_obj_dims["size"]) / np.product(self.native_bbox)
                size_ratio = np.cbrt(volume_ratio)
                scale *= size_ratio
        # Otherwise, if manual bounding box is specified, scale based on ratio between that and the native bbox
        elif self._load_config["bounding_box"] is not None:
            scale = np.ones(3)
            valid_idxes = self.native_bbox > 1e-4
            scale[valid_idxes] = np.array(self._load_config["bounding_box"])[valid_idxes] / self.native_bbox[valid_idxes]
        else:
            scale = np.ones(3) if self._load_config["scale"] is None else self._load_config["scale"]

        # Assert that the scale does not have too small dimensions
        assert np.all(scale > 1e-4), f"Scale of {self.name} is too small: {scale}"

        # Set this scale in the load config -- it will automatically scale the object during self.initialize()
        self._load_config["scale"] = scale

        # Run super last
        super()._post_load()

        # The loaded USD is from an already-deleted temporary file, so the asset paths for texture maps are wrong.
        # We explicitly provide the root_path to update all the asset paths: the asset paths are relative to the
        # original USD folder, i.e. <category>/<model>/usd.
        root_path = os.path.dirname(self._usd_path)
        for material in self.materials:
            material.shader_update_asset_paths_with_root_path(root_path)

        # Assign realistic density and mass based on average object category spec
        if self.avg_obj_dims is not None and self.avg_obj_dims["size"] is not None and self.avg_obj_dims["mass"] is not None:
            # Assume each link has the same density
            v_ratio = (np.product(self.native_bbox) * np.product(self.scale)) / np.product(self.avg_obj_dims["size"])
            mass = self.avg_obj_dims["mass"] * v_ratio
            if self._prim_type == PrimType.RIGID:
                density = mass / self.volume
                for link in self._links.values():
                    # If not a meta (virtual) link, set the density based on avg_obj_dims and a zero mass (ignored)
                    if link.has_collision_meshes:
                        link.mass = 0.0
                        link.density = density

            elif self._prim_type == PrimType.CLOTH:
                # Cloth cannot set density. Internally omni evenly distributes the mass to each particle
                mass = self.avg_obj_dims["mass"] * v_ratio
                self.root_link.mass = mass

    def _update_texture_change(self, object_state):
        """
        Update the texture based on the given object_state. E.g. if object_state is Frozen, update the diffuse color
        to match the frozen state. If object_state is None, update the diffuse color to the default value. It attempts
        to load the cached texture map named DIFFUSE/albedo_[STATE_NAME].png. If the cached texture map does not exist,
        it modifies the current albedo map by adding and scaling the values. See @self._update_albedo_value for details.

        Args:
            object_state (BooleanStateMixin or None): the object state that the diffuse color should match to
        """
        # TODO: uncomment these once our dataset has the object state-conditioned texture maps
        # DEFAULT_ALBEDO_MAP_SUFFIX = frozenset({"DIFFUSE", "COMBINED", "albedo"})
        # state_name = object_state.__class__.__name__ if object_state is not None else None
        for material in self.materials:
            # texture_path = material.diffuse_texture
            # assert texture_path is not None, f"DatasetObject [{self.prim_path}] has invalid diffuse texture map."
            #
            # # Get updated texture file path for state.
            # texture_path_split = texture_path.split("/")
            # filedir, filename = "/".join(texture_path_split[:-1]), texture_path_split[-1]
            # assert filename[-4:] == ".png", f"Texture file {filename} does not end with .png"
            #
            # filename_split = filename[:-4].split("_")
            # # Check all three file names for backward compatibility.
            # if len(filename_split) > 0 and filename_split[-1] not in DEFAULT_ALBEDO_MAP_SUFFIX:
            #     filename_split.pop()
            # target_texture_path = f"{filedir}/{'_'.join(filename_split)}"
            # target_texture_path += f"_{state_name}.png" if state_name is not None else ".png"
            #
            # if os.path.exists(target_texture_path):
            #     # Since we are loading a pre-cached texture map, we need to reset the albedo value to the default
            #     self._update_albedo_value(None, material)
            #     if material.diffuse_texture != target_texture_path:
            #         material.diffuse_texture = target_texture_path
            # else:
            #     print(f"Warning: DatasetObject [{self.prim_path}] does not have texture map: "
            #           f"[{target_texture_path}]. Falling back to directly updating albedo value.")
            self._update_albedo_value(object_state, material)

    def set_bbox_center_position_orientation(self, position=None, orientation=None):
        """
        Sets the center of the object's bounding box with respect to the world's frame.

        Args:
            position (None or 3-array): The desired global (x,y,z) position. None means it will not be changed
            orientation (None or 4-array): The desired global (x,y,z,w) quaternion orientation.
                None means it will not be changed
        """
        if orientation is None:
            orientation = self.get_orientation()
        if position is not None:
            rotated_offset = T.pose_transform([0, 0, 0], orientation,
                                              self.scaled_bbox_center_in_base_frame, [0, 0, 0, 1])[0]
            position = position + rotated_offset
        self.set_position_orientation(position, orientation)

    @property
    def model(self):
        """
        Returns:
            str: Unique model ID for this object
        """
        return self._model

    @property
    def in_rooms(self):
        """
        Returns:
            None or list of str: If specified, room(s) that this object should belong to
        """
        return self._in_rooms

    @in_rooms.setter
    def in_rooms(self, rooms):
        """
        Sets which room(s) this object should belong to. If no rooms, then should set to None

        Args:
            rooms (None or list of str): If specified, the room(s) this object should belong to
        """
        # Store the value to the internal variable and also update the init kwargs accordingly
        self._init_info["args"]["in_rooms"] = rooms
        self._in_rooms = rooms

    @property
    def native_bbox(self):
        """
        Get this object's native bounding box

        Returns:
            3-array: (x,y,z) bounding box
        """
        assert "ig:nativeBB" in self.property_names, \
            f"This dataset object '{self.name}' is expected to have native_bbox specified, but found none!"
        return np.array(self.get_attribute(attr="ig:nativeBB"))

    @property
    def base_link_offset(self):
        """
        Get this object's native base link offset

        Returns:
            3-array: (x,y,z) base link offset if it exists
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
    def orientations(self):
        """
        Returns:
            None or dict: Possible orientation information for this object, if it exists. Otherwise, returns None
        """
        metadata = self.metadata
        return None if metadata is None else metadata.get("orientations", None)

    @property
    def scale(self):
        # Just super call
        return super().scale

    @scale.setter
    def scale(self, scale):
        # call super first
        # A bit esoteric -- see https://gist.github.com/Susensio/979259559e2bebcd0273f1a95d7c1e79
        super(DatasetObject, type(self)).scale.fset(self, scale)

        # Remove bounding_box from scale if it's in our args
        if "bounding_box" in self._init_info["args"]:
            self._init_info["args"].pop("bounding_box")

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

    def get_base_aligned_bbox(self, link_name=None, visual=False, xy_aligned=False, fallback_to_aabb=False, link_bbox_type="axis_aligned"):
        """
        Get a bounding box for this object that's axis-aligned in the object's base frame.

        Args:
            link_name (None or str): If specified, only get the bbox for the given link
            visual (bool): Whether to aggregate the bounding boxes from the visual meshes. Otherwise, will use
                collision meshes
            xy_aligned (bool): Whether to align the bounding box to the global XY-plane
            fallback_to_aabb (bool): If set and a link's info is not found, the (global-frame) AABB will be
                dynamically computed directly from omniverse
            link_bbox_type (str): Which type of link bbox to use, "axis_aligned" means the bounding box is axis-aligned
                to the link frame, "oriented" means the bounding box has the minimum volume

        Returns:
            4-tuple:
                - 3-array: (x,y,z) bbox center position in world frame
                - 3-array: (x,y,z,w) bbox quaternion orientation in world frame
                - 3-array: (x,y,z) bbox extent in desired frame
                - 3-array: (x,y,z) bbox center in desired frame
        """
        assert self.prim_type == PrimType.RIGID, "get_base_aligned_bbox is only supported for rigid objects."
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
                # Check if the annotation is still missing.
                if bbox_type not in self.native_link_bboxes[link_name]:
                    raise ValueError(f"Could not find {bbox_type} bounding box for object {self.name} link {link_name}")

                # Get the extent and transform.
                bb_data = self.native_link_bboxes[link_name][bbox_type][link_bbox_type]
                extent_in_bbox_frame = np.array(bb_data["extent"])
                bbox_to_link_origin = np.array(bb_data["transform"])

                # # Get the link's pose in the base frame.
                link_frame_to_world = T.pose2mat(link.get_position_orientation())
                link_frame_to_base_frame = world_to_base_frame @ link_frame_to_world

                # Scale the bounding box in link origin frame. Here we create a transform that first puts the bounding
                # box's vertices into the link frame, and then scales them to match the scale applied to this object.
                # Note that once scaled, the vertices of the bounding box do not necessarily form a cuboid anymore but
                # instead a parallelepiped. This is not a problem because we later fit a bounding box to the points,
                # this time in the object's base link frame.
                scale_in_link_frame = np.diag(np.concatenate([self.scales_in_link_frame[link_name], [1]]))
                bbox_to_scaled_link_origin = np.dot(scale_in_link_frame, bbox_to_link_origin)

                # Compute the bounding box vertices in the base frame.
                # bbox_to_link_com = np.dot(link_origin_to_link_com, bbox_to_scaled_link_origin)
                bbox_center_in_base_frame = np.dot(link_frame_to_base_frame, bbox_to_scaled_link_origin)
                vertices_in_base_frame = np.array(list(itertools.product((1, -1), repeat=3))) * (extent_in_bbox_frame / 2)

                # Add the points to our collection of points.
                points.extend(trimesh.transformations.transform_points(vertices_in_base_frame, bbox_center_in_base_frame))
            elif fallback_to_aabb:
                # If we're visual and the mesh is not visible, there is no fallback so continue
                if bbox_type == "visual" and not np.all(tuple(mesh.visible for mesh in meshes.values())):
                    continue
                # If no BB annotation is available, get the AABB for this link.
                aabb_center, aabb_extent = BoundingBoxAPI.compute_center_extent(prim=link)
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
            name=name,
            category=self.category,
            class_id=self.class_id,
            scale=self.scale,
            visible=self.visible,
            fixed_base=self.fixed_base,
            visual_only=self._visual_only,
            prim_type=self._prim_type,
            load_config=load_config,
            abilities=self._abilities,
            in_rooms=self.in_rooms,
        )
