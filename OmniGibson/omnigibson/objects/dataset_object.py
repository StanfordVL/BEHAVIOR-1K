import math
import os
import random
from enum import IntEnum

import torch as th

import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.macros import create_module_macros, gm
from omnigibson.prims.rigid_dynamic_prim import RigidDynamicPrim
from omnigibson.objects.usd_object import USDObject
from omnigibson.utils.asset_utils import (
    check_minimum_behavior_1k_assets_version,
    get_all_object_category_models,
    get_avg_category_specs,
    get_dataset_path,
)
from omnigibson.utils.constants import (
    DEFAULT_PRISMATIC_JOINT_FRICTION,
    DEFAULT_REVOLUTE_JOINT_FRICTION,
    DEFAULT_PRISMATIC_JOINT_DAMPING,
    DEFAULT_REVOLUTE_JOINT_DAMPING,
    JointType,
    PrimType,
)
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)


# Create settings for this module
m = create_module_macros(module_path=__file__)


class DatasetObject(USDObject):
    """
    DatasetObjects are instantiated from a USD file. It is an object that is assumed to come from an iG-supported
    dataset. These objects should contain additional metadata, including aggregate statistics across the
    object's category, e.g., avg dims, bounding boxes, masses, etc.
    """

    def __init__(
        self,
        name,
        relative_prim_path=None,
        category="object",
        model=None,
        dataset_name="behavior-1k-assets",
        scale=None,
        visible=True,
        fixed_base=False,
        visual_only=False,
        kinematic_only=None,
        self_collisions=False,
        prim_type=PrimType.RIGID,
        link_physics_materials=None,
        load_config=None,
        abilities=None,
        include_default_states=True,
        bounding_box=None,
        in_rooms=None,
        expected_file_hash=None,
        **kwargs,
    ):
        """
        Args:
            name (str): Name for the object. Names need to be unique per scene
            relative_prim_path (None or str): The path relative to its scene prim for this object. If not specified, it defaults to /<name>.
            category (str): Category for the object. Defaults to "object".
            model (None or str): If specified, this is used in conjunction with
                @category to infer the usd filepath to load for this object, which evaluates to the following:

                    {gm.DATA_PATH}/{dataset_name}/objects/{category}/{model}/usd/{model}.usd

                Otherwise, will randomly sample a model given @category
            dataset_name (str): Dataset to search for this object. Default is behavior-1k-assets, corresponding to the
                proprietary (encrypted) BEHAVIOR-1K dataset. Any other dataset names are searched for
                under the gm.DATA_PATH directory.
            scale (None or float or 3-array): if specified, sets either the uniform (float) or x,y,z (3-array) scale
                for this object. A single number corresponds to uniform scaling along the x,y,z axes, whereas a
                3-array specifies per-axis scaling.
            visible (bool): whether to render this object or not in the stage
            fixed_base (bool): whether to fix the base of this object or not
            visual_only (bool): Whether this object should be visual only (and not collide with any other objects)
            kinematic_only (None or bool): Whether this object should be kinematic only (and not get affected by any
                collisions). If None, then this value will be set to True if @fixed_base is True and some other criteria
                are satisfied (see object_base.py post_load function), else False.
            self_collisions (bool): Whether to enable self collisions for this object
            prim_type (PrimType): Which type of prim the object is, Valid options are: {PrimType.RIGID, PrimType.CLOTH}
            link_physics_materials (None or dict): If specified, dictionary mapping link name to kwargs used to generate
                a specific physical material for that link's collision meshes, where the kwargs are arguments directly
                passed into the isaacsim.core.api.materials.physics_material.PhysicsMaterial constructor, e.g.: "static_friction",
                "dynamic_friction", and "restitution"
            load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
                loading this prim at runtime.
            abilities (None or dict): If specified, manually adds specific object states to this object. It should be
                a dict in the form of {ability: {param: value}} containing object abilities and parameters to pass to
                the object state instance constructor.
            include_default_states (bool): whether to include the default object states from @get_default_states
            bounding_box (None or 3-array): If specified, will scale this object such that it fits in the desired
                (x,y,z) object-aligned bounding box. Note that EITHER @bounding_box or @scale may be specified
                -- not both!
            in_rooms (None or str or list): If specified, sets the room(s) that this object should belong to. Either
                a list of room type(s) or a single room type
            expected_file_hash (str): The expected hash of the file to load. This is used to check if the file has changed. None to disable check.
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
        load_config["dataset_name"] = dataset_name
        # All DatasetObjects should have xform properties pre-loaded
        # TODO: enable this after next dataset release
        load_config["xform_props_pre_loaded"] = False

        # Infer the correct usd path to use
        if model is None:
            available_models = get_all_object_category_models(category=category)
            assert len(available_models) > 0, f"No available models found for category {category}!"
            model = random.choice(available_models)

        self._model = model
        usd_path = self.get_usd_path(category=category, model=model, dataset_name=dataset_name)

        # Run super init
        super().__init__(
            relative_prim_path=relative_prim_path,
            usd_path=usd_path,
            encrypted=dataset_name == "behavior-1k-assets",
            name=name,
            category=category,
            scale=scale,
            visible=visible,
            fixed_base=fixed_base,
            visual_only=visual_only,
            kinematic_only=kinematic_only,
            self_collisions=self_collisions,
            prim_type=prim_type,
            include_default_states=include_default_states,
            link_physics_materials=link_physics_materials,
            load_config=load_config,
            abilities=abilities,
            expected_file_hash=expected_file_hash,
            **kwargs,
        )

    @classmethod
    def get_usd_path(cls, category, model, dataset_name="behavior-1k-assets"):
        """
        Grabs the USD path for a DatasetObject corresponding to @category and @model.

        NOTE: This is the unencrypted path, NOT the encrypted path

        Args:
            category (str): Category for the object
            model (str): Specific model ID of the object
            dataset_name (str): Dataset type, used to infer dataset directory to search for @category and @model

        Returns:
            str: Absolute filepath to the corresponding USD asset file
        """
        dataset_path = get_dataset_path(dataset_name)

        # Here, we assert the minimum version of the BEHAVIOR-1K dataset this is compatible with.
        # This requirement was placed by the fact that we switched from USDZ back to USD due to
        # speed and caching issues.
        if dataset_name == "behavior-1k-assets":
            assert check_minimum_behavior_1k_assets_version("3.7.2rc1"), (
                "This version of BEHAVIOR-1K requires at least version 3.7.2rc1 of the behavior-1k-assets dataset! \n"
                "Please update your dataset by deleting datasets/behavior-1k-assets and running the below command to reinstall. \n\n"
                'python -c "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=True)"'
            )

        return os.path.join(dataset_path, "objects", category, model, "usd", f"{model}.usd")

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
            chosen_orientation = th.tensor([0, 0, 0, 1.0])
        else:
            probabilities = [o["prob"] for o in self.orientations.values()]
            probabilities = th.tensor(probabilities, dtype=th.float32) / th.sum(probabilities)
            option = th.multinomial(probabilities, 1).item()
            chosen_orientation = th.tensor(list(self.orientations.values())[option]["rotation"])

        # Randomize yaw from -pi to pi
        rot_lo, rot_hi = -1, 1
        rot_num = (th.rand(1) * (rot_hi - rot_lo) + rot_lo).item()
        rot_matrix = th.tensor(
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
                    child_prim.GetAttribute("inputs:intensity").Set(gm.FORCE_LIGHT_INTENSITY)
                    child_prim.GetAttribute("inputs:normalize").Set(True)

                for child_child_prim in child_prim.GetChildren():
                    recursive_light_update(child_child_prim)

            recursive_light_update(self._prim)

        # Set the joint frictions based on joint type
        for joint in self._joints.values():
            if joint.joint_type == JointType.JOINT_PRISMATIC:
                joint.friction = DEFAULT_PRISMATIC_JOINT_FRICTION
            elif joint.joint_type == JointType.JOINT_REVOLUTE:
                joint.friction = DEFAULT_REVOLUTE_JOINT_FRICTION

    def _post_load(self):
        # If manual bounding box is specified, scale based on ratio between that and the native bbox
        if self._load_config["bounding_box"] is not None:
            scale = th.ones(3)
            valid_idxes = self.native_bbox > 1e-4
            scale[valid_idxes] = (
                th.tensor(self._load_config["bounding_box"])[valid_idxes] / self.native_bbox[valid_idxes]
            )
        elif self._load_config["scale"] is not None:
            scale = self._load_config["scale"]
            scale = scale if th.is_tensor(scale) else th.tensor(scale, dtype=th.float32)
        else:
            scale = th.ones(3)

        # Assert that the scale does not have too small dimensions
        assert th.all(th.abs(scale) > 1e-4), f"Scale of {self.name} is too small: {scale}"

        # Set this scale in the load config -- it will automatically scale the object during self.initialize()
        self._load_config["scale"] = scale

        # Run super last
        super()._post_load()

        # Get the average mass/density for this object category
        avg_specs = get_avg_category_specs()
        if self.category in avg_specs:
            category_mass = avg_specs[self.category]["mass"]
            category_density = avg_specs[self.category]["density"]
        else:
            log.warning(
                f"Category {self.category} not found in average object specs! Defaulting to unit mass or density."
            )
            category_mass = 1.0
            category_density = 1.0

        if self._prim_type == PrimType.RIGID:
            total_volume = sum(link.volume for link in self._links.values())
            for link in self._links.values():
                # If not a meta (virtual) link, set the density based on avg_obj_dims and a zero mass (ignored)
                if link.has_collision_meshes and isinstance(link, RigidDynamicPrim):
                    if gm.FORCE_CATEGORY_MASS:
                        # Each link should get the appropriate fraction of the category mass
                        # based on the link volume
                        link.mass = max(category_mass * (link.volume / total_volume), 1e-6)
                        link.density = 0.0
                    else:
                        link.mass = 0.0
                        link.density = category_density

            # If there exists a center of mass annotation, apply it now
            if self.prim.HasAttribute("ig:centerOfMass"):
                center_of_mass_in_object_frame = th.tensor(self.get_attribute(attr="ig:centerOfMass"))

                # Here we assume that the local frame of the object is the same as the local frame of the root link. We also do NOT need to apply a scale
                # since the center of mass is already in the local frame of the object and thus the unscaled local frame of the root link.
                self.root_link.center_of_mass = center_of_mass_in_object_frame

            # For all joints under dataset objects,
            # 1. we use "acceleration" drive type instead of "force" to properly account for link mass
            # 2. we set non-zero damping to simulate dynamic friction:
            #       the friction coefficient only accounts for static friction
            from omnigibson.utils.asset_conversion_utils import find_all_prim_children_with_type

            prismatic_joints = find_all_prim_children_with_type(prim_type="PhysicsPrismaticJoint", root_prim=self._prim)
            revolute_joints = find_all_prim_children_with_type(prim_type="PhysicsRevoluteJoint", root_prim=self._prim)
            for prismatic_joint in prismatic_joints:
                prismatic_joint.GetAttribute("drive:linear:physics:type").Set("acceleration")
                prismatic_joint.GetAttribute("drive:linear:physics:damping").Set(DEFAULT_PRISMATIC_JOINT_DAMPING)
                prismatic_joint.GetAttribute("drive:linear:physics:stiffness").Set(0.0)
                prismatic_joint.GetAttribute("drive:linear:physics:targetPosition").Set(0.0)
                prismatic_joint.GetAttribute("drive:linear:physics:targetVelocity").Set(0.0)
            for revolute_joint in revolute_joints:
                revolute_joint.GetAttribute("drive:angular:physics:type").Set("acceleration")
                revolute_joint.GetAttribute("drive:angular:physics:damping").Set(DEFAULT_REVOLUTE_JOINT_DAMPING)
                revolute_joint.GetAttribute("drive:angular:physics:stiffness").Set(0.0)
                revolute_joint.GetAttribute("drive:angular:physics:targetPosition").Set(0.0)
                revolute_joint.GetAttribute("drive:angular:physics:targetVelocity").Set(0.0)

        elif self._prim_type == PrimType.CLOTH:
            self.root_link.mass = category_mass if gm.FORCE_CATEGORY_MASS else category_density * self.root_link.volume

    def set_bbox_center_position_orientation(self, position=None, orientation=None):
        """
        Sets the center of the object's bounding box with respect to the world's frame.

        Args:
            position (None or 3-array): The desired global (x,y,z) position. None means it will not be changed
            orientation (None or 4-array): The desired global (x,y,z,w) quaternion orientation.
                None means it will not be changed
        """
        if orientation is None:
            orientation = self.get_position_orientation()[1]
        if position is not None:
            rotated_offset = T.pose_transform(
                th.tensor([0, 0, 0], dtype=th.float32),
                orientation,
                self.scaled_bbox_center_in_base_frame,
                th.tensor([0, 0, 0, 1], dtype=th.float32),
            )[0]
            position = position + rotated_offset
        self.set_position_orientation(position=position, orientation=orientation)

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
        assert (
            "ig:nativeBB" in self.property_names
        ), f"This dataset object '{self.name}' is expected to have native_bbox specified, but found none!"
        return th.tensor(self.get_attribute(attr="ig:nativeBB"))

    @property
    def base_link_offset(self):
        """
        Get this object's native base link offset

        Returns:
            3-array: (x,y,z) base link offset if it exists
        """
        return th.tensor(self.get_attribute(attr="ig:offsetBaseLink"))

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
                        quat0 = lazy.isaacsim.core.utils.rotations.gf_quat_to_np_array(
                            prim.GetAttribute("physics:localRot0").Get()
                        )[[1, 2, 3, 0]]
                        quat1 = lazy.isaacsim.core.utils.rotations.gf_quat_to_np_array(
                            prim.GetAttribute("physics:localRot1").Get()
                        )[[1, 2, 3, 0]]
                        # Invert the child link relationship, and multiply the two rotations together to get the final rotation
                        local_ori = T.quat_multiply(
                            quaternion1=T.quat_inverse(th.from_numpy(quat1)), quaternion0=th.from_numpy(quat0)
                        )
                        jnt_frame_rot = T.quat2mat(local_ori)
                        scale_in_child_lf = th.abs(jnt_frame_rot.T @ th.tensor(scale_in_parent_lf))
                        scales[child_name] = scale_in_child_lf

        return scales

    def _create_prim_with_same_kwargs(self, relative_prim_path, name, load_config):
        # Add additional kwargs (bounding_box is already captured in load_config)
        return self.__class__(
            relative_prim_path=relative_prim_path,
            name=name,
            category=self.category,
            scale=self.scale,
            visible=self.visible,
            fixed_base=self.fixed_base,
            visual_only=self._visual_only,
            prim_type=self._prim_type,
            load_config=load_config,
            abilities=self._abilities,
            in_rooms=self.in_rooms,
        )
