import numpy as np

from omni.isaac.core.utils.rotations import gf_quat_to_np_array
from omni.isaac.core.utils.transformations import tf_matrix_from_pose
from omni.isaac.core.utils.types import DOFInfo
from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.core.utils.stage import get_current_stage
from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema
import omni

from omni.isaac.core.utils.prims import get_prim_property, set_prim_property, \
    get_prim_parent, get_prim_at_path

import omnigibson as og
from omnigibson.prims.cloth_prim import ClothPrim
from omnigibson.prims.joint_prim import JointPrim
from omnigibson.prims.rigid_prim import RigidPrim
from omnigibson.prims.xform_prim import XFormPrim
from omnigibson.utils.constants import PrimType, GEOM_TYPES
from omnigibson.utils.ui_utils import suppress_omni_log
from omnigibson.macros import gm


class EntityPrim(XFormPrim):
    """
    Provides high level functions to deal with an articulation prim and its attributes/ properties. Note that this
    type of prim cannot be created from scratch, and assumes there is already a pre-existing prim tree that should
    be converted into an articulation!

    Args:
        prim_path (str): prim path of the Prim to encapsulate or create.
        name (str): Name for the object. Names need to be unique per scene.
        load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
            loading this prim at runtime. Note that by default, this assumes an articulation already exists (i.e.:
            load() will raise NotImplementedError)! Subclasses must implement _load() for this prim to be able to be
            dynamically loaded after this class is created.

            visual_only (None or bool): If specified, whether this prim should include collisions or not.
                    Default is True.
        """

    def __init__(
        self,
        prim_path,
        name,
        load_config=None,
    ):
        # Other values that will be filled in at runtime
        self._dc = None                         # Dynamics control interface
        self._handle = None                     # Handle to this articulation
        self._root_handle = None                # Handle to the root rigid body of this articulation
        self._root_link_name = None             # Name of the root link
        self._dofs_infos = None
        self._n_dof = None                      # dof with dynamic control
        self._links = None
        self._joints = None
        self._materials = None
        self._visual_only = None

        # This needs to be initialized to be used for _load() of PrimitiveObject
        self._prim_type = load_config["prim_type"] if "prim_type" in load_config else PrimType.RIGID

        # Run super init
        super().__init__(
            prim_path=prim_path,
            name=name,
            load_config=load_config,
        )

    def _initialize(self):
        # Run super method
        super()._initialize()

        # Force populate inputs and outputs of the shaders of all materials
        # We suppress errors from omni.hydra if we're using encrypted assets, because we're loading from tmp location,
        # not the original location
        with suppress_omni_log(channels=["omni.hydra"] if gm.USE_ENCRYPTED_ASSETS else []):
            for material in self.materials:
                material.shader_force_populate(render=False)

        # Initialize all the links
        # This must happen BEFORE the handle is generated for this prim, because things changing in the RigidPrims may
        # cause the handle to change!
        for link in self._links.values():
            link.initialize()

        # Get dynamic control info
        self._dc = _dynamic_control.acquire_dynamic_control_interface()

        # Update joint information
        self.update_joints()

    def _load(self):
        # By default, this prim cannot be instantiated from scratch!
        raise NotImplementedError("By default, an entity prim cannot be created from scratch.")

    def _post_load(self):
        # Setup links info FIRST before running any other post loading behavior
        self.update_links()

        # Set visual only flag
        # This automatically handles setting collisions / gravity appropriately per-link
        self.visual_only = self._load_config["visual_only"] if \
            "visual_only" in self._load_config and self._load_config["visual_only"] is not None else False

        if self._prim_type == PrimType.CLOTH:
            assert not self._visual_only, "Cloth cannot be visual-only."
            assert len(self._links) == 1, f"Cloth entity prim can only have one link; got: {len(self._links)}"
            if gm.AG_CLOTH:
                self.create_attachment_point_link()

        # Disable any requested collision pairs
        for a_name, b_name in self.disabled_collision_pairs:
            link_a, link_b = self._links[a_name], self._links[b_name]
            link_a.add_filtered_collision_pair(prim=link_b)

        # Run super
        super()._post_load()

        # Cache material information
        materials = set()
        material_paths = set()
        for link in self._links.values():
            xforms = [link] + list(link.visual_meshes.values()) if self.prim_type == PrimType.RIGID else [link]
            for xform in xforms:
                if xform.has_material():
                    mat_path = xform.material.prim_path
                    if mat_path not in material_paths:
                        materials.add(xform.material)
                        material_paths.add(mat_path)

        self._materials = materials

    def update_links(self):
        """
        Helper function to refresh owned joints. Useful for synchronizing internal data if
        additional bodies are added manually
        """
        # Make sure to clean up all pre-existing names for all links
        if self._links is not None:
            for link in self._links.values():
                link.remove_names()

        # We iterate over all children of this object's prim,
        # and grab any that are presumed to be rigid bodies (i.e.: other Xforms)
        self._links = dict()
        joint_children = set()
        for prim in self._prim.GetChildren():
            link = None
            link_name = prim.GetName()
            if self._prim_type == PrimType.RIGID and prim.GetPrimTypeInfo().GetTypeName() == "Xform":
                # For rigid body object, process prims that are Xforms (e.g. rigid links)
                link = RigidPrim(
                    prim_path=prim.GetPrimPath().__str__(),
                    name=f"{self._name}:{link_name}",
                )
                # Also iterate through all children to infer joints and determine the children of those joints
                # We will use this info to infer which link is the base link!
                for child_prim in prim.GetChildren():
                    if "joint" in child_prim.GetPrimTypeInfo().GetTypeName().lower():
                        # Store the child target of this joint
                        relationships = {r.GetName(): r for r in child_prim.GetRelationships()}
                        # Only record if this is NOT a fixed link tying us to the world (i.e.: no target for body0)
                        if len(relationships["physics:body0"].GetTargets()) > 0:
                            joint_children.add(relationships["physics:body1"].GetTargets()[0].pathString.split("/")[-1])

            if self._prim_type == PrimType.CLOTH and prim.GetPrimTypeInfo().GetTypeName() in GEOM_TYPES:
                # For cloth object, process prims that belong to any of the GEOM_TYPES (e.g. Cube, Mesh, etc)
                link = ClothPrim(
                    prim_path=prim.GetPrimPath().__str__(),
                    name=f"{self._name}:{link_name}",
                )

            if link is not None:
                self._links[link_name] = link

        # Infer the correct root link name -- this corresponds to whatever link does not have any joint existing
        # in the children joints
        valid_root_links = list(set(self._links.keys()) - joint_children)

        # TODO: Uncomment safety check here after we figure out how to handle legacy multi-bodied assets like bed with pillow
        # assert len(valid_root_links) == 1, f"Only a single root link should have been found for this entity prim, " \
        #                                    f"but found multiple instead: {valid_root_links}"
        self._root_link_name = valid_root_links[0] if len(valid_root_links) == 1 else "base_link"

    def update_joints(self):
        """
        Helper function to refresh owned joints. Useful for synchronizing internal data if
        additional bodies are added manually
        """
        # Make sure to clean up all pre-existing names for all joints
        if self._joints is not None:
            for joint in self._joints.values():
                joint.remove_names()

        # Initialize joints dictionary
        self._joints = dict()
        self.update_handles()

        # Handle case separately based on whether the handle is valid (i.e.: whether we are actually articulated or not)
        if (not self.kinematic_only) and self._handle is not None:
            root_prim = get_prim_at_path(self._dc.get_rigid_body_path(self._root_handle))
            n_dof = self._dc.get_articulation_dof_count(self._handle)

            # Additionally grab DOF info if we have non-fixed joints
            if n_dof > 0:
                self._dofs_infos = dict()
                # Grab DOF info
                for index in range(n_dof):
                    dof_handle = self._dc.get_articulation_dof(self._handle, index)
                    dof_name = self._dc.get_dof_name(dof_handle)
                    # add dof to list
                    prim_path = self._dc.get_dof_path(dof_handle)
                    self._dofs_infos[dof_name] = DOFInfo(prim_path=prim_path, handle=dof_handle, prim=self.prim,
                                                         index=index)

                for i in range(self._dc.get_articulation_joint_count(self._handle)):
                    joint_handle = self._dc.get_articulation_joint(self._handle, i)
                    joint_name = self._dc.get_joint_name(joint_handle)
                    joint_path = self._dc.get_joint_path(joint_handle)
                    joint_prim = get_prim_at_path(joint_path)
                    # Only add the joint if it's not fixed (i.e.: it has DOFs > 0)
                    if self._dc.get_joint_dof_count(joint_handle) > 0:
                        joint = JointPrim(
                            prim_path=joint_path,
                            name=f"{self._name}:joint_{joint_name}",
                            articulation=self._handle,
                        )
                        joint.initialize()
                        self._joints[joint_name] = joint
        else:
            # TODO: May need to extend to clusters of rigid bodies, that aren't exactly joined
            # We assume this object contains a single rigid body
            body_path = f"{self._prim_path}/{self.root_link_name}"
            root_prim = get_prim_at_path(body_path)
            n_dof = 0

        # Make sure root prim stored is the same as the one found during initialization
        assert self.root_prim == root_prim, \
            f"Mismatch in root prims! Original was {self.root_prim.GetPrimPath()}, " \
            f"initialized is {root_prim.GetPrimPath()}!"

        # Store values internally
        self._n_dof = n_dof

    @property
    def prim_type(self):
        """
        Returns:
            str: Type of this entity prim, one of omnigibson.utils.constants.PrimType
        """
        return self._prim_type

    @property
    def articulated(self):
        """
        Returns:
             bool: Whether this prim is articulated or not
        """
        # An invalid handle implies that there is no articulation available for this object
        return self._handle is not None or self.articulation_root_path is not None

    @property
    def articulation_root_path(self):
        """
        Returns:
            None or str: Absolute USD path to the expected prim that represents the articulation root, if it exists. By default,
                this corresponds to self.prim_path
        """
        return self._prim_path if self.n_joints > 0 else None

    @property
    def root_link_name(self):
        """
        Returns:
            str: Name of this entity's root link
        """
        return self._root_link_name

    @property
    def root_link(self):
        """
        Returns:
            RigidPrim or ClothPrim: Root link of this object prim
        """
        return self._links[self.root_link_name]

    @property
    def root_prim(self):
        """
        Returns:
            UsdPrim: Root prim object associated with the root link of this object prim
        """
        # The root prim belongs to the link with name root_link_name
        return self._links[self.root_link_name].prim

    @property
    def handle(self):
        """
        Returns:
            None or int: ID (articulation) handle assigned to this prim from dynamic_control interface. Note that
                if this prim is not an articulation, it is None
        """
        return self._handle

    @property
    def root_handle(self):
        """
        Handle used by Isaac Sim's dynamic control module to reference the root body in this object.
        Note: while self.handle may be 0 (i.e.: invalid articulation, i.e.: object with no joints), root_handle should
            always be non-zero (i.e.: valid) if this object is initialized!

        Returns:
            int: ID handle assigned to this prim's root prim from dynamic_control interface
        """
        return self._root_handle

    @property
    def n_dof(self):
        """
        Returns:
            int: number of DoFs of the object
        """
        return self._n_dof

    @property
    def n_joints(self):
        """
        Returns:
            int: Number of joints owned by this articulation
        """
        if self.initialized:
            num = len(list(self._joints.keys()))
        else:
            # Manually iterate over all links and check for any joints that are not fixed joints!
            num = 0
            for link in self._links.values():
                for child_prim in link.prim.GetChildren():
                    prim_type = child_prim.GetPrimTypeInfo().GetTypeName().lower()
                    if "joint" in prim_type and "fixed" not in prim_type:
                        num += 1
        return num

    @property
    def n_fixed_joints(self):
        """
        Returns:
        int: Number of fixed joints owned by this articulation
        """
        # Manually iterate over all links and check for any joints that are not fixed joints!
        num = 0
        for link in self._links.values():
            for child_prim in link.prim.GetChildren():
                prim_type = child_prim.GetPrimTypeInfo().GetTypeName().lower()
                if "joint" in prim_type and "fixed" in prim_type:
                    num += 1

        return num

    @property
    def n_links(self):
        """
        Returns:
            int: Number of links owned by this articulation
        """
        return len(list(self._links.keys()))

    @property
    def joints(self):
        """
        Returns:
            dict: Dictionary mapping joint names (str) to joint prims (JointPrim) owned by this articulation
        """
        return self._joints

    @property
    def links(self):
        """
        Returns:
            dict: Dictionary mapping link names (str) to link prims (RigidPrim) owned by this articulation
        """
        return self._links

    @property
    def materials(self):
        """
        Loop through each link and their visual meshes to gather all the materials that belong to this object

        Returns:
            set of MaterialPrim: a set of MaterialPrim that belongs to this object
        """
        return self._materials

    @property
    def dof_properties(self):
        """
        Returns:
            n-array: Array of DOF properties assigned to this articulation's DoFs.
        """
        return self._dc.get_articulation_dof_properties(self._handle)

    @property
    def visual_only(self):
        """
        Returns:
            bool: Whether this link is a visual-only link (i.e.: no gravity or collisions applied)
        """
        return self._visual_only

    @visual_only.setter
    def visual_only(self, val):
        """
        Sets the visaul only state of this link

        Args:
            val (bool): Whether this link should be a visual-only link (i.e.: no gravity or collisions applied)
        """
        # Iterate over all owned links and set their respective visual-only properties accordingly
        for link in self._links.values():
            link.visual_only = val

        # Also set the internal value
        self._visual_only = val

    def contact_list(self):
        """
        Get list of all current contacts with this object prim

        Returns:
            list of CsRawData: raw contact info for this rigid body
        """
        contacts = []
        for link in self._links.values():
            contacts += link.contact_list()
        return contacts

    def enable_gravity(self) -> None:
        """
        Enables gravity for this entity
        """
        for link in self._links.values():
            link.enable_gravity()

    def disable_gravity(self) -> None:
        """
        Disables gravity for this entity
        """
        for link in self._links.values():
            link.disable_gravity()

    def set_joint_positions(self, positions, indices=None, normalized=False, drive=False):
        """
        Set the joint positions (both actual value and target values) in simulation. Note: only works if the simulator
        is actively running!

        Args:
            positions (np.ndarray): positions to set. This should be n-DOF length if all joints are being set,
                or k-length (k < n) if specific indices are being set. In this case, the length of @positions must
                be the same length as @indices!
            indices (None or k-array): If specified, should be k (k < n) length array of specific DOF positions to set.
                Default is None, which assumes that all joints are being set.
            normalized (bool): Whether the inputted joint positions should be interpreted as normalized values. Default
                is False
            drive (bool): Whether the positions being set are values that should be driven naturally by this entity's
                motors or manual values to immediately set. Default is False, corresponding to an instantaneous
                setting of the positions
        """
        # Run sanity checks -- make sure our handle is initialized and that we are articulated
        assert self._handle is not None, "handles are not initialized yet!"
        assert self.n_joints > 0, "Tried to call method not intended for entity prim with no joints!"

        # Possibly de-normalize the inputs
        if normalized:
            positions = self._denormalize_positions(positions=positions, indices=indices)

        # Grab current DOF states
        dof_states = self._dc.get_articulation_dof_states(self._handle, _dynamic_control.STATE_POS)

        # Possibly set specific values in the array if indies are specified
        if indices is None:
            assert len(positions) == self._n_dof, \
                "set_joint_positions called without specifying indices, but the desired positions do not match n_dof."
            new_positions = positions
        else:
            new_positions = dof_states["pos"]
            new_positions[indices] = positions

        # Set the DOF states
        dof_states["pos"] = new_positions
        if not drive:
            self._dc.set_articulation_dof_states(self._handle, dof_states, _dynamic_control.STATE_POS)

        # Also set the target
        self._dc.set_articulation_dof_position_targets(self._handle, new_positions.astype(np.float32))

    def set_joint_velocities(self, velocities, indices=None, normalized=False, drive=False):
        """
        Set the joint velocities (both actual value and target values) in simulation. Note: only works if the simulator
        is actively running!

        Args:
            velocities (np.ndarray): velocities to set. This should be n-DOF length if all joints are being set,
                or k-length (k < n) if specific indices are being set. In this case, the length of @velocities must
                be the same length as @indices!
            indices (None or k-array): If specified, should be k (k < n) length array of specific DOF velocities to set.
                Default is None, which assumes that all joints are being set.
            normalized (bool): Whether the inputted joint velocities should be interpreted as normalized values. Default
                is False
            drive (bool): Whether the velocities being set are values that should be driven naturally by this entity's
                motors or manual values to immediately set. Default is False, corresponding to an instantaneous
                setting of the velocities
        """
        # Run sanity checks -- make sure our handle is initialized and that we are articulated
        assert self._handle is not None, "handles are not initialized yet!"
        assert self.n_joints > 0, "Tried to call method not intended for entity prim with no joints!"

        # Possibly de-normalize the inputs
        if normalized:
            velocities = self._denormalize_velocities(velocities=velocities, indices=indices)

        # Grab current DOF states
        dof_states = self._dc.get_articulation_dof_states(self._handle, _dynamic_control.STATE_VEL)

        # Possibly set specific values in the array if indies are specified
        if indices is None:
            new_velocities = velocities
        else:
            new_velocities = dof_states["vel"]
            new_velocities[indices] = velocities

        # Set the DOF states
        dof_states["vel"] = new_velocities
        if not drive:
            self._dc.set_articulation_dof_states(self._handle, dof_states, _dynamic_control.STATE_VEL)

        # Also set the target
        self._dc.set_articulation_dof_velocity_targets(self._handle, new_velocities.astype(np.float32))

    def set_joint_efforts(self, efforts, indices=None, normalized=False):
        """
        Set the joint efforts (both actual value and target values) in simulation. Note: only works if the simulator
        is actively running!

        Args:
            efforts (np.ndarray): efforts to set. This should be n-DOF length if all joints are being set,
                or k-length (k < n) if specific indices are being set. In this case, the length of @efforts must
                be the same length as @indices!
            indices (None or k-array): If specified, should be k (k < n) length array of specific DOF efforts to set.
                Default is None, which assumes that all joints are being set.
            normalized (bool): Whether the inputted joint efforts should be interpreted as normalized values. Default
                is False
        """
        # Run sanity checks -- make sure our handle is initialized and that we are articulated
        assert self._handle is not None, "handles are not initialized yet!"
        assert self.n_joints > 0, "Tried to call method not intended for entity prim with no joints!"

        # Possibly de-normalize the inputs
        if normalized:
            efforts = self._denormalize_efforts(efforts=efforts, indices=indices)

        # Grab current DOF states
        dof_states = self._dc.get_articulation_dof_states(self._handle, _dynamic_control.STATE_EFFORT)

        # Possibly set specific values in the array if indies are specified
        if indices is None:
            new_efforts = efforts
        else:
            new_efforts = dof_states["effort"]
            new_efforts[indices] = efforts

        # Set the DOF states
        dof_states["effort"] = new_efforts
        self._dc.set_articulation_dof_states(self._handle, dof_states, _dynamic_control.STATE_EFFORT)

    def _normalize_positions(self, positions, indices=None):
        """
        Normalizes raw joint positions @positions

        Args:
            positions (n- or k-array): n-DOF raw positions to normalize, or k (k < n) specific positions to normalize.
                In the latter case, @indices should be specified
            indices (None or k-array): If specified, should be k (k < n) DOF indices corresponding to the specific
                positions to normalize. Default is None, which assumes the positions correspond to all DOF being
                normalized.

        Returns:
            n- or k-array: normalized positions in range [-1, 1] for the specified DOFs
        """
        low, high = self.joint_lower_limits, self.joint_upper_limits
        mean = (low + high) / 2.0
        magnitude = (high - low) / 2.0
        return (positions - mean) / magnitude if indices is None else (positions - mean[indices]) / magnitude[indices]

    def _denormalize_positions(self, positions, indices=None):
        """
        De-normalizes joint positions @positions

        Args:
            positions (n- or k-array): n-DOF normalized positions or k (k < n) specific positions in range [-1, 1]
                to de-normalize. In the latter case, @indices should be specified
            indices (None or k-array): If specified, should be k (k < n) DOF indices corresponding to the specific
                positions to de-normalize. Default is None, which assumes the positions correspond to all DOF being
                de-normalized.

        Returns:
            n- or k-array: de-normalized positions for the specified DOFs
        """
        low, high = self.joint_lower_limits, self.joint_upper_limits
        mean = (low + high) / 2.0
        magnitude = (high - low) / 2.0
        return positions * magnitude + mean if indices is None else positions * magnitude[indices] + mean[indices]

    def _normalize_velocities(self, velocities, indices=None):
        """
        Normalizes raw joint velocities @velocities

        Args:
            velocities (n- or k-array): n-DOF raw velocities to normalize, or k (k < n) specific velocities to normalize.
                In the latter case, @indices should be specified
            indices (None or k-array): If specified, should be k (k < n) DOF indices corresponding to the specific
                velocities to normalize. Default is None, which assumes the velocities correspond to all DOF being
                normalized.

        Returns:
            n- or k-array: normalized velocities in range [-1, 1] for the specified DOFs
        """
        return velocities / self.max_joint_velocities if indices is None else \
            velocities / self.max_joint_velocities[indices]

    def _denormalize_velocities(self, velocities, indices=None):
        """
        De-normalizes joint velocities @velocities

        Args:
            velocities (n- or k-array): n-DOF normalized velocities or k (k < n) specific velocities in range [-1, 1]
                to de-normalize. In the latter case, @indices should be specified
            indices (None or k-array): If specified, should be k (k < n) DOF indices corresponding to the specific
                velocities to de-normalize. Default is None, which assumes the velocities correspond to all DOF being
                de-normalized.

        Returns:
            n- or k-array: de-normalized velocities for the specified DOFs
        """
        return velocities * self.max_joint_velocities if indices is None else \
            velocities * self.max_joint_velocities[indices]

    def _normalize_efforts(self, efforts, indices=None):
        """
        Normalizes raw joint efforts @efforts

        Args:
            efforts (n- or k-array): n-DOF raw efforts to normalize, or k (k < n) specific efforts to normalize.
                In the latter case, @indices should be specified
            indices (None or k-array): If specified, should be k (k < n) DOF indices corresponding to the specific
                efforts to normalize. Default is None, which assumes the efforts correspond to all DOF being
                normalized.

        Returns:
            n- or k-array: normalized efforts in range [-1, 1] for the specified DOFs
        """
        return efforts / self.max_joint_efforts if indices is None else efforts / self.max_joint_efforts[indices]

    def _denormalize_efforts(self, efforts, indices=None):
        """
        De-normalizes joint efforts @efforts

        Args:
            efforts (n- or k-array): n-DOF normalized efforts or k (k < n) specific efforts in range [-1, 1]
                to de-normalize. In the latter case, @indices should be specified
            indices (None or k-array): If specified, should be k (k < n) DOF indices corresponding to the specific
                efforts to de-normalize. Default is None, which assumes the efforts correspond to all DOF being
                de-normalized.

        Returns:
            n- or k-array: de-normalized efforts for the specified DOFs
        """
        return efforts * self.max_joint_efforts if indices is None else efforts * self.max_joint_efforts[indices]

    def update_handles(self):
        """
        Updates all internal handles for this prim, in case they change since initialization
        """
        assert og.sim.is_playing(), "Simulator must be playing if updating handles!"

        # Grab the handle -- we know it might not return a valid value, so we suppress omni's warning here
        self._handle = None if self.articulation_root_path is None else \
            self._dc.get_articulation(self.articulation_root_path)

        # Sanity check -- make sure handle is not invalid handle -- it should only ever be None or a valid integer
        assert self._handle != _dynamic_control.INVALID_HANDLE, \
            f"Got invalid articulation handle for entity at {self.articulation_root_path}"

        # We only have a root handle if we're not a cloth prim
        if self._prim_type != PrimType.CLOTH:
            self._root_handle = self._dc.get_articulation_root_body(self._handle) if \
                self._handle is not None else self.root_link.handle

        # Update all links and joints as well
        for link in self._links.values():
            if not link.initialized:
                link.initialize()
            link.update_handles()

        for joint in self._joints.values():
            if not joint.initialized:
                joint.initialize()
            joint.update_handles()

    def get_joint_positions(self, normalized=False):
        """
        Grabs this entity's joint positions

        Args:
            normalized (bool): Whether returned values should be normalized to range [-1, 1] based on limits or not.

        Returns:
            n-array: n-DOF length array of positions
        """
        # Run sanity checks -- make sure our handle is initialized and that we are articulated
        assert self._handle is not None, "handles are not initialized yet!"
        assert self.n_joints > 0, "Tried to call method not intended for entity prim with no joints!"

        joint_positions = self._dc.get_articulation_dof_states(self._handle, _dynamic_control.STATE_POS)["pos"]

        # Possibly normalize values when returning
        return self._normalize_positions(positions=joint_positions) if normalized else joint_positions

    def get_joint_velocities(self, normalized=False):
        """
        Grabs this entity's joint velocities

        Args:
            normalized (bool): Whether returned values should be normalized to range [-1, 1] based on limits or not.

        Returns:
            n-array: n-DOF length array of velocities
        """
        # Run sanity checks -- make sure our handle is initialized and that we are articulated
        assert self._handle is not None, "handles are not initialized yet!"
        assert self.n_joints > 0, "Tried to call method not intended for entity prim with no joints!"

        joint_velocities = self._dc.get_articulation_dof_states(self._handle, _dynamic_control.STATE_VEL)["vel"]

        # Possibly normalize values when returning
        return self._normalize_velocities(velocities=joint_velocities) if normalized else joint_velocities

    def get_joint_efforts(self, normalized=False):
        """
        Grabs this entity's joint efforts

        Args:
            normalized (bool): Whether returned values should be normalized to range [-1, 1] based on limits or not.

        Returns:
            n-array: n-DOF length array of efforts
        """
        # Run sanity checks -- make sure our handle is initialized and that we are articulated
        assert self._handle is not None, "handles are not initialized yet!"
        assert self.n_joints > 0, "Tried to call method not intended for entity prim with no joints!"

        joint_efforts = self._dc.get_articulation_dof_states(self._handle, _dynamic_control.STATE_EFFORT)["effort"]

        # Possibly normalize values when returning
        return self._normalize_efforts(efforts=joint_efforts) if normalized else joint_efforts

    def set_linear_velocity(self, velocity: np.ndarray):
        """
        Sets the linear velocity of the root prim in stage.

        Args:
            velocity (np.ndarray): linear velocity to set the rigid prim to, in the world frame. Shape (3,).
        """
        self.root_link.set_linear_velocity(velocity)

    def get_linear_velocity(self):
        """
        Gets the linear velocity of the root prim in stage.

        Returns:
            velocity (np.ndarray): linear velocity to set the rigid prim to, in the world frame. Shape (3,).
        """
        return self.root_link.get_linear_velocity()

    def set_angular_velocity(self, velocity):
        """
        Sets the angular velocity of the root prim in stage.

        Args:
            velocity (np.ndarray): angular velocity to set the rigid prim to, in the world frame. Shape (3,).
        """
        self.root_link.set_angular_velocity(velocity)

    def get_angular_velocity(self):
        """Gets the angular velocity of the root prim in stage.

        Returns:
            velocity (np.ndarray): angular velocity to set the rigid prim to, in the world frame. Shape (3,).
        """
        return self.root_link.get_angular_velocity()

    def set_position_orientation(self, position=None, orientation=None):
        current_position, current_orientation = self.get_position_orientation()
        if position is None:
            position = current_position
        if orientation is None:
            orientation = current_orientation

        if self._prim_type == PrimType.CLOTH:
            if self._dc is not None and self._dc.is_simulating():
                self.root_link.set_position_orientation(position, orientation)
            else:
                super().set_position_orientation(position, orientation)
        else:
            if self._root_handle is not None and self._root_handle != _dynamic_control.INVALID_HANDLE and \
                    self._dc is not None and self._dc.is_simulating():
                self.root_link.set_position_orientation(position, orientation)
            else:
                super().set_position_orientation(position=position, orientation=orientation)

    def get_position_orientation(self):
        if self._prim_type == PrimType.CLOTH:
            if self._dc is not None and self._dc.is_simulating():
                return self.root_link.get_position_orientation()
            else:
                return super().get_position_orientation()
        else:
            if self._root_handle is not None and self._root_handle != _dynamic_control.INVALID_HANDLE and \
                    self._dc is not None and self._dc.is_simulating():
                return self.root_link.get_position_orientation()
            else:
                return super().get_position_orientation()

    def _set_local_pose_when_simulating(self, translation=None, orientation=None):
        """
        Sets prim's pose with respect to the local frame (the prim's parent frame) when simulation is running.

        Args:
            translation (None or 3-array): if specified, (x,y,z) translation in the local frame of the prim
                (with respect to its parent prim). Default is None, which means left unchanged.
            orientation (None or 4-array): if specified, (x,y,z,w) quaternion orientation in the local frame of the prim
                (with respect to its parent prim). Default is None, which means left unchanged.
        """
        current_translation, current_orientation = self.get_local_pose()
        if translation is None:
            translation = current_translation
        if orientation is None:
            orientation = current_orientation
        orientation = orientation[[3, 0, 1, 2]]
        local_transform = tf_matrix_from_pose(translation=translation, orientation=orientation)
        parent_world_tf = UsdGeom.Xformable(get_prim_parent(self._prim)).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        my_world_transform = np.matmul(parent_world_tf, local_transform)
        transform = Gf.Transform()
        transform.SetMatrix(Gf.Matrix4d(np.transpose(my_world_transform)))
        calculated_position = transform.GetTranslation()
        calculated_orientation = transform.GetRotation().GetQuat()
        self.set_position_orientation(
            position=np.array(calculated_position),
            orientation=gf_quat_to_np_array(calculated_orientation)[[1, 2, 3, 0]],
        )

    def set_local_pose(self, translation=None, orientation=None):
        if self._prim_type == PrimType.CLOTH:
            if self._dc is not None and self._dc.is_simulating():
                self._set_local_pose_when_simulating(translation=translation, orientation=orientation)
            else:
                super().set_local_pose(translation=translation, orientation=orientation)
        else:
            if self._root_handle is not None and self._root_handle != _dynamic_control.INVALID_HANDLE and \
                    self._dc is not None and self._dc.is_simulating():
                self._set_local_pose_when_simulating(translation=translation, orientation=orientation)
            else:
                super().set_local_pose(translation=translation, orientation=orientation)

    def _get_local_pose_when_simulating(self):
        """
        Gets prim's pose with respect to the prim's local frame (it's parent frame) when simulation is running

        Returns:
            2-tuple:
                - 3-array: (x,y,z) position in the local frame
                - 4-array: (x,y,z,w) quaternion orientation in the local frame
        """
        parent_world_tf = UsdGeom.Xformable(get_prim_parent(self._prim)).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        world_position, world_orientation = self.get_position_orientation()
        my_world_transform = tf_matrix_from_pose(translation=world_position,
                                                 orientation=world_orientation[[3, 0, 1, 2]])
        local_transform = np.matmul(np.linalg.inv(np.transpose(parent_world_tf)), my_world_transform)
        transform = Gf.Transform()
        transform.SetMatrix(Gf.Matrix4d(np.transpose(local_transform)))
        calculated_translation = transform.GetTranslation()
        calculated_orientation = transform.GetRotation().GetQuat()
        return np.array(calculated_translation), gf_quat_to_np_array(calculated_orientation)[[1, 2, 3, 0]]

    def get_local_pose(self):
        if self._prim_type == PrimType.CLOTH:
            if self._dc is not None and self._dc.is_simulating():
                return self._get_local_pose_when_simulating()
            else:
                return super().get_local_pose()
        else:
            if self._root_handle is not None and self._root_handle != _dynamic_control.INVALID_HANDLE and \
                    self._dc is not None and self._dc.is_simulating():
                return self._get_local_pose_when_simulating()
            else:
                return super().get_local_pose()

    # TODO: Is the omni joint damping (used for driving motors) same as dissipative joint damping (what we had in pb)?
    @property
    def joint_damping(self):
        """
        Returns:
            n-array: joint damping values for this prim
        """
        return np.concatenate([joint.damping for joint in self._joints.values()])

    @property
    def joint_lower_limits(self):
        """
        Returns:
            n-array: minimum values for this robot's joints. If joint does not have a range, returns -1000
                for that joint
        """
        return np.array([joint.lower_limit for joint in self._joints.values()])

    @property
    def joint_upper_limits(self):
        """
        Returns:
            n-array: maximum values for this robot's joints. If joint does not have a range, returns 1000
                for that joint
        """
        return np.array([joint.upper_limit for joint in self._joints.values()])

    @property
    def joint_range(self):
        """
        Returns:
            n-array: joint range values for this robot's joints
        """
        return self.joint_upper_limits - self.joint_lower_limits

    @property
    def max_joint_velocities(self):
        """
        Returns:
            n-array: maximum velocities for this robot's joints
        """
        return np.array([joint.max_velocity for joint in self._joints.values()])

    @property
    def max_joint_efforts(self):
        """
        Returns:
            n-array: maximum efforts for this robot's joints
        """
        return np.array([joint.max_effort for joint in self._joints.values()])

    @property
    def joint_position_limits(self):
        """
        Returns:
            2-tuple:
                - n-array: min joint position limits, where each is an n-DOF length array
                - n-array: max joint position limits, where each is an n-DOF length array
        """
        return self.joint_lower_limits, self.joint_upper_limits

    @property
    def joint_velocity_limits(self):
        """
        Returns:
            2-tuple:
                - n-array: min joint velocity limits, where each is an n-DOF length array
                - n-array: max joint velocity limits, where each is an n-DOF length array
        """
        return -self.max_joint_velocities, self.max_joint_velocities

    @property
    def joint_effort_limits(self):
        """
        Returns:
            2-tuple:
                - n-array: min joint effort limits, where each is an n-DOF length array
                - n-array: max joint effort limits, where each is an n-DOF length array
        """
        return -self.max_joint_efforts, self.max_joint_efforts

    @property
    def joint_at_limits(self):
        """
        Returns:
            n-array: n-DOF length array specifying whether joint is at its limit,
                with 1.0 --> at limit, otherwise 0.0
        """
        return 1.0 * (np.abs(self.get_joint_positions(normalized=True)) > 0.99)

    @property
    def joint_has_limits(self):
        """
        Returns:
            n-array: n-DOF length array specifying whether joint has a limit or not
        """
        return np.array([j.has_limit for j in self._joints.values()])

    @property
    def disabled_collision_pairs(self):
        """
        Returns:
            list of (str, str): List of rigid body collision pairs to disable within this object prim.
                Default is an empty list (no pairs)
        """
        return []

    @property
    def scale(self):
        # Since all rigid bodies owned by this object prim have the same scale, we simply grab it from the root prim
        return self.root_link.scale

    @scale.setter
    def scale(self, scale):
        # We iterate over all rigid bodies owned by this object prim and set their individual scales
        # We do this because omniverse cannot scale orientation of an articulated prim, so we get mesh mismatches as
        # they rotate in the world
        for link in self._links.values():
            link.scale = scale

    @property
    def solver_position_iteration_count(self):
        """
        Returns:
            int: How many position iterations to take per physics step by the physx solver
        """
        return get_prim_property(self.articulation_root_path, "physxArticulation:solverPositionIterationCount") if \
            self.articulated else self.root_link.solver_position_iteration_count

    @solver_position_iteration_count.setter
    def solver_position_iteration_count(self, count):
        """
        Sets how many position iterations to take per physics step by the physx solver

        Args:
            count (int): How many position iterations to take per physics step by the physx solver
        """
        if self.articulated:
            set_prim_property(self.articulation_root_path, "physxArticulation:solverPositionIterationCount", count)
        else:
            for link in self._links.values():
                link.solver_position_iteration_count = count

    @property
    def solver_velocity_iteration_count(self):
        """
        Returns:
            int: How many velocity iterations to take per physics step by the physx solver
        """
        return get_prim_property(self.articulation_root_path, "physxArticulation:solverVelocityIterationCount") if \
            self.articulated else self.root_link.solver_velocity_iteration_count

    @solver_velocity_iteration_count.setter
    def solver_velocity_iteration_count(self, count):
        """
        Sets how many velocity iterations to take per physics step by the physx solver

        Args:
            count (int): How many velocity iterations to take per physics step by the physx solver
        """
        if self.articulated:
            set_prim_property(self.articulation_root_path, "physxArticulation:solverVelocityIterationCount", count)
        else:
            for link in self._links.values():
                link.solver_velocity_iteration_count = count

    @property
    def stabilization_threshold(self):
        """
        Returns:
            float: threshold for stabilizing this articulation
        """
        return get_prim_property(self.articulation_root_path, "physxArticulation:stabilizationThreshold") if \
            self.articulated else self.root_link.stabilization_threshold

    @stabilization_threshold.setter
    def stabilization_threshold(self, threshold):
        """
        Sets threshold for stabilizing this articulation

        Args:
            threshold (float): Stabilization threshold
        """
        if self.articulated:
            set_prim_property(self.articulation_root_path, "physxArticulation:stabilizationThreshold", threshold)
        else:
            for link in self._links.values():
                link.stabilization_threshold = threshold

    @property
    def sleep_threshold(self):
        """
        Returns:
            float: threshold for sleeping this articulation
        """
        return get_prim_property(self.articulation_root_path, "physxArticulation:sleepThreshold") if \
            self.articulated else self.root_link.sleep_threshold

    @sleep_threshold.setter
    def sleep_threshold(self, threshold):
        """
        Sets threshold for sleeping this articulation

        Args:
            threshold (float): Sleeping threshold
        """
        if self.articulated:
            set_prim_property(self.articulation_root_path, "physxArticulation:sleepThreshold", threshold)
        else:
            for link in self._links.values():
                link.sleep_threshold = threshold

    @property
    def self_collisions(self):
        """
        Returns:
            bool: Whether self-collisions are enabled for this prim or not
        """
        assert self.articulated, "Cannot get self-collision for non-articulated EntityPrim!"
        return get_prim_property(self.articulation_root_path, "physxArticulation:enabledSelfCollisions")

    @self_collisions.setter
    def self_collisions(self, flag):
        """
        Sets whether self-collisions are enabled for this prim or not

        Args:
            flag (bool): Whether self collisions are enabled for this prim or not
        """
        assert self.articulated, "Cannot set self-collision for non-articulated EntityPrim!"
        set_prim_property(self.articulation_root_path, "physxArticulation:enabledSelfCollisions", flag)

    @property
    def kinematic_only(self):
        """
        Returns:
            bool: Whether this object is a kinematic-only object (otherwise, it is a rigid body). A kinematic-only
                object is not subject to simulator dynamics, and remains fixed unless the user explicitly sets the
                body's pose / velocities. See https://docs.omniverse.nvidia.com/app_create/prod_extensions/ext_physics/rigid-bodies.html?highlight=rigid%20body%20enabled#kinematic-rigid-bodies
                for more information
        """
        return self.root_link.kinematic_only

    @kinematic_only.setter
    def kinematic_only(self, val):
        """
        Args:
            val (bool): Whether this object is a kinematic-only object (otherwise, it is a rigid body). A kinematic-only
                object is not subject to simulator dynamics, and remains fixed unless the user explicitly sets the
                body's pose / velocities. See https://docs.omniverse.nvidia.com/app_create/prod_extensions/ext_physics/rigid-bodies.html?highlight=rigid%20body%20enabled#kinematic-rigid-bodies
                for more information
        """
        self.root_link.kinematic_only = val

    @property
    def aabb(self):
        # If we're a cloth prim type, we compute the bounding box from the limits of the particles. Otherwise, use the
        # normal method for computing bounding box
        if self._prim_type == PrimType.CLOTH:
            particle_positions = self.root_link.particle_positions
            aabb_lo, aabb_hi = np.min(particle_positions, axis=0), np.max(particle_positions, axis=0)
        else:
            aabb_lo, aabb_hi = super().aabb
            aabb_lo, aabb_hi = np.array(aabb_lo), np.array(aabb_hi)

        return aabb_lo, aabb_hi

    def wake(self):
        """
        Enable physics for this articulation
        """
        if self.articulated:
            self._dc.wake_up_articulation(self._handle)
        else:
            for link in self._links.values():
                link.wake()

    def sleep(self):
        """
        Disable physics for this articulation
        """
        if self.articulated:
            self._dc.sleep_articulation(self._handle)
        else:
            for link in self._links.values():
                link.sleep()

    def keep_still(self):
        """
        Zero out all velocities for this prim
        """
        self.set_linear_velocity(velocity=np.zeros(3))
        self.set_angular_velocity(velocity=np.zeros(3))
        for joint in self._joints.values():
            joint.keep_still()

    def create_attachment_point_link(self):
        """
        Create a collision-free, invisible attachment point link for the cloth object, and create an attachment between
        the ClothPrim and this attachment point link (RigidPrim).

        One use case for this is that we can create a fixed joint between this link and the world to enable AG fo cloth.
        During simulation, this joint will move and match the robot gripper frame, which will then drive the cloth.
        """

        assert self._prim_type == PrimType.CLOTH, "create_attachment_point_link should only be called for Cloth"
        link_name = "attachment_point"
        stage = get_current_stage()
        link_prim = stage.DefinePrim(f"{self._prim_path}/{link_name}", "Xform")
        vis_prim = UsdGeom.Sphere.Define(stage, f"{self._prim_path}/{link_name}/visuals").GetPrim()
        col_prim = UsdGeom.Sphere.Define(stage, f"{self._prim_path}/{link_name}/collisions").GetPrim()

        # Set the radius to be 0.03m. In theory, we want this radius to be as small as possible. Otherwise, the cloth
        # dynamics will be unrealistic. However, in practice, if the radius is too small, the attachment becomes very
        # unstable. Empirically 0.03m works reasonably well.
        vis_prim.GetAttribute("radius").Set(0.03)
        col_prim.GetAttribute("radius").Set(0.03)

        # Need to sync the extents
        extent = vis_prim.GetAttribute("extent").Get()
        extent[0] = Gf.Vec3f(-0.03, -0.03, -0.03)
        extent[1] = Gf.Vec3f(0.03, 0.03, 0.03)
        vis_prim.GetAttribute("extent").Set(extent)
        col_prim.GetAttribute("extent").Set(extent)

        # Add collision API to collision geom
        UsdPhysics.CollisionAPI.Apply(col_prim)
        UsdPhysics.MeshCollisionAPI.Apply(col_prim)
        PhysxSchema.PhysxCollisionAPI.Apply(col_prim)

        # Create a attachment point link
        link = RigidPrim(
            prim_path=link_prim.GetPrimPath().__str__(),
            name=f"{self._name}:{link_name}",
        )
        link.disable_collisions()
        # TODO (eric): Should we disable gravity for this link?
        # link.disable_gravity()
        link.visible = False
        # Set a very small mass
        link.mass = 1e-6

        self._links[link_name] = link

        # Create an attachment between the root link (ClothPrim) and the newly created attachment point link (RigidPrim)
        attachment_path = self.root_link.prim.GetPath().AppendElementString("attachment")
        omni.kit.commands.execute("CreatePhysicsAttachment", target_attachment_path=attachment_path,
                                  actor0_path=self.root_link.prim.GetPath(), actor1_path=link.prim.GetPath())

    def _dump_state(self):
        # We don't call super, instead, this state is simply the root link state and all joint states
        state = dict(root_link=self.root_link._dump_state())
        joint_state = dict()
        for prim_name, prim in self._joints.items():
            joint_state[prim_name] = prim._dump_state()
        state["joints"] = joint_state

        return state

    def _load_state(self, state):
        # Load base link state and joint states
        self.root_link._load_state(state=state["root_link"])
        for joint_name, joint_state in state["joints"].items():
            self._joints[joint_name]._load_state(state=joint_state)

    def _serialize(self, state):
        # We serialize by first flattening the root link state and then iterating over all joints and
        # adding them to the a flattened array
        state_flat = [self.root_link.serialize(state=state["root_link"])]
        if self.n_joints > 0:
            state_flat.append(
                np.concatenate(
                    [prim.serialize(state=state["joints"][prim_name]) for prim_name, prim in self._joints.items()]
                )
            )

        return np.concatenate(state_flat).astype(float)

    def _deserialize(self, state):
        # We deserialize by first de-flattening the root link state and then iterating over all joints and
        # sequentially grabbing from the flattened state array, incrementing along the way
        idx = self.root_link.state_size
        state_dict = dict(root_link=self.root_link.deserialize(state=state[:idx]))
        joint_state_dict = dict()
        for prim_name, prim in self._joints.items():
            joint_state_dict[prim_name] = prim.deserialize(state=state[idx:idx+prim.state_size])
            idx += prim.state_size
        state_dict["joints"] = joint_state_dict

        return state_dict, idx

    def _create_prim_with_same_kwargs(self, prim_path, name, load_config):
        # Subclass must implement this method for duplication functionality
        raise NotImplementedError("Subclass must implement _create_prim_with_same_kwargs() to enable duplication "
                                  "functionality for EntityPrim!")
