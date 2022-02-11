# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from typing import Optional, Tuple, Union, List
import numpy as np
from collections import OrderedDict, Iterable
from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.core.utils.types import DOFInfo
from omni.isaac.core.utils.types import JointsState, ArticulationAction
from omni.isaac.core.utils.transformations import tf_matrix_from_pose
from omni.isaac.core.utils.rotations import gf_quat_to_np_array
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from omni.isaac.core.controllers.articulation_controller import ArticulationController
import carb
from omni.isaac.core.utils.prims import is_prim_path_valid, get_prim_property, set_prim_property, \
    get_prim_parent, get_prim_at_path
from igibson.prims.xform_prim import XFormPrim
from igibson.prims.rigid_prim import RigidPrim
from igibson.prims.joint_prim import JointPrim

# TODO: REname, since not all of these may be articulated. Maybe "ObjectPrim" or something, to denote that this class captures a single entity that owns nested rigid bodies / joints?
class ArticulatedPrim(XFormPrim):
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
        self._root_prim = None
        self._dofs_infos = None
        self._num_dof = None
        self._articulation_controller = None
        self._default_joints_state = None
        self._links = None
        self._joints = None
        self._mass = None

        # Run super init
        super().__init__(
            prim_path=prim_path,
            name=name,
            load_config=load_config,
        )

    def _initialize(self):
        # Run super method
        super()._initialize()

        # Get dynamic control info
        self._dc = _dynamic_control.acquire_dynamic_control_interface()
        self._handle = self._dc.get_articulation(self._prim_path)
        self._joints = OrderedDict()

        # Initialize all the links
        for link in self._links.values():
            link.initialize()

        # Handle case separately based on whether the handle is valid (i.e.: whether we are actually articulated or not)
        if self._handle != _dynamic_control.INVALID_HANDLE:
            root_handle = self._dc.get_articulation_root_body(self._handle)
            root_prim = get_prim_at_path(self._dc.get_rigid_body_path(root_handle))
            num_dof = self._dc.get_articulation_dof_count(self._handle)

            # Additionally grab DOF info if we have non-fixed joints
            if num_dof > 0:
                self._articulation_controller = ArticulationController()
                self._dofs_infos = OrderedDict()
                # Grab DOF info
                for index in range(num_dof):
                    dof_handle = self._dc.get_articulation_dof(self._handle, index)
                    dof_name = self._dc.get_dof_name(dof_handle)
                    # add dof to list
                    prim_path = self._dc.get_dof_path(dof_handle)
                    self._dofs_infos[dof_name] = DOFInfo(prim_path=prim_path, handle=dof_handle, prim=self.prim, index=index)
                # Initialize articulation controller
                self._articulation_controller.initialize(self._handle, self._dofs_infos)
                # Default joints state is the info from the USD
                default_actions = self._articulation_controller.get_applied_action()
                self._default_joints_state = JointsState(
                    positions=np.array(default_actions.joint_positions),
                    velocities=np.array(default_actions.joint_velocities),
                    efforts=np.zeros_like(default_actions.joint_positions),
                )

                for i in range(self._dc.get_articulation_joint_count(self._handle)):
                    joint_handle = self._dc.get_articulation_joint(self._handle, i)
                    joint_name = self._dc.get_joint_name(joint_handle)
                    joint_path = self._dc.get_joint_path(joint_handle)
                    joint = JointPrim(
                        prim_path=joint_path,
                        name=f"{self._name}:{joint_name}",
                        articulation=self._handle,
                    )
                    joint.initialize()
                    self._joints[joint_name] = joint
        else:
            # TODO: May need to extend to clusters of rigid bodies, that aren't exactly joined
            # We assume this object contains a single rigid body
            body_path = f"{self._prim_path}/base_link"
            root_handle = self._dc.get_rigid_body(body_path)
            root_prim = get_prim_at_path(body_path)
            num_dof = 0

        # Store values internally
        self._root_handle = root_handle
        self._root_prim = root_prim
        self._num_dof = num_dof

        print(f"root handle: {self._root_handle}, root prim path: {self._dc.get_rigid_body_path(self._root_handle)}")

    def _load(self, simulator=None):
        # This should not be called, because this prim cannot be instantiated from scratch!
        raise NotImplementedError("By default, an articulated prim cannot be created from scratch.")

    def _post_load(self, simulator=None):
        # Setup links info FIRST before running any other post loading behavior
        # We iterate over all children of this object's prim,
        # and grab any that are presumed to be rigid bodies (i.e.: other Xforms)
        self._links = OrderedDict()
        for prim in self._prim.GetChildren():
            # Only process prims that are an Xform
            if prim.GetPrimTypeInfo().GetTypeName() == "Xform":
                link_name = prim.GetName()
                link = RigidPrim(
                    prim_path=prim.GetPrimPath().__str__(),
                    name=f"{self._name}:{link_name}",
                )
                self._links[link_name] = link

        # The root prim belongs to the first link
        self._root_prim = list(self._links.values())[0].prim

        # Disable any requested collision pairs
        for a_name, b_name in self.disabled_collision_pairs:
            link_a, link_b = self._links[a_name], self._links[b_name]
            link_a.add_filtered_collision_pair(prim=link_b)

        # Run super
        super()._post_load(simulator=simulator)

    @property
    def articulated(self):
        """
        Returns:
             bool: Whether this prim is articulated or not
        """
        # An invalid handle implies that there is no articulation available for this object
        return self._handle != _dynamic_control.INVALID_HANDLE and self.num_dof > 0

    def assert_articulated(self):
        """
        Sanity check to make sure this joint is articulated. Used as a gatekeeping function to prevent non-intended
        behavior (e.g.: trying to grab this joint's state if it's not articulated)
        """
        assert self.articulated, "Tried to call method not intended for non-articulated object prim!"

    @property
    def root_link(self):
        """
        Returns:
            RigidPrim: Root link of this object prim
        """
        return self._links[self._root_prim.GetName()]

    @property
    def handle(self):
        """[summary]

        Returns:
            int: [description]
        """
        return self._handle

    @property
    def num_dof(self):
        """[summary]

        Returns:
            int: [description]
        """
        return self._num_dof

    @property
    def num_joints(self):
        """
        Returns:
            int: Number of joints owned by this articulation
        """
        return len(list(self._joints.keys()))

    @property
    def num_links(self):
        """
        Returns:
            int: Number of links owned by this articulation
        """
        return len(list(self._links.keys()))

    @property
    def joints(self):
        """
        Returns:
            OrderedDict: Dictionary mapping joint names (str) to joint prims (JointPrim) owned by this articulation
        """
        return self._joints

    @property
    def links(self):
        """
        Returns:
            OrderedDict: Dictionary mapping link names (str) to link prims (RigidPrim) owned by this articulation
        """
        return self._links

    @property
    def dof_properties(self):
        """[summary]

        Returns:
            np.ndarray: [description]
        """
        return self._dc.get_articulation_dof_properties(self._handle)

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

    def in_contact_links(self):
        """
        Get set of unique rigid body handles that this object prim is in contact with

        Returns:
            set: Unique rigid body handles that this body is in contact with
        """
        contact_list = self.contact_list()
        link_handles = {link.handle for link in self._links.values()}
        body0_contacts = {c.body0 for c in contact_list if c.body0 not in link_handles}
        body1_contacts = {c.body1 for c in contact_list if c.body1 not in link_handles}
        return body0_contacts.union(body1_contacts)

    def in_contact(self, objects=None, links=None):
        """
        Returns whether this object is in contact with any object in @objects or link in @links. Note that at least
        one should be specified (both can be specified, in which case this will check for any contacts amongst the
        specified objects OR the specified links

        Args:
            objects (None or ArticulatedPrim or list of ArticulatedPrim): Object(s) to check for collision with
            links (None or RigidPrim or list of RigidPrim): Link(s) to check for collision with

        Returns:
            bool: Whether this object is in contact with the specified object(s) and / or link(s)
        """
        # Make sure at least one of objects or links are specified
        assert objects is not None or links is not None, "At least one of objects or links must be specified to check" \
                                                         "for contact!"

        # Standardize inputs
        objects = [] if objects is None else (objects if isinstance(objects, Iterable) else [objects])
        links = [] if links is None else (links if isinstance(objects, Iterable) else [links])

        # Get list of link handles to check for contact with
        link_handles = {[link.handle for link in links]}
        for obj in objects:
            link_handles = link_handles.union({link.handle for link in obj.links.values()})

        # Grab all contacts for this object prim
        valid_contacts = link_handles.intersection(self.in_contact_links())

        # We're in contact if any of our current contacts are the requested contact
        return len(valid_contacts) > 0

    def get_dof_index(self, dof_name: str) -> int:
        """[summary]

        Args:
            dof_name (str): [description]

        Returns:
            int: [description]
        """
        return self._dofs_infos[dof_name].index

    def read_kinematic_hierarchy(self) -> None:
        """[summary]
        """
        print("Articulation handle: {self._handle}")
        print("--- Hierarchy:\n", self._read_kinematic_hierarchy())
        return

    def _read_kinematic_hierarchy(self, body_index: Optional[int] = None, indent_level: int = 0) -> str:
        # Only can be called if this is articulated
        self.assert_articulated()

        if body_index is None:
            body_index = self._dc.get_articulation_root_body(self._handle)
        indent = "|" + "-" * indent_level
        body_name = self._dc.get_rigid_body_name(body_index)
        str_output = f"{indent}Body: {body_name}\n", "blue"
        for i in range(self._dc.get_rigid_body_child_joint_count(body_index)):
            joint = self._dc.get_rigid_body_child_joint(body_index, i)
            joint_name = self._dc.get_joint_name(joint)
            child = self._dc.get_joint_child_body(joint)
            child_name = self._dc.get_rigid_body_name(child)
            str_output += f"{indent}>>Joint: {joint_name} -> {child_name}\n", "green"
            str_output += self._read_kinematic_hierarchy(child, indent_level + 4)
        return str_output

    def disable_gravity(self) -> None:
        """[summary]
        """
        for link in self._links.values():
            self._dc.set_rigid_body_disable_gravity(link.handle, False)
        return

    def set_joint_positions(self, positions: np.ndarray, indices: Optional[Union[List, np.ndarray]] = None) -> None:
        """[summary]

        Args:
            positions (np.ndarray): [description]
            indices (Optional[Union[list, np.ndarray]], optional): [description]. Defaults to None.

        Raises:
            Exception: [description]
        """
        print(f"name: {self.name}, handle: {self._handle}, num dof: {self.num_dof}")
        # Only can be called if this is articulated
        self.assert_articulated()

        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        if self._handle is None:
            raise Exception("handles are not initialized yet")
        dof_states = self._dc.get_articulation_dof_states(self._handle, _dynamic_control.STATE_POS)
        if indices is None:
            new_joint_positions = positions
        else:
            new_joint_positions = self.get_joint_positions()
            for i in range(len(indices)):
                new_joint_positions[indices[i]] = positions[i]
        dof_states["pos"] = new_joint_positions
        self._dc.set_articulation_dof_states(self._handle, dof_states, _dynamic_control.STATE_POS)
        self._articulation_controller.apply_action(
            ArticulationAction(joint_positions=new_joint_positions, joint_velocities=None, joint_efforts=None)
        )
        return

    def set_joint_velocities(self, velocities: np.ndarray, indices: Optional[Union[List, np.ndarray]] = None) -> None:
        """[summary]

        Args:
            velocities (np.ndarray): [description]
            indices (Optional[Union[list, np.ndarray]], optional): [description]. Defaults to None.

        Raises:
            Exception: [description]
        """
        # Only can be called if this is articulated
        self.assert_articulated()

        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        if self._handle is None:
            raise Exception("handles are not initialized yet")
        dof_states = self._dc.get_articulation_dof_states(self._handle, _dynamic_control.STATE_VEL)
        if indices is None:
            new_joint_velocities = velocities
        else:
            new_joint_velocities = self.get_joint_velocities()
            for i in range(len(indices)):
                new_joint_velocities[indices[i]] = velocities[i]
        dof_states["vel"] = new_joint_velocities
        self._dc.set_articulation_dof_states(self._handle, dof_states, _dynamic_control.STATE_VEL)
        self._articulation_controller.apply_action(
            ArticulationAction(joint_positions=None, joint_velocities=new_joint_velocities, joint_efforts=None)
        )
        return

    def set_joint_efforts(self, efforts: np.ndarray, indices: Optional[Union[List, np.ndarray]] = None) -> None:
        """[summary]

        Args:
            efforts (np.ndarray): [description]
            indices (Optional[Union[list, np.ndarray]], optional): [description]. Defaults to None.

        Raises:
            Exception: [description]
        """
        # Only can be called if this is articulated
        self.assert_articulated()

        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        if self._handle is None:
            raise Exception("handles are not initialized yet")
        dof_states = self._dc.get_articulation_dof_states(self._handle, _dynamic_control.STATE_EFFORT)
        if indices is None:
            new_joint_efforts = efforts
        else:
            new_joint_efforts = [0] * self.num_dof
            for i in range(len(indices)):
                new_joint_efforts[indices[i]] = efforts[i]
        dof_states["effort"] = new_joint_efforts
        self._dc.set_articulation_dof_states(self._handle, dof_states, _dynamic_control.STATE_EFFORT)
        self._articulation_controller.apply_action(
            ArticulationAction(joint_positions=None, joint_velocities=None, joint_efforts=new_joint_efforts)
        )
        return

    def update_default_state(self):
        # Iterate over all links and joints and update their default states
        for link in self._links.values():
            link.update_default_state()
        for joint in self._joints.values():
            joint.update_default_state()

    def get_joint_positions(self) -> np.ndarray:
        """[summary]

        Raises:
            Exception: [description]

        Returns:
            np.ndarray: [description]
        """
        # Only can be called if this is articulated
        self.assert_articulated()

        if self._handle is None:
            raise Exception("handles are not initialized yet")
        joint_positions = self._dc.get_articulation_dof_states(self._handle, _dynamic_control.STATE_POS)
        joint_positions = [joint_positions[i][0] for i in range(len(joint_positions))]
        return np.array(joint_positions)

    def get_joint_velocities(self) -> np.ndarray:
        """[summary]

        Raises:
            Exception: [description]

        Returns:
            np.ndarray: [description]
        """
        # Only can be called if this is articulated
        self.assert_articulated()

        if self._handle is None:
            raise Exception("handles are not initialized yet")
        joint_velocities = self._dc.get_articulation_dof_states(self._handle, _dynamic_control.STATE_VEL)
        joint_velocities = [joint_velocities[i][1] for i in range(len(joint_velocities))]
        return np.array(joint_velocities)

    def get_joint_efforts(self) -> np.ndarray:
        """[summary]

        Raises:
            Exception: [description]

        Returns:
            np.ndarray: [description]
        """
        # Only can be called if this is articulated
        self.assert_articulated()

        if self._handle is None:
            raise Exception("handles are not initialized yet")
        joint_efforts = self._dc.get_articulation_dof_states(self._handle, _dynamic_control.STATE_EFFORT)
        joint_efforts = [joint_efforts[i][2] for i in range(len(joint_efforts))]
        return joint_efforts

    def set_joints_default_state(
        self,
        positions: Optional[np.ndarray] = None,
        velocities: Optional[np.ndarray] = None,
        efforts: Optional[np.ndarray] = None,
    ) -> None:
        """[summary]

        Args:
            positions (Optional[np.ndarray], optional): [description]. Defaults to None.
            velocities (Optional[np.ndarray], optional): [description]. Defaults to None.
            efforts (Optional[np.ndarray], optional): [description]. Defaults to None.
        """
        # Only can be called if this is articulated
        self.assert_articulated()

        if positions is not None:
            self._default_joints_state.positions = positions
        if velocities is not None:
            self._default_joints_state.velocities = velocities
        if efforts is not None:
            self._default_joints_state.efforts = efforts
        return

    def get_joints_state(self) -> JointsState:
        """[summary]

        Returns:
            JointsState: [description]
        """
        # Only can be called if this is articulated
        self.assert_articulated()

        return JointsState(
            positions=self.get_joint_positions(),
            velocities=self.get_joint_velocities(),
            efforts=self.get_joint_efforts(),
        )

    def reset(self):
        # Run super reset first to reset this articulation's pose
        super().reset()

        # Reset joint state if we're articulated
        if self.articulated:
            self.reset_joint_states()

    def reset_joint_states(self):
        """
        Resets the joint state based on self._default_joints_state
        """
        # Only can be called if this is articulated
        self.assert_articulated()

        # Reset state
        self.set_joint_positions(self._default_joints_state.positions)
        self.set_joint_velocities(self._default_joints_state.velocities)
        self.set_joint_efforts(self._default_joints_state.efforts)

    def get_articulation_controller(self) -> ArticulationController:
        """
        Returns:
            ArticulationController: PD Controller of all degrees of freedom of an articulation, can apply position targets, velocity targets and efforts.
        """
        # Only can be called if this is articulated
        self.assert_articulated()

        return self._articulation_controller

    def set_linear_velocity(self, velocity: np.ndarray):
        """Sets the linear velocity of the root prim in stage.

        Args:
            velocity (np.ndarray):linear velocity to set the rigid prim to. Shape (3,).
        """
        self.root_link.set_linear_velocity(velocity)

    def get_linear_velocity(self) -> np.ndarray:
        """[summary]

        Returns:
            np.ndarray: [description]
        """
        return self.root_link.get_linear_velocity()

    def set_angular_velocity(self, velocity: np.ndarray) -> None:
        """[summary]

        Args:
            velocity (np.ndarray): [description]
        """
        self.root_link.set_angular_velocity(velocity)

    def get_angular_velocity(self) -> np.ndarray:
        """[summary]

        Returns:
            np.ndarray: [description]
        """
        return self.root_link.get_angular_velocity()

    def set_position_orientation(self, position: Optional[np.ndarray] = None, orientation: Optional[np.ndarray] = None) -> None:
        """Sets prim's pose with respect to the world's frame.

        Args:
            position (Optional[np.ndarray], optional): position in the world frame of the prim. shape is (3, ).
                                                       Defaults to None, which means left unchanged.
            orientation (Optional[np.ndarray], optional): quaternion orientation in the world frame of the prim.
                                                          quaternion is scalar-last (x, y, z, w). shape is (4, ).
                                                          Defaults to None, which means left unchanged.
        """
        if self._root_handle is not None and self._dc.is_simulating():
            current_position, current_orientation = self.get_position_orientation()
            if position is None:
                position = current_position
            if orientation is None:
                orientation = current_orientation
            pose = _dynamic_control.Transform(position, orientation)
            self._dc.set_rigid_body_pose(self._root_handle, pose)
        else:
            super().set_position_orientation(position=position, orientation=orientation)

    def get_position_orientation(self) -> Tuple[np.ndarray, np.ndarray]:
        """Gets prim's pose with respect to the world's frame.

        Returns:
            Tuple[np.ndarray, np.ndarray]: first index is position in the world frame of the prim. shape is (3, ).
                                           second index is quaternion orientation in the world frame of the prim.
                                           quaternion is scalar-last (x, y, z, w). shape is (4, ).
        """
        if self._root_handle is not None and self._dc.is_simulating():
            pose = self._dc.get_rigid_body_pose(self._root_handle)
            return np.asarray(pose.p), np.asarray(pose.r)
        else:
            return super().get_position_orientation()

    def set_local_pose(
        self, translation: Optional[np.ndarray] = None, orientation: Optional[np.ndarray] = None
    ) -> None:
        """Sets prim's pose with respect to the local frame (the prim's parent frame).

        Args:
            translation (Optional[np.ndarray], optional): translation in the local frame of the prim
                                                          (with respect to its parent prim). shape is (3, ).
                                                          Defaults to None, which means left unchanged.
            orientation (Optional[np.ndarray], optional): quaternion orientation in the world frame of the prim.
                                                          quaternion is scalar-last (x, y, z, w). shape is (4, ).
                                                          Defaults to None, which means left unchanged.
        """
        if self._root_handle is not None and self._dc.is_simulating():
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
            return
        else:
            super().set_local_pose(translation=translation, orientation=orientation)
            return

    def get_local_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Gets prim's pose with respect to the local frame (the prim's parent frame).

        Returns:
            Tuple[np.ndarray, np.ndarray]: first index is position in the local frame of the prim. shape is (3, ).
                                           second index is quaternion orientation in the local frame of the prim.
                                           quaternion is scalar-last (x, y, z, w). shape is (4, ).
        """
        if self._root_handle is not None and self._dc.is_simulating():
            parent_world_tf = UsdGeom.Xformable(get_prim_parent(self._prim)).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            world_position, world_orientation = self.get_position_orientation()
            my_world_transform = tf_matrix_from_pose(translation=world_position, orientation=world_orientation[[3, 0, 1, 2]])
            local_transform = np.matmul(np.linalg.inv(np.transpose(parent_world_tf)), my_world_transform)
            transform = Gf.Transform()
            transform.SetMatrix(Gf.Matrix4d(np.transpose(local_transform)))
            calculated_translation = transform.GetTranslation()
            calculated_orientation = transform.GetRotation().GetQuat()
            return np.array(calculated_translation), gf_quat_to_np_array(calculated_orientation)[[1, 2, 3, 0]]
        else:
            return super().get_local_pose()

    def apply_action(
        self, control_actions: ArticulationAction, indices: Optional[Union[List, np.ndarray]] = None
    ) -> None:
        """[summary]

        Args:
            control_actions (ArticulationAction): actions to be applied for next physics step.
            indices (Optional[Union[list, np.ndarray]], optional): degree of freedom indices to apply actions to.
                                                                   Defaults to all degrees of freedom.

        Raises:
            Exception: [description]
        """
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        if self._handle is None:
            raise Exception("handles are not initialized yet")
        self._articulation_controller.apply_action(control_actions=control_actions, indices=indices)
        return

    def get_applied_action(self) -> ArticulationAction:
        """[summary]

        Raises:
            Exception: [description]

        Returns:
            ArticulationAction: [description]
        """
        if self._handle is None:
            raise Exception("handles are not initialized yet")
        return self._articulation_controller.get_applied_action()

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
    def solver_position_iteration_count(self) -> int:
        """[summary]

        Returns:
            int: [description]
        """
        return get_prim_property(self.prim_path, "physxArticulation:solverPositionIterationCount")

    @solver_position_iteration_count.setter
    def solver_position_iteration_count(self, count: int) -> None:
        """[summary]

        Args:
            count (int): [description]
        """
        set_prim_property(self.prim_path, "physxArticulation:solverPositionIterationCount", count)
        return

    @property
    def solver_velocity_iteration_count(self) -> int:
        """[summary]

        Returns:
            int: [description]
        """
        return get_prim_property(self.prim_path, "physxArticulation:solverVelocityIterationCount")

    @solver_velocity_iteration_count.setter
    def solver_velocity_iteration_count(self, count: int):
        """[summary]

        Args:
            count (int): [description]
        """
        set_prim_property(self.prim_path, "physxArticulation:solverVelocityIterationCount", count)
        return

    @property
    def stabilization_threshold(self) -> float:
        """[summary]

        Returns:
            float: [description]
        """
        return get_prim_property(self.prim_path, "physxArticulation:stabilizationThreshold")

    @stabilization_threshold.setter
    def stabilization_threshold(self, threshold: float) -> None:
        """[summary]

        Args:
            threshold (float): [description]
        """
        set_prim_property(self.prim_path, "physxArticulation:stabilizationThreshold", threshold)
        return

    @property
    def enabled_self_collisions(self) -> bool:
        """[summary]

        Returns:
            bool: [description]
        """
        return get_prim_property(self.prim_path, "physxArticulation:enabledSelfCollisions")

    @enabled_self_collisions.setter
    def enabled_self_collisions(self, flag: bool) -> None:
        """[summary]

        Args:
            flag (bool): [description]
        """
        set_prim_property(self.prim_path, "physxArticulation:enabledSelfCollisions", flag)
        return

    @property
    def sleep_threshold(self) -> float:
        """[summary]

        Returns:
            float: [description]
        """
        return get_prim_property(self.prim_path, "physxArticulation:sleepThreshold")

    @sleep_threshold.setter
    def sleep_threshold(self, threshold: float) -> None:
        """[summary]

        Args:
            threshold (float): [description]
        """
        set_prim_property(self.prim_path, "physxArticulation:sleepThreshold", threshold)
        return

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

    def save_state(self):
        # Iterate over all links and joints
        link_states = [link.save_state() for link in self._links.values()]
        joint_states = [joint.save_state() for joint in self._joints.values()]
        return np.concatenate([*link_states, *joint_states])
