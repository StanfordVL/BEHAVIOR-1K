import math
from functools import cached_property
from typing import Literal

import networkx as nx
import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.macros import create_module_macros, gm
from omnigibson.prims.cloth_prim import ClothPrim
from omnigibson.prims.joint_prim import JointPrim
from omnigibson.prims.rigid_dynamic_prim import RigidDynamicPrim
from omnigibson.prims.rigid_kinematic_prim import RigidKinematicPrim
from omnigibson.prims.xform_prim import XFormPrim
from omnigibson.utils.constants import JointAxis, JointType, PrimType
from omnigibson.utils.render_utils import force_pbr_material_for_link
from omnigibson.utils.usd_utils import PoseAPI, absolute_prim_path_to_scene_relative

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Default sleep threshold for all objects -- see https://docs.omniverse.nvidia.com/extensions/latest/ext_physics/simulation-control/physics-settings.html?highlight=sleep#sleeping
# Mass-normalized kinetic energy threshold below which an actor may go to sleep
m.DEFAULT_SLEEP_THRESHOLD = 0.00005


class EntityPrim(XFormPrim):
    """
    Provides high level functions to deal with an articulation prim and its attributes/ properties. Note that this
    type of prim cannot be created from scratch, and assumes there is already a pre-existing prim tree that should
    be converted into an articulation!

    Args:
        relative_prim_path (str): Scene-local prim path of the Prim to encapsulate or create.
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
        relative_prim_path,
        name,
        load_config=None,
    ):
        # Other values that will be filled in at runtime
        self._root_link_name = None  # Name of the root link
        self._n_dof = None
        self._links = None
        self._joints = None
        self._materials = None
        self._visual_only = None
        self._articulated = None
        self._articulation_tree = None
        self._articulation_view_direct = None

        # This needs to be initialized to be used for _load() of PrimitiveObject
        self._prim_type = (
            load_config["prim_type"] if load_config is not None and "prim_type" in load_config else PrimType.RIGID
        )
        assert self._prim_type in iter(PrimType), f"Unknown prim type {self._prim_type}!"

        # Run super init
        super().__init__(
            relative_prim_path=relative_prim_path,
            name=name,
            load_config=load_config,
        )

    def _initialize(self):
        # Run super method
        super()._initialize()

        if not self.kinematic_only:
            # Set the default sleep threshold
            self.sleep_threshold = m.DEFAULT_SLEEP_THRESHOLD

        # Initialize all the links
        for link in self._links.values():
            link.initialize()

        # Update joint information
        self.update_joints()

    def _load(self):
        # By default, this prim cannot be instantiated from scratch!
        raise NotImplementedError("By default, an entity prim cannot be created from scratch.")

    def _post_load(self):
        # If this is a cloth, delete the root link and replace it with the single nested mesh
        if self._prim_type == PrimType.CLOTH:
            # Verify only a single link and a single mesh exists
            old_link_prim = None
            cloth_mesh_prim = None
            for prim in self._prim.GetChildren():
                if prim.GetPrimTypeInfo().GetTypeName() == "Xform":
                    assert old_link_prim is None, "Found multiple XForm links for a Cloth entity prim! Expected: 1"
                    old_link_prim = prim
                    for child in prim.GetChildren():
                        if child.GetPrimTypeInfo().GetTypeName() == "Mesh" and not child.HasAPI(
                            lazy.pxr.UsdPhysics.CollisionAPI
                        ):
                            assert cloth_mesh_prim is None, "Found multiple meshes for a Cloth entity prim! Expected: 1"
                            cloth_mesh_prim = child

            # Move mesh prim one level up via copy, then delete the original link
            # NOTE: We copy because we cannot directly move the prim because it is ancestral
            # NOTE: We use this specific delete method because alternative methods (eg: "delete_prim") fail beacuse
            # the prim is ancestral. Note that because it is non-destructive, the original link prim path is still
            # tracked by omni, so we have to utilize a new unique prim path for the copied cloth mesh
            # See omni.kit.context_menu module for reference
            new_path = f"{self.prim_path}/{old_link_prim.GetName()}_cloth"
            lazy.omni.kit.commands.execute("CopyPrim", path_from=cloth_mesh_prim.GetPath(), path_to=new_path)
            lazy.omni.kit.commands.execute("DeletePrims", paths=[old_link_prim.GetPath()], destructive=False)

        self.update_links()
        self._compute_articulation_tree()

        # Prepare the articulation view.
        if self.n_joints > 0:
            # Import now to avoid too-eager load of Omni classes due to inheritance
            from omnigibson.utils.deprecated_utils import ArticulationView

            self._articulation_view_direct = ArticulationView(f"{self.prim_path}/{self.root_link_name}")

        # Set visual only flag
        # This automatically handles setting collisions / gravity appropriately per-link
        self.visual_only = (
            self._load_config["visual_only"]
            if "visual_only" in self._load_config and self._load_config["visual_only"] is not None
            else False
        ) or gm.VISUAL_ONLY

        if self._prim_type == PrimType.CLOTH:
            assert not self._visual_only, "Cloth cannot be visual-only."
            assert len(self._links) == 1, f"Cloth entity prim can only have one link; got: {len(self._links)}"
            if gm.AG_CLOTH:
                self.create_attachment_point_link()

        # Globally disable any requested collision links
        for link_name in self.disabled_collision_link_names:
            self._links[link_name].disable_collisions()

        # Disable any requested collision pairs
        for a_name, b_name in self.disabled_collision_pairs:
            link_a, link_b = self._links[a_name], self._links[b_name]
            link_a.add_filtered_collision_pair(prim=link_b)

        # Run super
        super()._post_load()

        assert th.all(self.original_scale == 1.0), "scale should be [1, 1, 1] at the EntityPrim (object) level"

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

        # Cache weather we are articulated or not
        self._articulated = self.articulation_root_path is not None

    def remove(self):
        # First remove all joints
        if self._joints is not None:
            for joint in self._joints.values():
                joint.remove()

        # Then links
        if self._links is not None:
            for link in self._links.values():
                link.remove()

        # Finally, remove this prim
        super().remove()

    def update_links(self):
        """
        Helper function to refresh owned joints. Useful for synchronizing internal data if
        additional bodies are added manually
        """
        # We iterate over all children of this object's prim,
        # and grab any that are presumed to be rigid bodies (i.e.: other Xforms)
        joint_children = set()
        # Keep track of all the links we will create. We can't create that just yet because we need to find
        # the base link first.
        links_to_create = {}
        for prim in self._prim.GetChildren():
            link_name = prim.GetName()
            prim_type_name = prim.GetPrimTypeInfo().GetTypeName()

            # Identify links based on prim type
            if self._prim_type == PrimType.RIGID and prim_type_name == "Xform":
                # For rigid body objects, process Xforms as potential rigid links
                # Mark this as a link to create (we'll determine exact class later)
                links_to_create[link_name] = (PrimType.RIGID, prim)

                # Also iterate through all children to infer joints and determine the children of those joints
                # We will use this info to infer which link is the base link!
                for child_prim in prim.GetChildren():
                    if "joint" in child_prim.GetPrimTypeInfo().GetTypeName().lower():
                        # Store the child target of this joint
                        relationships = {r.GetName(): r for r in child_prim.GetRelationships()}
                        # Only record if this is NOT a fixed link tying us to the world (i.e.: no target for body0)
                        if len(relationships["physics:body0"].GetTargets()) > 0:
                            joint_children.add(relationships["physics:body1"].GetTargets()[0].pathString.split("/")[-1])

            elif self._prim_type == PrimType.CLOTH and prim_type_name == "Mesh":
                # For cloth objects, process Meshes as cloth links
                links_to_create[link_name] = (PrimType.CLOTH, prim)

        # Infer the correct root link name -- this corresponds to whatever link does not have any joint existing
        # in the children joints
        valid_root_links = list(set(links_to_create.keys()) - joint_children)

        assert len(valid_root_links) == 1, (
            f"Exactly one single root link should have been found for {self.name}, "
            f"but found none/multiple instead: {valid_root_links}"
        )
        self._root_link_name = valid_root_links[0] if len(valid_root_links) == 1 else "base_link"

        # Now actually create the links with appropriate classes
        self._links = dict()
        for link_name, (link_type, prim) in links_to_create.items():
            # Determine link configuration
            is_root_link = link_name == self._root_link_name
            # Fixed child links of kinematic-only objects are not kinematic-only, to avoid the USD error:
            # PhysicsUSD: CreateJoint - cannot create a joint between static bodies, joint prim: ...
            is_kinematic = self._load_config.get("kinematic_only", False) if is_root_link else False

            link_load_config = {
                "kinematic_only": is_kinematic,
                "belongs_to_articulation": self._articulation_view is not None and not is_root_link,
                "remesh": self._load_config.get("remesh", True),
                "xform_props_pre_loaded": self._load_config.get("xform_props_pre_loaded", False),
                "scale": self._load_config.get("scale", None),
            }

            # Determine the correct class based on link type and kinematic property
            if link_type == PrimType.RIGID:
                link_cls = RigidKinematicPrim if is_kinematic else RigidDynamicPrim
            else:  # link_type == PrimType.CLOTH
                link_cls = ClothPrim

            # Apply the V-Ray to PBR material change if request by the macro
            if gm.USE_PBR_MATERIALS:
                force_pbr_material_for_link(self._prim, link_name)

            # Create and load the link
            self._links[link_name] = link_cls(
                relative_prim_path=absolute_prim_path_to_scene_relative(self.scene, prim.GetPrimPath().__str__()),
                name=f"{self._name}:{link_name}",
                load_config=link_load_config,
            )
            self._links[link_name].load(self.scene)

    def update_joints(self):
        """
        Helper function to refresh owned joints. Useful for synchronizing internal data if
        additional bodies are added manually
        """
        # Initialize joints dictionary
        self._joints = dict()
        self.update_handles()

        # Handle case separately based on whether we are actually articulated or not
        if self._articulation_view and not self.kinematic_only:
            self._n_dof = self._articulation_view.num_dof

            # Additionally grab DOF info if we have non-fixed joints
            if self._n_dof > 0:
                for i in range(self._articulation_view._metadata.joint_count):
                    # Only add the joint if it's not fixed (i.e.: it has DOFs > 0)
                    if self._articulation_view._metadata.joint_dof_counts[i] > 0:
                        joint_name = self._articulation_view._metadata.joint_names[i]
                        joint_dof_offset = self._articulation_view._metadata.joint_dof_offsets[i]
                        joint_path = self._articulation_view._dof_paths[0][joint_dof_offset]
                        joint = JointPrim(
                            relative_prim_path=absolute_prim_path_to_scene_relative(self.scene, joint_path),
                            name=f"{self._name}:joint_{joint_name}",
                            load_config={"driven": self.is_driven},
                            articulation_view=self._articulation_view_direct,
                        )
                        joint.load(self.scene)
                        joint.initialize()
                        self._joints[joint_name] = joint
        else:
            # TODO: May need to extend to clusters of rigid bodies, that aren't exactly joined
            # We assume this object contains a single rigid body
            self._n_dof = 0

        assert self.n_joints == len(self._joints), (
            f"Number of joints inferred from prim tree ({self.n_joints}) does not match number of joints "
            f"found in the articulation view ({len(self._joints)})!"
        )

        self._update_joint_limits()

    def _update_joint_limits(self):
        """
        Helper function to update internal joint limits for prismatic joints based on the object's scale
        """
        # If the scale is [1, 1, 1], we can skip this step
        if th.allclose(self.scale, th.ones(3)):
            return

        prismatic_joints = {
            j_name: j for j_name, j in self._joints.items() if j.joint_type == JointType.JOINT_PRISMATIC
        }

        # If there are no prismatic joints, we can skip this step
        if len(prismatic_joints) == 0:
            return

        uniform_scale = th.allclose(self.scale, self.scale[0])

        for joint_name, joint in prismatic_joints.items():
            if uniform_scale:
                scale_along_axis = self.scale[0]
            else:
                assert (
                    not self.initialized
                ), "Cannot update joint limits for a non-uniformly scaled object when already initialized."
                for link in self.links.values():
                    if joint.body0 == link.prim_path:
                        # Find the parent link frame orientation in the object frame. Note that we
                        # are OK getting this from XFormPrim since we actually want it relative to
                        # the object frame, notwithstanding the physics.
                        _, link_local_orn = XFormPrim.get_position_orientation(link, frame="parent")

                        # Find the joint frame orientation in the parent link frame
                        joint_local_orn = th.tensor(
                            lazy.isaacsim.core.utils.rotations.gf_quat_to_np_array(
                                joint.get_attribute("physics:localRot0")
                            )[[1, 2, 3, 0]],
                            dtype=th.float32,
                        )

                        # Compute the joint frame orientation in the object frame
                        joint_orn = T.quat_multiply(quaternion1=joint_local_orn, quaternion0=link_local_orn)

                        # assert T.check_quat_right_angle(joint_orn), \
                        #     f"Objects that are NOT uniformly scaled requires all joints to have orientations that " \
                        #     f"are factors of 90 degrees! Got orn: {joint_orn} for object {self.name}"

                        # Find the joint axis unit vector (e.g. [1, 0, 0] for "X", [0, 1, 0] for "Y", etc.)
                        axis_in_joint_frame = th.zeros(3)
                        axis_in_joint_frame[JointAxis.index(joint.axis)] = 1.0

                        # Compute the joint axis unit vector in the object frame
                        axis_in_obj_frame = T.quat2mat(joint_orn) @ axis_in_joint_frame

                        # Find the correct scale along the joint axis direction
                        scale_along_axis = self.scale[th.argmax(th.abs(axis_in_obj_frame))]

            joint.lower_limit = joint.lower_limit * scale_along_axis
            joint.upper_limit = joint.upper_limit * scale_along_axis

    @property
    def is_driven(self) -> bool:
        """
        Returns:
            bool: Whether this object is actively controlled/driven or not
        """
        return False

    @property
    def _articulation_view(self):
        if self._articulation_view_direct is None:
            return None

        # Validate that the articulation view is initialized and that if physics is running, the
        # view is valid.
        if og.sim.is_playing() and self.initialized:
            if not self._articulation_view_direct.is_physics_handle_valid():
                og.sim.update_handles()

        return self._articulation_view_direct

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
        # Note that this is not equivalent to self.n_joints > 0 because articulation root path is
        # overridden by the object classes
        assert self._articulated is not None, "Articulation state not initialized!"
        return self._articulated

    @property
    def articulation_root_path(self):
        """
        Returns:
            None or str: Absolute USD path to the expected prim that represents the articulation root, if it exists. By default,
                this corresponds to self.prim_path
        """
        return self.prim_path if self.n_joints > 0 else None

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
            num = len(self._joints)
        else:
            # Manually iterate over all links and check for any joints that are not fixed joints!
            num = 0
            children = list(self.prim.GetChildren())
            while children:
                child_prim = children.pop()
                children.extend(child_prim.GetChildren())
                prim_type = child_prim.GetPrimTypeInfo().GetTypeName().lower()
                if "joint" in prim_type and "fixed" not in prim_type:
                    num += 1
        return num

    @cached_property
    def n_fixed_joints(self):
        """
        Returns:
        int: Number of fixed joints owned by this articulation
        """
        # If the articulation view is available, use it.
        if self._articulation_view and self._articulation_view._metadata:
            return sum(1 for joint_dof in self._articulation_view._metadata.joint_dof_counts if joint_dof == 0)

        # Manually iterate over all links and check for any joints that are not fixed joints!
        num = 0
        children = list(self.prim.GetChildren())
        while children:
            child_prim = children.pop()
            children.extend(child_prim.GetChildren())
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

    @cached_property
    def link_prim_paths(self):
        return [link.prim_path for link in self._links.values()]

    @cached_property
    def has_attachment_points(self):
        """
        Returns:
            bool: Whether this object has any attachment points
        """
        children = list(self.prim.GetChildren())
        while children:
            child_prim = children.pop()
            children.extend(child_prim.GetChildren())
            if "attachment" in child_prim.GetName():
                return True
        return False

    def _compute_articulation_tree(self):
        """
        Get a graph of the articulation tree, where nodes are link names and edges
        correspond to joint names, where the joint name is accessible on the `joint_name`
        data field of the edge, and the joint type on the `joint_type` field.
        """
        G = nx.DiGraph()
        rename_later = {}

        # Add the links
        for link_name, link in self.links.items():
            prim_path = link.prim_path
            G.add_node(prim_path)
            rename_later[prim_path] = link_name

        # Add the joints
        children = list(self.prim.GetChildren())
        while children:
            child_prim = children.pop()
            children.extend(child_prim.GetChildren())
            prim_type = child_prim.GetPrimTypeInfo().GetTypeName()
            if "Joint" in prim_type:
                # Get body 0
                body0_targets = child_prim.GetRelationship("physics:body0").GetTargets()
                if not body0_targets:
                    continue
                body0 = str(body0_targets[0])

                # Get body 1
                body1_targets = child_prim.GetRelationship("physics:body1").GetTargets()
                if not body1_targets:
                    continue
                body1 = str(body1_targets[0])

                # Assert both bodies in links
                if body0 not in G.nodes or body1 not in G.nodes:
                    continue

                # Add the joint
                joint_type = JointType.get_type(prim_type.split("Physics")[-1])
                G.add_edge(body0, body1, joint_name=child_prim.GetName(), joint_type=joint_type)

        # Relabel nodes to use link name instead of prim path
        nx.relabel_nodes(G, rename_later, copy=False)

        # Assert all nodes have in-degree of 1 except root
        in_degrees = {node: G.in_degree(node) for node in G.nodes}
        assert in_degrees[self.root_link_name] == 0, "Root link should have in-degree of 0!"
        assert all(
            [in_degrees[node] == 1 for node in G.nodes if node != self.root_link_name]
        ), "All non-root links should have in-degree of 1!"

        self._articulation_tree = G

    @property
    def articulation_tree(self):
        return self._articulation_tree

    def get_fixed_link_names_in_subtree(self, subtree_root_link_name=None):
        """
        Find all the links that are fixed to a given search subtree root link.

        If the subtree root link name is not provided, the object's root link will be used.
        """
        if subtree_root_link_name is None:
            subtree_root_link_name = self.root_link_name
        is_edge_fixed = lambda f, t: self.articulation_tree[f][t]["joint_type"] == JointType.JOINT_FIXED
        only_fixed_joints = nx.subgraph_view(self.articulation_tree, filter_edge=is_edge_fixed)
        return nx.descendants(only_fixed_joints, subtree_root_link_name) | {subtree_root_link_name}

    @property
    def materials(self):
        """
        Loop through each link and their visual meshes to gather all the materials that belong to this object

        Returns:
            set of MaterialPrim: a set of MaterialPrim that belongs to this object
        """
        return self._materials

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
        NOTE: This method is slow and uncached, but it works even for sleeping objects.
        For frequent contact checks, consider using RigidContactAPI for performance.

        Returns:
            list of CsRawData: raw contact info for this rigid body
        """
        contacts = []
        for link in self._links.values():
            contacts.extend(link.contact_list())
        return contacts

    def enable_gravity(self) -> None:
        """
        Enables gravity for this entity
        """
        for link in self._links.values():
            if isinstance(link, RigidDynamicPrim):
                link.enable_gravity()

    def disable_gravity(self) -> None:
        """
        Disables gravity for this entity
        """
        for link in self._links.values():
            if isinstance(link, RigidDynamicPrim):
                link.disable_gravity()

    def reset(self):
        """
        Resets this entity to some default, pre-defined state
        """
        # Make sure simulation is playing, otherwise, we cannot reset because physx requires active running
        # simulation in order to set joints
        assert og.sim.is_playing(), "Simulator must be playing in order to reset controllable object's joints!"

        # If this is a cloth, reset the particle positions
        if self.prim_type == PrimType.CLOTH:
            self.root_link.reset()

        # Otherwise, set all joints to have 0 position and 0 velocity if this object has joints
        elif self.n_joints > 0:
            self.set_joint_positions(positions=th.zeros(self.n_dof), drive=False)
            self.set_joint_velocities(velocities=th.zeros(self.n_dof), drive=False)

    def set_joint_positions(self, positions, indices=None, normalized=False, drive=False):
        """
        Set the joint positions (both actual value and target values) in simulation. Note: only works if the simulator
        is actively running!

        Args:
            positions (th.tensor): positions to set. This should be n-DOF length if all joints are being set,
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
        # Run sanity checks -- make sure that we are articulated
        assert self.n_joints > 0, "Tried to call method not intended for entity prim with no joints!"

        # Possibly de-normalize the inputs
        if normalized:
            positions = self._denormalize_positions(positions=positions, indices=indices)

        # Set the DOF states
        if drive:
            self._articulation_view.set_joint_position_targets(positions, joint_indices=indices)
        else:
            self._articulation_view.set_joint_positions(positions, joint_indices=indices)
            PoseAPI.invalidate()

    def set_joint_velocities(self, velocities, indices=None, normalized=False, drive=False):
        """
        Set the joint velocities (both actual value and target values) in simulation. Note: only works if the simulator
        is actively running!

        Args:
            velocities (th.tensor): velocities to set. This should be n-DOF length if all joints are being set,
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
        # Run sanity checks -- make sure we are articulated
        assert self.n_joints > 0, "Tried to call method not intended for entity prim with no joints!"

        # Possibly de-normalize the inputs
        if normalized:
            velocities = self._denormalize_velocities(velocities=velocities, indices=indices)

        # Set the DOF states
        if drive:
            self._articulation_view.set_joint_velocity_targets(velocities, joint_indices=indices)
        else:
            self._articulation_view.set_joint_velocities(velocities, joint_indices=indices)

    def set_joint_efforts(self, efforts, indices=None, normalized=False):
        """
        Set the joint efforts (both actual value and target values) in simulation. Note: only works if the simulator
        is actively running!

        Args:
            efforts (th.tensor): efforts to set. This should be n-DOF length if all joints are being set,
                or k-length (k < n) if specific indices are being set. In this case, the length of @efforts must
                be the same length as @indices!
            indices (None or k-array): If specified, should be k (k < n) length array of specific DOF efforts to set.
                Default is None, which assumes that all joints are being set.
            normalized (bool): Whether the inputted joint efforts should be interpreted as normalized values. Default
                is False
        """
        # Run sanity checks -- make sure we are articulated
        assert self.n_joints > 0, "Tried to call method not intended for entity prim with no joints!"

        # Possibly de-normalize the inputs
        if normalized:
            efforts = self._denormalize_efforts(efforts=efforts, indices=indices)

        # Set the DOF states
        self._articulation_view.set_joint_efforts(efforts, joint_indices=indices)

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
        return (
            velocities / self.max_joint_velocities
            if indices is None
            else velocities / self.max_joint_velocities[indices]
        )

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
        return (
            velocities * self.max_joint_velocities
            if indices is None
            else velocities * self.max_joint_velocities[indices]
        )

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

        # Reinitialize the articulation view
        if self._articulation_view_direct is not None:
            self._articulation_view_direct.initialize(og.sim.physics_sim_view)

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
        # Run sanity checks -- make sure we are articulated
        assert self.n_joints > 0, "Tried to call method not intended for entity prim with no joints!"

        joint_positions = self._articulation_view.get_joint_positions().view(self.n_dof)

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
        # Run sanity checks -- make sure we are articulated
        assert self.n_joints > 0, "Tried to call method not intended for entity prim with no joints!"

        joint_velocities = self._articulation_view.get_joint_velocities().view(self.n_dof)

        # Possibly normalize values when returning
        return self._normalize_velocities(velocities=joint_velocities) if normalized else joint_velocities

    def get_joint_efforts(self, normalized=False):
        """
        Grabs this entity's "measured" joint efforts

        Args:
            normalized (bool): Whether returned values should be normalized to range [-1, 1] based on limits or not.

        Returns:
            n-array: n-DOF length array of efforts
        """
        # Run sanity checks -- make sure we are articulated
        assert self.n_joints > 0, "Tried to call method not intended for entity prim with no joints!"

        joint_efforts = self._articulation_view.get_measured_joint_efforts().view(self.n_dof)

        # Possibly normalize values when returning
        return self._normalize_efforts(efforts=joint_efforts) if normalized else joint_efforts

    def get_joint_position_targets(self, normalized=False):
        """
        Grabs this entity's joint position targets

        Args:
            normalized (bool): Whether returned values should be normalized to range [-1, 1] based on limits or not.

        Returns:
            n-array: n-DOF length array of position targets
        """
        # Run sanity checks -- make sure we are articulated
        assert self.n_joints > 0, "Tried to call method not intended for entity prim with no joints!"

        joint_positions = self._articulation_view.get_joint_position_targets().view(self.n_dof)

        # Possibly normalize values when returning
        return self._normalize_positions(positions=joint_positions) if normalized else joint_positions

    def get_joint_velocity_targets(self, normalized=False):
        """
        Grabs this entity's joint velocity targets

        Args:
            normalized (bool): Whether returned values should be normalized to range [-1, 1] based on limits or not.

        Returns:
            n-array: n-DOF length array of velocity targets
        """
        # Run sanity checks -- make sure we are articulated
        assert self.n_joints > 0, "Tried to call method not intended for entity prim with no joints!"

        joint_velocities = self._articulation_view.get_joint_velocity_targets().view(self.n_dof)

        # Possibly normalize values when returning
        return self._normalize_velocities(velocities=joint_velocities) if normalized else joint_velocities

    def set_linear_velocity(self, velocity: th.Tensor):
        """
        Sets the linear velocity of the root prim in stage.

        Args:
            velocity (th.tensor): linear velocity to set the rigid prim to, in the world frame. Shape (3,).
        """
        self.root_link.set_linear_velocity(velocity)

    def get_linear_velocity(self):
        """
        Gets the linear velocity of the root prim in stage.

        Returns:
            velocity (th.tensor): linear velocity to set the rigid prim to, in the world frame. Shape (3,).
        """
        return self.root_link.get_linear_velocity()

    def set_angular_velocity(self, velocity):
        """
        Sets the angular velocity of the root prim in stage.

        Args:
            velocity (th.tensor): angular velocity to set the rigid prim to, in the world frame. Shape (3,).
        """
        self.root_link.set_angular_velocity(velocity)

    def get_angular_velocity(self):
        """Gets the angular velocity of the root prim in stage.

        Returns:
            velocity (th.tensor): angular velocity to set the rigid prim to, in the world frame. Shape (3,).
        """
        return self.root_link.get_angular_velocity()

    def get_relative_linear_velocity(self):
        """
        Returns:
            3-array: (x,y,z) Linear velocity of root link in its own frame
        """
        return T.quat2mat(self.get_position_orientation()[1]).T @ self.get_linear_velocity()

    def get_relative_angular_velocity(self):
        """
        Returns:
            3-array: (ax,ay,az) angular velocity of root link in its own frame
        """
        return T.quat2mat(self.get_position_orientation()[1]).T @ self.get_angular_velocity()

    def set_position_orientation(self, position=None, orientation=None, frame: Literal["world", "scene"] = "world"):
        """
        Set the position and orientation of entry prim object.

        Args:
            position (None or 3-array): The position to set the object to. If None, the position is not changed.
            orientation (None or 4-array): The orientation to set the object to. If None, the orientation is not changed.
            frame (Literal): The frame in which to set the position and orientation. Defaults to world.
                scene frame sets position relative to the scene.
        """
        assert frame in ["world", "scene"], f"Invalid frame '{frame}'. Must be 'world', or 'scene'."

        # If kinematic only, clear cache for the root link
        if self.kinematic_only:
            self.root_link.clear_kinematic_only_cache()

        # If the simulation isn't running, we should set this prim's XForm (object-level) properties directly
        if og.sim.is_stopped():
            return XFormPrim.set_position_orientation(self, position=position, orientation=orientation, frame=frame)

        # Otherwise, we need to set our pose through PhysX.
        # If we are not articulated, we can use the RigidPrim API.
        if self._articulation_view is None:
            return self.root_link.set_position_orientation(position=position, orientation=orientation, frame=frame)

        # Otherwise, we use the articulation view.
        # If no position or no orientation are given, get the current position and orientation of the object
        if position is None or orientation is None:
            current_position, current_orientation = self.get_position_orientation(frame=frame)
        position = current_position if position is None else position
        orientation = current_orientation if orientation is None else orientation

        # Convert to th.Tensor if necessary
        position = th.as_tensor(position, dtype=th.float32)
        orientation = th.as_tensor(orientation, dtype=th.float32)

        # Convert to from scene-relative to world if necessary
        if frame == "scene":
            assert self.scene is not None, "cannot set position and orientation relative to scene without a scene"
            position, orientation = self.scene.convert_scene_relative_pose_to_world(position, orientation)

        # Assert validity of the orientation
        assert math.isclose(
            th.norm(orientation).item(), 1, abs_tol=1e-3
        ), f"{self.prim_path} desired orientation {orientation} is not a unit quaternion."

        # Actually set the pose.
        self._articulation_view.set_world_poses(
            positions=position[None, :], orientations=orientation[None, [3, 0, 1, 2]]
        )

        # Invalidate the pose cache.
        PoseAPI.invalidate()

    def get_position_orientation(self, frame: Literal["world", "scene"] = "world", clone=True):
        """
        Gets prim's pose with respect to the specified frame.

        Args:
            frame (Literal): frame to get the pose with respect to. Default to world.
                scene frame get position relative to the scene.
            clone (bool): Whether to clone the underlying tensor buffer or not

        Returns:
            2-tuple:
                - th.Tensor: (x,y,z) position in the specified frame
                - th.Tensor: (x,y,z,w) quaternion orientation in the specified frame
        """
        assert frame in ["world", "scene"], f"Invalid frame '{frame}'. Must be 'world' or 'scene'."

        # If the simulation isn't running, we should read from this prim's XForm (object-level) properties directly
        if og.sim.is_stopped():
            return XFormPrim.get_position_orientation(self, frame=frame, clone=clone)

        # Delegate to RigidPrim if we are not articulated
        if self._articulation_view is None:
            return self.root_link.get_position_orientation(frame=frame, clone=clone)

        # Otherwise, get the pose from the articulation view and convert to our format
        positions, orientations = self._articulation_view.get_world_poses(clone=clone)
        position = positions[0]
        orientation = orientations[0][[1, 2, 3, 0]]

        # Assert that the orientation is a unit quaternion
        assert math.isclose(
            th.norm(orientations).item(), 1, abs_tol=1e-3
        ), f"{self.prim_path} orientation {orientations} is not a unit quaternion."

        # If requested, compute the scene-local transform
        if frame == "scene":
            assert self.scene is not None, "Cannot get position and orientation relative to scene without a scene"
            position, orientation = self.scene.convert_world_pose_to_scene_relative(position, orientation)

        return position, orientation

    # TODO: Is the omni joint damping (used for driving motors) same as dissipative joint damping (what we had in pb)?
    @property
    def joint_damping(self):
        """
        Returns:
            n-array: joint damping values for this prim
        """
        return th.cat([joint.damping for joint in self._joints.values()])

    # TODO: These are cached, but they are not updated when the joint limit is changed
    @cached_property
    def joint_lower_limits(self):
        """
        Returns:
            n-array: minimum values for this robot's joints. If joint does not have a range, returns -1000
                for that joint
        """
        return th.tensor([joint.lower_limit for joint in self._joints.values()])

    # TODO: These are cached, but they are not updated when the joint limit is changed
    @cached_property
    def joint_upper_limits(self):
        """
        Returns:
            n-array: maximum values for this robot's joints. If joint does not have a range, returns 1000
                for that joint
        """
        return th.tensor([joint.upper_limit for joint in self._joints.values()])

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
        return th.tensor([joint.max_velocity for joint in self._joints.values()])

    @property
    def max_joint_efforts(self):
        """
        Returns:
            n-array: maximum efforts for this robot's joints
        """
        return th.tensor([joint.max_effort for joint in self._joints.values()])

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
        return 1.0 * (th.abs(self.get_joint_positions(normalized=True)) > 0.99)

    @property
    def joint_has_limits(self):
        """
        Returns:
            n-array: n-DOF length array specifying whether joint has a limit or not
        """
        return th.tensor([j.has_limit for j in self._joints.values()])

    @property
    def disabled_collision_link_names(self):
        """
        Returns:
            list of str: List of link names for this entity whose collisions should be globally disabled
        """
        return []

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
        # For the EntityPrim (object) level, @self.scale represents the scale with respect to the original scale of
        # the link (RigidPrim or ClothPrim), which might not be uniform ([1, 1, 1]) itself.
        return self.root_link.scale / self.root_link.original_scale

    @scale.setter
    def scale(self, scale):
        # For the EntityPrim (object) level, @self.scale represents the scale with respect to the original scale of
        # the link (RigidPrim or ClothPrim), which might not be uniform ([1, 1, 1]) itself.
        # We iterate over all rigid bodies owned by this object prim and set their individual scales
        # We do this because omniverse cannot scale orientation of an articulated prim, so we get mesh mismatches as
        # they rotate in the world.
        for link in self._links.values():
            link.scale = scale * link.original_scale

    @property
    def solver_position_iteration_count(self):
        """
        Returns:
            int: How many position iterations to take per physics step by the physx solver
        """
        return (
            lazy.isaacsim.core.utils.prims.get_prim_property(
                self.articulation_root_path, "physxArticulation:solverPositionIterationCount"
            )
            if self.articulated
            else self.root_link.solver_position_iteration_count
        )

    @solver_position_iteration_count.setter
    def solver_position_iteration_count(self, count):
        """
        Sets how many position iterations to take per physics step by the physx solver

        Args:
            count (int): How many position iterations to take per physics step by the physx solver
        """
        if self.articulated:
            lazy.isaacsim.core.utils.prims.set_prim_property(
                self.articulation_root_path, "physxArticulation:solverPositionIterationCount", count
            )
        else:
            for link in self._links.values():
                link.solver_position_iteration_count = count

    @property
    def solver_velocity_iteration_count(self):
        """
        Returns:
            int: How many velocity iterations to take per physics step by the physx solver
        """
        return (
            lazy.isaacsim.core.utils.prims.get_prim_property(
                self.articulation_root_path, "physxArticulation:solverVelocityIterationCount"
            )
            if self.articulated
            else self.root_link.solver_velocity_iteration_count
        )

    @solver_velocity_iteration_count.setter
    def solver_velocity_iteration_count(self, count):
        """
        Sets how many velocity iterations to take per physics step by the physx solver

        Args:
            count (int): How many velocity iterations to take per physics step by the physx solver
        """
        if self.articulated:
            lazy.isaacsim.core.utils.prims.set_prim_property(
                self.articulation_root_path, "physxArticulation:solverVelocityIterationCount", count
            )
        else:
            for link in self._links.values():
                link.solver_velocity_iteration_count = count

    @property
    def stabilization_threshold(self):
        """
        Returns:
            float: threshold for stabilizing this articulation
        """
        return (
            lazy.isaacsim.core.utils.prims.get_prim_property(
                self.articulation_root_path, "physxArticulation:stabilizationThreshold"
            )
            if self.articulated
            else self.root_link.stabilization_threshold
        )

    @stabilization_threshold.setter
    def stabilization_threshold(self, threshold):
        """
        Sets threshold for stabilizing this articulation

        Args:
            threshold (float): Stabilization threshold
        """
        if self.articulated:
            lazy.isaacsim.core.utils.prims.set_prim_property(
                self.articulation_root_path, "physxArticulation:stabilizationThreshold", threshold
            )
        else:
            for link in self._links.values():
                link.stabilization_threshold = threshold

    @property
    def is_asleep(self):
        """
        Returns:
            bool: whether this entity is asleep or not
        """
        # If we're kinematic only, immediately return False since it doesn't follow the sleep / wake paradigm
        if self.kinematic_only or self.prim_type != PrimType.RIGID:
            return False
        else:
            return (
                og.sim.psi.is_sleeping(
                    og.sim.stage_id, lazy.pxr.PhysicsSchemaTools.sdfPathToInt(self.articulation_root_path)
                )
                if self.articulated
                else self.root_link.is_asleep
            )

    @property
    def sleep_threshold(self):
        """
        Returns:
            float: threshold for sleeping this articulation
        """
        return (
            lazy.isaacsim.core.utils.prims.get_prim_property(
                self.articulation_root_path, "physxArticulation:sleepThreshold"
            )
            if self.articulated
            else self.root_link.sleep_threshold
        )

    @sleep_threshold.setter
    def sleep_threshold(self, threshold):
        """
        Sets threshold for sleeping this articulation

        Args:
            threshold (float): Sleeping threshold
        """
        if self.articulated:
            lazy.isaacsim.core.utils.prims.set_prim_property(
                self.articulation_root_path, "physxArticulation:sleepThreshold", threshold
            )
        else:
            for link in self._links.values():
                link.sleep_threshold = threshold

    @property
    def self_collisions(self):
        """
        Returns:
            bool: Whether self-collisions are enabled for this prim or not
        """
        return lazy.isaacsim.core.utils.prims.get_prim_property(
            self.articulation_root_path, "physxArticulation:enabledSelfCollisions"
        )

    @self_collisions.setter
    def self_collisions(self, flag):
        """
        Sets whether self-collisions are enabled for this prim or not

        Args:
            flag (bool): Whether self collisions are enabled for this prim or not
        """
        lazy.isaacsim.core.utils.prims.set_prim_property(
            self.articulation_root_path, "physxArticulation:enabledSelfCollisions", flag
        )

    @cached_property
    def kinematic_only(self):
        """
        Returns:
            bool: Whether this object is a kinematic-only object (otherwise, it is a rigid body). A kinematic-only
                object is not subject to simulator dynamics, and remains fixed unless the user explicitly sets the
                body's pose / velocities. See https://docs.omniverse.nvidia.com/app_create/prod_extensions/ext_physics/rigid-bodies.html?highlight=rigid%20body%20enabled#kinematic-rigid-bodies
                for more information
        """
        return isinstance(self.root_link, RigidKinematicPrim)

    @property
    def collision_points_world(self):
        """Get all collision points in world coordinates.

        For cloth objects, returns particle positions.
        For rigid objects, returns collision boundary points from all links.
        """
        if self._prim_type == PrimType.CLOTH:
            return self.root_link.compute_particle_positions()
        else:
            points_world = [link.collision_boundary_points_world for link in self._links.values()]
            return th.cat([p for p in points_world if p is not None], dim=0)

    @property
    def aabb(self):
        """Compute axis-aligned bounding box from collision mesh points."""
        points = self.collision_points_world
        if self._prim_type == PrimType.CLOTH:
            particle_contact_offset = self.root_link.cloth_system.particle_contact_offset
            aabb_lo = th.min(points, dim=0).values - particle_contact_offset
            aabb_hi = th.max(points, dim=0).values + particle_contact_offset
        else:
            aabb_lo = th.min(points, dim=0).values
            aabb_hi = th.max(points, dim=0).values
        return aabb_lo, aabb_hi

    @property
    def aabb_extent(self):
        """
        Get this xform's actual bounding box extent

        Returns:
            3-array: (x,y,z) bounding box
        """
        min_corner, max_corner = self.aabb
        return max_corner - min_corner

    @property
    def aabb_center(self):
        """
        Get this xform's actual bounding box center

        Returns:
            3-array: (x,y,z) bounding box center
        """
        min_corner, max_corner = self.aabb
        return (max_corner + min_corner) / 2.0

    def get_coriolis_and_centrifugal_forces(self, clone=True):
        """
        Args:
            clone (bool): Whether to clone the underlying tensor buffer or not

        Returns:
            n-array: (N,) shaped per-DOF coriolis and centrifugal forces experienced by the entity, if articulated
        """
        assert self.articulated, "Cannot get coriolis and centrifugal forces for non-articulated entity!"
        return self._articulation_view.get_coriolis_and_centrifugal_forces(clone=clone).view(self.n_dof)

    def get_generalized_gravity_forces(self, clone=True):
        """
        Args:
            clone (bool): Whether to clone the underlying tensor buffer or not

        Returns:
            n-array: (N, N) shaped per-DOF gravity forces, if articulated
        """
        assert self.articulated, "Cannot get generalized gravity forces for non-articulated entity!"
        return self._articulation_view.get_generalized_gravity_forces(clone=clone).view(self.n_dof)

    def get_mass_matrix(self, clone=True):
        """
        Args:
            clone (bool): Whether to clone the underlying tensor buffer or not

        Returns:
            n-array: (N, N) shaped per-DOF mass matrix, if articulated
        """
        assert self.articulated, "Cannot get mass matrix for non-articulated entity!"
        return self._articulation_view.get_mass_matrices(clone=clone).view(self.n_dof, self.n_dof)

    def get_jacobian(self, clone=True):
        """
        Args:
            clone (bool): Whether to clone the underlying tensor buffer or not

        Returns:
            n-array: (N_links - 1 [+ 1], 6, N_dof [+ 6]) shaped per-link jacobian, if articulated. Note that the first
                dimension is +1 and the final dimension is +6 if the entity does not have a fixed base
                (i.e.: there is an additional "floating" joint tying the robot to the world frame)
        """
        assert self.articulated, "Cannot get jacobian for non-articulated entity!"
        return self._articulation_view.get_jacobians(clone=clone).squeeze(dim=0)

    def get_relative_jacobian(self, clone=True):
        """
        Args:
            clone (bool): Whether to clone the underlying tensor buffer or not

        Returns:
            n-array: (N_links - 1 [+ 1], 6, N_dof [+ 6]) shaped per-link relative jacobian, if articulated (expressed in
                this entity's base frame). Note that the first dimension is +1 and the final dimension is +6 if the
                entity does not have a fixed base (i.e.: there is an additional "floating" joint tying the robot to
                the world frame)
        """
        jac = self.get_jacobian(clone=clone)
        ori_t = T.quat2mat(self.get_position_orientation()[1]).T
        tf = th.zeros((1, 6, 6), dtype=th.float32)
        tf[:, :3, :3] = ori_t
        tf[:, 3:, 3:] = ori_t
        return tf @ jac

    def wake(self):
        """
        Enable physics for this articulation
        """
        if self.articulated:
            prim_id = lazy.pxr.PhysicsSchemaTools.sdfPathToInt(self.articulation_root_path)
            og.sim.psi.wake_up(og.sim.stage_id, prim_id)
        else:
            for link in self._links.values():
                if isinstance(link, RigidDynamicPrim):
                    link.wake()

    def sleep(self):
        """
        Disable physics for this articulation
        """
        if self.articulated:
            prim_id = lazy.pxr.PhysicsSchemaTools.sdfPathToInt(self.articulation_root_path)
            og.sim.psi.put_to_sleep(og.sim.stage_id, prim_id)
        else:
            for link in self._links.values():
                if isinstance(link, RigidDynamicPrim):
                    link.sleep()

    def keep_still(self):
        """
        Zero out all velocities for this prim
        """
        if self.kinematic_only:
            return
        self.set_linear_velocity(velocity=th.zeros(3))
        self.set_angular_velocity(velocity=th.zeros(3))
        if self.initialized:
            for joint in self._joints.values():
                joint.keep_still()
        # Make sure object is awake
        self.wake()

    def create_attachment_point_link(self):
        """
        Create a collision-free, invisible attachment point link for the cloth object, and create an attachment between
        the ClothPrim and this attachment point link (RigidPrim).

        One use case for this is that we can create a fixed joint between this link and the world to enable AG fo cloth.
        During simulation, this joint will move and match the robot gripper frame, which will then drive the cloth.
        """

        assert self._prim_type == PrimType.CLOTH, "create_attachment_point_link should only be called for Cloth"
        link_name = "attachment_point"
        stage = lazy.isaacsim.core.utils.stage.get_current_stage()
        link_prim = stage.DefinePrim(f"{self.prim_path}/{link_name}", "Xform")
        vis_prim = lazy.pxr.UsdGeom.Sphere.Define(stage, f"{self.prim_path}/{link_name}/visuals").GetPrim()
        col_prim = lazy.pxr.UsdGeom.Sphere.Define(stage, f"{self.prim_path}/{link_name}/collisions").GetPrim()

        # Set the radius to be 0.03m. In theory, we want this radius to be as small as possible. Otherwise, the cloth
        # dynamics will be unrealistic. However, in practice, if the radius is too small, the attachment becomes very
        # unstable. Empirically 0.03m works reasonably well.
        vis_prim.GetAttribute("radius").Set(0.03)
        col_prim.GetAttribute("radius").Set(0.03)

        # Need to sync the extents
        extent = vis_prim.GetAttribute("extent").Get()
        extent[0] = lazy.pxr.Gf.Vec3f(-0.03, -0.03, -0.03)
        extent[1] = lazy.pxr.Gf.Vec3f(0.03, 0.03, 0.03)
        vis_prim.GetAttribute("extent").Set(extent)
        col_prim.GetAttribute("extent").Set(extent)

        # Add collision API to collision geom
        lazy.pxr.UsdPhysics.CollisionAPI.Apply(col_prim)
        lazy.pxr.UsdPhysics.MeshCollisionAPI.Apply(col_prim)
        lazy.pxr.PhysxSchema.PhysxCollisionAPI.Apply(col_prim)

        # Create a attachment point link
        link = RigidDynamicPrim(
            relative_prim_path=absolute_prim_path_to_scene_relative(self.scene, link_prim.GetPrimPath().pathString),
            name=f"{self._name}:{link_name}",
        )
        link.load(self.scene)
        link.disable_collisions()
        # TODO (eric): Should we disable gravity for this link?
        # link.disable_gravity()
        link.visible = False
        # Set a very small mass
        link.mass = 1e-6
        link.density = 0.0

        self._links[link_name] = link

        # Create an attachment between the root link (ClothPrim) and the newly created attachment point link (RigidDynamicPrim)
        attachment_path = self.root_link.prim.GetPath().AppendElementString("attachment")
        lazy.omni.kit.commands.execute(
            "CreatePhysicsAttachment",
            target_attachment_path=attachment_path,
            actor0_path=self.root_link.prim.GetPath(),
            actor1_path=link.prim.GetPath(),
        )

    def _dump_state(self):
        # We don't call super, instead, this state is simply the root link state and all joint states
        state = dict(is_asleep=self.is_asleep, root_link=self.root_link._dump_state())
        if self.n_joints > 0:
            state["joint_pos"] = self.get_joint_positions()
            state["joint_vel"] = self.get_joint_velocities()

            # We do NOT save joint pos / vel targets because this is only relevant for motorized joints (e.g.: robots).
            # Such control (a) only relies on the joint state, and not joint targets, when computing control, and
            # (b) these targets will be immediately overwritten as soon as the next physics step occurs.
            # In other words, when loading the sim state, we will load the joint pos / vel and the controller state (e.g. goal)
            # and then when we take a physics step, right before the physics step actually occurs, @og.sim._on_physics_step()
            # will be called, which computes the control based on the just-loaded joint state and overwrites the joint targets with the new control.
            # So because these values are not used and require additional memory / compute, we do not save targets

        return state

    def _load_state(self, state):
        # Load base link state and joint states
        self.root_link._load_state(state=state["root_link"])

        # TODO: Remove this backwards-compatible part once we re-sample scenes
        if "joints" in state:
            for joint_name, joint_state in state["joints"].items():
                self._joints[joint_name]._load_state(state=joint_state)

        elif self.n_joints > 0:
            self.set_joint_positions(state["joint_pos"])
            self.set_joint_velocities(state["joint_vel"])

        # Make sure this object is awake if it was not asleep during setting
        # TODO: Remove backwards compatibility once we re-sample scenes
        self.sleep() if state.get("is_asleep", False) else self.wake()

    def serialize(self, state):
        # We serialize by first flattening the root link state and then iterating over all joints and
        # adding them to the a flattened array
        state_flat = [th.tensor([state["is_asleep"]], dtype=th.int), self.root_link.serialize(state=state["root_link"])]
        if self.n_joints > 0:
            state_flat += [
                state["joint_pos"],
                state["joint_vel"],
            ]

        return th.cat(state_flat)

    def deserialize(self, state):
        # Get sleep state first
        is_asleep = bool(state[0].item())
        # We deserialize by first de-flattening the root link state and then iterating over all joints and
        # sequentially grabbing from the flattened state array, incrementing along the way
        root_link_state, idx = self.root_link.deserialize(state=state[1:])
        idx += 1  # Incremented 1 from is_asleep value
        state_dict = dict(is_asleep=is_asleep, root_link=root_link_state)
        if self.n_joints > 0:
            for jnt_state in ("pos", "vel"):
                state_dict[f"joint_{jnt_state}"] = state[idx : idx + self.n_joints]
                idx += self.n_joints

        return state_dict, idx

    def _create_prim_with_same_kwargs(self, relative_prim_path, name, load_config):
        # Subclass must implement this method for duplication functionality
        raise NotImplementedError(
            "Subclass must implement _create_prim_with_same_kwargs() to enable duplication "
            "functionality for EntityPrim!"
        )
