import trimesh.triangles
from omni.isaac.core.utils.prims import get_prim_at_path, get_prim_parent
from omni.isaac.core.utils.transformations import tf_matrix_from_pose
from omni.isaac.core.utils.rotations import gf_quat_to_np_array
from pxr import Gf, UsdPhysics, Usd, UsdGeom, PhysxSchema
import numpy as np
from omni.isaac.dynamic_control import _dynamic_control

from omnigibson.macros import gm, create_module_macros
from omnigibson.prims.xform_prim import XFormPrim
from omnigibson.prims.geom_prim import CollisionGeomPrim, VisualGeomPrim
from omnigibson.utils.constants import GEOM_TYPES
from omnigibson.utils.sim_utils import CsRawData
from omnigibson.utils.usd_utils import get_mesh_volume_and_com
import omnigibson.utils.transform_utils as T
from omnigibson.utils.ui_utils import create_module_logger

# Import omni sensor based on type
from omni.isaac.sensor import _sensor as _s

# Create module logger
log = create_module_logger(module_name=__name__)

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.DEFAULT_CONTACT_OFFSET = 0.001
m.DEFAULT_REST_OFFSET = 0.0


class RigidPrim(XFormPrim):
    """
    Provides high level functions to deal with a rigid body prim and its attributes/ properties.
    If there is an prim present at the path, it will use it. Otherwise, a new XForm prim at
    the specified prim path will be created.

    Notes: if the prim does not already have a rigid body api applied to it before it is loaded,
        it will apply it.

    Args:
        prim_path (str): prim path of the Prim to encapsulate or create.
        name (str): Name for the object. Names need to be unique per scene.
        load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
            loading this prim at runtime. Note that this is only needed if the prim does not already exist at
            @prim_path -- it will be ignored if it already exists. For this joint prim, the below values can be
            specified:

            scale (None or float or 3-array): If specified, sets the scale for this object. A single number corresponds
                to uniform scaling along the x,y,z axes, whereas a 3-array specifies per-axis scaling.
            mass (None or float): If specified, mass of this body in kg
            density (None or float): If specified, density of this body in kg / m^3
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
        self._dc = None                     # Dynamic control interface
        self._cs = None                     # Contact sensor interface
        self._handle = None
        self._contact_handle = None
        self._body_name = None
        self._rigid_api = None
        self._physx_rigid_api = None
        self._physx_contact_report_api = None
        self._mass_api = None

        self._visual_only = None
        self._collision_meshes = None
        self._visual_meshes = None

        # Run super init
        super().__init__(
            prim_path=prim_path,
            name=name,
            load_config=load_config,
        )

    def _post_load(self):
        # run super first
        super()._post_load()

        # Apply rigid body and mass APIs
        self._rigid_api = UsdPhysics.RigidBodyAPI(self._prim) if self._prim.HasAPI(UsdPhysics.RigidBodyAPI) else \
            UsdPhysics.RigidBodyAPI.Apply(self._prim)
        self._physx_rigid_api = PhysxSchema.PhysxRigidBodyAPI(self._prim) if \
            self._prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI) else PhysxSchema.PhysxRigidBodyAPI.Apply(self._prim)
        self._mass_api = UsdPhysics.MassAPI(self._prim) if self._prim.HasAPI(UsdPhysics.MassAPI) else \
            UsdPhysics.MassAPI.Apply(self._prim)

        # Only create contact report api if we're not visual only
        if not self._visual_only:
            self._physx_contact_report_api_api = PhysxSchema.PhysxContactReportAPI(self._prim) if \
                self._prim.HasAPI(PhysxSchema.PhysxContactReportAPI) else \
                PhysxSchema.PhysxContactReportAPI.Apply(self._prim)

        # Store references to owned visual / collision meshes
        # We iterate over all children of this object's prim,
        # and grab any that are presumed to be meshes
        self.update_meshes()

        # Possibly set the mass / density
        if not self.has_collision_meshes:
            # A meta (virtual) link has no collision meshes; set a negligible mass and a zero density (ignored)
            self.mass = 1e-6
            self.density = 0.0
        elif "mass" in self._load_config and self._load_config["mass"] is not None:
            self.mass = self._load_config["mass"]
        if "density" in self._load_config and self._load_config["density"] is not None:
            self.density = self._load_config["density"]

        # Set the visual-only attribute
        # This automatically handles setting collisions / gravity appropriately
        self.visual_only = self._load_config["visual_only"] if \
            "visual_only" in self._load_config and self._load_config["visual_only"] is not None else False

        # Create contact sensor
        self._cs = _s.acquire_contact_sensor_interface()
        # self._create_contact_sensor()

    def _initialize(self):
        # Run super method first
        super()._initialize()

        # Get dynamic control and contact sensing interfaces
        self._dc = _dynamic_control.acquire_dynamic_control_interface()

        # Initialize all owned meshes
        for mesh_group in (self._collision_meshes, self._visual_meshes):
            for mesh in mesh_group.values():
                mesh.initialize()

        # We grab contact info for the first time before setting our internal handle, because this changes the dc handle
        if self.contact_reporting_enabled:
            self._cs.get_rigid_body_raw_data(self._prim_path)

        # Grab handle to this rigid body and get name
        self.update_handles()
        self._body_name = self.prim_path.split("/")[-1]

    def update_meshes(self):
        """
        Helper function to refresh owned visual and collision meshes. Useful for synchronizing internal data if
        additional bodies are added manually
        """
        # Make sure to clean up all pre-existing names for all collision_meshes
        if self._collision_meshes is not None:
            for collision_mesh in self._collision_meshes.values():
                collision_mesh.remove_names()

        # Make sure to clean up all pre-existing names for all visual_meshes
        if self._visual_meshes is not None:
            for visual_mesh in self._visual_meshes.values():
                visual_mesh.remove_names()

        self._collision_meshes, self._visual_meshes = dict(), dict()
        prims_to_check = []
        coms, vols = [], []
        for prim in self._prim.GetChildren():
            prims_to_check.append(prim)
            for child in prim.GetChildren():
                prims_to_check.append(child)
        for prim in prims_to_check:
            if prim.GetPrimTypeInfo().GetTypeName() in GEOM_TYPES:
                mesh_name, mesh_path = prim.GetName(), prim.GetPrimPath().__str__()
                mesh_prim = get_prim_at_path(prim_path=mesh_path)
                mesh_kwargs = {"prim_path": mesh_path, "name": f"{self._name}:{mesh_name}"}
                if mesh_prim.HasAPI(UsdPhysics.CollisionAPI):
                    mesh = CollisionGeomPrim(**mesh_kwargs)
                    # We also modify the collision mesh's contact and rest offsets, since omni's default values result
                    # in lightweight objects sometimes not triggering contacts correctly
                    mesh.set_contact_offset(m.DEFAULT_CONTACT_OFFSET)
                    mesh.set_rest_offset(m.DEFAULT_REST_OFFSET)
                    self._collision_meshes[mesh_name] = mesh

                    is_volume, volume, com = get_mesh_volume_and_com(mesh_prim)
                    vols.append(volume)
                    # We need to translate the center of mass from the mesh's local frame to the link's local frame
                    local_pos, local_orn = mesh.get_local_pose()
                    coms.append(T.quat2mat(local_orn) @ (com * mesh.scale) + local_pos)
                    # If we're not a valid volume, use bounding box approximation for the underlying collision approx
                    if not is_volume:
                        log.warning(f"Got invalid (non-volume) collision mesh: {mesh.name}")
                        mesh.set_collision_approximation("boundingCube")
                else:
                    self._visual_meshes[mesh_name] = VisualGeomPrim(**mesh_kwargs)

        # If we have any collision meshes, we aggregate their center of mass and volume values to set the center of mass
        # for this link
        if len(coms) > 0:
            com = (np.array(coms) * np.array(vols).reshape(-1, 1)).sum(axis=0) / np.sum(vols)
            self.set_attribute("physics:centerOfMass", Gf.Vec3f(*com))

    def enable_collisions(self):
        """
        Enable collisions for this RigidPrim
        """
        # Iterate through all owned collision meshes and toggle on their collisions
        for col_mesh in self._collision_meshes.values():
            col_mesh.collision_enabled = True

    def disable_collisions(self):
        """
        Disable collisions for this RigidPrim
        """
        # Iterate through all owned collision meshes and toggle off their collisions
        for col_mesh in self._collision_meshes.values():
            col_mesh.collision_enabled = False

    def update_handles(self):
        """
        Updates all internal handles for this prim, in case they change since initialization
        """
        self._handle = None if self.kinematic_only else self._dc.get_rigid_body(self._prim_path)

    def contact_list(self):
        """
        Get list of all current contacts with this rigid body

        Returns:
            list of CsRawData: raw contact info for this rigid body
        """
        # # Make sure we have the ability to grab contacts for this object
        # assert self._physx_contact_report_api is not None, \
        #     "Cannot grab contacts for this rigid prim without Physx's contact report API being added!"
        contacts = []
        if self.contact_reporting_enabled:
            raw_data = self._cs.get_rigid_body_raw_data(self._prim_path)
            for c in raw_data:
                # contact sensor handles and dynamic articulation handles are not comparable
                # every prim has a cs to convert (cs) handle to prim path (decode_body_name)
                # but not every prim (e.g. groundPlane) has a dc to convert prim path to (dc) handle (get_rigid_body)
                # so simpler to convert both handles (int) to prim paths (str) for comparison
                c = [*c] # CsRawData enforces body0 and body1 types to be ints, but we want strings
                c[2] = self._cs.decode_body_name(c[2])
                c[3] = self._cs.decode_body_name(c[3])
                contacts.append(CsRawData(*c))
        return contacts

    def set_linear_velocity(self, velocity):
        """
        Sets the linear velocity of the prim in stage.

        Args:
            velocity (np.ndarray): linear velocity to set the rigid prim to. Shape (3,).
        """
        if self.dc_is_accessible:
            self._dc.set_rigid_body_linear_velocity(self._handle, velocity)
        else:
            self._rigid_api.GetVelocityAttr().Set(Gf.Vec3f(velocity.tolist()))

    def get_linear_velocity(self):
        """
        Returns:
            np.ndarray: current linear velocity of the the rigid prim. Shape (3,).
        """
        if self.dc_is_accessible:
            lin_vel = np.array(self._dc.get_rigid_body_linear_velocity(self._handle))
        else:
            lin_vel = self._rigid_api.GetVelocityAttr().Get()
        return np.array(lin_vel)

    def set_angular_velocity(self, velocity):
        """
        Sets the angular velocity of the prim in stage.

        Args:
            velocity (np.ndarray): angular velocity to set the rigid prim to. Shape (3,).
        """
        if self.dc_is_accessible:
            self._dc.set_rigid_body_angular_velocity(self._handle, velocity)
        else:
            self._rigid_api.GetAngularVelocityAttr().Set(Gf.Vec3f(velocity.tolist()))

    def get_angular_velocity(self):
        """
        Returns:
            np.ndarray: current angular velocity of the the rigid prim. Shape (3,).
        """
        if self.dc_is_accessible:
            return np.array(self._dc.get_rigid_body_angular_velocity(self._handle))
        else:
            return np.array(self._rigid_api.GetAngularVelocityAttr().Get())

    def set_position_orientation(self, position=None, orientation=None):
        if self.dc_is_accessible:
            current_position, current_orientation = self.get_position_orientation()
            if position is None:
                position = current_position
            if orientation is None:
                orientation = current_orientation
            pose = _dynamic_control.Transform(position, orientation)
            self._dc.set_rigid_body_pose(self._handle, pose)
        else:
            # Call super method by default
            super().set_position_orientation(position=position, orientation=orientation)

    def get_position_orientation(self):
        if self.dc_is_accessible:
            pose = self._dc.get_rigid_body_pose(self._handle)
            pos, ori = np.asarray(pose.p), np.asarray(pose.r)
        else:
            # Call super method by default
            pos, ori = super().get_position_orientation()

        return np.array(pos), np.array(ori)

    def set_local_pose(self, translation=None, orientation=None):
        if self.dc_is_accessible:
            current_translation, current_orientation = self.get_local_pose()
            translation = current_translation if translation is None else translation
            orientation = current_orientation if orientation is None else orientation
            orientation = orientation[[3, 0, 1, 2]]  # Flip from x,y,z,w to w,x,y,z
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
                position=np.array(calculated_position), orientation=gf_quat_to_np_array(calculated_orientation)
            )
        else:
            # Call super method by default
            super().set_local_pose(translation=translation, orientation=orientation)

    def get_local_pose(self):
        if self.dc_is_accessible:
            parent_world_tf = UsdGeom.Xformable(get_prim_parent(self._prim)).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            world_position, world_orientation = self.get_position_orientation()
            world_orientation = world_orientation[[3, 0, 1, 2]]  # Flip from x,y,z,w to w,x,y,z
            my_world_transform = tf_matrix_from_pose(translation=world_position, orientation=world_orientation)
            local_transform = np.matmul(np.linalg.inv(np.transpose(parent_world_tf)), my_world_transform)
            transform = Gf.Transform()
            transform.SetMatrix(Gf.Matrix4d(np.transpose(local_transform)))
            calculated_translation = transform.GetTranslation()
            calculated_orientation = transform.GetRotation().GetQuat()
            pos, ori = np.array(calculated_translation), gf_quat_to_np_array(calculated_orientation)[[1, 2, 3, 0]] # Flip from w,x,y,z to x,y,z,w to
        else:
            # Call super method by default
            pos, ori = super().get_local_pose()

        return np.array(pos), np.array(ori)

    @property
    def handle(self):
        """
        Handle used by Isaac Sim's dynamic control module to reference this rigid prim

        Returns:
            int: ID handle assigned to this prim from dynamic_control interface
        """
        return self._handle

    @property
    def body_name(self):
        """
        Returns:
            str: Name of this body
        """
        return self._body_name

    @property
    def collision_meshes(self):
        """
        Returns:
            dict: Dictionary mapping collision mesh names (str) to mesh prims (CollisionMeshPrim) owned by
                this rigid body
        """
        return self._collision_meshes

    @property
    def visual_meshes(self):
        """
        Returns:
            dict: Dictionary mapping visual mesh names (str) to mesh prims (VisualMeshPrim) owned by
                this rigid body
        """
        return self._visual_meshes

    @property
    def visual_only(self):
        """
        Returns:
            bool: Whether this link is a visual-only link (i.e.: no gravity or collisions applied)
        """
        return self._visual_only

    @property
    def has_collision_meshes(self):
        """
        Returns:
            bool: Whether this link has any collision mesh
        """
        return len(self._collision_meshes) > 0

    @visual_only.setter
    def visual_only(self, val):
        """
        Sets the visaul only state of this link

        Args:
            val (bool): Whether this link should be a visual-only link (i.e.: no gravity or collisions applied)
        """
        # Set gravity and collisions based on value
        if val:
            self.disable_collisions()
            self.disable_gravity()
        else:
            self.enable_collisions()
            self.enable_gravity()

        # Also set the internal value
        self._visual_only = val

    @property
    def volume(self):
        """
        Note: Currently it doesn't support Capsule type yet

        Returns:
            float: total volume of all the collision meshes of the rigid body in m^3.
        """
        # TODO (eric): revise this once omni exposes API to query volume of GeomPrims
        volume = 0.0
        for collision_mesh in self._collision_meshes.values():
            _, mesh_volume, _ = get_mesh_volume_and_com(collision_mesh.prim)
            volume += mesh_volume * np.product(collision_mesh.get_world_scale())

        return volume

    @volume.setter
    def volume(self, volume):
        raise NotImplementedError("Cannot set volume directly for an link!")

    @property
    def mass(self):
        """
        Returns:
            float: mass of the rigid body in kg.
        """
        raw_usd_mass = self._mass_api.GetMassAttr().Get()
        # If our raw_usd_mass isn't specified, we check dynamic control if possible (sim is playing),
        # otherwise we fallback to analytical computation of volume * density
        if raw_usd_mass != 0:
            mass = raw_usd_mass
        elif self.dc_is_accessible:
            mass = self.rigid_body_properties.mass
        else:
            mass = self.volume * self.density

        return mass

    @mass.setter
    def mass(self, mass):
        """
        Args:
            mass (float): mass of the rigid body in kg.
        """
        self._mass_api.GetMassAttr().Set(mass)

    @property
    def density(self):
        """
        Returns:
            float: density of the rigid body in kg / m^3.
        """
        raw_usd_mass = self._mass_api.GetMassAttr().Get()
        # We first check if the raw usd mass is specified, since mass overrides density
        # If it's specified, we infer density based on that value divided by volume
        # Otherwise, we try to directly grab the raw usd density value, and if that value
        # does not exist, we return 1000 since that is the canonical density assigned by omniverse
        if raw_usd_mass != 0:
            density = raw_usd_mass / self.volume
        else:
            density = self._mass_api.GetDensityAttr().Get()
            if density == 0:
                density = 1000.0

        return density

    @density.setter
    def density(self, density):
        """
        Args:
            density (float): density of the rigid body in kg / m^3.
        """
        self._mass_api.GetDensityAttr().Set(density)

    @property
    def kinematic_only(self):
        """
        Returns:
            bool: Whether this object is a kinematic-only object (otherwise, it is a rigid body). A kinematic-only
                object is not subject to simulator dynamics, and remains fixed unless the user explicitly sets the
                body's pose / velocities. See https://docs.omniverse.nvidia.com/app_create/prod_extensions/ext_physics/rigid-bodies.html?highlight=rigid%20body%20enabled#kinematic-rigid-bodies
                for more information
        """
        return self.get_attribute("physics:kinematicEnabled")

    @kinematic_only.setter
    def kinematic_only(self, val):
        """
        Args:
            val (bool): Whether this object is a kinematic-only object (otherwise, it is a rigid body). A kinematic-only
                object is not subject to simulator dynamics, and remains fixed unless the user explicitly sets the
                body's pose / velocities. See https://docs.omniverse.nvidia.com/app_create/prod_extensions/ext_physics/rigid-bodies.html?highlight=rigid%20body%20enabled#kinematic-rigid-bodies
                for more information
        """
        self.set_attribute("physics:kinematicEnabled", val)
        self.set_attribute("physics:rigidBodyEnabled", not val)

    @property
    def solver_position_iteration_count(self):
        """
        Returns:
            int: How many position iterations to take per physics step by the physx solver
        """
        return self.get_attribute("physxRigidBody:solverPositionIterationCount")

    @solver_position_iteration_count.setter
    def solver_position_iteration_count(self, count):
        """
        Sets how many position iterations to take per physics step by the physx solver

        Args:
            count (int): How many position iterations to take per physics step by the physx solver
        """
        self.set_attribute("physxRigidBody:solverPositionIterationCount", count)

    @property
    def solver_velocity_iteration_count(self):
        """
        Returns:
            int: How many velocity iterations to take per physics step by the physx solver
        """
        return self.get_attribute("physxRigidBody:solverVelocityIterationCount")

    @solver_velocity_iteration_count.setter
    def solver_velocity_iteration_count(self, count):
        """
        Sets how many velocity iterations to take per physics step by the physx solver

        Args:
            count (int): How many velocity iterations to take per physics step by the physx solver
        """
        self.set_attribute("physxRigidBody:solverVelocityIterationCount", count)

    @property
    def stabilization_threshold(self):
        """
        Returns:
            float: threshold for stabilizing this rigid body
        """
        return self.get_attribute("physxRigidBody:stabilizationThreshold")

    @stabilization_threshold.setter
    def stabilization_threshold(self, threshold):
        """
        Sets threshold for stabilizing this rigid body

        Args:
            threshold (float): stabilizing threshold
        """
        self.set_attribute("physxRigidBody:stabilizationThreshold", threshold)

    @property
    def sleep_threshold(self):
        """
        Returns:
            float: threshold for sleeping this rigid body
        """
        return self.get_attribute("physxRigidBody:sleepThreshold")

    @sleep_threshold.setter
    def sleep_threshold(self, threshold):
        """
        Sets threshold for sleeping this rigid body

        Args:
            threshold (float): Sleeping threshold
        """
        self.set_attribute("physxRigidBody:sleepThreshold", threshold)

    @property
    def ccd_enabled(self):
        """
        Returns:
            bool: whether CCD is enabled or not for this link
        """
        return self.get_attribute("physxRigidBody:enableCCD")

    @ccd_enabled.setter
    def ccd_enabled(self, enabled):
        """
        Args:
            enabled (bool): whether CCD should be enabled or not for this link
        """
        self.set_attribute("physxRigidBody:enableCCD", enabled)

    @property
    def contact_reporting_enabled(self):
        """
        Returns:
            bool: Whether contact reporting is enabled for this rigid prim or not
        """
        return self._prim.HasAPI(PhysxSchema.PhysxContactReportAPI)

    @property
    def rigid_body_properties(self):
        """
        Returns:
            None or RigidBodyProperty: Properties for this rigid body, if accessible. If they do not exist or
                dc cannot be queried, this will return None
        """
        return self._dc.get_rigid_body_properties(self._handle) if self.dc_is_accessible else None

    @property
    def dc_is_accessible(self):
        """
        Checks if dynamic control interface is accessible (checks whether we have a dc handle for this body
        and if dc is simulating)

        Returns:
            bool: Whether dc interface can be used or not
        """
        return self._handle is not None and self._dc.is_simulating() and not self.kinematic_only

    def enable_gravity(self):
        """
        Enables gravity for this rigid body
        """
        self.set_attribute("physxRigidBody:disableGravity", False)
        # self._dc.set_rigid_body_disable_gravity(self._handle, False)

    def disable_gravity(self):
        """
        Disables gravity for this rigid body
        """
        self.set_attribute("physxRigidBody:disableGravity", True)
        # self._dc.set_rigid_body_disable_gravity(self._handle, True)

    def wake(self):
        """
        Enable physics for this rigid body
        """
        if self.dc_is_accessible:
            self._dc.wake_up_rigid_body(self._handle)

    def sleep(self):
        """
        Disable physics for this rigid body
        """
        if self.dc_is_accessible:
            self._dc.sleep_rigid_body(self._handle)

    def _dump_state(self):
        # Grab pose from super class
        state = super()._dump_state()
        state["lin_vel"] = self.get_linear_velocity()
        state["ang_vel"] = self.get_angular_velocity()

        return state

    def _load_state(self, state):
        # Call super first
        super()._load_state(state=state)

        # Set velocities if not kinematic
        if not self.kinematic_only:
            self.set_linear_velocity(np.array(state["lin_vel"]))
            self.set_angular_velocity(np.array(state["ang_vel"]))

    def _serialize(self, state):
        # Run super first
        state_flat = super()._serialize(state=state)

        return np.concatenate([
            state_flat,
            state["lin_vel"],
            state["ang_vel"],
        ]).astype(float)

    def _deserialize(self, state):
        # Call supermethod first
        state_dic, idx = super()._deserialize(state=state)
        # We deserialize deterministically by knowing the order of values -- lin_vel, ang_vel
        state_dic["lin_vel"] = state[idx: idx+3]
        state_dic["ang_vel"] = state[idx + 3: idx + 6]

        return state_dic, idx + 6
