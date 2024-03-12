from functools import cached_property
from scipy.spatial import ConvexHull, QhullError
import numpy as np

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm, create_module_macros
from omnigibson.prims.xform_prim import XFormPrim
from omnigibson.prims.geom_prim import CollisionGeomPrim, VisualGeomPrim
from omnigibson.utils.constants import GEOM_TYPES
from omnigibson.utils.sim_utils import CsRawData
from omnigibson.utils.usd_utils import PoseAPI, get_mesh_volume_and_com, check_extent_radius_ratio
import omnigibson.utils.transform_utils as T
from omnigibson.utils.ui_utils import create_module_logger

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
            kinematic_only (None or bool): If specified, whether this prim should be kinematic-only or not.
    """

    def __init__(
        self,
        prim_path,
        name,
        load_config=None,
    ):
        # Other values that will be filled in at runtime
        self._rigid_prim_view_direct = None
        self._cs = None                     # Contact sensor interface
        self._body_name = None

        self._visual_only = None
        self._collision_meshes = None
        self._visual_meshes = None

        # Caches for kinematic-only objects
        # This exists because RigidPrimView uses USD pose read, which is very slow
        self._kinematic_world_pose_cache = None
        self._kinematic_local_pose_cache = None
    
        # Run super init
        super().__init__(
            prim_path=prim_path,
            name=name,
            load_config=load_config,
        )

    def _post_load(self):
        # Create the view
        # Import now to avoid too-eager load of Omni classes due to inheritance
        from omnigibson.utils.deprecated_utils import RigidPrimView
        self._rigid_prim_view_direct = RigidPrimView(self._prim_path)

        # Set it to be kinematic if necessary
        kinematic_only = "kinematic_only" in self._load_config and self._load_config["kinematic_only"]
        self.set_attribute("physics:kinematicEnabled", kinematic_only)
        self.set_attribute("physics:rigidBodyEnabled", not kinematic_only)

        # run super first
        super()._post_load()

        # Apply rigid body and mass APIs
        if not self._prim.HasAPI(lazy.pxr.UsdPhysics.RigidBodyAPI):
            lazy.pxr.UsdPhysics.RigidBodyAPI.Apply(self._prim)
        if not self._prim.HasAPI(lazy.pxr.PhysxSchema.PhysxRigidBodyAPI):
            lazy.pxr.PhysxSchema.PhysxRigidBodyAPI.Apply(self._prim)
        if not self._prim.HasAPI(lazy.pxr.UsdPhysics.MassAPI):
            lazy.pxr.UsdPhysics.MassAPI.Apply(self._prim)

        # Only create contact report api if we're not visual only
        if not self._visual_only:
            lazy.pxr.PhysxSchema.PhysxContactReportAPI(self._prim) if \
                self._prim.HasAPI(lazy.pxr.PhysxSchema.PhysxContactReportAPI) else \
                lazy.pxr.PhysxSchema.PhysxContactReportAPI.Apply(self._prim)

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
        self._cs = lazy.omni.isaac.sensor._sensor.acquire_contact_sensor_interface()
        # self._create_contact_sensor()

    def _initialize(self):
        # Run super method first
        super()._initialize()

        # Initialize all owned meshes
        for mesh_group in (self._collision_meshes, self._visual_meshes):
            for mesh in mesh_group.values():
                mesh.initialize()

        # Get contact info first
        if self.contact_reporting_enabled:
            self._cs.get_rigid_body_raw_data(self._prim_path)

        # Grab handle to this rigid body and get name
        self.update_handles()
        self._body_name = self.prim_path.split("/")[-1]

    def remove(self):
        # First remove the meshes
        if self._collision_meshes is not None:
            for collision_mesh in self._collision_meshes.values():
                collision_mesh.remove()

        # Make sure to clean up all pre-existing names for all visual_meshes
        if self._visual_meshes is not None:
            for visual_mesh in self._visual_meshes.values():
                visual_mesh.remove()

        # Then self
        super().remove()

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
            mesh_type = prim.GetPrimTypeInfo().GetTypeName()
            if mesh_type in GEOM_TYPES:
                mesh_name, mesh_path = prim.GetName(), prim.GetPrimPath().__str__()
                mesh_prim = lazy.omni.isaac.core.utils.prims.get_prim_at_path(prim_path=mesh_path)
                is_collision = mesh_prim.HasAPI(lazy.pxr.UsdPhysics.CollisionAPI)
                mesh_kwargs = {"prim_path": mesh_path, "name": f"{self._name}:{'collision' if is_collision else 'visual'}_{mesh_name}"}
                if is_collision:
                    mesh = CollisionGeomPrim(**mesh_kwargs)
                    # We also modify the collision mesh's contact and rest offsets, since omni's default values result
                    # in lightweight objects sometimes not triggering contacts correctly
                    mesh.set_contact_offset(m.DEFAULT_CONTACT_OFFSET)
                    mesh.set_rest_offset(m.DEFAULT_REST_OFFSET)
                    self._collision_meshes[mesh_name] = mesh

                    volume, com = get_mesh_volume_and_com(mesh_prim)
                    # We need to transform the volume and CoM from the mesh's local frame to the link's local frame
                    local_pos, local_orn = mesh.get_local_pose()
                    vols.append(volume * np.product(mesh.scale))
                    coms.append(T.quat2mat(local_orn) @ (com * mesh.scale) + local_pos)
                    # If the ratio between the max extent and min radius is too large (i.e. shape too oblong), use
                    # boundingCube approximation for the underlying collision approximation for GPU compatibility
                    if not check_extent_radius_ratio(mesh_prim):
                        log.warning(f"Got overly oblong collision mesh: {mesh.name}; use boundingCube approximation")
                        mesh.set_collision_approximation("boundingCube")
                else:
                    self._visual_meshes[mesh_name] = VisualGeomPrim(**mesh_kwargs)


        # If we have any collision meshes, we aggregate their center of mass and volume values to set the center of mass
        # for this link
        if len(coms) > 0:
            com = (np.array(coms) * np.array(vols).reshape(-1, 1)).sum(axis=0) / np.sum(vols)
            self.set_attribute("physics:centerOfMass", lazy.pxr.Gf.Vec3f(*com))

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
        # We only do this for non-kinematic objects, because while the USD APIs for kinematic-only
        # and dynamic objects are the same, physx tensor APIs do NOT exist for kinematic-only
        # objects, meaning initializing the view actively breaks the view.
        if not self.kinematic_only:
            self._rigid_prim_view_direct.initialize(og.sim.physics_sim_view)

    def contact_list(self):
        """
        Get list of all current contacts with this rigid body

        Returns:
            list of CsRawData: raw contact info for this rigid body
        """
        # Make sure we have the ability to grab contacts for this object
        contacts = []
        if self.contact_reporting_enabled:
            raw_data = self._cs.get_rigid_body_raw_data(self._prim_path)
            for c in raw_data:
                # convert handles to prim paths for comparison
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
        self._rigid_prim_view.set_linear_velocities(velocity[None, :])

    def get_linear_velocity(self):
        """
        Returns:
            np.ndarray: current linear velocity of the the rigid prim. Shape (3,).
        """
        return self._rigid_prim_view.get_linear_velocities()[0]

    def set_angular_velocity(self, velocity):
        """
        Sets the angular velocity of the prim in stage.

        Args:
            velocity (np.ndarray): angular velocity to set the rigid prim to. Shape (3,).
        """
        self._rigid_prim_view.set_angular_velocities(velocity[None, :])

    def get_angular_velocity(self):
        """
        Returns:
            np.ndarray: current angular velocity of the the rigid prim. Shape (3,).
        """
        return self._rigid_prim_view.get_angular_velocities()[0]

    def set_position_orientation(self, position=None, orientation=None):
        # Invalidate kinematic-only object pose caches when new pose is set
        if self.kinematic_only:
            self.clear_kinematic_only_cache()
        if position is not None:
            position = np.asarray(position)[None, :]
        if orientation is not None:
            assert np.isclose(np.linalg.norm(orientation), 1, atol=1e-3), \
                f"{self.prim_path} desired orientation {orientation} is not a unit quaternion."
            orientation = np.asarray(orientation)[None, [3, 0, 1, 2]]
        self._rigid_prim_view.set_world_poses(positions=position, orientations=orientation)
        PoseAPI.invalidate()

    def get_position_orientation(self):
        # Return cached pose if we're kinematic-only
        if self.kinematic_only and self._kinematic_world_pose_cache is not None:
            return self._kinematic_world_pose_cache
        
        pos, ori = self._rigid_prim_view.get_world_poses()

        assert np.isclose(np.linalg.norm(ori), 1, atol=1e-3), \
            f"{self.prim_path} orientation {ori} is not a unit quaternion."
        
        pos = pos[0]
        ori = ori[0][[1, 2, 3, 0]]
        if self.kinematic_only:
            self._kinematic_world_pose_cache = (pos, ori)
        return pos, ori

    def set_local_pose(self, position=None, orientation=None):
        # Invalidate kinematic-only object pose caches when new pose is set
        if self.kinematic_only:
            self.clear_kinematic_only_cache()
        if position is not None:
            position = np.asarray(position)[None, :]
        if orientation is not None:
            orientation = np.asarray(orientation)[None, [3, 0, 1, 2]]
        self._rigid_prim_view.set_local_poses(position, orientation)
        PoseAPI.invalidate()

    def get_local_pose(self):
        # Return cached pose if we're kinematic-only
        if self.kinematic_only and self._kinematic_local_pose_cache is not None:
            return self._kinematic_local_pose_cache
        
        positions, orientations = self._rigid_prim_view.get_local_poses()
        positions = positions[0]
        orientations = orientations[0][[1, 2, 3, 0]]
        if self.kinematic_only:
            self._kinematic_local_pose_cache = (positions, orientations)
        return positions, orientations

    @property
    def _rigid_prim_view(self):
        if self._rigid_prim_view_direct is None:
            return None

        # Validate that the if physics is running, the view is valid.
        if not self.kinematic_only and og.sim.is_playing() and self.initialized:
            assert self._rigid_prim_view_direct.is_physics_handle_valid() and \
                self._rigid_prim_view_direct._physics_view.check(), \
                "Rigid prim view must be valid if physics is running!"

        assert not (og.sim.is_playing() and not self._rigid_prim_view_direct.is_valid), \
            "Rigid prim view must be valid if physics is running!"
        
        return self._rigid_prim_view_direct

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
        return sum(get_mesh_volume_and_com(collision_mesh.prim, world_frame=True)[0] for collision_mesh in self._collision_meshes.values())

    @volume.setter
    def volume(self, volume):
        raise NotImplementedError("Cannot set volume directly for an link!")

    @property
    def mass(self):
        """
        Returns:
            float: mass of the rigid body in kg.
        """
        mass = self._rigid_prim_view.get_masses()[0]

        # Fallback to analytical computation of volume * density
        if mass == 0:
            return self.volume * self.density

        return mass

    @mass.setter
    def mass(self, mass):
        """
        Args:
            mass (float): mass of the rigid body in kg.
        """
        self._rigid_prim_view.set_masses([mass])

    @property
    def density(self):
        """
        Returns:
            float: density of the rigid body in kg / m^3.
        """
        mass = self._rigid_prim_view.get_masses()[0]
        # We first check if the mass is specified, since mass overrides density. If so, density = mass / volume.
        # Otherwise, we try to directly grab the raw usd density value, and if that value does not exist,
        # we return 1000 since that is the canonical density assigned by omniverse
        if mass != 0.0:
            density = mass / self.volume
        else:
            density = self._rigid_prim_view.get_densities()[0]
            if density == 0.0:
                density = 1000.0

        return density

    @density.setter
    def density(self, density):
        """
        Args:
            density (float): density of the rigid body in kg / m^3.
        """
        self._rigid_prim_view.set_densities([density])

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
    def is_asleep(self):
        """
        Returns:
            bool: whether this rigid prim is asleep or not
        """
        # If we're kinematic only, immediately return False since it doesn't follow the sleep / wake paradigm
        return False if self.kinematic_only \
            else og.sim.psi.is_sleeping(og.sim.stage_id, lazy.pxr.PhysicsSchemaTools.sdfPathToInt(self.prim_path))

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
        return self._prim.HasAPI(lazy.pxr.PhysxSchema.PhysxContactReportAPI)

    def _compute_points_on_convex_hull(self, visual):
        """
        Returns:
            np.ndarray or None: points on the convex hull of all points from child geom prims
        """
        meshes = self._visual_meshes if visual else self._collision_meshes
        points = []

        for mesh in meshes.values():
            mesh_points = mesh.points_in_parent_frame
            if mesh_points is not None and len(mesh_points) > 0:
                points.append(mesh_points)
        
        if not points:
            return None

        points = np.concatenate(points, axis=0)
        
        try:
            hull = ConvexHull(points)
            return points[hull.vertices, :]
        except:
            # Handle the case where a convex hull cannot be formed (e.g., collinear points)
            # return all the points in this case
            return points
        
    @cached_property
    def visual_boundary_points_local(self):
        """
        Returns:
            np.ndarray: local coords of points on the convex hull of all points from child geom prims
        """
        return self._compute_points_on_convex_hull(visual=True)
    
    @property
    def visual_boundary_points_world(self):
        """
        Returns:
            np.ndarray: world coords of points on the convex hull of all points from child geom prims
        """
        local_points = self.visual_boundary_points_local
        if local_points is None:
            return None
        return self.transform_local_points_to_world(local_points)
    
    @cached_property
    def collision_boundary_points_local(self):
        """
        Returns:
            np.ndarray: local coords of points on the convex hull of all points from child geom prims
        """
        return self._compute_points_on_convex_hull(visual=False)
    
    @property
    def collision_boundary_points_world(self):
        """
        Returns:
            np.ndarray: world coords of points on the convex hull of all points from child geom prims
        """
        local_points = self.collision_boundary_points_local
        if local_points is None:
            return None
        return self.transform_local_points_to_world(local_points)
        
    @property
    def aabb(self):
        position, _ = self.get_position_orientation()
        hull_points = self.collision_boundary_points_world
        
        if hull_points is None:
            # When there's no points on the collision meshes
            return position, position
        
        aabb_lo = np.min(hull_points, axis=0)
        aabb_hi = np.max(hull_points, axis=0)
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
    
    @property
    def visual_aabb(self):
        hull_points = self.visual_boundary_points_world
        assert hull_points is not None, "No visual boundary points found for this rigid prim"

        # Calculate and return the AABB
        aabb_lo = np.min(hull_points, axis=0)
        aabb_hi = np.max(hull_points, axis=0)

        return aabb_lo, aabb_hi

    @property
    def visual_aabb_extent(self):
        """
        Get this xform's actual bounding box extent

        Returns:
            3-array: (x,y,z) bounding box
        """
        min_corner, max_corner = self.visual_aabb
        return max_corner - min_corner

    @property
    def visual_aabb_center(self):
        """
        Get this xform's actual bounding box center

        Returns:
            3-array: (x,y,z) bounding box center
        """
        min_corner, max_corner = self.visual_aabb
        return (max_corner + min_corner) / 2.0
    
    def enable_gravity(self):
        """
        Enables gravity for this rigid body
        """
        self._rigid_prim_view.enable_gravities()

    def disable_gravity(self):
        """
        Disables gravity for this rigid body
        """
        self._rigid_prim_view.disable_gravities()

    def wake(self):
        """
        Enable physics for this rigid body
        """
        prim_id = lazy.pxr.PhysicsSchemaTools.sdfPathToInt(self.prim_path)
        og.sim.psi.wake_up(og.sim.stage_id, prim_id)

    def sleep(self):
        """
        Disable physics for this rigid body
        """
        prim_id = lazy.pxr.PhysicsSchemaTools.sdfPathToInt(self.prim_path)
        og.sim.psi.put_to_sleep(og.sim.stage_id, prim_id)

    def clear_kinematic_only_cache(self):
        """
        Clears the internal kinematic only cached pose. Useful if the parent prim's pose
        changes without explicitly calling this prim's pose setter
        """
        assert self.kinematic_only
        self._kinematic_local_pose_cache = None
        self._kinematic_world_pose_cache = None

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
