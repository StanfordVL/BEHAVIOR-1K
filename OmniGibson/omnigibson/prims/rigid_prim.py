from functools import cached_property
import re
import math

import torch as th
from scipy.spatial import ConvexHull

import omnigibson as og
from omnigibson.macros import gm
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.macros import create_module_macros
from omnigibson.prims.geom_prim import CollisionGeomPrim, VisualGeomPrim
from omnigibson.prims.xform_prim import XFormPrim
from omnigibson.utils.constants import GEOM_TYPES
from omnigibson.utils.sim_utils import CsRawData
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.usd_utils import (
    absolute_prim_path_to_scene_relative,
    check_extent_radius_ratio,
    get_mesh_volume_and_com,
)

# Create module logger
log = create_module_logger(module_name=__name__)

# Create settings for this module
m = create_module_macros(module_path=__file__)

m.DEFAULT_CONTACT_OFFSET = 0.001
m.DEFAULT_REST_OFFSET = 0.0
m.LEGACY_META_LINK_PATTERN = re.compile(r".*:(\w+)_([A-Za-z0-9]+)_(\d+)_link")


class RigidPrim(XFormPrim):
    """
    Base class that provides common functionality for all rigid prim types.
    This serves as the parent class for RigidDynamicPrim and RigidKinematicPrim.

    Provides high level functions to deal with a rigid body prim and its attributes/properties.
    If there is a prim present at the path, it will use it. Otherwise, a new XForm prim at
    the specified prim path will be created.

    Args:
        relative_prim_path (str): Scene-local prim path of the Prim to encapsulate or create.
        name (str): Name for the object. Names need to be unique per scene.
        load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
            loading this prim at runtime. Note that this is only needed if the prim does not already exist at
            @relative_prim_path -- it will be ignored if it already exists. For this joint prim, the below values can be
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
        relative_prim_path,
        name,
        load_config=None,
    ):
        # Common values that will be used by both kinematic and dynamic rigid prims
        self._body_name = None
        self._visual_only = None
        self._collision_meshes = None
        self._visual_meshes = None
        self._belongs_to_articulation = None

        # Run super init
        super().__init__(
            relative_prim_path=relative_prim_path,
            name=name,
            load_config=load_config,
        )

    def _post_load(self):
        # Run super first
        super()._post_load()

        # Apply rigid body and mass APIs
        if not self._prim.HasAPI(lazy.pxr.UsdPhysics.RigidBodyAPI):
            lazy.pxr.UsdPhysics.RigidBodyAPI.Apply(self._prim)
        if not self._prim.HasAPI(lazy.pxr.PhysxSchema.PhysxRigidBodyAPI):
            lazy.pxr.PhysxSchema.PhysxRigidBodyAPI.Apply(self._prim)
        if not self._prim.HasAPI(lazy.pxr.UsdPhysics.MassAPI):
            lazy.pxr.UsdPhysics.MassAPI.Apply(self._prim)

        # Check if it's part of an articulation view
        self._belongs_to_articulation = (
            "belongs_to_articulation" in self._load_config and self._load_config["belongs_to_articulation"]
        )

        # Only create contact report api if we're not visual only
        self._visual_only = (
            self._load_config["visual_only"]
            if "visual_only" in self._load_config and self._load_config["visual_only"] is not None
            else False
        )

        if not self._visual_only:
            contact_api = (
                lazy.pxr.PhysxSchema.PhysxContactReportAPI(self._prim)
                if self._prim.HasAPI(lazy.pxr.PhysxSchema.PhysxContactReportAPI)
                else lazy.pxr.PhysxSchema.PhysxContactReportAPI.Apply(self._prim)
            )
            contact_api.GetThresholdAttr().Set(0.0)

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
        self.visual_only = self._visual_only or gm.VISUAL_ONLY

    def _initialize(self):
        # Run super method first
        super()._initialize()

        # Initialize all owned meshes
        for mesh_group in (self._collision_meshes, self._visual_meshes):
            for mesh in mesh_group.values():
                mesh.initialize()

        # Get contact info first
        if self.contact_reporting_enabled:
            og.sim.contact_sensor.get_rigid_body_raw_data(self.prim_path)

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
                mesh_prim = lazy.isaacsim.core.utils.prims.get_prim_at_path(prim_path=mesh_path)
                is_collision = mesh_prim.HasAPI(lazy.pxr.UsdPhysics.CollisionAPI)
                mesh_kwargs = {
                    "relative_prim_path": absolute_prim_path_to_scene_relative(self.scene, mesh_path),
                    "name": f"{self._name}:{'collision' if is_collision else 'visual'}_{mesh_name}",
                    "load_config": {"xform_props_pre_loaded": self._load_config["xform_props_pre_loaded"]},
                }
                if is_collision:
                    mesh = CollisionGeomPrim(**mesh_kwargs)
                    mesh.load(self.scene)
                    # We also modify the collision mesh's contact and rest offsets, since omni's default values result
                    # in lightweight objects sometimes not triggering contacts correctly
                    mesh.set_contact_offset(m.DEFAULT_CONTACT_OFFSET)
                    mesh.set_rest_offset(m.DEFAULT_REST_OFFSET)
                    self._collision_meshes[mesh_name] = mesh

                    volume, com = get_mesh_volume_and_com(mesh_prim)
                    # We need to transform the volume and CoM from the mesh's local frame to the link's local frame
                    local_pos, local_orn = mesh.get_position_orientation(frame="parent")
                    vols.append(volume * th.prod(mesh.scale))
                    coms.append(T.quat2mat(local_orn) @ (com * mesh.scale) + local_pos)
                    # If the ratio between the max extent and min radius is too large (i.e. shape too oblong), use
                    # boundingCube approximation for the underlying collision approximation for GPU compatibility
                    if not check_extent_radius_ratio(mesh, com):
                        log.warning(f"Got overly oblong collision mesh: {mesh.name}; use boundingCube approximation")
                        mesh.set_collision_approximation("boundingCube")
                else:
                    self._visual_meshes[mesh_name] = VisualGeomPrim(**mesh_kwargs)
                    self._visual_meshes[mesh_name].load(self.scene)

        # If we have any collision meshes, we aggregate their center of mass and volume values to set the center of mass
        # for this link
        if len(coms) > 0:
            coms_tensor = th.stack(coms)
            vols_tensor = th.tensor(vols).unsqueeze(1)
            com = th.sum(coms_tensor * vols_tensor, dim=0) / th.sum(vols_tensor)
            self.center_of_mass = com

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
        Updates all internal handles for this prim, in case they change since initialization.
        To be implemented by subclasses as needed.
        """
        pass

    def contact_list(self):
        """
        Get list of all current contacts with this rigid body
        NOTE: This method is slow and uncached, but it works even for sleeping objects.
        For frequent contact checks, consider using RigidContactAPI for performance.

        Returns:
            list of CsRawData: raw contact info for this rigid body
        """
        # Make sure we have the ability to grab contacts for this object
        contacts = []
        if self.contact_reporting_enabled:
            raw_data = og.sim.contact_sensor.get_rigid_body_raw_data(self.prim_path)
            for c in raw_data:
                # convert handles to prim paths for comparison
                c = [*c]  # CsRawData enforces body0 and body1 types to be ints, but we want strings
                c[2] = og.sim.contact_sensor.decode_body_name(c[2])
                c[3] = og.sim.contact_sensor.decode_body_name(c[3])
                contacts.append(CsRawData(*c))
        return contacts

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
        Sets the visual only state of this link

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
        meshes = self._visual_meshes.values() if self.is_meta_link else self._collision_meshes.values()
        return sum(get_mesh_volume_and_com(mesh.prim, world_frame=True)[0] for mesh in meshes)

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
            th.tensor or None: points on the convex hull of all points from child geom prims
        """
        meshes = self._visual_meshes if visual else self._collision_meshes
        points = []

        for mesh in meshes.values():
            mesh_points = mesh.points_in_parent_frame
            if mesh_points is not None and len(mesh_points) > 0:
                points.append(mesh_points)

        if not points:
            return None

        points = th.cat(points, dim=0)

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
            th.tensor: local coords of points on the convex hull of all points from child geom prims
        """
        return self._compute_points_on_convex_hull(visual=True)

    @property
    def visual_boundary_points_world(self):
        """
        Returns:
            th.tensor: world coords of points on the convex hull of all points from child geom prims
        """
        local_points = self.visual_boundary_points_local
        if local_points is None:
            return None
        return self.transform_local_points_to_world(local_points)

    @cached_property
    def collision_boundary_points_local(self):
        """
        Returns:
            th.tensor: local coords of points on the convex hull of all points from child geom prims
        """
        return self._compute_points_on_convex_hull(visual=False)

    @property
    def collision_boundary_points_world(self):
        """
        Returns:
            th.tensor: world coords of points on the convex hull of all points from child geom prims
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

        aabb_lo = th.min(hull_points, dim=0).values
        aabb_hi = th.max(hull_points, dim=0).values
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
        aabb_lo = th.min(hull_points, dim=0).values
        aabb_hi = th.max(hull_points, dim=0).values

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

    @cached_property
    def is_meta_link(self):
        # Check using the new format first
        new_format = self.prim.HasAttribute("ig:isMetaLink") and self.get_attribute("ig:isMetaLink")
        if new_format:
            return True

        # Check using the old format.
        # TODO: Remove this after the next dataset release
        old_format = m.LEGACY_META_LINK_PATTERN.fullmatch(self.name) is not None
        if old_format:
            return True

        return False

    @cached_property
    def meta_link_type(self):
        assert self.is_meta_link, f"{self.name} is not a meta link"
        if self.prim.HasAttribute("ig:metaLinkType"):
            return self.get_attribute("ig:metaLinkType")

        # Check using the old format.
        # TODO: Remove this after the next dataset release
        return m.LEGACY_META_LINK_PATTERN.fullmatch(self.name).group(1)

    @cached_property
    def meta_link_id(self):
        """The meta link id of this link, if the link is a meta link.

        The meta link ID is a semantic identifier for the meta link within the meta link type. It is
        used when an object has multiple meta links of the same type. It can be just a numerical index,
        or for some objects, it will be a string that can be matched to other meta links. For example,
        a stove might have toggle buttons named "left" and "right", and heat sources named "left" and
        "right". The meta link ID can be used to match the toggle button to the heat source.
        """
        assert self.is_meta_link, f"{self.name} is not a meta link"
        if self.prim.HasAttribute("ig:metaLinkId"):
            return self.get_attribute("ig:metaLinkId")

        # Check using the old format.
        # TODO: Remove this after the next dataset release
        return m.LEGACY_META_LINK_PATTERN.fullmatch(self.name).group(2)

    @cached_property
    def meta_link_sub_id(self):
        """The integer meta link sub id of this link, if the link is a meta link.

        The meta link sub ID identifies this link as one of the parts of a meta link. For example, an
        attachment meta link's ID will be the attachment pair name, and each attachment point that
        works with that pair will show up as a separate link with a unique sub ID.
        """
        assert self.is_meta_link, f"{self.name} is not a meta link"
        if self.prim.HasAttribute("ig:metaLinkSubId"):
            return int(self.get_attribute("ig:metaLinkSubId"))

        # Check using the old format.
        # TODO: Remove this after the next dataset release
        return int(m.LEGACY_META_LINK_PATTERN.fullmatch(self.name).group(3))

    def check_points_in_volume(
        self,
        particle_positions_world,
        use_visual_meshes=True,
    ):
        """
        Args:
            particle_positions_world (th.tensor): (N, 3) array of particle positions to check
            use_visual_meshes (bool): Whether to use @volume_link's visual or collision meshes to generate points fcn.
                In either case, this assumes the given meshes are convex hulls. For visual meshes, we enforce this
                constraint by explicitly converting them into convex hulls
        """
        in_volume = th.zeros(particle_positions_world.shape[0], dtype=th.bool)
        meshes_to_check = self.visual_meshes if use_visual_meshes else self.collision_meshes
        for mesh in meshes_to_check.values():
            in_volume |= mesh.check_points_in_volume(particle_positions_world)
        return in_volume

    @cached_property
    def world_volume(self, precision=1e-5):
        # We use monte-carlo sampling to approximate the voluem up to @precision
        # NOTE: precision defines the RELATIVE precision of the volume computation -- i.e.: the relative error with
        # respect to the volume link's global AABB

        # Convert precision to minimum number of particles to sample
        min_n_particles = int(math.ceil(1.0 / precision))

        # Determine equally-spaced sampling distance to achieve this minimum particle count
        aabb_volume = th.prod(self.visual_aabb_extent)
        sampling_distance = th.pow(aabb_volume / min_n_particles, 1 / 3.0)
        low, high = self.aabb
        n_particles_per_axis = ((high - low) / sampling_distance).int() + 1
        assert th.all(n_particles_per_axis), "Must increase precision for calculate_volume -- too coarse for sampling!"
        # 1e-10 is added because the extent might be an exact multiple of particle radius
        arrs = [th.arange(l, h, sampling_distance) for l, h, n in zip(low, high, n_particles_per_axis)]
        # Generate 3D-rectangular grid of points, and only keep the ones inside the mesh
        points = th.stack([arr.flatten() for arr in th.meshgrid(*arrs)]).T

        # Return the fraction of the link AABB's volume based on fraction of points enclosed within it
        return aabb_volume * th.mean(self.check_points_in_volume(points).float())
