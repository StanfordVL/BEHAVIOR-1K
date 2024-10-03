# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import math
from collections.abc import Iterable

import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.macros import create_module_macros, gm
from omnigibson.prims.geom_prim import GeomPrim
from omnigibson.utils.numpy_utils import vtarray_to_torch
from omnigibson.utils.sim_utils import CsRawData
from omnigibson.utils.usd_utils import array_to_vtarray, mesh_prim_to_trimesh_mesh, sample_mesh_keypoints

# Create settings for this module
m = create_module_macros(module_path=__file__)

# Subsample cloth particle points to boost performance
m.N_CLOTH_KEYPOINTS = 1000
m.KEYPOINT_COVERAGE_THRESHOLD = 0.75
m.N_CLOTH_KEYFACES = 500


class ClothPrim(GeomPrim):
    """
    Provides high level functions to deal with a cloth prim and its attributes/ properties.
    If there is an prim present at the path, it will use it. Otherwise, a new XForm prim at
    the specified prim path will be created.

    Notes: if the prim does not already have a cloth api applied to it before it is loaded,
        it will apply it.

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
    """

    def __init__(
        self,
        relative_prim_path,
        name,
        load_config=None,
    ):
        # Internal vars stored
        self._centroid_idx = None
        self._keypoint_idx = None
        self._keyface_idx = None

        # Run super init
        super().__init__(
            relative_prim_path=relative_prim_path,
            name=name,
            load_config=load_config,
        )

    def _post_load(self):
        # run super first
        super()._post_load()

        # Make sure flatcache is not being used -- if so, raise an error, since we lose most of our needed functionality
        # (such as R/W to specific particle states) when flatcache is enabled
        assert not gm.ENABLE_FLATCACHE, "Cannot use flatcache with ClothPrim!"

        self._mass_api = (
            lazy.pxr.UsdPhysics.MassAPI(self._prim)
            if self._prim.HasAPI(lazy.pxr.UsdPhysics.MassAPI)
            else lazy.pxr.UsdPhysics.MassAPI.Apply(self._prim)
        )

        # Possibly set the mass / density
        if "mass" in self._load_config and self._load_config["mass"] is not None:
            self.mass = self._load_config["mass"]

        # Clothify this prim, which is assumed to be a mesh
        self.cloth_system.clothify_mesh_prim(mesh_prim=self._prim, remesh=self._load_config.get("remesh", True))

        # Track generated particle count
        positions = self.compute_particle_positions()
        self._n_particles = len(positions)

        # Sample mesh keypoints / keyvalues and sanity check the AABB of these subsampled points vs. the actual points
        success = False
        for i in range(10):
            self._keypoint_idx, self._keyface_idx = sample_mesh_keypoints(
                mesh_prim=self._prim,
                n_keypoints=m.N_CLOTH_KEYPOINTS,
                n_keyfaces=m.N_CLOTH_KEYFACES,
                seed=i,
            )

            keypoint_positions = positions[self._keypoint_idx]
            keypoint_aabb = keypoint_positions.min(dim=0).values, keypoint_positions.max(dim=0).values
            true_aabb = positions.min(dim=0).values, positions.max(dim=0).values
            overlap_x = th.max(
                th.min(true_aabb[1][0], keypoint_aabb[1][0]) - th.max(true_aabb[0][0], keypoint_aabb[0][0]),
                th.tensor(0),
            )
            overlap_y = th.max(
                th.min(true_aabb[1][1], keypoint_aabb[1][1]) - th.max(true_aabb[0][1], keypoint_aabb[0][1]),
                th.tensor(0),
            )
            overlap_z = th.max(
                th.min(true_aabb[1][2], keypoint_aabb[1][2]) - th.max(true_aabb[0][2], keypoint_aabb[0][2]),
                th.tensor(0),
            )
            overlap_vol = overlap_x * overlap_y * overlap_z
            true_vol = th.prod(true_aabb[1] - true_aabb[0])
            if true_vol == 0.0 or (overlap_vol / true_vol > m.KEYPOINT_COVERAGE_THRESHOLD).item():
                success = True
                break
        assert success, f"Did not adequately subsample keypoints for cloth {self.name}!"

        # Compute centroid particle idx based on AABB
        aabb_min, aabb_max = th.min(positions, dim=0).values, th.max(positions, dim=0).values
        aabb_center = (aabb_min + aabb_max) / 2.0
        dists = th.norm(positions - aabb_center.reshape(1, 3), dim=-1)
        self._centroid_idx = th.argmin(dists)

        # Store the default position of the points in the local frame
        self._default_positions = vtarray_to_torch(self.get_attribute(attr="points"))

    @property
    def visual_aabb(self):
        return self.aabb

    @property
    def visual_aabb_extent(self):
        return self.aabb_extent

    @property
    def visual_aabb_center(self):
        return self.aabb_center

    @property
    def cloth_system(self):
        return self.scene.get_system("cloth")

    @property
    def n_particles(self):
        """
        Returns:
            int: Number of particles owned by this cloth prim
        """
        return self._n_particles

    @property
    def kinematic_only(self):
        """
        Returns:
            bool: Whether this object is a kinematic-only object. For ClothPrim, always return False.
        """
        return False

    def compute_particle_positions(self, idxs=None):
        """
        Compute individual particle positions for this cloth prim

        Args:
            idxs (n-array or None): If set, will only calculate the requested indexed particle state

        Returns:
            th.tensor: (N, 3) numpy array, where each of the N particles' positions are expressed in (x,y,z)
                cartesian coordinates relative to the world frame
        """
        pos, ori = self.get_position_orientation()
        ori = T.quat2mat(ori)
        scale = self.scale

        # Don't copy to save compute, since we won't be returning a reference to the underlying object anyways
        p_local = vtarray_to_torch(self.get_attribute(attr="points"))
        p_local = p_local[idxs] if idxs is not None else p_local
        p_world = (ori @ (p_local * scale).T).T + pos

        return p_world

    def set_particle_positions(self, positions, idxs=None):
        """
        Sets individual particle positions for this cloth prim

        Args:
            positions (n-array): (N, 3) numpy array, where each of the N particles' positions are expressed in (x,y,z)
                cartesian coordinates relative to the world frame
            idxs (n-array or None): If set, will only set the requested indexed particle state
        """
        n_expected = self._n_particles if idxs is None else len(idxs)
        assert (
            len(positions) == n_expected
        ), f"Got mismatch in particle setting size: {len(positions)}, vs. number of expected particles {n_expected}!"

        translation, rotation = self.get_position_orientation()
        rotation = T.quat2mat(rotation)
        scale = self.scale
        p_local = (rotation.T @ (positions - translation).T).T / scale

        # Fill the idxs if requested
        if idxs is not None:
            p_local_old = vtarray_to_torch(self.get_attribute(attr="points"))
            p_local_old[idxs] = p_local
            p_local = p_local_old

        self.set_attribute(attr="points", val=lazy.pxr.Vt.Vec3fArray(p_local.tolist()))

    @property
    def keypoint_idx(self):
        """
        Returns:
            n-array: (N,) array specifying the keypoint particle IDs
        """
        return self._keypoint_idx

    @property
    def keyface_idx(self):
        """
        Returns:
            n-array: (N,) array specifying the keyface IDs
        """
        return self._keyface_idx

    @property
    def faces(self):
        """
        Grabs particle indexes defining each of the faces for this cloth prim

        Returns:
             th.tensor: (N, 3) numpy array, where each of the N faces are defined by the 3 particle indices
                corresponding to that face's vertices
        """
        return th.tensor(self.get_attribute("faceVertexIndices")).reshape(-1, 3)

    @property
    def keyfaces(self):
        """
        Grabs particle indexes defining each of the keyfaces for this cloth prim.
        Total number of keyfaces is m.N_CLOTH_KEYFACES

        Returns:
             th.tensor: (N, 3) numpy array, where each of the N keyfaces are defined by the 3 particle indices
                corresponding to that face's vertices
        """
        return self.faces[self._keyface_idx]

    @property
    def keypoint_particle_positions(self):
        """
        Grabs individual keypoint particle positions for this cloth prim.
        Total number of keypoints is m.N_CLOTH_KEYPOINTS

        Returns:
            th.tensor: (N, 3) numpy array, where each of the N keypoint particles' positions are expressed in (x,y,z)
                cartesian coordinates relative to the world frame
        """
        return self.compute_particle_positions(idxs=self._keypoint_idx)

    @property
    def centroid_particle_position(self):
        """
        Grabs the individual particle that was pre-computed to be the closest to the centroid of this cloth prim.

        Returns:
            th.tensor: centroid particle's (x,y,z) cartesian coordinates relative to the world frame
        """
        return self.compute_particle_positions(idxs=[self._centroid_idx])

    @property
    def particle_velocities(self):
        """
        Grabs individual particle velocities for this cloth prim

        Returns:
            th.tensor: (N, 3) numpy array, where each of the N particles' velocities are expressed in (x,y,z)
                cartesian coordinates with respect to the world frame.
        """
        # the velocities attribute is w.r.t the world frame already
        return vtarray_to_torch(self.get_attribute(attr="velocities"))

    @particle_velocities.setter
    def particle_velocities(self, vel):
        """
        Set the particle velocities of this cloth

        Args:
            th.tensor: (N, 3) numpy array, where each of the N particles' velocities are expressed in (x,y,z)
                cartesian coordinates with respect to the world frame
        """
        assert (
            vel.shape[0] == self._n_particles
        ), f"Got mismatch in particle setting size: {vel.shape[0]}, vs. number of particles {self._n_particles}!"

        # the velocities attribute is w.r.t the world frame already
        self.set_attribute(attr="velocities", val=lazy.pxr.Vt.Vec3fArray(vel.tolist()))

    def compute_face_normals(self, face_ids=None):
        """
        Grabs individual face normals for this cloth prim

        Args:
            face_ids (None or n-array): If specified, list of face IDs whose corresponding normals should be computed
                If None, all faces will be used

        Returns:
            th.tensor: (N, 3) numpy array, where each of the N faces' normals are expressed in (x,y,z)
                cartesian coordinates with respect to the world frame.
        """
        faces = self.faces if face_ids is None else self.faces[face_ids]
        points = self.compute_particle_positions(idxs=faces.flatten()).reshape(-1, 3, 3)
        return self.compute_face_normals_from_particle_positions(positions=points)

    def compute_face_normals_from_particle_positions(self, positions):
        """
        Grabs individual face normals for this cloth prim

        Args:
            positions (n-array): (N, 3, 3) array specifying the per-face particle positions

        Returns:
            th.tensor: (N, 3) numpy array, where each of the N faces' normals are expressed in (x,y,z)
                cartesian coordinates with respect to the world frame.
        """
        # Shape [F, 3]
        v1 = positions[:, 2, :] - positions[:, 0, :]
        v2 = positions[:, 1, :] - positions[:, 0, :]
        normals = th.linalg.cross(v1, v2)
        return normals / th.norm(normals, dim=1).reshape(-1, 1)

    def contact_list(self, keypoints_only=True):
        """
        Get list of all current contacts with this cloth body

        Args:
            keypoints_only (bool): If True, will only check contact with this cloth's keypoints

        Returns:
            list of CsRawData: raw contact info for this cloth body
        """
        contacts = []

        def report_hit(hit):
            contacts.append(
                CsRawData(
                    time=0.0,  # dummy value
                    dt=0.0,  # dummy value
                    body0=self.prim_path,
                    body1=hit.rigid_body,
                    position=pos,
                    normal=th.zeros(3),  # dummy value
                    impulse=th.zeros(3),  # dummy value
                )
            )
            return True

        positions = self.keypoint_particle_positions if keypoints_only else self.compute_particle_positions()
        for pos in positions:
            og.sim.psqi.overlap_sphere(self.cloth_system.particle_contact_offset, pos.tolist(), report_hit, False)

        return contacts

    def update_handles(self):
        # no handles to update
        pass

    @property
    def volume(self):
        mesh = mesh_prim_to_trimesh_mesh(self.prim, include_normals=False, include_texcoord=False, world_frame=True)
        return mesh.volume if mesh.is_volume else mesh.convex_hull.volume

    @volume.setter
    def volume(self, volume):
        raise NotImplementedError("Cannot set volume directly for a link!")

    @property
    def mass(self):
        """
        Returns:
            float: mass of the rigid body in kg.
        """
        # We have to read the mass directly in the cloth prim
        return self._mass_api.GetMassAttr().Get()

    @mass.setter
    def mass(self, mass):
        """
        Args:
            mass (float): mass of the rigid body in kg.
        """
        # We have to set the mass directly in the cloth prim
        self._mass_api.GetMassAttr().Set(mass)

    @property
    def density(self):
        raise NotImplementedError("Cannot get density for ClothPrim")

    @density.setter
    def density(self, density):
        raise NotImplementedError("Cannot set density for ClothPrim")

    @property
    def body_name(self):
        """
        Returns:
            str: Name of this body
        """
        return self.prim_path.split("/")[-1]

    def get_linear_velocity(self):
        """
        Returns:
            th.tensor: current average linear velocity of the particles of the cloth prim. Shape (3,).
        """
        return vtarray_to_torch(self._prim.GetAttribute("velocities").Get()).mean(dim=0)

    def get_angular_velocity(self):
        """
        Returns:
            th.tensor: zero vector as a placeholder because a cloth prim doesn't have an angular velocity. Shape (3,).
        """
        return th.zeros(3)

    def set_linear_velocity(self, velocity):
        """
        Sets the linear velocity of all the particles of the cloth prim.

        Args:
            velocity (th.tensor): linear velocity to set all the particles of the cloth prim to. Shape (3,).
        """
        vel = self.particle_velocities
        vel[:] = velocity
        self.particle_velocities = vel

    def set_angular_velocity(self, velocity):
        """
        Simply returns because a cloth prim doesn't have an angular velocity

        Args:
            velocity (th.tensor): linear velocity to set all the particles of the cloth prim to. Shape (3,).
        """
        return

    def wake(self):
        # TODO (eric): Just a pass through for now.
        return

    @property
    def bend_stiffness(self):
        """
        Returns:
            float: spring bend stiffness of the particle system
        """
        return self.get_attribute("physxAutoParticleCloth:springBendStiffness")

    @bend_stiffness.setter
    def bend_stiffness(self, bend_stiffness):
        """
        Args:
            bend_stiffness (float): spring bend stiffness of the particle system
        """
        self.set_attribute("physxAutoParticleCloth:springBendStiffness", bend_stiffness)

    @property
    def damping(self):
        """
        Returns:
            float: spring damping of the particle system
        """
        return self.get_attribute("physxAutoParticleCloth:springDamping")

    @damping.setter
    def damping(self, damping):
        """
        Args:
            damping (float): spring damping of the particle system
        """
        self.set_attribute("physxAutoParticleCloth:springDamping", damping)

    @property
    def shear_stiffness(self):
        """
        Returns:
            float: spring shear_stiffness of the particle system
        """
        return self.get_attribute("physxAutoParticleCloth:springShearStiffness")

    @shear_stiffness.setter
    def shear_stiffness(self, shear_stiffness):
        """
        Args:
            shear_stiffness (float): spring shear_stiffness of the particle system
        """
        self.set_attribute("physxAutoParticleCloth:springShearStiffness", shear_stiffness)

    @property
    def stretch_stiffness(self):
        """
        Returns:
            float: spring stretch_stiffness of the particle system
        """
        return self.get_attribute("physxAutoParticleCloth:springStretchStiffness")

    @stretch_stiffness.setter
    def stretch_stiffness(self, stretch_stiffness):
        """
        Args:
            stretch_stiffness (float): spring stretch_stiffness of the particle system
        """
        self.set_attribute("physxAutoParticleCloth:springStretchStiffness", stretch_stiffness)

    @property
    def particle_group(self):
        """
        Returns:
            int: Particle group this instancer belongs to
        """
        return self.get_attribute(attr="physxParticle:particleGroup")

    @particle_group.setter
    def particle_group(self, group):
        """
        Args:
            group (int): Particle group this instancer belongs to
        """
        self.set_attribute(attr="physxParticle:particleGroup", val=group)

    def _dump_state(self):
        # Run super first
        state = super()._dump_state()
        state["particle_group"] = self.particle_group
        state["n_particles"] = self.n_particles
        state["particle_positions"] = self.compute_particle_positions()
        state["particle_velocities"] = self.particle_velocities
        return state

    def _load_state(self, state):
        # Run super first
        super()._load_state(state=state)
        # Sanity check the identification number and particle group
        assert self.particle_group == state["particle_group"], (
            f"Got mismatch in particle group for this cloth "
            f"when loading state! Should be: {self.particle_group}, got: {state['particle_group']}."
        )

        # Set values appropriately
        self._n_particles = state["n_particles"]
        # Make sure the loaded state is a numpy array, it could have been accidentally casted into a list during
        # JSON-serialization
        self.particle_velocities = (
            th.tensor(state["particle_velocities"])
            if not isinstance(state["particle_velocities"], th.Tensor)
            else state["particle_velocities"]
        )
        self.set_particle_positions(positions=state["particle_positions"])

    def serialize(self, state):
        # Run super first
        state_flat = super().serialize(state=state)

        return th.cat(
            [
                state_flat,
                th.tensor([state["particle_group"], state["n_particles"]], dtype=th.float32),
                state["particle_positions"].reshape(-1),
                state["particle_velocities"].reshape(-1),
            ]
        )

    def deserialize(self, state):
        # Run super first
        state_dict, idx = super().deserialize(state=state)

        particle_group = int(state[idx])
        n_particles = int(state[idx + 1])

        # Sanity check the identification number
        assert self.particle_group == particle_group, (
            f"Got mismatch in particle group for this particle "
            f"instancer when deserializing state! Should be: {self.particle_group}, got: {particle_group}."
        )

        # De-compress from 1D array
        state_dict["particle_group"] = particle_group
        state_dict["n_particles"] = n_particles

        # Process remaining keys and reshape automatically
        keys = ("particle_positions", "particle_velocities")
        sizes = ((n_particles, 3), (n_particles, 3))

        idx += 2
        for key, size in zip(keys, sizes):
            length = math.prod(size)
            state_dict[key] = state[idx : idx + length].reshape(size)
            idx += length

        return state_dict, idx

    def reset(self):
        """
        Reset the points to their default positions in the local frame, and also zeroes out velocities
        """
        if self.initialized:
            self.set_attribute(attr="points", val=lazy.pxr.Vt.Vec3fArray(self._default_positions.tolist()))
            self.particle_velocities = th.zeros((self._n_particles, 3))
