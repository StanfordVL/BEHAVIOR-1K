# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import math
from functools import cached_property
import os
from pathlib import Path
import tempfile
from typing import Literal
import uuid

from omnigibson.prims.xform_prim import XFormPrim
from omnigibson.utils.python_utils import multi_dim_linspace
import torch as th
from omnigibson.utils.ui_utils import create_module_logger
import trimesh

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.macros import create_module_macros, gm
from omnigibson.prims.geom_prim import GeomPrim
from omnigibson.utils.numpy_utils import vtarray_to_torch
from omnigibson.utils.sim_utils import CsRawData
from omnigibson.utils.usd_utils import (
    mesh_prim_to_trimesh_mesh,
    sample_mesh_keypoints,
    delete_or_deactivate_prim,
)

# Create module logger
log = create_module_logger(module_name=__name__)

# Create settings for this module
m = create_module_macros(module_path=__file__)

CLOTH_CONFIGURATIONS = ["default", "settled", "folded", "crumpled"]

# Subsample cloth particle points to boost performance
m.ALLOW_MULTIPLE_CLOTH_MESH_COMPONENTS = True  # TODO: Disable after new dataset release
m.N_CLOTH_KEYPOINTS = 1000
m.FOLDING_INCREMENTS = 100
m.CRUMPLING_INCREMENTS = 100
m.KEYPOINT_COVERAGE_THRESHOLD = 0.75
m.N_CLOTH_KEYFACES = 500
m.MAX_CLOTH_PARTICLES = 20000  # Comes from a limitation in physx - do not increase
m.CLOTH_PARTICLE_CONTACT_OFFSET = 0.0075
m.CLOTH_REMESHING_ERROR_THRESHOLD = 0.05


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

        # Save the default point configuration
        self.save_configuration("default", self.points)

        # Remesh the object if necessary
        force_remesh = self._load_config.get("force_remesh", False)
        should_remesh_because_of_scale = self._load_config.get("remesh", True) and not th.allclose(
            self.scale, th.ones(3)
        )
        # TODO: Remove the legacy check after the next dataset release
        should_remesh_because_legacy = self._load_config.get(
            "remesh", True
        ) and self.get_available_configurations() == ["default"]
        if should_remesh_because_of_scale or should_remesh_because_legacy or force_remesh:
            # Remesh the object if necessary
            log.warning(
                f"Remeshing cloth {self.name}. This happens when cloth is loaded with non-unit scale or forced using the forced_remesh argument. It invalidates precached info for settled, folded, and crumpled configurations."
            )
            self._remesh()

        # Set the mesh to use the desired configuration
        points_configuration = self._load_config.get("default_point_configuration", "default")
        self.reset_points_to_configuration(configuration=points_configuration)

        # Clothify this prim, which is assumed to be a mesh
        self.cloth_system.clothify_mesh_prim(mesh_prim=self._prim)

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

    def _remesh(self):
        assert self.prim is not None, "Cannot remesh a non-existent prim!"
        has_uv_mapping = self.prim.GetAttribute("primvars:st").Get() is not None

        # We will remesh in pymeshlab, but it doesn't allow programmatic construction of a mesh with texcoords so
        # we convert our mesh into a trimesh mesh, then export it to a temp file, then load it into pymeshlab
        scaled_world_transform = self.scaled_transform
        # Convert to trimesh mesh (in world frame)
        tm = mesh_prim_to_trimesh_mesh(
            mesh_prim=self.prim, include_normals=True, include_texcoord=True, world_frame=True
        )
        # Tmp file written to: {tmp_dir}/{tmp_fname}/{tmp_fname}.obj
        tmp_name = str(uuid.uuid4())
        tmp_dir = os.path.join(tempfile.gettempdir(), tmp_name)
        tmp_fpath = os.path.join(tmp_dir, f"{tmp_name}.obj")
        Path(tmp_dir).mkdir(parents=True, exist_ok=True)
        tm.export(tmp_fpath)

        # Start with the default particle distance
        particle_distance = self.cloth_system.particle_contact_offset * 2 / 1.5

        # Repetitively re-mesh at lower resolution until we have a mesh that has less than MAX_CLOTH_PARTICLES vertices
        import pymeshlab  # We import this here because it takes a few seconds to load.

        for _ in range(10):
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(tmp_fpath)

            # Re-mesh based on @particle_distance - distance chosen such that at rest particles should be just touching
            # each other. The 1.5 magic number comes from the particle cloth demo from omni
            # Note that this means that the particles will overlap with each other, since at dist = 2 * contact_offset
            # the particles are just touching each other at rest

            avg_edge_percentage_mismatch = 1.0
            # Loop re-meshing until average edge percentage is within error threshold or we reach the max number of tries
            for _ in range(5):
                if avg_edge_percentage_mismatch <= m.CLOTH_REMESHING_ERROR_THRESHOLD:
                    break

                ms.meshing_isotropic_explicit_remeshing(
                    iterations=5, adaptive=True, targetlen=pymeshlab.AbsoluteValue(particle_distance)
                )

                # If the cloth has multiple pieces, only keep the largest one
                ms.generate_splitting_by_connected_components(delete_source_mesh=True)
                if len(ms) > 1:
                    assert m.ALLOW_MULTIPLE_CLOTH_MESH_COMPONENTS, "Cloth mesh has multiple components!"

                    log.warning(
                        f"The cloth mesh has {len(ms)} disconnected components. To simplify, we only keep the mesh with largest face number."
                    )
                    biggest_face_num = 0
                    for split_mesh in ms:
                        face_num = split_mesh.face_number()
                        if face_num > biggest_face_num:
                            biggest_face_num = face_num
                    new_ms = pymeshlab.MeshSet()
                    for split_mesh in ms:
                        if split_mesh.face_number() == biggest_face_num:
                            new_ms.add_mesh(split_mesh)
                    ms = new_ms

                avg_edge_percentage_mismatch = abs(
                    1.0 - particle_distance / ms.get_geometric_measures()["avg_edge_length"]
                )
            else:
                # Terminate anyways, but don't fail
                log.warning("The generated cloth may not have evenly distributed particles.")

            # Check if we have too many vertices
            cm = ms.current_mesh()
            if cm.vertex_number() > m.MAX_CLOTH_PARTICLES:
                # We have too many vertices, so we will re-mesh again
                particle_distance *= math.sqrt(2)  # halve the number of vertices
                log.warning(
                    f"Too many vertices ({cm.vertex_number()})! Re-meshing with particle distance {particle_distance}..."
                )
            else:
                break
        else:
            raise ValueError(f"Could not remesh with less than MAX_CLOTH_PARTICLES ({m.MAX_CLOTH_PARTICLES}) vertices!")

        # Re-write data to @mesh_prim
        new_faces = cm.face_matrix()
        new_vertices = cm.vertex_matrix()
        new_normals = cm.vertex_normal_matrix()
        texcoord = cm.wedge_tex_coord_matrix() if has_uv_mapping else None
        tm = trimesh.Trimesh(
            vertices=new_vertices,
            faces=new_faces,
            vertex_normals=new_normals,
        )
        # Apply the inverse of the world transform to get the mesh back into its local frame
        tm.apply_transform(th.linalg.inv_ex(scaled_world_transform).inverse)

        # Update the mesh prim to store the new information. First update the non-configuration-
        # dependent fields
        face_vertex_counts = th.tensor([len(face) for face in tm.faces], dtype=int).cpu().numpy()
        self.prim.GetAttribute("faceVertexCounts").Set(face_vertex_counts)
        self.prim.GetAttribute("faceVertexIndices").Set(tm.faces.flatten())
        self.prim.GetAttribute("normals").Set(lazy.pxr.Vt.Vec3fArray.FromNumpy(tm.vertex_normals))
        if has_uv_mapping:
            self.prim.GetAttribute("primvars:st").Set(lazy.pxr.Vt.Vec2fArray.FromNumpy(texcoord))

        # Remove the properties for all configurations
        for config in CLOTH_CONFIGURATIONS:
            attr_name = f"points_{config}"
            if self.prim.HasAttribute(attr_name):
                self.prim.RemoveProperty(attr_name)

        # Then update the configuration-dependent fields
        self.save_configuration("default", th.tensor(tm.vertices, dtype=th.float32))

        # Then update the points to the default configuration
        self.reset_points_to_configuration("default")

    def generate_settled_configuration(self):
        """
        Generate a settled configuration for the cloth by running a few steps of simulation to let the cloth settle
        """
        # Reset position first (moving to a position where the AABB is just on the floor)
        self.reset_points_to_configuration("default")
        self.set_position_orientation(position=th.zeros(3), orientation=th.tensor([0, 0, 0, 1.0]))
        self.set_position_orientation(
            position=th.tensor([0, 0, self.aabb_extent[2] / 2.0 - self.aabb_center[2]]),
            orientation=th.tensor([0, 0, 0, 1.0]),
        )

        # Run a few steps of simulation to let the cloth settle
        for _ in range(300):
            og.sim.step()

        # Save the settled configuration
        self.save_configuration("settled", self.points)

    def generate_folded_configuration(self):
        # Settle and reset position first (moving to a position where the AABB is just on the floor)
        self.reset_points_to_configuration("settled")
        self.set_position_orientation(position=th.zeros(3), orientation=th.tensor([0, 0, 0, 1.0]))
        self.set_position_orientation(
            position=th.tensor([0, 0, self.aabb_extent[2] / 2.0 - self.aabb_center[2]]),
            orientation=th.tensor([0, 0, 0, 1.0]),
        )

        for _ in range(100):
            og.sim.step()

        # Fold - first stage. Take the bottom third of Y positions and fold them over towards
        # the middle third.
        pos = self.compute_particle_positions()
        y_min = th.min(pos[:, 1])
        y_max = th.max(pos[:, 1])
        y_bottom_third = y_min + (y_max - y_min) / 3
        y_top_third = y_min + 2 * (y_max - y_min) / 3
        first_folding_indices = th.where(pos[:, 1] < y_bottom_third)[0]
        first_folding_initial_pos = th.clone(pos[first_folding_indices])
        first_staying_pos = pos[pos[:, 1] >= y_bottom_third]
        first_staying_z_height = th.max(first_staying_pos[:, 2]) - th.min(first_staying_pos[:, 2])
        first_folding_final_pos = th.clone(first_folding_initial_pos)
        first_folding_final_pos[:, 1] = (
            2 * y_bottom_third - first_folding_final_pos[:, 1]
        )  # Mirror bottom to the above of y_mid_bottom
        first_folding_final_pos[:, 2] += first_staying_z_height  # Add a Z offset to keep particles from overlapping
        for ctrl_pts in multi_dim_linspace(first_folding_initial_pos, first_folding_final_pos, m.FOLDING_INCREMENTS):
            all_pts = th.clone(pos)
            all_pts[first_folding_indices] = ctrl_pts
            self.set_particle_positions(all_pts)
            og.sim.step()

        # Fold - second stage. Take the top third of original Y positions and fold them over towards the
        # middle too.
        pos = self.compute_particle_positions()
        second_folding_indices = th.where(pos[:, 1] > y_top_third)[0]
        second_folding_initial_pos = th.clone(pos[second_folding_indices])
        second_staying_pos = pos[pos[:, 1] <= y_top_third]
        second_staying_z_height = th.max(second_staying_pos[:, 2]) - th.min(second_staying_pos[:, 2])
        second_folding_final_pos = th.clone(second_folding_initial_pos)
        second_folding_final_pos[:, 1] = (
            2 * y_top_third - second_folding_final_pos[:, 1]
        )  # Mirror top to the below of y_mid
        second_folding_final_pos[:, 2] += second_staying_z_height  # Add a Z offset to keep particles from overlapping
        for ctrl_pts in multi_dim_linspace(second_folding_initial_pos, second_folding_final_pos, m.FOLDING_INCREMENTS):
            all_pts = th.clone(pos)
            all_pts[second_folding_indices] = ctrl_pts
            self.set_particle_positions(all_pts)
            og.sim.step()

        # Fold - third stage. Fold along the X axis, in half.
        pos = self.compute_particle_positions()
        x_min = th.min(pos[:, 0])
        x_max = th.max(pos[:, 0])
        x_mid = (x_min + x_max) / 2
        third_folding_indices = th.where(pos[:, 0] > x_mid)[0]
        third_folding_initial_pos = th.clone(pos[third_folding_indices])
        third_staying_pos = pos[pos[:, 0] <= x_mid]
        third_staying_z_height = th.max(third_staying_pos[:, 2]) - th.min(third_staying_pos[:, 2])
        third_folding_final_pos = th.clone(third_folding_initial_pos)
        third_folding_final_pos[:, 0] = 2 * x_mid - third_folding_final_pos[:, 0]
        third_folding_final_pos[:, 2] += third_staying_z_height  # Add a Z offset to keep particles from overlapping
        for ctrl_pts in multi_dim_linspace(third_folding_initial_pos, third_folding_final_pos, m.FOLDING_INCREMENTS):
            all_pts = th.clone(pos)
            all_pts[third_folding_indices] = ctrl_pts
            self.set_particle_positions(all_pts)
            og.sim.step()

        # Let things settle
        for _ in range(100):
            og.sim.step()

        # Save the folded configuration
        self.save_configuration("folded", self.points)

    def generate_crumpled_configuration(self):
        # Settle and reset position first (moving to a position where the AABB is just on the floor)
        self.reset_points_to_configuration("settled")
        self.set_position_orientation(position=th.zeros(3), orientation=th.tensor([0, 0, 0, 1.0]))
        self.set_position_orientation(
            position=th.tensor([0, 0, self.aabb_extent[2] / 2.0 - self.aabb_center[2]]),
            orientation=th.tensor([0, 0, 0, 1.0]),
        )
        for _ in range(100):
            og.sim.step()

        # We just need to generate the side planes
        box_half_extent = self.aabb_extent / 2
        plane_centers = (
            th.tensor(
                [
                    [1, 0, 1],
                    [0, 1, 1],
                    [-1, 0, 1],
                    [0, -1, 1],
                ]
            )
            * box_half_extent
        )
        plane_prims = []
        plane_motions = []
        for i, pc in enumerate(plane_centers):
            plane = lazy.isaacsim.core.api.objects.ground_plane.GroundPlane(
                prim_path=f"/World/plane_{i}",
                name=f"plane_{i}",
                z_position=0,
                size=box_half_extent[2].item(),
                color=None,
                visible=False,
            )

            plane_as_prim = XFormPrim(
                relative_prim_path=f"/plane_{i}",
                name=plane.name,
            )
            plane_as_prim.load(None)

            # Build the plane orientation from the plane normal
            horiz_dir = pc - th.tensor([0, 0, box_half_extent[2]])
            plane_z = -1 * horiz_dir / th.norm(horiz_dir)
            plane_x = th.tensor([0, 0, 1], dtype=th.float32)
            plane_y = th.cross(plane_z, plane_x)
            plane_mat = th.stack([plane_x, plane_y, plane_z], dim=1)
            plane_quat = T.mat2quat(plane_mat)
            plane_as_prim.set_position_orientation(pc, plane_quat)

            plane_prims.append(plane_as_prim)
            plane_motions.append(plane_z)

        # Calculate end positions for all walls
        end_positions = []
        for i in range(4):
            plane_prim = plane_prims[i]
            position = plane_prim.get_position_orientation()[0]
            end_positions.append(position + plane_motions[i] * box_half_extent)

        for step in range(m.CRUMPLING_INCREMENTS):
            # Move all walls a small amount
            for i in range(4):
                plane_prim = plane_prims[i]
                current_pos = multi_dim_linspace(
                    plane_prim.get_position_orientation()[0], end_positions[i], m.CRUMPLING_INCREMENTS
                )[step]
                plane_prim.set_position_orientation(position=current_pos)

            og.sim.step()

            # Check cloth height
            cloth_positions = self.compute_particle_positions()
            max_height = th.max(cloth_positions[:, 2])

            # Get distance between facing walls (assuming walls 0-2 and 1-3 are facing pairs)
            wall_dist_1 = th.linalg.norm(
                plane_prims[0].get_position_orientation()[0] - plane_prims[2].get_position_orientation()[0]
            )
            wall_dist_2 = th.linalg.norm(
                plane_prims[1].get_position_orientation()[0] - plane_prims[3].get_position_orientation()[0]
            )
            min_wall_dist = min(wall_dist_1, wall_dist_2)

            # Stop if cloth height exceeds wall distance
            if max_height > min_wall_dist:
                break

        # Let things settle
        for _ in range(100):
            og.sim.step()

        # Save the folded configuration
        self.save_configuration("crumpled", self.points)

        # Remove the planes
        for plane_prim in plane_prims:
            delete_or_deactivate_prim(plane_prim.prim_path)

    def get_available_configurations(self):
        """
        Returns:
            list: List of available configurations for this cloth prim
        """
        return [x for x in CLOTH_CONFIGURATIONS if self.prim.HasAttribute(f"points_{x}")]

    def save_configuration(self, configuration: Literal["default", "settled", "folded", "crumpled"], points: th.tensor):
        """
        Save the current configuration of the cloth to a specific configuration

        Args:
            configuration (Literal["default", "settled", "folded", "crumpled"]): Configuration to save the cloth to
        """
        # Get the points arguments stored on the USD prim
        assert configuration in CLOTH_CONFIGURATIONS, f"Invalid cloth configuration {configuration}!"
        assert self.prim is not None, "Cannot save configuration for a non-existent prim!"
        attr_name = f"points_{configuration}"
        points_default_attrib = (
            self.prim.GetAttribute(attr_name)
            if self.prim.HasAttribute(attr_name)
            else self.prim.CreateAttribute(attr_name, lazy.pxr.Sdf.ValueTypeNames.Float3Array)
        )
        points_default_attrib.Set(lazy.pxr.Vt.Vec3fArray.FromNumpy(points.cpu().numpy()))

    def reset_points_to_configuration(self, configuration: Literal["default", "settled", "folded", "crumpled"]):
        """
        Reset the cloth to a specific configuration

        Args:
            configuration (Literal["default", "settled", "folded", "crumpled"]): Configuration to reset the cloth to
        """
        # Get the points arguments stored on the USD prim
        assert (
            configuration in self.get_available_configurations()
        ), f"Invalid or unavailable cloth configuration {configuration}!"
        attr_name = f"points_{configuration}"
        points = self.get_attribute(attr=attr_name)
        self.set_attribute(attr="points", val=points)

        # Reset velocities to zero if velocities are present
        if self.prim.HasAttribute("velocities"):
            self.set_attribute(attr="velocities", val=lazy.pxr.Vt.Vec3fArray(th.zeros((len(points), 3)).tolist()))

    # For cloth, points should NOT be @cached_property because their local poses change over time
    @property
    def points(self):
        """
        Returns:
            th.tensor: Local poses of all points
        """
        # If the geom is a mesh we can directly return its points.
        mesh = self.prim
        mesh_type = mesh.GetPrimTypeInfo().GetTypeName()
        assert mesh_type == "Mesh", f"Expected a mesh prim, got {mesh_type} instead!"
        return vtarray_to_torch(mesh.GetAttribute("points").Get(), dtype=th.float32)

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

    @cached_property
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
            points_configuration = self._load_config.get("default_point_configuration", "default")
            self.reset_points_to_configuration(points_configuration)

    @cached_property
    def is_meta_link(self):
        return False

    @cached_property
    def meta_link_type(self):
        raise ValueError(f"{self.name} is not a meta link")

    @cached_property
    def meta_link_id(self):
        """The meta link id of this link, if the link is a meta link.

        The meta link ID is a semantic identifier for the meta link within the meta link type. It is
        used when an object has multiple meta links of the same type. It can be just a numerical index,
        or for some objects, it will be a string that can be matched to other meta links. For example,
        a stove might have toggle buttons named "left" and "right", and heat sources named "left" and
        "right". The meta link ID can be used to match the toggle button to the heat source.
        """
        raise ValueError(f"{self.name} is not a meta link")

    @cached_property
    def meta_link_sub_id(self):
        """The integer meta link sub id of this link, if the link is a meta link.

        The meta link sub ID identifies this link as one of the parts of a meta link. For example, an
        attachment meta link's ID will be the attachment pair name, and each attachment point that
        works with that pair will show up as a separate link with a unique sub ID.
        """
        raise ValueError(f"{self.name} is not a meta link")
