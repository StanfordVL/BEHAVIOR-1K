from functools import cached_property

from scipy.spatial import Delaunay
import torch as th

import omnigibson.lazy as lazy
from omnigibson.utils.geometry_utils import (
    check_points_in_cone,
    check_points_in_cube,
    check_points_in_cylinder,
    check_points_in_sphere,
)
import omnigibson.utils.transform_utils as T
from omnigibson.prims.xform_prim import XFormPrim
from omnigibson.utils.numpy_utils import vtarray_to_torch
from omnigibson.utils.ui_utils import create_module_logger
from omnigibson.utils.usd_utils import mesh_prim_shape_to_trimesh_mesh

# Create module logger
log = create_module_logger(module_name=__name__)


class GeomPrim(XFormPrim):
    """
    Provides high level functions to deal with a geom prim and its attributes / properties.
    If there is an geom prim present at the path, it will use it. By default, a geom prim cannot be directly
    created from scratch.

    Geom prims are not inherently distinguished as collision or visual. Instead, at the link level
    (RigidPrim), they are tracked separately based on whether they appear under a collision scope prim.
    Collision-related APIs and methods live on RigidPrim and operate on all collision meshes of a link.

    Args:
        relative_prim_path (str): Scene-local prim path of the Prim to encapsulate or create.
        name (str): Name for the object. Names need to be unique per scene.
        load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
            loading this prim at runtime. For this mesh prim, the below values can be specified:
    """

    def __init__(
        self,
        relative_prim_path,
        name,
        load_config=None,
    ):
        self._mesh_type = None
        self._applied_physics_material = None

        # Run super method
        super().__init__(
            relative_prim_path=relative_prim_path,
            name=name,
            load_config=load_config,
        )

    def _load(self):
        # This should not be called, because this prim cannot be instantiated from scratch!
        raise NotImplementedError("By default, a geom prim cannot be created from scratch.")

    def _post_load(self):
        super()._post_load()
        self._mesh_type = self.prim.GetTypeName()

    @property
    def purpose(self):
        """
        Returns:
            str: the purpose used for this geom, one of {"default", "render", "proxy", "guide"}
        """
        return self.get_attribute("purpose")

    @purpose.setter
    def purpose(self, purpose):
        """
        Sets the purpose of this geom

        Args:
            purpose (str): the purpose used for this geom, one of {"default", "render", "proxy", "guide"}
        """
        self.set_attribute("purpose", purpose)

    @property
    def color(self):
        """
        Returns:
            None or 3-array: If set, the default RGB color used for this visual geom
        """
        if self.has_material():
            return self.material.diffuse_color_constant
        else:
            color = self.get_attribute("primvars:displayColor")
            return None if color is None else th.tensor(color)[0]

    @color.setter
    def color(self, rgb):
        """
        Sets the RGB color of this visual mesh

        Args:
            3-array: The default RGB color used for this visual geom
        """
        rgb = th.as_tensor(rgb)
        if self.has_material():
            self.material.diffuse_color_constant = rgb
        else:
            self.set_attribute("primvars:displayColor", rgb.cpu().numpy())

    @property
    def opacity(self):
        """
        Returns:
            None or float: If set, the default opacity used for this visual geom
        """
        if self.has_material():
            return self.material.opacity_constant
        else:
            opacity = self.get_attribute("primvars:displayOpacity")
            return None if opacity is None else th.tensor(opacity)[0]

    @opacity.setter
    def opacity(self, opacity):
        """
        Sets the opacity of this visual mesh

        Args:
            opacity: The default opacity used for this visual geom
        """
        if self.has_material():
            self.material.opacity_constant = opacity
        else:
            self.set_attribute("primvars:displayOpacity", [opacity])

    @cached_property
    def points(self):
        """
        Returns:
            th.tensor: Local poses of all points
        """
        # If the geom is a mesh we can directly return its points.
        mesh = self.prim
        mesh_type = mesh.GetPrimTypeInfo().GetTypeName()
        if mesh_type == "Mesh":
            # If the geom is a mesh we can directly return its points.
            return vtarray_to_torch(mesh.GetAttribute("points").Get(), dtype=th.float32)
        else:
            # Return the vertices of the trimesh
            return th.tensor(mesh_prim_shape_to_trimesh_mesh(mesh).vertices, dtype=th.float32)

    @cached_property
    def faces(self):
        mesh = self.prim
        mesh_type = mesh.GetPrimTypeInfo().GetTypeName()
        if mesh_type != "Mesh":
            log.warning(f"Geom {self.prim_path} is not a mesh, returning None for faces.")
            return None

        face_vertex_counts = vtarray_to_torch(mesh.GetAttribute("faceVertexCounts").Get(), dtype=th.int)
        face_indices = vtarray_to_torch(mesh.GetAttribute("faceVertexIndices").Get(), dtype=th.int)

        faces = []
        i = 0
        for count in face_vertex_counts:
            for j in range(count - 2):
                faces.append([face_indices[i], face_indices[i + j + 1], face_indices[i + j + 2]])
            i += count
        faces = th.tensor(faces, dtype=th.int)

        return faces

    @cached_property
    def delaunay_triangulation(self):
        return Delaunay(self.points.numpy())

    @property
    def geom_type(self):
        """
        Returns:
            str: the type of the geom prim, one of {"Sphere", "Cube", "Cone", "Cylinder", "Mesh"}
        """
        return self._prim.GetPrimTypeInfo().GetTypeName()

    @cached_property
    def mesh_face_centroids(self):
        return self.points[self.faces].mean(dim=1)

    @cached_property
    def mesh_face_normals(self):
        # Get the vertices for each triangle
        vertices = self.points[self.faces]  # Shape: (N_triangles, 3, 3)

        # Compute two edges of each triangle
        edge1 = vertices[:, 1] - vertices[:, 0]  # Shape: (N_triangles, 3)
        edge2 = vertices[:, 2] - vertices[:, 0]  # Shape: (N_triangles, 3)

        # Compute the cross product of the two edges to get the normal vector
        face_normals = th.cross(edge1, edge2, dim=1)  # Shape: (N_triangles, 3)

        # Normalize the normal vectors
        face_normals_norm = th.norm(face_normals, dim=1, keepdim=True)

        # Handle potential division by zero for degenerate faces
        epsilon = 1e-8
        face_normals_norm = th.clamp(face_normals_norm, min=epsilon)

        face_normals = face_normals / face_normals_norm

        return face_normals

    def check_local_points_in_volume(self, particle_positions_in_mesh_frame):
        if self._mesh_type == "Mesh":
            return th.as_tensor(self.delaunay_triangulation.find_simplex(particle_positions_in_mesh_frame.numpy())) >= 0
        elif self._mesh_type == "Sphere":
            return check_points_in_sphere(
                size=self.get_attribute("radius"),
                particle_positions=particle_positions_in_mesh_frame,
            )
        elif self._mesh_type == "Cylinder":
            return check_points_in_cylinder(
                size=[self.get_attribute("radius"), self.get_attribute("height")],
                particle_positions=particle_positions_in_mesh_frame,
            )
        elif self._mesh_type == "Cone":
            return check_points_in_cone(
                size=[self.get_attribute("radius"), self.get_attribute("height")],
                particle_positions=particle_positions_in_mesh_frame,
            )
        elif self._mesh_type == "Cube":
            return check_points_in_cube(
                size=self.get_attribute("size"),
                particle_positions=particle_positions_in_mesh_frame,
            )
        else:
            raise ValueError(f"Cannot check in volume for mesh of type: {self._mesh_type}")

    def check_points_in_volume(self, particle_positions_world):
        # Move particles into local frame
        world_pose_w_scale = self.scaled_transform
        particle_positions_world_homogeneous = th.cat(
            (particle_positions_world, th.ones((particle_positions_world.shape[0], 1))), dim=1
        )
        particle_positions_local = (particle_positions_world_homogeneous @ th.linalg.inv(world_pose_w_scale).T)[:, :3]
        return self.check_local_points_in_volume(particle_positions_local)

    @property
    def points_in_parent_frame(self):
        points = self.points
        if points is None:
            return None
        position, orientation = self.get_position_orientation(frame="parent")
        scale = self.scale
        points_scaled = points * scale
        points_rotated = (T.quat2mat(orientation) @ points_scaled.T).T
        points_transformed = points_rotated + position
        return points_transformed

    @property
    def aabb(self):
        world_pose_w_scale = self.scaled_transform

        # transform self.points into world frame
        points = self.points
        points_homogeneous = th.cat((points, th.ones((points.shape[0], 1))), dim=1)
        points_transformed = (points_homogeneous @ world_pose_w_scale.T)[:, :3]

        aabb_lo = th.min(points_transformed, dim=0).values
        aabb_hi = th.max(points_transformed, dim=0).values
        return aabb_lo, aabb_hi

    @property
    def aabb_extent(self):
        """
        Bounding box extent of this geom prim

        Returns:
            3-array: (x,y,z) bounding box
        """
        min_corner, max_corner = self.aabb
        return max_corner - min_corner

    @property
    def aabb_center(self):
        """
        Bounding box center of this geom prim

        Returns:
            3-array: (x,y,z) bounding box center
        """
        min_corner, max_corner = self.aabb
        return (max_corner + min_corner) / 2.0

    @cached_property
    def extent(self):
        """
        Returns:
            th.tensor: The unscaled 3d extent of the mesh in its local frame.
        """
        points = self.points
        return th.max(points, dim=0).values - th.min(points, dim=0).values

    def apply_physics_material(self, physics_material, weaker_than_descendants=False):
        """
        Used to apply physics material to the held prim and optionally its descendants.

        Args:
            physics_material (PhysicsMaterial): physics material to be applied to the held prim. This where you want to
                                                define friction, restitution..etc. Note: if a physics material is not
                                                defined, the defaults will be used from PhysX.
            weaker_than_descendants (bool, optional): True if the material shouldn't override the descendants
                                                      materials, otherwise False. Defaults to False.
        """
        if weaker_than_descendants:
            self._binding_api.Bind(
                physics_material.material,
                bindingStrength=lazy.pxr.UsdShade.Tokens.weakerThanDescendants,
                materialPurpose="physics",
            )
        else:
            self._binding_api.Bind(
                physics_material.material,
                bindingStrength=lazy.pxr.UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )
        self._applied_physics_material = physics_material
        return

    def get_applied_physics_material(self):
        """
        Returns the current applied physics material in case it was applied using apply_physics_material or not.

        Returns:
            PhysicsMaterial: the current applied physics material.
        """
        if self._applied_physics_material is not None:
            return self._applied_physics_material
        else:
            physics_binding = self._binding_api.GetDirectBinding(materialPurpose="physics")
            path = physics_binding.GetMaterialPath()
            if path == "":
                return None
            else:
                self._applied_physics_material = lazy.isaacsim.core.api.materials.PhysicsMaterial(prim_path=path)
                return self._applied_physics_material
