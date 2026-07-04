"""USD visual mesh import for the Newton backend."""

from dataclasses import dataclass

import newton
import numpy as np
import warp as wp
from newton._src.usd import utils as usd_utils
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade


@dataclass(frozen=True)
class VisualImportResult:
    """Summary of render-only visual shapes added to a Newton builder."""

    shape_indices: tuple[int, ...]
    body_indices: tuple[int, ...]
    mesh_sources: tuple[object, ...]


def add_usd_visual_shapes(builder, usd_path, import_info, *, root_xform=None, root_scale=None, label_prefix=None):
    """Add visible-only USD meshes for an object already imported for physics.

    Newton's USD importer can load visual geometry directly, but BEHAVIOR assets
    currently expose native importer crashes when full scenes combine many
    visual and collision meshes. Keep the stable collision-only physics import
    and add render-only meshes in a separate pass, bound to the body indices
    returned by that import.
    """
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"USD failed to open for visual import: {usd_path}")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    root_mat = _xform_to_mat44(root_xform) if root_xform is not None else None
    scale_mat = _scale_to_mat44(root_scale)
    body_path_map = import_info.get("path_body_map") or {}
    visual_shape_indices = []
    visual_body_indices = []
    visual_mesh_sources = []

    visual_cfg = newton.ModelBuilder.ShapeConfig(
        has_shape_collision=False,
        has_particle_collision=False,
        is_visible=True,
        is_solid=True,
    )

    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        if _is_enabled_collider(prim) or not _is_effectively_visible(prim):
            continue

        body_path = _nearest_imported_body_path(prim, body_path_map)
        if body_path is None:
            continue

        body_idx = body_path_map[body_path]
        meshes = _load_render_meshes(prim)
        if not meshes:
            continue

        xform, scale = _mesh_xform_relative_to_body(prim, body_idx, xform_cache, builder, root_mat, scale_mat)
        for mesh_label, mesh in meshes:
            label = str(prim.GetPath())
            if mesh_label:
                label = f"{label}/{mesh_label}"
            if label_prefix:
                label = f"{label_prefix}/{label}"
            shape_idx = builder.add_shape_mesh(
                body_idx,
                xform=xform,
                scale=scale,
                mesh=mesh,
                cfg=visual_cfg,
                color=_viewer_color_for_mesh(mesh),
                label=label,
            )
            visual_shape_indices.append(shape_idx)
            visual_body_indices.append(body_idx)
            visual_mesh_sources.append(mesh)

    del stage
    return VisualImportResult(tuple(visual_shape_indices), tuple(visual_body_indices), tuple(visual_mesh_sources))


def _load_render_meshes(prim):
    try:
        mesh = UsdGeom.Mesh(prim)
        points = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
        indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        counts = np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
    except Exception:
        return None
    if len(points) == 0 or len(indices) == 0 or len(counts) == 0:
        return ()

    flip_winding = False
    orientation_attr = mesh.GetOrientationAttr()
    if orientation_attr:
        orientation = orientation_attr.Get()
        flip_winding = bool(orientation and orientation.lower() == "lefthanded")

    uvs = _load_uvs(prim, points, indices, counts)
    material_subsets = _material_subsets(prim, len(counts))
    if material_subsets:
        meshes = []
        covered_faces = set()
        for subset in material_subsets:
            covered_faces.update(subset.face_indices)
            faces, corner_indices = _fan_triangulate_faces(counts, indices, flip_winding, subset.face_indices)
            render_mesh = _make_render_mesh(points, indices, faces, corner_indices, uvs, subset.material_props)
            if render_mesh is not None:
                meshes.append((subset.name, render_mesh))

        # USD materialBind subsets are expected to partition mesh faces. Keep a
        # parent-material fallback for malformed assets so unassigned faces do
        # not silently disappear from the viewer.
        remaining_faces = tuple(face_idx for face_idx in range(len(counts)) if face_idx not in covered_faces)
        if remaining_faces:
            faces, corner_indices = _fan_triangulate_faces(counts, indices, flip_winding, remaining_faces)
            render_mesh = _make_render_mesh(
                points,
                indices,
                faces,
                corner_indices,
                uvs,
                _direct_material_properties(prim),
            )
            if render_mesh is not None:
                meshes.append(("unassigned_material", render_mesh))
        return tuple(meshes)

    faces, corner_indices = _fan_triangulate_faces(counts, indices, flip_winding)
    render_mesh = _make_render_mesh(
        points, indices, faces, corner_indices, uvs, usd_utils.resolve_material_properties_for_prim(prim)
    )
    return (("", render_mesh),) if render_mesh is not None else ()


def _make_render_mesh(points, indices, faces, corner_indices, uvs, material_props):
    if len(faces) == 0:
        return None

    points, faces, uvs = _compact_mesh(points, indices, faces, corner_indices, uvs)
    texture = material_props.get("texture") if uvs is not None else None
    color = None if texture is not None else material_props.get("color")
    return newton.Mesh(
        points,
        faces.reshape(-1),
        uvs=uvs,
        compute_inertia=False,
        is_solid=False,
        color=color,
        roughness=material_props.get("roughness"),
        metallic=material_props.get("metallic"),
        texture=texture,
    )


@dataclass(frozen=True)
class _MaterialSubset:
    name: str
    face_indices: tuple[int, ...]
    material_props: dict


def _material_subsets(prim, face_count):
    subsets = []
    for child in prim.GetChildren():
        if not child.IsA(UsdGeom.Subset):
            continue
        subset = UsdGeom.Subset(child)
        if subset.GetFamilyNameAttr().Get() != UsdShade.Tokens.materialBind:
            continue
        face_indices = subset.GetIndicesAttr().Get()
        if face_indices is None:
            continue
        face_indices = tuple(int(face_idx) for face_idx in face_indices if 0 <= int(face_idx) < face_count)
        if not face_indices:
            continue
        subsets.append(
            _MaterialSubset(
                name=child.GetName(),
                face_indices=face_indices,
                material_props=usd_utils.resolve_material_properties_for_prim(child),
            )
        )
    return tuple(subsets)


def _direct_material_properties(prim):
    resolver = getattr(usd_utils, "_resolve_prim_material_properties", None)
    if resolver is not None:
        props = resolver(prim)
        if props is not None:
            return props
    empty = getattr(usd_utils, "_empty_material_properties", None)
    if empty is not None:
        return empty()
    return {"color": None, "metallic": None, "roughness": None, "texture": None}


def _compact_mesh(points, indices, faces, corner_indices, uvs):
    if uvs is not None and len(uvs) != len(points):
        points = points[indices[corner_indices]]
        uvs = uvs[corner_indices]
        faces = np.arange(len(points), dtype=np.int32).reshape(-1, 3)
        return points, faces, uvs

    used_vertex_indices, remapped_faces = np.unique(faces.reshape(-1), return_inverse=True)
    points = points[used_vertex_indices]
    if uvs is not None:
        uvs = uvs[used_vertex_indices]
    faces = remapped_faces.astype(np.int32).reshape(-1, 3)
    return points, faces, uvs


def _viewer_color_for_mesh(mesh):
    # ViewerGL multiplies mesh textures by the per-shape color buffer. Use a
    # neutral color for textured meshes so authored texture albedo is not tinted
    # by USD displayColor or Newton's fallback debug palette.
    if getattr(mesh, "texture", None) is not None:
        return (1.0, 1.0, 1.0)
    return mesh.color if mesh.color is not None else (1.0, 1.0, 1.0)


def _fan_triangulate_faces(counts, indices, flip_winding, selected_face_indices=None):
    selected_face_indices = set(selected_face_indices) if selected_face_indices is not None else None
    faces = []
    corner_indices = []
    cursor = 0
    for face_idx, count in enumerate(counts):
        face = indices[cursor : cursor + count]
        if selected_face_indices is None or face_idx in selected_face_indices:
            for tri_idx in range(1, count - 1):
                tri = [face[0], face[tri_idx], face[tri_idx + 1]]
                corners = [cursor, cursor + tri_idx, cursor + tri_idx + 1]
                if flip_winding:
                    tri = tri[::-1]
                    corners = corners[::-1]
                faces.append(tri)
                corner_indices.extend(corners)
        cursor += count
    return np.array(faces, dtype=np.int32), np.array(corner_indices, dtype=np.int32)


def _load_uvs(prim, points, indices, counts):
    primvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
    if not primvar:
        return None

    values = primvar.Get()
    if values is None:
        return None
    uvs = np.array(values, dtype=np.float32)
    if primvar.IsIndexed():
        authored_indices = primvar.GetIndices()
        if authored_indices is None:
            return None
        authored_indices = np.array(authored_indices, dtype=np.int32)
        if len(authored_indices) == len(indices):
            uvs = uvs[authored_indices]

    interpolation = primvar.GetInterpolation()
    if interpolation == UsdGeom.Tokens.faceVarying:
        return uvs if len(uvs) == len(indices) else None
    if len(uvs) == len(points):
        return uvs
    return None


def _nearest_imported_body_path(prim, body_path_map):
    path = prim.GetPath()
    while path != path.absoluteRootPath:
        path_str = str(path)
        if path_str in body_path_map:
            return path_str
        path = path.GetParentPath()
    return None


def _mesh_xform_relative_to_body(prim, body_idx, xform_cache, builder, root_mat, scale_mat):
    mesh_world = usd_utils.get_transform_matrix(prim, local=False, xform_cache=xform_cache)
    if scale_mat is not None:
        mesh_world = scale_mat @ mesh_world
    if root_mat is not None:
        mesh_world = root_mat @ mesh_world
    if body_idx == -1:
        rel_mat = mesh_world
    else:
        body_world = _xform_to_mat44(builder.body_q[body_idx])
        rel_mat = wp.inverse(body_world) @ mesh_world

    pos, rot, scale = wp.transform_decompose(rel_mat)
    return wp.transform(pos, rot), scale


def _xform_to_mat44(xform):
    return wp.transform_compose(xform.p, xform.q, wp.vec3(1.0))


def _scale_to_mat44(scale):
    if scale is None:
        return None
    if hasattr(scale, "detach"):
        values = scale.detach().cpu().flatten().tolist()
    elif isinstance(scale, (int, float)):
        values = [float(scale)] * 3
    else:
        values = list(scale)
    if len(values) == 1:
        values = values * 3
    values = tuple(float(value) for value in values[:3])
    if values == (1.0, 1.0, 1.0):
        return None
    return wp.transform_compose(wp.vec3(0.0), wp.quat_identity(), wp.vec3(*values))


def _is_enabled_collider(prim):
    collider = UsdPhysics.CollisionAPI(prim)
    if not collider:
        return False
    enabled = collider.GetCollisionEnabledAttr().Get()
    return enabled is not False


def _is_effectively_visible(prim):
    imageable = UsdGeom.Imageable(prim)
    return bool(imageable) and imageable.ComputeVisibility() != UsdGeom.Tokens.invisible
