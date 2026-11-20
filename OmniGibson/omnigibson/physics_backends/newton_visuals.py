"""USD visual mesh import for the Newton physics backend.

Bake visual-only shapes into the simulation stage in ``physics_backends.newton_backend.NewtonBackend``.
"""

import functools
import inspect
from dataclasses import dataclass

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade


@dataclass(frozen=True)
class VisualImportResult:
    """Summary of render-only visual shapes added to a Newton builder."""

    shape_indices: tuple
    body_indices: tuple
    mesh_sources: tuple


def add_usd_visual_shapes(newton, builder, stage, root_path, path_body_map, *, label_prefix=None):
    """Add USD meshes (under ``root_path`` on the current stage) as render-only shapes."""
    visual_shape_indices = []
    visual_body_indices = []
    visual_mesh_sources = []

    visual_cfg = newton.ModelBuilder.ShapeConfig(
        has_shape_collision=False,
        has_particle_collision=False,
        is_visible=True,
        is_solid=True,
        density=0.0,
    )

    root_prim = stage.GetPrimAtPath(root_path)
    for prim in Usd.PrimRange(root_prim):
        if prim.GetTypeName() != "Mesh":
            continue
        if _is_enabled_collider(prim) or not _is_effectively_visible(prim) or not _has_render_purpose(prim):
            continue

        body_path = _nearest_imported_body_path(prim, path_body_map)
        if body_path is None:
            continue

        body_idx = path_body_map[body_path]
        xform, scale = _mesh_xform_relative_to_body(prim, body_idx, body_path)
        # Bake scale into the mesh points
        meshes = _load_render_meshes(newton, prim, scale)
        if not meshes:
            continue

        for mesh_label, mesh in meshes:
            label = str(prim.GetPath())
            if mesh_label:
                label = f"{label}/{mesh_label}"
            if label_prefix:
                label = f"{label_prefix}/{label}"
            shape_idx = builder.add_shape_mesh(
                body_idx,
                xform=xform,
                mesh=mesh,
                cfg=visual_cfg,
                color=_viewer_color_for_mesh(mesh),
                label=label,
            )
            visual_shape_indices.append(shape_idx)
            visual_body_indices.append(body_idx)
            visual_mesh_sources.append(mesh)

    return VisualImportResult(tuple(visual_shape_indices), tuple(visual_body_indices), tuple(visual_mesh_sources))


def _load_render_meshes(newton, prim, scale):
    from newton._src.usd import utils as usd_utils

    try:
        mesh = UsdGeom.Mesh(prim)
        points = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
        indices = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
        counts = np.array(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
    except Exception:
        return None
    if len(points) == 0 or len(indices) == 0 or len(counts) == 0:
        return ()
    points = points * np.array([scale[0], scale[1], scale[2]], dtype=np.float64)

    flip_winding = False
    orientation_attr = mesh.GetOrientationAttr()
    if orientation_attr:
        orientation = orientation_attr.Get()
        flip_winding = bool(orientation and orientation.lower() == "lefthanded")

    uvs = _load_uvs(prim, points, indices, counts)
    material_subsets = _material_subsets(usd_utils, prim, len(counts))
    if material_subsets:
        meshes = []
        covered_faces = set()
        for subset in material_subsets:
            covered_faces.update(subset.face_indices)
            faces, corner_indices = _fan_triangulate_faces(counts, indices, flip_winding, subset.face_indices)
            render_mesh = _make_render_mesh(newton, points, indices, faces, corner_indices, uvs, subset.material_props)
            if render_mesh is not None:
                meshes.append((subset.name, render_mesh))

        # USD materialBind subsets are expected to partition mesh faces. Keep a parent-material
        # fallback for malformed assets so unassigned faces do not silently disappear from the viewer.
        remaining_faces = tuple(face_idx for face_idx in range(len(counts)) if face_idx not in covered_faces)
        if remaining_faces:
            faces, corner_indices = _fan_triangulate_faces(counts, indices, flip_winding, remaining_faces)
            render_mesh = _make_render_mesh(
                newton,
                points,
                indices,
                faces,
                corner_indices,
                uvs,
                _direct_material_properties(usd_utils, prim),
            )
            if render_mesh is not None:
                meshes.append(("unassigned_material", render_mesh))
        return tuple(meshes)

    faces, corner_indices = _fan_triangulate_faces(counts, indices, flip_winding)
    render_mesh = _make_render_mesh(
        newton,
        points,
        indices,
        faces,
        corner_indices,
        uvs,
        _with_mdl_material(usd_utils.resolve_material_properties_for_prim(prim), prim),
    )
    return (("", render_mesh),) if render_mesh is not None else ()


def _make_render_mesh(newton, points, indices, faces, corner_indices, uvs, material_props):
    if len(faces) == 0:
        return None

    points, faces, uvs = _compact_mesh(points, indices, faces, corner_indices, uvs)
    texture = material_props.get("texture") if uvs is not None else None
    color = None if texture is not None else material_props.get("color")
    extra = {}
    if _mesh_supports_opacity(newton):
        opacity = material_props.get("opacity")
        if opacity is not None:
            extra["opacity"] = opacity
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
        **extra,
    )


@functools.lru_cache(maxsize=1)
def _mesh_supports_opacity(newton):
    return "opacity" in inspect.signature(newton.Mesh.__init__).parameters


@dataclass(frozen=True)
class _MaterialSubset:
    name: str
    face_indices: tuple
    material_props: dict


def _material_subsets(usd_utils, prim, face_count):
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
                material_props=_with_mdl_material(usd_utils.resolve_material_properties_for_prim(child), child),
            )
        )
    return tuple(subsets)


# OmniGibson's dataset assets express glass with Omniverse's MDL ``OmniGlass`` shader rather than a
# USD Preview Surface ``opacity`` input, so newton renders the objects as opaque. Map the shader itself to an opacity.
_OMNI_GLASS_OPACITY = 0.25


def _bound_material_is_omni_glass(prim):
    """True when ``prim``'s bound material is an MDL OmniGlass shader."""
    try:
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not material:
            return False
        for child in Usd.PrimRange(material.GetPrim()):
            if not child.IsA(UsdShade.Shader):
                continue
            shader = UsdShade.Shader(child)
            sub = shader.GetSourceAssetSubIdentifier("mdl")
            if sub and "omniglass" in str(sub).lower():
                return True
            asset = shader.GetSourceAsset("mdl")
            if asset and "omniglass" in str(asset.path).lower():
                return True
    except Exception:
        return False
    return False


def _with_glass_opacity(props, prim):
    if props is not None and props.get("opacity") is None and _bound_material_is_omni_glass(prim):
        props = dict(props)
        props["opacity"] = _OMNI_GLASS_OPACITY
    return props


# OmniGibson's dataset assets mostly bind the MDL ``OmniGibsonVRayMtl``, which keeps a material's specular look
# in separate reflection / glossiness maps. newton's resolver reads only its ``diffuse_texture``.
# Bake the reflection layer into the albedo, and derive the scalar metallic/roughness from the maps.
_VRAY_MIN_REFLECTION = 0.05


def _bound_vray_texture_paths(prim):
    """{name: path} of the ``*_texture`` inputs of ``prim``'s bound OmniGibsonVRayMtl shader, or None."""
    try:
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not material:
            return None
        for child in Usd.PrimRange(material.GetPrim()):
            if not child.IsA(UsdShade.Shader):
                continue
            shader = UsdShade.Shader(child)
            if "omnigibsonvraymtl" not in str(shader.GetSourceAssetSubIdentifier("mdl") or "").lower():
                continue
            paths = {}
            for inp in shader.GetInputs():
                name, value = inp.GetBaseName(), inp.Get()
                if name.endswith("_texture") and value:
                    paths[name[: -len("_texture")]] = value.resolvedPath or value.path
            return paths
    except Exception:
        return None
    return None


def _load_texture_unit(path, size=None):
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGBA")
        if size is not None and image.size != size:
            image = image.resize(size, Image.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0


@functools.lru_cache(maxsize=512)
def _bake_vray_maps(diffuse_path, reflection_path, glossiness_path):
    """(albedo uint8 RGBA, metallic, roughness) for a VRay material, or None if it has no real reflection layer."""
    try:
        diffuse = _load_texture_unit(diffuse_path)
        size = (diffuse.shape[1], diffuse.shape[0])
        reflection = _load_texture_unit(reflection_path, size)[..., :3]
    except Exception:
        return None
    # Per-texel reflection strength, in the map's own (display) space like the albedo it is mixed into
    weight = reflection @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    if float(weight.mean()) < _VRAY_MIN_REFLECTION:
        return None
    albedo = diffuse.copy()
    albedo[..., :3] = diffuse[..., :3] * (1.0 - weight[..., None]) + reflection * weight[..., None]
    roughness = None
    if glossiness_path is not None:
        try:
            glossiness = _load_texture_unit(glossiness_path, size)[..., 0]
            # Glossiness only matters where the surface reflects
            roughness = float(((1.0 - glossiness) * weight).sum() / max(float(weight.sum()), 1e-6))
        except Exception:
            pass
    albedo = np.clip(np.round(albedo * 255.0), 0, 255).astype(np.uint8)
    return albedo, float(weight.mean()), roughness


def _with_vray_reflection(props, prim):
    if props is None or not isinstance(props.get("texture"), str) or props.get("metallic") is not None:
        return props
    paths = _bound_vray_texture_paths(prim)
    if not paths or "reflection" not in paths:
        return props
    baked = _bake_vray_maps(props["texture"], paths["reflection"], paths.get("glossiness"))
    if baked is None:
        return props
    props = dict(props)
    props["texture"], props["metallic"], roughness = baked
    if roughness is not None and props.get("roughness") is None:
        props["roughness"] = roughness
    return props


def _with_mdl_material(props, prim):
    return _with_vray_reflection(_with_glass_opacity(props, prim), prim)


def _direct_material_properties(usd_utils, prim):
    resolver = getattr(usd_utils, "_resolve_prim_material_properties", None)
    if resolver is not None:
        props = resolver(prim)
        if props is not None:
            return _with_mdl_material(props, prim)
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
    # ViewerGL multiplies mesh textures by the per-shape color buffer. Use a neutral color for
    # textured meshes so authored texture albedo is not tinted by USD displayColor or Newton's
    # fallback debug palette.
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


def _mesh_xform_relative_to_body(prim, body_idx, body_path):
    """Mesh-to-body local transform.

    Uses :func:`raw_world_transform` for BOTH the mesh and the body reference.
    """
    from pxr import Gf

    from omnigibson.physics_backends.newton_backend import raw_world_transform

    mesh_world = raw_world_transform(prim)
    if body_idx == -1:
        rel_mat = mesh_world
        body_world_scale = None
    else:
        body_prim = prim.GetStage().GetPrimAtPath(body_path)
        body_world = raw_world_transform(body_prim)
        rel_mat = mesh_world * body_world.GetInverse()
        body_world_scale = Gf.Transform(body_world).GetScale()

    gf_xf = Gf.Transform(rel_mat)
    t = gf_xf.GetTranslation()
    q = gf_xf.GetRotation().GetQuat()
    s = gf_xf.GetScale()
    if body_world_scale is not None:
        s = (s[0] * body_world_scale[0], s[1] * body_world_scale[1], s[2] * body_world_scale[2])
        # rel_mat's translation is in the body's unscaled frame too, so the mesh's offset from its
        # body needs the same scale
        t = (t[0] * body_world_scale[0], t[1] * body_world_scale[1], t[2] * body_world_scale[2])
    pos = wp.vec3(t[0], t[1], t[2])
    rot = wp.quat(*q.GetImaginary(), q.GetReal())
    scale = wp.vec3(s[0], s[1], s[2])
    return wp.transform(pos, rot), scale


def _is_enabled_collider(prim):
    collider = UsdPhysics.CollisionAPI(prim)
    if not collider:
        return False
    enabled = collider.GetCollisionEnabledAttr().Get()
    return enabled is not False


def _is_effectively_visible(prim):
    imageable = UsdGeom.Imageable(prim)
    return bool(imageable) and imageable.ComputeVisibility() != UsdGeom.Tokens.invisible


def _has_render_purpose(prim):
    # Some BEHAVIOR-1K assets carry a normally-hidden duplicate/proxy mesh with a huge world-space
    # extent, authored with purpose="proxy".
    purpose = UsdGeom.Imageable(prim).GetPurposeAttr().Get()
    return purpose in ("default", "render", "", None)
