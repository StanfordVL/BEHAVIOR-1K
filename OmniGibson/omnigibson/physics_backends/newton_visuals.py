"""USD visual mesh import for the Newton physics backend.

Ported from ``feat/newton``'s own ``omnigibson/newton/visuals.py`` (the reference Newton-native
rewrite), adapted to bake visual-only shapes into the SAME already-open simulation stage/builder our
``physics_backends.newton_backend.NewtonBackend`` already uses (feat/newton opens a fresh per-object
USD file for this pass; we don't have per-object files, so this walks ``stage`` directly under
``root_path`` instead).

Only used when ``not og.sim.render_backend.runs_inside_kit`` (Newton's own standalone viewers,
``ViewerGL``/``ViewerRTX`` -- see ``omnigibson/utils/newton_viewer_recording.py``): under Kit
rendering, OmniGibson's Newton physics model carries collision geometry only, and Kit renders the
original USD's separate visual mesh subtree directly, so baking visual shapes into the model there
would be pure waste (and doubles shape count for no benefit, since Kit never reads them).

Critically, this is also what fixes real per-texel texturing: Newton's own ``viewer.log_state()``
renders shapes baked into the model (via ``ModelBuilder.add_shape_mesh``) through a different internal
path than the public ``viewer.log_mesh()``/``log_instances()`` API -- confirmed empirically that the
latter renders textured meshes as a flat, untextured color (a minimal, OmniGibson-independent repro:
a bare quad with a real texture image renders uniform gray, not the image, via ``log_mesh()``), while
the former (this module's approach, matching feat/newton) renders correctly.
"""

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
    """Add visible-only USD meshes (under ``root_path`` on the already-open ``stage``) as render-only
    shapes on the bodies already imported into ``builder`` for this same object (via ``add_usd()``,
    whose returned ``path_body_map`` is passed in here).
    """
    visual_shape_indices = []
    visual_body_indices = []
    visual_mesh_sources = []

    visual_cfg = newton.ModelBuilder.ShapeConfig(
        has_shape_collision=False,
        has_particle_collision=False,
        is_visible=True,
        is_solid=True,
        # Render-only shapes must not contribute mass: the ShapeConfig default density (1000) would
        # silently add each visual mesh's volume in kilograms to its body, inflating every object and
        # robot in the scene (feat/newton's docs/other/newton_migration.md workaround W13).
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
        # Bake scale into the mesh's own points rather than passing it to add_shape_mesh()'s
        # separate `scale` argument -- confirmed empirically (a door rendering flat/horizontal
        # instead of upright) that add_shape_mesh composes `xform` (rotation) and `scale` in a
        # different order than expected for non-uniform scale + non-trivial rotation together (the
        # near-identity-rotation cases -- walls, floors, windows -- looked fine regardless, since
        # rotation order doesn't matter when the rotation is ~identity). Baking scale into the raw
        # points removes the ambiguity entirely, matching this same codebase's prior workaround for
        # anisotropic per-instance Cube scales corrupting unrelated meshes under ViewerRTX.
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
        newton, points, indices, faces, corner_indices, uvs, usd_utils.resolve_material_properties_for_prim(prim)
    )
    return (("", render_mesh),) if render_mesh is not None else ()


def _make_render_mesh(newton, points, indices, faces, corner_indices, uvs, material_props):
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
                material_props=usd_utils.resolve_material_properties_for_prim(child),
            )
        )
    return tuple(subsets)


def _direct_material_properties(usd_utils, prim):
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


def raw_world_transform(prim):
    """Compose a prim's world transform by directly composing each ancestor's own LOCAL transform
    (``UsdGeom.Xformable.GetLocalTransformation()``, which reads only that ONE prim's own
    xformOpOrder/values -- no ancestor traversal, no "world pose" resolution, no Fabric/live-simulation
    involvement of any kind) -- confirmed empirically that higher-level "world pose" APIs
    (``get_world_pose()``, and even ``UsdGeom.XformCache.GetLocalToWorldTransform()`` in some calling
    contexts) can return a DIFFERENT, WRONG transform for a body with live Newton-simulated state (a
    fixed-base but articulated object, e.g. a door with a hinge) than manually composing the SAME raw
    per-prim local transforms by hand. This is the one ground-truth source immune to whatever mechanism
    causes that divergence, since it never asks USD/Fabric to resolve "world" anything -- it only ever
    reads a single prim's own authored xformOps and does the multiplication itself.

    Returns a ``Gf.Matrix4d``.
    """
    from pxr import Gf, UsdGeom

    stage = prim.GetStage()
    pseudo_root_path = stage.GetPseudoRoot().GetPath()
    world = Gf.Matrix4d(1.0)
    p = prim
    while p and p.GetPath() != pseudo_root_path:
        xformable = UsdGeom.Xformable(p)
        if xformable:
            world = world * xformable.GetLocalTransformation()
        p = p.GetParent()
    return world


def _mesh_xform_relative_to_body(prim, body_idx, body_path):
    """Mesh-to-body local transform.

    Deliberately does NOT use ``builder.body_q[body_idx]`` (feat/newton's own approach) as the body's
    world reference, unlike the rest of this port: ``add_usd()``'s up-axis handling applies a spurious
    rotation to each floating-base object's synthetic free joint that only gets corrected, for the
    ROOT joint specifically, by ``physics_backends/newton_backend.py`` AFTER ``builder.finalize()`` (by
    overwriting ``model.joint_q``/``model.joint_X_p`` from raw USD ground truth) -- forward kinematics
    then propagates that correction to every DOWNSTREAM child body once the model actually starts
    simulating. At the point this function runs (mid-import, well before that finalize()+correction
    step), ``builder.body_q`` for any body reached through the affected chain is still stale/wrong.

    Also deliberately does the actual matrix algebra via ``pxr.Gf`` (``Gf.Matrix4d``/``Gf.Transform``),
    not Warp's ``wp.transform_compose``/``wp.inverse``/``wp.transform_decompose`` -- confirmed
    empirically, for a door object, that composing the SAME (position, quaternion) via
    ``wp.transform_compose`` produced a DIFFERENT rotation matrix (rows 1 and 2 permuted) depending on
    calling context, even with byte-for-byte identical inputs -- a genuine Warp-level
    non-determinism/context-dependence this function has no way to work around directly.

    Uses :func:`raw_world_transform` for BOTH the mesh and the body reference (not
    ``get_world_pose()``): the two sides of this division must come from the SAME ground-truth source
    for the result to be meaningful, and ``physics_backends/newton_backend.py``'s own joint-anchor
    correction (which this pairs with) is written using that same raw source -- so once that correction
    lands, the live simulated body pose used at render time matches this function's ``body_world``
    exactly, and the render composes back out to the true mesh position.
    """
    from pxr import Gf

    mesh_world = raw_world_transform(prim)
    if body_idx == -1:
        rel_mat = mesh_world
        body_world_scale = None
    else:
        body_prim = prim.GetStage().GetPrimAtPath(body_path)
        body_world = raw_world_transform(body_prim)
        rel_mat = mesh_world * body_world.GetInverse()
        # Newton's body_q is a pure rigid transform (position + orientation) -- it has no scale slot
        # at all, so any scale the body prim ITSELF carries (e.g. these dataset objects commonly
        # author their bounding-box-fit object scale as an xformOp:scale directly on the body/link
        # prim, not on the mesh) cancels out of `rel_mat` above (mesh_world and body_world both
        # inherit it identically as a shared ancestor) and would otherwise be silently dropped at
        # render time -- confirmed empirically: a straight_chair with base_link xformOp:scale
        # (1.32, 1.32, 1.27) rendered at its native unscaled mesh size, correctly centered on its
        # true position but ~25% short, leaving its legs visibly floating above the true floor
        # contact point. Reintroduce it here since this is the only place that scale can still be
        # captured and baked into the mesh's own points (see _load_render_meshes).
        body_world_scale = Gf.Transform(body_world).GetScale()

    gf_xf = Gf.Transform(rel_mat)
    t = gf_xf.GetTranslation()
    q = gf_xf.GetRotation().GetQuat()
    s = gf_xf.GetScale()
    if body_world_scale is not None:
        s = (s[0] * body_world_scale[0], s[1] * body_world_scale[1], s[2] * body_world_scale[2])
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
    # extent, authored with purpose="proxy" rather than being marked invisible -- confirmed
    # empirically (via omnigibson/utils/newton_viewer_recording.py's earlier, now-removed manual
    # registration path) that including one of these produced a giant dark artifact dominating every
    # camera angle, and that it's excluded from Kit's own "render" traversal purely by purpose, not
    # visibility. ``_is_effectively_visible`` alone does not catch this (a proxy-purposed mesh is not
    # necessarily "invisible" per USD's Imageable visibility opinion).
    purpose = UsdGeom.Imageable(prim).GetPurposeAttr().Get()
    return purpose in ("default", "render", "", None)
