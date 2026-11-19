"""Extract renderable geometry, materials, lights and cameras from the composed BEHAVIOR stage.

The output is renderer-agnostic: triangle soup (3 corners per triangle) with per-corner normals/UVs,
a material table, a de-duplicated texture table, and the authored UsdLux lights. Triangles that belong
to a Newton rigid body are stored in that body's frame so a renderer can pose them from ``state.body_q``;
everything else is baked into world space.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdShade

MAX_TEXTURE_SIZE = 1024


@dataclass
class Material:
    name: str
    base_color: tuple = (0.6, 0.6, 0.6)  # linear RGB multiplier
    roughness: float = 0.6
    metallic: float = 0.0
    specular: float = 1.0  # VRay reflection intensity I (luminance of the reflection colour)
    spec_tint: tuple = (1.0, 1.0, 1.0)  # reflection colour / I
    r0: float = 0.04  # coat reflectivity at normal incidence, ((1 - ior) / (1 + ior))^2
    transmission: float = 0.0  # 1 = clear dielectric (glass / water)
    ior: float = 1.5
    tex_diffuse: int = -1
    tex_gloss: int = -1  # VRay reflection glossiness (roughness = 1 - gloss)
    tex_metal: int = -1
    tex_normal: int = -1
    tex_reflection: int = -1
    tex_r0: int = -1  # R0 map derived from VRay's Fresnel IOR map (EXR)


@dataclass
class Lights:
    # dicts: kind (0 rect, 1 disk, 2 sphere), center, u, v (half extents / radius-scaled axes), normal, radius, radiance
    rects: list = field(default_factory=list)
    dome: tuple = (0.0, 0.0, 0.0)  # constant radiance


@dataclass
class RenderScene:
    # static triangles, world space
    s_pos: np.ndarray
    s_nrm: np.ndarray
    s_uv: np.ndarray
    s_mat: np.ndarray
    # dynamic triangles, body space
    d_pos: np.ndarray
    d_nrm: np.ndarray
    d_uv: np.ndarray
    d_mat: np.ndarray
    d_body: np.ndarray
    materials: list
    textures: list  # list of (H, W, 4) uint8 arrays
    texture_paths: list
    lights: Lights
    mesh_paths: list = field(default_factory=list)

    @property
    def triangle_count(self):
        return len(self.s_pos) + len(self.d_pos)


def _matrix(gf_matrix):
    # USD uses row vectors (p' = p M); return the column-vector form.
    return np.array(gf_matrix, dtype=np.float64).T


def _rigid(matrix):
    """Strip scale/shear from a 4x4 transform (Newton bodies are rigid)."""
    out = np.eye(4)
    u, _, vt = np.linalg.svd(matrix[:3, :3])
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    out[:3, :3] = r
    out[:3, 3] = matrix[:3, 3]
    return out


def _kelvin_to_rgb(kelvin):
    """Approximate blackbody colour (Tanner Helland fit), normalised to unit luminance, linear RGB."""
    t = kelvin / 100.0
    if t <= 66:
        r = 255.0
        g = 99.4708025861 * np.log(t) - 161.1195681661
        b = 0.0 if t <= 19 else 138.5177312231 * np.log(t - 10) - 305.0447927307
    else:
        r = 329.698727446 * (t - 60) ** -0.1332047592
        g = 288.1221695283 * (t - 60) ** -0.0755148492
        b = 255.0
    srgb = np.clip(np.array([r, g, b]) / 255.0, 0, 1)
    lin = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return lin / max(lum, 1e-6)


class _TextureTable:
    def __init__(self, max_size):
        self.max_size = max_size
        self.paths = []
        self.images = []
        self._index = {}

    def get_r0(self, asset_path):
        """Load a VRay reflection-IOR map (EXR) and store R0 = ((1 - ior) / (1 + ior))^2 as an 8-bit map."""
        if asset_path is None:
            return -1
        path = asset_path.resolvedPath or asset_path.path
        if not path or not Path(path).is_file():
            return -1
        key = path + "#r0"
        if key in self._index:
            return self._index[key]
        import OpenEXR

        with OpenEXR.File(path) as f:
            ior = np.asarray(next(iter(f.channels().values())).pixels, dtype=np.float32)
        if ior.ndim == 3:
            ior = ior[..., 0]
        r0 = np.clip(((1.0 - ior) / (1.0 + ior)) ** 2, 0, 1)
        image = Image.fromarray((r0 * 255 + 0.5).astype(np.uint8)).convert("RGBA")
        if max(image.size) > self.max_size:
            scale = self.max_size / max(image.size)
            image = image.resize(
                (max(1, round(image.size[0] * scale)), max(1, round(image.size[1] * scale))), Image.BILINEAR
            )
        self._index[key] = len(self.images)
        self.paths.append(key)
        self.images.append(np.ascontiguousarray(np.asarray(image, dtype=np.uint8)))
        return self._index[key]

    def get(self, asset_path):
        if asset_path is None:
            return -1
        path = asset_path.resolvedPath or asset_path.path
        if not path or not Path(path).is_file() or path.endswith(".exr"):
            return -1
        if path in self._index:
            return self._index[path]
        image = Image.open(path).convert("RGBA")
        if max(image.size) > self.max_size:
            scale = self.max_size / max(image.size)
            image = image.resize(
                (max(1, round(image.size[0] * scale)), max(1, round(image.size[1] * scale))), Image.LANCZOS
            )
        self._index[path] = len(self.images)
        self.paths.append(path)
        self.images.append(np.ascontiguousarray(np.asarray(image, dtype=np.uint8)))
        return self._index[path]


def _srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _input(shader, name):
    inp = shader.GetInput(name)
    if not inp:
        return None
    if inp.HasConnectedSource():
        source, _, _ = inp.GetConnectedSource()
        file_input = UsdShade.Shader(source.GetPrim()).GetInput("file")
        return file_input.Get() if file_input else None
    return inp.Get()


def _parse_material(material_prim, textures):
    mat = Material(name=str(material_prim.GetPath()) if material_prim else "default")
    if not material_prim:
        return mat
    for prim in Usd.PrimRange(material_prim):
        if not prim.IsA(UsdShade.Shader):
            continue
        shader = UsdShade.Shader(prim)
        sub = shader.GetSourceAssetSubIdentifier("mdl") or ""
        shader_id = shader.GetIdAttr().Get() or ""
        if sub == "OmniGibsonVRayMtl":
            mat.base_color = (1.0, 1.0, 1.0)
            tint = _input(shader, "diffuse_tint")
            if tint is not None:
                mat.base_color = tuple(float(x) for x in tint)
            mat.tex_diffuse = textures.get(_input(shader, "diffuse_texture"))
            mat.tex_gloss = textures.get(_input(shader, "glossiness_texture"))
            mat.tex_metal = textures.get(_input(shader, "metalness_texture"))
            mat.tex_normal = textures.get(_input(shader, "normal_texture"))
            mat.tex_reflection = textures.get(_input(shader, "reflection_texture"))
            mat.tex_r0 = textures.get_r0(_input(shader, "reflection_ior_texture"))
            mat.roughness, mat.metallic, mat.specular, mat.r0 = 0.5, 0.0, 1.0, ((1 - 1.6) / (1 + 1.6)) ** 2
            return mat
        if sub in ("OmniGlass", "OmniSurfaceGlass") or "glass" in sub.lower():
            color = _input(shader, "glass_color")
            mat.base_color = tuple(float(x) for x in color) if color is not None else (1.0, 1.0, 1.0)
            ior = _input(shader, "glass_ior")
            mat.ior = float(ior) if ior is not None else 1.5
            rough = _input(shader, "frosting_roughness")
            mat.roughness = float(rough) if rough is not None else 0.0
            mat.transmission, mat.metallic, mat.specular = 1.0, 0.0, 1.0
            return mat
        if sub == "OmniPBR":
            color = _input(shader, "diffuse_color_constant")
            if color is not None:
                mat.base_color = tuple(_srgb_to_linear(color))
            mat.tex_diffuse = textures.get(_input(shader, "diffuse_texture"))
            if mat.tex_diffuse >= 0:
                mat.base_color = (1.0, 1.0, 1.0)
            mat.roughness = float(_input(shader, "reflection_roughness_constant") or 0.5)
            mat.metallic = float(_input(shader, "metallic_constant") or 0.0)
            mat.tex_normal = textures.get(_input(shader, "normalmap_texture"))
            return mat
        if shader_id == "UsdPreviewSurface":
            color = _input(shader, "diffuseColor")
            if isinstance(color, Gf.Vec3f):
                mat.base_color = tuple(float(x) for x in color)
            elif color is not None:  # connected texture
                mat.tex_diffuse = textures.get(color)
                mat.base_color = (1.0, 1.0, 1.0)
            mat.roughness = float(_input(shader, "roughness") or 0.5)
            mat.metallic = float(_input(shader, "metallic") or 0.0)
            opacity = _input(shader, "opacity")
            if opacity is not None and float(opacity) < 0.5:
                mat.transmission = 1.0
                mat.ior = float(_input(shader, "ior") or 1.5)
            return mat
    return mat


def _corner_attribute(values, interpolation, vertex_indices, corner_count):
    """Expand a primvar to one value per face-vertex corner."""
    if values is None:
        return None
    values = np.asarray(values)
    if interpolation == UsdGeom.Tokens.faceVarying and len(values) == corner_count:
        return values
    if interpolation in (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying) and len(values) > vertex_indices.max():
        return values[vertex_indices]
    return None


def _triangulate(counts):
    """Fan-triangulate polygons; returns (T, 3) indices into the face-vertex corner array, and face id per tri."""
    counts = np.asarray(counts, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    ntri = np.maximum(counts - 2, 0)
    face = np.repeat(np.arange(len(counts)), ntri)
    k = np.arange(ntri.sum()) - np.repeat(np.cumsum(ntri) - ntri, ntri)
    s = starts[face]
    return np.stack([s, s + k + 1, s + k + 2], axis=1), face


def _is_renderable(prim, time):
    if not prim.IsA(UsdGeom.Mesh):
        return False
    path = str(prim.GetPath())
    if "/collisions" in path or "/meta__" in path:
        return False
    imageable = UsdGeom.Imageable(prim)
    if imageable.ComputeVisibility(time) == UsdGeom.Tokens.invisible:
        return False
    return imageable.ComputePurpose() in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render)


def extract_scene(stage, body_labels, extra_materials=None, max_texture_size=MAX_TEXTURE_SIZE):
    """Collect every renderable mesh under /World.

    Args:
        stage: composed Usd.Stage
        body_labels: Newton ``model.body_label`` (prim paths of simulated bodies)
    """
    time = Usd.TimeCode.Default()
    cache = UsdGeom.XformCache(time)
    body_index = {label: i for i, label in enumerate(body_labels)}
    body_inv = {}
    for label, i in body_index.items():
        prim = stage.GetPrimAtPath(label)
        if prim:
            body_inv[i] = np.linalg.inv(_rigid(_matrix(cache.GetLocalToWorldTransform(prim))))

    textures = _TextureTable(max_size=max_texture_size)
    materials, material_index = [], {}

    def material_id(material_prim):
        key = str(material_prim.GetPath()) if material_prim else ""
        if key not in material_index:
            material_index[key] = len(materials)
            materials.append(_parse_material(material_prim, textures))
        return material_index[key]

    static, dynamic, mesh_paths = [], [], []
    predicate = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), predicate):
        if not _is_renderable(prim, time):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get(time)
        counts = mesh.GetFaceVertexCountsAttr().Get(time)
        indices = mesh.GetFaceVertexIndicesAttr().Get(time)
        if not points or not counts or not indices:
            continue
        points = np.asarray(points, dtype=np.float64)
        indices = np.asarray(indices, dtype=np.int64)
        tri_corners, tri_face = _triangulate(counts)
        if len(tri_corners) == 0:
            continue
        tri_vertex = indices[tri_corners]

        # per-corner normals (authored or flat), uvs
        normals = _corner_attribute(
            mesh.GetNormalsAttr().Get(time), mesh.GetNormalsInterpolation(), indices, len(indices)
        )
        primvars = UsdGeom.PrimvarsAPI(prim)
        st = primvars.GetPrimvar("st") or primvars.GetPrimvar("st0") or primvars.GetPrimvar("uv")
        uvs = None
        if st and st.HasValue():
            uvs = _corner_attribute(st.ComputeFlattened(time), st.GetInterpolation(), indices, len(indices))

        world = _matrix(cache.GetLocalToWorldTransform(prim))
        # find owning body
        body, ancestor = -1, prim
        while ancestor and ancestor.GetPath() != ancestor.GetPath().absoluteRootPath:
            if str(ancestor.GetPath()) in body_index:
                body = body_index[str(ancestor.GetPath())]
                break
            ancestor = ancestor.GetParent()
        frame = body_inv[body] @ world if body >= 0 else world

        p = points @ frame[:3, :3].T + frame[:3, 3]
        pos = p[tri_vertex]  # (T,3,3)
        if normals is not None:
            nmat = np.linalg.inv(frame[:3, :3]).T
            n = np.asarray(normals, dtype=np.float64)[tri_corners] @ nmat.T
        else:
            fn = np.cross(pos[:, 1] - pos[:, 0], pos[:, 2] - pos[:, 0])
            n = np.repeat(fn[:, None, :], 3, axis=1)
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
        if np.linalg.det(frame[:3, :3]) < 0:
            pos = pos[:, ::-1]
            n = n[:, ::-1]
        uv = np.asarray(uvs, dtype=np.float64)[tri_corners] if uvs is not None else np.zeros((len(tri_corners), 3, 2))
        if len(uv) != len(pos):
            uv = np.zeros((len(pos), 3, 2))
        if uvs is not None and np.asarray(uvs).ndim == 2 and len(uvs) and len(uv) and uv.shape[-1] != 2:
            uv = np.zeros((len(pos), 3, 2))

        # material per triangle (with GeomSubset support)
        binding = UsdShade.MaterialBindingAPI(prim)
        mats = np.full(len(tri_corners), material_id(binding.ComputeBoundMaterial()[0].GetPrim()), dtype=np.int32)
        for subset in UsdShade.MaterialBindingAPI(prim).GetMaterialBindSubsets():
            faces = np.asarray(subset.GetIndicesAttr().Get(time) or [], dtype=np.int64)
            sub_mat = UsdShade.MaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial()[0]
            if len(faces) and sub_mat:
                mats[np.isin(tri_face, faces)] = material_id(sub_mat.GetPrim())

        entry = (pos.astype(np.float32), n.astype(np.float32), uv.astype(np.float32), mats)
        if body >= 0:
            dynamic.append(entry + (np.full(len(pos), body, dtype=np.int32),))
        else:
            static.append(entry)
        mesh_paths.append((str(prim.GetPath()), body, len(pos)))

    def cat(items, i, shape, dtype):
        return np.concatenate([x[i] for x in items]) if items else np.zeros(shape, dtype=dtype)

    lights = Lights()
    for prim in stage.Traverse():
        if prim.IsA(UsdLux.RectLight) or prim.IsA(UsdLux.DiskLight) or prim.IsA(UsdLux.SphereLight):
            if UsdGeom.Imageable(prim).ComputeVisibility(time) == UsdGeom.Tokens.invisible:
                continue
            light = UsdLux.LightAPI(prim)
            m = _matrix(cache.GetLocalToWorldTransform(prim))
            color = np.array(light.GetColorAttr().Get() or (1, 1, 1), dtype=np.float64)
            if light.GetEnableColorTemperatureAttr().Get():
                color = color * _kelvin_to_rgb(light.GetColorTemperatureAttr().Get())
            radiance = light.GetIntensityAttr().Get() * 2.0 ** (light.GetExposureAttr().Get() or 0.0) * color
            # USD lights emit along local -Z; a scaled light's extent scales with it
            sx, sy, sz = (np.linalg.norm(m[:3, k]) for k in range(3))
            entry = dict(center=m[:3, 3], normal=-m[:3, 2] / sz, radius=0.0)
            if prim.IsA(UsdLux.RectLight):
                rect = UsdLux.RectLight(prim)
                w, h = rect.GetWidthAttr().Get() * 0.5, rect.GetHeightAttr().Get() * 0.5
                entry.update(kind=0, u=m[:3, 0] * w, v=m[:3, 1] * h)
                surface = 4 * w * h * sx * sy
            elif prim.IsA(UsdLux.DiskLight):
                r = UsdLux.DiskLight(prim).GetRadiusAttr().Get()
                entry.update(kind=1, u=m[:3, 0] * r, v=m[:3, 1] * r, radius=r * max(sx, sy))
                surface = np.pi * entry["radius"] ** 2
            else:
                r = UsdLux.SphereLight(prim).GetRadiusAttr().Get() * max(sx, sy, sz)
                entry.update(kind=2, u=np.array([r, 0, 0]), v=np.array([0, r, 0]), radius=r)
                surface = 4 * np.pi * r**2
            if light.GetNormalizeAttr().Get():
                # "normalize": intensity is power-like, divided by the emitting surface area
                radiance = radiance / max(surface, 1e-12)
            if not np.any(radiance > 0):
                continue
            entry["radiance"] = radiance
            lights.rects.append(entry)
        elif prim.IsA(UsdLux.DomeLight):
            light = UsdLux.DomeLight(prim)
            color = np.array(light.GetColorAttr().Get() or (1, 1, 1), dtype=np.float64)
            lights.dome = tuple(light.GetIntensityAttr().Get() * 2.0 ** (light.GetExposureAttr().Get() or 0.0) * color)

    return RenderScene(
        s_pos=cat(static, 0, (0, 3, 3), np.float32),
        s_nrm=cat(static, 1, (0, 3, 3), np.float32),
        s_uv=cat(static, 2, (0, 3, 2), np.float32),
        s_mat=cat(static, 3, (0,), np.int32),
        d_pos=cat(dynamic, 0, (0, 3, 3), np.float32),
        d_nrm=cat(dynamic, 1, (0, 3, 3), np.float32),
        d_uv=cat(dynamic, 2, (0, 3, 2), np.float32),
        d_mat=cat(dynamic, 3, (0,), np.int32),
        d_body=cat(dynamic, 4, (0,), np.int32),
        materials=materials,
        textures=textures.images,
        texture_paths=textures.paths,
        lights=lights,
        mesh_paths=mesh_paths,
    )
