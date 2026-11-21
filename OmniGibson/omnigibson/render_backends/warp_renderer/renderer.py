"""A compute-only ray tracer in NVIDIA Warp (no RT cores, no OpenGL/Vulkan, no Kit).

Everything here is a plain CUDA kernel over Warp's software BVH (``wp.Mesh``), so it runs on any CUDA GPU,
including ones without RT cores or graphics TPCs (A100/H100).

Geometry: three BVHs -- static triangles (built once), dynamic rigid-body triangles (re-posed from
``body_q`` and refit every frame), and the water surface (rebuilt per frame). Lights: UsdLux rect area
lights plus a constant dome. Materials: BEHAVIOR's baked VRay maps (diffuse, glossiness, metalness, normal,
reflection), glass/water dielectrics, UsdPreviewSurface metals.

Quality tiers (``preset``), all sharing one megakernel:

``direct``  rl-tools-style: GGX + Lambert direct lighting from each light's centre with a hard shadow ray,
            constant ambient, one specular continuation (glass refraction / metal mirror). Deterministic.
``probe``   + soft area-light shadows, glossy GGX-sampled reflections and Fresnel-split glass, and diffuse
            global illumination from an irradiance-probe grid baked once per (static) scene by path tracing
            (ambient cubes, DDGI-like invalid-probe rejection). A few spp + optional a-trous denoise.
``pt``      unidirectional path tracer with next-event estimation, Russian roulette, dielectrics; N spp +
            edge-aware a-trous denoiser guided by albedo/normal/depth taken through specular chains.
"""

import math
import time

import numpy as np
import warp as wp

PI = wp.constant(math.pi)
GEOM_STATIC = wp.constant(0)
GEOM_DYNAMIC = wp.constant(1)
GEOM_WATER = wp.constant(2)
MAX_LIGHTS = 64
LIGHT_RECT = wp.constant(0)
LIGHT_DISK = wp.constant(1)
LIGHT_SPHERE = wp.constant(2)


# ----------------------------------------------------------------------------------------------------------
# GPU data
# ----------------------------------------------------------------------------------------------------------


@wp.struct
class Geometry:
    mesh: wp.array(dtype=wp.uint64)  # [static, dynamic, water], 0 = absent
    # per-corner attributes, one array per BVH (corner index = 3 * face + k)
    s_nrm: wp.array(dtype=wp.vec3)
    s_uv: wp.array(dtype=wp.vec2)
    s_mat: wp.array(dtype=wp.int32)
    s_pts: wp.array(dtype=wp.vec3)
    d_nrm: wp.array(dtype=wp.vec3)
    d_uv: wp.array(dtype=wp.vec2)
    d_mat: wp.array(dtype=wp.int32)
    d_pts: wp.array(dtype=wp.vec3)
    w_pts: wp.array(dtype=wp.vec3)
    w_idx: wp.array(dtype=wp.int32)
    w_nrm: wp.array(dtype=wp.vec3)
    water_mat: wp.int32


@wp.struct
class Materials:
    base: wp.array(dtype=wp.vec3)
    rough: wp.array(dtype=wp.float32)
    metal: wp.array(dtype=wp.float32)
    spec: wp.array(dtype=wp.float32)  # VRay reflection intensity I (luminance of the reflection colour)
    tint: wp.array(dtype=wp.vec3)  # reflection colour / I
    r0: wp.array(dtype=wp.float32)  # normal-incidence reflectivity of the coat, ((1 - ior) / (1 + ior))^2
    trans: wp.array(dtype=wp.float32)
    ior: wp.array(dtype=wp.float32)
    tex_diffuse: wp.array(dtype=wp.int32)
    tex_gloss: wp.array(dtype=wp.int32)
    tex_metal: wp.array(dtype=wp.int32)
    tex_normal: wp.array(dtype=wp.int32)
    tex_refl: wp.array(dtype=wp.int32)
    tex_r0: wp.array(dtype=wp.int32)
    # texture pool: packed RGBA8 texels
    texels: wp.array(dtype=wp.uint32)
    tex_offset: wp.array(dtype=wp.int32)
    tex_w: wp.array(dtype=wp.int32)
    tex_h: wp.array(dtype=wp.int32)
    srgb_lut: wp.array(dtype=wp.float32)


@wp.struct
class LightSet:
    count: wp.int32
    kind: wp.array(dtype=wp.int32)  # LIGHT_RECT / LIGHT_DISK / LIGHT_SPHERE
    center: wp.array(dtype=wp.vec3)
    u: wp.array(dtype=wp.vec3)  # half extent vectors (rect), radius-scaled axes (disk)
    v: wp.array(dtype=wp.vec3)
    normal: wp.array(dtype=wp.vec3)  # emission direction (rect, disk; one-sided)
    radius: wp.array(dtype=wp.float32)  # disk, sphere
    radiance: wp.array(dtype=wp.vec3)
    area: wp.array(dtype=wp.float32)  # sampled area: rect, disk, and a sphere's facing disk (pi r^2)
    dome: wp.vec3
    ambient: wp.vec3  # constant ambient radiance for the 'direct' preset


@wp.struct
class ProbeGrid:
    enabled: wp.int32
    origin: wp.vec3
    cell: wp.float32
    nx: wp.int32
    ny: wp.int32
    nz: wp.int32
    cube: wp.array(dtype=wp.vec3)  # 6 per probe (+x,-x,+y,-y,+z,-z), outgoing radiance of a white lambertian
    valid: wp.array(dtype=wp.float32)


@wp.struct
class Settings:
    spp: wp.int32
    max_depth: wp.int32
    soft_shadows: wp.int32  # 0: light centre, 1: stochastic point on the light, n>=2: n x n fixed grid
    indirect: wp.int32  # 0: constant ambient, 1: probe grid, 2: path traced
    glossy_sampling: wp.int32  # 0: mirror reflections, 1: GGX-sampled
    seed: wp.int32
    exposure: wp.float32
    lights_visible: wp.int32
    clamp: wp.float32  # per-sample luminance clamp (fireflies)


@wp.struct
class Hit:
    t: wp.float32
    geom: wp.int32
    face: wp.int32
    u: wp.float32
    v: wp.float32


@wp.struct
class Surface:
    p: wp.vec3
    ng: wp.vec3  # geometric normal, facing the incoming ray
    n: wp.vec3  # shading normal, facing the incoming ray
    front: wp.int32  # 1 if the ray hit the outward side
    albedo: wp.vec3
    rough: wp.float32
    metal: wp.float32
    spec: wp.float32
    tint: wp.vec3
    r0: wp.float32
    trans: wp.float32
    ior: wp.float32


# ----------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------


@wp.func
def unpack_rgba(c: wp.uint32, mat: Materials, srgb: wp.int32) -> wp.vec4:
    r = wp.int32(c & wp.uint32(255))
    g = wp.int32((c >> wp.uint32(8)) & wp.uint32(255))
    b = wp.int32((c >> wp.uint32(16)) & wp.uint32(255))
    a = wp.int32((c >> wp.uint32(24)) & wp.uint32(255))
    if srgb != 0:
        return wp.vec4(mat.srgb_lut[r], mat.srgb_lut[g], mat.srgb_lut[b], wp.float32(a) / 255.0)
    return wp.vec4(wp.float32(r) / 255.0, wp.float32(g) / 255.0, wp.float32(b) / 255.0, wp.float32(a) / 255.0)


@wp.func
def sample_tex(mat: Materials, tex: wp.int32, uv: wp.vec2, srgb: wp.int32) -> wp.vec4:
    """Bilinear, repeat-wrapped lookup in the packed texture pool (V flipped: USD st origin is bottom-left)."""
    w = mat.tex_w[tex]
    h = mat.tex_h[tex]
    off = mat.tex_offset[tex]
    x = (uv[0] - wp.floor(uv[0])) * wp.float32(w) - 0.5
    y = (1.0 - (uv[1] - wp.floor(uv[1]))) * wp.float32(h) - 0.5
    x0f = wp.floor(x)
    y0f = wp.floor(y)
    fx = x - x0f
    fy = y - y0f
    x0 = (wp.int32(x0f) % w + w) % w
    y0 = (wp.int32(y0f) % h + h) % h
    x1 = (x0 + 1) % w
    y1 = (y0 + 1) % h
    c00 = unpack_rgba(mat.texels[off + y0 * w + x0], mat, srgb)
    c10 = unpack_rgba(mat.texels[off + y0 * w + x1], mat, srgb)
    c01 = unpack_rgba(mat.texels[off + y1 * w + x0], mat, srgb)
    c11 = unpack_rgba(mat.texels[off + y1 * w + x1], mat, srgb)
    return (c00 * (1.0 - fx) + c10 * fx) * (1.0 - fy) + (c01 * (1.0 - fx) + c11 * fx) * fy


@wp.func
def reflect(d: wp.vec3, n: wp.vec3) -> wp.vec3:
    return d - n * (2.0 * wp.dot(d, n))


@wp.func
def luminance(c: wp.vec3) -> wp.float32:
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


@wp.func
def onb_u(n: wp.vec3) -> wp.vec3:
    s = wp.where(n[2] >= 0.0, 1.0, -1.0)
    a = -1.0 / (s + n[2])
    b = n[0] * n[1] * a
    return wp.vec3(1.0 + s * n[0] * n[0] * a, s * b, -s * n[0])


@wp.func
def onb_v(n: wp.vec3) -> wp.vec3:
    s = wp.where(n[2] >= 0.0, 1.0, -1.0)
    a = -1.0 / (s + n[2])
    b = n[0] * n[1] * a
    return wp.vec3(b, s + n[1] * n[1] * a, -n[1])


@wp.func
def cosine_dir(n: wp.vec3, r1: wp.float32, r2: wp.float32) -> wp.vec3:
    phi = 2.0 * PI * r1
    sr = wp.sqrt(r2)
    return wp.normalize(onb_u(n) * (wp.cos(phi) * sr) + onb_v(n) * (wp.sin(phi) * sr) + n * wp.sqrt(1.0 - r2))


@wp.func
def ggx_sample_h(n: wp.vec3, alpha: wp.float32, r1: wp.float32, r2: wp.float32) -> wp.vec3:
    phi = 2.0 * PI * r1
    cos_t = wp.sqrt((1.0 - r2) / (1.0 + (alpha * alpha - 1.0) * r2))
    sin_t = wp.sqrt(wp.max(0.0, 1.0 - cos_t * cos_t))
    return wp.normalize(onb_u(n) * (wp.cos(phi) * sin_t) + onb_v(n) * (wp.sin(phi) * sin_t) + n * cos_t)


@wp.func
def ggx_d(ndh: wp.float32, alpha: wp.float32) -> wp.float32:
    a2 = alpha * alpha
    d = ndh * ndh * (a2 - 1.0) + 1.0
    return a2 / (PI * d * d + 1.0e-7)


@wp.func
def smith_g1(ndx: wp.float32, alpha: wp.float32) -> wp.float32:
    a2 = alpha * alpha
    return 2.0 * ndx / (ndx + wp.sqrt(a2 + (1.0 - a2) * ndx * ndx) + 1.0e-7)


@wp.func
def fresnel_schlick(f0: wp.vec3, cos_t: wp.float32) -> wp.vec3:
    m = wp.pow(wp.clamp(1.0 - cos_t, 0.0, 1.0), 5.0)
    return f0 + (wp.vec3(1.0) - f0) * m


@wp.func
def fresnel_dielectric(cos_i: wp.float32, eta: wp.float32) -> wp.float32:
    """Unpolarised Fresnel reflectance; eta = n_t / n_i."""
    c = wp.abs(cos_i)
    g2 = eta * eta - 1.0 + c * c
    if g2 < 0.0:
        return 1.0
    g = wp.sqrt(g2)
    a = (g - c) / (g + c)
    b = (c * (g + c) - 1.0) / (c * (g - c) + 1.0)
    return 0.5 * a * a * (1.0 + b * b)


@wp.func
def coat_weight(s: Surface, cos_t: wp.float32) -> wp.float32:
    """VRayMtl's reflection layer weight: color_custom_curve_layer(R0 * I, I) (see vray_materials.mdl)."""
    return s.spec * (s.r0 + (1.0 - s.r0) * wp.pow(wp.clamp(1.0 - cos_t, 0.0, 1.0), 5.0))


@wp.func
def glossy_reflectance(s: Surface, cos_t: wp.float32) -> wp.vec3:
    """Albedo of the two GGX lobes: the tinted reflection coat plus the metallic base (tinted by diffuse)."""
    w = coat_weight(s, cos_t)
    return s.tint * w + s.albedo * ((1.0 - w) * s.metal)


@wp.func
def diffuse_reflectance(s: Surface, cos_t: wp.float32) -> wp.vec3:
    w = coat_weight(s, cos_t)
    return s.albedo * ((1.0 - w) * (1.0 - s.metal) * (1.0 - s.trans))


@wp.func
def eval_brdf_cos(s: Surface, wo: wp.vec3, wi: wp.vec3) -> wp.vec3:
    """VRayMtl-style BRDF times cos(theta_i): coat(GGX, tint) over mix(diffuse, metallic GGX)."""
    ndl = wp.dot(s.n, wi)
    ndv = wp.dot(s.n, wo)
    if ndl <= 0.0 or ndv <= 0.0 or wp.dot(s.ng, wi) <= 0.0:
        return wp.vec3(0.0)
    h = wp.normalize(wi + wo)
    ndh = wp.max(wp.dot(s.n, h), 0.0)
    alpha = wp.max(s.rough * s.rough, 0.002)
    dg = ggx_d(ndh, alpha) * smith_g1(ndl, alpha) * smith_g1(ndv, alpha) / (4.0 * ndl * ndv + 1.0e-6)
    return (glossy_reflectance(s, ndv) * dg + diffuse_reflectance(s, ndv) / PI) * ndl


# ----------------------------------------------------------------------------------------------------------
# tracing
# ----------------------------------------------------------------------------------------------------------


@wp.func
def trace(geo: Geometry, o: wp.vec3, d: wp.vec3, tmax: wp.float32) -> Hit:
    hit = Hit()
    hit.t = tmax
    hit.geom = -1
    for g in range(3):
        mid = geo.mesh[g]
        if mid != wp.uint64(0):
            q = wp.mesh_query_ray(mid, o, d, hit.t)
            if q.result and q.t < hit.t:
                hit.t = q.t
                hit.geom = g
                hit.face = q.face
                hit.u = q.u
                hit.v = q.v
    return hit


@wp.func
def corner_pts(geo: Geometry, geom: wp.int32, face: wp.int32, k: wp.int32) -> wp.vec3:
    if geom == GEOM_STATIC:
        return geo.s_pts[3 * face + k]
    if geom == GEOM_DYNAMIC:
        return geo.d_pts[3 * face + k]
    return geo.w_pts[geo.w_idx[3 * face + k]]


@wp.func
def material_of(geo: Geometry, hit: Hit) -> wp.int32:
    if hit.geom == GEOM_STATIC:
        return geo.s_mat[hit.face]
    if hit.geom == GEOM_DYNAMIC:
        return geo.d_mat[hit.face]
    return geo.water_mat


@wp.func
def transmittance_of(geo: Geometry, mats: Materials, hit: Hit) -> wp.float32:
    return mats.trans[material_of(geo, hit)]


@wp.func
def shade_surface(geo: Geometry, mats: Materials, hit: Hit, o: wp.vec3, d: wp.vec3) -> Surface:
    s = Surface()
    w0 = hit.u
    w1 = hit.v
    w2 = 1.0 - hit.u - hit.v
    p0 = corner_pts(geo, hit.geom, hit.face, 0)
    p1 = corner_pts(geo, hit.geom, hit.face, 1)
    p2 = corner_pts(geo, hit.geom, hit.face, 2)
    s.p = p0 * w0 + p1 * w1 + p2 * w2
    ng = wp.cross(p1 - p0, p2 - p0)
    ngl = wp.length(ng)
    if ngl > 0.0:
        ng = ng / ngl
    else:
        ng = -d
    uv = wp.vec2(0.0)
    n = ng
    m = wp.int32(0)
    if hit.geom == GEOM_STATIC:
        c = 3 * hit.face
        n = geo.s_nrm[c] * w0 + geo.s_nrm[c + 1] * w1 + geo.s_nrm[c + 2] * w2
        uv = geo.s_uv[c] * w0 + geo.s_uv[c + 1] * w1 + geo.s_uv[c + 2] * w2
        m = geo.s_mat[hit.face]
    elif hit.geom == GEOM_DYNAMIC:
        c = 3 * hit.face
        n = geo.d_nrm[c] * w0 + geo.d_nrm[c + 1] * w1 + geo.d_nrm[c + 2] * w2
        uv = geo.d_uv[c] * w0 + geo.d_uv[c + 1] * w1 + geo.d_uv[c + 2] * w2
        m = geo.d_mat[hit.face]
    else:
        i0 = geo.w_idx[3 * hit.face]
        i1 = geo.w_idx[3 * hit.face + 1]
        i2 = geo.w_idx[3 * hit.face + 2]
        n = geo.w_nrm[i0] * w0 + geo.w_nrm[i1] * w1 + geo.w_nrm[i2] * w2
        m = geo.water_mat
    nl = wp.length(n)
    if nl > 1.0e-6:
        n = n / nl
    else:
        n = ng
    # keep the interpolated normal on the same side as the geometric one
    if wp.dot(n, ng) < 0.0:
        n = -n

    s.front = wp.where(wp.dot(ng, d) < 0.0, 1, 0)
    albedo = mats.base[m]
    rough = mats.rough[m]
    metal = mats.metal[m]
    spec = mats.spec[m]
    td = mats.tex_diffuse[m]
    if td >= 0:
        tc = sample_tex(mats, td, uv, 1)
        albedo = wp.cw_mul(albedo, wp.vec3(tc[0], tc[1], tc[2]))
    tg = mats.tex_gloss[m]
    if tg >= 0:
        rough = 1.0 - sample_tex(mats, tg, uv, 1)[0]
    tm = mats.tex_metal[m]
    if tm >= 0:
        metal = sample_tex(mats, tm, uv, 1)[0]
    tint = mats.tint[m]
    tr = mats.tex_refl[m]
    if tr >= 0:
        rc = sample_tex(mats, tr, uv, 1)
        rcol = wp.vec3(rc[0], rc[1], rc[2])
        spec = luminance(rcol)
        tint = rcol / wp.max(spec, 1.0e-4)
    r0 = mats.r0[m]
    tr0 = mats.tex_r0[m]
    if tr0 >= 0:
        r0 = sample_tex(mats, tr0, uv, 0)[0]
    # BEHAVIOR's baked VRay maps are 8-bit PNGs that the MDL loads with texture_2d()'s default gamma, i.e.
    # sRGB-decoded -- including glossiness/metalness/normal maps (a flat normal is stored as 186,186,255).
    tn = mats.tex_normal[m]
    if tn >= 0 and hit.geom != GEOM_WATER:
        # per-triangle tangent frame from UV derivatives
        c = 3 * hit.face
        uv0 = wp.vec2(0.0)
        uv1 = wp.vec2(0.0)
        uv2 = wp.vec2(0.0)
        if hit.geom == GEOM_STATIC:
            uv0 = geo.s_uv[c]
            uv1 = geo.s_uv[c + 1]
            uv2 = geo.s_uv[c + 2]
        else:
            uv0 = geo.d_uv[c]
            uv1 = geo.d_uv[c + 1]
            uv2 = geo.d_uv[c + 2]
        e1 = p1 - p0
        e2 = p2 - p0
        du1 = uv1[0] - uv0[0]
        dv1 = uv1[1] - uv0[1]
        du2 = uv2[0] - uv0[0]
        dv2 = uv2[1] - uv0[1]
        det = du1 * dv2 - du2 * dv1
        if wp.abs(det) > 1.0e-12:
            tangent = (e1 * dv2 - e2 * dv1) / det
            tangent = tangent - n * wp.dot(n, tangent)
            tl = wp.length(tangent)
            if tl > 1.0e-8:
                tangent = tangent / tl
                bitangent = wp.cross(n, tangent) * wp.where(det < 0.0, -1.0, 1.0)
                tc = sample_tex(mats, tn, uv, 1)
                tsn = wp.vec3(tc[0] * 2.0 - 1.0, tc[1] * 2.0 - 1.0, tc[2] * 2.0 - 1.0)
                pn = wp.normalize(tangent * tsn[0] + bitangent * tsn[1] + n * wp.max(tsn[2], 0.05))
                if wp.dot(pn, ng) > 0.05:
                    n = pn

    # face the incoming ray
    if s.front == 0:
        ng = -ng
        n = -n
    view_cos = wp.dot(n, -d)
    if view_cos < 0.01:
        # Project along the geometric normal far enough to face the viewer. Without the
        # denominator, smoothed normals can remain back-facing at grazing angles and shade black.
        n = wp.normalize(n + ng * ((0.01 - view_cos) / wp.max(wp.dot(ng, -d), 1.0e-6)))
    s.ng = ng
    s.n = n
    s.albedo = albedo
    s.rough = wp.clamp(rough, 0.02, 1.0)
    s.metal = wp.clamp(metal, 0.0, 1.0)
    s.spec = wp.clamp(spec, 0.0, 1.0)
    s.tint = tint
    s.r0 = wp.clamp(r0, 0.0, 1.0)
    s.trans = mats.trans[m]
    s.ior = mats.ior[m]
    return s


@wp.func
def shadow_transmittance(geo: Geometry, mats: Materials, o: wp.vec3, d: wp.vec3, dist: wp.float32) -> wp.vec3:
    """Visibility along a shadow ray; clear dielectrics attenuate (tinted) instead of blocking."""
    tr = wp.vec3(1.0)
    start = o
    remaining = dist
    for _ in range(6):
        hit = trace(geo, start, d, remaining)
        if hit.geom < 0:
            return tr
        m = material_of(geo, hit)
        if mats.trans[m] <= 0.0:
            return wp.vec3(0.0)
        tr = wp.cw_mul(tr, mats.base[m]) * 0.92
        start = start + d * (hit.t + 1.0e-3)
        remaining = remaining - hit.t - 1.0e-3
        if remaining <= 0.0:
            return tr
    return tr


@wp.func
def direct_light(
    geo: Geometry,
    mats: Materials,
    lights: LightSet,
    s: Surface,
    wo: wp.vec3,
    soft: wp.int32,
    r1: wp.float32,
    r2: wp.float32,
) -> wp.vec3:
    """Area-light direct lighting. soft: 0 = light centre (hard shadow), 1 = one random point (noisy, unbiased),
    n >= 2 = fixed n x n stratified grid of points (deterministic soft shadows, n*n shadow rays per light), each
    point at offset (r1, r2) inside its cell (0.5 = cell centre; shifted per AA sample)."""
    out = wp.vec3(0.0)
    o = s.p + s.ng * 2.0e-4
    grid_n = wp.max(soft, 1)
    inv_n = 1.0 / wp.float32(grid_n)
    for li in range(lights.count):
        kind = lights.kind[li]
        c = lights.center[li]
        nrm = lights.normal[li]
        uu = lights.u[li]
        vv = lights.v[li]
        if kind == LIGHT_SPHERE:
            # a small sphere seen from o: sample the disk of the same radius facing o (same solid angle)
            nrm = wp.normalize(o - c)
            uu = onb_u(nrm) * lights.radius[li]
            vv = onb_v(nrm) * lights.radius[li]
        for k in range(grid_n * grid_n):
            # sample offset in [-1, 1]^2: light centre, one random point, or a stratified grid cell
            sa = float(0.0)
            sb = float(0.0)
            if soft == 1:
                sa = 2.0 * r1 - 1.0
                sb = 2.0 * r2 - 1.0
            elif soft >= 2:
                sa = 2.0 * (wp.float32(k % grid_n) + r1) * inv_n - 1.0
                sb = 2.0 * (wp.float32(k // grid_n) + r2) * inv_n - 1.0
            if kind != LIGHT_RECT:
                # Shirley-Chiu concentric square -> disk mapping (keeps the strata)
                rr = float(0.0)
                phi = float(0.0)
                if sa * sa > sb * sb:
                    rr = sa
                    phi = (PI / 4.0) * (sb / sa)
                elif sb != 0.0:
                    rr = sb
                    phi = (PI / 2.0) - (PI / 4.0) * (sa / sb)
                sa = rr * wp.cos(phi)
                sb = rr * wp.sin(phi)
            lp = c + uu * sa + vv * sb
            to_l = lp - o
            dist2 = wp.dot(to_l, to_l)
            dist = wp.sqrt(dist2)
            wi = to_l / dist
            cos_l = wp.dot(nrm, -wi)
            if cos_l <= 0.0:
                continue
            f = eval_brdf_cos(s, wo, wi)
            if f[0] + f[1] + f[2] <= 0.0:
                continue
            vis = shadow_transmittance(geo, mats, o, wi, dist - 1.0e-3)
            g = cos_l * lights.area[li] / dist2 * inv_n * inv_n
            out = out + wp.cw_mul(wp.cw_mul(f, lights.radiance[li]), vis) * g
    return out


@wp.func
def probe_index(grid: ProbeGrid, ix: wp.int32, iy: wp.int32, iz: wp.int32) -> wp.int32:
    return (iz * grid.ny + iy) * grid.nx + ix


@wp.func
def probe_irradiance(grid: ProbeGrid, p: wp.vec3, n: wp.vec3) -> wp.vec3:
    """Trilinear ambient-cube lookup; returns outgoing radiance of a white lambertian at p facing n."""
    q = (p + n * (0.5 * grid.cell) - grid.origin) / grid.cell
    fx = wp.clamp(q[0], 0.0, wp.float32(grid.nx - 1) - 1.0e-3)
    fy = wp.clamp(q[1], 0.0, wp.float32(grid.ny - 1) - 1.0e-3)
    fz = wp.clamp(q[2], 0.0, wp.float32(grid.nz - 1) - 1.0e-3)
    x0 = wp.int32(wp.floor(fx))
    y0 = wp.int32(wp.floor(fy))
    z0 = wp.int32(wp.floor(fz))
    tx = fx - wp.float32(x0)
    ty = fy - wp.float32(y0)
    tz = fz - wp.float32(z0)
    nn = wp.cw_mul(n, n)
    total = wp.vec3(0.0)
    wsum = float(0.0)
    for c in range(8):
        dx = c & 1
        dy = (c >> 1) & 1
        dz = (c >> 2) & 1
        idx = probe_index(grid, x0 + dx, y0 + dy, z0 + dz)
        w = wp.where(dx == 1, tx, 1.0 - tx) * wp.where(dy == 1, ty, 1.0 - ty) * wp.where(dz == 1, tz, 1.0 - tz)
        # reject probes behind the surface and probes buried in geometry
        probe_p = grid.origin + wp.vec3(wp.float32(x0 + dx), wp.float32(y0 + dy), wp.float32(z0 + dz)) * grid.cell
        facing = wp.dot(wp.normalize(probe_p - p + n * 1.0e-3), n)
        w = w * grid.valid[idx] * wp.max(0.05, (facing + 1.0) * 0.5)
        if w <= 0.0:
            continue
        b = 6 * idx
        e = (
            grid.cube[b + wp.where(n[0] >= 0.0, 0, 1)] * nn[0]
            + grid.cube[b + wp.where(n[1] >= 0.0, 2, 3)] * nn[1]
            + grid.cube[b + wp.where(n[2] >= 0.0, 4, 5)] * nn[2]
        )
        total = total + e * w
        wsum = wsum + w
    if wsum <= 0.0:
        return wp.vec3(0.0)
    return total / wsum


@wp.func
def hit_light(lights: LightSet, o: wp.vec3, d: wp.vec3, tmax: wp.float32) -> wp.vec3:
    """Radiance of the nearest light hit before tmax (rect and disk lights: front side only)."""
    best = tmax
    out = wp.vec3(0.0)
    for li in range(lights.count):
        c = lights.center[li]
        if lights.kind[li] == LIGHT_SPHERE:
            oc = o - c
            bq = wp.dot(oc, d)
            disc = bq * bq - (wp.dot(oc, oc) - lights.radius[li] * lights.radius[li])
            if disc < 0.0:
                continue
            t = -bq - wp.sqrt(disc)
            if t > 1.0e-4 and t < best:
                best = t
                out = lights.radiance[li]
            continue
        nrm = lights.normal[li]
        denom = wp.dot(nrm, d)
        if denom >= 0.0:
            continue
        t = wp.dot(c - o, nrm) / denom
        if t <= 1.0e-4 or t >= best:
            continue
        rel = o + d * t - c
        uu = lights.u[li]
        vv = lights.v[li]
        a = wp.dot(rel, uu) / wp.dot(uu, uu)
        b = wp.dot(rel, vv) / wp.dot(vv, vv)
        inside = wp.abs(a) <= 1.0 and wp.abs(b) <= 1.0
        if lights.kind[li] == LIGHT_DISK:
            inside = a * a + b * b <= 1.0
        if inside:
            best = t
            out = lights.radiance[li]
    return out


# ----------------------------------------------------------------------------------------------------------
# the megakernel
# ----------------------------------------------------------------------------------------------------------


@wp.kernel
def render_kernel(
    geo: Geometry,
    mats: Materials,
    lights: LightSet,
    grid: ProbeGrid,
    cfg: Settings,
    cam_pos: wp.array(dtype=wp.vec3),
    cam_rot: wp.array(dtype=wp.mat33),  # columns: right, up, back (camera looks down -Z)
    cam_tan: wp.array(dtype=wp.float32),  # tan(vfov / 2)
    width: wp.int32,
    height: wp.int32,
    radiance_out: wp.array(dtype=wp.vec3, ndim=3),
    albedo_out: wp.array(dtype=wp.vec3, ndim=3),
    normal_out: wp.array(dtype=wp.vec3, ndim=3),
    depth_out: wp.array(dtype=wp.float32, ndim=3),
):
    cam, py, px = wp.tid()
    rng = wp.rand_init(cfg.seed, (cam * height + py) * width + px)
    aspect = wp.float32(width) / wp.float32(height)
    rot = cam_rot[cam]
    total = wp.vec3(0.0)
    aov_albedo = wp.vec3(0.0)
    aov_normal = wp.vec3(0.0)
    aov_depth = float(0.0)
    for sample in range(cfg.spp):
        jx = float(0.5)
        jy = float(0.5)
        if cfg.spp > 1:
            if cfg.indirect != 2:
                # deterministic tiers: fixed stratified grid (ceil(sqrt(spp))^2 cells, spp of them used)
                g = wp.int32(wp.ceil(wp.sqrt(wp.float32(cfg.spp))))
                jx = (wp.float32(sample % g) + 0.5) / wp.float32(g)
                jy = (wp.float32(sample // g) + 0.5) / wp.float32(g)
            else:
                jx = wp.randf(rng)
                jy = wp.randf(rng)
        sx = (2.0 * (wp.float32(px) + jx) / wp.float32(width) - 1.0) * cam_tan[cam] * aspect
        sy = (1.0 - 2.0 * (wp.float32(py) + jy) / wp.float32(height)) * cam_tan[cam]
        d = wp.normalize(rot * wp.vec3(sx, sy, -1.0))
        o = cam_pos[cam]
        throughput = wp.vec3(1.0)
        radiance = wp.vec3(0.0)
        specular_path = int(1)  # camera ray or only specular bounces so far
        alive = int(1)  # cleared when the path terminates on its own (vs. running out of depth)
        aov_done = int(0)
        path_len = float(0.0)
        for depth in range(cfg.max_depth):
            hit = trace(geo, o, d, 1.0e4)
            if specular_path == 1 and (depth > 0 or cfg.lights_visible != 0):
                le = hit_light(lights, o, d, hit.t)
                if le[0] + le[1] + le[2] > 0.0:
                    radiance = radiance + wp.cw_mul(throughput, le)
                    if aov_done == 0:
                        aov_albedo = aov_albedo + wp.vec3(1.0)
                        aov_normal = aov_normal + (-d)
                        aov_depth = aov_depth + path_len + hit.t
                        aov_done = 1
                    alive = int(0)
                    break
            if hit.geom < 0:
                radiance = radiance + wp.cw_mul(throughput, lights.dome)
                if aov_done == 0:
                    aov_albedo = aov_albedo + wp.vec3(1.0)
                    aov_depth = aov_depth + 1.0e3
                    aov_done = 1
                alive = int(0)
                break
            path_len = path_len + hit.t
            s = shade_surface(geo, mats, hit, o, d)
            wo = -d
            r1 = wp.randf(rng)
            r2 = wp.randf(rng)
            r3 = wp.randf(rng)
            r4 = wp.randf(rng)

            # ---- clear dielectric (glass, water): Fresnel split ----
            if s.trans > 0.5:
                eta = wp.where(s.front == 1, s.ior, 1.0 / s.ior)
                cos_i = wp.dot(s.n, wo)
                fr = fresnel_dielectric(cos_i, eta)
                refl = wp.normalize(reflect(d, s.n))
                # refraction direction (Snell), eta_ratio = n_i / n_t
                k = 1.0 / eta
                c2 = 1.0 - k * k * (1.0 - cos_i * cos_i)
                if cfg.indirect != 2:
                    # deterministic: follow refraction, add the reflection's environment estimate (a random
                    # reflect-or-refract choice per sample would leave salt-and-pepper speckle at low spp)
                    env = lights.ambient
                    if cfg.indirect == 1:
                        env = probe_irradiance(grid, s.p, refl)
                    radiance = radiance + wp.cw_mul(throughput, env) * fr
                    if c2 <= 0.0:
                        d = refl
                    else:
                        d = wp.normalize(d * k + s.n * (k * cos_i - wp.sqrt(c2)))
                        throughput = throughput * (1.0 - fr)
                else:
                    if c2 <= 0.0 or r1 < fr:
                        d = refl
                    else:
                        d = wp.normalize(d * k + s.n * (k * cos_i - wp.sqrt(c2)))
                # tint: half the absorption per interface crossing
                throughput = wp.cw_mul(
                    throughput, wp.vec3(wp.sqrt(s.albedo[0]), wp.sqrt(s.albedo[1]), wp.sqrt(s.albedo[2]))
                )
                o = s.p + d * 5.0e-4
                continue

            # first non-specular vertex: denoiser guides
            if aov_done == 0:
                aov_albedo = aov_albedo + s.albedo
                aov_normal = aov_normal + s.n
                aov_depth = aov_depth + path_len
                aov_done = 1

            # ---- direct lighting ----
            l1 = r1
            l2 = r2
            if cfg.soft_shadows >= 2:
                l1 = wp.float32(0.5) + wp.float32(sample) * 0.7548777
                l2 = wp.float32(0.5) + wp.float32(sample) * 0.5698403
                l1 = l1 - wp.floor(l1)
                l2 = l2 - wp.floor(l2)
            radiance = radiance + wp.cw_mul(
                throughput, direct_light(geo, mats, lights, s, wo, cfg.soft_shadows, l1, l2)
            )

            ndv = wp.max(wp.dot(s.n, wo), 1.0e-4)
            fspec = glossy_reflectance(s, ndv)
            kd = diffuse_reflectance(s, ndv)

            if cfg.indirect == 2:
                # ---- path tracing: sample diffuse or GGX lobe ----
                p_spec = wp.clamp(luminance(fspec) / (luminance(fspec) + luminance(kd) + 1.0e-6), 0.1, 0.9)
                if luminance(kd) < 1.0e-4:
                    p_spec = 1.0
                wi = wp.vec3(0.0)
                if r3 < p_spec:
                    alpha = wp.max(s.rough * s.rough, 0.002)
                    hvec = ggx_sample_h(s.n, alpha, wp.randf(rng), wp.randf(rng))
                    wi = reflect(-wo, hvec)
                else:
                    wi = cosine_dir(s.n, wp.randf(rng), wp.randf(rng))
                ndl = wp.dot(s.n, wi)
                if ndl <= 0.0 or wp.dot(s.ng, wi) <= 0.0:
                    alive = int(0)
                    break
                # combined pdf of the two-lobe mixture
                alpha2 = wp.max(s.rough * s.rough, 0.002)
                hh = wp.normalize(wi + wo)
                ndh = wp.max(wp.dot(s.n, hh), 1.0e-4)
                pdf_spec = ggx_d(ndh, alpha2) * ndh / (4.0 * wp.max(wp.dot(wo, hh), 1.0e-4))
                pdf_diff = ndl / PI
                pdf = p_spec * pdf_spec + (1.0 - p_spec) * pdf_diff
                throughput = wp.cw_mul(throughput, eval_brdf_cos(s, wo, wi)) / wp.max(pdf, 1.0e-6)
                specular_path = 0
                if depth >= 2:
                    q = wp.clamp(wp.max(throughput[0], wp.max(throughput[1], throughput[2])), 0.05, 0.95)
                    if r4 > q:
                        alive = int(0)
                        break
                    throughput = throughput / q
                o = s.p + s.ng * 2.0e-4
                d = wi
                continue

            # ---- approximate indirect diffuse ----
            env_d = lights.ambient
            if cfg.indirect == 1:
                env_d = probe_irradiance(grid, s.p, s.n)
            radiance = radiance + wp.cw_mul(throughput, wp.cw_mul(kd, env_d))

            # ---- specular continuation for shiny surfaces, environment estimate for the rest ----
            if s.rough < 0.3 and luminance(fspec) > 0.05:
                wi = reflect(d, s.n)
                if cfg.glossy_sampling != 0:
                    alpha = wp.max(s.rough * s.rough, 0.002)
                    hvec = ggx_sample_h(s.n, alpha, r3, r4)
                    wi = reflect(d, hvec)
                if wp.dot(wi, s.ng) <= 0.0:
                    alive = int(0)
                    break
                throughput = wp.cw_mul(throughput, fspec)
                o = s.p + s.ng * 2.0e-4
                d = wp.normalize(wi)
                continue
            env_s = lights.ambient
            if cfg.indirect == 1:
                env_s = probe_irradiance(grid, s.p, wp.normalize(reflect(d, s.n)))
            radiance = radiance + wp.cw_mul(throughput, wp.cw_mul(fspec, env_s))
            alive = int(0)
            break
        if alive == 1 and cfg.indirect != 2:
            # out of depth inside a specular chain (e.g. glass walls + water): close with the ambient estimate
            env = lights.ambient
            if cfg.indirect == 1:
                env = probe_irradiance(grid, o, d)
            radiance = radiance + wp.cw_mul(throughput, env)
        # clamp fireflies
        lum = luminance(radiance)
        if lum > cfg.clamp:
            radiance = radiance * (cfg.clamp / lum)
        total = total + radiance
    inv = 1.0 / wp.float32(cfg.spp)
    radiance_out[cam, py, px] = total * inv
    albedo_out[cam, py, px] = aov_albedo * inv
    nl = wp.length(aov_normal)
    normal_out[cam, py, px] = wp.where(nl > 0.0, aov_normal / nl, wp.vec3(0.0))
    depth_out[cam, py, px] = aov_depth * inv


# ----------------------------------------------------------------------------------------------------------
# probe bake
# ----------------------------------------------------------------------------------------------------------


@wp.kernel
def bake_probes_kernel(
    geo: Geometry,
    mats: Materials,
    lights: LightSet,
    prev: ProbeGrid,
    use_prev: wp.int32,
    origin: wp.vec3,
    cell: wp.float32,
    nx: wp.int32,
    ny: wp.int32,
    samples: wp.int32,
    seed: wp.int32,
    cube_out: wp.array(dtype=wp.vec3),
    backface_out: wp.array(dtype=wp.float32),
):
    probe, axis = wp.tid()
    ix = probe % nx
    iy = (probe // nx) % ny
    iz = probe // (nx * ny)
    p = origin + wp.vec3(wp.float32(ix), wp.float32(iy), wp.float32(iz)) * cell
    sgn = wp.where(axis % 2 == 0, 1.0, -1.0)
    ax = axis // 2
    n = wp.vec3(wp.where(ax == 0, sgn, 0.0), wp.where(ax == 1, sgn, 0.0), wp.where(ax == 2, sgn, 0.0))
    rng = wp.rand_init(seed, probe * 6 + axis)
    acc = wp.vec3(0.0)
    back = float(0.0)
    for _ in range(samples):
        d = cosine_dir(n, wp.randf(rng), wp.randf(rng))
        hit = trace(geo, p, d, 1.0e4)
        # emitters are excluded: shading adds direct light itself, probes only carry indirect light
        le = hit_light(lights, p, d, hit.t)
        if le[0] + le[1] + le[2] > 0.0:
            continue
        if hit.geom < 0:
            acc = acc + lights.dome
            continue
        s = shade_surface(geo, mats, hit, p, d)
        if s.front == 0 and s.trans < 0.5:
            back = back + 1.0
            continue
        if s.trans > 0.5:
            # see through glass/water approximately: skip it
            continue
        wo = -d
        # diffuse-only outgoing radiance at the hit: direct + previous-bounce indirect
        s.metal = 0.0
        s.spec = 0.0
        lo = direct_light(geo, mats, lights, s, wo, 1, wp.randf(rng), wp.randf(rng))
        if use_prev != 0:
            lo = lo + wp.cw_mul(s.albedo, probe_irradiance(prev, s.p, s.n))
        acc = acc + lo
    cube_out[probe * 6 + axis] = acc / wp.float32(samples)
    backface_out[probe * 6 + axis] = back / wp.float32(samples)


# ----------------------------------------------------------------------------------------------------------
# dynamic geometry update, denoise, tonemap
# ----------------------------------------------------------------------------------------------------------


@wp.kernel
def smooth_normals_scatter(idx: wp.array(dtype=wp.int32), nrm: wp.array(dtype=wp.vec3), acc: wp.array(dtype=wp.vec3)):
    t = wp.tid()
    i0 = idx[3 * t]
    i1 = idx[3 * t + 1]
    i2 = idx[3 * t + 2]
    total = nrm[i0] + nrm[i1] + nrm[i2]
    wp.atomic_add(acc, i0, total)
    wp.atomic_add(acc, i1, total)
    wp.atomic_add(acc, i2, total)


@wp.kernel
def normalize_kernel(acc: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    n = wp.length(acc[i])
    if n > 1.0e-12:
        out[i] = acc[i] / n


@wp.kernel
def pose_dynamic_kernel(
    local_pts: wp.array(dtype=wp.vec3),
    local_nrm: wp.array(dtype=wp.vec3),
    body: wp.array(dtype=wp.int32),
    body_q: wp.array(dtype=wp.transform),
    out_pts: wp.array(dtype=wp.vec3),
    out_nrm: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    x = body_q[body[i // 3]]
    out_pts[i] = wp.transform_point(x, local_pts[i])
    out_nrm[i] = wp.transform_vector(x, local_nrm[i])


@wp.kernel
def atrous_kernel(
    color_in: wp.array(dtype=wp.vec3, ndim=3),
    albedo: wp.array(dtype=wp.vec3, ndim=3),
    normal: wp.array(dtype=wp.vec3, ndim=3),
    depth: wp.array(dtype=wp.float32, ndim=3),
    step: wp.int32,
    sigma_l: wp.float32,
    color_out: wp.array(dtype=wp.vec3, ndim=3),
):
    """One iteration of an edge-avoiding a-trous wavelet filter on demodulated irradiance."""
    cam, y, x = wp.tid()
    h = color_in.shape[1]
    w = color_in.shape[2]
    c0 = color_in[cam, y, x]
    n0 = normal[cam, y, x]
    z0 = depth[cam, y, x]
    l0 = luminance(c0)
    acc = wp.vec3(0.0)
    wsum = float(0.0)
    for j in range(-2, 3):
        for i in range(-2, 3):
            xx = wp.clamp(x + i * step, 0, w - 1)
            yy = wp.clamp(y + j * step, 0, h - 1)
            k = (1.0 - 0.375 * wp.float32(wp.abs(i))) * (1.0 - 0.375 * wp.float32(wp.abs(j)))  # ~B3 spline
            c = color_in[cam, yy, xx]
            nn = normal[cam, yy, xx]
            wn = wp.pow(wp.max(0.0, wp.dot(n0, nn)), 64.0)
            if wp.dot(n0, n0) < 0.5 and wp.dot(nn, nn) < 0.5:
                wn = 1.0  # both background / emitter pixels
            wz = wp.exp(-wp.abs(depth[cam, yy, xx] - z0) / (0.02 * z0 * wp.float32(step) + 1.0e-3))
            wl = wp.exp(-wp.abs(luminance(c) - l0) / (sigma_l * (l0 + 0.05) + 1.0e-4))
            wgt = k * wn * wz * wl
            acc = acc + c * wgt
            wsum = wsum + wgt
    color_out[cam, y, x] = acc / wp.max(wsum, 1.0e-6)


@wp.kernel
def outlier_clamp_kernel(color_in: wp.array(dtype=wp.vec3, ndim=3), color_out: wp.array(dtype=wp.vec3, ndim=3)):
    """Clamp each pixel to the min/max of its 8 neighbours (removes isolated fireflies and dark speckles,
    which the edge-stopping a-trous filter would otherwise preserve as 'edges')."""
    cam, y, x = wp.tid()
    h = color_in.shape[1]
    w = color_in.shape[2]
    lo = wp.vec3(1.0e30)
    hi = wp.vec3(-1.0e30)
    for j in range(-1, 2):
        for i in range(-1, 2):
            if i != 0 or j != 0:
                c = color_in[cam, wp.clamp(y + j, 0, h - 1), wp.clamp(x + i, 0, w - 1)]
                lo = wp.min(lo, c)
                hi = wp.max(hi, c)
    color_out[cam, y, x] = wp.min(wp.max(color_in[cam, y, x], lo), hi)


@wp.kernel
def demodulate_kernel(
    color: wp.array(dtype=wp.vec3, ndim=3),
    albedo: wp.array(dtype=wp.vec3, ndim=3),
    out: wp.array(dtype=wp.vec3, ndim=3),
):
    cam, y, x = wp.tid()
    a = albedo[cam, y, x]
    c = color[cam, y, x]
    out[cam, y, x] = wp.vec3(c[0] / wp.max(a[0], 0.02), c[1] / wp.max(a[1], 0.02), c[2] / wp.max(a[2], 0.02))


@wp.kernel
def remodulate_kernel(
    irr: wp.array(dtype=wp.vec3, ndim=3), albedo: wp.array(dtype=wp.vec3, ndim=3), out: wp.array(dtype=wp.vec3, ndim=3)
):
    cam, y, x = wp.tid()
    a = albedo[cam, y, x]
    out[cam, y, x] = wp.cw_mul(irr[cam, y, x], wp.vec3(wp.max(a[0], 0.02), wp.max(a[1], 0.02), wp.max(a[2], 0.02)))


@wp.func
def aces(x: wp.float32) -> wp.float32:
    # Narkowicz ACES filmic fit
    return wp.clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)


@wp.func
def to_srgb(x: wp.float32) -> wp.float32:
    if x <= 0.0031308:
        return 12.92 * x
    return 1.055 * wp.pow(x, 1.0 / 2.4) - 0.055


@wp.kernel
def tonemap_kernel(color: wp.array(dtype=wp.vec3, ndim=3), exposure: wp.float32, out: wp.array(dtype=wp.uint8, ndim=4)):
    cam, y, x = wp.tid()
    c = color[cam, y, x] * exposure
    out[cam, y, x, 0] = wp.uint8(wp.int32(to_srgb(aces(c[0])) * 255.0 + 0.5))
    out[cam, y, x, 1] = wp.uint8(wp.int32(to_srgb(aces(c[1])) * 255.0 + 0.5))
    out[cam, y, x, 2] = wp.uint8(wp.int32(to_srgb(aces(c[2])) * 255.0 + 0.5))


# ----------------------------------------------------------------------------------------------------------
# host side
# ----------------------------------------------------------------------------------------------------------

PRESETS = {
    # name: settings + denoise iterations
    "direct": dict(spp=1, max_depth=10, soft_shadows=0, indirect=0, glossy_sampling=0, denoise=0),
    "direct_aa4": dict(spp=4, max_depth=10, soft_shadows=0, indirect=0, glossy_sampling=0, denoise=0),
    "probe": dict(spp=1, max_depth=10, soft_shadows=0, indirect=1, glossy_sampling=0, denoise=0),
    "probe_soft": dict(spp=1, max_depth=10, soft_shadows=2, indirect=1, glossy_sampling=0, denoise=0),
    "probe_aa4": dict(spp=4, max_depth=10, soft_shadows=2, indirect=1, glossy_sampling=0, denoise=0),
    "pt4": dict(spp=4, max_depth=8, soft_shadows=1, indirect=2, glossy_sampling=1, denoise=5),
    "pt16": dict(spp=16, max_depth=10, soft_shadows=1, indirect=2, glossy_sampling=1, denoise=5),
    "pt64": dict(spp=64, max_depth=12, soft_shadows=1, indirect=2, glossy_sampling=1, denoise=4),
    "pt1024": dict(spp=1024, max_depth=16, soft_shadows=1, indirect=2, glossy_sampling=1, denoise=0),
}


def look_at_rotation(pos, target, up=(0.0, 0.0, 1.0)):
    pos, target, up = (np.asarray(x, dtype=np.float64) for x in (pos, target, up))
    back = pos - target
    back /= np.linalg.norm(back)
    right = np.cross(up, back)
    right /= np.linalg.norm(right)
    true_up = np.cross(back, right)
    return np.stack([right, true_up, back], axis=1)


def _empty(dtype, device):
    return wp.zeros(1, dtype=dtype, device=device)


class WarpRenderer:
    def __init__(self, scene, device="cuda:0", ambient=0.0, water_ior=1.333, lights_visible=True):
        self.device = device
        self.scene = scene
        wp.init()
        with wp.ScopedDevice(device):
            self._build_materials(scene, water_ior)
            self._build_lights(scene, ambient)
            self._build_geometry(scene)
        self.grid = ProbeGrid()
        self.grid.enabled = 0
        self.grid.cube = _empty(wp.vec3, device)
        self.grid.valid = _empty(wp.float32, device)
        self.grid.nx = self.grid.ny = self.grid.nz = 1
        self.grid.cell = 1.0
        self.grid.origin = wp.vec3(0.0)
        self.lights_visible = lights_visible
        # The water surface changes topology every frame, so its BVH is rebuilt (not refit): LBVH builds fastest.
        self.water_bvh = "lbvh"
        self.water_smooth = 3  # neighbour-averaging passes over the water's vertex normals
        self.bake_seconds = 0.0
        self._buffers = {}

    # -- setup --
    def _build_materials(self, scene, water_ior):
        mats = list(scene.materials)
        from omnigibson.render_backends.warp_renderer.extract import Material

        mats.append(
            Material(name="water", base_color=(0.93, 0.97, 1.0), roughness=0.0, transmission=1.0, ior=water_ior)
        )
        self.water_mat = len(mats) - 1
        m = Materials()
        dev = self.device
        m.base = wp.array(np.array([x.base_color for x in mats], dtype=np.float32), dtype=wp.vec3, device=dev)
        m.rough = wp.array(np.array([x.roughness for x in mats], dtype=np.float32), device=dev)
        m.metal = wp.array(np.array([x.metallic for x in mats], dtype=np.float32), device=dev)
        m.spec = wp.array(np.array([x.specular for x in mats], dtype=np.float32), device=dev)
        m.tint = wp.array(np.array([x.spec_tint for x in mats], dtype=np.float32), dtype=wp.vec3, device=dev)
        m.r0 = wp.array(np.array([x.r0 for x in mats], dtype=np.float32), device=dev)
        m.tex_r0 = wp.array(np.array([x.tex_r0 for x in mats], dtype=np.int32), device=dev)
        m.trans = wp.array(np.array([x.transmission for x in mats], dtype=np.float32), device=dev)
        m.ior = wp.array(np.array([x.ior for x in mats], dtype=np.float32), device=dev)
        for field in ("tex_diffuse", "tex_gloss", "tex_metal", "tex_normal"):
            setattr(m, field, wp.array(np.array([getattr(x, field) for x in mats], dtype=np.int32), device=dev))
        m.tex_refl = wp.array(np.array([x.tex_reflection for x in mats], dtype=np.int32), device=dev)
        texels, offsets, ws, hs, off = [], [], [], [], 0
        for img in scene.textures:
            h, w = img.shape[:2]
            packed = img.astype(np.uint32)
            packed = packed[..., 0] | (packed[..., 1] << 8) | (packed[..., 2] << 16) | (packed[..., 3] << 24)
            texels.append(packed.reshape(-1))
            offsets.append(off)
            ws.append(w)
            hs.append(h)
            off += w * h
        m.texels = wp.array(np.concatenate(texels) if texels else np.zeros(1, np.uint32), dtype=wp.uint32, device=dev)
        m.tex_offset = wp.array(np.array(offsets or [0], np.int32), device=dev)
        m.tex_w = wp.array(np.array(ws or [1], np.int32), device=dev)
        m.tex_h = wp.array(np.array(hs or [1], np.int32), device=dev)
        c = np.arange(256) / 255.0
        m.srgb_lut = wp.array(
            np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32), device=dev
        )
        self.mats = m
        self.texture_bytes = int(off * 4)

    def _build_lights(self, scene, ambient):
        rects = scene.lights.rects[:MAX_LIGHTS]
        lights = LightSet()
        dev = self.device
        lights.count = len(rects)

        def arr(key):
            return wp.array(np.array([r[key] for r in rects] or [[0, 0, 0]], np.float32), dtype=wp.vec3, device=dev)

        def area(r):
            if r.get("kind", 0) == 0:
                return 4 * np.linalg.norm(r["u"]) * np.linalg.norm(r["v"])
            return np.pi * r["radius"] ** 2

        lights.kind = wp.array(np.array([r.get("kind", 0) for r in rects] or [0], np.int32), device=dev)
        lights.center, lights.u, lights.v = arr("center"), arr("u"), arr("v")
        lights.normal, lights.radiance = arr("normal"), arr("radiance")
        lights.radius = wp.array(np.array([r.get("radius", 0.0) for r in rects] or [0], np.float32), device=dev)
        lights.area = wp.array(np.array([area(r) for r in rects] or [0], np.float32), device=dev)
        lights.dome = wp.vec3(*scene.lights.dome)
        lights.ambient = wp.vec3(ambient, ambient, ambient)
        self.lights = lights
        # firefly clamp: the brightest emitter (lights, the dome, or the constant ambient)
        lum = [
            0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2] for c in [r["radiance"] for r in rects] + [scene.lights.dome]
        ]
        self.max_light_luminance = float(max(lum + [ambient, 1.0]))

    def _build_geometry(self, scene):
        dev = self.device
        g = Geometry()
        g.water_mat = self.water_mat
        sp = scene.s_pos.reshape(-1, 3)
        if len(sp) == 0:
            # e.g. OmniGibson, where every mesh belongs to a body: warp cannot build a BVH over zero triangles, so
            # keep one degenerate triangle far away
            sp = np.full((3, 3), 1.0e6, np.float32)
        self.s_pts = wp.array(sp, dtype=wp.vec3, device=dev)
        self.s_mesh = wp.Mesh(
            points=self.s_pts, indices=wp.array(np.arange(len(sp), dtype=np.int32), device=dev), bvh_constructor="sah"
        )
        g.s_pts = self.s_pts
        n_static = len(sp) // 3
        g.s_nrm = wp.array(
            scene.s_nrm.reshape(-1, 3) if len(scene.s_pos) else np.zeros((3, 3), np.float32), dtype=wp.vec3, device=dev
        )
        g.s_uv = wp.array(
            scene.s_uv.reshape(-1, 2) if len(scene.s_pos) else np.zeros((3, 2), np.float32), dtype=wp.vec2, device=dev
        )
        g.s_mat = wp.array(
            scene.s_mat if len(scene.s_pos) else np.zeros(n_static, np.int32), dtype=wp.int32, device=dev
        )
        nd = len(scene.d_pos)
        self.d_mesh = None
        if nd:
            self.d_local = wp.array(scene.d_pos.reshape(-1, 3), dtype=wp.vec3, device=dev)
            self.d_local_n = wp.array(scene.d_nrm.reshape(-1, 3), dtype=wp.vec3, device=dev)
            self.d_body = wp.array(scene.d_body, dtype=wp.int32, device=dev)
            self.d_pts = wp.zeros(nd * 3, dtype=wp.vec3, device=dev)
            self.d_nrm = wp.zeros(nd * 3, dtype=wp.vec3, device=dev)
            g.d_pts, g.d_nrm = self.d_pts, self.d_nrm
            g.d_uv = wp.array(scene.d_uv.reshape(-1, 2), dtype=wp.vec2, device=dev)
            g.d_mat = wp.array(scene.d_mat, dtype=wp.int32, device=dev)
        else:
            g.d_pts, g.d_nrm = _empty(wp.vec3, dev), _empty(wp.vec3, dev)
            g.d_uv, g.d_mat = _empty(wp.vec2, dev), _empty(wp.int32, dev)
        g.w_pts, g.w_nrm, g.w_idx = _empty(wp.vec3, dev), _empty(wp.vec3, dev), _empty(wp.int32, dev)
        self.w_mesh = None
        self.geo = g
        self._mesh_ids = np.zeros(3, dtype=np.uint64)
        self._mesh_ids[0] = self.s_mesh.id
        self.geo.mesh = wp.array(self._mesh_ids, dtype=wp.uint64, device=dev)

    # -- per frame --
    def set_body_q(self, body_q):
        if self.d_mesh is None and len(self.scene.d_pos) == 0:
            return
        bq = wp.array(np.asarray(body_q, np.float32), dtype=wp.transform, device=self.device)
        n = len(self.scene.d_pos) * 3
        wp.launch(
            pose_dynamic_kernel,
            dim=n,
            inputs=[self.d_local, self.d_local_n, self.d_body, bq],
            outputs=[self.d_pts, self.d_nrm],
            device=self.device,
        )
        if self.d_mesh is None:
            self.d_mesh = wp.Mesh(
                points=self.d_pts,
                indices=wp.array(np.arange(n, dtype=np.int32), device=self.device),
                bvh_constructor="sah",
            )
            self._mesh_ids[1] = self.d_mesh.id
            self.geo.mesh = wp.array(self._mesh_ids, dtype=wp.uint64, device=self.device)
        else:
            self.d_mesh.refit()

    def set_water(self, points, indices, normals=None):
        dev = self.device
        if points is None or len(points) == 0 or len(indices) == 0:
            self.w_mesh = None
            self._mesh_ids[2] = 0
            self.geo.mesh = wp.array(self._mesh_ids, dtype=wp.uint64, device=dev)
            return
        points = np.asarray(points, np.float32)
        indices = np.asarray(indices, np.int32).reshape(-1)
        if normals is None or len(normals) != len(points):
            tri = indices.reshape(-1, 3)
            fn = np.cross(points[tri[:, 1]] - points[tri[:, 0]], points[tri[:, 2]] - points[tri[:, 0]])
            normals = np.zeros_like(points)
            for k in range(3):
                np.add.at(normals, tri[:, k], fn)
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        self.geo.w_pts = wp.array(points, dtype=wp.vec3, device=dev)
        self.geo.w_idx = wp.array(indices, dtype=wp.int32, device=dev)
        self.geo.w_nrm = wp.array(np.asarray(normals, np.float32), dtype=wp.vec3, device=dev)
        # The reconstructed MPM surface is bumpy at the voxel scale; refracting through it sparkles. Average each
        # vertex normal with its neighbours' (shading only; the geometry is unchanged).
        acc = wp.empty_like(self.geo.w_nrm)
        for _ in range(self.water_smooth):
            acc.zero_()
            wp.launch(
                smooth_normals_scatter,
                dim=len(indices) // 3,
                inputs=[self.geo.w_idx, self.geo.w_nrm],
                outputs=[acc],
                device=dev,
            )
            wp.launch(normalize_kernel, dim=len(points), inputs=[acc], outputs=[self.geo.w_nrm], device=dev)
        self.w_mesh = wp.Mesh(points=self.geo.w_pts, indices=self.geo.w_idx, bvh_constructor=self.water_bvh)
        self._mesh_ids[2] = self.w_mesh.id
        self.geo.mesh = wp.array(self._mesh_ids, dtype=wp.uint64, device=dev)

    def bake_probes(self, lo, hi, cell=0.15, samples=256, bounces=2, seed=7):
        """Bake the irradiance-probe grid over the box [lo, hi] (static lighting; call after set_body_q)."""
        t0 = time.perf_counter()
        lo = np.asarray(lo, np.float64)
        hi = np.asarray(hi, np.float64)
        dims = np.maximum(np.ceil((hi - lo) / cell).astype(int) + 1, 2)
        count = int(np.prod(dims))
        dev = self.device
        grid = ProbeGrid()
        grid.enabled = 1
        grid.origin = wp.vec3(*lo.astype(np.float32))
        grid.cell = float(cell)
        grid.nx, grid.ny, grid.nz = (int(x) for x in dims)
        grid.valid = wp.ones(count, dtype=wp.float32, device=dev)
        grid.cube = wp.zeros(count * 6, dtype=wp.vec3, device=dev)
        back = wp.zeros(count * 6, dtype=wp.float32, device=dev)
        for b in range(bounces):
            cube = wp.zeros(count * 6, dtype=wp.vec3, device=dev)
            wp.launch(
                bake_probes_kernel,
                dim=(count, 6),
                inputs=[
                    self.geo,
                    self.mats,
                    self.lights,
                    grid,
                    int(b > 0),
                    grid.origin,
                    grid.cell,
                    grid.nx,
                    grid.ny,
                    samples,
                    seed + b,
                ],
                outputs=[cube, back],
                device=dev,
            )
            grid.cube = cube
            if b == 0:
                frac = back.numpy().reshape(count, 6).mean(axis=1)
                grid.valid = wp.array((frac < 0.25).astype(np.float32), device=dev)
        wp.synchronize_device(dev)
        self.grid = grid
        self.bake_seconds = time.perf_counter() - t0
        self.probe_count = count
        return self.bake_seconds

    def _buffer(self, name, shape, dtype):
        key = (name, shape)
        if key not in self._buffers:
            self._buffers[key] = wp.zeros(shape, dtype=dtype, device=self.device)
        return self._buffers[key]

    def render(self, cameras, width, height, preset="probe", exposure=1.0, seed=0, return_aovs=False, **overrides):
        """Render ``cameras`` (list of dicts with pos, look_at, fov) into an (N, H, W, 3) uint8 array."""
        p = dict(PRESETS[preset])
        p.update(overrides)
        if p["indirect"] == 1 and not self.grid.enabled:
            raise RuntimeError("preset needs bake_probes() first")
        if p["soft_shadows"] != 1 and p["indirect"] != 2 and p["glossy_sampling"] == 0:
            # Deterministic lighting: keep the anti-aliasing jitter pattern fixed across frames, so static pixels
            # do not shimmer in video (only stochastic tiers take a per-frame seed).
            seed = 0
        cfg = Settings()
        cfg.spp, cfg.max_depth = p["spp"], p["max_depth"]
        cfg.soft_shadows, cfg.indirect, cfg.glossy_sampling = p["soft_shadows"], p["indirect"], p["glossy_sampling"]
        cfg.seed, cfg.exposure, cfg.lights_visible = seed, exposure, int(self.lights_visible)
        cfg.clamp = float(p.get("clamp", self.max_light_luminance))
        n = len(cameras)
        dev = self.device
        cam_pos = wp.array(np.array([c["pos"] for c in cameras], np.float32), dtype=wp.vec3, device=dev)
        cam_rot = wp.array(
            np.array([look_at_rotation(c["pos"], c["look_at"], c.get("up", (0, 0, 1))) for c in cameras], np.float32),
            dtype=wp.mat33,
            device=dev,
        )
        cam_tan = wp.array(np.array([math.tan(math.radians(c["fov"]) / 2) for c in cameras], np.float32), device=dev)
        shape = (n, height, width)
        radiance = self._buffer("radiance", shape, wp.vec3)
        albedo = self._buffer("albedo", shape, wp.vec3)
        normal = self._buffer("normal", shape, wp.vec3)
        depth = self._buffer("depth", shape, wp.float32)
        wp.launch(
            render_kernel,
            dim=shape,
            inputs=[self.geo, self.mats, self.lights, self.grid, cfg, cam_pos, cam_rot, cam_tan, width, height],
            outputs=[radiance, albedo, normal, depth],
            device=dev,
        )
        color = radiance
        if p["denoise"] > 0:
            a = self._buffer("tmp_a", shape, wp.vec3)
            b = self._buffer("tmp_b", shape, wp.vec3)
            wp.launch(demodulate_kernel, dim=shape, inputs=[radiance, albedo], outputs=[b], device=dev)
            wp.launch(outlier_clamp_kernel, dim=shape, inputs=[b], outputs=[a], device=dev)
            for it in range(p["denoise"]):
                wp.launch(
                    atrous_kernel, dim=shape, inputs=[a, albedo, normal, depth, 1 << it, 0.8], outputs=[b], device=dev
                )
                a, b = b, a
            color = self._buffer("denoised", shape, wp.vec3)
            wp.launch(remodulate_kernel, dim=shape, inputs=[a, albedo], outputs=[color], device=dev)
        out = self._buffer("ldr", (n, height, width, 3), wp.uint8)
        wp.launch(tonemap_kernel, dim=shape, inputs=[color, float(exposure)], outputs=[out], device=dev)
        if return_aovs:
            return out, dict(radiance=color, albedo=albedo, normal=normal, depth=depth)
        return out
