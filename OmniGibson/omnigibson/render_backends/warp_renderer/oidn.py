"""Intel Open Image Denoise on the GPU, on Warp arrays in place (ctypes over the OIDN 2.x C API).

Uses the CUDA build of OIDN installed under the active environment's opt directory. OIDN's CUDA device
runs ordinary CUDA kernels (tensor cores where available), not RT cores.
"""

import ctypes
import os
import sys
from pathlib import Path

import warp as wp

from omnigibson.render_backends.warp_renderer.renderer import tonemap_kernel

DEVICE_TYPE_CUDA = 3
FORMAT_FLOAT3 = 3
QUALITY = {"fast": 4, "balanced": 5, "high": 6}


def _lib():
    path = Path(
        os.environ.get(
            "OIDN_LIBRARY", str(Path(sys.prefix) / "opt/oidn-2.5.1.x86_64.linux/lib/libOpenImageDenoise.so.2")
        )
    )
    if not path.is_file():
        raise RuntimeError(
            "Install the CUDA-enabled OIDN 2.5.1 Linux archive under $CONDA_PREFIX/opt, or set OIDN_LIBRARY to libOpenImageDenoise.so.2"
        )
    lib = ctypes.CDLL(str(path))
    vp, cp, i, sz = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_size_t
    lib.oidnNewCUDADevice.restype = vp
    lib.oidnNewCUDADevice.argtypes = [ctypes.POINTER(i), ctypes.POINTER(vp), i]
    lib.oidnCommitDevice.argtypes = [vp]
    lib.oidnGetDeviceError.restype = i
    lib.oidnGetDeviceError.argtypes = [vp, ctypes.POINTER(cp)]
    lib.oidnNewFilter.restype = vp
    lib.oidnNewFilter.argtypes = [vp, cp]
    lib.oidnSetSharedFilterImage.argtypes = [vp, cp, vp, i, sz, sz, sz, sz, sz]
    lib.oidnSetFilterBool.argtypes = [vp, cp, ctypes.c_bool]
    lib.oidnSetFilterInt.argtypes = [vp, cp, i]
    lib.oidnCommitFilter.argtypes = [vp]
    lib.oidnExecuteFilter.argtypes = [vp]
    lib.oidnSyncDevice.argtypes = [vp]
    lib.oidnReleaseFilter.argtypes = [vp]
    lib.oidnReleaseDevice.argtypes = [vp]
    return lib


class Denoiser:
    """Denoise an (N, H, W) vec3 radiance array with albedo + normal guides, writing into ``output``."""

    def __init__(self, device="cuda:0", quality="balanced"):
        self.lib = _lib()
        dev = wp.get_device(device)
        ids = (ctypes.c_int * 1)(dev.ordinal)
        streams = (ctypes.c_void_p * 1)(wp.get_stream(dev).cuda_stream)
        self.device = self.lib.oidnNewCUDADevice(ids, streams, 1)
        if not self.device:
            raise RuntimeError("OIDN could not create a CUDA device")
        self.lib.oidnCommitDevice(self.device)
        self._check()
        self.quality = QUALITY[quality]
        self.filters = {}

    def close(self):
        for f in self.filters.values():
            self.lib.oidnReleaseFilter(f)
        self.filters.clear()
        if self.device:
            self.lib.oidnReleaseDevice(self.device)
            self.device = None

    def _check(self):
        msg = ctypes.c_char_p()
        if self.lib.oidnGetDeviceError(self.device, ctypes.byref(msg)):
            raise RuntimeError(f"OIDN: {msg.value.decode() if msg.value else 'error'}")

    def __call__(self, color, albedo, normal, output):
        n, h, w = color.shape
        stride = h * w * 12
        for cam in range(n):
            key = (cam, h, w, color.ptr, albedo.ptr, normal.ptr, output.ptr)
            f = self.filters.get(key)
            if f is None:
                f = self.lib.oidnNewFilter(self.device, b"RT")
                for name, arr in ((b"color", color), (b"albedo", albedo), (b"normal", normal), (b"output", output)):
                    self.lib.oidnSetSharedFilterImage(f, name, arr.ptr + cam * stride, FORMAT_FLOAT3, w, h, 0, 0, 0)
                self.lib.oidnSetFilterBool(f, b"hdr", True)
                self.lib.oidnSetFilterInt(f, b"quality", self.quality)
                self.lib.oidnCommitFilter(f)
                self._check()
                self.filters[key] = f
            self.lib.oidnExecuteFilter(f)
        self._check()
        return output


def parse_preset(name):
    """``pt<N>_oidn`` / ``probe_oidn`` -> (warp preset, overrides).

    pt<N>: the path tracer at N spp. probe: baked probes for indirect light, with stochastic soft shadows and
    GGX-sampled glossy reflections (the noise OIDN removes).
    """
    base = name.removesuffix("_oidn")
    if base == "probe":
        return "probe_soft", dict(soft_shadows=1, glossy_sampling=1)
    return "pt4", dict(spp=int(base[2:]))


def warp_with_oidn(renderer, denoiser, cameras, width, height, preset, exposure, seed=0, **overrides):
    """Render ``preset`` without its own denoiser, OIDN the radiance, tonemap. Returns (N, H, W, 3) uint8 wp.array."""
    _, aovs = renderer.render(
        cameras, width, height, preset=preset, exposure=exposure, seed=seed, return_aovs=True, denoise=0, **overrides
    )
    shape = aovs["radiance"].shape
    out = renderer._buffer("oidn", shape, wp.vec3)
    denoiser(aovs["radiance"], aovs["albedo"], aovs["normal"], out)
    ldr = renderer._buffer("ldr", (*shape, 3), wp.uint8)
    wp.launch(tonemap_kernel, dim=shape, inputs=[out, float(exposure)], outputs=[ldr], device=renderer.device)
    return ldr
