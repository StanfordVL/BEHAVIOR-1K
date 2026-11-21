"""Run with OMNIGIBSON_PHYSICS_BACKEND=newton OMNIGIBSON_RENDER_BACKEND=warp.

Requires the newton environment, a CUDA GPU, and CUDA-enabled OIDN.
"""

import numpy as np
import pytest
import torch as th

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.objects import PrimitiveObject


@pytest.mark.skipif(
    gm.PHYSICS_BACKEND != "newton" or gm.RENDER_BACKEND != "warp",
    reason="Requires Newton physics and Warp rendering",
)
def test_warp_rgb_rebuild_and_clear():
    env = og.Environment(
        configs={
            "scene": {"type": "Scene"},
            "env": {"device": "cuda:0"},
            "render": {"viewer_width": 320, "viewer_height": 240},
            "objects": [
                {
                    "type": "PrimitiveObject",
                    "name": "box",
                    "primitive_type": "Cube",
                    "size": 0.5,
                    "position": [0, 0, 1],
                    "fixed_base": True,
                }
            ],
        }
    )
    camera = og.sim.viewer_camera
    camera.set_position_orientation(
        position=th.tensor([0.0, -3.0, 1.0]),
        orientation=th.tensor([0.70710678, 0.0, 0.0, 0.70710678]),
    )
    backend = og.sim.render_backend
    target = th.tensor([0.0, 0.0, 1.0])
    backend.focus_viewer_camera(target)
    parameters = backend._camera_parameters(camera.prim_path, 320, 240)
    assert parameters["fov"] == pytest.approx(60.0)
    np.testing.assert_allclose(
        parameters["pos"] + (parameters["look_at"] - parameters["pos"]) * backend._view_distance,
        target.numpy(),
        atol=1e-5,
    )
    env.step(th.empty(0, device=env.device))
    obs, _ = camera.get_obs()
    rgb = obs["rgb"].cpu().numpy()
    assert rgb.shape == (240, 320, 4) and rgb.dtype == np.uint8
    assert np.ptp(rgb[..., :3]) > 0
    # Smoothed normals on a visible floor must not remain back-facing after correction.
    import warp as wp

    from omnigibson.render_backends.warp_renderer import renderer as wr

    @wp.kernel
    def check_grazing_normal(geo: wr.Geometry, mats: wr.Materials, output: wp.array(dtype=wp.float32)):
        hit = wr.Hit()
        hit.geom = 0
        hit.face = 0
        hit.u = 0.25
        hit.v = 0.25
        direction = wp.vec3(-0.8, 0.0, -0.6)
        surface = wr.shade_surface(geo, mats, hit, wp.vec3(0.0), direction)
        output[0] = wp.dot(surface.n, -direction)

    device = backend._renderer.device
    geometry = wr.Geometry()
    geometry.s_pts = wp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=wp.vec3, device=device)
    geometry.s_nrm = wp.array([[-0.70710678, 0, 0.70710678]] * 3, dtype=wp.vec3, device=device)
    geometry.s_uv = wp.zeros(3, dtype=wp.vec2, device=device)
    material = next(i for i, m in enumerate(backend._renderer.scene.materials) if m.tex_normal < 0)
    geometry.s_mat = wp.array([material], dtype=wp.int32, device=device)
    facing = wp.zeros(1, dtype=wp.float32, device=device)
    wp.launch(check_grazing_normal, dim=1, inputs=[geometry, backend._renderer.mats], outputs=[facing], device=device)
    assert facing.numpy()[0] > 0
    if gm.HEADLESS:
        assert backend._viewer is None
    with pytest.raises(NotImplementedError, match="only RGB"):
        backend.create_annotator("distance_to_camera")
    model = backend._model
    og.sim.stop()
    env.scene.add_object(
        PrimitiveObject(
            name="extra",
            primitive_type="Sphere",
            radius=0.2,
            position=[1, 0, 1],
            fixed_base=True,
        )
    )
    og.sim.play()
    env.step(th.empty(0, device=env.device))
    camera.get_obs()
    assert backend._model is not model
    og.clear()
    assert backend._renderer is None and backend._denoiser is None
