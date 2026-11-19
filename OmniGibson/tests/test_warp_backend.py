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
    env.step(th.empty(0, device=env.device))
    obs, _ = camera.get_obs()
    rgb = obs["rgb"].cpu().numpy()
    assert rgb.shape == (240, 320, 4) and rgb.dtype == np.uint8
    assert np.ptp(rgb[..., :3]) > 0
    backend = og.sim.render_backend
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
