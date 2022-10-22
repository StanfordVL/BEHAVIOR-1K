import logging
import os
import platform

import numpy as np
import torch

import omnigibson
from omnigibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from omnigibson.render.mesh_renderer.mesh_renderer_tensor import MeshRendererG2G


def test_tensor_render_rendering():
    if platform.system() != "Linux":
        logging.warning("Skip test_tensor_render_rendering on non-Linux platforms.")
        return
    w = 800
    h = 600
    setting = MeshRendererSettings(enable_pbr=False, msaa=True, enable_shadow=False)
    renderer = MeshRendererG2G(w, h, rendering_settings=setting)
    test_dir = os.path.join(omnigibson.assets_path, "test")
    renderer.load_object(os.path.join(test_dir, "mesh/bed1a77d92d64f5cbbaaae4feed64ec1_new.obj"))
    renderer.add_instance_group([0])

    renderer.set_camera([0, 0, 1.2], [0, 1, 1.2], [0, 1, 0])
    renderer.set_fov(90)
    tensor, tensor2 = renderer.render(modes=("rgb", "normal"))

    img_np = tensor.flip(0).data.cpu().numpy().reshape(h, w, 4)
    img_np2 = tensor2.flip(0).data.cpu().numpy().reshape(h, w, 4)

    # print(np.mean(img_np.astype(np.float32), axis=(0, 1)))
    # plt.subplot(1,2,1)
    # plt.imshow(img_np)
    # plt.subplot(1,2,2)
    # plt.imshow(img_np2)
    # plt.show()
    assert np.allclose(
        np.mean(img_np.astype(np.float32), axis=(0, 1)),
        np.array([131.72513, 128.30641, 121.820724, 255.56105]),
        rtol=1e-3,
    )

    renderer.release()
