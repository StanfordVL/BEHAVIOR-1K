import os
import time

import GPUtil
import numpy as np

import omnigibson
from omnigibson.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from omnigibson.utils.asset_utils import download_assets


def test_render_loading_cleaning():
    renderer = MeshRenderer(width=800, height=600)
    renderer.release()


def test_projection_matrix():
    renderer = MeshRenderer(width=128, height=128)
    K_original = np.array([[134.64, 0, 60.44], [0, 134.64, 45.14], [0, 0, 1]])

    renderer.set_projection_matrix(K_original[0, 0], K_original[1, 1], K_original[0, 2], K_original[1, 2], 0.1, 100)
    print(renderer.P)
    K_recovered = np.array(renderer.get_intrinsics())
    print(K_original, K_recovered)
    max_error = np.max(np.abs(K_original - K_recovered))
    print(max_error)
    assert max_error < 1e-3
    renderer.release()


def test_projection_matrix_and_fov():
    renderer = MeshRenderer(width=128, height=128)
    renderer.set_fov(90)
    intrinsics = renderer.get_intrinsics()
    correct_intrinsics = np.array([[64, 0, 64], [0, 64, 64], [0, 0, 1]])
    max_error = np.max(np.abs(intrinsics - correct_intrinsics))
    assert max_error < 1e-3
    print(max_error)
    renderer.release()


def test_transform_point():
    renderer = MeshRenderer(width=128, height=128)
    renderer.set_fov(90)
    renderer.set_camera(camera=[1, 1, 1], target=[1, 2, 1], up=[0, 0, 1])  # look at y
    transformed_point = renderer.transform_point([1.1, 3, 2])  # z backward, y up, x right
    correct_transformed_point = np.array([0.1, 1.0, -2.0])
    max_error = np.max(np.abs(correct_transformed_point - transformed_point))
    assert max_error < 1e-3
    renderer.release()


def test_project_point():
    renderer = MeshRenderer(width=128, height=128)
    renderer.set_fov(90)
    renderer.set_camera(camera=[1, 1, 1], target=[1, 2, 1], up=[0, 0, 1])  # look at y

    n_trials = 10
    np.random.seed(1)
    for _ in range(n_trials):
        transformed_point = renderer.transform_point(np.random.uniform(-5, 5, size=(3,)))  # z backward, y up, x right
        projected_point_h = renderer.get_intrinsics().dot(transformed_point)
        projected_point_using_k = projected_point_h[:2] / projected_point_h[2]  # y pointing down, x left
        projected_point_using_k = (
            np.array([renderer.width, renderer.height]) - projected_point_using_k
        )  # y pointing up, x right

        clip_space = renderer.P.T.dot(np.concatenate([transformed_point, [1]]))
        ndc_space = clip_space[:3] / clip_space[3]

        projected_point_using_ndc = (
            ndc_space[:2]
        ) * renderer.width / 2.0 + renderer.width / 2.0  # y pointing up, x right

        max_error = np.max(np.abs(projected_point_using_k - projected_point_using_ndc))
        # print(projected_point_using_ndc, projected_point_using_k)
        assert max_error < 1e-3
    renderer.release()


def test_render_rendering(record_property):
    download_assets()
    test_dir = os.path.join(omnigibson.assets_path, "test")

    renderer = MeshRenderer(width=800, height=600)
    start = time.time()
    renderer.load_object(os.path.join(test_dir, "mesh/bed1a77d92d64f5cbbaaae4feed64ec1_new.obj"))
    elapsed = time.time() - start
    renderer.add_instance_group([0])
    renderer.set_camera([0, 0, 1.2], [0, 1, 1.2], [0, 1, 0])
    renderer.set_fov(90)
    rgb = renderer.render(("rgb"))[0]
    record_property("object_loading_time", elapsed)

    assert np.sum(rgb, axis=(0, 1, 2)) > 0
    renderer.release()


def test_render_rendering_cleaning():
    download_assets()
    test_dir = os.path.join(omnigibson.assets_path, "test")

    for i in range(5):
        renderer = MeshRenderer(width=800, height=600)
        renderer.load_object(os.path.join(test_dir, "mesh/bed1a77d92d64f5cbbaaae4feed64ec1_new.obj"))
        renderer.add_instance_group([0])
        renderer.set_camera([0, 0, 1.2], [0, 1, 1.2], [0, 1, 0])
        renderer.set_fov(90)
        rgb = renderer.render(("rgb"))[0]
        assert np.sum(rgb, axis=(0, 1, 2)) > 0

        GPUtil.showUtilization()
        renderer.release()
        GPUtil.showUtilization()
