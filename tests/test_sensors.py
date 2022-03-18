import os

import numpy as np

import igibson
from igibson.envs.igibson_env import iGibsonEnv
from igibson.sensors.scan_sensor import ScanSensor
from igibson.sensors.vision_sensor import VisionSensor
from igibson.utils.assets_utils import download_assets, download_demo_data
from igibson.utils.constants import MAX_CLASS_COUNT


def test_vision_sensor():
    download_assets()
    download_demo_data()
    config_filename = os.path.join(os.path.dirname(__file__), "test_house.yaml")
    env = iGibsonEnv(config_file=config_filename, mode="headless")
    vision_modalities = ["rgb", "depth", "pc", "normal", "seg"]
    vision_sensor = VisionSensor(env, vision_modalities)
    vision_obs = vision_sensor.get_obs(env)

    assert vision_obs["rgb"].shape == (env.image_height, env.image_width, 3)
    assert np.all(0 <= vision_obs["rgb"]) and np.all(vision_obs["rgb"] <= 1.0)

    assert vision_obs["depth"].shape == (env.image_height, env.image_width, 1)
    assert np.all(0 <= vision_obs["depth"]) and np.all(vision_obs["depth"] <= 1.0)

    assert vision_obs["pc"].shape == (env.image_height, env.image_width, 3)

    assert vision_obs["normal"].shape == (env.image_height, env.image_width, 3)
    normal_norm = np.linalg.norm(vision_obs["normal"] * 2 - 1, axis=2)
    assert np.sum(np.abs(normal_norm - 1) > 0.1) / (env.image_height * env.image_width) < 0.05

    assert vision_obs["seg"].shape == (env.image_height, env.image_width, 1)
    assert np.all(0 <= vision_obs["seg"]) and np.all(vision_obs["seg"] <= MAX_CLASS_COUNT)


def test_scan_sensor():
    download_assets()
    download_demo_data()
    config_filename = os.path.join(os.path.dirname(__file__), "test_house.yaml")
    env = iGibsonEnv(config_file=config_filename, mode="headless")
    scan_sensor = ScanSensor(env, ["scan"])
    scan_obs = scan_sensor.get_obs(env)["scan"]

    assert scan_obs.shape == (scan_sensor.n_horizontal_rays, scan_sensor.n_vertical_beams)
    assert np.all(0 <= scan_obs) and np.all(scan_obs <= 1.0)
