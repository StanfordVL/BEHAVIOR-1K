import argparse
import logging
import os
import random
import subprocess
from sys import platform

import numpy as np
from PIL import Image

import igibson
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.render.profiler import Profiler
from igibson.scenes.interactive_traversable_scene import InteractiveIndoorScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_ig_scene_path


def main(random_selection=False, headless=False, short_exec=False):
    """
    Generates videos navigating in the iG scenes
    Loads an iG scene, predefined paths and produces a video. Alternate random textures and/or objects, on demand.
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    # Assuming that if random_selection=True, headless=True, short_exec=True, we are calling it from tests and we
    # do not want to parse args (it would fail because the calling function is pytest "testfile.py")
    if not (random_selection and headless and short_exec):
        parser = argparse.ArgumentParser()
        parser.add_argument("--scene", type=str, help="Name of the scene in the iG Dataset", default="Rs_int")
        parser.add_argument("--save_dir", type=str, help="Directory to save the frames.", default="misc")
        parser.add_argument("--seed", type=int, default=15, help="Random seed.")
        parser.add_argument("--domain_rand", dest="domain_rand", action="store_true")
        parser.add_argument("--domain_rand_interval", dest="domain_rand_interval", type=int, default=50)
        parser.add_argument("--object_rand", dest="object_rand", action="store_true")
        args = parser.parse_args()
        scene_name = args.scene
        save_dir = args.save_dir
        seed = args.seed
        domain_rand = args.domain_rand
        domain_rand_interval = args.domain_rand_interval
        object_rand = args.object_rand
    else:
        scene_name = "Rs_int"
        save_dir = "misc"
        seed = 15
        domain_rand = False
        domain_rand_interval = 50
        object_rand = False

    # hdr_texture1 = os.path.join(
    # igibson.ig_dataset_path, 'scenes', 'background', 'photo_studio_01_2k.hdr')
    hdr_texture1 = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_03.hdr")
    hdr_texture2 = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_02.hdr")
    light_map = os.path.join(get_ig_scene_path(scene_name), "layout", "floor_lighttype_0.png")

    background_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "urban_street_01.jpg")

    settings = MeshRendererSettings(
        env_texture_filename=hdr_texture1,
        env_texture_filename2=hdr_texture2,
        env_texture_filename3=background_texture,
        light_modulation_map_filename=light_map,
        enable_shadow=True,
        msaa=True,
        skybox_size=36.0,
        light_dimming_factor=0.8,
        texture_scale=0.5 if platform == "darwin" else 1.0,  # Reduce scale if in Mac
    )

    s = Simulator(mode="headless", image_width=1080, image_height=720, vertical_fov=60, rendering_settings=settings)

    random.seed(seed)
    scene = InteractiveIndoorScene(scene_name, texture_randomization=domain_rand, object_randomization=object_rand)

    s.import_scene(scene)

    # Load trajectory path
    traj_path = os.path.join(get_ig_scene_path(scene_name), "misc", "tour_cam_trajectory.txt")
    save_dir = os.path.join(get_ig_scene_path(scene_name), save_dir)
    os.makedirs(save_dir, exist_ok=True)
    tmp_dir = os.path.join(save_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    with open(traj_path, "r") as fp:
        points = [l.rstrip().split(",") for l in fp.readlines()]

    for _ in range(60):
        s.step()
    s.sync()

    for i in range(len(points)):
        if domain_rand and i % domain_rand_interval == 0:
            scene.randomize_texture()
        x, y, dir_x, dir_y = [float(p) for p in points[i]]
        z = 1.7
        tar_x = x + dir_x
        tar_y = y + dir_y
        tar_z = 1.4
        # cam_loc = np.array([x, y, z])
        s.renderer.set_camera([x, y, z], [tar_x, tar_y, tar_z], [0, 0, 1])

        with Profiler("Render"):
            frame = s.renderer.render(modes=("rgb"))
        img = Image.fromarray((255 * np.concatenate(frame, axis=1)[:, :, :3]).astype(np.uint8))
        img.save(os.path.join(tmp_dir, "{:05d}.png".format(i)))

    cmd = "ffmpeg -i {t}/%5d.png -y -an -c:v libx264 -crf 18 -preset veryslow -r 30 {s}/tour.mp4".format(
        t=tmp_dir, s=save_dir
    )
    subprocess.call(cmd, shell=True)
    cmd = "rm -r {}".format(tmp_dir)
    subprocess.call(cmd, shell=True)

    s.disconnect()


if __name__ == "__main__":
    main()
