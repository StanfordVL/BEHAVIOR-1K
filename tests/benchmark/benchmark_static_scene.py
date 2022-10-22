import os
import time

import matplotlib.pyplot as plt

import omnigibson
from omnigibson.robots.turtlebot import Turtlebot
from omnigibson.scenes.gibson_indoor_scene import StaticIndoorScene
from omnigibson.simulator import Simulator
from omnigibson.utils.config_utils import parse_config


def benchmark(render_to_tensor=False, resolution=512):
    config = parse_config(os.path.join(os.path.dirname(__file__), "..", "test.yaml"))
    if render_to_tensor:
        mode = "headless_tensor"
    else:
        mode = "headless"
    s = Simulator(mode=mode, image_width=resolution, image_height=resolution)
    scene = StaticIndoorScene("Rs", build_graph=True, pybullet_load_texture=True)
    s.import_scene(scene)
    turtlebot = Turtlebot()
    s.import_object(turtlebot)

    n_frame = 500
    start = time.time()
    for i in range(n_frame):
        turtlebot.apply_action([0.1, 0.1])
        s.step()
        rgb = s.renderer.render_robot_cameras(modes=("rgb"))

    physics_render_elapsed = time.time() - start
    physics_render_fps = n_frame / physics_render_elapsed
    print(
        "physics simulation + rendering rgb, resolution {}, render_to_tensor {}: {} fps".format(
            resolution, render_to_tensor, physics_render_fps
        )
    )

    start = time.time()
    for i in range(n_frame):
        rgb = s.renderer.render_robot_cameras(modes=("rgb"))

    render_elapsed = time.time() - start
    rgb_fps = n_frame / render_elapsed
    print(
        "Rendering rgb, resolution {}, render_to_tensor {}: {} fps".format(
            resolution, render_to_tensor, n_frame / render_elapsed
        )
    )

    start = time.time()
    for i in range(n_frame):
        rgb = s.renderer.render_robot_cameras(modes=("3d"))

    render_elapsed = time.time() - start
    pc_fps = n_frame / render_elapsed
    print(
        "Rendering 3d, resolution {}, render_to_tensor {}: {} fps".format(
            resolution, render_to_tensor, n_frame / render_elapsed
        )
    )

    start = time.time()
    for i in range(n_frame):
        rgb = s.renderer.render_robot_cameras(modes=("normal"))

    normal_fps = n_frame / render_elapsed
    render_elapsed = time.time() - start
    print(
        "Rendering normal, resolution {}, render_to_tensor {}: {} fps".format(
            resolution, render_to_tensor, n_frame / render_elapsed
        )
    )
    plt.figure()
    plt.bar([0, 1, 2, 3], [physics_render_fps, rgb_fps, pc_fps, normal_fps], color="g")
    plt.xticks([0, 1, 2, 3], ["sim+render", "rgb", "3d", "normal"])
    plt.ylabel("fps")
    plt.xlabel("mode")
    plt.title("Static Scene Benchmark, resolution {}, to_tensor {}".format(resolution, render_to_tensor))
    plt.savefig("static_scene_benchmark_res{}_tensor{}.pdf".format(resolution, render_to_tensor))

    s.disconnect()


def main():
    benchmark(render_to_tensor=True, resolution=512)
    benchmark(render_to_tensor=False, resolution=512)


if __name__ == "__main__":
    main()
