import os
import numpy as np
import time
import yaml

import omnigibson as og
from omnigibson.macros import gm
from omnigibson.learning.utils.obs_utils import create_video_writer, write_video

NUM_STEPS = 100
NUM_ENVS = 5
MAX_NUM_ENVS_TO_RECORD_VIDEO = 10


def main(random_selection=False, headless=False, short_exec=False):
    # Load the config
    gm.RENDER_VIEWER_CAMERA = False
    gm.ENABLE_FLATCACHE = True
    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_TRANSITION_RULES = False
    gm.ENABLE_OBJECT_STATES = False

    config_filename = os.path.join(og.example_config_path, "franka_vector_env.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    config["scene"]["load_object_categories"] = ["floors", "walls", "coffee_table"]

    # store a video
    if MAX_NUM_ENVS_TO_RECORD_VIDEO > 0:
        video_writer = create_video_writer(
            fpath="tiled_output.mp4", resolution=(224 * min(NUM_ENVS, MAX_NUM_ENVS_TO_RECORD_VIDEO), 448)
        )

    # Load the environment
    vec_env = og.VectorEnvironment(NUM_ENVS, config)

    max_iterations = 10 if not short_exec else 1
    for _ in range(max_iterations):
        times = []
        for _ in range(NUM_STEPS):
            start_time = time.time()
            actions = []
            for e in vec_env.envs:
                actions.append(e.action_space.sample())
            obs = vec_env.step(actions)[0]
            times.append(time.time() - start_time)
            if MAX_NUM_ENVS_TO_RECORD_VIDEO > 0:
                write_video(
                    obs=np.expand_dims(
                        np.concatenate(
                            [
                                obs[i]["robot:eef_link:Camera:0::rgb"].cpu().numpy()
                                for i in range(min(NUM_ENVS, MAX_NUM_ENVS_TO_RECORD_VIDEO))
                            ]
                        ),
                        axis=0,
                    ),
                    video_writer=video_writer,
                    batch_size=1,
                    mode="rgb",
                )
        fps = NUM_STEPS / np.sum(times)
        effective_fps = fps * NUM_ENVS
        print("fps", fps)
        print("effective fps", effective_fps)

    # close video writer
    if MAX_NUM_ENVS_TO_RECORD_VIDEO > 0:
        (container, stream) = video_writer
        for packet in stream.encode():
            container.mux(packet)
        container.close()
    # Always close the environment at the end
    og.shutdown()


if __name__ == "__main__":
    main()
