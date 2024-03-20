import numpy as np

import omnigibson as og
import omnigibson.lazy as lazy
from omnigibson.macros import gm
from omnigibson.utils.asset_utils import get_available_g_scenes, get_available_og_scenes
from omnigibson.utils.ui_utils import choose_from_options, KeyboardEventHandler
from PIL import Image, ImageDraw
from omnigibson.utils.constants import semantic_class_name_to_id


def main():

    scenes = get_available_og_scenes()

    background_id = semantic_class_name_to_id()["background"]

    for scene_model in scenes:
        print(f"scene model: {scene_model}")

        if og.sim is not None:
            og.sim.stop()

        config = {
            "scene": {
                "type": "InteractiveTraversableScene",
                "scene_model": scene_model,
            }
        }

        env = og.Environment(configs=config)

        env.reset()

        # og.sim.viewer_width = 1920
        # og.sim.viewer_height = 1920

        cam_mover = og.sim.enable_viewer_camera_teleoperation()
        og.sim.viewer_camera.add_modality("seg_semantic")

        # terminate = [False]

        # KeyboardEventHandler.add_keyboard_callback(
        #     key=lazy.carb.input.KeyboardInput.ESCAPE,
        #     callback_fn=lambda: terminate.__setitem__(0, True),
        # )

        # print("Press ESC to take a birds-eye view screenshot")

        # while not terminate[0]:
        #     env.step([])

        for _ in range(10):
            og.sim.step()

        birds_eye_view = og.sim.viewer_camera.get_obs()[0]["rgb"]
        semantic_image = og.sim.viewer_camera.get_obs()[0]["seg_semantic"]

        # # remove background
        # # for all the pixels in birds_eye_view, if the pixel is background, set it to white
        # birds_eye_view[semantic_image == background_id] = 255

        # img = Image.fromarray(birds_eye_view)
        # img.save(f"/scr/home/yinhang/birds-eye-views/{scene_model}.png")

        # print("Press ESC to take a screenshot of the scene")

        # terminate = [False]

        # while not terminate[0]:
        #     env.step([])

        # screenshot = og.sim.viewer_camera.get_obs()[0]['rgb']

        # # save this screenshot as a png file with PIL
        # img = Image.fromarray(screenshot)
        # img.save(f"/scr/home/yinhang/scene_assets/{scene_model}.png")

        # # env.close()
        og.sim.clear()

    env.close()


if __name__ == "__main__":
    main()
