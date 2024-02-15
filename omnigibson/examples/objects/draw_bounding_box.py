import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import colorsys
import random

import numpy as np
import omnigibson as og
import omnigibson.lazy as lazy


def main(random_selection=False, headless=False, short_exec=False):
    """
    Shows how to obtain the bounding box of an articulated object.
    Draws the bounding box around the loaded object, a cabinet, and writes the visualized image to disk at the
    current directory named 'bbox_2d_[loose / tight]_img.png'.

    NOTE: In the GUI, bounding boxes can be natively viewed by clicking on the sensor ((*)) icon at the top,
    and then selecting the appropriate bounding box modalities, and clicking "Show". See:

    https://docs.omniverse.nvidia.com/prod_extensions/prod_extensions/ext_replicator/visualization.html#the-visualizer
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Specify objects to load
    banana_cfg = dict(
        type="DatasetObject",
        name="banana",
        category="banana",
        model="vvyyyv",
        bounding_box=[0.643, 0.224, 0.269],
        position=[-0.906661, -0.545106,  0.136824],
        orientation=[0, 0, 0.76040583, -0.6494482],
    )

    door_cfg = dict(
        type="DatasetObject",
        name="door",
        category="door",
        model="ohagsq",
        bounding_box=[1.528, 0.064, 1.299],
        position=[-2.0, 0, 0.70000001],
        orientation=[0, 0, -0.38268343,  0.92387953],
    )

    # Create the scene config to load -- empty scene with a few objects
    cfg = {
        "scene": {
            "type": "Scene",
        },
        "objects": [banana_cfg, door_cfg],
    }

    # Create the environment
    env = og.Environment(configs=cfg)

    # Set camera to appropriate viewing pose
    cam = og.sim.viewer_camera
    cam.set_position_orientation(
        position=np.array([-4.62785 , -0.418575,  0.933943]),
        orientation=np.array([ 0.52196595, -0.4231939 , -0.46640436,  0.5752612 ]),
    )

    # Add bounding boxes to camera sensor
    bbox_modalities = ["bbox_3d", "bbox_2d_loose", "bbox_2d_tight"]
    for bbox_modality in bbox_modalities:
        cam.add_modality(bbox_modality)

    # Take a few steps to let objects settle
    for i in range(100):
        env.step(np.array([]))

    # Grab observations from viewer camera and write them to disk
    obs = cam.get_obs()[0]

    for bbox_modality in bbox_modalities:
        # Print out each of the modalities
        og.log.info(f"Observation modality {bbox_modality}:\n{obs[bbox_modality]}")

        # Also write the 2d loose bounding box to disk
        if "3d" not in bbox_modality:
            colorized_img = colorize_bboxes(bboxes_2d_data=obs[bbox_modality], bboxes_2d_rgb=obs["rgb"], num_channels=4)
            fpath = f"{bbox_modality}_img.png"
            plt.imsave(fpath, colorized_img)
            og.log.info(f"Saving modality [{bbox_modality}] image to: {fpath}")

    # Always close environment down at end
    env.close()
    
def random_colours(N, enable_random=True, num_channels=3):
    """
    Generate random colors.
    Generate visually distinct colours by linearly spacing the hue
    channel in HSV space and then convert to RGB space.
    """
    start = 0
    if enable_random:
        random.seed(10)
        start = random.random()
    hues = [(start + i / N) % 1.0 for i in range(N)]
    colours = [list(colorsys.hsv_to_rgb(h, 0.9, 1.0)) for i, h in enumerate(hues)]
    if num_channels == 4:
        for color in colours:
            color.append(1.0)
    if enable_random:
        random.shuffle(colours)
    return colours

def colorize_bboxes(bboxes_2d_data, bboxes_2d_rgb, num_channels=3):
    """Colorizes 2D bounding box data for visualization.


    Args:
        bboxes_2d_data (numpy.ndarray): 2D bounding box data from the sensor.
        bboxes_2d_rgb (numpy.ndarray): RGB data from the sensor to embed bounding box.
        num_channels (int): Specify number of channels i.e. 3 or 4.
    """
    semantic_id_list = []
    bbox_2d_list = []
    rgb_img = Image.fromarray(bboxes_2d_rgb)
    rgb_img_draw = ImageDraw.Draw(rgb_img)
    for bbox_2d in bboxes_2d_data:
        semantic_id_list.append(bbox_2d[0])
        bbox_2d_list.append(bbox_2d)
    semantic_id_list_np = np.unique(np.array(semantic_id_list))
    color_list = random_colours(len(semantic_id_list_np.tolist()), True, num_channels)
    for bbox_2d in bbox_2d_list:
        index = np.where(semantic_id_list_np == bbox_2d[0])[0][0]
        bbox_color = color_list[index]
        outline = (int(255 * bbox_color[0]), int(255 * bbox_color[1]), int(255 * bbox_color[2]))
        if num_channels == 4:
            outline = (
                int(255 * bbox_color[0]),
                int(255 * bbox_color[1]),
                int(255 * bbox_color[2]),
                int(255 * bbox_color[3]),
            )
        rgb_img_draw.rectangle([(bbox_2d[1], bbox_2d[2]), (bbox_2d[3], bbox_2d[4])], outline=outline, width=2)
    bboxes_2d_rgb = np.array(rgb_img)
    return bboxes_2d_rgb

if __name__ == "__main__":
    main()
