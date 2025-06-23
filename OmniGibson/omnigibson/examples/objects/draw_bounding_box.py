import matplotlib.pyplot as plt
import torch as th

import omnigibson as og


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
        position=[-0.906661, -0.545106, 0.136824],
        orientation=[0, 0, 0.76040583, -0.6494482],
    )

    door_cfg = dict(
        type="DatasetObject",
        name="door",
        category="door",
        model="ohagsq",
        bounding_box=[1.528, 0.064, 1.299],
        position=[-2.0, 0, 0.70000001],
        orientation=[0, 0, -0.38268343, 0.92387953],
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
        position=th.tensor([-4.62785, -0.418575, 0.933943]),
        orientation=th.tensor([0.52196595, -0.4231939, -0.46640436, 0.5752612]),
    )

    # Add bounding boxes to camera sensor
    bbox_modalities = ["bbox_3d", "bbox_2d_loose", "bbox_2d_tight"]
    for bbox_modality in bbox_modalities:
        cam.add_modality(bbox_modality)

    # Take a few steps to let objects settle
    for i in range(100):
        env.step(th.empty(0))

    # Grab observations from viewer camera and write them to disk
    obs, _ = cam.get_obs()

    for bbox_modality in bbox_modalities:
        # Print out each of the modalities
        og.log.info(f"Observation modality {bbox_modality}:\n{obs[bbox_modality]}")

        # Also write the 2d loose bounding box to disk
        if "3d" not in bbox_modality:
            from omnigibson.utils.deprecated_utils import colorize_bboxes

            colorized_img = colorize_bboxes(
                bboxes_2d_data=obs[bbox_modality], bboxes_2d_rgb=obs["rgb"].cpu().numpy(), num_channels=4
            )
            fpath = f"{bbox_modality}_img.png"
            plt.imsave(fpath, colorized_img)
            og.log.info(f"Saving modality [{bbox_modality}] image to: {fpath}")

    # Always close environment down at end
    og.clear()


if __name__ == "__main__":
    main()
