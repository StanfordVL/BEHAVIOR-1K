import pathlib
import sys

import cv2
import numpy as np
import torch as th
from PIL import Image
import os

import omnigibson as og
from omnigibson.macros import gm
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject

CENTER_HEIGHT = 1.0  # meters

IMG_WIDTH = 1280
IMG_HEIGHT = 720


def main():
    dataset_path = sys.argv[1]
    output_path = sys.argv[2]
    batch = sys.argv[3:]

    # Set the dataset path
    gm.DATASET_PATH = dataset_path

    # Create the scene config to load -- empty scene
    cfg = {
        "scene": {
            "type": "Scene",
            "floor_plane_visible": False,
        },
        "render": {
            "viewer_width": IMG_WIDTH,
            "viewer_height": IMG_HEIGHT,
        },
    }

    # Create the environment
    env = og.Environment(configs=cfg)

    # Make it brighter
    dome_light = og.sim.skybox
    dome_light.intensity = 0.25e4

    for item in batch:
        obj_category, obj_model = pathlib.Path(item).parts[-2:]
        og.sim.stop()
        obj = DatasetObject(
            name="obj",
            category=obj_category,
            model=obj_model,
            visual_only=True,
        )
        env.scene.add_object(obj)
        # Make sure the object fits in a box of unit size 1
        obj.scale = (th.ones(3) / obj.aabb_extent).min()
        og.sim.play()

        # Move the object so that its center is at [0, 0, 1]
        camera_dist = obj.aabb_extent[:2].max() * 4.0
        height_range = th.maximum(
            th.sin(th.deg2rad(th.tensor(45.0))) * camera_dist,  # Either 45 degrees
            obj.aabb_extent[2] * 2.0,  # Or twice the object height
        )
        center_offset = obj.get_position() - obj.aabb_center
        target_pos = th.tensor([0.0, 0.0, CENTER_HEIGHT])

        # Place the center at [0, 0, CENTER_HEIGHT]
        obj.set_position_orientation(
            position=center_offset + target_pos, orientation=[0, 0, 0, 1]
        )
        og.sim.viewer_camera.set_position_orientation(
            position=th.tensor([camera_dist, 0, CENTER_HEIGHT]),
            orientation=T.euler2quat(th.tensor([th.pi / 2.0, 0.0, th.pi / 2.0])),
        )

        og.sim.step()
        for _ in range(10):
            og.sim.render()

        total_time_ms = 6000
        fps = 30
        total_frames = (total_time_ms * fps) // 1000
        total_horizontal_revolutions = 3
        total_vertical_revolutions = 2
        total_joint_loops = 1.5
        pitch_angular_velocity = th.tensor(
            2 * th.pi * total_vertical_revolutions / total_frames
        )
        yaw_angular_velocity = th.tensor(
            2 * th.pi * total_horizontal_revolutions / total_frames
        )
        joint_angular_velocity = th.tensor(2 * th.pi * total_joint_loops / total_frames)

        # Make the collision meshes also visible by just the visibility argument
        # (if they have the "guide" purpose, they are not visible even if the visibility is True)
        for link in obj.links.values():
            for mesh in link.visual_meshes.values():
                # Set the purpose to "default" for all visual meshes and all meta meshes except containers.
                mesh.purpose = (
                    "default"
                    if not link.is_meta_link or link.meta_link_type != "container"
                    else "guide"
                )
                mesh.visible = not link.is_meta_link
            for mesh in link.collision_meshes.values():
                mesh.purpose = "default"
                mesh.visible = False

        # Open the video file
        filename = os.path.join(output_path, f"{obj_model}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(filename, fourcc, 30.0, (IMG_WIDTH, IMG_HEIGHT))

        # Open/close joints for 3 times (lower_limit -> upper_limit -> lower_limit, 3 times)
        for view_collision_mesh in [False, True]:
            for link in obj.links.values():
                for mesh in link.visual_meshes.values():
                    mesh.visible = not view_collision_mesh
                for mesh in link.collision_meshes.values():
                    mesh.visible = view_collision_mesh

            for i in range(total_frames):
                camera_yaw = th.tensor(yaw_angular_velocity * i)
                camera_pos = th.tensor(
                    [
                        th.cos(camera_yaw) * camera_dist,
                        th.sin(camera_yaw) * camera_dist,
                        th.sin(pitch_angular_velocity * i) * height_range
                        + CENTER_HEIGHT,
                    ]
                )
                camera_pitch = th.arctan((CENTER_HEIGHT - camera_pos[2]) / camera_dist)
                og.sim.viewer_camera.set_position_orientation(
                    position=camera_pos,
                    orientation=T.euler2quat(
                        th.tensor(
                            [camera_pitch + th.pi / 2.0, 0.0, camera_yaw + th.pi / 2.0]
                        )
                    ),
                )
                if obj.n_dof > 0:
                    j_frac = th.sin(joint_angular_velocity * i)
                    obj.set_joint_positions(
                        positions=j_frac * th.ones(obj.n_dof),
                        normalized=True,
                        drive=False,
                    )

                og.sim.step()
                og.sim.render()
                og.sim.render()

                # Get the image as a 3-channel uint8 rgb image
                image = (
                    og.sim.viewer_camera.get_obs()[0]["rgb"]
                    .cpu()
                    .numpy()[:, :, :3]
                    .astype(np.uint8)
                )

                # Convert to BGR format for OpenCV
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                video_writer.write(image)

        video_writer.release()
        with open(filename.replace(".mp4", ".success"), "w") as f:
            pass

        env.scene.remove_object(obj)

    # Shut down at the end
    og.shutdown()


if __name__ == "__main__":
    main()
