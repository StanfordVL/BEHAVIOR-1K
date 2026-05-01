import os

from concurrent.futures import as_completed
import fs.path
from fs.copy import copy_fs
from fs.osfs import OSFS
from fs.tempfs import TempFS
from fs.zipfs import ZipFS
import tqdm

from b1k_pipeline.utils import (
    PipelineFS,
    TMP_DIR,
    launch_cluster,
    submit_og_task,
)


WORKER_COUNT = 6

CENTER_HEIGHT = 1.0  # meters
IMG_WIDTH = 1280
IMG_HEIGHT = 720

OG_MACROS = {
    "HEADLESS": True,
    "USE_ENCRYPTED_ASSETS": True,
}


def process_obj(dataset_path, out_path, obj_path):
    """Render a turntable video for one object on a worker."""
    import pathlib

    import cv2
    import numpy as np
    import torch as th

    import omnigibson as og
    from omnigibson.macros import gm
    import omnigibson.utils.transform_utils as T
    from omnigibson.objects.dataset_object import DatasetObject

    gm.DATASET_PATH = dataset_path

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
    env = og.Environment(configs=cfg)
    og.sim.skybox.intensity = 0.25e4

    obj_category, obj_model = pathlib.Path(obj_path).parts[-2:]
    og.sim.stop()
    obj = DatasetObject(
        name="obj",
        category=obj_category,
        model=obj_model,
        visual_only=True,
    )
    env.scene.add_object(obj)
    obj.scale = (th.ones(3) / obj.aabb_extent).min()
    og.sim.play()

    camera_dist = obj.aabb_extent[:2].max() * 4.0
    height_range = th.maximum(
        th.sin(th.deg2rad(th.tensor(45.0))) * camera_dist,  # Either 45 degrees
        obj.aabb_extent[2] * 2.0,  # Or twice the object height
    )
    center_offset = obj.get_position() - obj.aabb_center
    target_pos = th.tensor([0.0, 0.0, CENTER_HEIGHT])

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
            mesh.purpose = (
                "default"
                if not link.is_meta_link or link.meta_link_type != "container"
                else "guide"
            )
            mesh.visible = not link.is_meta_link
        for mesh in link.collision_meshes.values():
            mesh.purpose = "default"
            mesh.visible = False

    filename = os.path.join(out_path, f"{obj_model}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(filename, fourcc, 30.0, (IMG_WIDTH, IMG_HEIGHT))

    try:
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

                image = (
                    og.sim.viewer_camera.get_obs()[0]["rgb"]
                    .cpu()
                    .numpy()[:, :, :3]
                    .astype(np.uint8)
                )
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                video_writer.write(image)
    finally:
        video_writer.release()

    with open(filename.replace(".mp4", ".success"), "w") as f:
        pass


def main():
    with PipelineFS() as pipeline_fs:
        with (
            ZipFS(pipeline_fs.open("artifacts/og_dataset.zip", "rb")) as dataset_zip_fs,
            TempFS(temp_dir=str(TMP_DIR)) as dataset_fs,
            OSFS(
                pipeline_fs.makedirs(
                    "artifacts/pipeline/object_images", recreate=True
                ).getsyspath("/")
            ) as out_temp_fs,
        ):
            # Copy everything over to the dataset FS
            print("Copy everything over to the dataset FS...")
            objdir_glob = list(dataset_zip_fs.glob("objects/*/*/"))
            for item in tqdm.tqdm(objdir_glob):
                if (
                    dataset_zip_fs.opendir(item.path).glob("usd/*.usd").count().files
                    == 0
                ):
                    continue
                objdir_normalized = fs.path.normpath(item.path)
                obj_id = fs.path.basename(objdir_normalized)
                if out_temp_fs.exists(f"{obj_id}.success"):
                    continue
                copy_fs(
                    dataset_zip_fs.opendir(item.path), dataset_fs.makedirs(item.path)
                )

            print("Launching cluster...")
            executor = launch_cluster(WORKER_COUNT, og_macros=OG_MACROS)

            object_glob = [
                fs.path.normpath(x.path) for x in dataset_fs.glob("objects/*/*/")
            ]

            print("Queueing objects.")
            print("Total count: ", len(object_glob))
            futures = {}
            for obj_path in object_glob:
                worker_future = submit_og_task(
                    executor,
                    process_obj,
                    dataset_fs.getsyspath("/"),
                    out_temp_fs.getsyspath("/"),
                    obj_path,
                )
                futures[worker_future] = obj_path

            print("Queued all objects. Waiting for them to finish...")
            for future in tqdm.tqdm(as_completed(futures.keys()), total=len(futures)):
                obj_path = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Object {obj_path} raised: {e}")

                obj_id = fs.path.basename(obj_path)
                if not out_temp_fs.exists(f"{obj_id}.success"):
                    print(f"Failed: {obj_path}")

        print("Archiving results...")

        # At this point, out_temp_fs's contents will be zipped. Save the success file.
        pipeline_fs.pipeline_output().touch("generate_images.success")


if __name__ == "__main__":
    main()
