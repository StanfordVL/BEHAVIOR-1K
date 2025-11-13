import math
import omnigibson.lazy as lazy
import torch as th
from omnigibson.sensors.vision_sensor import VisionSensor
from typing import Any, Tuple, List


class TiledVisionSensor:
    """
    Tiled vision sensor that leverages Omniverse Replicator to render tiled images from multiple cameras
    Args:
        envs (Iterable[og.Environment]): List of environments to create tiled sensors for.
        tile_all_sensors (bool): Whether to tile all cameras in the envs or only those with matching names. Defaults to False.
            NOTE: If True, all cameras must have the same modalities and resolution.
    """

    def __init__(self, envs: List[Any], tile_all_sensors: bool = False) -> None:
        camera_prim_paths = dict()
        self.modalities = dict()
        self._camera_resolution = dict()
        self._camera_prims = dict()
        self._annotators = dict()

        stage = lazy.omni.usd.get_context().get_stage()
        for env in envs:
            for robot in env.robots:
                for sensor_name, sensor in robot.sensors.items():
                    if tile_all_sensors:
                        sensor_name = "sensor"
                    if isinstance(sensor, VisionSensor):
                        if sensor_name not in camera_prim_paths:
                            camera_prim_paths[sensor_name] = []
                            self.modalities[sensor_name] = sensor.modalities
                            self._camera_resolution[sensor_name] = (sensor.image_width, sensor.image_height)
                            self._camera_prims[sensor_name] = []
                        assert (
                            sensor.modalities == self.modalities[sensor_name]
                        ), f"All sensors named {sensor_name} must have the same modalities!"
                        assert self._camera_resolution[sensor_name] == (
                            sensor.image_width,
                            sensor.image_height,
                        ), f"All sensors named {sensor_name} must have the same resolution!"
                        camera_prim_paths[sensor_name].append(sensor.prim_path)
                        camera_prim = stage.GetPrimAtPath(sensor.prim_path)
                        self._camera_prims[sensor_name].append(lazy.pxr.UsdGeom.Camera(camera_prim))
        self._render_product_paths = {
            sensor_name: lazy.omni.replicator.core.create.render_product_tiled(
                cameras=camera_prim_paths[sensor_name], tile_resolution=self._camera_resolution[sensor_name]
            )
            for sensor_name in camera_prim_paths
        }
        self._create_annotators()
        # Attach the annotator to the render product
        for sensor_name in self._annotators:
            for modality in self._annotators[sensor_name]:
                self._annotators[sensor_name][modality].attach([self._render_product_paths[sensor_name]])

        # Create internal buffers
        self._create_buffers()

    def _create_annotators(self) -> None:
        self._annotators = dict()
        for sensor_name in self.modalities:
            self._annotators[sensor_name] = dict()
            for modality in self.modalities[sensor_name]:
                annotator_type = VisionSensor.RAW_SENSOR_TYPES[modality]
                annotator = lazy.omni.replicator.core.AnnotatorRegistry.get_annotator(
                    annotator_type, device="cuda:0", do_array_copy=False
                )
                self._annotators[sensor_name][modality] = annotator

    def _create_buffers(self, device: str = "cuda:0") -> None:
        self._output_buffer = dict()
        for sensor_name in self._camera_prims:
            self._output_buffer[sensor_name] = dict()
            for modality in self.modalities[sensor_name]:
                if modality == "rgb":
                    self._output_buffer[sensor_name][modality] = th.zeros(
                        (
                            self._camera_count(sensor_name=sensor_name),
                            self._camera_resolution[sensor_name][1],
                            self._camera_resolution[sensor_name][0],
                            4,
                        ),
                        device=device,
                        dtype=th.uint8,
                    ).contiguous()
                elif modality == "depth" or modality == "depth_linear":
                    self._output_buffer[sensor_name][modality] = th.zeros(
                        (
                            self._camera_count(sensor_name=sensor_name),
                            self._camera_resolution[sensor_name][1],
                            self._camera_resolution[sensor_name][0],
                            1,
                        ),
                        device=device,
                        dtype=th.float32,
                    ).contiguous()
                elif modality == "seg_semantic" or modality == "seg_instance" or modality == "seg_instance_id":
                    self._output_buffer[sensor_name][modality] = th.zeros(
                        (
                            self._camera_count(sensor_name=sensor_name),
                            self._camera_resolution[sensor_name][1],
                            self._camera_resolution[sensor_name][0],
                            1,
                        ),
                        device=device,
                        dtype=th.uint32,
                    ).contiguous()
                elif modality == "flow":
                    self._output_buffer[sensor_name][modality] = th.zeros(
                        (
                            self._camera_count(sensor_name=sensor_name),
                            self._camera_resolution[sensor_name][1],
                            self._camera_resolution[sensor_name][0],
                            2,
                        ),
                        device=device,
                        dtype=th.float32,
                    ).contiguous()
                elif modality == "normal":
                    self._output_buffer[sensor_name][modality] = th.zeros(
                        (
                            self._camera_count(sensor_name=sensor_name),
                            self._camera_resolution[sensor_name][1],
                            self._camera_resolution[sensor_name][0],
                            3,
                        ),
                        device=device,
                        dtype=th.uint8,
                    ).contiguous()
                else:
                    raise ValueError(f"Unsupported modality {modality} for tiled vision sensor!")

    def _camera_count(self, sensor_name: str) -> int:
        return len(self._camera_prims[sensor_name])

    def _tiled_grid_shape(self, sensor_name: str) -> Tuple[int, int]:
        cols = math.ceil(math.sqrt(self._camera_count(sensor_name=sensor_name)))
        rows = math.ceil(self._camera_count(sensor_name=sensor_name) / cols)
        return (cols, rows)

    def _tiled_img_shape(self, sensor_name: str) -> Tuple[int, int]:
        cols, rows = self._tiled_grid_shape(sensor_name=sensor_name)
        width, height = self._camera_resolution[sensor_name]
        return (width * cols, height * rows)

    def get_obs(self):
        from omnigibson.utils.deprecated_utils import reshape_tiled_image

        for sensor_name in self._annotators:
            for modality in self.modalities[sensor_name]:
                tiled_data_buffer = self._annotators[sensor_name][modality].get_data().to("cuda:0")
                # For flow, we only require the first two channels of the tiled buffer
                # Note: Not doing this breaks the alignment of the data (check: https://github.com/isaac-sim/IsaacLab/issues/2003)
                if modality == "flow":
                    tiled_data_buffer = tiled_data_buffer[:, :, :2].contiguous()
                lazy.warp.launch(
                    kernel=reshape_tiled_image,
                    dim=(
                        self._camera_count(sensor_name=sensor_name),
                        self._camera_resolution[sensor_name][1],
                        self._camera_resolution[sensor_name][0],
                    ),
                    inputs=[
                        tiled_data_buffer.flatten(),
                        lazy.warp.from_torch(self._output_buffer[sensor_name][modality]),  # zero-copy alias
                        *list(self._output_buffer[sensor_name][modality].shape[1:]),  # height, width, num_channels
                        self._tiled_grid_shape(sensor_name=sensor_name)[0],  # num_tiles_x
                    ],
                    device="cuda:0",
                )

        return self._output_buffer
