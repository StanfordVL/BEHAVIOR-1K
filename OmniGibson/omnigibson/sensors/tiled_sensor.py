import math

import torch as th

import omnigibson.lazy as lazy
from omnigibson.sensors.vision_sensor import VisionSensor


class TiledSensor:
    """
    Args:
        modalities (list of str): Modality(s) supported by this sensor. Default is "rgb", can also include "depth".
    """

    def __init__(
        self,
        modalities=["rgb"],
    ):
        self.modalities = modalities
        self._camera_resolution = None
        camera_prim_paths = []
        for sensor in VisionSensor.SENSORS.values():
            if self._camera_resolution is None:
                self._camera_resolution = (sensor.image_width, sensor.image_height)
            else:
                assert self._camera_resolution == (
                    sensor.image_width,
                    sensor.image_height,
                ), "All cameras must have the same resolution!"
            camera_prim_paths.append(sensor.prim_path)
        stage = lazy.omni.usd.get_context().get_stage()
        self._camera_prims = []
        for path in camera_prim_paths:
            camera_prim = stage.GetPrimAtPath(path)
            self._camera_prims.append(lazy.pxr.UsdGeom.Camera(camera_prim))
        tiled_camera = lazy.omni.replicator.core.create.tiled_sensor(
            cameras=camera_prim_paths,
            camera_resolution=self._camera_resolution,
            tiled_resolution=self._tiled_img_shape(),
            output_types=self.modalities,
        )
        self._render_product_path = lazy.omni.replicator.core.create.render_product(
            camera=tiled_camera, resolution=self._tiled_img_shape()
        )
        self._annotator = lazy.omni.replicator.core.AnnotatorRegistry.get_annotator(
            "RtxSensorGpu", device="cuda:0", do_array_copy=False
        )
        self._annotator.attach([self._render_product_path])

        self._output_buffer = dict()
        if "rgb" in self.modalities:
            self._output_buffer["rgb"] = th.zeros(
                (self._camera_count(), self._camera_resolution[1], self._camera_resolution[0], 3), device="cuda:0"
            ).contiguous()
        if "depth" in self.modalities:
            self._output_buffer["depth"] = th.zeros(
                (self._camera_count(), self._camera_resolution[1], self._camera_resolution[0], 1), device="cuda:0"
            ).contiguous()

        super().__init__()

    def _camera_count(self):
        return len(self._camera_prims)

    def _tiled_grid_shape(self):
        cols = round(math.sqrt(self._camera_count()))
        rows = math.ceil(self._camera_count() / cols)
        return (cols, rows)

    def _tiled_img_shape(self):
        cols, rows = self._tiled_grid_shape()
        width, height = self._camera_resolution
        return (width * cols, height * rows)

    def get_obs(self):
        # TODO: somehow isaac 4.1.0 introduced a bug: this always return a warp array on cpu instead of gpu, even when explicitly specifying device="cuda:0"
        tiled_data = self._annotator.get_data().to(device="cuda:0")
        breakpoint()
        from omnigibson.utils.deprecated_utils import reshape_tiled_image

        for modality in self.modalities:
            lazy.warp.launch(
                kernel=reshape_tiled_image,
                dim=(self._camera_count(), self._camera_resolution[1], self._camera_resolution[0]),
                inputs=[
                    tiled_data,
                    lazy.warp.from_torch(self._output_buffer[modality]),  # zero-copy alias
                    *list(self._output_buffer[modality].shape[1:]),  # height, width, num_channels
                    self._tiled_grid_shape()[0],  # num_tiles_x
                    (
                        self._output_buffer["rgb"].numel() if "depth" in self.modalities else 0
                    ),  # rgb always comes first; needs an offset for depth
                ],
                device="cuda:0",
            )
        return self._output_buffer
