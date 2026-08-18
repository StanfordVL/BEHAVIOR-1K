import math
import omnigibson as og
import omnigibson.lazy as lazy
import torch as th
from omnigibson.sensors.vision_sensor import VisionSensor
from omnigibson.utils.ui_utils import create_module_logger
from typing import Any, Tuple, List

# Create module logger
log = create_module_logger(module_name=__name__)


class TiledVisionSensor:
    """
    Tiled vision sensor that leverages Omniverse Replicator to render tiled images from multiple cameras
    Args:
        envs (Iterable): Robot containers to create tiled sensors for -- any objects exposing a .robots list
            (e.g. the per-env Scene instances of a multi-env Environment), one per environment, in env order.
        tile_all_sensors (bool): Whether to tile all cameras in the envs or only those with matching names. Defaults to False.
            NOTE: If True, all cameras must have the same modalities and resolution.
    """

    # Registry of all created tiled sensors, used for cleanup on og.clear()
    TILED_SENSORS = []

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

        # Register for cleanup on og.clear()
        TiledVisionSensor.TILED_SENSORS.append(self)

    def remove(self) -> None:
        """Detach all annotators and destroy the tiled render products. Idempotent."""
        if self not in TiledVisionSensor.TILED_SENSORS:
            return
        TiledVisionSensor.TILED_SENSORS.remove(self)
        with og.sim.editing_usd():
            for sensor_name in self._annotators:
                for annotator in self._annotators[sensor_name].values():
                    # Passing an explicit list is bugged -- see VisionSensor._remove_modality_from_backend
                    try:
                        annotator.detach(self._render_product_paths[sensor_name])
                    except TypeError:
                        log.warning(f"Failed to cleanly detach tiled annotator for {sensor_name}; skipping")
        # Render so the syntheticdata graph settles before the render product is destroyed -- destroying
        # with pending graph edits leaves invalid nodes that break subsequent annotator detach calls
        og.sim.render()
        with og.sim.editing_usd():
            for render_product in self._render_product_paths.values():
                render_product.destroy()
        og.sim.render()
        self._annotators = dict()
        self._render_product_paths = dict()

    @classmethod
    def clear(cls) -> None:
        """Remove all tiled vision sensors. Called on og.clear() before scenes (and their cameras) are removed."""
        for sensor in tuple(cls.TILED_SENSORS):
            sensor.remove()

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
        self._output_warp_buffer = dict()
        self._reshape_dims = dict()
        self._num_tiles_x = dict()
        for sensor_name in self._camera_prims:
            self._output_buffer[sensor_name] = dict()
            self._output_warp_buffer[sensor_name] = dict()
            self._reshape_dims[sensor_name] = (
                self._camera_count(sensor_name=sensor_name),
                self._camera_resolution[sensor_name][1],
                self._camera_resolution[sensor_name][0],
            )
            self._num_tiles_x[sensor_name] = self._tiled_grid_shape(sensor_name=sensor_name)[0]
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
                    # unlike VisionSensor.get_obs(), the tiled path returns RAW Replicator segmentation
                    # IDs -- they are arbitrary, change between sessions, and are NOT remapped to OmniGibson's
                    # stable semantic/instance IDs (see VisionSensor._remap_semantic_segmentation). No id->name
                    # mapping info dict is available either. Do NOT train on these values until a batched remap
                    # is implemented; use the single-env per-sensor path for correct segmentation labels.
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
                # The output tensors are persistent. Cache their zero-copy Warp aliases instead of
                # recreating Python/Warp wrappers on every rendered frame.
                self._output_warp_buffer[sensor_name][modality] = lazy.warp.from_torch(
                    self._output_buffer[sensor_name][modality]
                )

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
                    dim=self._reshape_dims[sensor_name],
                    inputs=[
                        tiled_data_buffer.flatten(),
                        self._output_warp_buffer[sensor_name][modality],
                        *list(self._output_buffer[sensor_name][modality].shape[1:]),  # height, width, num_channels
                        self._num_tiles_x[sensor_name],
                    ],
                    device="cuda:0",
                )

        return self._output_buffer
