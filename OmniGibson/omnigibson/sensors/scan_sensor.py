import math
from collections.abc import Iterable

import cv2
import torch as th

import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.sensors.sensor_base import BaseSensor
from omnigibson.utils.constants import OccupancyGridState
from omnigibson.utils.numpy_utils import NumpyTypes
from omnigibson.utils.python_utils import classproperty


class ScanSensor(BaseSensor):
    """
    General 2D LiDAR range sensor and occupancy grid sensor.

    Args:
        relative_prim_path (str): Scene-local prim path of the Sensor to encapsulate or create.
        name (str): Name for the object. Names need to be unique per scene.
        modalities (str or list of str): Modality(s) supported by this sensor. Default is "all", which corresponds
            to all modalities being used. Otherwise, valid options should be part of cls.all_modalities.
            For this scan sensor, this includes any of:
                {scan, occupancy_grid}
            Note that in order for "occupancy_grid" to be used, "scan" must also be included.
        enabled (bool): Whether this sensor should be enabled by default
        noise (None or BaseSensorNoise): If specified, sensor noise model to apply to this sensor.
        load_config (None or dict): If specified, should contain keyword-mapped values that are relevant for
            loading this sensor's prim at runtime.
        min_range (float): Minimum range to sense in meters
        max_range (float): Maximum range to sense in meters
        horizontal_fov (float): Field of view of sensor, in degrees
        vertical_fov (float): Field of view of sensor, in degrees
        yaw_offset (float): Degrees for offsetting this sensors horizontal FOV.
            Useful in cases where this sensor's forward direction is different than expected
        horizontal_resolution (float): Degrees in between each horizontal scan hit
        vertical_resolution (float): Degrees in between each vertical scan hit
        rotation_rate (float): How fast the range sensor is rotating, in rotations per sec. Set to 0 for all scans
            be to hit at once
        draw_points (bool): Whether to draw the points hit by this sensor
        draw_lines (bool): Whether to draw the lines representing the scans from this sensor
        occupancy_grid_resolution (int): How many discretized nodes in the occupancy grid. This will specify the
            height == width of the map
        occupancy_grid_range (float): Range of the occupancy grid, in meters
        occupancy_grid_inner_radius (float): Inner range of the occupancy grid that will assumed to be empty, in meters
        occupancy_grid_local_link (None or XFormPrim): XForm prim that represents the "origin" of any generated
            occupancy grid, e.g.: if this scan sensor is attached to a robot, then this should possibly be the base link
            for that robot. If None is specified, then this will default to this own sensor's frame as the origin.
    """

    def __init__(
        self,
        relative_prim_path,
        name,
        modalities="all",
        enabled=True,
        noise=None,
        load_config=None,
        # Basic LIDAR kwargs
        min_range=0.05,
        max_range=10.0,
        horizontal_fov=360.0,
        vertical_fov=1.0,
        yaw_offset=0.0,
        horizontal_resolution=1.0,
        vertical_resolution=1.0,
        rotation_rate=0.0,
        draw_points=False,
        draw_lines=False,
        # Occupancy Grid kwargs
        occupancy_grid_resolution=128,
        occupancy_grid_range=5.0,
        occupancy_grid_inner_radius=0.5,
        occupancy_grid_local_link=None,
    ):
        # Store settings
        self.occupancy_grid_resolution = occupancy_grid_resolution
        self.occupancy_grid_range = occupancy_grid_range
        self.occupancy_grid_inner_radius = int(
            occupancy_grid_inner_radius * occupancy_grid_resolution / occupancy_grid_range
        )
        self.occupancy_grid_local_link = self if occupancy_grid_local_link is None else occupancy_grid_local_link

        # Create variables that will be filled in at runtime
        self._rs = None  # Range sensor interface, analagous to others, e.g.: dynamic control interface

        # Create load config from inputs
        load_config = dict() if load_config is None else load_config
        load_config["min_range"] = min_range
        load_config["max_range"] = max_range
        load_config["horizontal_fov"] = horizontal_fov
        load_config["vertical_fov"] = vertical_fov
        load_config["yaw_offset"] = yaw_offset
        load_config["horizontal_resolution"] = horizontal_resolution
        load_config["vertical_resolution"] = vertical_resolution
        load_config["rotation_rate"] = rotation_rate
        load_config["draw_points"] = draw_points
        load_config["draw_lines"] = draw_lines

        # Sanity check modalities -- if we're using occupancy_grid without scan modality, raise an error
        if isinstance(modalities, Iterable) and not isinstance(modalities, str) and "occupancy_grid" in modalities:
            assert "scan" in modalities, "'scan' modality must be included in order to get occupancy_grid modality!"

        # Run super method
        super().__init__(
            relative_prim_path=relative_prim_path,
            name=name,
            modalities=modalities,
            enabled=enabled,
            noise=noise,
            load_config=load_config,
        )

    def _load(self):
        # Define a LIDAR prim at the current stage
        result, lidar = lazy.omni.kit.commands.execute("RangeSensorCreateLidar", path=self.prim_path)

        return lidar.GetPrim()

    def _post_load(self):
        # run super first
        super()._post_load()

        # Set all the lidar kwargs
        self.min_range = self._load_config["min_range"]
        self.max_range = self._load_config["max_range"]
        self.horizontal_fov = self._load_config["horizontal_fov"]
        self.vertical_fov = self._load_config["vertical_fov"]
        self.yaw_offset = self._load_config["yaw_offset"]
        self.horizontal_resolution = self._load_config["horizontal_resolution"]
        self.vertical_resolution = self._load_config["vertical_resolution"]
        self.rotation_rate = self._load_config["rotation_rate"]
        self.draw_points = self._load_config["draw_points"]
        self.draw_lines = self._load_config["draw_lines"]

    def _initialize(self):
        # run super first
        super()._initialize()

        # Initialize lidar sensor interface
        self._rs = lazy.isaacsim.sensors.physx._range_sensor.acquire_lidar_sensor_interface()

    @property
    def _obs_space_mapping(self):
        # Set the remaining modalities' values
        # (obs modality, shape, low, high)
        obs_space_mapping = dict(
            scan=((self.n_horizontal_rays, self.n_vertical_rays), 0.0, 1.0, NumpyTypes.FLOAT32),
            occupancy_grid=(
                (self.occupancy_grid_resolution, self.occupancy_grid_resolution, 1),
                0.0,
                1.0,
                NumpyTypes.FLOAT32,
            ),
        )

        return obs_space_mapping

    def get_local_occupancy_grid(self, scan):
        """
        Get local occupancy grid based on current 1D scan

        Args:
            scan (array): 1D LiDAR scan

        Returns:
            2D-array: (occupancy_grid_resolution, occupancy_grid_resolution)-sized numpy array of the local occupancy grid
        """
        # Run sanity checks first
        assert "occupancy_grid" in self._modalities, "Occupancy grid is not enabled for this range sensor!"
        assert self.n_vertical_rays == 1, "Occupancy grid is only valid for a 1D range sensor (n_vertical_rays = 1)!"

        # Calculate the number of points
        num_points = math.ceil(self.horizontal_fov / self.horizontal_resolution)

        # Generate angles using linspace
        angles = th.linspace(
            -th.deg2rad(th.tensor([self.horizontal_fov / 2])).item(),
            th.deg2rad(th.tensor([self.horizontal_fov / 2])).item(),
            num_points,
        )

        # Convert into 3D unit vectors for each angle
        unit_vector_laser = th.stack([th.cos(angles), th.sin(angles), th.zeros_like(angles)], dim=1)

        # Scale unit vectors by corresponding laser scan distnaces
        assert ((scan >= 0.0) & (scan <= 1.0)).all(), "scan out of valid range [0, 1]"
        scan_laser = unit_vector_laser * (scan * (self.max_range - self.min_range) + self.min_range)

        # Convert scans from laser frame to world frame
        pos, ori = self.get_position_orientation()
        scan_world = (T.quat2mat(ori) @ scan_laser.T).T + pos

        # Convert scans from world frame to local base frame
        base_pos, base_ori = self.occupancy_grid_local_link.get_position_orientation()
        scan_local = (T.quat2mat(base_ori).T @ (scan_world - base_pos).T).T
        scan_local = scan_local[:, :2]
        scan_local = th.cat([th.tensor([[0, 0]]), scan_local, th.tensor([[0, 0]])], dim=0)

        # flip y axis
        scan_local[:, 1] *= -1

        # Initialize occupancy grid -- default is unknown values
        occupancy_grid = th.full(
            (self.occupancy_grid_resolution, self.occupancy_grid_resolution),
            fill_value=int(OccupancyGridState.UNKNOWN * 2.0),
            dtype=th.uint8,
        )

        # Convert local scans into the corresponding OG square it should belong to (note now all values are > 0, since
        # OG ranges from [0, resolution] x [0, resolution])
        scan_local_in_map = scan_local / self.occupancy_grid_range * self.occupancy_grid_resolution + (
            self.occupancy_grid_resolution / 2
        )
        scan_local_in_map = scan_local_in_map.reshape((1, -1, 1, 2)).int()

        occupancy_grid = occupancy_grid.cpu().numpy()
        scan_local_in_map = scan_local_in_map.cpu().numpy()

        # For each scan hit,
        for i in range(scan_local_in_map.shape[1]):
            cv2.circle(
                img=occupancy_grid,
                center=(scan_local_in_map[0, i, 0, 0], scan_local_in_map[0, i, 0, 1]),
                radius=2,
                color=int(OccupancyGridState.OBSTACLES * 2.0),
                thickness=-1,
            )
        cv2.fillPoly(
            img=occupancy_grid, pts=scan_local_in_map, color=int(OccupancyGridState.FREESPACE * 2.0), lineType=1
        )
        cv2.circle(
            img=occupancy_grid,
            center=(self.occupancy_grid_resolution // 2, self.occupancy_grid_resolution // 2),
            radius=self.occupancy_grid_inner_radius,
            color=int(OccupancyGridState.FREESPACE * 2.0),
            thickness=-1,
        )

        return th.tensor(occupancy_grid[:, :, None]) / 2.0

    def _get_obs(self):
        # Run super first to grab any upstream obs
        obs, info = super()._get_obs()

        # Add scan info (normalized to [0.0, 1.0])
        if "scan" in self._modalities:
            raw_scan = th.tensor(self._rs.get_linear_depth_data(self.prim_path), dtype=th.float32)
            # Sometimes get_linear_depth_data will return values that are slightly out of range, needs clipping
            raw_scan = th.clip(raw_scan, self.min_range, self.max_range)
            obs["scan"] = (raw_scan - self.min_range) / (self.max_range - self.min_range)

            # Optionally add occupancy grid info
            if "occupancy_grid" in self._modalities:
                obs["occupancy_grid"] = self.get_local_occupancy_grid(scan=obs["scan"])

        return obs, info

    @property
    def n_horizontal_rays(self):
        """
        Returns:
            int: Number of horizontal rays for this range sensor
        """
        return int(self.horizontal_fov // self.horizontal_resolution)

    @property
    def n_vertical_rays(self):
        """
        Returns:
            int: Number of vertical rays for this range sensor
        """
        return int(self.vertical_fov // self.vertical_resolution)

    @property
    def min_range(self):
        """
        Gets this range sensor's min_range (minimum distance in meters which will register a hit)

        Returns:
            float: minimum range for this range sensor, in meters
        """
        return self.get_attribute("minRange")

    @min_range.setter
    def min_range(self, val):
        """
        Sets this range sensor's min_range (minimum distance in meters which will register a hit)

        Args:
            val (float): minimum range for this range sensor, in meters
        """
        self.set_attribute("minRange", val)

    @property
    def max_range(self):
        """
        Gets this range sensor's max_range (maximum distance in meters which will register a hit)

        Returns:
            float: maximum range for this range sensor, in meters
        """
        return self.get_attribute("maxRange")

    @max_range.setter
    def max_range(self, val):
        """
        Sets this range sensor's max_range (maximum distance in meters which will register a hit)

        Args:
            val (float): maximum range for this range sensor, in meters
        """
        self.set_attribute("maxRange", val)

    @property
    def draw_lines(self):
        """
        Gets whether range lines are drawn for this sensor

        Returns:
            bool: Whether range lines are drawn for this sensor
        """
        return self.get_attribute("drawLines")

    @draw_lines.setter
    def draw_lines(self, draw):
        """
        Sets whether range lines are drawn for this sensor

        Args:
            draw (float): Whether range lines are drawn for this sensor
        """
        self.set_attribute("drawLines", draw)

    @property
    def draw_points(self):
        """
        Gets whether range points are drawn for this sensor

        Returns:
            bool: Whether range points are drawn for this sensor
        """
        return self.get_attribute("drawPoints")

    @draw_points.setter
    def draw_points(self, draw):
        """
        Sets whether range points are drawn for this sensor

        Args:
            draw (float): Whether range points are drawn for this sensor
        """
        self.set_attribute("drawPoints", draw)

    @property
    def horizontal_fov(self):
        """
        Gets this range sensor's horizontal_fov

        Returns:
            float: horizontal field of view for this range sensor
        """
        return self.get_attribute("horizontalFov")

    @horizontal_fov.setter
    def horizontal_fov(self, fov):
        """
        Sets this range sensor's horizontal_fov

        Args:
            fov (float): horizontal field of view to set
        """
        self.set_attribute("horizontalFov", fov)

    @property
    def horizontal_resolution(self):
        """
        Gets this range sensor's horizontal_resolution (degrees in between each horizontal hit)

        Returns:
            float: horizontal resolution for this range sensor, in degrees
        """
        return self.get_attribute("horizontalResolution")

    @horizontal_resolution.setter
    def horizontal_resolution(self, resolution):
        """
        Sets this range sensor's horizontal_resolution (degrees in between each horizontal hit)

        Args:
            resolution (float): horizontal resolution to set, in degrees
        """
        self.set_attribute("horizontalResolution", resolution)

    @property
    def vertical_fov(self):
        """
        Gets this range sensor's vertical_fov

        Returns:
            float: vertical field of view for this range sensor
        """
        return self.get_attribute("verticalFov")

    @vertical_fov.setter
    def vertical_fov(self, fov):
        """
        Sets this range sensor's vertical_fov

        Args:
            fov (float): vertical field of view to set
        """
        self.set_attribute("verticalFov", fov)

    @property
    def vertical_resolution(self):
        """
        Gets this range sensor's vertical_resolution (degrees in between each vertical hit)

        Returns:
            float: vertical resolution for this range sensor, in degrees
        """
        return self.get_attribute("verticalResolution")

    @vertical_resolution.setter
    def vertical_resolution(self, resolution):
        """
        Sets this range sensor's vertical_resolution (degrees in between each vertical hit)

        Args:
            resolution (float): vertical resolution to set, in degrees
        """
        self.set_attribute("verticalResolution", resolution)

    @property
    def yaw_offset(self):
        """
        Gets this range sensor's yaw_offset (used in cases where this sensor's forward direction is different than expected)

        Returns:
            float: yaw offset for this range sensor in degrees
        """
        return self.get_attribute("yawOffset")

    @yaw_offset.setter
    def yaw_offset(self, offset):
        """
        Sets this range sensor's yaw_offset (used in cases where this sensor's forward direction is different than expected)

        Args:
            offset (float): yaw offset to set in degrees.
        """
        self.set_attribute("yawOffset", offset)

    @property
    def rotation_rate(self):
        """
        Gets this range sensor's rotation_rate, in degrees per second. Note that a 0 value corresponds to no rotation,
        and all range hits are assumed to be received at the exact same time.

        Returns:
            float: rotation rate for this range sensor in degrees per second
        """
        return self.get_attribute("rotationRate")

    @rotation_rate.setter
    def rotation_rate(self, rate):
        """
        Sets this range sensor's rotation_rate, in degrees per second. Note that a 0 value corresponds to no rotation,
        and all range hits are assumed to be received at the exact same time.

        Args:
            rate (float): rotation rate for this range sensor in degrees per second
        """
        self.set_attribute("rotationRate", rate)

    @classproperty
    def all_modalities(cls):
        return {"scan", "occupancy_grid"}

    @classproperty
    def no_noise_modalities(cls):
        # Occupancy grid should have no noise
        return {"occupancy_grid"}

    @property
    def enabled(self):
        # Just use super
        return super().enabled

    @enabled.setter
    def enabled(self, enabled):
        # We must use super and additionally directly en/disable the sensor in the simulation
        # Note: weird syntax below required to "extend" super class's implementation, see:
        # https://stackoverflow.com/a/37663266
        super(ScanSensor, self.__class__).enabled.fset(self, enabled)
        self.set_attribute("enabled", enabled)
