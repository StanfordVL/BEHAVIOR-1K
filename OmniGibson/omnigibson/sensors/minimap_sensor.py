import cv2
import math
import numpy as np
import torch as th

import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from omnigibson.sensors.sensor_base import BaseSensor
from omnigibson.utils.numpy_utils import NumpyTypes
from omnigibson.utils.python_utils import classproperty
from omnigibson.utils.sim_utils import set_carb_setting


# Duplicate of simulator's render method, used for UI updates
def render():
    """
    Refreshes the Isaac Sim app rendering components including UI elements and view ports.
    """
    set_carb_setting(og.app._carb_settings, "/app/player/playSimulations", False)
    og.app.update()
    set_carb_setting(og.app._carb_settings, "/app/player/playSimulations", True)


class MinimapSensor(BaseSensor):
    """
    Custom sensor that generates and displays a 2D minimap using omni.ui.
    Displays floor segmentation with robot position as an arrow.
    """

    @classproperty
    def all_modalities(cls):
        """
        Returns:
            set: All modalities supported by this sensor (only RGB)
        """
        return {"rgb"}

    @property
    def _obs_space_mapping(self):
        """
        Returns:
            dict: Observation space mapping for minimap RGB output
        """
        return {
            "rgb": ((self.resolution, self.resolution, 3), 0, 255, NumpyTypes.UINT8),
        }

    @classproperty
    def no_noise_modalities(cls):
        """
        Returns:
            set: Modalities that should not have noise applied.
                 Minimap is a visualization, not a real sensor reading.
        """
        return {"rgb"}

    def __init__(
        self, scene, robot, name="minimap", resolution=256, arrow_size=5, arrow_color=(255, 0, 0), seg_alpha=0.5
    ):
        """
        Args:
            scene: Scene object containing seg_map and trav_map
            robot: Robot object to track
            name (str): Sensor name
            resolution (int): Minimap resolution in pixels
            arrow_size (int): Size of the robot arrow indicator in pixels
            arrow_color (tuple): RGB color of the robot arrow (default: red)
            seg_alpha (float): Alpha/opacity for segmentation overlay on traversability map (0.0-1.0)
        """
        # Store references (use different names to avoid conflicts with parent class properties)
        self._scene_ref = scene
        self._robot_ref = robot
        self.resolution = resolution
        self.seg_map = scene.seg_map
        self.arrow_size = arrow_size
        self.arrow_color = arrow_color
        self.seg_alpha = seg_alpha

        # Will be populated in _load() and _post_load()
        self.base_map = None
        self._crop_offset_x = 0  # Column offset after cropping
        self._crop_offset_y = 0  # Row offset after cropping
        self._ui_window = None
        self._byte_provider = None
        self._image_widget = None

        super().__init__(
            relative_prim_path=f"/{name}_sensor",
            name=name,
            modalities={"rgb"},
        )

    def _load(self):
        """Load sensor - create combined traversability + segmentation map and sensor prim"""
        # Get the traversability map (floor 0)
        trav_map = self._scene_ref.trav_map.floor_map[0]

        # Create the colored segmentation map
        colored_seg_map = self._create_colored_segmentation_map()

        # Combine traversability map with segmentation overlay
        combined_map = self._create_combined_map(trav_map, colored_seg_map)

        # Crop to valid regions only (removes grey/black background areas)
        self.base_map = self._crop_to_valid_region(combined_map)

        # Create and return an Xform prim for this sensor (required by BasePrim)
        return og.sim.stage.DefinePrim(self.prim_path, "Xform")

    def _post_load(self):
        """Create omni.ui window with ByteImageProvider for fast minimap display"""
        self._create_ui_window()

    def _initialize(self):
        """
        Initialize the sensor after it has been fully loaded.
        Ensures that robot references are valid before the first display update.
        """
        super()._initialize()

        # Initialize display with base map and current robot pose
        self.update_display()

    def _create_ui_window(self):
        """
        Create a UI window with an image widget backed by ByteImageProvider.
        The image automatically resizes to fit the window.
        """
        # Create the window with docking preference
        self._ui_window = lazy.omni.ui.Window(
            f"Minimap: {self.name}",
            width=self.resolution + 20,
            height=self.resolution + 20,
            dockPreference=lazy.omni.ui.DockPreference.LEFT_BOTTOM,
        )

        with self._ui_window.frame:
            # Create ByteImageProvider for fast texture updates
            self._byte_provider = lazy.omni.ui.ByteImageProvider()

            # Create image widget that fills the frame using percentage-based sizing
            self._image_widget = lazy.omni.ui.ImageWithProvider(
                self._byte_provider,
                width=lazy.omni.ui.Percent(100),
                height=lazy.omni.ui.Percent(100),
            )

        # Render to make window visible
        render()

    def _create_colored_segmentation_map(self):
        """Convert instance segmentation to RGB colored visualization"""
        # Get unique instance IDs
        unique_ids = th.unique(self.seg_map.room_ins_map)

        # Create RGB map
        height, width = self.seg_map.room_ins_map.shape
        colored_map = np.zeros((height, width, 3), dtype=np.uint8)

        for ins_id in unique_ids:
            ins_id_val = ins_id.item()
            if ins_id_val == 0:
                # Background/boundaries - dark gray
                color = (50, 50, 50)
            else:
                # Deterministic color from instance ID using a local RNG
                rng = np.random.RandomState(ins_id_val)
                color = tuple(rng.randint(80, 255, size=3).tolist())

            # Apply color to all pixels with this instance ID
            mask = (self.seg_map.room_ins_map == ins_id).numpy()
            colored_map[mask] = color

        return colored_map

    def _create_combined_map(self, trav_map, colored_seg_map):
        """
        Overlay semi-transparent colored segmentation map on top of traversability map.

        Args:
            trav_map (torch.Tensor): Traversability map (grayscale, 255=traversable, 0=not)
            colored_seg_map (np.ndarray): RGB colored segmentation map

        Returns:
            np.ndarray: Combined RGB map with segmentation overlay on traversability
        """
        # Convert traversability map to numpy and ensure proper size match
        trav_np = trav_map.numpy() if hasattr(trav_map, "numpy") else trav_map

        # Handle size mismatch by resizing traversability map to match segmentation map
        seg_height, seg_width = colored_seg_map.shape[:2]
        trav_height, trav_width = trav_np.shape[:2]

        if (trav_height, trav_width) != (seg_height, seg_width):
            trav_np = cv2.resize(trav_np, (seg_width, seg_height), interpolation=cv2.INTER_NEAREST)

        # Convert grayscale traversability map to RGB
        trav_rgb = cv2.cvtColor(trav_np.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        # Blend: combined = trav * (1 - alpha) + seg * alpha
        alpha = self.seg_alpha
        combined = trav_rgb.astype(np.float32) * (1 - alpha) + colored_seg_map.astype(np.float32) * alpha
        combined = np.clip(combined, 0, 255).astype(np.uint8)

        return combined

    def _crop_to_valid_region(self, colored_map, margin=5):
        """
        Crop the map to only include valid regions (non-background pixels).

        Args:
            colored_map (np.ndarray): RGB colored map
            margin (int): Padding around the valid region in pixels

        Returns:
            np.ndarray: Cropped map containing only valid regions
        """
        # Create mask of valid pixels (non-background, i.e., room_ins_map != 0)
        valid_mask = (self.seg_map.room_ins_map != 0).numpy()

        # Find bounding box of valid regions
        valid_rows, valid_cols = np.where(valid_mask)

        if len(valid_rows) == 0 or len(valid_cols) == 0:
            # No valid regions found, return original map
            return colored_map

        # Get bounding box with margin
        min_row = max(0, valid_rows.min() - margin)
        max_row = min(colored_map.shape[0], valid_rows.max() + margin + 1)
        min_col = max(0, valid_cols.min() - margin)
        max_col = min(colored_map.shape[1], valid_cols.max() + margin + 1)

        # Store crop offsets for coordinate transformation
        self._crop_offset_y = min_row  # Row offset
        self._crop_offset_x = min_col  # Column offset

        # Crop the map
        cropped_map = colored_map[min_row:max_row, min_col:max_col].copy()

        return cropped_map

    def _draw_robot_arrow(self, img, center, yaw_angle, size=None, color=None):
        """
        Draw filled triangle arrow pointing in robot's direction.
        Uses a fixed arrow template rotated by yaw_angle for consistent shape.

        Args:
            img (np.ndarray): Image to draw on
            center (tuple): (x, y) center position
            yaw_angle (float): Yaw angle in radians
            size (int): Arrow size in pixels (uses self.arrow_size if None)
            color (tuple): RGB color (uses self.arrow_color if None)
        """
        if size is None:
            size = self.arrow_size
        if color is None:
            color = self.arrow_color

        # Calculate arrow start and end points based on yaw angle
        cos_a = math.cos(yaw_angle)
        sin_a = math.sin(yaw_angle)

        # Arrow from back to tip, centered at robot position
        half_len = size / 2
        pt1 = (int(center[0] - half_len * cos_a), int(center[1] - half_len * sin_a))  # Back
        pt2 = (int(center[0] + half_len * cos_a), int(center[1] + half_len * sin_a))  # Tip

        # Draw arrow line with arrowhead
        cv2.arrowedLine(img, pt1, pt2, color, thickness=1, line_type=cv2.LINE_8, tipLength=0.6)

    def _get_obs(self):
        """Generate minimap with robot arrow overlay"""
        # Clone base map and flip horizontally to match world orientation
        minimap = np.flip(self.base_map, axis=1).copy()

        # Get original map dimensions for coordinate scaling
        orig_height, orig_width = minimap.shape[:2]

        # Resize to target resolution first (arrow drawn after for better quality)
        minimap_resized = cv2.resize(minimap, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)

        # Calculate scale factors for coordinate transformation
        scale_x = self.resolution / orig_width
        scale_y = self.resolution / orig_height

        # Get robot position and orientation
        robot_pos, robot_quat = self._robot_ref.get_position_orientation()

        # Convert world coords to map pixel coords
        # world_to_map returns (row, col) format after flipping x/y
        map_coords = self.seg_map.world_to_map(robot_pos[:2])
        map_row, map_col = int(map_coords[0].item()), int(map_coords[1].item())

        # Apply crop offset and horizontal flip to get coordinates in flipped cropped map
        # After horizontal flip: new_x = width - 1 - old_x
        original_x = map_col - self._crop_offset_x
        center_x = orig_width - 1 - original_x  # x after horizontal flip
        center_y = map_row - self._crop_offset_y  # y = row (unchanged by horizontal flip)

        # Scale coordinates to resized image
        center_x_scaled = center_x * scale_x
        center_y_scaled = center_y * scale_y

        # Check if robot is within cropped map bounds (using original bounds)
        if 0 <= center_x < orig_width and 0 <= center_y < orig_height:
            # Extract yaw angle from quaternion
            roll, pitch, yaw = T.quat2euler(robot_quat)

            # Adjust yaw for coordinate system:
            # - Horizontal flip requires pi + yaw for correct X direction
            # - Negate to flip Y direction (image Y increases downward)
            adjusted_yaw = -(math.pi + yaw.item())
            self._draw_robot_arrow(minimap_resized, (center_x_scaled, center_y_scaled), adjusted_yaw)

        # Convert to torch tensor (keep as uint8 to match observation space)
        minimap_tensor = th.from_numpy(minimap_resized)

        return {"rgb": minimap_tensor}, {}

    def _update_byte_provider(self, rgb_array):
        """
        Update the ByteImageProvider with new image data (fast, no disk I/O).

        Args:
            rgb_array (np.ndarray): RGB image array of shape (H, W, 3), dtype uint8
        """
        if self._byte_provider is None:
            return

        # ByteImageProvider expects RGBA format
        if rgb_array.shape[-1] == 3:
            # Add alpha channel (fully opaque)
            rgba = np.concatenate([rgb_array, np.full((*rgb_array.shape[:2], 1), 255, dtype=np.uint8)], axis=-1)
        else:
            rgba = rgb_array

        # Ensure contiguous array
        rgba = np.ascontiguousarray(rgba)

        # Update the byte provider with raw bytes
        # set_bytes_data expects (data_bytes, [width, height])
        self._byte_provider.set_bytes_data(
            rgba.flatten().tolist(),
            [rgba.shape[1], rgba.shape[0]],  # [width, height]
        )

    def update_display(self):
        """
        Update the minimap display with current robot position.
        Call this every frame or whenever you want the minimap to update.

        This is fast because it uses in-memory ByteImageProvider (no disk I/O).
        """
        if self._byte_provider is None:
            return

        # Get latest minimap observation
        obs, _ = self._get_obs()
        minimap_img = obs["rgb"].numpy().astype(np.uint8)

        # Update the UI image widget directly in memory
        self._update_byte_provider(minimap_img)

    @property
    def visible(self):
        """
        Returns:
            bool: Whether the minimap window is visible
        """
        if self._ui_window is None:
            return False
        return self._ui_window.visible

    @visible.setter
    def visible(self, value):
        """
        Set minimap window visibility.

        Args:
            value (bool): Whether the minimap should be visible
        """
        if self._ui_window is not None:
            self._ui_window.visible = value

    def remove(self):
        """Clean up the minimap sensor and its UI window"""
        if self._ui_window is not None:
            self._ui_window.destroy()
            self._ui_window = None
        self._byte_provider = None
        self._image_widget = None

        # Call parent cleanup
        super().remove()

    def __del__(self):
        """Destructor - clean up UI resources"""
        try:
            self.remove()
        except:
            pass
