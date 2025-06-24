from pathlib import Path
from typing import Optional, Protocol, Tuple

import numpy as np


class CameraDriver(Protocol):
    """Camera protocol.

    A protocol for a camera driver. This is used to abstract the camera from the rest of the code.
    """

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read a frame from the camera.

        Args:
            img_size: The size of the image to return. If None, the original size is returned.
            farthest: The farthest distance to map to 255.

        Returns:
            np.ndarray: The color image.
            np.ndarray: The depth image.
        """


class DummyCamera(CameraDriver):
    """A dummy camera for testing."""

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read a frame from the camera.

        Args:
            img_size: The size of the image to return. If None, the original size is returned.
            farthest: The farthest distance to map to 255.

        Returns:
            np.ndarray: The color image.
            np.ndarray: The depth image.
        """
        if img_size is None:
            return (
                np.random.randint(255, size=(480, 640, 3), dtype=np.uint8),
                np.random.randint(255, size=(480, 640, 1), dtype=np.uint16),
            )
        else:
            return (
                np.random.randint(
                    255, size=(img_size[0], img_size[1], 3), dtype=np.uint8
                ),
                np.random.randint(
                    255, size=(img_size[0], img_size[1], 1), dtype=np.uint16
                ),
            )


class SavedCamera(CameraDriver):
    def __init__(self, path: str = "example"):
        self.path = str(Path(__file__).parent / path)
        from PIL import Image

        self._color_img = Image.open(f"{self.path}/image.png")
        self._depth_img = Image.open(f"{self.path}/depth.png")

    def read(
        self,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if img_size is not None:
            color_img = self._color_img.resize(img_size)
            depth_img = self._depth_img.resize(img_size)
        else:
            color_img = self._color_img
            depth_img = self._depth_img

        return np.array(color_img), np.array(depth_img)[:, :, 0:1]
