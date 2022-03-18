import colorsys
import random

import numpy as np
from PIL import Image
import os

from igibson import app, assets_path

# Make sure synthetic data extension is enabled
ext_manager = app.app.get_extension_manager()
ext_manager.set_extension_enabled("omni.syntheticdata", True)

# Continue with omni synethic data imports afterwards
from omni.syntheticdata import sensors as sensors_util
import omni.syntheticdata._syntheticdata as sd
sensor_types = sd.SensorType

try:
    import accimage
except ImportError:
    accimage = None


# This is "the goggle" net. For space-purposes, we only (try to) load this network at runtime on demand
GOGGLE = None


class RandomScale(object):
    """Rescale the input PIL.Image to the given size.
    Args:
    size (sequence or int): Desired output size. If size is a sequence like
    (w, h), output size will be matched to this. If size is an int,
    smaller edge of the image will be matched to this number.
    i.e, if height > width, then image will be rescaled to
    (size * height / width, size)
    interpolation (int, optional): Desired interpolation. Default is
    ``PIL.Image.BILINEAR``
    """

    def __init__(self, minsize, maxsize, interpolation=Image.BILINEAR):
        assert isinstance(minsize, int)
        assert isinstance(maxsize, int)
        self.minsize = minsize
        self.maxsize = maxsize
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
        img (PIL.Image): Image to be scaled.
        Returns:
        PIL.Image: Rescaled image.
        """

        size = random.randint(self.minsize, self.maxsize)

        if isinstance(size, int):
            w, h = img.size
            if (w <= h and w == size) or (h <= w and h == size):
                return img
            if w < h:
                ow = size
                oh = int(size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = size
                ow = int(size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            raise NotImplementedError()


def randomize_colors(N, bright=True):
    """
    Modified from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py#L59
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.5
    hsv = [(1.0 * i / N, 1, brightness) for i in range(N)]
    colors = np.array(list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv)))
    rstate = np.random.RandomState(seed=20)
    np.random.shuffle(colors)
    colors[0] = [0, 0, 0]  # First color is black
    return colors


def segmentation_to_rgb(seg_im, N, colors=None):
    """
    Helper function to visualize segmentations as RGB frames.
    NOTE: assumes that geom IDs go up to N at most - if not,
    multiple geoms might be assigned to the same color.
    """
    # ensure all values lie within [0, N]
    seg_im = np.mod(seg_im, N)

    if colors is None:
        use_colors = randomize_colors(N=N, bright=True)
    else:
        use_colors = colors

    if N <= 256:
        return (255.0 * use_colors[seg_im]).astype(np.uint8)
    else:
        return (use_colors[seg_im]).astype(np.float)


def get_rgb_filled(viewport):
    rgb = sensors_util.get_rgb(viewport=viewport)
    return fill_rgb_from_goggle(rgb=rgb)


def fill_rgb_from_goggle(rgb):
    """
    Fills a given RGB data buffer @rgb by passing it through "the goggle" neural network. Note that this requires torch
    to be installed!

    Args:
        rgb (np.array): (H, W, 3) RGB array

    Returns:
        np.array: RGB array passed through "the goggle" network
    """
    global GOGGLE

    # If we haven't loaded the goggle net yet, load it now
    if GOGGLE is None:
        # Try to import torch
        try:
            import torch
            import torch.nn as nn
            from torchvision import transforms

            from igibson.learn.completion import CompletionNet
        except ImportError:
            raise Exception(
                'Trying to use fill_rgb_from_goggle ("the goggle"), but torch is not installed. '
                'Try "pip install torch torchvision".'
            )

        # Load the goggle net
        GOGGLE = CompletionNet(norm=nn.BatchNorm2d, nf=64)
        GOGGLE = torch.nn.DataParallel(GOGGLE).cuda()
        GOGGLE.load_state_dict(torch.load(os.path.join(assets_path, "networks", "model.pth")))
        GOGGLE.eval()

    # Pass the input through the network and return
    with torch.no_grad():
        tensor = transforms.ToTensor()((rgb * 255).astype(np.uint8)).cuda()
        rgb_filled = GOGGLE(tensor[None, :, :, :])[0]
        return rgb_filled.permute(1, 2, 0).cpu().numpy()
