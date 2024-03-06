import colorsys
import random

import numpy as np
from PIL import Image

try:
    import accimage
except ImportError:
    accimage = None


class RandomScale:
    """Rescale the input PIL.Image to the given size.
    Args:
        minsize (sequence or int): Desired min output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        maxsize (sequence or int): Desired max output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is ``PIL.Image.BILINEAR``
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

    Args:
        N (int): Number of colors to generate

    Returns:
        bright (bool): whether to increase the brightness of the colors or not
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

    Args:
        seg_im ((W, H)-array): Segmentation image
        N (int): Maximum segmentation ID from @seg_im
        colors (None or list of 3-array): If specified, colors to apply
            to different segmentation IDs. Otherwise, will be generated randomly
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

class Remapper:
    def __init__(self):
        self.key_array = np.array([], dtype=np.uint32)  # Initialize the key_array as empty
        self.known_ids = set()

    def clear(self):
        """Resets the key_array to empty."""
        self.key_array = np.array([], dtype=np.uint32)
        self.known_ids = set()

    def remap(self, old_mapping, new_mapping, image):
        """
        Remaps values in the given image from old_mapping to new_mapping using an efficient key_array.
        
        Args:
            old_mapping (dict): The old mapping dictionary that maps a set of image values to labels, e.g. {1:'desk',2:'chair'}.
            new_mapping (dict): The new mapping dictionary that maps another set of image values to labels, e.g. {5:'desk',7:'chair',9:'sofa'}.
            image (np.ndarray): The 2D image to remap, e.g. [[1,1],[1,2]].
        
        Returns:
            np.ndarray: The remapped image, e.g. [[5,5],[5,7]].
            dict: The remapped labels dictionary, e.g. {5:'desk',7:'chair'}.
        """
        # assert np.all([x in old_mapping for x in np.unique(image)]), "Not all keys in the image are in the old mapping!"
        # assert np.all([x in new_mapping.values() for x in old_mapping.values()]), "Not all values in the old mapping are in the new mapping!"

        new_keys = old_mapping.keys() - self.known_ids
        max_key = max(np.max(image), len(self.key_array))

        # If key_array is empty or does not cover all old keys, rebuild it
        if new_keys or max_key >= len(self.key_array):
            self.known_ids.update(new_keys)
            
            # Using max uint32 as a placeholder for unmapped values may not be safe
            assert np.all(new_mapping != np.iinfo(np.uint32).max), "New mapping contains default unmapped value!"
            prev_key_array = self.key_array.copy()
            # we are doing this because there are numbers in image that don't necessarily show up in the old_mapping i.e. particle systems
            self.key_array = np.full(max_key + 1, np.iinfo(np.uint32).max, dtype=np.uint32)

            if prev_key_array.size > 0:
                self.key_array[:len(prev_key_array)] = prev_key_array
            
            # populate key_array with new keys
            for key in new_keys:
                label = old_mapping[key]
                new_key = next((k for k, v in new_mapping.items() if v == label), None)
                assert new_key is not None, f"Could not find a new key for label {label} in new_mapping!"
                self.key_array[key] = new_key
            
            # for all the labels that exist in np.unique(image) but not in old_mapping.keys(), we map them to whichever key in new_mapping that equals to 'unlabelled'
            for key in np.unique(image):
                if key not in old_mapping.keys():
                    new_key = next((k for k, v in new_mapping.items() if v == 'unlabelled'), None)
                    assert new_key is not None, f"Could not find a new key for label 'unlabelled' in new_mapping!"
                    self.key_array[key] = new_key

        # Apply remapping
        remapped_img = self.key_array[image]
        assert np.all(remapped_img != np.iinfo(np.uint32).max), "Not all keys in the image are in the key array!"
        remapped_labels = {}
        for key in np.unique(remapped_img):
            remapped_labels[key] = new_mapping[key]
        
        return remapped_img, remapped_labels
