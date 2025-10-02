import colorsys
from typing import Dict, List, Tuple
import cv2
import numpy as np

import torch as th
from PIL import Image, ImageDraw
import omnigibson.utils.transform_utils as T
from skimage.color import rgb2lab, deltaE_ciede2000
from scipy.ndimage import binary_dilation

try:
    # import accimage
    pass
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

        size = th.randint(self.minsize, self.maxsize + 1)

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


class Remapper:
    """
    Remaps values in an image from old_mapping to new_mapping using an efficient key_array.
    See more details in the remap method.
    """

    def __init__(self):
        self.key_array = None  # Will be initialized with the correct device on first use
        self.known_ids = set()
        self.unlabelled_ids = set()
        self.warning_printed = set()

    def clear(self):
        """Resets the key_array to empty."""
        self.key_array = None  # Will be reinitialized with the correct device on next use
        self.known_ids = set()
        self.unlabelled_ids = set()

    def remap(self, old_mapping, new_mapping, image, image_keys=None):
        """
        Remaps values in the given image from old_mapping to new_mapping using an efficient key_array.
        If the image contains values that are not in old_mapping, they are remapped to the value in new_mapping
        that corresponds to 'unlabelled'.

        Args:
            old_mapping (dict): The old mapping dictionary that maps a set of image values to labels
                e.g. {1: 'desk', 2: 'chair'}.
            new_mapping (dict): The new mapping dictionary that maps another set of image values to labels,
                e.g. {5: 'desk', 7: 'chair', 100: 'unlabelled'}.
            image (th.tensor): The 2D image to remap, e.g. [[1, 3], [1, 2]].
            image_keys (th.tensor): The unique keys in the image, e.g. [1, 2, 3].

        Returns:
            th.tensor: The remapped image, e.g. [[5,100],[5,7]].
            dict: The remapped labels dictionary, e.g. {5: 'desk', 7: 'chair', 100: 'unlabelled'}.
        """
        # Make sure that max int32 doesn't match any value in the new mapping
        assert th.all(
            th.tensor(list(new_mapping.keys())) != th.iinfo(th.int32).max
        ), "New mapping contains default unmapped value!"
        image_max_key = max(th.max(image).item(), max(old_mapping.keys()))

        if self.key_array is None:
            # First time initialization
            self.key_array = th.full((image_max_key + 1,), th.iinfo(th.int32).max, dtype=th.int32, device=image.device)
        else:
            key_array_max_key = len(self.key_array) - 1
            if image_max_key > key_array_max_key:
                prev_key_array = self.key_array.clone()
                # We build a new key array and use max int32 as the default value.
                self.key_array = th.full(
                    (image_max_key + 1,), th.iinfo(th.int32).max, dtype=th.int32, device=image.device
                )
                # Copy the previous key array into the new key array
                self.key_array[: len(prev_key_array)] = prev_key_array

        # Retrospectively inspect our cached ids against the old mapping and update the key array
        updated_ids = set()
        for unlabelled_id in self.unlabelled_ids:
            if unlabelled_id in old_mapping and old_mapping[unlabelled_id] != "unlabelled":
                # If an object was previously unlabelled but now has a label, we need to update the key array
                updated_ids.add(unlabelled_id)
        self.unlabelled_ids -= updated_ids

        # For updated ids, we need to update their key_array entries and then mark them as known
        for updated_id in updated_ids:
            label = old_mapping[updated_id]
            new_key = next((k for k, v in new_mapping.items() if v == label), None)
            assert new_key is not None, f"Could not find a new key for label {label} in new_mapping!"
            self.key_array[updated_id] = new_key

        # Add updated ids to known_ids since they now have valid mappings
        self.known_ids.update(updated_ids)

        # Check if any objects in known_ids have changed their labels and need updating
        changed_known_ids = set()
        for known_id in self.known_ids:
            if known_id in old_mapping:
                # Get the current label from old_mapping
                current_label = old_mapping[known_id]
                # Get the currently mapped value from key_array
                current_mapped_value = self.key_array[known_id].item()
                # Find what label this mapped value corresponds to
                current_mapped_label = None
                for k, v in new_mapping.items():
                    if k == current_mapped_value:
                        current_mapped_label = v
                        break

                # If the labels don't match, we need to update this known_id
                if current_mapped_label != current_label:
                    changed_known_ids.add(known_id)

        # Update the key_array for changed known_ids
        for changed_id in changed_known_ids:
            label = old_mapping[changed_id]
            new_key = next((k for k, v in new_mapping.items() if v == label), None)
            assert new_key is not None, f"Could not find a new key for label {label} in new_mapping!"
            self.key_array[changed_id] = new_key

        new_keys = old_mapping.keys() - self.known_ids

        if new_keys:
            self.known_ids.update(new_keys)
            # Populate key_array with new keys
            for key in new_keys:
                label = old_mapping[key]
                new_key = next((k for k, v in new_mapping.items() if v == label), None)
                assert new_key is not None, f"Could not find a new key for label {label} in new_mapping!"
                self.key_array[key] = new_key
                if label == "unlabelled":
                    # Some objects in the image might be unlabelled first but later get a valid label later, so we keep track of them
                    self.unlabelled_ids.add(key)

        # For all the values that exist in the image but not in old_mapping.keys(), we map them to whichever key in
        # new_mapping that equals to 'unlabelled'. This is needed because some values in the image don't necessarily
        # show up in the old_mapping, i.e. particle systems.
        for key in th.unique(image) if image_keys is None else image_keys:
            if key.item() not in old_mapping.keys():
                new_key = next((k for k, v in new_mapping.items() if v == "unlabelled"), None)
                assert new_key is not None, "Could not find a new key for label 'unlabelled' in new_mapping!"
                self.key_array[key] = new_key

        # Apply remapping
        remapped_img = self.key_array[image]
        # Make sure all values are correctly remapped and not equal to the default value
        assert th.all(remapped_img != th.iinfo(th.int32).max), "Not all keys in the image are in the key array!"
        remapped_labels = {}
        for key in th.unique(remapped_img):
            remapped_labels[key.item()] = new_mapping[key.item()]

        return remapped_img, remapped_labels


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
    colors = th.tensor(list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv)))
    colors = colors[th.randperm(colors.size(0))]
    colors[0] = th.tensor([0, 0, 0], dtype=th.float32)  # First color is black
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
    seg_im = th.fmod(seg_im, N).cpu()

    if colors is None:
        use_colors = randomize_colors(N=N, bright=True)
    else:
        use_colors = colors

    if N <= 256:
        return (255.0 * use_colors[seg_im]).to(th.uint8)
    else:
        return use_colors[seg_im]

def instance_to_bbox(obs: th.Tensor, instance_mapping: Dict[int, str], unique_ins_ids: List[int]) -> List[Tuple[int, int, int, int, int]]:
    """
    Convert instance segmentation to bounding boxes.
    
    Args:
        obs (th.Tensor): (H, W) tensor of instance IDs
        instance_mapping (Dict[int, str]): Dict mapping instance IDs to instance names
            Note: this does not need to include all instance IDs, only the ones that we want to generate bbox for
        unique_ins_ids (List[int]): List of unique instance IDs
    Returns:
        List of tuples (x_min, y_min, x_max, y_max, instance_id) for each instance
    """
    bboxes = []
    valid_ids = [id for id in instance_mapping if id in unique_ins_ids]
    for instance_id in valid_ids:
        # Create mask for this instance
        mask = (obs == instance_id)  # (H, W)
        if not mask.any():
            continue
        # Find non-zero indices (where instance exists)
        y_coords, x_coords = th.where(mask)
        if len(y_coords) == 0:
            continue
        # Calculate bounding box
        x_min = x_coords.min().item()
        x_max = x_coords.max().item()
        y_min = y_coords.min().item()
        y_max = y_coords.max().item()
        bboxes.append((x_min, y_min, x_max, y_max, instance_id))

    return bboxes

def find_non_overlapping_text_position(x1, y1, x2, y2, text_size, occupied_regions, img_height, img_width):
    """Find a text position that doesn't overlap with existing text."""
    text_w, text_h = text_size
    padding = 5

    # Try different positions in order of preference
    positions = [
        # Above bbox
        (x1, y1 - text_h - padding),
        # Below bbox
        (x1, y2 + text_h + padding),
        # Right of bbox
        (x2 + padding, y1 + text_h),
        # Left of bbox
        (x1 - text_w - padding, y1 + text_h),
        # Inside bbox (top-left)
        (x1 + padding, y1 + text_h + padding),
        # Inside bbox (bottom-right)
        (x2 - text_w - padding, y2 - padding),
    ]

    for text_x, text_y in positions:
        # Check bounds
        if text_x < 0 or text_y < text_h or text_x + text_w > img_width or text_y > img_height:
            continue

        # Check for overlap with existing text
        text_rect = (text_x - padding, text_y - text_h - padding, text_x + text_w + padding, text_y + padding)

        overlap = False
        for occupied_rect in occupied_regions:
            if (
                text_rect[0] < occupied_rect[2]
                and text_rect[2] > occupied_rect[0]
                and text_rect[1] < occupied_rect[3]
                and text_rect[3] > occupied_rect[1]
            ):
                overlap = True
                break

        if not overlap:
            return text_x, text_y, text_rect

    # Fallback: use the first position even if it overlaps
    text_x, text_y = positions[0]
    text_rect = (text_x - padding, text_y - text_h - padding, text_x + text_w + padding, text_y + padding)
    return text_x, text_y, text_rect

def get_consistent_color(instance_id):
    import colorsys

    colors = [
        (52, 73, 94),  # Dark blue-gray
        (142, 68, 173),  # Purple
        (39, 174, 96),  # Emerald green
        (230, 126, 34),  # Orange
        (231, 76, 60),  # Red
        (41, 128, 185),  # Blue
        (155, 89, 182),  # Amethyst
        (26, 188, 156),  # Turquoise
        (241, 196, 15),  # Yellow (darker)
        (192, 57, 43),  # Dark red
        (46, 204, 113),  # Green
        (52, 152, 219),  # Light blue
        (155, 89, 182),  # Violet
        (22, 160, 133),  # Dark turquoise
        (243, 156, 18),  # Dark yellow
        (211, 84, 0),  # Dark orange
        (154, 18, 179),  # Dark purple
        (31, 81, 255),  # Royal blue
        (20, 90, 50),  # Forest green
        (120, 40, 31),  # Maroon
    ]

    # # a test, consistently use green color
    # return (39, 174, 96)

    # Use hash to consistently select a color from the palette
    hash_val = hash(str(instance_id))
    base_color_idx = hash_val % len(colors)
    base_color = colors[base_color_idx]

    # Add slight variation while maintaining sophistication
    # Convert to HSV for easier manipulation
    r, g, b = [c / 255.0 for c in base_color]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    # Add small random variation to hue (±10 degrees) and saturation/value
    hue_variation = ((hash_val >> 8) % 20 - 10) / 360.0  # ±10 degrees
    sat_variation = ((hash_val >> 16) % 20 - 10) / 200.0  # ±5% saturation
    val_variation = ((hash_val >> 24) % 20 - 10) / 200.0  # ±5% value

    # Apply variations with bounds checking
    h = (h + hue_variation) % 1.0
    s = max(0.4, min(0.9, s + sat_variation))  # Keep saturation between 40-90%
    v = max(0.3, min(0.7, v + val_variation))  # Keep value between 30-70% (darker for contrast)

    # Convert back to RGB
    r, g, b = colorsys.hsv_to_rgb(h, s, v)

    # Convert to 0-255 range
    return (int(r * 255), int(g * 255), int(b * 255))

def clean_single_part(part: str) -> str:
    """Clean a single part that has no underscores."""
    # Remove trailing numbers for single parts
    cleaned = ''.join(c for c in part if not c.isdigit())
    return cleaned if cleaned else part

def is_strange_string(part: str) -> bool:
    """
    Detect if a 6-character string is likely a strange/generated identifier.
    
    Characteristics of strange strings:
    - Exactly 6 characters
    - Mix of letters that don't form common English patterns
    - Often have repeated characters or unusual patterns
    - Sequential patterns like "abcdef"
    
    Args:
        part: String part to check
        
    Returns:
        bool: True if likely a strange string
    """
    if len(part) != 6:
        return False
    
    # Check for common English prefixes/suffixes in 6-char words
    common_6char_words = {'cutter', 'broken', 'paving', 'marker', 'napkin', 'edible', 'tablet', 'saddle', 'flower', 'wooden', 'square', 'peeler', 'shovel', 'nickel', 'pestle', 'gravel', 'french', 'sesame', 'bleach', 'pewter', 'outlet', 'fabric', 'staple', 'banana', 'almond', 'masher', 'carpet', 'fridge', 'swivel', 'normal', 'potato', 'litter', 'button', 'pomelo', 'hanger', 'trophy', 'drying', 'hamper', 'radish', 'grater', 'pillow', 'skates', 'canvas', 'cloche', 'nutmeg', 'indoor', 'slicer', 'lotion', 'rolled', 'starch', 'chives', 'tomato', 'dinner', 'tartar', 'goblet', 'polish', 'liners', 'runner', 'danish', 'tissue', 'shaped', 'tassel', 'quartz', 'muffin', 'lights', 'hoodie', 'burlap', 'wrench', 'shorts', 'hotdog', 'lemons', 'turnip', 'cookie', 'salmon', 'abacus', 'guitar', 'paddle', 'boxers', 'cherry', 'liquid', 'helmet', 'folder', 'silver', 'record', 'floors', 'middle', 'eraser', 'hinged', 'carton', 'wicker', 'coffee', 'smoker', 'tender', 'zipper', 'public', 'pastry', 'mallet', 'mussel', 'flakes', 'system', 'snacks', 'pebble', 'cereal', 'mixing', 'window', 'sandal', 'orange', 'toilet', 'fennel', 'cooker', 'puzzle', 'oyster', 'switch', 'barley', 'funnel', 'blower', 'infant', 'shaker', 'sensor', 'butter', 'jigger', 'ground', 'mirror', 'soccer', 'pallet', 'garage', 'longue', 'urinal', 'celery', 'bikini', 'shears', 'dental', 'waffle', 'handle', 'noodle', 'stairs', 'easter', 'socket', 'boiled', 'poster', 'drawer', 'chisel', 'holder', 'tackle', 'breast', 'pickle', 'plugin', 'jigsaw', 'collar', 'mortar', 'pepper', 'teapot', 'trowel', 'credit', 'screen', 'boxing', 'walker', 'garlic', 'laptop', 'server', 'deicer', 'omelet', 'pruner', 'makeup', 'chaise', 'shrimp', 'tennis', 'feeder', 'sticky', 'gloves', 'yogurt', 'pencil', 'icicle', 'powder', 'carafe', 'leaves', 'jersey', 'ginger', 'bucket', 'diaper', 'shower', 'grains', 'medium', 'pellet', 'honing', 'dahlia', 'gaming', 'chilli', 'router', 'icetea', 'pickup', 'figure', 'baking', 'cymbal', 'violin', 'frying', 'bottle', 'kettle', 'spirit', 'ticket', 'washer', 'burner', 'durian', 'carrot', 'statue', 'basket', 'blouse', 'roller', 'squash', 'webcam', 'candle', 'jacket', 'ladder', 'kidney', 'thread', 'dipper', 'loofah', 'tights', 'branch', 'ripsaw', 'pommel', 'heater', 'cactus', 'peanut', 'canned', 'walnut', 'pillar', 'cooler', 'cloves', 'hammer', 'wreath', 'hummus', 'hiking', 'letter', 'teacup', 'cotton', 'weight', 'fillet', 'juicer', 'cheese', 'crayon', 'bottom', 'garden', 'tinsel', 'camera', 'wading', 'analog', 'sponge', 'wallet', 'center', 'locker', 'copper', 'tripod', 'filter', 'raisin', 'rubber', 'ribbon', 'hockey', 'beaker', 'catsup', 'output', 'sodium', 'turkey', 'quiche', 'vacuum', 'saucer', 'papaya', 'sliced', 'hammam', 'grated', 'racket', 'motion', 'onesie'}
    
    if part in common_6char_words:
        return False
    
    return True

def remove_trailing_numbers(part: str) -> str:
    """Remove numbers from the end of a string."""
    # Find the last non-digit character
    last_alpha_idx = -1
    for i in range(len(part) - 1, -1, -1):
        if not part[i].isdigit():
            last_alpha_idx = i
            break
    
    if last_alpha_idx >= 0:
        return part[:last_alpha_idx + 1]
    else:
        # All digits, return original (shouldn't happen due to earlier checks)
        return part

def process_name_part(part: str, position: int, total_parts: int, all_parts: list) -> str:
    """
    Process a single part of an object name with advanced logic.
    
    Args:
        part: The part to process
        position: Position in the parts list (0-indexed)
        total_parts: Total number of parts
        all_parts: All parts for context
        
    Returns:
        str: Cleaned part or None if should be removed
    """
    # Skip empty parts
    if not part:
        return None
    
    # Skip pure numbers at the end (like "_90", "_1")
    if part.isdigit():
        return None
    
    # Handle robot special case: keep meaningful robot IDs like "r1", "r2"
    if position > 0 and any(prev_part in ['robot', 'agent', 'player'] for prev_part in all_parts[:position]):
        if part.startswith('r') and len(part) <= 3 and part[1:].isdigit():
            return part  # Keep "r1", "r2", etc.
    
    # Detect and remove 6-character strange strings (like "tynnnw")
    if len(part) == 6 and is_strange_string(part) and position != 0:
        return None
    
    # Remove numbers from the end of meaningful parts
    # But keep the meaningful part (e.g., "processor90" -> "processor")
    cleaned = remove_trailing_numbers(part)
    
    # Only keep if it has meaningful content
    if len(cleaned) >= 2 and cleaned.isalpha():
        return cleaned
    
    # Special case: if it's a very short part and not obviously garbage, keep it
    if len(part) <= 3 and part.isalpha():
        return part
    
    return None

def format_object_name(name: str) -> str:
    """
    Format object name for natural language with advanced pattern recognition.
    
    Handles specific patterns:
    - robot_r1 -> "robot r1" (keep meaningful suffixes, remove underscores)
    - food_processor_90 -> "food processor" (remove numeric suffixes)
    - top_cabinet_tynnnw_1 -> "top cabinet" (remove 6-char strange strings + numbers)
    
    Args:
        name: Raw object name from scene graph
        
    Returns:
        str: Formatted object name with "the" article
    """
    if not name:
        return "the object"
    
    # Convert to lowercase for processing
    original_name = name
    name = name.lower()
    
    # Split by underscores
    parts = name.split('_')
    
    if len(parts) == 1:
        # No underscores, just clean and return
        cleaned = clean_single_part(parts[0])
        return f"the {cleaned}" if cleaned else "the object"
    
    cleaned_parts = []
    
    # Process each part with advanced logic
    for i, part in enumerate(parts):
        cleaned_part = process_name_part(part, i, len(parts), parts)
        if cleaned_part:
            cleaned_parts.append(cleaned_part)
    
    if not cleaned_parts:
        print(f"No cleaned parts found for {original_name}")
        exit()
    
    return " ".join(cleaned_parts)


def overlay_bboxes_with_names(
    img: np.ndarray,
    bbox_2d_data: List[Tuple[int, int, int, int, int]],
    instance_mapping: Dict[int, str],
    task_relevant_objects: List[str],
) -> np.ndarray:
    """
    Overlays bounding boxes with object names on the given image.

    Args:
        img (np.ndarray): The input image (RGB) to overlay on.
        bbox_2d_data (List[Tuple[int, int, int, int, int]]): Bounding box data with format (x1, y1, x2, y2, instance_id)
        instance_mapping (Dict[int, str]): Mapping from instance ID to object name
        task_relevant_objects (List[str]): List of task relevant objects
    Returns:
        np.ndarray: The image with bounding boxes and object names overlaid.
    """
    # Create a copy of the image to draw on
    overlay_img = img.copy()
    img_height, img_width = img.shape[:2]

    # Track occupied text regions to avoid overlap
    occupied_text_regions = []

    # Process each bounding box
    for bbox in bbox_2d_data:
        x1, y1, x2, y2, instance_id = bbox
        object_name = instance_mapping[instance_id]
        # Only overlay task relevant objects
        if object_name not in task_relevant_objects:
            continue

        # Generate a consistent color based on instance_id
        color = get_consistent_color(instance_id)

        object_name = format_object_name(object_name)

        # Draw the bounding box
        cv2.rectangle(overlay_img, (x1, y1), (x2, y2), color, 2)

        # Draw the object name
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(object_name, font, font_scale, font_thickness)[0]
        # Find non-overlapping position for text
        text_x, text_y, text_rect = find_non_overlapping_text_position(
            x1, y1, x2, y2, text_size, occupied_text_regions, img_height, img_width
        )
        # Add this text region to occupied regions
        occupied_text_regions.append(text_rect)

        # Draw background rectangle for text
        cv2.rectangle(
            overlay_img, (int(text_rect[0]), int(text_rect[1])), (int(text_rect[2]), int(text_rect[3])), color, -1
        )

        # Draw the text
        cv2.putText(
            overlay_img,
            object_name,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    return overlay_img

def overlay_bboxes(
    img: np.ndarray,
    bbox_2d_data: List[Tuple[int, int, int, int, int]],
    instance_mapping: Dict[int, str],
    task_relevant_objects: List[str],
    in_place: bool = False,
) -> np.ndarray:
    """
    Overlays bounding boxes on the given image for task-relevant objects.

    Args:
        img (np.ndarray): The input image (RGB) to overlay on.
        bbox_2d_data (List[Tuple[int, int, int, int, int]]): Bounding box data
            with format (x1, y1, x2, y2, instance_id).
        instance_mapping (Dict[int, str]): Mapping from instance ID to object name.
        task_relevant_objects (List[str]): List of task-relevant objects.
        in_place (bool): If True, modifies the input image directly instead of creating a copy.

    Returns:
        np.ndarray: The image with bounding boxes overlaid.
    """
    # Create a copy of the image to draw on, unless in_place is True
    overlay_img = img if in_place else img.copy()

    # Process each bounding box
    for bbox in bbox_2d_data:
        x1, y1, x2, y2, instance_id = bbox
        object_name = instance_mapping.get(instance_id)

        # Only overlay task-relevant objects
        if object_name not in task_relevant_objects:
            continue

        # Generate a consistent color based on instance_id
        color = get_consistent_color(instance_id)

        # Draw the bounding box
        cv2.rectangle(overlay_img, (x1, y1), (x2, y2), color, 2)

    return overlay_img

def overlay_segmentation_mask(
    img: np.ndarray,
    visibility_matrix: th.Tensor,
    unique_ins_ids: List[int],
    instance_mapping: Dict[int, str],
    task_relevant_objects: List[str],
    in_place: bool = False,
) -> np.ndarray:
    """
    Overlays segmentation mask on the given image for task-relevant objects.
    
    Args:
        img (np.ndarray): The input image (RGB) to overlay on.
        visibility_matrix (th.Tensor): (H, W) tensor of instance IDs.
        unique_ins_ids (List[int]): List of unique instance IDs in the current view.
        instance_mapping (Dict[int, str]): Mapping from instance ID to object name.
        task_relevant_objects (List[str]): List of task-relevant objects.
        in_place (bool): If True, modifies the input image directly instead of creating a copy.
    
    Returns:
        np.ndarray: The image with segmentation masks overlaid.
    """
    overlay_img = img if in_place else img.copy()
    
    # Convert visibility matrix to numpy if it's a tensor
    if isinstance(visibility_matrix, th.Tensor):
        vis_matrix = visibility_matrix.cpu().numpy()
    else:
        vis_matrix = visibility_matrix
    
    # Process each unique instance ID
    valid_ids = [id for id in instance_mapping if id in unique_ins_ids]
    
    for instance_id in valid_ids:
        object_name = instance_mapping.get(instance_id)
        
        # Only overlay task-relevant objects
        if object_name not in task_relevant_objects:
            continue
        
        # Create mask for this instance
        mask = (vis_matrix == instance_id)
        if not mask.any():
            continue
        
        # Generate consistent color for this instance
        color = get_consistent_color(instance_id)
        
        # Apply 50% transparency overlay where mask is true
        alpha = 0.5
        
        # Blend the colors: (1-alpha) * original + alpha * overlay_color
        overlay_img[mask] = ((1.0 - alpha) * overlay_img[mask].astype(np.float32) + 
                            alpha * np.array(color, dtype=np.float32)).astype(overlay_img.dtype)
    
    return overlay_img

def calculate_delta_e(
    obs_ids: th.Tensor,
    rgb_image: th.Tensor,
    instance_mapping: Dict[int, str],
    unique_ins_ids: List[int]
) -> Dict[int, float]:
    """
    Calculates the CIEDE2000 Delta E color difference for each instance
    against its immediate surrounding background.

    Args:
        obs_ids (th.Tensor): (H, W) tensor of instance IDs.
        rgb_image (th.Tensor): (H, W, 3) tensor of the original RGB image (values 0-255).
        instance_mapping (Dict[int, str]): Dict mapping instance IDs to names.
        unique_ins_ids (List[int]): List of unique instance IDs in the current view.

    Returns:
        Dict[int, float]: A dictionary mapping each valid instance ID to its Delta E score.
    """
    # Convert PyTorch tensors to NumPy arrays for image processing, as SciPy and Scikit-image work with them.
    # We only need to do this once.
    ids_np = obs_ids.cpu().numpy()
    rgb_np = rgb_image.cpu().numpy().astype(np.uint8)

    delta_e_scores = {}
    valid_ids = [id for id in instance_mapping if id in unique_ins_ids]

    for instance_id in valid_ids:
        # 1. Create a boolean mask for the current instance.
        object_mask = (ids_np == instance_id)
        if not object_mask.any():
            continue

        # 2. Create a "background ring" mask using morphological dilation.
        # This creates a border of pixels immediately surrounding the object.
        dilated_mask = binary_dilation(object_mask, iterations=2) # iterations control ring thickness
        background_mask = dilated_mask & ~object_mask
        
        # If the object is at the edge of the image, the background ring might be empty.
        if not background_mask.any():
            delta_e_scores[instance_id] = 0.0 # Or float('nan') if you prefer
            continue

        # 3. Calculate the average color for the object and its background.
        # NumPy's boolean array indexing makes this very efficient.
        avg_rgb_object = rgb_np[object_mask][:, :3].mean(axis=0)
        avg_rgb_background = rgb_np[background_mask][:, :3].mean(axis=0)

        # 4. Convert the two average RGB colors to LAB color space.
        # The library expects a 3D array, so we reshape our single color vectors.
        lab_object = rgb2lab(avg_rgb_object.reshape(1, 1, 3))
        lab_background = rgb2lab(avg_rgb_background.reshape(1, 1, 3))

        # 5. Calculate the CIEDE2000 Delta E score.
        delta_e = deltaE_ciede2000(lab_object, lab_background)
        
        # The result is in a nested array, so we extract the float value.
        delta_e_scores[instance_id] = int(float(delta_e[0, 0]) + 0.5)

    return delta_e_scores




def change_pcd_frame(pcd: th.Tensor, rel_pose: th.Tensor) -> th.Tensor:
    """
    Transform point cloud from one frame to another frame.

    Args:
        pcd: (N, 6) point cloud tensor in world frame
        rel_pose: (7) relative pose tensor [pos, quat] from current to target frame
    Returns:
        th.Tensor: (B, N, 6) point cloud tensor in target frame
    """
    rel_pos = rel_pose[:3]
    rel_quat = rel_pose[3:]
    rel_rot = T.quat2mat(rel_quat)

    # Create transformation matrix
    tf = th.eye(4)
    tf[:3, :3] = rel_rot
    tf[:3, 3] = rel_pos

    # Add homogeneous coordinate to point cloud
    xyz = pcd[..., 3:]  # (N, 3)
    xyz_homo = th.cat([xyz, th.ones_like(xyz[..., :1])], dim=-1)  # (N, 4)

    # Transform point cloud from camera frame to base frame
    new_xyz = th.matmul(xyz_homo, th.inverse(tf).T)  # (N, 4)
    pcd[..., 3:] = new_xyz[..., :3]
    return pcd
