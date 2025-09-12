import colorsys

import torch as th
from PIL import Image, ImageDraw
import omnigibson.utils.transform_utils as T

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


def colorize_bboxes_3d(bbox_3d_data, rgb_image, camera_params):
    """
    Project 3D bounding box data onto 2D and colorize the bounding boxes for visualization.
    Reference: https://forums.developer.nvidia.com/t/mathematical-definition-of-3d-bounding-boxes-annotator-nvidia-omniverse-isaac-sim/223416

    Args:
        bbox_3d_data (th.tensor): 3D bounding box data
        rgb_image (th.tensor): RGB image
        camera_params (dict): Camera parameters

    Returns:
        th.tensor: RGB image with 3D bounding boxes drawn
    """

    def world_to_image_pinhole(world_points, camera_params):
        # Project corners to image space (assumes pinhole camera model)
        proj_mat = camera_params["cameraProjection"].reshape(4, 4)
        view_mat = camera_params["cameraViewTransform"].reshape(4, 4)
        view_proj_mat = view_mat @ proj_mat
        world_points_homo = th.nn.functional.pad(world_points, (0, 1, 0, 0), value=1.0)
        tf_points = th.dot(world_points_homo, view_proj_mat)
        tf_points = tf_points / (tf_points[..., -1:])
        return 0.5 * (tf_points[..., :2] + 1)

    def draw_lines_and_points_for_boxes(img, all_image_points):
        width, height = img.size
        draw = ImageDraw.Draw(img)

        # Define connections between the corners of the bounding box
        connections = [
            (0, 1),
            (1, 3),
            (3, 2),
            (2, 0),  # Front face
            (4, 5),
            (5, 7),
            (7, 6),
            (6, 4),  # Back face
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),  # Side edges connecting front and back faces
        ]

        # Calculate the number of bounding boxes
        num_boxes = len(all_image_points) // 8

        # Generate random colors for each bounding box
        from omni.replicator.core import random_colours

        box_colors = random_colours(num_boxes, enable_random=True, num_channels=3)

        # Ensure colors are in the correct format for drawing (255 scale)
        box_colors = [(int(r), int(g), int(b)) for r, g, b in box_colors]

        # Iterate over each set of 8 points (each bounding box)
        for i in range(0, len(all_image_points), 8):
            image_points = all_image_points[i : i + 8]
            image_points[:, 1] = height - image_points[:, 1]  # Flip Y-axis to match image coordinates

            # Use a distinct color for each bounding box
            line_color = box_colors[i // 8]

            # Draw lines for each connection
            for start, end in connections:
                draw.line(
                    (image_points[start][0], image_points[start][1], image_points[end][0], image_points[end][1]),
                    fill=line_color,
                    width=2,
                )

    rgb = Image.fromarray(rgb_image)

    # Get 3D corners
    from omni.syntheticdata.scripts.helpers import get_bbox_3d_corners

    corners_3d = get_bbox_3d_corners(bbox_3d_data)
    corners_3d = corners_3d.reshape(-1, 3)

    # Project to image space
    corners_2d = world_to_image_pinhole(corners_3d, camera_params)
    width, height = rgb.size
    corners_2d *= th.tensor([[width, height]])

    # Now, draw all bounding boxes
    draw_lines_and_points_for_boxes(rgb, corners_2d)

    return th.tensor(rgb)


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
