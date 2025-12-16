from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import pathlib
import tempfile
import traceback
from cryptography.fernet import Fernet
from pxr import Usd, UsdGeom, Gf, Sdf
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import fs.path
from fs.zipfs import ZipFS
import torch as th
import cv2

DATASET_DIR = pathlib.Path(r"/scr/BEHAVIOR-1K/OmniGibson/omnigibson/data/og_dataset")
KEY_PATH = pathlib.Path(r"/scr/BEHAVIOR-1K/OmniGibson/omnigibson/data/omnigibson.key")
MODE = "USD"  # use one of "ATTRIBUTE", "USD" or "JSON"


class BaseMap:
    """
    Base map class.
    Contains basic interface for converting from map to world frame, and vise-versa
    """

    def __init__(
        self,
        map_resolution=0.1,
    ):
        """
        Args:
            map_resolution (float): map resolution
        """
        # Set internal values
        self.map_resolution = map_resolution
        self.map_size = None

    def load_map(self, *args, **kwargs):
        """
        Load's this map internally
        """
        # Run internal method and store map size
        self.map_size = self._load_map(*args, **kwargs)

    def _load_map(self, *args, **kwargs):
        """
        Arbitrary function to load this map. Should be implemented by subclass

        Returns:
            int: Size of the loaded map
        """
        raise NotImplementedError()

    def map_to_world(self, xy):
        """
        Transforms a 2D point in map reference frame into world (simulator) reference frame

        Args:
            xy (2-array or (N, 2)-array): 2D location(s) in map reference frame (in image pixel space)

        Returns:
            2-array or (N, 2)-array: 2D location(s) in world reference frame (in metric space)
        """
        dims = 0 if xy.dim() == 1 else 1
        return th.flip((xy - self.map_size / 2.0) * self.map_resolution, dims=(dims,))

    def world_to_map(self, xy):
        """
        Transforms a 2D point in world (simulator) reference frame into map reference frame

            xy: 2D location in world reference frame (metric)
        :return: 2D location in map reference frame (image)
        """

        xy = th.as_tensor(xy)
        point_wrt_map = xy / self.map_resolution + self.map_size / 2.0
        return th.flip(point_wrt_map, dims=tuple(range(point_wrt_map.dim()))).int()


class TraversableMap(BaseMap):
    """
    Traversable scene class.
    Contains the functionalities for navigation such as shortest path computation
    """

    def __init__(
        self,
        map_resolution=0.1,
        default_erosion_radius=0.0,
        trav_map_with_objects=True,
        num_waypoints=10,
        waypoint_resolution=0.2,
    ):
        """
        Args:
            map_resolution (float): map resolution in meters, each pixel represents this many meters;
                                    normally, this should be between 0.01 and 0.1
            default_erosion_radius (float): default map erosion radius in meters
            trav_map_with_objects (bool): whether to use objects or not when constructing graph
            num_waypoints (int): number of way points returned
            waypoint_resolution (float): resolution of adjacent way points
        """
        # Set internal values
        self.map_default_resolution = (
            0.01  # each pixel == 0.01m in the dataset representation
        )
        self.default_erosion_radius = default_erosion_radius
        self.trav_map_with_objects = trav_map_with_objects
        self.num_waypoints = num_waypoints
        self.waypoint_interval = int(waypoint_resolution / map_resolution)

        # Values loaded at runtime
        self.trav_map_original_size = None
        self.trav_map_size = None
        self.mesh_body_id = None
        self.floor_heights = None
        self.floor_map = None

        # Run super method
        super().__init__(map_resolution=map_resolution)

    def _load_map(self, maps_path, floor_heights=(0.0,)):
        """
        Loads the traversability maps for all floors

        Args:
            maps_path (str): Path to the folder containing the traversability maps
            floor_heights (n-array): Height(s) of the floors for this map

        Returns:
            int: Size of the loaded map
        """
        if not os.path.exists(maps_path):
            print("trav map does not exist: {}".format(maps_path))
            return

        self.floor_heights = floor_heights
        self.floor_map = []
        map_size = None
        for floor in range(len(self.floor_heights)):
            if self.trav_map_with_objects:
                # TODO: Shouldn't this be generated dynamically?
                trav_map = th.tensor(
                    cv2.imread(
                        os.path.join(maps_path, "floor_trav_{}.png".format(floor)),
                        cv2.IMREAD_GRAYSCALE,
                    )
                )
            else:
                trav_map = th.tensor(
                    cv2.imread(
                        os.path.join(
                            maps_path, "floor_trav_no_obj_{}.png".format(floor)
                        ),
                        cv2.IMREAD_GRAYSCALE,
                    )
                )

            # If we do not initialize the original size of the traversability map, we obtain it from the image
            # Then, we compute the final map size as the factor of scaling (default_resolution/resolution) times the
            # original map size
            if self.trav_map_original_size is None:
                height, width = trav_map.shape
                assert height == width, "trav map is not a square"
                self.trav_map_original_size = height
                map_size = int(
                    self.trav_map_original_size
                    * self.map_default_resolution
                    / self.map_resolution
                )

            # We resize the traversability map to the new size computed before
            trav_map = th.tensor(
                cv2.resize(trav_map.cpu().numpy(), (map_size, map_size))
            )

            # We make the pixels of the image to be either 0 or 255
            trav_map[trav_map < 255] = 0

            self.floor_map.append(trav_map)

        return map_size


def _set_xform_properties(prim, pos, quat):
    properties_to_remove = [
        "xformOp:rotateX",
        "xformOp:rotateXZY",
        "xformOp:rotateY",
        "xformOp:rotateYXZ",
        "xformOp:rotateYZX",
        "xformOp:rotateZ",
        "xformOp:rotateZYX",
        "xformOp:rotateZXY",
        "xformOp:rotateXYZ",
        "xformOp:transform",
    ]
    prop_names = prim.GetPropertyNames()
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    # TODO: wont be able to delete props for non root links on articulated objects
    for prop_name in prop_names:
        if prop_name in properties_to_remove:
            prim.RemoveProperty(prop_name)
    if "xformOp:scale" not in prop_names:
        xform_op_scale = xformable.AddXformOp(
            UsdGeom.XformOp.TypeScale, UsdGeom.XformOp.PrecisionDouble, ""
        )
        xform_op_scale.Set(Gf.Vec3d([1.0, 1.0, 1.0]))
    else:
        xform_op_scale = UsdGeom.XformOp(prim.GetAttribute("xformOp:scale"))

    if "xformOp:translate" not in prop_names:
        xform_op_translate = xformable.AddXformOp(
            UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.PrecisionDouble, ""
        )
    else:
        xform_op_translate = UsdGeom.XformOp(prim.GetAttribute("xformOp:translate"))

    if "xformOp:orient" not in prop_names:
        xform_op_rot = xformable.AddXformOp(
            UsdGeom.XformOp.TypeOrient, UsdGeom.XformOp.PrecisionDouble, ""
        )
    else:
        xform_op_rot = UsdGeom.XformOp(prim.GetAttribute("xformOp:orient"))
    xformable.SetXformOpOrder([xform_op_translate, xform_op_rot, xform_op_scale])

    position = Gf.Vec3d(*pos.tolist())
    xform_op_translate.Set(position)

    orientation = quat[[3, 0, 1, 2]].tolist()
    if xform_op_rot.GetTypeName() == "quatf":
        rotq = Gf.Quatf(*orientation)
    else:
        rotq = Gf.Quatd(*orientation)
    xform_op_rot.Set(rotq)


def decrypt_file(encrypted_filename, decrypted_filename):
    with open(KEY_PATH, "rb") as filekey:
        key = filekey.read()
    fernet = Fernet(key)

    with open(encrypted_filename, "rb") as enc_f:
        encrypted = enc_f.read()

    decrypted = fernet.decrypt(encrypted)

    with open(decrypted_filename, "wb") as decrypted_file:
        decrypted_file.write(decrypted)


def compute_bbox(prim: Usd.Prim) -> Gf.Range3d:
    """
    Compute Bounding Box using ComputeWorldBound at UsdGeom.Imageable
    See https://openusd.org/release/api/class_usd_geom_imageable.html

    Args:
        prim: A prim to compute the bounding box.
    Returns:
        A range (i.e. bounding box), see more at: https://openusd.org/release/api/class_gf_range3d.html
    """
    imageable = UsdGeom.Imageable(prim)
    time = Usd.TimeCode.Default()  # The time at which we compute the bounding box
    bound = imageable.ComputeWorldBound(time, UsdGeom.Tokens.default_)
    bound_range = bound.ComputeAlignedBox()
    return np.array(bound_range.GetMin()), np.array(bound_range.GetMax())


def keep_paths_to_all_visuals(stage):
    """
    Keeps only the paths to all prims named 'visuals' and their ancestors.

    Args:
        stage (Usd.Stage): The USD stage to modify.
    """

    # Find all prims named 'visuals'
    def find_visuals_prims(prim, visuals_paths):
        for child in prim.GetChildren():
            if child.GetName() == "visuals":
                visuals_paths.add(child.GetPath())
            find_visuals_prims(child, visuals_paths)

    visuals_paths = set()
    root_prim = stage.GetPseudoRoot()
    find_visuals_prims(root_prim, visuals_paths)

    if not visuals_paths:
        print("No prims named 'visuals' found.")
        return

    # Collect paths to keep (visuals and their ancestors)
    paths_to_keep = set()
    for path in visuals_paths:
        current_path = path
        while current_path != Sdf.Path.emptyPath:
            paths_to_keep.add(current_path)
            current_path = current_path.GetParentPath()

    # Traverse and prune
    def traverse_and_prune(prim):
        for child in prim.GetChildren():
            traverse_and_prune(child)

        # Remove the prim if its path is not in paths_to_keep
        if prim.GetPath() not in paths_to_keep:
            stage.RemovePrim(prim.GetPath())

    traverse_and_prune(root_prim)


def get_bounding_box_from_usd(input_usd, pos, quat, tempdir):
    encrypted_filename = input_usd
    fd, decrypted_filename = tempfile.mkstemp(suffix=".usd", dir=tempdir)
    os.close(fd)
    decrypt_file(encrypted_filename, decrypted_filename)
    stage = Usd.Stage.Open(str(decrypted_filename))
    prim = stage.GetDefaultPrim()
    keep_paths_to_all_visuals(stage)
    stage.Save()

    # Rotate the object by the rotmat and get the bounding box
    _set_xform_properties(prim, np.array(pos), np.array(quat))
    bb_min, bb_max = compute_bbox(prim)
    return bb_min, bb_max


def find_nearest_unoccupied_cell(grid, start_coord):
    """
    Find the nearest unoccupied cell's coordinate from a given start coordinate.

    Parameters:
    - grid: 2D numpy array where 1 represents occupied cells, 0 represents unoccupied cells
    - start_coord: tuple (row, col) of the starting coordinate

    Returns:
    - tuple of (row, col) of the nearest unoccupied cell, or None if no unoccupied cell exists
    """
    rows, cols = grid.shape
    start_row, start_col = start_coord

    # If the start cell is already unoccupied, return its coordinate
    if grid[start_row, start_col] == 255:
        return start_coord

    # Possible movement directions: 4-directional (up, right, down, left)
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    # Queue for BFS, initialized with the start coordinate
    queue = deque([(start_row, start_col, 0)])

    # Keep track of visited cells to avoid revisiting
    visited = set([(start_row, start_col)])

    while queue:
        current_row, current_col, distance = queue.popleft()

        # Check all 4 directions
        for dx, dy in directions:
            next_row = current_row + dx
            next_col = current_col + dy

            # Check if the next cell is within grid boundaries and not visited
            if (
                0 <= next_row < rows
                and 0 <= next_col < cols
                and (next_row, next_col) not in visited
            ):
                # Mark as visited
                visited.add((next_row, next_col))

                # If unoccupied cell is found, return its coordinate
                if grid[next_row, next_col] == 255:
                    return (next_row, next_col)

                # Add to queue for further exploration
                queue.append((next_row, next_col, distance + 1))

    # No unoccupied cell found
    return None


def get_distance_from_traversability(trav_map, bbox_min, bbox_max):
    bbox_corners_2d = [
        [bbox_min[0], bbox_min[1]],
        [bbox_min[0], bbox_max[1]],
        [bbox_max[0], bbox_min[1]],
        [bbox_max[0], bbox_max[1]],
    ]
    min_dist = None
    for corner_2d in bbox_corners_2d:
        corner_on_map = trav_map.world_to_map(corner_2d)

        # Clip the corner to the map coordinate range
        corner_on_map = np.clip(
            corner_on_map, 0, np.array(trav_map.floor_map[0].shape) - 1
        )

        # Find the point on the map that is closest to the corner that is marked as traversable
        closest_traversable_point_on_map = find_nearest_unoccupied_cell(
            trav_map.floor_map[0], tuple(corner_on_map.int().tolist())
        )

        # Convert back to world coordinates
        closest_traversable_point = trav_map.map_to_world(
            th.tensor(closest_traversable_point_on_map, dtype=float)
        )

        # Compute the distance between the corner and the closest traversable point
        distance = (th.tensor(corner_2d) - closest_traversable_point).tolist()
        if distance is None:
            continue

        # Update the minimum distance
        if min_dist is None or np.linalg.norm(distance) < np.linalg.norm(min_dist):
            min_dist = distance

    return min_dist


def get_tro_positions(task_file, tempdir, include_fixed=False):
    # Load the traversability map
    traversability_map = TraversableMap()
    traversability_map.load_map(maps_path=str(task_file.parent.parent / "layout"))

    example_task = json.loads(task_file.read_text())
    tro_names = [
        x
        for syn, x in example_task["metadata"]["task"]["inst_to_name"].items()
        if syn != "agent.n.01_1"
    ]
    objects = example_task["state"]["object_registry"]
    object_init_infos = example_task["objects_info"]["init_info"]
    tros = [
        (tro_name, objects[tro_name], object_init_infos[tro_name])
        for tro_name in tro_names
        if tro_name in objects
    ]
    tro_ranges = {}
    for tro_name, tro_state, tro_init in tros:
        if not include_fixed and tro_init["args"].get("fixed_base", False):
            continue

        # Get the position and rotation
        tro_pos = tro_state["root_link"]["pos"]
        tro_rot = tro_state["root_link"]["ori"]

        # Get the bounding box
        category = tro_init["args"]["category"]
        model_id = tro_init["args"]["model"]
        usd_path = (
            DATASET_DIR
            / "objects"
            / category
            / model_id
            / "usd"
            / f"{model_id}.encrypted.usd"
        )

        # Get the oriented bounding box
        bbox_min, bbox_max = get_bounding_box_from_usd(
            usd_path, tro_pos, tro_rot, tempdir
        )

        # Get the distance from the traversability map
        dist_from_trav = get_distance_from_traversability(
            traversability_map, bbox_min, bbox_max
        )

        tro_ranges[tro_name] = {
            "pos": tro_pos,
            "bbmin": bbox_min.tolist(),
            "bbmax": bbox_max.tolist(),
            "dist_from_trav": dist_from_trav,
        }

    return tro_ranges


def main():
    print("Globbing tasks")
    task_files = list(DATASET_DIR.glob("scenes/*/json/*_0_0_template.json"))
    print(f"Found {len(task_files)} tasks")

    # Scale up
    futures = {}
    with tempfile.TemporaryDirectory() as tempdir:
        with ProcessPoolExecutor() as executor:
            for p in tqdm(task_files, desc="Queueing up jobs"):
                task_name = p.stem.replace("_0_0_template", "")
                future = executor.submit(get_tro_positions, p, tempdir)
                futures[future] = task_name

            # Gather the results (with a tqdm progress bar)
            results = {}
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing results"
            ):
                task_name = futures[future]
                try:
                    results[task_name] = future.result()
                except Exception as e:
                    print(f"Error processing {task_name}: {e}")
                    traceback.print_exc()

    with open(DATASET_DIR / "task_object_infos.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
