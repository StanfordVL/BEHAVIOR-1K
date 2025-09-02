from omnigibson.robots import REGISTERED_ROBOTS
import numpy as np


def merge_scene_files(scene_a, scene_b, keep_robot_from="b"):
    """
    Merge two scene files, keeping the robot from the specified scene
    and merging all other objects.

    Design decisions:
    - Scene B is considered the most up-to-date version and its data is preferred
      when there are conflicts
    - For objects that exist in both scenes, we perform sanity checks on objects_info
      but always use Scene B's version
    - For states of objects that exist in both scenes, we always use Scene B's version
      without sanity checking
    - For system registry, we always use Scene B's version (no merging)

    Args:
        scene_a (dict): First scene file
        scene_b (dict): Second scene file (considered the most up-to-date version for conflicts)
        keep_robot_from (str or None): Which scene to keep the robot from ('a', 'b', or None)
            If None, no robot will be included

    Returns:
        dict: Merged scene file
    """
    assert isinstance(scene_a, dict), "Scene A must be a dictionary"
    assert isinstance(scene_b, dict), "Scene B must be a dictionary"
    assert keep_robot_from in ["a", "b", None], "keep_robot_from must be 'a', 'b', or None"

    # Sanity check to make sure the two scene files are compatible
    if "init_info" in scene_a and "init_info" in scene_b:
        assert (
            scene_a["init_info"]["args"]["scene_model"] == scene_b["init_info"]["args"]["scene_model"]
        ), "Scene models must match for merging"

    # Initialize merged scene file
    result = {
        "metadata": scene_b["metadata"],  # Just use metadata from scene_b
        "init_info": scene_b["init_info"],  # Start with init_info from scene_b
        "objects_info": {"init_info": {}},
        "state": {
            "pos": [0.0, 0.0, 0.0],
            "ori": [0.0, 0.0, 0.0, 1.0],
            "registry": {
                "system_registry": {},
                "object_registry": {},
            },
        },
    }

    # Update init_info settings
    result["init_info"]["args"]["load_room_types"] = None
    result["init_info"]["args"]["load_room_instances"] = None
    result["init_info"]["args"]["include_robots"] = keep_robot_from is not None

    # Find robots in both scenes
    robots_a = {}
    robots_b = {}

    for obj_name, obj in scene_a["objects_info"]["init_info"].items():
        if obj["class_name"] in REGISTERED_ROBOTS.keys():
            robots_a[obj_name] = obj

    for obj_name, obj in scene_b["objects_info"]["init_info"].items():
        if obj["class_name"] in REGISTERED_ROBOTS.keys():
            robots_b[obj_name] = obj

    # Merge non-robot objects from both scenes
    # Start with all objects from scene_a (excluding robots)
    for obj_name, obj in scene_a["objects_info"]["init_info"].items():
        if obj_name not in robots_a:
            result["objects_info"]["init_info"][obj_name] = obj

    # Add all objects from scene_b (excluding robots) and check for conflicts
    for obj_name, obj in scene_b["objects_info"]["init_info"].items():
        if obj_name not in robots_b:
            if obj_name in result["objects_info"]["init_info"]:
                # Object exists in both scenes, perform sanity check
                sanity_check_object_compatibility(obj_name, result["objects_info"]["init_info"][obj_name], obj)
            # Always use scene_b's version as the most up-to-date
            result["objects_info"]["init_info"][obj_name] = obj

    # Add robot based on keep_robot_from
    if keep_robot_from == "a" and robots_a:
        for robot_name, robot in robots_a.items():
            result["objects_info"]["init_info"][robot_name] = robot
    elif keep_robot_from == "b" and robots_b:
        for robot_name, robot in robots_b.items():
            result["objects_info"]["init_info"][robot_name] = robot

    # Handle state information
    # First add states for all non-robot objects from scene_a
    for obj_name in scene_a["objects_info"]["init_info"]:
        if obj_name not in robots_a and obj_name in scene_a["state"]["registry"]["object_registry"]:
            result["state"]["registry"]["object_registry"][obj_name] = scene_a["state"]["registry"]["object_registry"][
                obj_name
            ]

    # Then add states for all non-robot objects from scene_b
    # Always use scene_b's version without sanity checking
    for obj_name in scene_b["objects_info"]["init_info"]:
        if obj_name not in robots_b and obj_name in scene_b["state"]["registry"]["object_registry"]:
            result["state"]["registry"]["object_registry"][obj_name] = scene_b["state"]["registry"]["object_registry"][
                obj_name
            ]

    # Add robot state based on keep_robot_from
    if keep_robot_from == "a":
        for robot_name in robots_a:
            if robot_name in scene_a["state"]["registry"]["object_registry"]:
                result["state"]["registry"]["object_registry"][robot_name] = scene_a["state"]["registry"][
                    "object_registry"
                ][robot_name]
    elif keep_robot_from == "b":
        for robot_name in robots_b:
            if robot_name in scene_b["state"]["registry"]["object_registry"]:
                result["state"]["registry"]["object_registry"][robot_name] = scene_b["state"]["registry"][
                    "object_registry"
                ][robot_name]

    # Use system_registry from scene_b (considered more up-to-date)
    result["state"]["registry"]["system_registry"] = scene_b["state"]["registry"]["system_registry"]

    # Validate merged scene
    validate_merged_scene(result, require_robot=(keep_robot_from is not None))

    return result


def sanity_check_object_compatibility(obj_name, obj_a, obj_b):
    """
    Sanity check if two object definitions are compatible.
    This is only applied to objects_info, not to states.

    Args:
        obj_name (str): Name of the object
        obj_a (dict): Object definition from scene_a
        obj_b (dict): Object definition from scene_b

    Raises:
        AssertionError: If objects are incompatible
    """
    # Check basic properties must match exactly
    assert obj_a["class_module"] == obj_b["class_module"], f"Object {obj_name} has different class_module in two scenes"
    assert obj_a["class_name"] == obj_b["class_name"], f"Object {obj_name} has different class_name in two scenes"

    # For args, we'll check critical properties exactly and allow small differences in numeric values
    args_a = obj_a["args"]
    args_b = obj_b["args"]

    # Critical properties must match exactly
    assert args_a["name"] == args_b["name"], f"Object {obj_name} has different name in args"
    assert args_a["category"] == args_b["category"], f"Object {obj_name} has different category in args"
    assert args_a["model"] == args_b["model"], f"Object {obj_name} has different model in args"
    if "fixed_base" in args_a or "fixed_base" in args_b:
        assert args_a["fixed_base"] == args_b["fixed_base"], f"Object {obj_name} has different fixed_base in args"
    if "visual_only" in args_a or "visual_only" in args_b:
        assert args_a["visual_only"] == args_b["visual_only"], f"Object {obj_name} has different visual_only in args"
    assert args_a["in_rooms"] == args_b["in_rooms"], f"Object {obj_name} has different in_rooms in args"

    # For scale, they should be identical
    if "scale" in args_a and "scale" in args_b:
        scale_a = np.array(args_a["scale"])
        scale_b = np.array(args_b["scale"])
        assert np.allclose(scale_a, scale_b), f"Object {obj_name} has different scale in args"


def validate_merged_scene(scene, require_robot=True):
    """
    Validate that the merged scene is coherent and complete.

    Args:
        scene (dict): Scene to validate
        require_robot (bool): Whether to require exactly one robot
    """
    # Check that all objects in objects_info have corresponding state
    for obj_name in scene["objects_info"]["init_info"]:
        assert obj_name in scene["state"]["registry"]["object_registry"], f"Missing state for object: {obj_name}"

    # Robot check (if required)
    if require_robot:
        robot_count = 0
        for obj in scene["objects_info"]["init_info"].values():
            if obj["class_name"] in REGISTERED_ROBOTS.keys():
                robot_count += 1

        assert robot_count == 1, f"Scene must have exactly one robot, found {robot_count}"
