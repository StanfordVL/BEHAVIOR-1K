import math

import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.macros import gm
from omnigibson.systems import FluidSystem, GranularSystem, MacroVisualParticleSystem
from omnigibson.utils.constants import PrimType

SYSTEM_EXAMPLES = {
    "water": FluidSystem,
    "white_rice": GranularSystem,
    # TODO: disabled due to broken particle physics, see issue #2065
    # "diced__apple": MacroPhysicalParticleSystem,
    "stain": MacroVisualParticleSystem,
}


def retrieve_obj_cfg(obj):
    return {
        "name": obj.name,
        "category": obj.category,
        "model": obj.model,
        "prim_type": obj.prim_type,
        "position": obj.get_position_orientation()[0],
        "scale": obj.scale,
        "abilities": obj.abilities,
        "visual_only": obj.visual_only,
    }


def get_random_pose(pos_low=10.0, pos_hi=20.0):
    pos = th.rand(3) * (pos_hi - pos_low) + pos_low
    ori_lo, ori_hi = -math.pi, math.pi
    orn = T.euler2quat(th.rand(3) * (ori_hi - ori_lo) + ori_lo)
    return pos, orn


def place_objA_on_objB_bbox(objA, objB, x_offset=0.0, y_offset=0.0, z_offset=0.001):
    objA.keep_still()
    objB.keep_still()
    # Reset pose if cloth object
    if objA.prim_type == PrimType.CLOTH:
        objA.root_link.reset()

    objA_aabb_center, objA_aabb_extent = objA.aabb_center, objA.aabb_extent
    objB_aabb_center, objB_aabb_extent = objB.aabb_center, objB.aabb_extent
    objA_aabb_offset = objA.get_position_orientation()[0] - objA_aabb_center

    target_objA_aabb_pos = (
        objB_aabb_center
        + th.tensor([0, 0, (objB_aabb_extent[2] + objA_aabb_extent[2]) / 2.0])
        + th.tensor([x_offset, y_offset, z_offset])
    )
    objA.set_position_orientation(position=target_objA_aabb_pos + objA_aabb_offset)


def place_obj_on_floor_plane(obj, x_offset=0.0, y_offset=0.0, z_offset=0.01):
    obj.keep_still()
    # Reset pose if cloth object
    if obj.prim_type == PrimType.CLOTH:
        obj.root_link.reset()

    obj_aabb_center, obj_aabb_extent = obj.aabb_center, obj.aabb_extent
    obj_aabb_offset = obj.get_position_orientation()[0] - obj_aabb_center

    target_obj_aabb_pos = th.tensor([0, 0, obj_aabb_extent[2] / 2.0]) + th.tensor([x_offset, y_offset, z_offset])
    obj.set_position_orientation(position=target_obj_aabb_pos + obj_aabb_offset)


def remove_all_systems(scene):
    for system in scene.active_systems.values():
        system.remove_all_particles()
    og.sim.step()


# ---------------------------------------------------------------------------
# Shared helpers for test_multiple_envs_*.py
# ---------------------------------------------------------------------------

MULTI_ENV_ROBOTS = ["fetch", "tiago", "r1", "r1pro"]

MULTI_ENV_TASK_TYPES = ["DummyTask", "PointNavigationTask", "PointReachingTask", "GraspTask"]

GRASP_OBJECTS_CFG = [
    {
        "type": "PrimitiveObject",
        "name": "grasp_obj",
        "primitive_type": "Cube",
        "rgba": [1.0, 0, 0, 1.0],
        "scale": [0.04, 0.04, 0.04],
        "mass": 0.1,
        "position": [0.5, 0.0, 0.55],
    }
]

ROBOT_JOINT_COUNTS = {"fetch": 8, "tiago": 8, "r1": 10, "r1pro": 11}


def _grasp_reset_pose_path(robot):
    """Create a temporary precached reset pose file sized for the given robot."""
    import json
    import tempfile

    n_joints = ROBOT_JOINT_COUNTS[robot]
    reset_poses = [{"joint_pos": [0.0] * n_joints, "base_pos": [0.0, 0.0, 0.0], "base_ori": [0.0, 0.0, 0.0, 1.0]}]
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(reset_poses, f)
    f.close()
    return f.name


def multi_env_task_cfg(task_type, robot="fetch"):
    """Return the task config dict for a given task type string."""
    if task_type == "GraspTask":
        return {
            "type": "GraspTask",
            "obj_name": "grasp_obj",
            "objects_config": GRASP_OBJECTS_CFG,
            "termination_config": {"max_steps": 10},
            "precached_reset_pose_path": _grasp_reset_pose_path(robot),
        }
    return {"type": task_type}


def _init_multi_env_macros():
    """Set simulator macros (only once, before first Environment is created)."""
    if og.sim is None:
        gm.RENDER_VIEWER_CAMERA = False
        gm.ENABLE_OBJECT_STATES = True
        gm.USE_GPU_DYNAMICS = True
        gm.ENABLE_FLATCACHE = False
        gm.ENABLE_TRANSITION_RULES = False
    else:
        og.sim.stop()


def setup_multi_environment(num_of_envs, robot="fetch", task_type="DummyTask", additional_objects_cfg=None):
    """Create an Environment with *num_of_envs* cloned scenes."""
    _init_multi_env_macros()

    cfg = {
        "env": {"num_envs": num_of_envs},
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "load_object_categories": ["floors", "walls"],
        },
        "robots": [{"model": robot, "obs_modalities": []}],
        "task": multi_env_task_cfg(task_type, robot=robot),
    }
    if additional_objects_cfg:
        cfg["objects"] = additional_objects_cfg

    print(f"  Setting up env: num_envs={num_of_envs}, robot={robot}, task={task_type}")
    env = og.Environment(configs=cfg)
    print("  Environment created successfully")
    return env
