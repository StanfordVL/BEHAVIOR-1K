import math

import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.systems import FluidSystem, GranularSystem, MacroPhysicalParticleSystem, MacroVisualParticleSystem
from omnigibson.utils.constants import PrimType

SYSTEM_EXAMPLES = {
    "water": FluidSystem,
    "white_rice": GranularSystem,
    "diced__apple": MacroPhysicalParticleSystem,
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
