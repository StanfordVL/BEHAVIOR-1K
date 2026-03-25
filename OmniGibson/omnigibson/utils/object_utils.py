"""
Helper utility functions for computing relevant object information
"""

import torch as th

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.scenes import Scene
from omnigibson.utils.geometry_utils import get_particle_positions_from_frame
from omnigibson.utils.ui_utils import create_module_logger

log = create_module_logger(module_name=__name__)


def add_object_with_parts(scene, obj, pos=None, orn=None):
    """
    Add @obj to @scene, together with any connectedpart/extrapart objects defined in its metadata.
    For connectedpart objects, sets up the attachment joint automatically after loading.

    Args:
        scene (Scene): Scene to add objects to
        obj (DatasetObject): Main object to add
        pos (None or 3-array): World position (x,y,z) for the main object's root link.
            If None, leaves the object at its default spawn position.
        orn (None or 4-array): World orientation quaternion (x,y,z,w) for the main object.
            If None, leaves the object at its default spawn orientation.

    Returns:
        list of DatasetObject: All added objects, main object first followed by its parts.
    """
    from omnigibson.object_states.attached_to import AttachedTo
    from omnigibson.objects.dataset_object import DatasetObject

    scene.add_object(obj)
    if pos is not None or orn is not None:
        cur_pos, cur_orn = obj.get_position_orientation()
        obj.set_position_orientation(
            position=pos if pos is not None else cur_pos,
            orientation=orn if orn is not None else cur_orn,
        )

    metadata = obj.metadata
    if metadata is None or not metadata.get("object_parts"):
        return [obj]

    # object_parts is stored as a list in JSON but USD round-trips it to a dict
    # with string-integer keys {"0": {...}, "1": {...}}. Handle both formats.
    raw_parts = metadata["object_parts"]
    parts = raw_parts.values() if isinstance(raw_parts, dict) else raw_parts

    if AttachedTo not in obj.states:
        log.warning(
            f"{obj.name} has connectedpart/extrapart metadata but no AttachedTo state. "
            f"Create it with abilities={{'attachable': {{}}}} for connectedpart attachment to work."
        )

    parent_pos, parent_orn = obj.get_position_orientation()
    all_objs = [obj]
    connectedparts = []

    for i, part in enumerate(parts):
        if part["type"] not in ("connectedpart", "extrapart"):
            continue

        part_bb_pos = th.tensor(part["bb_pos"], dtype=th.float32)
        part_bb_orn = th.tensor(part["bb_orn"], dtype=th.float32)
        part_bb_size = th.tensor(part["bb_size"], dtype=th.float32)

        # Scale the offset to account for non-unit parent scale (same logic as SlicingRule)
        if th.all(obj.scale == obj.scale[0]):
            scale = obj.scale
        else:
            assert T.check_quat_right_angle(part_bb_orn), (
                f"Part {part['category']}/{part['model']} of {obj.name} must have orientations that are multiples "
                f"of 90 degrees when the parent has non-uniform scale!"
            )
            scale = th.abs(T.quat2mat(part_bb_orn) @ obj.scale)

        # Compute global bounding box pose for this part
        global_bb_pos = parent_pos + T.quat2mat(parent_orn) @ (part_bb_pos * scale)
        global_bb_orn = T.quat_multiply(parent_orn, part_bb_orn)

        # connectedpart objects need attachable so AttachedTo is included in their states
        part_abilities = {"attachable": {}} if part["type"] == "connectedpart" else {}
        part_obj = DatasetObject(
            name=f"{obj.name}_{part['category']}_{i}",
            category=part["category"],
            model=part["model"],
            bounding_box=part_bb_size * scale,
            abilities=part_abilities,
        )
        scene.add_object(part_obj)
        part_obj.set_bbox_center_position_orientation(position=global_bb_pos, orientation=global_bb_orn)

        if part["type"] == "connectedpart":
            connectedparts.append(part_obj)

        all_objs.append(part_obj)

    # Step the simulator to initialize all newly added objects before setting states
    if connectedparts:
        og.sim.step()
        for part_obj in connectedparts:
            if AttachedTo in part_obj.states:
                success = part_obj.states[AttachedTo].set_value(obj, True, bypass_alignment_checking=True)
                if not success:
                    log.warning(
                        f"Failed to attach connectedpart {part_obj.name} to {obj.name}. "
                        f"Check that both objects have matching attachment meta links."
                    )
            else:
                log.warning(f"connectedpart {part_obj.name} has no AttachedTo state — missing attachment meta links?")

    return all_objs


def sample_stable_orientations(obj, n_samples=10, drop_aabb_offset=0.1):
    """
    Samples random stable orientations for obj @obj by stochastically dropping the object and recording its
    resulting orientations

    Args:
        obj (USDObject): Object whose orientations will be sampled
        n_samples (int): How many sampled orientations will be recorded
        drop_aabb_offset (float): Offset to apply in the z-direction when dropping the object

    Returns:
        n-array: (N, 4) array, where each of the N rows are sampled (x,y,z,w) stable orientations
    """
    og.sim.play()
    assert th.all(obj.scale == 1.0)
    aabb_extent = obj.aabb_extent
    radius = th.norm(aabb_extent) / 2.0
    drop_pos = th.tensor([0, 0, radius + drop_aabb_offset])
    center_offset = obj.get_position_orientation()[0] - obj.aabb_center
    drop_orientations = T.random_quaternion(n_samples)
    stable_orientations = th.zeros_like(drop_orientations)
    for i, drop_orientation in enumerate(drop_orientations):
        # Sample orientation, drop, wait to stabilize, then record
        pos = drop_pos + T.quat2mat(drop_orientation) @ center_offset
        obj.set_position_orientation(position=pos, orientation=drop_orientation)
        obj.keep_still()
        for j in range(25):
            og.sim.step()
        stable_orientations[i] = obj.get_position_orientation()[1]

    return stable_orientations


def compute_bbox_offset(obj):
    """
    Computes the base link offset of @obj, specifying the relative position of the object's bounding box center wrt to
    its root link frame, expressed in the world frame

    Args:
        obj (USDObject): Object whose bbox offset will be computed

    Returns:
        n-array: (x,y,z) offset specifying the relative position from the root link to @obj's bounding box center
    """
    og.sim.stop()
    assert th.all(obj.scale == 1.0)
    obj.set_position_orientation(position=th.zeros(3), orientation=th.tensor([0, 0, 0, 1.0]))
    return obj.aabb_center - obj.get_position_orientation()[0]


def compute_native_bbox_extent(obj):
    """
    Computes the native bounding box extent for @obj, which is the extent with the obj placed at (0, 0, 0) with
    orientation (0, 0, 0, 1) and scale (1, 1, 1)

    Args:
        obj (USDObject): Object whose native bbox extent will be computed

    Returns:
        n-array: (x,y,z) native bounding box extent
    """
    og.sim.stop()
    assert th.all(obj.scale == 1.0)
    obj.set_position_orientation(position=th.zeros(3), orientation=th.tensor([0, 0, 0, 1.0]))
    return obj.aabb_extent


def compute_base_aligned_bboxes(obj):
    link_bounding_boxes = {}
    for link_name, link in obj.links.items():
        link_bounding_boxes[link_name] = {}
        for mesh_type, mesh_list in zip(("collision", "visual"), (link.collision_meshes, link.visual_meshes)):
            pts_in_link_frame = []
            for mesh_name, mesh in mesh_list.items():
                pts = mesh.get_attribute("points")
                local_pos, local_orn = mesh.get_position_orientation(frame="parent")
                pts_in_link_frame.append(get_particle_positions_from_frame(local_pos, local_orn, mesh.scale, pts))
            pts_in_link_frame = th.cat(pts_in_link_frame, dim=0)
            max_pt = th.max(pts_in_link_frame, dim=0).values
            min_pt = th.min(pts_in_link_frame, dim=0).values
            extent = max_pt - min_pt
            center = (max_pt + min_pt) / 2.0
            transform = T.pose2mat((center, th.tensor([0, 0, 0, 1.0])))
            print(pts_in_link_frame.shape)
            link_bounding_boxes[link_name][mesh_type] = {
                "extent": extent,
                "transform": transform,
            }
    return link_bounding_boxes


def compute_obj_kinematic_metadata(obj):
    """
    Computes relevant kinematic metadata for @obj, such as stable_orientations, bounding box offsets,
    bounding box extents, and base_aligned_bboxes

    Args:
        obj (USDObject): Object whose metadata will be computed

    Returns:
        dict: Relevant metadata, with the following keys:

        - "stable_orientations": 2D (N, 4)-array of sampled stable (x,y,z,w) quaternion orientations
        - "bbox_offset": (x,y,z) relative position from the root link to @obj's bounding box center
        - "native_bbox_extent": (x,y,z) native bounding box extent
        - "base_aligned_bboxes": TODO
    """
    assert obj.scene is not None
    assert og.sim.floor_plane is not None
    assert type(obj.scene) is Scene, "An empty scene must be used in order to compute kinematic metadata!"
    assert th.all(obj.scale == 1.0), "Object must have scale [1, 1, 1] in order to compute kinematic metadata!"
    og.sim.stop()

    return {
        "stable_orientations": sample_stable_orientations(obj=obj),
        "bbox_offset": compute_bbox_offset(obj=obj),
        "native_bbox_extent": compute_native_bbox_extent(obj=obj),
        "base_aligned_bboxes": compute_base_aligned_bboxes(obj=obj),
    }
