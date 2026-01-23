"""
Helper script to perform batch QA on OmniGibson objects.
"""

import hashlib
import os
import pathlib
import sys
import json
import argparse
import time
from omnigibson.prims.xform_prim import XFormPrim
import trimesh
import omnigibson as og
from omnigibson.utils.asset_utils import (
    get_all_object_categories,
    get_all_object_category_models,
)
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.object_states import REGISTERED_OBJECT_STATES, Open
from omnigibson.object_states.link_based_state_mixin import LinkBasedStateMixin
import omnigibson.utils.transform_utils as T
import omnigibson.lazy as lazy
from omnigibson.utils.asset_utils import decrypted
from omnigibson.utils.ui_utils import (
    KeyboardEventHandler,
    draw_text,
    clear_debug_drawing,
)
from omnigibson.macros import gm
from nltk.corpus import wordnet

import multiprocessing
from fs.zipfs import ZipFS
from PIL import Image
import bddl.object_taxonomy
from pathlib import Path
from nltk.corpus import wordnet as wn
import torch as th


gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = False

PI = th.tensor(th.pi)
LOW_PRECISION_ANGLE_INCREMENT = PI / 4
ANGLE_INCREMENT = PI / 90
FIXED_Y_SPACING = 0.1
INTERESTING_ABILITIES = {
    "fillable",
    "openable",
    "cloth",
    "heatSource",
    "coldSource",
    "particleApplier",
    "particleRemover",
    "toggleable",
    "particleSource",
    "particleSink",
}
JOINT_SECONDS_PER_CYCLE = 4.0

ANNOTATION_TYPE_MAPPING = {
    0: "canonical scale + ori tuning",
    1: "synset interchangeability checking",
    2: "visual mesh + collision mesh checking",
    5: "joints checking",
    6: "metalink checking",
    7: "previous complaint checking",
}

IGNORE_METALINK_TYPES = {"attachment"}
CUSTOM_INCLUDE_METALINK_TYPES = {"fillable"}
METALINK_STATES = [
    state
    for state in REGISTERED_OBJECT_STATES.values()
    if issubclass(state, LinkBasedStateMixin) and state.requires_meta_link
]
METALINK_TYPES = (
    {state.meta_link_type for state in METALINK_STATES} - IGNORE_METALINK_TYPES
).union(CUSTOM_INCLUDE_METALINK_TYPES)


def add_keyboard_callback(key, callback_fn, description):
    KeyboardEventHandler.add_keyboard_callback(key=key, callback_fn=callback_fn)
    key_name = str(key).replace("KeyboardInput.", "").replace("_", " ")
    print(f"Press {key_name} to {description}.")


class BatchQAViewer:
    def __init__(
        self, record_path, your_id, total_ids, seed, pipeline_root, annotation_type
    ):
        self.env = None
        self.pipeline_root = pipeline_root
        assert (
            annotation_type in ANNOTATION_TYPE_MAPPING
        ), f"Got invalid annotation_type. Expected one of: {ANNOTATION_TYPE_MAPPING}"
        self.annotation_type = annotation_type
        self.taxonomy = bddl.object_taxonomy.ObjectTaxonomy()
        self.record_path = record_path + f"_{self.annotation_type}"
        self.your_id = your_id
        self.total_ids = total_ids
        self.seed = seed
        self.all_objs = {
            (cat, model)
            for cat in get_all_object_categories()
            for model in get_all_object_category_models(cat)
        }
        filtered_objs_by_id = {
            this_id: sorted(
                {
                    (cat, model)
                    for cat, model in self.all_objs
                    if int(hashlib.md5((cat + self.seed).encode()).hexdigest(), 16)
                    % self.total_ids
                    == this_id
                }
            )
            for this_id in range(self.total_ids)
        }
        # print("Filtered objects by id:", {k: len(v) for k, v in filtered_objs_by_id.items()})
        self.filtered_objs = filtered_objs_by_id[self.your_id]
        self.processed_objects = self.load_processed_objects()
        print("-" * 80)
        print("IMPORTANT: VERIFY THIS NUMBER!")
        print("There are a total of", len(self.filtered_objs), "objects in this batch.")
        print(
            f"You are running with annotation type: {self.annotation_type}: {ANNOTATION_TYPE_MAPPING[self.annotation_type]}"
        )
        print("You are running the 5.0.3 version of this script.")
        print("-" * 80)
        input("Press Enter to continue...")
        self.complaint_handler = ObjectComplaintHandler(pipeline_root)

        # Reference objects
        self.human = None
        self.phone = None

        # Precision mode
        self.precision_mode = False

        # Camera parameters
        self.pan = th.tensor(0.0)
        self.tilt = th.tensor(0.0)
        self.dist = th.tensor(3.0)

    @property
    def angle_increment(self):
        return ANGLE_INCREMENT if self.precision_mode else LOW_PRECISION_ANGLE_INCREMENT

    @property
    def scale_increment(self):
        return th.tensor(1.1) if self.precision_mode else th.tensor(10.0)

    def _toggle_precision(self):
        self.precision_mode = not self.precision_mode
        print(f"Precision mode: {'ON' if self.precision_mode else 'OFF'}")

    def load_processed_objects(self):
        processed_objs = set()
        for cat, mdl in self.filtered_objs:
            path = os.path.join(self.record_path, cat, mdl + ".json")
            if os.path.exists(path):
                processed_objs.add(mdl)
        return processed_objs

    @property
    def remaining_objects(self):
        return sorted(
            {
                (cat, model)
                for cat, model in self.filtered_objs
                if model not in self.processed_objects
            }
        )

    def group_objects_by_category(self, objects):
        grouped_objs = {}
        for cat, model in objects:
            if cat not in grouped_objs:
                grouped_objs[cat] = []
            grouped_objs[cat].append(model)
        return grouped_objs

    def import_objects(self, category, batch):
        all_objects = []
        for i, obj_model in enumerate(batch):
            obj = DatasetObject(
                name="obj_" + obj_model,
                category=category,
                model=obj_model,
                visual_only=True,
            )
            self.env.scene.add_object(obj)
            all_objects.append(obj)

            # Make the collision meshes also visible
            for link in obj.links.values():
                for mesh in link.visual_meshes.values():
                    mesh.purpose = "default"
                    mesh.visible = not link.is_meta_link
                for mesh in link.collision_meshes.values():
                    mesh.purpose = "default"
                    mesh.visible = False

        return all_objects

    def position_objects(self, all_objects, should_draw=True):
        y_coordinate = 0
        prev_obj_radius = 0

        if should_draw:
            clear_debug_drawing()

        for index, obj in enumerate(all_objects):
            obj_radius = th.linalg.norm(obj.aabb_extent[:2]) / 2.0
            if index != 0:
                y_coordinate += prev_obj_radius + FIXED_Y_SPACING + obj_radius
            obj_in_min = obj.get_position_orientation()[0] - obj.aabb[0]

            obj.set_position_orientation(
                position=[obj_in_min[0], y_coordinate, obj_in_min[2] + 0.05]
            )

            if should_draw:
                draw_text(
                    obj.name.replace("obj_", ""),
                    [0, y_coordinate, -0.1],
                    T.euler2quat(th.tensor([th.pi / 2, 0, th.pi / 2])),
                    color=(1.0, 0.0, 0.0, 1.0),
                    line_size=3.0,
                    anchor="topcenter",
                    max_width=obj_radius,
                    max_height=0.2,
                )

            prev_obj_radius = obj_radius

        # Write the category name across the total dimension
        min_y = all_objects[0].aabb[0][1]
        max_y = all_objects[-1].aabb[1][1]
        center_y = (min_y + max_y) / 2
        length_y = max_y - min_y
        if should_draw:
            draw_text(
                obj.category,
                [0, center_y, -0.3],
                T.euler2quat(th.tensor([th.pi / 2, 0, th.pi / 2])),
                color=(1.0, 0.0, 0.0, 1.0),
                line_size=3.0,
                anchor="topcenter",
                max_width=length_y,
            )
        og.sim.step()
        og.sim.step()
        og.sim.step()

    def save_object_results(self, obj, orientation, scale, complaints):
        orientation = obj.get_position_orientation()[1]
        scale = obj.scale
        if not os.path.exists(os.path.join(self.record_path, obj.category)):
            os.makedirs(os.path.join(self.record_path, obj.category))
        with open(
            os.path.join(self.record_path, obj.category, obj.model + ".json"), "w"
        ) as f:
            json.dump(
                {
                    "orientation": orientation.tolist(),
                    "scale": scale.tolist(),
                    "complaints": complaints,
                },
                f,
                indent=4,
            )

    def save_empty_results(self, category, model):
        if not os.path.exists(os.path.join(self.record_path, category)):
            os.makedirs(os.path.join(self.record_path, category))
        with open(os.path.join(self.record_path, category, model + ".json"), "w") as f:
            json.dump(
                {
                    "orientation": [0, 0, 0, 1],
                    "scale": [1, 1, 1],
                    "complaints": [],
                },
                f,
                indent=4,
            )

    def set_camera_bindings(self, default_dist=3.0):
        self.pan, self.tilt, self.dist = PI, th.tensor(0.0), th.tensor(default_dist)

        def update_camera(d_pan, d_tilt, d_dist):
            self.pan = (self.pan + d_pan) % (2 * th.pi)
            self.tilt = th.clip(self.tilt + d_tilt, -th.pi / 2, th.pi / 2)
            self.dist = th.clip(self.dist + d_dist, 0, 100)

        def reset_camera():
            self.pan, self.tilt, self.dist = PI, th.tensor(0.0), th.tensor(default_dist)

        add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.UP,
            callback_fn=lambda: update_camera(0, ANGLE_INCREMENT, 0),
            description="tilt camera up",
        )
        add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.DOWN,
            callback_fn=lambda: update_camera(0, -ANGLE_INCREMENT, 0),
            description="tilt camera down",
        )
        add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.LEFT,
            callback_fn=lambda: update_camera(-ANGLE_INCREMENT, 0, 0),
            description="pan camera left",
        )
        add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.RIGHT,
            callback_fn=lambda: update_camera(ANGLE_INCREMENT, 0, 0),
            description="pan camera right",
        )
        add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.PAGE_DOWN,
            callback_fn=lambda: update_camera(0, 0, 0.1),
            description="zoom in",
        )
        add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.PAGE_UP,
            callback_fn=lambda: update_camera(0, 0, -0.1),
            description="zoom out",
        )
        add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.END,
            callback_fn=lambda: reset_camera(),
            description="reset camera",
        )

    def update_camera(self, target):
        # Get the camera position by starting at the target point and moving back by the distance
        # along the negative pan / tilt direction
        camera_pos = target - self.dist * th.tensor(
            [
                th.cos(self.pan) * th.cos(self.tilt),
                th.sin(self.pan) * th.cos(self.tilt),
                -th.sin(self.tilt),
            ]
        )
        # Camera matrix: note that this is the OpenGL frame, so the camera is looking down the negative z-axis
        # and the up vector is the positive y-axis.
        weird_camera_frame = T.mat2quat(
            th.tensor(
                [
                    [0, 0, -1],
                    [-1, 0, 0],
                    [0, 1, 0],
                ]
            )
        )
        camera_orn = T.euler2quat(th.tensor([0.0, self.tilt, self.pan]))
        camera_orn = T.quat_multiply(camera_orn, weird_camera_frame)
        og.sim.viewer_camera.set_position_orientation(camera_pos, camera_orn)

    def whole_batch_preview(self, all_objects):
        KeyboardEventHandler.initialize()
        self.set_camera_bindings()

        done = False
        skip = False
        y_min = th.min(th.tensor([obj.aabb[0][1] for obj in all_objects]))
        y_max = th.max(th.tensor([obj.aabb[1][1] for obj in all_objects]))
        average_pos = th.mean(th.stack([obj.aabb_center for obj in all_objects]), dim=0)

        offset = 0.0

        def _set_done():
            nonlocal done
            done = True

        def _set_skip():
            nonlocal skip
            skip = True
            _set_done()

        def change_offset(d_offset):
            nonlocal offset
            offset += d_offset

        # Set done when the user presses 'C'
        add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.ENTER,
            callback_fn=_set_done,
            description="continue to object editing",
        )
        add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.HOME,
            callback_fn=_set_skip,
            description="skip to next category",
        )
        add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.LEFT_BRACKET,
            callback_fn=lambda: change_offset(-1),
            description="pan camera left",
        )
        add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.RIGHT_BRACKET,
            callback_fn=lambda: change_offset(1),
            description="pan camera right",
        )

        def _rotate_category(axis, angle):
            current_rot = all_objects[0].get_position_orientation()[1]
            rotation_delta = th.zeros(3)
            rotation_delta["xyz".index(axis)] = angle
            new_rot = T.quat_multiply(T.euler2quat(rotation_delta), current_rot)
            # Round the new rotation to the nearest degree
            rounded_rot = T.euler2quat(
                th.deg2rad(th.round(th.rad2deg(T.quat2euler(new_rot))))
            )

            for obj in all_objects:
                obj.set_position_orientation(orientation=rounded_rot)

            # Reposition everything
            self.position_objects(all_objects)
            self.position_reference_objects(target_y=all_objects[0].aabb_center[1])

        add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.O,
            callback_fn=lambda: _rotate_category("z", self.angle_increment),
            description="rotate category counterclockwise around z-axis",
        )
        add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.P,
            callback_fn=lambda: _rotate_category("z", -self.angle_increment),
            description="rotate category clockwise around z-axis",
        )
        add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.K,
            callback_fn=lambda: _rotate_category("y", self.angle_increment),
            description="rotate category counterclockwise around y-axis",
        )
        add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.L,
            callback_fn=lambda: _rotate_category("y", -self.angle_increment),
            description="rotate category clockwise around y-axis",
        )
        add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.COMMA,
            callback_fn=lambda: _rotate_category("x", self.angle_increment),
            description="rotate category counterclockwise around x-axis",
        )
        add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.PERIOD,
            callback_fn=lambda: _rotate_category("x", -self.angle_increment),
            description="rotate category clockwise around x-axis",
        )
        add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.BACKSLASH,
            callback_fn=lambda: [
                obj.set_position_orientation(orientation=[0, 0, 0, 1])
                for obj in all_objects
            ],
            description="reset category orientation",
        )

        while not done:
            y = th.clip(y_min + offset, y_min, y_max)
            target = th.tensor([average_pos[0], y, average_pos[2]])
            self.update_camera(target)
            og.sim.step()

        KeyboardEventHandler.reset()
        return skip

    def evaluate_single_object(self, all_objects, i):
        obj = all_objects[i]
        print(f"\n\n\n\nNow editing object {obj.name.replace('obj_', '')}\n")

        # If we're checking joints and the object has none, skip
        synset = self.taxonomy.get_synset_from_category(category=obj.category)
        terminate_early = False
        metalink_types = None
        if self.annotation_type == 5 and obj.n_joints == 0:  # joints checking
            terminate_early = True
        elif self.annotation_type == 6:
            metalink_types = set()
            for link_name in obj.links.keys():
                for metalink_type in METALINK_TYPES:
                    if metalink_type in link_name:
                        metalink_types.add(metalink_type)
            # metalink_types = {state.meta_link_type for state in obj.states.keys() if issubclass(state, LinkBasedStateMixin) and state.requires_meta_link} - IGNORE_METALINK_TYPES
            if len(metalink_types) == 0:
                terminate_early = True
        elif self.annotation_type == 7:
            previous_complaints = self.complaint_handler._get_existing_complaints(
                obj.model
            )
            if len(previous_complaints) == 0:
                terminate_early = True

        if terminate_early:
            self.save_empty_results(obj.category, obj.model)
            self.processed_objects.add(obj.name.replace("obj_", ""))
            return  # skip this model

        KeyboardEventHandler.initialize()
        self.set_camera_bindings(default_dist=obj.aabb_extent[0] * 2.5)

        done = False
        should_show_photo = False
        quick_complaints = []
        scale_queue = []  # We queue scales to apply them all at once to avoid .play getting called from .step
        obj_first_pca_angle_map = {}
        obj_second_pca_angle_map = {}

        # Immediately set done to be True if we're only handling prior complaints
        if self.annotation_type == 7:
            done = True

        def _set_done():
            nonlocal done
            done = True

        def _set_complaint(message):
            quick_complaints.append(
                {
                    "object": f"{obj.category}-{obj.model}",
                    "message": "QC: " + message,
                    "complaint": "quick complaint added during category & collision QA",
                    "processed": False,
                    "new": True,
                }
            )
            print("Added", message, "complaint.\n")

        def _toggle_gravity():
            obj.visual_only = not obj.visual_only

            # Reposition everything
            self.position_objects(all_objects)
            self.position_reference_objects(target_y=obj.aabb_center[1])

        collision_visibility = True

        def _toggle_collision_visibility():
            # Disable all the visual meshes and enable all the collision ones
            nonlocal collision_visibility
            collision_visibility = not collision_visibility
            for link in obj.links.values():
                for mesh in link.visual_meshes.values():
                    mesh.visible = not collision_visibility
                for mesh in link.collision_meshes.values():
                    mesh.visible = collision_visibility

        # Warm this up
        _toggle_collision_visibility()

        meta_visibility = True

        def _toggle_meta_visibility():
            # Toggle the visibility of all the links
            nonlocal meta_visibility
            meta_visibility = not meta_visibility
            for link in obj.links.values():
                if not link.is_meta_link:
                    continue
                for mesh in link.visual_meshes.values():
                    mesh.visible = meta_visibility
                for mesh in link.collision_meshes.values():
                    mesh.visible = meta_visibility

        # Warm this up
        _toggle_meta_visibility()

        joint_position_seed = th.tensor(
            0.0
        )  # monotonically increasing, to be passed into th.sin
        joints_moving = False

        def _toggle_joints():
            nonlocal joints_moving
            joints_moving = not joints_moving

        def _align_to_pca(pca_axis):
            if pca_axis == 1 and obj in obj_first_pca_angle_map:
                angle = obj_first_pca_angle_map[obj]
            elif pca_axis == 2 and obj in obj_second_pca_angle_map:
                angle = obj_second_pca_angle_map[obj]
            else:
                # Collecting points from the object
                points = []
                for link in obj.links.values():
                    for mesh in link.visual_meshes.values():
                        mesh_points = mesh.prim.GetAttribute("points").Get()
                        pos, ori = mesh.get_position_orientation()
                        transform = T.pose2mat((pos, ori))
                        if mesh_points is None or len(mesh_points) == 0:
                            continue
                        points.append(
                            trimesh.transformations.transform_points(
                                mesh_points, transform
                            )
                        )
                points = th.concat(points, dim=0)

                # Apply PCA to 3D points
                from sklearn.decomposition import PCA

                pca = PCA(n_components=3)
                pca.fit(points)

                if pca_axis == 1:
                    # The first principal component
                    pc = pca.components_[0]
                else:
                    # The second principal component
                    pc = pca.components_[1]

                # Compute the angle between the first principal component and the x-axis
                angle = th.arctan2(pc[1], pc[0])

                if pca_axis == 1:
                    obj_first_pca_angle_map[obj] = angle
                else:
                    obj_second_pca_angle_map[obj] = angle

            # Create a quaternion from this angle
            rot = T.euler2quat(th.tensor([0, 0, angle]))

            # Apply the rotation to the object
            obj.set_position_orientation(orientation=rot)

            # Reposition everything
            self.position_objects(all_objects)
            self.position_reference_objects(target_y=obj.aabb_center[1])

        def _rotate_object(axis, angle):
            current_rot = obj.get_position_orientation()[1]
            rotation_delta = th.zeros(3)
            rotation_delta["xyz".index(axis)] = angle
            new_rot = T.quat_multiply(T.euler2quat(rotation_delta), current_rot)
            # Round the new rotation to the nearest degree
            rounded_rot = T.euler2quat(
                th.deg2rad(th.round(th.rad2deg(T.quat2euler(new_rot))))
            )
            obj.set_position_orientation(orientation=rounded_rot)

            # Reposition everything
            self.position_objects(all_objects)
            self.position_reference_objects(target_y=obj.aabb_center[1])

        def _set_scale(new_scale):
            object_poses = {
                o: o.get_position_orientation() for o in self.env.scene.objects
            }
            og.sim.stop()
            obj.scale = new_scale
            og.sim.play()
            for o, pose in object_poses.items():
                o.set_position_orientation(*pose)

            # Reposition everything
            self.position_objects(all_objects)
            self.position_reference_objects(target_y=obj.aabb_center[1])
            self.dist = obj.aabb_extent[0] * 2.5

        def _show_photo():
            nonlocal should_show_photo
            should_show_photo = True

        # Other controls
        add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.ENTER,
            callback_fn=_set_done,
            description="continue to complaint process",
        )
        # add_keyboard_callback(
        #     key=lazy.carb.input.KeyboardInput.NUMPAD_7,
        #     callback_fn=lambda: _align_to_pca(1),
        #     description="align object to first principal component"
        # )
        # add_keyboard_callback(
        #     key=lazy.carb.input.KeyboardInput.NUMPAD_9,
        #     callback_fn=lambda: _align_to_pca(2),
        #     description="align object to second principal component"
        # )

        if self.annotation_type == 0:
            add_keyboard_callback(
                key=lazy.carb.input.KeyboardInput.O,
                callback_fn=lambda: _rotate_object("z", self.angle_increment),
                description="rotate object counterclockwise around z-axis",
            )
            add_keyboard_callback(
                key=lazy.carb.input.KeyboardInput.P,
                callback_fn=lambda: _rotate_object("z", -self.angle_increment),
                description="rotate object clockwise around z-axis",
            )
            add_keyboard_callback(
                key=lazy.carb.input.KeyboardInput.K,
                callback_fn=lambda: _rotate_object("y", self.angle_increment),
                description="rotate object counterclockwise around y-axis",
            )
            add_keyboard_callback(
                key=lazy.carb.input.KeyboardInput.L,
                callback_fn=lambda: _rotate_object("y", -self.angle_increment),
                description="rotate object clockwise around y-axis",
            )
            add_keyboard_callback(
                key=lazy.carb.input.KeyboardInput.COMMA,
                callback_fn=lambda: _rotate_object("x", self.angle_increment),
                description="rotate object counterclockwise around x-axis",
            )
            add_keyboard_callback(
                key=lazy.carb.input.KeyboardInput.PERIOD,
                callback_fn=lambda: _rotate_object("x", -self.angle_increment),
                description="rotate object clockwise around x-axis",
            )
            add_keyboard_callback(
                key=lazy.carb.input.KeyboardInput.BACKSLASH,
                callback_fn=lambda: obj.set_position_orientation(
                    orientation=[0, 0, 0, 1]
                ),
                description="reset object orientation",
            )
            add_keyboard_callback(
                key=lazy.carb.input.KeyboardInput.G,
                callback_fn=_toggle_gravity,
                description="toggle gravity for selected object",
            )
            add_keyboard_callback(
                key=lazy.carb.input.KeyboardInput.T,
                callback_fn=lambda: self._toggle_precision(),
                description="toggle precision mode",
            )
            add_keyboard_callback(
                key=lazy.carb.input.KeyboardInput.RIGHT_BRACKET,
                callback_fn=lambda: scale_queue.append(self.scale_increment),
                description="increase object scale",
            )
            add_keyboard_callback(
                key=lazy.carb.input.KeyboardInput.LEFT_BRACKET,
                callback_fn=lambda: scale_queue.append(1 / self.scale_increment),
                description="decrease object scale",
            )
            add_keyboard_callback(
                key=lazy.carb.input.KeyboardInput.KEY_0,
                callback_fn=lambda: scale_queue.append(th.tensor(0)),
                description="reset object scale",
            )

        elif self.annotation_type == 1:
            add_keyboard_callback(
                key=lazy.carb.input.KeyboardInput.KEY_1,
                callback_fn=lambda: _set_complaint("category"),
                description="add a category or synset complaint",
            )

        elif self.annotation_type == 2:
            add_keyboard_callback(
                key=lazy.carb.input.KeyboardInput.C,
                callback_fn=_toggle_collision_visibility,
                description="toggle collision mesh visibility",
            )
            add_keyboard_callback(
                key=lazy.carb.input.KeyboardInput.KEY_2,
                callback_fn=lambda: _set_complaint("appearance"),
                description="add a visual appearance complaint",
            )
            add_keyboard_callback(
                key=lazy.carb.input.KeyboardInput.KEY_3,
                callback_fn=lambda: _set_complaint("handle"),
                description="add a handle-specific collision mesh complaint",
            )
            add_keyboard_callback(
                key=lazy.carb.input.KeyboardInput.KEY_4,
                callback_fn=lambda: _set_complaint("collision"),
                description="add a general collision complaint",
            )

        elif self.annotation_type == 5:
            add_keyboard_callback(
                key=lazy.carb.input.KeyboardInput.J,
                callback_fn=_toggle_joints,
                description="toggle joint movement",
            )
            add_keyboard_callback(
                key=lazy.carb.input.KeyboardInput.KEY_5,
                callback_fn=lambda: _set_complaint("joint"),
                description="add a joint complaint",
            )
        elif self.annotation_type == 6:
            add_keyboard_callback(
                key=lazy.carb.input.KeyboardInput.M,
                callback_fn=_toggle_meta_visibility,
                description="toggle meta link mesh visibility",
            )
            add_keyboard_callback(
                key=lazy.carb.input.KeyboardInput.KEY_6,
                callback_fn=lambda: _set_complaint("metalink"),
                description="add a meta link complaint",
            )

        add_keyboard_callback(
            key=lazy.carb.input.KeyboardInput.KEY_7,
            callback_fn=lambda: _set_complaint("unknown"),
            description="add a generic complaint to be re-examined by the team",
        )
        print("-" * 80)

        if self.annotation_type == 0:
            print(
                "\nEdit the currently selected object to match the realistic size of the category."
            )
            print(
                "It should also face the same way as the other objects, and should be stable in its"
            )
            print("canonical orientation.\n")

        # position reference objects to be next to the inspected object
        self.position_reference_objects(target_y=obj.aabb_center[1])

        # Prompt the user to fix the scale and orientation of the object. Keep the camera in position, too.
        step = 0
        while not done:
            og.sim.step()

            if should_show_photo:
                should_show_photo = False
                # First, load the background image
                background_path = os.path.join(
                    self.pipeline_root, "b1k_pipeline", "tools", "background.jpg"
                )
                background = Image.open(background_path).resize((800, 800))

                # Open the zip file
                zip_path = os.path.join(
                    self.pipeline_root, "artifacts", "pipeline", "max_object_images.zip"
                )
                with ZipFS(zip_path) as zip_fs:
                    # Find and show photos of this object.
                    image_paths = sorted(
                        [
                            x
                            for x in zip_fs.listdir("/")
                            if obj.name.replace("obj_", "") in x
                        ]
                    )
                    for image_path in image_paths:
                        with zip_fs.open(image_path, "rb") as f:
                            image = background.copy()
                            max_image = Image.open(f)
                            image.paste(max_image, (0, 0), mask=max_image)
                            image.show()

            # Apply any scale changes
            if len(scale_queue) > 0:
                scale = th.tensor(obj.scale)
                if any(s == 0 for s in scale_queue):
                    scale = th.ones(3)  # Reset the scale
                else:
                    scale *= th.prod(th.tensor(scale_queue))
                _set_scale(scale)
                scale_queue.clear()

            # Apply joint motion
            if obj.n_dof > 0:
                if joints_moving:
                    joint_position_seed += (
                        2 * th.pi * og.sim.get_rendering_dt() / JOINT_SECONDS_PER_CYCLE
                    )
                joint_positions = th.ones(obj.n_dof) * th.sin(joint_position_seed)
                obj.set_joint_positions(
                    positions=joint_positions, normalized=True, drive=False
                )

            self.update_camera(obj.aabb_center)

            if self.annotation_type == 0:
                if step % 100 == 0:
                    scale_str = (
                        f"{obj.scale[0]:.2f}, {obj.scale[1]:.2f}, {obj.scale[2]:.2f}"
                    )
                    rotation = th.rad2deg(
                        T.quat2euler(obj.get_position_orientation()[1])
                    )
                    rotation_str = (
                        f"{rotation[0]:.2f}, {rotation[1]:.2f}, {rotation[2]:.2f}"
                    )
                    bbox_str = f"{obj.aabb_extent[0] * 100:.2f}cm, {obj.aabb_extent[1] * 100:.2f}cm, {obj.aabb_extent[2] * 100:.2f}cm"
                    print(
                        f"Bounding box extent: {bbox_str}. Scale: {scale_str}. Rotation: {rotation_str}              ",
                        end="\r",
                    )
            elif self.annotation_type == 1:
                if step == 0:
                    wordnet_synsets = wordnet.synsets(synset.split(".")[0])
                    definition = None
                    for wordnet_synset in wordnet_synsets:
                        if wordnet_synset.name() == synset:
                            definition = wordnet_synset.definition()
                            break
                    print(f"expected category / synset: {obj.category} / {synset}")
                    print(f"Definition: {definition}")
            elif self.annotation_type == 6:
                if step == 0:
                    print(f"Object has metalinks: {metalink_types}")
            step += 1
        print()
        print("-" * 80)

        # Now we're done with bbox and scale and orientation. Save the data.
        orientation = obj.get_position_orientation()[1]
        scale = obj.scale

        # Set the object back to visual only and reposition everything
        obj.visual_only = True
        self.position_objects(all_objects)
        self.position_reference_objects(target_y=obj.aabb_center[1])

        # Set the keyboard bindings back to camera only
        KeyboardEventHandler.reset()
        KeyboardEventHandler.initialize()
        self.set_camera_bindings(default_dist=obj.aabb_extent[0] * 2.5)

        # Launch the complaint thread
        multiprocess_queue = multiprocessing.Queue()
        questions = self.complaint_handler.get_questions(obj)
        handle_previous_complaints = self.annotation_type == 7
        complaint_process = multiprocessing.Process(
            target=self.complaint_handler.process_complaints,
            args=[
                multiprocess_queue,
                obj.category,
                obj.name.replace("obj_", ""),
                questions,
                quick_complaints,
                sys.stdin.fileno(),
                handle_previous_complaints,
            ],
            daemon=True,
        )
        complaint_process.start()

        # Wait to receive the complaints
        step = 0
        while complaint_process.is_alive():
            step += 1
            og.sim.step()
            self.update_camera(obj.aabb_center)

            # During this part, we want to be moving the joints
            # for joint in obj.joints.values():
            #     seconds_since_start = step * og.sim.get_rendering_dt()
            #     interpolation_point = 0.5 * th.sin(seconds_since_start / JOINT_SECONDS_PER_CYCLE * 2 * PI) + 0.5
            #     target_pos = joint.lower_limit + interpolation_point * (joint.upper_limit - joint.lower_limit)
            #     joint.set_pos(target_pos)

            if not multiprocess_queue.empty():
                # Got a response, we can stop.
                break

        # Wait for the complaint process to finish to not have to kill it
        time.sleep(0.1)

        # If the complaint process is still alive, kill it.
        if complaint_process.is_alive():
            # Join the finished thread
            complaint_process.join()
        assert (
            complaint_process.exitcode == 0
        ), "Complaint process exited with error code."

        assert (
            not multiprocess_queue.empty()
        ), "Complaint process did not return a message."
        message = multiprocess_queue.get()
        complaints = json.loads(message)

        # Save the object results / mark the object as processed
        self.save_object_results(obj, orientation, scale, complaints)
        self.processed_objects.add(obj.name.replace("obj_", ""))

        KeyboardEventHandler.reset()

    def evaluate_batch(self, batch, category):
        synset = self.taxonomy.get_synset_from_category(category=category)
        abilities = self.taxonomy.get_abilities(synset=synset)

        # If we're checking joints or metalinks, skip this entire batch is no object has either joints nor metalinks
        if self.annotation_type in {5, 6, 7}:
            if self.annotation_type == 5:  # joints checking
                check_is_valid = lambda prim, category, model: Open.is_compatible_asset(
                    prim
                )[0]
            elif self.annotation_type == 6:

                def has_metalink_link(prim):
                    def _find_prims_with_condition(condition, root_prim):
                        found_prims = []
                        if condition(root_prim):
                            found_prims.append(root_prim)

                        for child in root_prim.GetChildren():
                            found_prims += _find_prims_with_condition(
                                condition=condition, root_prim=child
                            )

                        return found_prims

                    def find_all_rigid_bodies(root_prim):
                        return _find_prims_with_condition(
                            condition=is_rigid_body,
                            root_prim=root_prim,
                        )
                        return False

                    def is_rigid_body(prim):
                        prim_type = prim.GetPrimTypeInfo().GetTypeName().lower()
                        has_rigid_api = prim.HasAPI(lazy.pxr.UsdPhysics.RigidBodyAPI)
                        return "xform" in prim_type and has_rigid_api

                    rb_prims = find_all_rigid_bodies(root_prim=prim)
                    for rb_prim in rb_prims:
                        name = rb_prim.GetName().lower()
                        for metalink_type in METALINK_TYPES:
                            if metalink_type in name:
                                return True

                    return False

                check_is_valid = lambda prim, category, model: has_metalink_link(prim)

            elif self.annotation_type == 7:
                check_is_valid = (
                    lambda prim, category, model: len(
                        self.complaint_handler._get_existing_complaints(model)
                    )
                    > 0
                )

            else:
                raise ValueError()

            has_valid_model = False
            for model in batch:
                usd_path = DatasetObject.get_usd_path(category=category, model=model)
                usd_path = usd_path.replace(".usd", ".encrypted.usd")
                with decrypted(usd_path) as fpath:
                    stage = lazy.pxr.Usd.Stage.Open(fpath)
                    prim = stage.GetDefaultPrim()
                    if check_is_valid(prim, category, model):
                        has_valid_model = True
                        break

            if not has_valid_model:
                for model in batch:
                    self.save_empty_results(category, model)
                    self.processed_objects.add(model)
                return False  # don't skip the rest of the category

        all_objects = self.import_objects(category, batch)
        og.sim.step()
        self.position_objects(all_objects)
        self.position_reference_objects(target_y=0.0)

        # Phase 1: Continuously pan across the full category to show the user all objects
        # Only need to show all objects for canonical orientation purposes
        skip = False
        if self.annotation_type == 0:  # canonical orientation / scale checking
            skip = self.whole_batch_preview(all_objects)

        if not skip:
            # Phase 2: Allow the user to interact with the objects one by one
            for i in range(len(all_objects)):
                self.evaluate_single_object(all_objects, i)

        # Clean up.
        for obj in all_objects:
            self.env.scene.remove_object(obj)

        return skip

    def position_reference_objects(self, target_y):
        obj_in_center_frame = (
            self.phone.get_position_orientation()[0] - self.phone.aabb_center
        )
        obj_in_min_frame = self.phone.get_position_orientation()[0] - self.phone.aabb[0]
        obj_in_max_frame = self.phone.get_position_orientation()[0] - self.phone.aabb[1]
        self.phone.set_position_orientation(
            position=[
                -0.05 + obj_in_max_frame[0],
                target_y + obj_in_center_frame[1],
                obj_in_min_frame[2],
            ]
        )

        human_in_center_frame = (
            self.human.get_position_orientation()[0] - self.human.aabb_center
        )
        human_in_min_frame = (
            self.human.get_position_orientation()[0] - self.human.aabb[0]
        )
        human_in_max_frame = (
            self.human.get_position_orientation()[0] - self.human.aabb[1]
        )
        self.human.set_position_orientation(
            position=[
                -0.1 + self.phone.aabb_extent[0] + human_in_max_frame[0],
                target_y + human_in_center_frame[1],
                human_in_min_frame[2],
            ]
        )

    def add_reference_objects(self):
        # Add a cellphone into the scene
        phone = DatasetObject(
            name="dbhfuh",
            category="cell_phone",
            model="dbhfuh",
            visual_only=True,
        )
        self.env.scene.add_object(phone)
        og.sim.step()
        phone.links["meta__base_link_togglebutton_0_0_link"].visible = False
        phone.set_position_orientation(
            orientation=T.euler2quat(th.tensor([th.pi / 2, th.pi / 2, 0]))
        )

        # Add a human into the scene
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        human_usd_path = os.path.join(curr_dir, "HumanFemale/HumanFemale.usd")
        human_prim_path = "/World/scene_0/human"
        lazy.omni.isaac.core.utils.stage.add_reference_to_stage(
            usd_path=human_usd_path, prim_path=human_prim_path, prim_type="Xform"
        )
        human_prim = XFormPrim(name="human", relative_prim_path="/human")
        human_prim.load(self.env.scene)
        human_prim.set_position_orientation(
            orientation=T.euler2quat(th.tensor([0, 0, th.pi / 2]))
        )
        human_prim.scale = [0.012, 0.012, 0.012]
        og.sim.step()

        return human_prim, phone

    def run(self):
        if self.your_id < 0 or self.your_id >= self.total_ids:
            print("Invalid id!")
            sys.exit(1)

        print(
            f"{len(self.processed_objects)}/{len(self.filtered_objs)} objects processed. {len(self.remaining_objects)} objects remaining."
        )

        # Load the environment and set the lighting parameters.
        cfg = {"scene": {"type": "Scene", "floor_plane_visible": False}}
        self.env = og.Environment(configs=cfg)
        dome_light = og.sim.skybox
        dome_light.intensity = 7.5e2
        dome_light.color = [1.0, 1.0, 1.0]
        dome_light.light_link.prim.GetAttribute("inputs:texture:file").Clear()

        self.human, self.phone = self.add_reference_objects()

        remaining_objs_by_cat = self.group_objects_by_category(self.remaining_objects)

        batch_size = 50

        for cat, models in remaining_objs_by_cat.items():
            print(f"Processing category {cat}...")
            sorted_models = sorted(models)
            for batch_start in range(0, len(sorted_models), batch_size):
                batch = sorted_models[batch_start : batch_start + batch_size]
                skip = self.evaluate_batch(batch, cat)
                if skip:
                    print("Skipping the rest of the category", cat)
                    break
                print(
                    f"\n\n{len(self.processed_objects)}/{len(self.filtered_objs)} objects processed. {len(self.remaining_objects)} objects remaining.\n"
                )
                # time.sleep(0.1)


class ObjectComplaintHandler:
    def __init__(self, pipeline_root):
        self.pipeline_root = Path(pipeline_root)
        self.inventory_dict = self._load_inventory()
        self.taxonomy = bddl.object_taxonomy.ObjectTaxonomy()

    def _load_inventory(self):
        inventory_path = self.pipeline_root / "artifacts/pipeline/object_inventory.json"
        with open(inventory_path, "r") as file:
            return json.load(file)["providers"]

    def _get_existing_complaints(self, model):
        provider = None
        for obj, provider_candidate in self.inventory_dict.items():
            if obj.split("-")[-1] == model:
                provider = provider_candidate
                break
        assert provider is not None, f"Provider not found for object {model}"

        complaints_file = self.pipeline_root / "cad" / provider / "complaints.json"
        if not complaints_file.exists():
            return []
        with open(complaints_file, "r") as file:
            all_complaints = json.load(file)
        filtered_complaints = []
        for complaint in all_complaints:
            if complaint["object"].split("-")[-1] != model:
                continue
            filtered_complaints.append(complaint)
        return filtered_complaints

    def process_complaints(
        self,
        queue,
        category,
        model,
        messages,
        quick_complaints,
        stdin_fileno,
        parse_existing_complaints,
    ):
        sys.stdin = os.fdopen(stdin_fileno)

        # Get existing complaints only if running the check on prior complaints
        existing_complaints = (
            self._get_existing_complaints(model) if parse_existing_complaints else []
        )

        # Take note of the unresolved ones.
        unresolved_indices = [
            idx
            for idx, complaint in enumerate(existing_complaints)
            if not complaint["processed"]
        ]

        # Mark all as resolved
        for complaint in existing_complaints:
            complaint["processed"] = True

        # Allow the user to pick which complaints to keep unresolved.
        if len(unresolved_indices) > 0:
            print(f"Found {len(unresolved_indices)} unresolved complaints.")
            for i, idx in enumerate(unresolved_indices, start=1):
                print(f"\nComplaint {i}:")
                complaint = existing_complaints[idx]
                print(f"Prompt: {complaint['message']}\n")
                print(f"Complaint: {complaint['complaint']}\n")

            print(
                "\nALL complaints except the ones you enter below will be marked as RESOLVED."
            )
            response = input(
                "Enter complaint numbers to KEEP as UNRESOLVED (e.g., 1,2,3): "
            )
            if response:
                response = response.split(",")
                for idx in response:
                    complaint_idx = unresolved_indices[int(idx) - 1]
                    existing_complaints[complaint_idx]["processed"] = False
        else:
            print("No unresolved complaints found.")

        print("-" * 80)

        for message in messages:
            complaint = self._process_single_complaint(message, category, model)
            if complaint:
                existing_complaints.append(complaint)
            print("-" * 80)

        all_complaints = existing_complaints + quick_complaints

        queue.put(json.dumps(all_complaints))

    def _process_single_complaint(self, message, category, model):
        print(message)
        response = input("Enter a complaint (or hit enter if all's good): ")
        if response:
            return {
                "object": f"{category}-{model}",
                "message": message,
                "complaint": response,
                "processed": False,
                "new": True,
            }

        return None

    def get_questions(self, obj):
        messages = [
            # self._get_synset_question(obj),
            # self._get_substanceness_question(obj),
            # self._get_ability_question(obj),
            # self._get_category_question(obj),
            # self._get_single_rigid_body_question(obj),
            # self._get_appearance_question(obj),
            # self._get_articulation_question(obj),
            # self._get_collision_question(obj),
        ]

        # if "cloth" in self._get_synset_and_abilities(obj.category)[1]:
        #     self._get_unfolded_question(obj),

        return messages

    def _get_synset_and_abilities(self, category):
        synset = self.taxonomy.get_synset_from_category(category)
        if synset is None:
            synset = self.taxonomy.get_synset_from_substance(category)
        assert synset is not None, f"Synset not found for category {category}"
        return synset, self.taxonomy.get_abilities(synset)

    def _get_synset_and_definition(self, category):
        synset, _ = self._get_synset_and_abilities(category)
        try:
            s = wn.synset(synset)
            return s.name(), s.definition()
        except:
            s = wn.synset(self.taxonomy.get_parents(synset)[0])
            return (
                f"{synset} (custom synset)",
                f"(hypernyms: {s.name()}): {s.definition()}",
            )

    def _get_synset_question(self, obj):
        synset, definition = self._get_synset_and_definition(obj.category)
        message = (
            "SYNSET: Confirm object synset assignment.\n\n"
            f"Object assigned to synset: {synset}\n"
            f"Definition: {definition}\n\n"
            "Reminder: synset should match the object and not its contents.\n"
            "(e.g., orange_juice.n.01 is the fluid orange juice and not a bottle).\n"
            "For containers, note that usable containers (e.g. ones that have an open cap and\n"
            "can be filled should be assigned to x__bottle.n.01 etc. synsets/categories while\n"
            "non-usable containers should be assigned to bottle__of__x.n.01 etc. synsets/categories).\n\n"
            "If the synset is wrong, please type in the name of the correct synset the object\n"
            "should be assigned to. In the next question also type in the correct category."
        )
        return message

    def _get_category_question(self, obj):
        message = (
            "CATEGORY: Confirm object category assignment.\n\n"
            f"Object assigned to category: {obj.category}\n\n"
            "If the object is not compatible with the rest of the objects in this category,\n"
            "(e.g. tabletop sink vs regular sink), change the category here.\n"
            "Confirm that the object is the same kind of object as the rest of the objects\n"
            "in this category.\n\n"
            "If the category is wrong, please type in the correct category."
        )
        return message

    def _get_substanceness_question(self, obj):
        _, abilities = self._get_synset_and_abilities(obj.category)
        substance_types = set(abilities.keys()) & {
            "rigidBody",
            "liquid",
            "macroPhysicalSubstance",
            "microPhysicalSubstance",
            "visualSubstance",
            "cloth",
            "softBody",
            "rope",
        }
        assert (
            len(substance_types) == 1
        ), f"Multiple substance types found for object {obj.name.replace('obj_', '')}"
        (substance_type,) = substance_types
        message = (
            "SUBSTANCE: Confirm if object should be a rigid body, cloth, or substance.\n"
            "If it's marked softBody or rope, that means it will be a rigid body for now, which is OK.\n\n"
            f"Currently, it is annotated as: {substance_type}.\n\n"
            'Enter one of "rigidBody", "liquid", "macroPhysicalSubstance", "microPhysicalSubstance", "visualSubstance", "cloth", "softBody", "rope" to change.'
        )
        return message

    def _get_ability_question(self, obj):
        _, abilities = self._get_synset_and_abilities(obj.category)
        interesting_abilities = [
            f"    {a}: {a in abilities}" for a in sorted(INTERESTING_ABILITIES)
        ]
        message = (
            "ABILITIES: Confirm that this object can support all of the abilities seen below.\n\n"
            "Abilities to evaluate (subset of all abilities): \n"
        )
        message += "\n".join(sorted(interesting_abilities))
        message += (
            "\n\nIf this object looks like it should have some of these abilities flipped, please\n"
            "list the abilities that we should flip for this object as a comma separated list.\n"
            "For example, for an unopenable window you would put in 'openable'."
        )
        return message

    def _get_single_rigid_body_question(self, obj):
        message = "CONNECTED: Confirm object is a single body. An object cannot contain disconnected parts."
        return message

    def _get_appearance_question(self, obj):
        message = (
            "APPEARANCE: Confirm object visual appearance.\n"
            "Requirements:\n"
            "- make sure the object has a valid texture or appearance (e.g., texture not UNEXPECTEDLY black,\n"
            "       transparency rendered correctly, etc).\n"
            "- make sure any glass parts are transparent (would this object contain glass? e.g.\n"
            "       wall pictures, clocks, etc. - anything wrong)\n"
        )
        return message

    def _get_collision_question(self, obj):
        message = (
            "COLLISION: Confirm object collision meshes (C to toggle on/off).\n"
            "Requirements:\n"
            "- make sure the collision meshes well approximate the original visual meshes\n"
            "- make sure the collision meshes don't lose any affordance (e.g., holes and handles are preserved)."
        )
        return message

    def _get_articulation_question(self, obj):
        message = "ARTICULATION: Confirm articulation:\n"
        message += "This object has the below movable links annotated:\n"
        if len(obj.joints) == 0:
            message += "- None\n"
        else:
            for j_name, j in obj.joints.items():
                message += f"- {j_name}, {j.joint_type}\n"
        message += "\nVerify that the joint limits look reasonable and that the object is not\n"
        message += "missing articulations that would make it useless. Do NOT be overly ambitious - we\n"
        message += "only care about MUST-HAVE articulations here."
        return message

    def _get_unfolded_question(self, obj):
        message = "Confirm the default state of the rope/cloth object is unfolded."
        return message

    def _get_meta_link_question(self, obj):
        meta_links = sorted(
            {
                meta_name
                for link_metas in obj.metadata["meta_links"].values()
                for meta_name in link_metas
            }
        )
        message = "Confirm object meta links listed below:\n"
        if len(meta_links) == 0:
            message += "- None\n"
        else:
            for meta_link in meta_links:
                message += f"- {meta_link}\n"
        message += "Make sure these match mechanisms you expect from this object."
        return message


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--record_path",
        type=str,
        required=True,
        help="The path to save recorded orientations and scales.",
    )
    parser.add_argument(
        "--id",
        type=int,
        required=True,
        help=f"Your assigned id in range (0, total_ids-1).",
    )
    parser.add_argument(
        "--total_ids", type=int, required=True, help=f"Total number of IDs."
    )
    parser.add_argument("--seed", type=str, required=True, help=f"The shuffling seed.")
    parser.add_argument(
        "--annotation_type",
        type=int,
        required=True,
        help=f"Subtype of annotation: [{ANNOTATION_TYPE_MAPPING}]",
    )
    args = parser.parse_args()

    pipeline_root = Path(__file__).resolve().parents[2]
    viewer = BatchQAViewer(
        args.record_path,
        args.id,
        args.total_ids,
        args.seed,
        pipeline_root=str(pipeline_root),
        annotation_type=args.annotation_type,
    )
    viewer.run()
    og.shutdown()
