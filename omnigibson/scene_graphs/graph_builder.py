import itertools
import os

import cv2
import networkx as nx
import numpy as np
import pybullet as p
from matplotlib import pyplot as plt

from igibson import object_states
from igibson.activity.activity_base import iGBEHAVIORActivityInstance
from igibson.external.pybullet_tools.utils import aabb_union, get_aabb, get_aabb_center, get_aabb_extent
from igibson.object_states.factory import get_state_name
from igibson.object_states.object_state_base import AbsoluteObjectState, BooleanState, RelativeObjectState
from igibson.utils.utils import z_rotation_from_quat

DRAW_EVERY = 1


def get_robot(activity: iGBEHAVIORActivityInstance):
    assert len(activity.simulator.robots) == 1, "Exactly one robot should be available."
    return activity.simulator.robots[0]


def get_robot_bbox(robot):
    # The robot doesn't have a nicely annotated bounding box so we just append one for now.
    aabb = aabb_union([get_aabb(part.get_body_id()) for part in robot.parts.values()])
    center = get_aabb_center(aabb)
    extent = get_aabb_extent(aabb)
    return (center, [0, 0, 0, 1]), extent


def get_robot_to_world_transform(robot):
    # TODO: Maybe keep the graph in the base reference frame and only switch to eye for visualization.
    robot_to_world = robot.parts["body"].get_position_orientation()

    # Get rid of any rotation outside xy plane
    robot_to_world = robot_to_world[0], z_rotation_from_quat(robot_to_world[1])

    return robot_to_world


def get_unary_states(obj, only_true=False):
    states = {}
    for state_type, state_inst in obj.states.items():
        if not issubclass(state_type, BooleanState) or not issubclass(state_type, AbsoluteObjectState):
            continue

        value = state_inst.get_value()
        if only_true and not value:
            continue

        states[get_state_name(state_type)] = value

    return states


def get_all_binary_states(objs, only_true=False):
    states = []
    for obj1 in objs:
        for obj2 in objs:
            if obj1 == obj2:
                continue

            for state_type, state_inst in obj1.states.items():
                if not issubclass(state_type, BooleanState) or not issubclass(state_type, RelativeObjectState):
                    continue

                value = state_inst.get_value(obj2)
                if only_true and not value:
                    continue

                states.append((obj1, obj2, get_state_name(state_type), {"value": value}))

    return states


class SceneGraphBuilder(object):
    def __init__(self, egocentric=False, full_obs=False, only_true=False, merge_parallel_edges=False):
        """
        @param egocentric: Whether the objects should have poses in the world frame or robot frame.
        @param full_obs: Whether all objects should be updated or only those in FOV of the robot.
        @param only_true: Whether edges should be created only for relative states that have a value True, or for all
            relative states (with the appropriate value attached as an attribute).
        @param merge_parallel_edges: Whether parallel edges (e.g. different states of the same pair of objects) should
            exist (making the graph a MultiDiGraph) or should be merged into a single edge instead.
        """
        self.G = None
        self.egocentric = egocentric
        self.full_obs = full_obs
        self.only_true = only_true
        self.merge_parallel_edges = merge_parallel_edges
        self.last_desired_frame_to_world = None

    def _get_desired_frame(self, robot):
        desired_frame_to_world = ([0, 0, 0], [0, 0, 0, 1])
        world_to_desired_frame = ([0, 0, 0], [0, 0, 0, 1])
        if self.egocentric:
            desired_frame_to_world = get_robot_to_world_transform(robot)
            world_to_desired_frame = p.invertTransform(*desired_frame_to_world)

        return desired_frame_to_world, world_to_desired_frame

    def start(self, activity, log_reader):
        assert self.G is None, "Cannot start graph builder multiple times."

        robot = get_robot(activity)
        self.G = nx.DiGraph() if self.merge_parallel_edges else nx.MultiDiGraph()

        desired_frame_to_world, world_to_desired_frame = self._get_desired_frame(robot)
        robot_pose = p.multiplyTransforms(*world_to_desired_frame, *get_robot_to_world_transform(robot))
        robot_bbox_pose, robot_bbox_extent = get_robot_bbox(robot)
        robot_bbox_pose = p.multiplyTransforms(*world_to_desired_frame, *robot_bbox_pose)
        self.G.add_node(
            robot.parts["body"], pose=robot_pose, bbox_pose=robot_bbox_pose, bbox_extent=robot_bbox_extent, states={}
        )
        self.last_desired_frame_to_world = desired_frame_to_world

    def step(self, activity, log_reader):
        assert self.G is not None, "Cannot step graph builder before starting it."

        # Prepare the necessary transformations.
        robot = get_robot(activity)
        desired_frame_to_world, world_to_desired_frame = self._get_desired_frame(robot)

        # Update the position of everything that's already in the scene by using our relative position to last frame.
        old_desired_to_new_desired = p.multiplyTransforms(*world_to_desired_frame, *self.last_desired_frame_to_world)
        for obj in self.G.nodes:
            self.G.nodes[obj]["pose"] = p.multiplyTransforms(*old_desired_to_new_desired, *self.G.nodes[obj]["pose"])
            self.G.nodes[obj]["bbox_pose"] = p.multiplyTransforms(
                *old_desired_to_new_desired, *self.G.nodes[obj]["bbox_pose"]
            )

        # Update the robot's pose. We don't want to accumulate errors because of the repeated transforms.
        self.G.nodes[robot.parts["body"]]["pose"] = p.multiplyTransforms(
            *world_to_desired_frame, *get_robot_to_world_transform(robot)
        )
        robot_bbox_pose, robot_bbox_extent = get_robot_bbox(robot)
        robot_bbox_pose = p.multiplyTransforms(*world_to_desired_frame, *robot_bbox_pose)
        self.G.nodes[robot.parts["body"]]["bbox_pose"] = robot_bbox_pose
        self.G.nodes[robot.parts["body"]]["bbox_extent"] = robot_bbox_extent

        # Go through the objects in FOV of the robot.
        objs_to_add = set(activity.simulator.scene.get_objects()) | set(activity.object_scope.values())
        if not self.full_obs:
            # If we're not in full observability mode, only pick the objects in FOV of robot.
            bids_in_fov = robot.parts["body"].states[object_states.ObjectsInFOVOfRobot].get_value()
            objs_in_fov = set(
                activity.simulator.scene.objects_by_id[bid]
                for bid in bids_in_fov
                if bid in activity.simulator.scene.objects_by_id
            )
            objs_to_add &= objs_in_fov

        # Filter out any agent parts.
        objs_to_add = {obj for obj in objs_to_add if obj.category != "agent"}

        for obj in objs_to_add:
            # Add the object if not already in the graph
            if obj not in self.G.nodes:
                self.G.add_node(obj)

            # Get the relative position of the object & update it (reducing accumulated errors)
            self.G.nodes[obj]["pose"] = p.multiplyTransforms(*world_to_desired_frame, *obj.get_position_orientation())

            # Get the bounding box.
            bbox_center, bbox_orn, bbox_extent, _ = obj.get_base_aligned_bounding_box(visual=True)
            self.G.nodes[obj]["bbox_pose"] = p.multiplyTransforms(*world_to_desired_frame, bbox_center, bbox_orn)
            self.G.nodes[obj]["bbox_extent"] = bbox_extent

            # Update the states of the object
            self.G.nodes[obj]["states"] = get_unary_states(obj, only_true=self.only_true)

        # Update the binary states for seen objects.
        self.G.remove_edges_from(list(itertools.product(objs_to_add, objs_to_add)))
        edges = get_all_binary_states(objs_to_add, only_true=self.only_true)
        if self.merge_parallel_edges:
            new_edges = {}
            for edge in edges:
                edge_pair = (edge[0], edge[1])
                if edge_pair not in new_edges:
                    new_edges[edge_pair] = []

                new_edges[edge_pair].append((edge[2], edge[3]["value"]))

            edges = [(k[0], k[1], {"states": v}) for k, v in new_edges.items()]

        self.G.add_edges_from(edges)

        # Save the robot's transform in this frame.
        self.last_desired_frame_to_world = desired_frame_to_world


class SceneGraphBuilderWithVisualization(SceneGraphBuilder):
    def __init__(self, show_window=True, out_path=None, realistic_positioning=False, *args, **kwargs):
        """
        @param show_window: Whether a cv2 GUI window containing the visualization should be shown.
        @param out_path: Directory to output visualization frames to. If None, no frames will be saved.
        @param realistic_positioning: Whether nodes should be positioned based on their position in the scene (if True)
            or placed using a graphviz layout (neato) that makes it easier to read edges & find clusters.
        @param args: Any positional arguments to forward to the SceneGraphBuilder constructor.
        @param kwargs: Any keyword arguments to forward to the SceneGraphBuilder constructor. Note that the
            merge_parallel_edges argument is forced to True here because the graph drawing mechanism cannot show
            parallel edges.
        """
        super(SceneGraphBuilderWithVisualization, self).__init__(*args, **kwargs, merge_parallel_edges=True)
        assert show_window or out_path, "One of show_window or out_path should be set."
        self.show_window = show_window
        self.out_path = out_path
        self.out_writer = None
        self.realistic_positioning = realistic_positioning

    def draw_graph(self):
        nodes = list(self.G.nodes)
        node_labels = {obj: obj.category for obj in nodes}
        colors = [
            "yellow"
            if obj.category == "agent"
            else ("green" if obj.states[object_states.InFOVOfRobot].get_value() else "red")
            for obj in nodes
        ]
        positions = (
            {obj: (-pose[0][1], pose[0][0]) for obj, pose in self.G.nodes.data("pose")}
            if self.realistic_positioning
            else nx.nx_pydot.pydot_layout(self.G, prog="neato")
        )
        nx.drawing.draw_networkx(
            self.G,
            pos=positions,
            labels=node_labels,
            nodelist=nodes,
            node_color=colors,
            font_size=4,
            arrowsize=5,
            node_size=150,
        )

        edge_labels = {
            edge: ", ".join(
                state + "=" + str(value) if not self.only_true else state  # Don't print value in only_true mode.
                for state, value in self.G.edges[edge]["states"]
            )
            for edge in self.G.edges
        }
        nx.drawing.draw_networkx_edge_labels(self.G, pos=positions, edge_labels=edge_labels, font_size=4)

    def step(self, activity, *args):
        super(SceneGraphBuilderWithVisualization, self).step(activity, *args)

        if activity.simulator.frame_count % DRAW_EVERY != 0:
            return

        # Prepare pyplot figure that's sized to match the robot video.
        robot_view = (get_robot(activity).render_camera_image()[0][..., :3] * 255).astype(np.uint8)
        imgheight, imgwidth, _ = robot_view.shape

        figheight = 4.8
        figdpi = imgheight / figheight
        figwidth = imgwidth / figdpi

        # Draw the graph onto the figure.
        fig = plt.figure(figsize=(figwidth, figheight), dpi=figdpi)
        self.draw_graph()
        fig.canvas.draw()

        # Convert the canvas to image
        graph_view = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
        graph_view = graph_view.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        assert graph_view.shape == robot_view.shape
        plt.close(fig)

        # Combine the two images side-by-side
        img = np.hstack((robot_view, graph_view))

        # Convert to BGR for cv2-based viewing.
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if self.out_path:
            frame_path = os.path.join(self.out_path, "%05d.png" % activity.simulator.frame_count)
            cv2.imwrite(frame_path, img)

        if self.show_window:
            # display image with opencv or any operation you like
            cv2.imshow("SceneGraph", img)
            cv2.waitKey(1)
