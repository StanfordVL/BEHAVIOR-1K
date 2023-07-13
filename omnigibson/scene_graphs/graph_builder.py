import itertools
import os

import networkx as nx
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from omnigibson import object_states
from omnigibson.sensors import VisionSensor
from omnigibson.object_states.factory import get_state_name
from omnigibson.object_states.object_state_base import AbsoluteObjectState, BooleanState, RelativeObjectState
from omnigibson.utils import transform_utils as T

DRAW_EVERY = 1


def get_robot(scene):
    assert len(scene.robots) == 1, "Exactly one robot should be available."
    return scene.robots[0]


def get_robot_bbox(robot):
    # The robot doesn't have a nicely annotated bounding box so we just return AABB for now.
    return T.pose2mat((robot.aabb_center, [0, 0, 0, 1])), robot.aabb_extent


def get_robot_to_world_transform(robot):
    # TODO: Maybe do this in eye frame
    robot_to_world = robot.get_position_orientation()

    # Get rid of any rotation outside xy plane
    robot_to_world = T.pose2mat((robot_to_world[0], T.z_rotation_from_quat(robot_to_world[1])))

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

    def get_scene_graph(self):
        return self.G.copy()

    def _get_desired_frame(self, robot):
        desired_frame_to_world = np.eye(4)
        world_to_desired_frame = np.eye(4)
        if self.egocentric:
            desired_frame_to_world = get_robot_to_world_transform(robot)
            world_to_desired_frame = T.pose_inv(desired_frame_to_world)

        return desired_frame_to_world, world_to_desired_frame

    def start(self, scene):
        assert self.G is None, "Cannot start graph builder multiple times."

        robot = get_robot(scene)
        self.G = nx.DiGraph() if self.merge_parallel_edges else nx.MultiDiGraph()

        desired_frame_to_world, world_to_desired_frame = self._get_desired_frame(robot)
        robot_pose = world_to_desired_frame @ get_robot_to_world_transform(robot)
        robot_bbox_pose, robot_bbox_extent = get_robot_bbox(robot)
        robot_bbox_pose = world_to_desired_frame @ robot_bbox_pose
        self.G.add_node(
            robot, pose=robot_pose, bbox_pose=robot_bbox_pose, bbox_extent=robot_bbox_extent, states={}
        )
        self.last_desired_frame_to_world = desired_frame_to_world

    def step(self, scene):
        assert self.G is not None, "Cannot step graph builder before starting it."

        # Prepare the necessary transformations.
        robot = get_robot(scene)
        desired_frame_to_world, world_to_desired_frame = self._get_desired_frame(robot)

        # Update the position of everything that's already in the scene by using our relative position to last frame.
        old_desired_to_new_desired = world_to_desired_frame @ self.last_desired_frame_to_world
        for obj in self.G.nodes:
            self.G.nodes[obj]["pose"] = old_desired_to_new_desired @ self.G.nodes[obj]["pose"]
            self.G.nodes[obj]["bbox_pose"] = old_desired_to_new_desired @ self.G.nodes[obj]["bbox_pose"]

        # Update the robot's pose. We don't want to accumulate errors because of the repeated transforms.
        self.G.nodes[robot]["pose"] = world_to_desired_frame @ get_robot_to_world_transform(robot)
        robot_bbox_pose, robot_bbox_extent = get_robot_bbox(robot)
        robot_bbox_pose = world_to_desired_frame @ robot_bbox_pose
        self.G.nodes[robot]["bbox_pose"] = robot_bbox_pose
        self.G.nodes[robot]["bbox_extent"] = robot_bbox_extent

        # Go through the objects in FOV of the robot.
        objs_to_add = set(scene.objects)
        if not self.full_obs:
            # TODO: This probably does not work
            # If we're not in full observability mode, only pick the objects in FOV of robot.
            bids_in_fov = robot.states[object_states.ObjectsInFOVOfRobot].get_value()
            objs_in_fov = set(
                scene.objects_by_id[bid]
                for bid in bids_in_fov
                if bid in scene.objects_by_id
            )
            objs_to_add &= objs_in_fov

        for obj in objs_to_add:
            # Add the object if not already in the graph
            if obj not in self.G.nodes:
                self.G.add_node(obj)

            # Get the relative position of the object & update it (reducing accumulated errors)
            self.G.nodes[obj]["pose"] = world_to_desired_frame @ T.pose2mat(obj.get_position_orientation())

            # Get the bounding box.
            if hasattr(obj, "get_base_aligned_bbox"):
                bbox_center, bbox_orn, bbox_extent, _ = obj.get_base_aligned_bbox(visual=True)
                bbox_pose = T.pose2mat((bbox_center, bbox_orn))
            else:
                bbox_pose, bbox_extent = get_robot_bbox(robot)
            self.G.nodes[obj]["bbox_pose"] = world_to_desired_frame @ bbox_pose
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


def visualize_scene_graph(scene, G, show_window=True, realistic_positioning=False):
    """
    @param show_window: Whether a cv2 GUI window containing the visualization should be shown.
    @param realistic_positioning: Whether nodes should be positioned based on their position in the scene (if True)
        or placed using a graphviz layout (neato) that makes it easier to read edges & find clusters.
    """

    def _draw_graph():
        nodes = list(G.nodes)
        node_labels = {obj: obj.category for obj in nodes}
        colors = [
            "yellow"
            if obj.category == "agent"
            else ("green" if obj.states[object_states.InFOVOfRobot].get_value() else "red")
            for obj in nodes
        ]
        positions = (
            {obj: (-pose[0][1], pose[0][0]) for obj, pose in G.nodes.data("pose")}
            if realistic_positioning
            else nx.nx_pydot.pydot_layout(G, prog="neato")
        )
        nx.drawing.draw_networkx(
            G,
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
                state + "=" + str(value)
                for state, value in G.edges[edge]["states"]
            )
            for edge in G.edges
        }
        nx.drawing.draw_networkx_edge_labels(G, pos=positions, edge_labels=edge_labels, font_size=4)

    # Prepare pyplot figure that's sized to match the robot video.
    robot = get_robot(scene)
    robot_camera_sensor, = [s for s in robot.sensors.values() if isinstance(s, VisionSensor) and "rgb" in s.modalities]
    robot_view = (robot_camera_sensor.get_obs()["rgb"][..., :3]).astype(np.uint8)
    imgheight, imgwidth, _ = robot_view.shape

    figheight = 4.8
    figdpi = imgheight / figheight
    figwidth = imgwidth / figdpi

    # Draw the graph onto the figure.
    fig = plt.figure(figsize=(figwidth, figheight), dpi=figdpi)
    _draw_graph()
    fig.canvas.draw()

    # if show_window:
    #     plt.show()

    # # Convert the canvas to image
    graph_view = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    graph_view = graph_view.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    assert graph_view.shape == robot_view.shape
    plt.close(fig)

    # # Combine the two images side-by-side
    img = np.hstack((robot_view, graph_view))

    # # Convert to BGR for cv2-based viewing.
    # if show_window:
    #     # display image with opencv or any operation you like
    #     cv_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #     cv2.imshow("SceneGraph", cv_img)
    #     cv2.waitKey(1)

    return Image.fromarray(img).save(r"D:\test.png")
