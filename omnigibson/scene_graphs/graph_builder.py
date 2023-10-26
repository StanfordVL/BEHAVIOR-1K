import itertools
import os

import networkx as nx
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from omnigibson import object_states
from omnigibson.macros import create_module_macros
from omnigibson.sensors import VisionSensor
from omnigibson.object_states.factory import get_state_name
from omnigibson.object_states.object_state_base import AbsoluteObjectState, BooleanStateMixin, RelativeObjectState
from omnigibson.utils import transform_utils as T

m = create_module_macros(module_path=__file__)

m.DRAW_EVERY = 1


def _formatted_aabb(obj):
    return T.pose2mat((obj.aabb_center, [0, 0, 0, 1])), obj.aabb_extent


class SceneGraphBuilder(object):
    def __init__(
            self,
            robot_name=None,
            egocentric=False,
            full_obs=False,
            only_true=False,
            merge_parallel_edges=False,
            exclude_states=(object_states.Touching,)
        ):
        """
        A utility that builds a scene graph with objects as nodes and relative states as edges,
        alongside additional metadata.

        Args:
            robot_name (str): Name of the robot whose POV the scene graph will be from. If None, we assert that there
                is exactly one robot in the scene and use that robot.
            egocentric (bool): Whether the objects should have poses in the world frame or robot frame.
            full_obs (bool): Whether all objects should be updated or only those in FOV of the robot.
            only_true (bool): Whether edges should be created only for relative states that have a value True, or for all
                relative states (with the appropriate value attached as an attribute).
            merge_parallel_edges (bool): Whether parallel edges (e.g. different states of the same pair of objects) should
                exist (making the graph a MultiDiGraph) or should be merged into a single edge instead.
            exclude_states (Iterable): Object state classes that should be ignored when building the graph.
        """
        self._G = None
        self._robot = None
        self._robot_name = robot_name
        self._egocentric = egocentric
        self._full_obs = full_obs
        self._only_true = only_true
        self._merge_parallel_edges = merge_parallel_edges
        self._last_desired_frame_to_world = None
        self._exclude_states = set(exclude_states)

    def get_scene_graph(self):
        return self._G.copy()

    def _get_desired_frame(self):
        desired_frame_to_world = np.eye(4)
        world_to_desired_frame = np.eye(4)
        if self._egocentric:
            desired_frame_to_world = self._get_robot_to_world_transform()
            world_to_desired_frame = T.pose_inv(desired_frame_to_world)

        return desired_frame_to_world, world_to_desired_frame

    def _get_robot_to_world_transform(self):
        robot_to_world = self._robot.get_position_orientation()

        # Get rid of any rotation outside xy plane
        robot_to_world = T.pose2mat((robot_to_world[0], T.z_rotation_from_quat(robot_to_world[1])))

        return robot_to_world

    def _get_boolean_unary_states(self, obj):
        states = {}
        for state_type, state_inst in obj.states.items():
            if not issubclass(state_type, BooleanStateMixin) or not issubclass(state_type, AbsoluteObjectState):
                continue

            if state_type in self._exclude_states:
                continue

            value = state_inst.get_value()
            if self._only_true and not value:
                continue

            states[get_state_name(state_type)] = value

        return states


    def _get_boolean_binary_states(self, objs):
        states = []
        for obj1 in objs:
            for obj2 in objs:
                if obj1 == obj2:
                    continue

                for state_type, state_inst in obj1.states.items():
                    if not issubclass(state_type, BooleanStateMixin) or not issubclass(state_type, RelativeObjectState):
                        continue

                    if state_type in self._exclude_states:
                        continue

                    try:
                        value = state_inst.get_value(obj2)
                        if self._only_true and not value:
                            continue

                        states.append((obj1, obj2, get_state_name(state_type), {"value": value}))
                    except:
                        pass

        return states

    def start(self, scene):
        assert self._G is None, "Cannot start graph builder multiple times."

        if self._robot_name is None:
            assert len(scene.robots) == 1, "Cannot build scene graph without specifying robot name if there are multiple robots."
            self._robot = scene.robots[0]
        else:
            self._robot = scene.object_registry("name", self._robot_name)
            assert self._robot, f"Robot with name {self._robot_name} not found in scene."
        self._G = nx.DiGraph() if self._merge_parallel_edges else nx.MultiDiGraph()

        desired_frame_to_world, world_to_desired_frame = self._get_desired_frame()
        robot_pose = world_to_desired_frame @ self._get_robot_to_world_transform()
        robot_bbox_pose, robot_bbox_extent = _formatted_aabb(self._robot)
        robot_bbox_pose = world_to_desired_frame @ robot_bbox_pose
        self._G.add_node(
            self._robot, pose=robot_pose, bbox_pose=robot_bbox_pose, bbox_extent=robot_bbox_extent, states={}
        )
        self._last_desired_frame_to_world = desired_frame_to_world

        # Let's also take the first step.
        self.step(scene)

    def step(self, scene):
        assert self._G is not None, "Cannot step graph builder before starting it."

        # Prepare the necessary transformations.
        desired_frame_to_world, world_to_desired_frame = self._get_desired_frame(self._robot)

        # Update the position of everything that's already in the scene by using our relative position to last frame.
        old_desired_to_new_desired = world_to_desired_frame @ self._last_desired_frame_to_world
        for obj in self._G.nodes:
            self._G.nodes[obj]["pose"] = old_desired_to_new_desired @ self._G.nodes[obj]["pose"]
            self._G.nodes[obj]["bbox_pose"] = old_desired_to_new_desired @ self._G.nodes[obj]["bbox_pose"]

        # Update the robot's pose. We don't want to accumulate errors because of the repeated transforms.
        self._G.nodes[self._robot]["pose"] = world_to_desired_frame @ self._get_robot_to_world_transform()
        robot_bbox_pose, robot_bbox_extent = _formatted_aabb(self._robot)
        robot_bbox_pose = world_to_desired_frame @ robot_bbox_pose
        self._G.nodes[self._robot]["bbox_pose"] = robot_bbox_pose
        self._G.nodes[self._robot]["bbox_extent"] = robot_bbox_extent

        # Go through the objects in FOV of the robot.
        objs_to_add = set(scene.objects)
        if not self._full_obs:
            # TODO: Reenable this once InFOV state is fixed.
            # If we're not in full observability mode, only pick the objects in FOV of robot.
            # bids_in_fov = self._robot.states[object_states.ObjectsInFOVOfRobot].get_value()
            # objs_in_fov = set(
            #     scene.objects_by_id[bid]
            #     for bid in bids_in_fov
            #     if bid in scene.objects_by_id
            # )
            # objs_to_add &= objs_in_fov
            raise NotImplementedError("Partial observability not supported in scene graph builder yet.")

        for obj in objs_to_add:
            # Add the object if not already in the graph
            if obj not in self._G.nodes:
                self._G.add_node(obj)

            # Get the relative position of the object & update it (reducing accumulated errors)
            self._G.nodes[obj]["pose"] = world_to_desired_frame @ T.pose2mat(obj.get_position_orientation())

            # Get the bounding box.
            if hasattr(obj, "get_base_aligned_bbox"):
                bbox_center, bbox_orn, bbox_extent, _ = obj.get_base_aligned_bbox(visual=True, fallback_to_aabb=True)
                bbox_pose = T.pose2mat((bbox_center, bbox_orn))
            else:
                bbox_pose, bbox_extent = _formatted_aabb(obj)
            self._G.nodes[obj]["bbox_pose"] = world_to_desired_frame @ bbox_pose
            self._G.nodes[obj]["bbox_extent"] = bbox_extent

            # Update the states of the object
            self._G.nodes[obj]["states"] = self._get_boolean_unary_states(obj)

        # Update the binary states for seen objects.
        self._G.remove_edges_from(list(itertools.product(objs_to_add, objs_to_add)))
        edges = self._get_boolean_binary_states(objs_to_add)
        if self._merge_parallel_edges:
            new_edges = {}
            for edge in edges:
                edge_pair = (edge[0], edge[1])
                if edge_pair not in new_edges:
                    new_edges[edge_pair] = []

                new_edges[edge_pair].append((edge[2], edge[3]["value"]))

            edges = [(k[0], k[1], {"states": v}) for k, v in new_edges.items()]

        self._G.add_edges_from(edges)

        # Save the robot's transform in this frame.
        self._last_desired_frame_to_world = desired_frame_to_world


def visualize_scene_graph(scene, G, show_window=True, realistic_positioning=False):
    """
    Converts the graph into an image and shows it in a cv2 window if preferred.

    Args:
        show_window (bool): Whether a cv2 GUI window containing the visualization should be shown.
        realistic_positioning (bool): Whether nodes should be positioned based on their position in the scene (if True)
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
    robot = scene.robots[0]
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

    # Convert the canvas to image
    graph_view = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    graph_view = graph_view.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    assert graph_view.shape == robot_view.shape
    plt.close(fig)

    # Combine the two images side-by-side
    img = np.hstack((robot_view, graph_view))

    # # Convert to BGR for cv2-based viewing.
    if show_window:
        import cv2
        cv_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("SceneGraph", cv_img)
        cv2.waitKey(1)

    return Image.fromarray(img).save(r"D:\test.png")
