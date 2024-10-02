import itertools

import networkx as nx
import torch as th
from PIL import Image

from omnigibson import object_states
from omnigibson.object_states.factory import get_state_name
from omnigibson.object_states.object_state_base import AbsoluteObjectState, BooleanStateMixin, RelativeObjectState
from omnigibson.robots import BaseRobot
from omnigibson.sensors import VisionSensor
from omnigibson.utils import transform_utils as T
from omnigibson.utils.numpy_utils import pil_to_tensor


def _formatted_aabb(obj):
    return T.pose2mat((obj.aabb_center, th.tensor([0, 0, 0, 1], dtype=th.float32))), obj.aabb_extent


class SceneGraphBuilder(object):
    def __init__(
        self,
        robot_names=None,
        egocentric=False,
        full_obs=False,
        only_true=True,
        merge_parallel_edges=False,
        exclude_states=(
            object_states.Touching,
            object_states.NextTo,
        ),
    ):
        """
        A utility that builds a scene graph with objects as nodes and relative states as edges,
        alongside additional metadata.

        Args:
            robot_names (list of str): Names of the robots whose POV the scene graph will be from. If None, we assert that there
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
        self._robots = None
        if robot_names is not None and len(robot_names) > 1:
            assert not egocentric, "Cannot have multiple robots in egocentric mode."
        self._robot_names = robot_names
        self._egocentric = egocentric
        self._full_obs = full_obs
        self._only_true = only_true
        self._merge_parallel_edges = merge_parallel_edges
        self._last_desired_frame_to_world = None
        self._exclude_states = set(exclude_states)

    def get_scene_graph(self):
        return self._G.copy()

    def _get_desired_frame(self):
        desired_frame_to_world = th.eye(4)
        world_to_desired_frame = th.eye(4)
        if self._egocentric:
            desired_frame_to_world = self._get_robot_to_world_transform(self._robots[0])
            world_to_desired_frame = T.pose_inv(desired_frame_to_world)

        return desired_frame_to_world, world_to_desired_frame

    def _get_robot_to_world_transform(self, robot):
        robot_to_world = robot.get_position_orientation()

        # Get rid of any rotation outside xy plane
        z_angle = T.z_angle_from_quat(robot_to_world[1])
        robot_to_world = T.pose2mat((robot_to_world[0], T.euler2quat(th.tensor([0, 0, z_angle], dtype=th.float32))))

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

        if self._robot_names is None:
            assert (
                len(scene.robots) == 1
            ), "Cannot build scene graph without specifying robot name if there are multiple robots."
            self._robots = [scene.robots[0]]
        else:
            self._robots = [scene.object_registry("name", name) for name in self._robot_names]
            assert len(self._robots) > 0, f"Robots with names {self._robot_names} not found in scene."
        self._G = nx.DiGraph() if self._merge_parallel_edges else nx.MultiDiGraph()

        desired_frame_to_world, world_to_desired_frame = self._get_desired_frame()

        for robot in self._robots:
            robot_pose = world_to_desired_frame @ self._get_robot_to_world_transform(robot)
            robot_bbox_pose, robot_bbox_extent = _formatted_aabb(robot)
            robot_bbox_pose = world_to_desired_frame @ robot_bbox_pose
            self._G.add_node(
                robot, pose=robot_pose, bbox_pose=robot_bbox_pose, bbox_extent=robot_bbox_extent, states={}
            )

        self._last_desired_frame_to_world = desired_frame_to_world

        # Let's also take the first step.
        self.step(scene)

    def step(self, scene):
        assert self._G is not None, "Cannot step graph builder before starting it."

        # Prepare the necessary transformations.
        desired_frame_to_world, world_to_desired_frame = self._get_desired_frame()

        # Update the position of everything that's already in the scene by using our relative position to last frame.
        old_desired_to_new_desired = world_to_desired_frame @ self._last_desired_frame_to_world
        nodes = list(self._G.nodes)
        poses = th.stack([self._G.nodes[obj]["pose"] for obj in nodes])
        bbox_poses = th.stack([self._G.nodes[obj]["bbox_pose"] for obj in nodes])
        updated_poses = old_desired_to_new_desired @ poses
        updated_bbox_poses = old_desired_to_new_desired @ bbox_poses
        for i, obj in enumerate(nodes):
            self._G.nodes[obj]["pose"] = updated_poses[i]
            self._G.nodes[obj]["bbox_pose"] = updated_bbox_poses[i]

        # Update the robots' poses. We don't want to accumulate errors because of the repeated transforms.
        for robot in self._robots:
            self._G.nodes[robot]["pose"] = world_to_desired_frame @ self._get_robot_to_world_transform(robot)
            robot_bbox_pose, robot_bbox_extent = _formatted_aabb(robot)
            robot_bbox_pose = world_to_desired_frame @ robot_bbox_pose
            self._G.nodes[robot]["bbox_pose"] = robot_bbox_pose
            self._G.nodes[robot]["bbox_extent"] = robot_bbox_extent

        # Go through the objects in FOV of the robot.
        objs_to_add = set(scene.objects)
        if not self._full_obs:
            # If we're not in full observability mode, only pick the objects in FOV of robots.
            for robot in self._robots:
                objs_in_fov = robot.states[object_states.ObjectsInFOVOfRobot].get_value()
                objs_to_add &= set(objs_in_fov)

        # Remove all BaseRobot objects from the set of objects to add.
        base_robots = [obj for obj in objs_to_add if isinstance(obj, BaseRobot)]
        objs_to_add -= set(base_robots)

        for obj in objs_to_add:
            # Add the object if not already in the graph
            if obj not in self._G.nodes:
                self._G.add_node(obj)

            # Get the relative position of the object & update it (reducing accumulated errors)
            self._G.nodes[obj]["pose"] = world_to_desired_frame @ T.pose2mat(obj.get_position_orientation())

            # Get the bounding box.
            if hasattr(obj, "get_base_aligned_bbox"):
                bbox_center, bbox_orn, bbox_extent, _ = obj.get_base_aligned_bbox(visual=True)
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


def visualize_scene_graph(scene, G, show_window=True, cartesian_positioning=False):
    """
    Converts the graph into an image and shows it in a cv2 window if preferred.
    Note: Currently, this function only works when we merge parallel edges, i.e. the graph is a DiGraph.

    Args:
        show_window (bool): Whether a cv2 GUI window containing the visualization should be shown.
        realistic_positioning (bool): Whether nodes should be positioned based on their position in the scene (if True)
            or placed using a graphviz layout (neato) that makes it easier to read edges & find clusters.
    """

    nodes = list(G.nodes)
    all_robots = [robot for robot in nodes if isinstance(robot, BaseRobot)]

    def _draw_graph():
        node_labels = {obj: obj.category for obj in nodes}

        # get all objects in fov of robots
        objects_in_fov = []
        for robot in all_robots:
            objects_in_fov += robot.states[object_states.ObjectsInFOVOfRobot].get_value()
        colors = [
            ("yellow" if obj.category == "agent" else ("green" if obj in objects_in_fov else "red")) for obj in nodes
        ]
        positions = (
            {obj: (-pose[1][-1], pose[0][-1]) for obj, pose in G.nodes.data("pose")}
            if cartesian_positioning
            else nx.nx_pydot.pydot_layout(G, prog="neato")
        )
        nx.drawing.draw_networkx(
            G,
            pos=positions,
            labels=node_labels,
            nodelist=nodes,
            node_color=colors,
            font_size=5,
            arrowsize=5,
            node_size=200,
        )

        edge_labels = {}
        for edge in G.edges:
            state_value_pairs = []
            if len(edge) == 3:
                # When we don't merge parallel edges
                raise ValueError("Visualization does not support parallel edges.")
            else:
                # When we merge parallel edges
                assert len(edge) == 2, "Invalid graph format for scene graph visualization."
            for state, value in G.edges[edge]["states"]:
                state_value_pairs.append(state + "=" + str(value))
            edge_labels[edge] = ", ".join(state_value_pairs)

        nx.drawing.draw_networkx_edge_labels(G, pos=positions, edge_labels=edge_labels, font_size=4)

    # Prepare pyplot figure that's sized to match the robot video.
    robot = all_robots[0]  # If there are multiple robots, we only include the first one
    (robot_camera_sensor,) = [
        s for s in robot.sensors.values() if isinstance(s, VisionSensor) and "rgb" in s.modalities
    ]
    robot_view = (robot_camera_sensor.get_obs()[0]["rgb"][..., :3]).to(th.uint8)
    imgheight, imgwidth, _ = robot_view.shape

    # check imgheight and imgwidth; if they are too small, we need to upsample the image to 640x640
    if imgheight < 640 or imgwidth < 640:
        # Convert to PIL Image to upsample, then write back to tensor
        robot_view = pil_to_tensor(Image.fromarray(robot_view.cpu().numpy()).resize((640, 640), Image.BILINEAR))
        imgheight, imgwidth, _ = robot_view.shape

    figheight = 4.8
    figdpi = imgheight / figheight
    figwidth = imgwidth / figdpi

    # Draw the graph onto the figure.
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(figwidth, figheight), dpi=figdpi)
    _draw_graph()
    fig.canvas.draw()

    # Convert the canvas to image
    graph_view = th.frombuffer(fig.canvas.tostring_rgb(), dtype=th.uint8)
    graph_view = graph_view.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    assert graph_view.shape == robot_view.shape
    plt.close(fig)

    # Combine the two images side-by-side
    img = th.cat((robot_view, graph_view), dim=1)

    # # Convert to BGR for cv2-based viewing.
    if show_window:
        plt.imshow(img)
        plt.title("SceneGraph")
        plt.axis("off")
        plt.show()

    return img
