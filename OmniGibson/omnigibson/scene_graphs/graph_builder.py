import itertools
import math

import networkx as nx
import numpy as np
import torch as th
from PIL import Image

import omnigibson as og
from omnigibson import object_states
from omnigibson.object_states.factory import get_state_name
from omnigibson.object_states.object_state_base import AbsoluteObjectState, BooleanStateMixin, RelativeObjectState
from omnigibson.robots import BaseRobot
from omnigibson.sensors import VisionSensor
from omnigibson.systems.system_base import BaseSystem
from omnigibson.utils import transform_utils as T
from omnigibson.utils.numpy_utils import pil_to_tensor
from omnigibson.utils.scene_graph_utils import CustomizedBinaryStates, CustomizedUnaryStates
from omnigibson.objects.stateful_object import StatefulObject

from dataclasses import fields
from copy import deepcopy

EXTRA_OBJECT_WHITE_LIST = {
    "straight_chair_uofiqj_0",
    "door_bexenl_0",
    "drop_in_sink_lkklqs_0",
    "floors_qgmjvd_0",
    "floors_kxcpgy_0",
    "coffee_table_rlsebe_0",
    "door_bexenl_0",
    "breakfast_table_xftrki_0",
    "garden_chair_lraplz_0",
    "garden_chair_cottya_0",
    "garden_chair_cottya_1",
    "garden_chair_cottya_2",
    "toilet_udiezm_0",
    "lawn_aztwla_0",
    "door_vudhlc_1",
    "sliding_door_tprpvb_10",
    "bottom_cabinet_rhdbzv_0",
    "door_bexenl_0",
    "drop_in_sink_awvzkn_0"
}

MAGNITUDE_TOLERANCE = 1e-1

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
            object_states.Frozen,
            object_states.Heated,
            object_states.SlicerActive,
            object_states.Filled
        ),
        only_task_relevant_objects=False,
        semantic_only=True
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
            only_task_relevant_objects (bool): Whether to only consider task-relevant objects.
            semantic_only (bool): Whether to only include semantic states in the graph.
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

        self._initialized = False
        self._exclude_states = set(exclude_states)

        self._all_objects = []

        self._last_removed_object = []

        self._obj_parent_dict = {}

        # Heuristics fields for efficient scenegraph states updating
        self._task_relevant_objects = None
        self._only_task_relevant_objects = only_task_relevant_objects
        self._contact_objects = set()
        # self._contact_callback = og.sim._physics_context._physx_sim_interface.subscribe_contact_report_events(
        #     self._on_contact_handler
        # )

        self._objects_remove_callback = og.sim.add_callback_on_remove_obj(
            "scene_graph_obj_remove_callback",
            self._object_remove_handler
        )
        self._objects_add_callback = og.sim.add_callback_on_add_obj(
            "scene_graph_obj_add_callback",
            self._object_add_handler
        )

        self._systems_remove_callback = og.sim.add_callback_on_system_clear(
            "scene_graph_system_remove_callback",
            self._system_remove_handler
        )

        self._systems_add_callback = og.sim.add_callback_on_system_init(
            "scene_graph_system_add_callback",
            self._system_add_handler
        )

        # Whether to only include semantic states in the graph
        self._semantic_only = semantic_only
    
    @staticmethod
    def _get_impulse_magnitude(impulse):
        return math.sqrt(sum(x**2 for x in impulse))

    def _on_contact_handler(self, contact_headers, contact_data):
        '''
        This callback will be invoked after every PHYSICS step if there is any contact. Will record the contact objects in the current step.
        '''
        current_contact_points_num = 0
        for contact_header in contact_headers:
            # TODO: Not all contact headers have valid impulse contact points
            # We need to skip those contact headers
            # This implementation has some issues
            current_contact_points = contact_data[current_contact_points_num:current_contact_points_num + contact_header.num_contact_data]
            current_contact_points_num += contact_header.num_contact_data # always update this

            has_evident_contact = any(self._get_impulse_magnitude(cp.impulse) >= MAGNITUDE_TOLERANCE 
                                    for cp in current_contact_points)
            if not has_evident_contact:
                continue

            actor0_obj = og.sim._link_id_to_objects.get(contact_header.actor0, None)
            actor1_obj = og.sim._link_id_to_objects.get(contact_header.actor1, None)

            if actor0_obj is not None:
                self._contact_objects.add(actor0_obj)
            if actor1_obj is not None:
                self._contact_objects.add(actor1_obj)


    def _system_remove_handler(self, system):
        if self._G is not None and system in self._G.nodes:
            self._G.remove_node(system)
        
        if self._initialized:
            self._all_objects = [o for o in self._all_objects if o.name != system.name]
            self._last_removed_object.append(system)

    def _system_add_handler(self, system):
        if self._initialized:
            self._all_objects.append(system) if all(o.name != system.name for o in self._all_objects) else None
            system_parent_name = system.name.replace('diced__', '')
            if system.name not in self._obj_parent_dict:
                self._obj_parent_dict[system.name] = [system_parent_name]
            
    def _object_remove_handler(self, obj):
        # 1. Remove the object from scene graph if it exists
        # breakpoint()
        if self._G is not None and obj in self._G.nodes:
            self._G.remove_node(obj)  # This also removes all edges connected to the object

        if self._initialized:
            # 2. Remove the object from self._all_objects by query obj name
            self._all_objects = [o for o in self._all_objects if o.name != obj.name]

            # 3. Add the object to self._last_removed_object
            self._last_removed_object.append(obj)

    def _object_add_handler(self, obj):        
        # 1. Add the object to self._all_objects
        if self._initialized:
            self._all_objects.append(obj) if all(o.name != obj.name for o in self._all_objects) and isinstance(obj, StatefulObject) and obj.states else None
        
            # 2. Add the object to self._obj_parent_dict, and the parent is the first object in self._last_removed_object
            if len(self._last_removed_object) == 1:
                if obj.name not in self._obj_parent_dict:
                    self._obj_parent_dict[obj.name] = [self._last_removed_object[0].name]
                else:
                    self._obj_parent_dict[obj.name].append(self._last_removed_object[0].name)
            else:
                print(f"Warning: {len(self._last_removed_object)} objects removed, expected 1.")


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
    
    def _get_binary_filtered_edge(self, edge, all_edges):
        '''
        Rules to filter out edges that are not what we want.
        '''

        obj_1, obj_2, state_dict = edge
        states = [s[0] for s in state_dict["states"]]
        filtered_edge = (obj_1, obj_2, deepcopy(state_dict))

        # 0. if obj_1 is a robot and obj_2 is system, and the state is "Covered" filtered
        if isinstance(obj_1, BaseRobot) and isinstance(obj_2, BaseSystem):
            filtered_edge[2]["states"] = [s for s in filtered_edge[2]["states"] if s[0] != "Covered"]

        if isinstance(obj_1, BaseSystem) or isinstance(obj_2, BaseSystem):
            return filtered_edge

        if (isinstance(obj_1, BaseRobot) or isinstance(obj_2, BaseRobot)) \
            and ("OnTop" in states or "Inside" in states or 'Under' in states):
            filtered_edge[2]["states"] = [s for s in filtered_edge[2]["states"] if s[0] != "OnTop" and s[0] != "Inside" and s[0] != 'Under']
           
        # 2. if obj_1 and obj_2 have the relations 'Inside' and 'OnTop' and their values are both True, filtered 'OnTop'
        ## 2.1 remove 'OnTop' for (obj_1, obj_2)
        if "Inside" in states and "OnTop" in states:
            filtered_edge[2]["states"] = [s for s in filtered_edge[2]["states"] if s[0] != "OnTop"]
        ## 2.2 if obj_1 is 'Under' obj_2 and obj_2 is 'Inside' obj_1, remove 'Under'
        ## 2.3 if obj_1 is 'Inisde' obj_2, we should never consider the 'Under' states for obj_1 and obj_3
        ## 2.4 if obj_1 is 'Inside' obj_2, we should never consider the 'OnTop' states for obj_1 and obj_3
        ## it is never possible to do 2.3 and 2.4 checking, so we need to check reversely
        ## 2.3 reverse: if obj_1 is 'Under' obj_2, and obj_1 is 'Inside' obj_3, remove 'Under'
        ## 2.4 reverse: if obj_1 is 'OnTop' obj_2, and obj_1 is 'Inside' obj_3, remove 'OnTop'
        ## merging 2.2 2.3 2.4, this basically means, for current edge (obj_1, obj_2), if detect 'Under' or 'OnTop', and obj_1 is 'Inside' something or the container, and obj_2 is inside something, then we do not consider this edge

        if 'Under' in states or 'OnTop' in states:
            # first find the (obj_2, obj_1) edge
            for find_obj1, find_obj2, find_states in all_edges:
                if find_obj1.name == obj_2.name \
                and find_obj2.name == obj_1.name \
                and 'Inside' in [s[0] for s in find_states["states"]]:
                    filtered_edge[2]["states"] = [s for s in filtered_edge[2]["states"] if s[0] != 'Under' and s[0] != 'OnTop']
                    break

        if 'Under' in states or 'OnTop' in states:
            obj_1_inside = False
            obj_2_inside = False
            for find_obj1, find_obj2, find_states in all_edges:
                if find_obj1.name == obj_1.name \
                and 'Inside' in [s[0] for s in find_states["states"]]:
                    obj_1_inside = True
                if find_obj1.name == obj_2.name \
                and 'Inside' in [s[0] for s in find_states["states"]]:
                    obj_2_inside = True
                if obj_1_inside and obj_2_inside:
                    filtered_edge[2]["states"] = [s for s in filtered_edge[2]["states"] if s[0] != 'Under' and s[0] != 'OnTop']
                    break
        
        
        # 3. if obj_1 is a robot and is grasping obj_2, filtered 'Contact'
        if isinstance(obj_1, BaseRobot) and ("LeftContact" in states and "LeftGrasping" in states):
            filtered_edge[2]["states"] = [s for s in filtered_edge[2]["states"] if s[0] != "LeftContact"]
        if isinstance(obj_1, BaseRobot) and ("RightContact" in states and "RightGrasping" in states):
            filtered_edge[2]["states"] = [s for s in filtered_edge[2]["states"] if s[0] != "RightContact"]
        
        # 4. filter robot or objectis under ceiling, directly remove the edge
        if "Under" in states and "ceiling" in obj_2.category:
            filtered_edge[2]["states"] = [s for s in filtered_edge[2]["states"] if s[0] != "Under"]
        
        if obj_1.category == obj_2.category:
            filtered_edge[2]["states"] = [s for s in filtered_edge[2]["states"] if s[0] != "OnTop" and s[0] != "Under" and s[0] != "Touching"]

        # 5. if floors in under obj_2, filter
        if (obj_1.category == "floors" or obj_1.category == "driveway" or obj_1.category == "lawn") and "Under" in states:
            filtered_edge[2]["states"] = [s for s in filtered_edge[2]["states"] if s[0] != "Under"]

        # 6. Experimental: if obj_1 is under obj_2 and obj_1 is being grasped, filtered
        if "Under" in states:
            for find_obj1, find_obj2, find_states in all_edges:
                if (find_obj2.name == obj_1.name or find_obj2.name == obj_2.name) \
                and isinstance(find_obj1, BaseRobot) \
                and ("LeftGrasping" in [s[0] for s in find_states["states"]] or "RightGrasping" in [s[0] for s in find_states["states"]]):
                    filtered_edge[2]["states"] = [s for s in filtered_edge[2]["states"] if s[0] != "Under"]
                    break

        return filtered_edge


    def _get_binary_filtered_edges(self, edges):
        '''
        Filter out edges that are not what we want, according to the binary rules.
        '''
        filtered_edges = []
        for edge in edges:
            filtered_edge = self._get_binary_filtered_edge(edge, edges)
            if len(filtered_edge[2]["states"]) > 0:
                filtered_edges.append(filtered_edge)
        return filtered_edges

    def _get_boolean_unary_states(self, obj):
        states = {}

        if isinstance(obj, BaseSystem):
            return {}

        # Add customized unary states
        customized_unary_states = CustomizedUnaryStates()
        for field in fields(customized_unary_states):
            state_name = field.name
            state_func = getattr(customized_unary_states, state_name)
            if not callable(state_func):
                continue

            value = state_func(obj)
            if self._only_true and not value:
                continue

            states[state_name] = value

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

        # Add customized binary states
        customized_binary_states = CustomizedBinaryStates()
        for obj1 in objs:

            if isinstance(obj1, BaseSystem):
                continue

            for obj2 in objs:
                if obj1 == obj2:
                    continue
                
                for field in fields(customized_binary_states):
                    state_name = field.name
                    state_func = getattr(customized_binary_states, state_name)
                    if not callable(state_func):
                        continue

                    if isinstance(obj2, BaseSystem):
                        continue
                    
                    value = state_func(obj1, obj2)
                    if self._only_true and not value:
                        continue

                    states.append((obj1, obj2, state_name, {"value": value}))
                    # print(f"Added ({obj1.name}, {state_name}, {obj2.name}) = {value}")

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
    
    def _get_object_candidates_via_heuristics(
            self,
            scene,
        ):
        '''
        Before checking, we use some heuristics to filter out some impossible objects.
        '''

        objects_to_add = set()
        # 0. if we only want task-relevant objects, filter them out first and directly return
        if self._task_relevant_objects is not None:
            objects_to_add = set(self._task_relevant_objects)
        # 1.1 temp method: we only consider the active white list objects
        # 1. first get all objects with contact changes
        # objects_to_add.update(self._contact_objects)

        # 2. then get all objects with acceleration changes
        # TODO: implement this

        # 3. after that, we get all objects with special properties. particles, temperature, fluid, etc
        # TODO: implement this

        # add a fallback here for now
        if len(objects_to_add) == 0:
            objects_to_add = set(scene.objects)
        return objects_to_add

    def start(self, scene, task=None):
        assert self._G is None, "Cannot start graph builder multiple times."

        if task is not None and self._only_task_relevant_objects:
            task_objects = [bddl_obj.wrapped_obj for bddl_obj in task.object_scope.values() 
                            if bddl_obj.wrapped_obj is not None and bddl_obj.exists]
            self._task_relevant_objects = [obj for obj in task_objects 
                                          if not isinstance(obj, BaseSystem)]
            system_objects = [obj for obj in task_objects 
                              if isinstance(obj, BaseSystem)]
            
            # add task relevant systems into tracking list if they are in the scene
            for sys_obj in system_objects:
                for obj in scene.systems:
                    if obj.name == sys_obj.name:
                        self._task_relevant_objects.append(obj)

                                          
            ## Jul 19 2025, this part is added only for B50 cases
            self._white_list_objects = [obj for obj in scene.objects if obj.name in EXTRA_OBJECT_WHITE_LIST]
            self._all_objects = self._task_relevant_objects + self._white_list_objects
            
            # print(f"Loaded {len(self._task_relevant_objects)} task relevant objects")
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

        self._initialized = True

    def step(self, scene):
        assert self._G is not None, "Cannot step graph builder before starting it."

        # Prepare the necessary transformations.
        desired_frame_to_world, world_to_desired_frame = self._get_desired_frame()

        # Update the position of everything that's already in the scene by using our relative position to last frame.
        old_desired_to_new_desired = world_to_desired_frame @ self._last_desired_frame_to_world
        nodes = list(self._G.nodes)

        if not self._semantic_only:
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


        # objs_to_add = self._get_object_candidates_via_heuristics(scene) # this is a common implementation, useful, but needs revision
        objs_to_add = self._all_objects
        
        if not self._full_obs:
            # If we're not in full observability mode, only pick the objects in FOV of robots.
            for robot in self._robots:
                objs_in_fov = robot.states[object_states.ObjectsInFOVOfRobot].get_value()
                objs_to_add &= objs_in_fov

        # # Remove all BaseRobot objects from the set of objects to add.
        # base_robots = [obj for obj in objs_to_add if isinstance(obj, BaseRobot)]
        # objs_to_add -= set(base_robots)
        

        for obj in objs_to_add:
            # Add the object if not already in the graph
            if obj not in self._G.nodes:
                self._G.add_node(obj)

                # 1. If the object is within self._obj_parent_dict, add a field called 'parent' to the object
                if obj.name in self._obj_parent_dict:
                    self._G.nodes[obj]["parent"] = self._obj_parent_dict[obj.name]
                    # print(f"Added parent {self._obj_parent_dict[obj.name]} to {obj.name}")

            if not self._semantic_only:
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
            edges = self._get_binary_filtered_edges(edges)
        self._G.add_edges_from(edges)

        # Save the robot's transform in this frame.
        self._last_desired_frame_to_world = desired_frame_to_world

        # 3. clear self._last_removed_object
        self._last_removed_object = []


def visualize_scene_graph(scene, G, show_window=True, cartesian_positioning=False):
    """
    Converts the graph into an image and shows it in a cv2 window if preferred.
    Note: Currently, this function only works when we merge parallel edges, i.e. the graph is a DiGraph.

    Args:
        scene (Scene): The scene object containing the scene graph.
        G (Graph): The graph object to visualize.
        show_window (bool): Whether a cv2 GUI window containing the visualization should be shown.
        cartesian_positioning (bool): Whether nodes should be positioned based on their position in the scene (if True)
            or placed using a graphviz layout (neato) that makes it easier to read edges & find clusters.
    """

    nodes = list(G.nodes)
    all_robots = [robot for robot in nodes if isinstance(robot, BaseRobot)]

    def _draw_graph():
        node_labels = {obj: obj.category for obj in nodes}

        # get all objects in fov of robots
        objects_in_fov = set()
        for robot in all_robots:
            objects_in_fov.update(robot.states[object_states.ObjectsInFOVOfRobot].get_value())
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
        s for s in robot.sensors.values() if isinstance(s, VisionSensor) and "eyes" in s.name and "rgb" in s.modalities
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
    graph_view = th.from_numpy(np.asarray(fig.canvas.renderer.buffer_rgba())[:, :, :3])
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
