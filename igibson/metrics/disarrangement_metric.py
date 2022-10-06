import copy

import numpy as np

from igibson.metrics.metric_base import MetricBase
from igibson.object_states import Inside, NextTo, OnFloor, OnTop, Pose, Touching, Under
from igibson.object_states.object_state_base import AbsoluteObjectState, BooleanState
from igibson.object_states.on_floor import RoomFloor
from igibson.objects.multi_object_wrappers import ObjectMultiplexer
from igibson.robots.robot_base import BaseRobot

SIMULATOR_SETTLE_TIME = 150


class KinematicDisarrangement(MetricBase):
    def __init__(self):
        self.initialized = False

        self.integrated_disarrangement = 0
        self.delta_disarrangement = []

        self.delta_obj_disp_dict = {}
        self.int_obj_disp_dict = {}

    def update_state_cache(self, env):
        state_cache = {}
        for obj_id, obj in env.scene.objects_by_name.items():
            if isinstance(obj, BaseRobot):
                continue
            if type(obj) == ObjectMultiplexer:
                assert (
                    len(obj._multiplexed_objects[1].objects) == 2
                ), "Kinematic caching only supported for multiplexed objects of len 2"
                part_pose = [
                    obj._multiplexed_objects[1].objects[0].states[Pose].get_value(),
                    obj._multiplexed_objects[1].objects[1].states[Pose].get_value(),
                ]
                state_cache[obj_id] = {
                    "pose": {
                        "base": obj._multiplexed_objects[0].states[Pose].get_value(),
                        "children": part_pose,
                    },
                    "active": obj.current_index,
                }
            else:
                state_cache[obj_id] = {
                    "pose": {
                        "base": obj.states[Pose].get_value(),
                    },
                    "active": 0,
                }
        return state_cache

    @staticmethod
    def calculate_object_disarrangement(obj, prev_state_cache, cur_state_cache):
        """
        This function implements logic to check the current index in order to aggregate kinematic state as
        an object is split/joined. This logic should be shifted into the multiplexer class, as it requires
        a significant amount of (error prone) handling.
        """
        obj_disarrangement = {"base": 0, "children": [0, 0]}
        if prev_state_cache[obj]["active"] == 0 and cur_state_cache[obj]["active"] == 0:
            obj_disarrangement["base"] = np.linalg.norm(
                cur_state_cache[obj]["pose"]["base"][0] - prev_state_cache[obj]["pose"]["base"][0]
            )
        elif prev_state_cache[obj]["active"] == 0 and cur_state_cache[obj]["active"] == 1:
            obj_disarrangement["children"][0] = np.linalg.norm(
                cur_state_cache[obj]["pose"]["children"][0][0] - prev_state_cache[obj]["pose"]["base"][0]
            )
            obj_disarrangement["children"][1] = np.linalg.norm(
                cur_state_cache[obj]["pose"]["children"][1][0] - prev_state_cache[obj]["pose"]["base"][0]
            )
        elif prev_state_cache[obj]["active"] == 1 and cur_state_cache[obj]["active"] == 0:
            obj_disarrangement["children"][0] = np.linalg.norm(
                cur_state_cache[obj]["pose"]["base"][0] - prev_state_cache[obj]["pose"]["children"][0][0]
            )
            obj_disarrangement["children"][1] = np.linalg.norm(
                cur_state_cache[obj]["pose"]["base"][0] - prev_state_cache[obj]["pose"]["children"][1][0]
            )
        elif prev_state_cache[obj]["active"] == 1 and cur_state_cache[obj]["active"] == 1:
            obj_disarrangement["children"][0] = np.linalg.norm(
                cur_state_cache[obj]["pose"]["children"][0][0] - prev_state_cache[obj]["pose"]["children"][0][0]
            )
            obj_disarrangement["children"][1] = np.linalg.norm(
                cur_state_cache[obj]["pose"]["children"][1][0] - prev_state_cache[obj]["pose"]["children"][1][0]
            )
        else:
            raise Exception
        return obj_disarrangement

    def step_callback(self, env, _):
        total_disarrangement = 0
        self.cur_state_cache = self.update_state_cache(env)

        if not self.initialized:
            self.prev_state_cache = copy.deepcopy(self.cur_state_cache)
            self.initial_state_cache = copy.deepcopy(self.cur_state_cache)
            self.delta_obj_disp_dict = {obj: {"base": [], "children": []} for obj in self.cur_state_cache}
            self.int_obj_disp_dict = {obj: {"base": 0, "children": [0, 0]} for obj in self.cur_state_cache}
            self.initialized = True

        for obj in self.prev_state_cache:
            obj_disarrangement = self.calculate_object_disarrangement(obj, self.prev_state_cache, self.cur_state_cache)
            total_disarrangement += obj_disarrangement["base"]
            total_disarrangement += np.sum(obj_disarrangement["children"])

            self.delta_obj_disp_dict[obj]["base"].append(obj_disarrangement["base"])
            self.delta_obj_disp_dict[obj]["children"].append(obj_disarrangement["children"])

            self.int_obj_disp_dict[obj]["base"] += obj_disarrangement["base"]
            self.int_obj_disp_dict[obj]["children"] = np.array(self.int_obj_disp_dict[obj]["children"]) + np.array(
                obj_disarrangement["children"]
            )

        self.prev_state_cache = copy.deepcopy(self.cur_state_cache)
        self.integrated_disarrangement += total_disarrangement
        self.delta_disarrangement.append(total_disarrangement)

        return total_disarrangement

    @property
    def relative_disarrangement(self):
        relative_disarrangement = 0
        for obj in self.initial_state_cache:
            disarrangement = self.calculate_object_disarrangement(obj, self.initial_state_cache, self.cur_state_cache)
            relative_disarrangement += disarrangement["base"]
            relative_disarrangement += np.sum(disarrangement["children"])
        return relative_disarrangement

    def gather_results(self):
        return {
            "kinematic_disarrangement": {
                "relative": self.relative_disarrangement,
                "timestep": self.delta_disarrangement,
                "integrated": self.integrated_disarrangement,
            }
        }


class LogicalDisarrangement(MetricBase):
    def __init__(self):
        self.initialized = False

        self.state_cache = {}
        self.next_state_cache = {}

    @staticmethod
    def cache_single_object(obj_id, obj, room_floors, env):
        obj_cache = {}
        for state_class, state in obj.states.items():
            if not isinstance(state, BooleanState):
                continue
            if isinstance(state, AbsoluteObjectState):
                obj_cache[state_class] = state.get_value()
            # TODO (mjlbach): room floors are not currently proper objects, this means special logic
            # is needed to handle onFloor until this is fixed
            elif isinstance(state, OnFloor):
                relational_state_cache = {}
                for floor_id, floor in room_floors.items():
                    relational_state_cache[floor_id] = state.get_value(floor)
                obj_cache[state_class] = relational_state_cache
            else:
                relational_state_cache = {}
                for target_obj_id, target_obj in env.scene.objects_by_name.items():
                    if obj_id == target_obj_id or isinstance(target_obj, BaseRobot):
                        continue
                    relational_state_cache[target_obj_id] = False
                    # Relational states with multiplexed target objects currently unhandled
                    # For example, inside apple cabinet is supported, inside cabinet apple is not
                    if type(target_obj) == ObjectMultiplexer:
                        pass
                    else:
                        relational_state_cache[target_obj_id] = state.get_value(target_obj)
                obj_cache[state_class] = relational_state_cache
        return obj_cache

    def create_object_logical_state_cache(self, env):
        room_floors = {
            "room_floor_"
            + room_inst: RoomFloor(
                category="room_floor",
                name="room_floor_" + room_inst,
                scene=env.scene,
                room_instance=room_inst,
                floor_obj=env.scene.objects_by_name["floors"],
            )
            for room_inst in env.scene.room_ins_name_to_ins_id.keys()
        }

        state_cache = {}
        for obj_id, obj in env.scene.objects_by_name.items():
            if isinstance(obj, BaseRobot):
                continue
            state_cache[obj_id] = {}
            if type(obj) == ObjectMultiplexer:
                if obj.current_index == 0:
                    cache_base = self.cache_single_object(obj_id, obj._multiplexed_objects[0], room_floors, env)
                    cache_part_1 = None
                    cache_part_2 = None
                else:
                    cache_base = None
                    cache_part_1 = self.cache_single_object(
                        obj_id, obj._multiplexed_objects[1].objects[0], room_floors, env.task
                    )
                    cache_part_2 = self.cache_single_object(
                        obj_id, obj._multiplexed_objects[1].objects[1], room_floors, env.task
                    )
                state_cache[obj_id] = {
                    "base_states": cache_base,
                    "part_states": [cache_part_1, cache_part_2],
                    "active": obj.current_index,
                    "type": "multiplexer",
                }
            else:
                cache_base = self.cache_single_object(obj_id, obj, room_floors, env.task)
                state_cache[obj_id] = {
                    "base_states": cache_base,
                    "type": "standard",
                }
        return state_cache

    def diff_object_states(self, obj_1_states, obj_2_states):
        total_states = 0
        non_kinematic_edits = 0
        kinematic_edits = 0
        for state in obj_1_states:
            total_states += 1
            if obj_1_states[state] != obj_2_states[state]:
                if state in [Inside, Under, OnTop, Touching, OnFloor]:
                    kinematic_edits = 1
                elif state in [NextTo]:
                    pass
                else:
                    non_kinematic_edits += 1
        return total_states, kinematic_edits, non_kinematic_edits

    def compute_logical_disarrangement(self, object_state_cache_1, object_state_cache_2):
        total_edit_distance = 0
        total_objects = 0
        total_states = 0
        for obj_id in object_state_cache_1:
            kinematic_edits = 0
            non_kinematic_edits = 0
            total_objects += 1
            obj_total_states = 0
            obj_kinematic_edits = 0
            obj_non_kinematic_edits = 0
            if object_state_cache_1[obj_id]["type"] == "multiplexer":
                if object_state_cache_1[obj_id]["active"] == 0 and object_state_cache_2[obj_id]["active"] == 0:
                    obj_1_states = object_state_cache_1[obj_id]["base_states"]
                    obj_2_states = object_state_cache_2[obj_id]["base_states"]
                    (
                        obj_total_states,
                        obj_kinematic_edits,
                        obj_non_kinematic_edits,
                    ) = self.diff_object_states(obj_1_states, obj_2_states)
                elif object_state_cache_1[obj_id]["active"] == 0 and object_state_cache_2[obj_id]["active"] == 1:
                    obj_1_states = object_state_cache_1[obj_id]["base_states"]
                    obj_2_1_states = object_state_cache_2[obj_id]["part_states"][0]
                    obj_2_2_states = object_state_cache_2[obj_id]["part_states"][1]
                    (
                        obj_total_states,
                        obj_kinematic_edits,
                        obj_non_kinematic_edits,
                    ) = self.diff_object_states(obj_1_states, obj_2_1_states)
                    total_states += obj_total_states
                    kinematic_edits += obj_kinematic_edits
                    kinematic_edits += obj_non_kinematic_edits
                    (
                        obj_total_states,
                        obj_kinematic_edits,
                        obj_non_kinematic_edits,
                    ) = self.diff_object_states(obj_1_states, obj_2_2_states)
                elif object_state_cache_1[obj_id]["active"] == 1 and object_state_cache_2[obj_id]["active"] == 0:
                    obj_1_1_states = object_state_cache_1[obj_id]["part_states"][0]
                    obj_1_2_states = object_state_cache_1[obj_id]["part_states"][1]
                    obj_2_states = object_state_cache_2[obj_id]["base_states"]
                    (
                        obj_total_states,
                        obj_kinematic_edits,
                        obj_non_kinematic_edits,
                    ) = self.diff_object_states(obj_1_1_states, obj_2_states)
                    total_states += obj_total_states
                    kinematic_edits += obj_kinematic_edits
                    kinematic_edits += obj_non_kinematic_edits
                    (
                        obj_total_states,
                        obj_kinematic_edits,
                        obj_non_kinematic_edits,
                    ) = self.diff_object_states(obj_1_2_states, obj_2_states)
                elif object_state_cache_1[obj_id]["active"] == 1 and object_state_cache_2[obj_id]["active"] == 1:
                    obj_1_1_states = object_state_cache_1[obj_id]["part_states"][0]
                    obj_1_2_states = object_state_cache_1[obj_id]["part_states"][1]
                    obj_2_1_states = object_state_cache_1[obj_id]["part_states"][0]
                    obj_2_2_states = object_state_cache_1[obj_id]["part_states"][1]
                    (
                        obj_total_states,
                        obj_kinematic_edits,
                        obj_non_kinematic_edits,
                    ) = self.diff_object_states(obj_1_1_states, obj_2_1_states)
                    total_states += obj_total_states
                    kinematic_edits += obj_kinematic_edits
                    kinematic_edits += obj_non_kinematic_edits
                    (
                        obj_total_states,
                        obj_kinematic_edits,
                        obj_non_kinematic_edits,
                    ) = self.diff_object_states(obj_1_2_states, obj_2_2_states)
                else:
                    raise Exception
            else:
                obj_1_states = object_state_cache_1[obj_id]["base_states"]
                obj_2_states = object_state_cache_2[obj_id]["base_states"]
                (
                    obj_total_states,
                    obj_kinematic_edits,
                    obj_non_kinematic_edits,
                ) = self.diff_object_states(obj_1_states, obj_2_states)
            total_states += obj_total_states
            kinematic_edits += obj_kinematic_edits
            kinematic_edits += obj_non_kinematic_edits

            total_edit_distance += non_kinematic_edits
            total_edit_distance += kinematic_edits

        return {
            "total_objects": total_objects,
            "total_edit_distance": total_edit_distance,
            "total_states": total_states,
        }

    def step_callback(self, env, _):
        if not self.initialized and env.simulator.frame_count == SIMULATOR_SETTLE_TIME:
            self.initial_state_cache = self.create_object_logical_state_cache(env)
            self.initialized = True
        else:
            return

    def end_callback(self, env, _):
        """
        When pybullet sleeps objects, getContactPoints is no longer refreshed
        Setting collision groups on the agent (which happens when users first activate the agent)
        Wipes active collision groups. To get the logical disarrangement, we must wake up all objects in the scene
        This can only be done at the end of the scene so as to not affect determinism.
        """
        for obj in env.scene.objects_by_name.values():
            obj.force_wakeup()
        env.simulator.step()

        self.cur_state_cache = self.create_object_logical_state_cache(env)

        self.relative_logical_disarrangement = self.compute_logical_disarrangement(
            self.initial_state_cache, self.cur_state_cache
        )

    def gather_results(self):
        return {
            "logical_disarrangement": {
                "relative": self.relative_logical_disarrangement["total_edit_distance"],
                "total_objects": self.relative_logical_disarrangement["total_objects"],
                "total_states": self.relative_logical_disarrangement["total_states"],
            }
        }
