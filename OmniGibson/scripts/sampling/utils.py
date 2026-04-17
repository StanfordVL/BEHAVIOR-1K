import torch as th
import csv
import json
import os
import copy
import omnigibson as og
import omnigibson.lazy as lazy
import omnigibson.utils.transform_utils as T
from bddl.knowledge_base import CompiledTask
from omnigibson.objects import DatasetObject
from omnigibson.systems import MicroPhysicalParticleSystem
from omnigibson.utils.asset_utils import get_dataset_path
from omnigibson.macros import gm, macros


folder_path = os.path.dirname(os.path.abspath(__file__))

SCENE_INFO_FPATH = os.path.join(folder_path, "BEHAVIOR-1K Scenes.csv")

UNSUPPORTED_PREDICATES = {"broken", "assembled"}


def prune_unevaluatable_predicates(init_conditions):
    pruned_conditions = []
    for condition in init_conditions:
        if condition.body[0] in {"insource", "future", "real"}:
            continue
        pruned_conditions.append(condition)

    return pruned_conditions


def get_predicates(conds):
    preds = []
    if isinstance(conds, str):
        return preds
    assert isinstance(conds, list)
    contains_list = any(isinstance(ele, list) for ele in conds)
    if contains_list:
        for ele in conds:
            preds += get_predicates(ele)
    else:
        preds.append(conds[0])
    return preds


def get_subjects(conds):
    subjs = []
    if isinstance(conds, str):
        return subjs
    assert isinstance(conds, list)
    contains_list = any(isinstance(ele, list) for ele in conds)
    if contains_list:
        for ele in conds:
            subjs += get_subjects(ele)
    else:
        subjs.append(conds[1])
    return subjs


def get_rooms(conds):
    rooms = []
    if isinstance(conds, str):
        return rooms
    assert isinstance(conds, list)
    contains_list = any(isinstance(ele, list) for ele in conds)
    if contains_list:
        for ele in conds:
            rooms += get_rooms(ele)
    elif conds[0] == "inroom":
        rooms.append(conds[2])
    return rooms


def get_scenes():
    scenes = set()
    with open(SCENE_INFO_FPATH) as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        for i, row in enumerate(reader):
            # Skip first row since it's the header
            if i == 0:
                continue
            scenes.add(row[0])

    return tuple(sorted(scenes))


def get_valid_tasks():
    from omnigibson.utils.bddl_utils import get_behavior_activities

    return set(get_behavior_activities())


def get_all_lights(prim):
    prims = []
    for child in prim.GetChildren():
        if "Light" in child.GetPrimTypeInfo().GetTypeName():
            prims.append(child)
        prims += get_all_lights(child)

    return prims


def hide_all_lights():
    lights = get_all_lights(prim=og.sim.world_prim)
    for light in lights:
        imageable = lazy.pxr.UsdGeom.Imageable(light)
        imageable.MakeInvisible()


def parse_task_mapping_new():
    if os.path.exists("task_mapping.json"):
        with open("task_mapping.json", "r") as f:
            mapping = json.load(f)
        return mapping

    from omnigibson.utils.bddl_utils import get_knowledge_base

    tasks = get_knowledge_base().all_tasks()
    mapping = dict()
    for task in tasks:
        task_name = task.name[:-2]
        scenes = []
        for scene, status in task.scene_matching_dict.items():
            if status["matched"]:
                scenes.append(scene.name)
        mapping[task_name] = scenes
    with open("task_mapping.json", "w") as f:
        json.dump(mapping, f, indent=4)
    return mapping


def get_scene_compatible_activities(scene_model, mapping):
    return [activity for activity, scenes in mapping.items() if scene_model in scenes]


def _validate_object_state_stability(obj_name, obj_dict, strict=False):
    lin_vel_threshold = 0.001 if strict else 1.0
    ang_vel_threshold = 0.005 if strict else th.pi
    joint_vel_threshold = 0.01 if strict else 1.0
    # Check close to zero root link velocity
    for key, atol in zip(("lin_vel", "ang_vel"), (lin_vel_threshold, ang_vel_threshold)):
        val = obj_dict["root_link"].get(key, th.tensor(0.0))
        if not isinstance(val, th.Tensor):
            val = th.tensor(val)
        if not th.all(th.isclose(val, th.tensor(0.0), atol=atol, rtol=0.0)).item():
            raise ValueError(f"{obj_name} root link {key} is not close to 0: {val}")

    # Check close to zero joint velocities
    if "joint_vel" in obj_dict.keys():
        val = obj_dict["joint_vel"]
        if not isinstance(val, th.Tensor):
            val = th.tensor(val)
        if not th.all(th.isclose(val, th.tensor(0.0), atol=joint_vel_threshold, rtol=0.0)).item():
            raise ValueError(f"{obj_name} joint velocity is not close to 0: {val}")

    # If all passes, return True
    return True


def create_stable_scene_json(scene_model):
    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": scene_model,
        },
    }

    # Disable sleeping
    macros.prims.entity_prim.DEFAULT_SLEEP_THRESHOLD = 0.0

    # Create the environment
    env = og.Environment(configs=copy.deepcopy(cfg))

    # Take a few steps to let objects settle, then update the scene initial state
    for _ in range(300):
        og.sim.step()

    # Sanity check for zero velocities for all objects
    stable_state = og.sim.dump_state()[0]
    if "registry" in stable_state:
        stable_state = stable_state["registry"]
    invalid_msgs = []
    for obj_name, obj_info in stable_state["object_registry"].items():
        valid_obj, err_msg = _validate_object_state_stability(obj_name, obj_info, strict=False)
        if not valid_obj:
            invalid_msgs.append(err_msg)

    if len(invalid_msgs) > 0:
        print("Creating stable scene failed! Invalid messages:")
        for msg in invalid_msgs:
            print(msg)
        raise ValueError("Scene is not stable!")

    for obj in env.scene.objects:
        obj.keep_still()
    env.scene.update_initial_file()

    # Save this as a stable file
    path = os.path.join(
        get_dataset_path("behavior-1k-assets"), "scenes", env.scene.scene_model, "json", f"{scene_model}_stable.json"
    )
    og.sim.save(json_paths=[path])

    og.sim.stop()
    og.clear()


def validate_task(task, task_scene_dict, default_scene_dict):
    assert og.sim.is_playing()

    conditions = task._base_conditions
    relevant_rooms = set(get_rooms(conditions.parsed_initial_conditions))
    active_obj_names = set()
    for obj in og.sim.scenes[0].objects:
        if isinstance(obj, DatasetObject):
            obj_rooms = {"_".join(room.split("_")[:-1]) for room in obj.in_rooms}
            active = len(relevant_rooms.intersection(obj_rooms)) > 0 or obj.category in {"floors", "walls"}
            if active:
                active_obj_names.add(obj.name)

    # 1. Sanity check all object poses wrt their original pre-loaded poses
    print("Step 1: Checking loaded task environment...")

    def _validate_identical_object_kinematic_state(obj_name, default_obj_dict, obj_dict, check_vel=True):
        # Check root link state
        for key, val in default_obj_dict["root_link"].items():
            # Skip velocities if requested
            if not check_vel and "vel" in key:
                continue
            obj_val = obj_dict["root_link"][key]
            val = th.tensor(val) if not isinstance(val, th.Tensor) else val
            obj_val = th.tensor(obj_val) if not isinstance(obj_val, th.Tensor) else obj_val
            atol = 1.0 if "vel" in key else 0.05
            # # TODO: Update ori value to be larger tolerance
            # tol = 0.15 if "ori" in key else 0.05
            # If particle positions are being checked, only check the min / max
            if "particle" in key:
                # Only check particle position
                if "position" in key:
                    pos_min, pos_max = th.min(val, dim=0), th.max(val, dim=0)
                    curr_pos_min, curr_pos_max = (
                        th.min(obj_val, dim=0),
                        th.max(obj_val, dim=0),
                    )
                    for name, pos, curr_pos in zip(("min", "max"), (pos_min, pos_max), (curr_pos_min, curr_pos_max)):
                        if not th.all(th.isclose(pos, curr_pos, atol=0.05)).item():
                            raise ValueError(
                                f"Got mismatch in cloth {obj_name} particle positions range: {name} min {pos_min} max {pos_max} vs. min {curr_pos_min} max {curr_pos_max}"
                            )
                else:
                    continue
            else:
                if key == "ori":
                    if not th.all(
                        th.isclose(T.quat_distance(val, obj_val), th.tensor([0.0, 0.0, 0.0, 1.0]), atol=atol)
                    ).item():
                        raise ValueError(
                            f"{obj_name} root link mismatch in {key}: default_obj_dict has: {val}, obj_dict has: {obj_val}"
                        )
                else:
                    if not th.all(th.isclose(val, obj_val, atol=atol, rtol=0.0)).item():
                        raise ValueError(
                            f"{obj_name} root link mismatch in {key}: default_obj_dict has: {val}, obj_dict has: {obj_val}"
                        )

        # Check any non-robot joint values
        # This is because the controller can cause the robot to drift over time
        if "robot" not in obj_name:
            # Check joint states
            if "joint_pos" in default_obj_dict.keys():
                for key in ("pos", "vel"):
                    val = default_obj_dict[f"joint_{key}"]
                    obj_val = obj_dict[f"joint_{key}"]
                    atol = 1.0 if key == "vel" else 0.05
                    if not isinstance(val, th.Tensor):
                        val = th.tensor(val)
                    if not isinstance(obj_val, th.Tensor):
                        obj_val = th.tensor(obj_val)
                    if not th.all(th.isclose(val, obj_val, atol=atol, rtol=0.0)).item():
                        raise ValueError(
                            f"{obj_name} joint mismatch in {key}: default_obj_dict has: {val}, obj_dict has: {obj_val}"
                        )

        # If all passes, return True
        return True

    task_state_t0 = og.sim.dump_state(serialized=False)[0]
    if "registry" in task_state_t0:
        task_state_t0 = task_state_t0["registry"]
    for obj_name, obj_info in task_scene_dict["state"]["registry"]["object_registry"].items():
        current_obj_info = task_state_t0["object_registry"][obj_name]
        _validate_identical_object_kinematic_state(obj_name, obj_info, current_obj_info, check_vel=True)

    # We should never use this after
    task_scene_dict = None

    # 2. Validate the native USDs jsons are stable and similar -- compare all object kinematics (poses, joint
    #       states) with respect to the native scene file
    print("Step 2: Checking poses and joint states for non-task-relevant objects and velocities for all objects...")

    # Sanity check all non-task-relevant object poses
    for obj_name, default_obj_info in default_scene_dict["state"]["registry"]["object_registry"].items():
        # Skip any active objects since they may have changed
        if obj_name in active_obj_names:
            continue
        # HACK: need to skip objects in default scene when partial loading during sampling
        if obj_name not in task_state_t0["object_registry"]:
            continue
        obj_info = task_state_t0["object_registry"][obj_name]
        _validate_identical_object_kinematic_state(obj_name, default_obj_info, obj_info, check_vel=True)

    # Sanity check for zero velocities for all objects
    for obj_name, obj_info in task_state_t0["object_registry"].items():
        _validate_object_state_stability(obj_name, obj_info, strict=False)

    # Need to enable transition rules before running step 3 and 4
    original_transition_rule_flag = gm.ENABLE_TRANSITION_RULES
    with gm.unlocked():
        gm.ENABLE_TRANSITION_RULES = True

    # 3. Validate object set is consistent (no faulty transition rules occurring) -- we expect the number
    #       of active systems (and number of active particles) and the number of objects to be the same after
    #       taking a physics step, and also make sure init state is True
    print("Step 3: Checking BehaviorTask initial conditions and scene stability...")
    # Take a single physics step
    og.sim.step()
    task_state_t1 = og.sim.dump_state(serialized=False)[0]
    if "registry" in task_state_t1:
        task_state_t1 = task_state_t1["registry"]

    def _validate_scene_stability(task, task_state, current_state, check_particle_positions=True):
        def _validate_particle_system_consistency(
            system_name, system_state, current_system_state, check_particle_positions=True
        ):
            is_micro_physical = isinstance(og.sim.scenes[0].get_system(system_name), MicroPhysicalParticleSystem)
            n_particles_key = "instancer_particle_counts" if is_micro_physical else "n_particles"
            if (
                is_micro_physical
                and not (
                    th.isclose(
                        th.tensor(system_state[n_particles_key]),
                        th.tensor(current_system_state[n_particles_key]),
                    ).all()
                )
            ) or (not is_micro_physical and system_state[n_particles_key] != current_system_state[n_particles_key]):
                raise ValueError(
                    f"Got inconsistent number of system {system_name} particles: {system_state[n_particles_key]} vs. {current_system_state[n_particles_key]}"
                )

            # Validate that no particles went flying -- maximum ranges of positions should be roughly close
            n_particles = (
                th.tensor(system_state[n_particles_key]).sum().item()
                if is_micro_physical
                else system_state[n_particles_key]
            )
            if n_particles > 0 and check_particle_positions:
                if is_micro_physical:
                    particle_positions = th.concatenate(
                        [inst_state["particle_positions"] for inst_state in system_state["particle_states"].values()],
                        dim=0,
                    )
                    current_particle_positions = th.concatenate(
                        [
                            inst_state["particle_positions"]
                            for inst_state in current_system_state["particle_states"].values()
                        ],
                        dim=0,
                    )
                else:
                    particle_positions = th.tensor(system_state["positions"])
                    current_particle_positions = th.tensor(current_system_state["positions"])
                pos_min, pos_max = th.min(particle_positions, dim=0).values, th.max(particle_positions, dim=0).values
                curr_pos_min, curr_pos_max = (
                    th.min(current_particle_positions, dim=0).values,
                    th.max(current_particle_positions, dim=0).values,
                )
                for name, pos, curr_pos in zip(("min", "max"), (pos_min, pos_max), (curr_pos_min, curr_pos_max)):
                    if not th.all(th.isclose(pos, curr_pos, atol=0.05)).item():
                        raise ValueError(
                            f"Got mismatch in system {system_name} particle positions range: {name} {pos} vs. {curr_pos}"
                        )

            return True

        # Sanity check consistent objects
        task_objects = {obj_name for obj_name in task_state["object_registry"].keys()}
        curr_objects = {obj_name for obj_name in current_state["object_registry"].keys()}
        mismatched_objs = set.union(task_objects, curr_objects) - set.intersection(task_objects, curr_objects)
        if len(mismatched_objs) > 0:
            raise ValueError(f"Got mismatch in active objects: {mismatched_objs}")

        for obj_name, obj_info in task_state["object_registry"].items():
            current_obj_info = current_state["object_registry"][obj_name]
            _validate_identical_object_kinematic_state(obj_name, obj_info, current_obj_info, check_vel=True)

        # Sanity check consistent particle systems
        task_systems = {system_name for system_name in task_state["system_registry"].keys() if system_name != "cloth"}
        curr_systems = {
            system_name for system_name in current_state["system_registry"].keys() if system_name != "cloth"
        }
        mismatched_systems = set.union(task_systems, curr_systems) - set.intersection(task_systems, curr_systems)
        if len(mismatched_systems) > 0:
            raise ValueError(f"Got mismatch in active systems: {mismatched_systems}")

        for system_name in task_systems:
            system_state = task_state["system_registry"][system_name]
            curr_system_state = current_state["system_registry"][system_name]
            _validate_particle_system_consistency(
                system_name, system_state, curr_system_state, check_particle_positions=check_particle_positions
            )

        # Sanity check initial state
        valid_init_state, results = CompiledTask.evaluate_conditions(
            prune_unevaluatable_predicates(task.activity_initial_conditions), task._evaluate_predicate
        )
        if not valid_init_state:
            raise ValueError(f"BDDL Task init conditions were invalid. Results: {results}")

        return True

    # Sanity check scene
    _validate_scene_stability(
        task=task, task_state=task_state_t0, current_state=task_state_t1, check_particle_positions=True
    )

    # 4. Validate longer-term stability -- take N=10 timesteps, and make sure all object positions and velocities
    #       are still stable (positions don't drift too much, and velocities are close to 0), as well as verifying
    #       that all BDDL conditions are satisfied
    print("Step 4: Checking longer-term BehaviorTask initial conditions and scene stability...")

    # Take 10 steps
    for _ in range(10):
        og.sim.step()

    # Sanity check scene
    # Don't check particle positions since some particles may be falling
    # TODO: Tighten this constraint once we figure out a way to stably sample particles
    task_state_t11 = og.sim.dump_state(serialized=False)[0]
    if "registry" in task_state_t11:
        task_state_t11 = task_state_t11["registry"]
    _validate_scene_stability(
        task=task, task_state=task_state_t0, current_state=task_state_t11, check_particle_positions=False
    )
    with gm.unlocked():
        gm.ENABLE_TRANSITION_RULES = original_transition_rule_flag

    return True
