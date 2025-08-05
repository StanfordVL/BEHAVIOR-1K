import os
import random
import re
from collections import defaultdict
from copy import deepcopy

import bddl
import networkx as nx
import torch as th
from bddl.activity import get_goal_conditions, get_ground_goal_state_options, get_initial_conditions
from bddl.backend_abc import BDDLBackend
from bddl.condition_evaluation import Negation
from bddl.config import get_definition_filename
from bddl.logic_base import AtomicFormula, BinaryAtomicFormula, UnaryAtomicFormula
from bddl.object_taxonomy import ObjectTaxonomy

import omnigibson as og
from omnigibson import object_states
from omnigibson.macros import create_module_macros
from omnigibson.object_states.factory import _KINEMATIC_STATE_SET, get_system_states
from omnigibson.object_states.object_state_base import AbsoluteObjectState, RelativeObjectState
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.utils.asset_utils import (
    get_all_object_categories,
    get_all_object_category_models_with_abilities,
    get_attachment_meta_links,
)
from omnigibson.utils.constants import PrimType
from omnigibson.utils.python_utils import Wrapper
from omnigibson.utils.ui_utils import create_module_logger

# Create module logger
log = create_module_logger(module_name=__name__)


# Create settings for this module
m = create_module_macros(module_path=__file__)

m.MIN_DYNAMIC_SCALE = 0.5
m.DYNAMIC_SCALE_INCREMENT = 0.1

GOOD_MODELS = {
    # "jar": {"kijnrj"},
    # "carton": {"causya", "msfzpz", "sxlklf"},
    # "hamper": {"drgdfh", "hlgjme", "iofciz", "pdzaca", "ssfvij"},
    # "hanging_plant": set(),
    # "hardback": {"esxakn"},
    # "notebook": {"hwhisw"},
    # "paperback": {"okcflv"},
    # "plant_pot": {"ihnfbi", "vhglly", "ygrtaz"},
    # "pot_plant": {"cvthyv", "dbjcic", "cecdwu"},
    # "recycling_bin": {"nuoypc"},
    # "tray": {"gsxbym", "huwhjg", "txcjux", "uekqey", "yqtlhy"},
}

GOOD_BBOXES = {
    # "basil": {
    #     "dkuhvb": [0.07286304, 0.0545199, 0.03108144],
    # },
    # "basil_jar": {
    #     "swytaw": [0.22969539, 0.19492961, 0.30791675],
    # },
    # "bicycle_chain": {
    #     "czrssf": [0.242, 0.012, 0.021],
    # },
    # "clam": {
    #     "ihhbfj": [0.078, 0.081, 0.034],
    # },
    # "envelope": {
    #     "urcigc": [0.004, 0.06535058, 0.10321216],
    # },
    # "mail": {
    #     "azunex": [0.19989018, 0.005, 0.12992871],
    #     "gvivdi": [0.28932137, 0.005, 0.17610794],
    #     "mbbwhn": [0.27069291, 0.005, 0.13114884],
    #     "ojkepk": [0.19092424, 0.005, 0.13252979],
    #     "qpwlor": [0.22472473, 0.005, 0.18983322],
    # },
    # "pill_bottle": {
    #     "csvdbe": [0.078, 0.078, 0.109],
    #     "wsasmm": [0.078, 0.078, 0.109],
    # },
    # "plant_pot": {
    #     "ihnfbi": [0.24578613, 0.2457865, 0.18862737],
    # },
    # "razor": {
    #     "jocsgp": [0.046, 0.063, 0.204],
    # },
    # "recycling_bin": {
    #     "nuoypc": [0.69529409, 0.80712041, 1.07168694],
    # },
    # "tupperware": {
    #     "mkstwr": [0.33, 0.33, 0.21],
    # },
    # "copper_wire": {
    #     "nzafel": [0.1762, 0.17655, 0.0631],
    # },
    # "backpack": {
    #     "gvbiwl": [0.7397, 0.6109, 0.6019],
    # },
}

BAD_CLOTH_MODELS = {
    # "bandana": {"wbhliu"},
    # "curtain": {"ohvomi", "shbakk"},
    # "cardigan": {"itrkhr"},
    # "sweatshirt": {"nowqqh"},
    # "jeans": {"nmvvil", "pvzxyp"},
    # "pajamas": {"rcgdde"},
    # "polo_shirt": {"vqbvph"},
    # "vest": {"girtqm"},  # bddl NOT FIXED
    # "onesie": {"pbytey"},
    # "dishtowel": {"ltydgg"},
    # "dress": {"gtghon"},
    # "hammock": {"aiftuk", "fglfga", "klhkgd", "lqweda", "qewdqa"},
    # "jacket": {"kiiium", "nogevo", "remcyk"},
    # "quilt": {"mksdlu", "prhems"},
    # "pennant": {"tfnwti"},
    # "pillowcase": {"dtoahb", "yakvci"},
    # "rubber_glove": {"leuiso"},
    # "scarf": {"kclcrj"},
    # "sock": {"vpafgj"},
    # "tank_top": {"fzldgi"},
}


class UnsampleablePredicate:
    def _sample(self, *args, **kwargs):
        raise NotImplementedError()


class ObjectStateInsourcePredicate(UnsampleablePredicate, BinaryAtomicFormula):
    def _evaluate(self, entity, **kwargs):
        # Always returns True
        return True


class ObjectStateFuturePredicate(UnsampleablePredicate, UnaryAtomicFormula):
    STATE_NAME = "future"

    def _evaluate(self, entity, **kwargs):
        return not entity.exists


class ObjectStateRealPredicate(UnsampleablePredicate, UnaryAtomicFormula):
    STATE_NAME = "real"

    def _evaluate(self, entity, **kwargs):
        return entity.exists


class ObjectStateUnaryPredicate(UnaryAtomicFormula):
    STATE_CLASS = None
    STATE_NAME = None

    def _evaluate(self, entity, **kwargs):
        return entity.get_state(self.STATE_CLASS, **kwargs)

    def _sample(self, entity, binary_state, **kwargs):
        return entity.set_state(self.STATE_CLASS, binary_state, **kwargs)


class ObjectStateBinaryPredicate(BinaryAtomicFormula):
    STATE_CLASS = None
    STATE_NAME = None

    def _evaluate(self, entity1, entity2, **kwargs):
        return (
            entity1.get_state(self.STATE_CLASS, entity2.wrapped_obj, **kwargs)
            if (entity2.exists and entity2.initialized)
            else False
        )

    def _sample(self, entity1, entity2, binary_state, **kwargs):
        return (
            entity1.set_state(self.STATE_CLASS, entity2.wrapped_obj, binary_state, **kwargs)
            if (entity2.exists and entity2.initialized)
            else None
        )


def get_unary_predicate_for_state(state_class, state_name):
    return type(
        state_class.__name__ + "StateUnaryPredicate",
        (ObjectStateUnaryPredicate,),
        {"STATE_CLASS": state_class, "STATE_NAME": state_name},
    )


def get_binary_predicate_for_state(state_class, state_name):
    return type(
        state_class.__name__ + "StateBinaryPredicate",
        (ObjectStateBinaryPredicate,),
        {"STATE_CLASS": state_class, "STATE_NAME": state_name},
    )


def is_substance_synset(synset):
    return "substance" in OBJECT_TAXONOMY.get_abilities(synset)


def get_system_name_by_synset(synset):
    system_names = OBJECT_TAXONOMY.get_subtree_substances(synset)
    assert len(system_names) == 1, f"Got zero or multiple systems for {synset}: {system_names}"
    return system_names[0]


def process_single_condition(condition):
    """
    Processes a single BDDL condition

    Args:
        condition (Condition): Condition to process

    Returns:
        2-tuple:
            - Expression: Condition's expression
            - bool: Whether this evaluated condition is positive or negative
    """
    if not isinstance(condition.children[0], Negation) and not isinstance(condition.children[0], AtomicFormula):
        log.debug(("Skipping over sampling of predicate that is not a negation or an atomic formula"))
        return None, None

    if isinstance(condition.children[0], Negation):
        condition = condition.children[0].children[0]
        positive = False
    else:
        condition = condition.children[0]
        positive = True

    return condition, positive


# TODO: Add remaining predicates.
SUPPORTED_PREDICATES = {
    "inside": get_binary_predicate_for_state(object_states.Inside, "inside"),
    "nextto": get_binary_predicate_for_state(object_states.NextTo, "nextto"),
    "ontop": get_binary_predicate_for_state(object_states.OnTop, "ontop"),
    "under": get_binary_predicate_for_state(object_states.Under, "under"),
    "touching": get_binary_predicate_for_state(object_states.Touching, "touching"),
    "covered": get_binary_predicate_for_state(object_states.Covered, "covered"),
    "contains": get_binary_predicate_for_state(object_states.Contains, "contains"),
    "saturated": get_binary_predicate_for_state(object_states.Saturated, "saturated"),
    "filled": get_binary_predicate_for_state(object_states.Filled, "filled"),
    "cooked": get_unary_predicate_for_state(object_states.Cooked, "cooked"),
    "burnt": get_unary_predicate_for_state(object_states.Burnt, "burnt"),
    "frozen": get_unary_predicate_for_state(object_states.Frozen, "frozen"),
    "hot": get_unary_predicate_for_state(object_states.Heated, "hot"),
    "open": get_unary_predicate_for_state(object_states.Open, "open"),
    "toggled_on": get_unary_predicate_for_state(object_states.ToggledOn, "toggled_on"),
    "on_fire": get_unary_predicate_for_state(object_states.OnFire, "on_fire"),
    "attached": get_binary_predicate_for_state(object_states.AttachedTo, "attached"),
    "overlaid": get_binary_predicate_for_state(object_states.Overlaid, "overlaid"),
    "folded": get_unary_predicate_for_state(object_states.Folded, "folded"),
    "unfolded": get_unary_predicate_for_state(object_states.Unfolded, "unfolded"),
    "draped": get_binary_predicate_for_state(object_states.Draped, "draped"),
    "future": ObjectStateFuturePredicate,
    "real": ObjectStateRealPredicate,
    "insource": ObjectStateInsourcePredicate,
}

KINEMATIC_STATES_BDDL = frozenset([state.__name__.lower() for state in _KINEMATIC_STATE_SET] + ["attached"])


# BEHAVIOR-related
OBJECT_TAXONOMY = ObjectTaxonomy()
BEHAVIOR_ACTIVITIES = sorted(os.listdir(os.path.join(os.path.dirname(bddl.__file__), "activity_definitions")))


def _populate_input_output_objects_systems(og_recipe, input_synsets, output_synsets):
    # Map input/output synsets into input/output objects and systems.
    for synsets, obj_key, system_key in zip(
        (input_synsets, output_synsets), ("input_objects", "output_objects"), ("input_systems", "output_systems")
    ):
        for synset, count in synsets.items():
            assert OBJECT_TAXONOMY.is_leaf(synset), f"Synset {synset} must be a leaf node in the taxonomy!"
            if is_substance_synset(synset):
                og_recipe[system_key].append(get_system_name_by_synset(synset))
            else:
                obj_categories = OBJECT_TAXONOMY.get_categories(synset)
                assert (
                    len(obj_categories) == 1
                ), f"Object synset {synset} must map to exactly one object category! Now: {obj_categories}."
                og_recipe[obj_key][obj_categories[0]] = count

    # Assert only one of output_objects or output_systems is not None
    assert (
        len(og_recipe["output_objects"]) == 0 or len(og_recipe["output_systems"]) == 0
    ), "Recipe can only generate output objects or output systems, but not both!"


def _populate_input_output_states(og_recipe, input_states, output_states):
    # Apply post-processing for input/output states if specified
    for synsets_to_states, states_key in zip((input_states, output_states), ("input_states", "output_states")):
        if synsets_to_states is None:
            continue
        for synsets, states in synsets_to_states.items():
            # For unary/binary states, synsets is a single synset or a comma-separated pair of synsets, respectively
            synset_split = synsets.split(",")
            if len(synset_split) == 1:
                first_synset = synset_split[0]
                second_synset = None
            else:
                first_synset, second_synset = synset_split

            # Assert the first synset is an object because the systems don't have any states.
            assert OBJECT_TAXONOMY.is_leaf(
                first_synset
            ), f"Input/output state synset {first_synset} must be a leaf node in the taxonomy!"
            assert not is_substance_synset(
                first_synset
            ), f"Input/output state synset {first_synset} must be applied to an object, not a substance!"
            obj_categories = OBJECT_TAXONOMY.get_categories(first_synset)
            assert (
                len(obj_categories) == 1
            ), f"Input/output state synset {first_synset} must map to exactly one object category! Now: {obj_categories}."
            first_obj_category = obj_categories[0]

            if second_synset is None:
                # Unary states for the first synset
                for state_type, state_value in states:
                    state_class = SUPPORTED_PREDICATES[state_type].STATE_CLASS
                    assert issubclass(
                        state_class, AbsoluteObjectState
                    ), f"Input/output state type {state_type} must be a unary state!"
                    # Example: (Cooked, True)
                    og_recipe[states_key][first_obj_category]["unary"].append((state_class, state_value))
            else:
                assert OBJECT_TAXONOMY.is_leaf(
                    second_synset
                ), f"Input/output state synset {second_synset} must be a leaf node in the taxonomy!"
                obj_categories = OBJECT_TAXONOMY.get_categories(second_synset)
                if is_substance_synset(second_synset):
                    second_obj_category = get_system_name_by_synset(second_synset)
                    is_substance = True
                else:
                    obj_categories = OBJECT_TAXONOMY.get_categories(second_synset)
                    assert (
                        len(obj_categories) == 1
                    ), f"Input/output state synset {second_synset} must map to exactly one object category! Now: {obj_categories}."
                    second_obj_category = obj_categories[0]
                    is_substance = False

                for state_type, state_value in states:
                    state_class = SUPPORTED_PREDICATES[state_type].STATE_CLASS
                    assert issubclass(
                        state_class, RelativeObjectState
                    ), f"Input/output state type {state_type} must be a binary state!"
                    assert is_substance == (
                        state_class in get_system_states()
                    ), f"Input/output state type {state_type} system state inconsistency found!"
                    if is_substance:
                        # Non-kinematic binary states, e.g. Covered, Saturated, Filled, Contains.
                        # Example: (Covered, "sesame_seed", True)
                        og_recipe[states_key][first_obj_category]["binary_system"].append(
                            (state_class, second_obj_category, state_value)
                        )
                    else:
                        # Kinematic binary states w.r.t. the second object.
                        # Example: (OnTop, "raw_egg", True)
                        assert (
                            states_key != "output_states"
                        ), f"Output state type {state_type} can only be used in input states!"
                        og_recipe[states_key][first_obj_category]["binary_object"].append(
                            (state_class, second_obj_category, state_value)
                        )


def _populate_filter_categories(og_recipe, filter_name, synsets):
    # Map synsets to categories.
    if synsets is not None:
        og_recipe[f"{filter_name}_categories"] = set()
        for synset in synsets:
            assert OBJECT_TAXONOMY.is_leaf(synset), f"Synset {synset} must be a leaf node in the taxonomy!"
            assert not is_substance_synset(synset), f"Synset {synset} must be applied to an object, not a substance!"
            for category in OBJECT_TAXONOMY.get_categories(synset):
                og_recipe[f"{filter_name}_categories"].add(category)


def translate_bddl_recipe_to_og_recipe(
    name,
    input_synsets,
    output_synsets,
    input_states=None,
    output_states=None,
    fillable_synsets=None,
    heatsource_synsets=None,
    timesteps=None,
):
    """
    Translate a BDDL recipe to an OG recipe.
    Args:
        name (str): Name of the recipe
        input_synsets (dict): Maps synsets to number of instances required for the recipe
        output_synsets (dict): Maps synsets to number of instances to be spawned in the container when the recipe executes
        input_states (dict or None): Maps input synsets to states that must be satisfied for the recipe to execute,
            or None if no states are required
        otuput_states (dict or None): Map output synsets to states that should be set when spawned when the recipe executes,
            or None if no states are required
        fillable_synsets (None or set of str): If specified, set of fillable synsets which are allowed for this recipe.
            If None, any fillable is allowed
        heatsource_synsets (None or set of str): If specified, set of heatsource synsets which are allowed for this recipe.
            If None, any heatsource is allowed
        timesteps (None or int): Number of subsequent heating steps required for the recipe to execute. If None,
            it will be set to be 1, i.e.: instantaneous execution
    """
    og_recipe = {
        "name": name,
        # Maps object categories to number of instances required for the recipe
        "input_objects": dict(),
        # List of system names required for the recipe
        "input_systems": list(),
        # Maps object categories to number of instances to be spawned in the container when the recipe executes
        "output_objects": dict(),
        # List of system names to be spawned in the container when the recipe executes. Currently the length is 1.
        "output_systems": list(),
        # Maps object categories to ["unary", "bianry_system", "binary_object"] to a list of states that must be satisfied for the recipe to execute
        "input_states": defaultdict(lambda: defaultdict(list)),
        # Maps object categories to ["unary", "bianry_system"] to a list of states that should be set after the output objects are spawned
        "output_states": defaultdict(lambda: defaultdict(list)),
        # Set of fillable categories which are allowed for this recipe
        "fillable_categories": None,
        # Set of heatsource categories which are allowed for this recipe
        "heatsource_categories": None,
        # Number of subsequent heating steps required for the recipe to execute
        "timesteps": timesteps if timesteps is not None else 1,
    }

    _populate_input_output_objects_systems(
        og_recipe=og_recipe, input_synsets=input_synsets, output_synsets=output_synsets
    )
    _populate_input_output_states(og_recipe=og_recipe, input_states=input_states, output_states=output_states)
    _populate_filter_categories(og_recipe=og_recipe, filter_name="fillable", synsets=fillable_synsets)
    _populate_filter_categories(og_recipe=og_recipe, filter_name="heatsource", synsets=heatsource_synsets)

    return og_recipe


def translate_bddl_washer_rule_to_og_washer_rule(conditions):
    """
    Translate BDDL washer rule to OG washer rule.

    Args:
        conditions (dict): Dictionary mapping the synset of ParticleSystem (str) to None or list of synsets of
            ParticleSystem (str). None represents "never", empty list represents "always", or non-empty list represents
            at least one of the systems in the list needs to be present in the washer for the key system to be removed.
            E.g. "rust.n.01" -> None: "never remove rust.n.01 from the washer"
            E.g. "dust.n.01" -> []: "always remove dust.n.01 from the washer"
            E.g. "cooking_oil.n.01" -> ["sodium_carbonate.n.01", "vinegar.n.01"]: "remove cooking_oil.n.01 from the
            washer if either sodium_carbonate.n.01 or vinegar.n.01 is present"
            For keys not present in the dictionary, the default is []: "always remove"
    Returns:
        dict: Dictionary mapping the system name (str) to None or list of system names (str). None represents "never",
            empty list represents "always", or non-empty list represents at least one of the systems in the list needs
            to be present in the washer for the key system to be removed.
    """
    og_washer_rule = dict()
    for solute, solvents in conditions.items():
        assert OBJECT_TAXONOMY.is_leaf(solute), f"Synset {solute} must be a leaf node in the taxonomy!"
        assert is_substance_synset(solute), f"Synset {solute} must be a substance synset!"
        solute_name = get_system_name_by_synset(solute)
        if solvents is None:
            og_washer_rule[solute_name] = None
        else:
            solvent_names = []
            for solvent in solvents:
                assert OBJECT_TAXONOMY.is_leaf(solvent), f"Synset {solvent} must be a leaf node in the taxonomy!"
                assert is_substance_synset(solvent), f"Synset {solvent} must be a substance synset!"
                solvent_name = get_system_name_by_synset(solvent)
                solvent_names.append(solvent_name)
            og_washer_rule[solute_name] = solvent_names
    return og_washer_rule


def get_processed_bddl(behavior_activity, activity_definition, scene):
    """
    Reads the raw bddl defined by @activity_name and @instance and converts it into OmniGibson-compatible
    bddl by expanding any wildcard specified scene objects into a concrete set of objects

    Args:
        behavior_activity (str): behavior activity being used
        activity_definition (int): specific definition of behavior_activity
        scene (Scene): Current active OmniGibson scene

    Returns:
        str: Post-processed BDDL string
    """
    # Load bddl string
    problem_filename = get_definition_filename(behavior_activity, activity_definition)
    with open(problem_filename, "r") as f:
        raw_bddl = f.readlines()

    # Manually parse BDDL to hot-swap wildcard scene objects
    swap_info = dict()
    in_goal = False
    start_init_idx = None
    end_init_idx = None
    for idx, line in enumerate(raw_bddl):
        if "*" in line:
            # Make sure we're not in the goal conditions -- we ONLY expect the wildcard to be
            # specified in either the object scope or init conditions
            assert (
                not in_goal
            ), "Found wildcard in BDDL goal conditions, but only expected in object_scope and init conditions!"

            # Infer whether this line is part of the object scope or goal conditions
            if "-" in line:
                # This is the object scope
                # Split by the delimiter, and then split by spaces
                instances, synset = line.strip(" \n\t").split(" - ")
                instances = instances.split(" ")

                # Synset should be a scene object instance
                abilities = OBJECT_TAXONOMY.get_abilities(synset)
                assert (
                    "sceneObject" in abilities
                ), f"Wildcard can only be used on sceneObject synsets, but got synset: {synset}"

                # Get all valid categories that are mapped to this synset
                og_categories = OBJECT_TAXONOMY.get_subtree_categories(synset)

                # Wildcard should be specified in the final instance
                wildcard_instance = instances[-1]
                assert (
                    "*" in wildcard_instance
                ), f"Expected wildcard to be specified in final instance in raw BDDL object scope line:\n{line}"

                # Make sure this hasn't been specified yet
                assert wildcard_instance not in swap_info, f"Already found wildcard previously for synset {synset}!"
                n_minimum_instances = len(instances) - 1

                # Add swap info
                swap_info[wildcard_instance] = {
                    "object_scope_idx": idx,
                    "n_minimum_instances": n_minimum_instances,
                    "categories": set(og_categories),
                    "synset": synset,
                }

            else:
                # This is the init condition
                # For now, we ONLY support inroom condition, so assert that this is the case
                tokens = line.strip(" ()\n\t").split(" ")
                assert len(tokens) == 3, f"Expected 3 total parsed tokens for wildcard init condition line:\n{line}"
                assert (
                    tokens[0] == "inroom"
                ), f"Only inroom is supported for wildcard init condition, but found: {tokens[0]}"
                _, wildcard_instance, room = tokens
                assert (
                    wildcard_instance in swap_info
                ), f"Expected wildcard instance {wildcard_instance} to already be specified in object_scope, but found none!"
                swap_info[wildcard_instance]["room"] = room
                swap_info[wildcard_instance]["init_cond_idx"] = idx

        elif ":init" in line:
            start_init_idx = idx
        elif ":goal" in line:
            end_init_idx = idx

    raw_bddl_init_cond_lines = deepcopy(raw_bddl[start_init_idx:end_init_idx])
    new_init_cond_lines = []
    for line in raw_bddl_init_cond_lines:
        if "*" in line:
            # parse line to get the space-delimited token that includes the star
            tokens = line.split(" ")
            wildcard_instance = None
            for token in tokens:
                if "*" in token:
                    wildcard_instance = token
                    break
            assert wildcard_instance is not None, f"Expected to find wildcard synset in line: {line}"
            # Search for swap info and hot swap in condition
            info = swap_info[wildcard_instance]
            # Make sure we have the minimum number of objects requested
            n_min_instances = info["n_minimum_instances"]
            synset = info["synset"]

            valid_objs = set()
            for category in info["categories"]:
                valid_objs = valid_objs.union(scene.object_registry("category", category, default_val=set()))
            # TODO: This is a temporary fix before we properly implement room instance handling for wildcards
            n_valid_objects = 0
            for i in range(11):  # Check room instances 0 through 10
                in_room_objs = scene.object_registry("in_rooms", f"{info['room']}_{i}")
                if in_room_objs is not None:
                    n_valid_objects = max(n_valid_objects, len(valid_objs.intersection(in_room_objs)))

            assert (
                n_valid_objects >= n_min_instances
            ), f"BDDL requires at least {n_min_instances} instances of synset {synset}, but only found {n_valid_objects} in rooms of type {info['room']}!"

            # Hot swap this information into the BDDL
            extra_instances = [f"{synset}_{i + 1}" for i in range(n_min_instances, n_valid_objects)]
            extra_instances_str = " ".join(extra_instances)
            obj_scope_idx = info["object_scope_idx"]
            init_cond_idx = info["init_cond_idx"]
            raw_bddl[obj_scope_idx] = raw_bddl[obj_scope_idx].replace(wildcard_instance, extra_instances_str)
            init_cond_line = raw_bddl[init_cond_idx]
            extra_cond_lines = [
                init_cond_line.replace(wildcard_instance, extra_instance) for extra_instance in extra_instances
            ]
            new_init_cond_lines = new_init_cond_lines + extra_cond_lines
        else:
            new_init_cond_lines.append(line)

    raw_bddl = raw_bddl[:start_init_idx] + new_init_cond_lines + raw_bddl[end_init_idx:]

    # Return the compiled processed BDDL as a single string
    return "".join(raw_bddl)


class OmniGibsonBDDLBackend(BDDLBackend):
    def get_predicate_class(self, predicate_name):
        return SUPPORTED_PREDICATES[predicate_name]


class BDDLEntity(Wrapper):
    """
    Thin wrapper class that wraps an object or system if it exists, or nothing if it does not exist. Will
    dynamically reference an object / system as they become real in the sim
    """

    def __init__(
        self,
        bddl_inst,
        entity=None,
    ):
        """
        Args:
            bddl_inst (str): BDDL synset instance of the entity, e.g.: "almond.n.01_1"
            entity (None or DatasetObject or BaseSystem): If specified, the BDDL entity to wrap. If not
                specified, will initially wrap nothing, but may dynamically reference an actual object or system
                if it exists in the future
        """
        # Store synset and other info, and pass entity internally
        self.bddl_inst = bddl_inst
        self.synset = "_".join(self.bddl_inst.split("_")[:-1])
        self.is_system = is_substance_synset(self.synset)

        # Infer the correct category to assign
        self.og_categories = (
            OBJECT_TAXONOMY.get_subtree_substances(self.synset)
            if self.is_system
            else OBJECT_TAXONOMY.get_subtree_categories(self.synset)
        )

        super().__init__(obj=entity)

    @property
    def name(self):
        """
        Returns:
            None or str: Name of this entity, if it exists, else None
        """
        if self.exists:
            return self.og_categories[0] if self.is_system else self.wrapped_obj.name
        else:
            return None

    @property
    def exists(self):
        """
        Checks whether the entity referenced by @synset exists

        Returns:
            bool: Whether the entity referenced by @synset exists
        """
        return self.wrapped_obj is not None

    def set_entity(self, entity):
        """
        Sets the internal entity, overriding any if it already exists

        Args:
            entity (BaseSystem or BaseObject): Entity to set internally
        """
        self.wrapped_obj = entity

    def clear_entity(self):
        """
        Clears the internal entity, if any
        """
        self.wrapped_obj = None

    def get_state(self, state, *args, **kwargs):
        """
        Helper function to grab wrapped entity's state @state

        Args:
            state (BaseObjectState): State whose get_value() should be called
            *args (tuple): Any arguments to pass to getter, in order
            **kwargs (dict): Any keyword arguments to pass to getter, in order

        Returns:
            any: Returned value(s) from @state if self.wrapped_obj exists (i.e.: not None), else False
        """
        return (
            self.wrapped_obj.states[state].get_value(*args, **kwargs) if (self.exists and self.initialized) else False
        )

    def set_state(self, state, *args, **kwargs):
        """
        Helper function to set wrapped entity's state @state. Note: Should only be called if the entity exists!

        Args:
            state (BaseObjectState): State whose set_value() should be called
            *args (tuple): Any arguments to pass to getter, in order
            **kwargs (dict): Any keyword arguments to pass to getter, in order

        Returns:
            any: Returned value(s) from @state if self.wrapped_obj exists (i.e.: not None)
        """
        assert (
            self.exists and self.initialized
        ), f"Cannot call set_state() for BDDLEntity {self.synset} when the entity does not exist or is not initialized!"
        return self.wrapped_obj.states[state].set_value(*args, **kwargs)


class BDDLSampler:
    def __init__(self, env, activity_conditions, object_scope, backend):
        # Avoid circular imports here
        from omnigibson.scenes.traversable_scene import TraversableScene

        # Store internal variables from inputs
        self._env = env
        self._scene_model = self._env.scene.scene_model if isinstance(self._env.scene, TraversableScene) else None
        self._agent = self._env.robots[0]
        self._backend = backend
        self._activity_conditions = activity_conditions
        self._object_scope = object_scope
        self._object_instance_to_synset = {
            obj_inst: obj_cat
            for obj_cat in self._activity_conditions.parsed_objects
            for obj_inst in self._activity_conditions.parsed_objects[obj_cat]
        }
        self._substance_instances = {
            obj_inst
            for obj_inst in self._object_scope.keys()
            if is_substance_synset(self._object_instance_to_synset[obj_inst])
        }

        # Initialize other variables that will be filled in later
        self._sampling_whitelist = None  # Maps str to str to list
        self._sampling_blacklist = None  # Maps str to str to list
        self._room_type_to_object_instance = None  # dict
        self._inroom_object_instances = None  # set of str
        self._object_sampling_orders = None  # dict mapping str to list of str
        self._sampled_objects = None  # set of BaseObject
        self._future_obj_instances = None  # set of str
        self._inroom_object_conditions = None  # list of (condition, positive) tuple
        self._inroom_object_scope_filtered_initial = None  # dict mapping str to BDDLEntity
        self._attached_objects = defaultdict(set)  # dict mapping str to set of str

    def sample(self, validate_goal=False, sampling_whitelist=None, sampling_blacklist=None):
        """
        Run sampling for this BEHAVIOR task

        Args:
            validate_goal (bool): Whether the goal should be validated or not
            sampling_whitelist (None or dict): If specified, should map synset name (e.g.: "table.n.01" to a dictionary
                mapping category name (e.g.: "breakfast_table") to a list of valid models to be sampled from
                that category. During sampling, if a given synset is found in this whitelist, only the specified
                models will be used as options
            sampling_blacklist (None or dict): If specified, should map synset name (e.g.: "table.n.01" to a dictionary
                mapping category name (e.g.: "breakfast_table") to a list of invalid models that should not be sampled from
                that category. During sampling, if a given synset is found in this blacklist, all specified
                models will not be used as options

        Returns:
            2-tuple:
                - bool: Whether sampling was successful or not
                - None or str: None if successful, otherwise the associated error message
        """
        log.info("Sampling task...")
        # Store sampling white / blacklists
        self._sampling_whitelist = sampling_whitelist
        self._sampling_blacklist = sampling_blacklist

        # Reject scenes with missing non-sampleable objects
        # Populate object_scope with sampleable objects and the robot
        accept_scene, feedback = self._prepare_scene_for_sampling()
        if not accept_scene:
            return accept_scene, feedback
        # Sample objects to satisfy initial conditions
        accept_scene, feedback = self._sample_all_conditions(validate_goal=validate_goal)
        if not accept_scene:
            return accept_scene, feedback

        log.info("Sampling succeeded!")

        return True, None

    def _sample_all_conditions(self, validate_goal=False):
        """
        Run sampling for this BEHAVIOR task

        Args:
            validate_goal (bool): Whether the goal should be validated or not

        Returns:
            2-tuple:
                - bool: Whether sampling was successful or not
                - None or str: None if successful, otherwise the associated error message
        """
        # Auto-initialize all sampleable objects
        with og.sim.playing():
            # Update the scene to include the latest robots / objects
            self._env.scene.update_initial_file()
            self._env.scene.reset()

            error_msg = self._sample_initial_conditions()
            if error_msg:
                log.error(error_msg)
                return False, error_msg

            if validate_goal:
                error_msg = self._sample_goal_conditions()
                if error_msg:
                    log.error(error_msg)
                    return False, error_msg

            error_msg = self._sample_initial_conditions_final()
            if error_msg:
                log.error(error_msg)
                return False, error_msg

            self._env.scene.update_initial_file()

        return True, None

    def _prepare_scene_for_sampling(self):
        """
        Runs sanity checks for the current scene for the given BEHAVIOR task

        Returns:
            2-tuple:
                - bool: Whether the generated scene activity should be accepted or not
                - dict: Any feedback from the sampling / initialization process
        """
        error_msg = self._parse_inroom_object_room_assignment()
        if error_msg:
            log.error(error_msg)
            return False, error_msg

        error_msg = self._parse_attached_states()
        if error_msg:
            log.error(error_msg)
            return False, error_msg

        error_msg = self._build_sampling_order()
        if error_msg:
            log.error(error_msg)
            return False, error_msg

        error_msg = self._build_inroom_object_scope()
        if error_msg:
            log.error(error_msg)
            return False, error_msg

        error_msg = self._import_sampleable_objects()
        if error_msg:
            log.error(error_msg)
            return False, error_msg

        self._object_scope["agent.n.01_1"] = BDDLEntity(bddl_inst="agent.n.01_1", entity=self._agent)

        return True, None

    def _parse_inroom_object_room_assignment(self):
        """
        Infers which rooms each object is assigned to
        """
        self._room_type_to_object_instance = dict()
        self._inroom_object_instances = set()
        for cond in self._activity_conditions.parsed_initial_conditions:
            if cond[0] == "inroom":
                obj_inst, room_type = cond[1], cond[2]
                obj_synset = self._object_instance_to_synset[obj_inst]
                abilities = OBJECT_TAXONOMY.get_abilities(obj_synset)
                if "sceneObject" not in abilities:
                    # Invalid room assignment
                    return (
                        f"You have assigned room type for [{obj_synset}], but [{obj_synset}] is sampleable. "
                        f"Only non-sampleable (scene) objects can have room assignment."
                    )
                if self._scene_model is not None and room_type not in self._env.scene.seg_map.room_sem_name_to_ins_name:
                    # Missing room type
                    return f"Room type [{room_type}] missing in scene [{self._scene_model}]."
                if room_type not in self._room_type_to_object_instance:
                    self._room_type_to_object_instance[room_type] = []
                self._room_type_to_object_instance[room_type].append(obj_inst)

                if obj_inst in self._inroom_object_instances:
                    # Duplicate room assignment
                    return f"Object [{obj_inst}] has more than one room assignment"

                self._inroom_object_instances.add(obj_inst)

    def _parse_attached_states(self):
        """
        Infers which objects are attached to which other objects.
        If a category-level attachment is specified, it will be expanded to all instances of that category.
        E.g. if the goal condition requires corks to be attached to bottles, every cork needs to be able to
        attach to every bottle.
        """
        for cond in self._activity_conditions.parsed_initial_conditions:
            if cond[0] == "attached":
                obj_inst, parent_inst = cond[1], cond[2]
                if obj_inst not in self._object_scope or parent_inst not in self._object_scope:
                    return f"Object [{obj_inst}] or parent [{parent_inst}] in attached initial condition not found in object scope"
                self._attached_objects[obj_inst].add(parent_inst)

        ground_attached_conditions = []
        conditions_to_check = self._activity_conditions.parsed_goal_conditions.copy()
        while conditions_to_check:
            new_conditions_to_check = []
            for cond in conditions_to_check:
                if cond[0] == "attached":
                    ground_attached_conditions.append(cond)
                else:
                    new_conditions_to_check.extend([ele for ele in cond if isinstance(ele, list)])
            conditions_to_check = new_conditions_to_check

        for cond in ground_attached_conditions:
            obj_inst, parent_inst = cond[1].lstrip("?"), cond[2].lstrip("?")
            if obj_inst in self._object_scope:
                obj_insts = [obj_inst]
            elif obj_inst in self._activity_conditions.parsed_objects:
                obj_insts = self._activity_conditions.parsed_objects[obj_inst]
            else:
                return f"Object [{obj_inst}] in attached goal condition not found in object scope or parsed objects"

            if parent_inst in self._object_scope:
                parent_insts = [parent_inst]
            elif parent_inst in self._activity_conditions.parsed_objects:
                parent_insts = self._activity_conditions.parsed_objects[parent_inst]
            else:
                return f"Parent [{parent_inst}] in attached goal condition not found in object scope or parsed objects"

            for obj_inst in obj_insts:
                for parent_inst in parent_insts:
                    self._attached_objects[obj_inst].add(parent_inst)

    def _build_sampling_order(self):
        """
        Sampling orders is a list of lists: [[batch_1_inst_1, ... batch_1_inst_N], [batch_2_inst_1, batch_2_inst_M], ...]
        Sampling should happen for batch 1 first, then batch 2, so on and so forth
        Example: OnTop(plate, table) should belong to batch 1, and OnTop(apple, plate) should belong to batch 2
        """
        unsampleable_conditions = []
        sampling_groups = {group: [] for group in ("kinematic", "particle", "unary")}
        self._object_sampling_conditions = {group: [] for group in ("kinematic", "particle", "unary")}
        self._object_sampling_orders = {group: [] for group in ("kinematic", "particle", "unary")}
        self._inroom_object_conditions = []

        # First, sort initial conditions into kinematic, particle and unary groups
        # bddl.condition_evaluation.HEAD, each with one child.
        # This child is either a ObjectStateUnaryPredicate/ObjectStateBinaryPredicate or
        # a Negation of a ObjectStateUnaryPredicate/ObjectStateBinaryPredicate
        for condition in get_initial_conditions(self._activity_conditions, self._backend, self._object_scope):
            condition, positive = process_single_condition(condition)
            if condition is None:
                continue

            # Sampled conditions must always be positive
            # Non-positive (e.g.: NOT onTop) is not restrictive enough for sampling
            if condition.STATE_NAME in KINEMATIC_STATES_BDDL and not positive:
                return "Initial condition has negative kinematic conditions: {}".format(condition.body)

            # Store any unsampleable conditions separately
            if isinstance(condition, UnsampleablePredicate):
                unsampleable_conditions.append(condition)
                continue

            # Infer the group the condition and its object instances belong to
            # (a) Kinematic (binary) conditions, where (ent0, ent1) are both objects
            # (b) Particle (binary) conditions, where (ent0, ent1) are (object, substance)
            # (d) Unary conditions, where (ent0,) is an object
            # Binary conditions have length 2: (ent0, ent1)
            if len(condition.body) == 2:
                group = "particle" if condition.body[1] in self._substance_instances else "kinematic"
            else:
                assert len(condition.body) == 1, (
                    f"Got invalid parsed initial condition; body length should either be 2 or 1. "
                    f"Got body: {condition.body} for condition: {condition}"
                )
                group = "unary"
            sampling_groups[group].append(condition.body)
            self._object_sampling_conditions[group].append((condition, positive))

            # If the condition involves any non-sampleable object (e.g.: furniture), it's a non-sampleable condition
            # This means that there's no ordering constraint in terms of sampling, because we know the, e.g., furniture
            # object already exists in the scene and is placed, so these specific conditions can be sampled without
            # any dependencies
            if len(self._inroom_object_instances.intersection(set(condition.body))) > 0:
                self._inroom_object_conditions.append((condition, positive))

        # Now, sort each group, ignoring the futures (since they don't get sampled)
        # First handle kinematics, then particles, then unary

        # Start with the non-sampleable objects as the first sampled set, then infer recursively
        cur_batch = self._inroom_object_instances
        while len(cur_batch) > 0:
            next_batch = set()
            for cur_batch_inst in cur_batch:
                inst_batch = set()
                for condition, _ in self._object_sampling_conditions["kinematic"]:
                    if condition.body[1] == cur_batch_inst:
                        inst_batch.add(condition.body[0])
                        next_batch.add(condition.body[0])
                if len(inst_batch) > 0:
                    self._object_sampling_orders["kinematic"].append(inst_batch)
            cur_batch = next_batch

        # Now parse particles -- simply unordered, since particle systems shouldn't impact each other
        self._object_sampling_orders["particle"].append({cond[0] for cond in sampling_groups["particle"]})
        sampled_particle_entities = {cond[1] for cond in sampling_groups["particle"]}

        # Finally, parse unaries -- this is simply unordered, since it is assumed that unary predicates do not
        # affect each other
        self._object_sampling_orders["unary"].append({cond[0] for cond in sampling_groups["unary"]})

        # Aggregate future objects and any unsampleable obj instances
        # Unsampleable obj instances are strictly a superset of future obj instances
        unsampleable_obj_instances = {cond.body[-1] for cond in unsampleable_conditions}
        self._future_obj_instances = {
            cond.body[0] for cond in unsampleable_conditions if isinstance(cond, ObjectStateFuturePredicate)
        }

        nonparticle_entities = set(self._object_scope.keys()) - self._substance_instances

        # Sanity check kinematic objects -- any non-system must be kinematically sampled
        remaining_kinematic_entities = (
            nonparticle_entities
            - unsampleable_obj_instances
            - self._inroom_object_instances
            - set.union(*(self._object_sampling_orders["kinematic"] + [set()]))
        )

        # Possibly remove the agent entity if we're in an empty scene -- i.e.: no kinematic sampling needed for the
        # agent
        if self._scene_model is None:
            remaining_kinematic_entities -= {"agent.n.01_1"}

        if len(remaining_kinematic_entities) != 0:
            return (
                f"Some objects do not have any kinematic condition defined for them in the initial conditions: "
                f"{', '.join(remaining_kinematic_entities)}"
            )

        # Sanity check particle systems -- any non-future system must be sampled as part of particle groups
        remaining_particle_entities = self._substance_instances - unsampleable_obj_instances - sampled_particle_entities
        if len(remaining_particle_entities) != 0:
            return (
                f"Some systems do not have any particle condition defined for them in the initial conditions: "
                f"{', '.join(remaining_particle_entities)}"
            )

    def _build_inroom_object_scope(self):
        """
        Store simulator object options for non-sampleable objects in self.inroom_object_scope
        {
            "living_room": {
                "table1": {
                    "living_room_0": [URDFObject, URDFObject, URDFObject],
                    "living_room_1": [URDFObject]
                },
                "table2": {
                    "living_room_0": [URDFObject, URDFObject],
                    "living_room_1": [URDFObject, URDFObject]
                },
                "chair1": {
                    "living_room_0": [URDFObject],
                    "living_room_1": [URDFObject]
                },
            }
        }
        """
        room_type_to_scene_objs = {}
        for room_type in self._room_type_to_object_instance:
            room_type_to_scene_objs[room_type] = {}
            for obj_inst in self._room_type_to_object_instance[room_type]:
                room_type_to_scene_objs[room_type][obj_inst] = {}
                obj_synset = self._object_instance_to_synset[obj_inst]
                synset_whitelist = (
                    None if self._sampling_whitelist is None else self._sampling_whitelist.get(obj_synset, None)
                )
                synset_blacklist = (
                    None if self._sampling_blacklist is None else self._sampling_blacklist.get(obj_synset, None)
                )

                # We allow burners to be used as if they are stoves
                # No need to safeguard check for subtree_substances because inroom objects will never be substances
                categories = OBJECT_TAXONOMY.get_subtree_categories(obj_synset)

                # Grab all models that fully support all abilities for the corresponding category
                valid_models = {
                    cat: set(
                        get_all_object_category_models_with_abilities(
                            cat, OBJECT_TAXONOMY.get_abilities(OBJECT_TAXONOMY.get_synset_from_category(cat))
                        )
                    )
                    for cat in categories
                }
                valid_models = {
                    cat: (models if cat not in GOOD_MODELS else models.intersection(GOOD_MODELS[cat]))
                    - BAD_CLOTH_MODELS.get(cat, set())
                    for cat, models in valid_models.items()
                }
                valid_models = {
                    cat: self._filter_model_choices_by_attached_states(models, cat, obj_inst)
                    for cat, models in valid_models.items()
                }

                # Filter based on white / blacklist
                if synset_whitelist is not None:
                    valid_models = {
                        cat: models.intersection(set(synset_whitelist[cat].keys()))
                        if cat in synset_whitelist
                        else set()
                        for cat, models in valid_models.items()
                    }

                if synset_blacklist is not None:
                    valid_models = {
                        cat: models - set(synset_blacklist[cat].keys()) if cat in synset_blacklist else models
                        for cat, models in valid_models.items()
                    }

                room_insts = (
                    [None]
                    if self._scene_model is None
                    else self._env.scene.seg_map.room_sem_name_to_ins_name[room_type]
                )
                for room_inst in room_insts:
                    # A list of scene objects that satisfy the requested categories
                    room_objs = self._env.scene.object_registry("in_rooms", room_inst, default_val=[])
                    scene_objs = [
                        obj
                        for obj in room_objs
                        if obj.category in categories and obj.model in valid_models[obj.category]
                    ]

                    if len(scene_objs) != 0:
                        room_type_to_scene_objs[room_type][obj_inst][room_inst] = scene_objs

        error_msg = self._consolidate_room_instance(room_type_to_scene_objs, "initial_pre-sampling")
        if error_msg:
            return error_msg
        self._inroom_object_scope = room_type_to_scene_objs

    def _filter_object_scope(self, input_object_scope, conditions, condition_type):
        """
        Filters the object scope based on given @input_object_scope, @conditions, and @condition_type

        Args:
            input_object_scope (dict):
            conditions (list): List of conditions to filter scope with, where each list entry is
                a tuple of (condition, positive), where @positive is True if the condition has a positive
                evaluation.
            condition_type (str): What type of condition to sample, e.g., "initial"

        Returns:
            2-tuple:

                - dict: Filtered object scope
                - list of str: The name of children object(s) that have the highest proportion of kinematic sampling
                    failures
        """
        filtered_object_scope = {}
        # Maps child obj name (SCOPE name) to parent obj name (OBJECT name) to T / F,
        # ie: if the kinematic relationship was sampled successfully
        problematic_objs = defaultdict(dict)
        for room_type in input_object_scope:
            filtered_object_scope[room_type] = {}
            for scene_obj in input_object_scope[room_type]:
                filtered_object_scope[room_type][scene_obj] = {}
                for room_inst in input_object_scope[room_type][scene_obj]:
                    # These are a list of candidate simulator objects that need sampling test
                    for obj in input_object_scope[room_type][scene_obj][room_inst]:
                        # Temporarily set object_scope to point to this candidate object
                        self._object_scope[scene_obj] = BDDLEntity(bddl_inst=scene_obj, entity=obj)

                        success = True
                        # If this candidate object is not involved in any conditions,
                        # success will be True by default and this object will qualify
                        parent_obj_name = obj.name
                        conditions_to_sample = []
                        for condition, positive in conditions:
                            # Sample positive kinematic conditions that involve this candidate object
                            if (
                                condition.STATE_NAME in KINEMATIC_STATES_BDDL
                                and positive
                                and scene_obj in condition.body
                            ):
                                child_scope_name = condition.body[0]
                                entity = self._object_scope[child_scope_name]
                                conditions_to_sample.append((condition, positive, entity, child_scope_name))

                        # If we're sampling kinematics, sort children based on (a) whether they are cloth or not, and
                        # then (b) their AABB, so that first all rigid objects are sampled before all cloth objects,
                        # and within each group the larger objects are sampled first. This is needed because rigid
                        # objects currently don't detect collisions with cloth objects (rigid_obj.states[ContactBodies]
                        # is empty even when a cloth object is in contact with it).
                        rigid_conditions = [c for c in conditions_to_sample if c[2].prim_type != PrimType.CLOTH]
                        cloth_conditions = [c for c in conditions_to_sample if c[2].prim_type == PrimType.CLOTH]
                        conditions_to_sample = list(
                            reversed(sorted(rigid_conditions, key=lambda x: th.prod(x[2].aabb_extent)))
                        ) + list(reversed(sorted(cloth_conditions, key=lambda x: th.prod(x[2].aabb_extent))))

                        # Sample!
                        for condition, positive, entity, child_scope_name in conditions_to_sample:
                            kwargs = dict()
                            # Reset if we're sampling a kinematic state
                            if condition.STATE_NAME in {"inside", "ontop", "under"}:
                                kwargs["reset_before_sampling"] = True
                            elif condition.STATE_NAME in {"attached"}:
                                kwargs["bypass_alignment_checking"] = True
                                kwargs["check_physics_stability"] = True
                                kwargs["can_joint_break"] = False
                            success = condition.sample(binary_state=positive, **kwargs)
                            log_msg = " ".join(
                                [
                                    f"{condition_type} kinematic condition sampling",
                                    room_type,
                                    scene_obj,
                                    str(room_inst),
                                    parent_obj_name,
                                    condition.STATE_NAME,
                                    str(condition.body),
                                    str(success),
                                ]
                            )
                            log.warning(log_msg)

                            # Record the result for the child object
                            assert (
                                parent_obj_name not in problematic_objs[child_scope_name]
                            ), f"Multiple kinematic relationships attempted for pair {condition.body}"
                            problematic_objs[child_scope_name][parent_obj_name] = success
                            # If any condition fails for this candidate object, skip
                            if not success:
                                break

                        # If this candidate object fails, move on to the next candidate object
                        if not success:
                            continue

                        if room_inst not in filtered_object_scope[room_type][scene_obj]:
                            filtered_object_scope[room_type][scene_obj][room_inst] = []
                        filtered_object_scope[room_type][scene_obj][room_inst].append(obj)

        # Compute most problematic objects
        if len(problematic_objs) == 0:
            max_problematic_objs = []
        else:
            problematic_objs_by_proportion = defaultdict(list)
            for child_scope_name, parent_obj_names in problematic_objs.items():
                problematic_objs_by_proportion[
                    th.mean(th.tensor(list(parent_obj_names.values()), dtype=th.float32)).item()
                ].append(child_scope_name)
            max_problematic_objs = problematic_objs_by_proportion[min(problematic_objs_by_proportion.keys())]

        return filtered_object_scope, max_problematic_objs

    def _consolidate_room_instance(self, filtered_object_scope, condition_type):
        """
        Consolidates room instances

        Args:
            filtered_object_scope (dict): Filtered object scope
            condition_type (str): What type of condition to sample, e.g., "initial"

        Returns:
            None or str: Error message, if any
        """
        for room_type in filtered_object_scope:
            # For each room_type, filter in room_inst that has successful
            # sampling options for all obj_inst in this room_type
            room_inst_satisfied = set.intersection(
                *[
                    set(filtered_object_scope[room_type][obj_inst].keys())
                    for obj_inst in filtered_object_scope[room_type]
                ]
            )

            if len(room_inst_satisfied) == 0:
                error_msg = "{}: Room type [{}] of scene [{}] do not contain or cannot sample all the objects needed.\nThe following are the possible room instances for each object, the intersection of which is an empty set.\n".format(
                    condition_type, room_type, self._scene_model
                )
                for obj_inst in filtered_object_scope[room_type]:
                    error_msg += (
                        "{}: ".format(obj_inst) + ", ".join(filtered_object_scope[room_type][obj_inst].keys()) + "\n"
                    )

                return error_msg

            for obj_inst in filtered_object_scope[room_type]:
                filtered_object_scope[room_type][obj_inst] = {
                    key: val
                    for key, val in filtered_object_scope[room_type][obj_inst].items()
                    if key in room_inst_satisfied
                }

    def _filter_model_choices_by_attached_states(self, model_choices, category, obj_inst):
        # If obj_inst is a child object that depends on a parent object that has been imported or exists in the scene,
        # we filter in only models that match the parent object's attachment meta links.
        if obj_inst in self._attached_objects:
            parent_insts = self._attached_objects[obj_inst]
            parent_objects = []
            for parent_inst in parent_insts:
                # If parent_inst is not an inroom object, it must be a non-sampleable object that has already been imported.
                # Grab it from the object_scope
                if parent_inst not in self._inroom_object_instances:
                    assert self._object_scope[parent_inst] is not None
                    parent_objects.append([self._object_scope[parent_inst].wrapped_obj])
                # If parent_inst is an inroom object, it can refer to multiple objects in the scene in different rooms.
                # We gather all of them and require that the model choice supports attachment to at least one of them.
                else:
                    for _, parent_inst_to_parent_objs in self._inroom_object_scope.items():
                        if parent_inst in parent_inst_to_parent_objs:
                            parent_objects.append(sum(parent_inst_to_parent_objs[parent_inst].values(), []))

            # Help function to check if a child object can attach to a parent object
            def can_attach(child_attachment_links, parent_attachment_links):
                for child_link_name in child_attachment_links:
                    child_category = re.search(r"attachment_(.+?)_\d+_link$", child_link_name).group(1)
                    if child_category.endswith("F"):
                        continue
                    assert child_category.endswith("M")
                    target_parent_category = child_category[:-1] + "F"
                    for parent_link_name in parent_attachment_links:
                        parent_category = re.search(r"attachment_(.+?)_\d+_link$", parent_link_name).group(1)
                        if parent_category == target_parent_category:
                            return True
                return False

            # Filter out models that don't support the attached states
            new_model_choices = set()
            for model_choice in model_choices:
                child_attachment_links = get_attachment_meta_links(category, model_choice)
                # The child model choice needs to be able to attach to all parent instances.
                # For in-room parent instances, there might be multiple parent objects (e.g. different wall nails),
                # and the child object needs to be able to attach to at least one of them.
                if all(
                    any(
                        can_attach(
                            child_attachment_links, get_attachment_meta_links(parent_obj.category, parent_obj.model)
                        )
                        for parent_obj in parent_objs_per_inst
                    )
                    for parent_objs_per_inst in parent_objects
                ):
                    new_model_choices.add(model_choice)

            return new_model_choices

        # If obj_inst is a prent object that other objects depend on, we filter in only models that have at least some
        # attachment links.
        elif any(obj_inst in parents for parents in self._attached_objects.values()):
            # Filter out models that don't support the attached states
            new_model_choices = set()
            for model_choice in model_choices:
                if len(get_attachment_meta_links(category, model_choice)) > 0:
                    new_model_choices.add(model_choice)
            return new_model_choices

        # If neither of the above cases apply, we don't need to filter the model choices
        else:
            return model_choices

    def _import_sampleable_objects(self):
        """
        Import all objects that can be sampled

        Args:
            env (Environment): Current active environment instance
        """
        assert og.sim.is_stopped(), "Simulator should be stopped when importing sampleable objects"

        # Move the robot object frame to a far away location, similar to other newly imported objects below
        self._agent.set_position_orientation(
            position=th.tensor([300, 300, 300], dtype=th.float32), orientation=th.tensor([0, 0, 0, 1], dtype=th.float32)
        )

        self._sampled_objects = set()
        num_new_obj = 0
        # Only populate self.object_scope for sampleable objects
        available_categories = set(get_all_object_categories())

        # Attached states introduce dependencies among objects during import time.
        # For example, when importing a child object instance, we need to make sure the imported model can be attached
        # to the parent object instance. We sort the object instances such that parent object instances are imported
        # before child object instances.
        dependencies = {key: self._attached_objects.get(key, {}) for key in self._object_instance_to_synset.keys()}
        for obj_inst in list(reversed(list(nx.algorithms.topological_sort(nx.DiGraph(dependencies))))):
            obj_synset = self._object_instance_to_synset[obj_inst]
            synset_whitelist = (
                None if self._sampling_whitelist is None else self._sampling_whitelist.get(obj_synset, None)
            )
            synset_blacklist = (
                None if self._sampling_blacklist is None else self._sampling_blacklist.get(obj_synset, None)
            )

            # Don't populate agent
            if obj_synset == "agent.n.01":
                continue

            # Populate based on whether it's a substance or not
            if is_substance_synset(obj_synset):
                assert len(self._activity_conditions.parsed_objects[obj_synset]) == 1, "Systems are singletons"
                obj_inst = self._activity_conditions.parsed_objects[obj_synset][0]
                system_name = OBJECT_TAXONOMY.get_subtree_substances(obj_synset)[0]
                self._object_scope[obj_inst] = BDDLEntity(
                    bddl_inst=obj_inst,
                    entity=(
                        None if obj_inst in self._future_obj_instances else self._env.scene.get_system(system_name)
                    ),
                )
            else:
                valid_categories = set(OBJECT_TAXONOMY.get_subtree_categories(obj_synset))
                categories = list(valid_categories.intersection(available_categories))
                if len(categories) == 0:
                    return (
                        f"None of the following categories could be found in the dataset for synset {obj_synset}: "
                        f"{valid_categories}"
                    )

                # Don't explicitly sample if future
                if obj_inst in self._future_obj_instances:
                    self._object_scope[obj_inst] = BDDLEntity(bddl_inst=obj_inst)
                    continue
                # Don't sample if already in room
                if obj_inst in self._inroom_object_instances:
                    continue

                # Shuffle categories and sample to find a valid model
                random.shuffle(categories)
                model_choices = set()
                for category in categories:
                    # Get all available models that support all of its synset abilities
                    model_choices = set(
                        get_all_object_category_models_with_abilities(
                            category=category,
                            abilities=OBJECT_TAXONOMY.get_abilities(OBJECT_TAXONOMY.get_synset_from_category(category)),
                        )
                    )
                    model_choices = (
                        model_choices
                        if category not in GOOD_MODELS
                        else model_choices.intersection(GOOD_MODELS[category])
                    )
                    model_choices -= BAD_CLOTH_MODELS.get(category, set())
                    model_choices = self._filter_model_choices_by_attached_states(model_choices, category, obj_inst)

                    # Filter based on white / blacklist
                    if synset_whitelist is not None:
                        model_choices = (
                            model_choices.intersection(set(synset_whitelist[category].keys()))
                            if category in synset_whitelist
                            else set()
                        )

                    if synset_blacklist is not None:
                        model_choices = (
                            model_choices - set(synset_blacklist[category].keys())
                            if category in synset_blacklist
                            else model_choices
                        )

                    # Filter by category
                    if len(model_choices) > 0:
                        break

                if len(model_choices) == 0:
                    # We failed to find ANY valid model across ALL valid categories
                    return f"Missing valid object models for all categories: {categories}"

                # Randomly select an object model
                model = random.choice(list(model_choices))

                # Potentially add additional kwargs
                obj_kwargs = dict()

                size = synset_whitelist.get(category, dict()).get(model, None) if synset_whitelist else None

                # if size is one dimension, this is a scale; if it's three dimensions, this is a bounding box
                if size is not None:
                    if isinstance(size, (int, float)):
                        obj_kwargs["scale"] = th.tensor(size, dtype=th.float32)
                    elif isinstance(size, list) and len(size) == 3:
                        obj_kwargs["bounding_box"] = th.tensor(size, dtype=th.float32)
                    else:
                        return f"Invalid size for object {obj_inst} with model {model} in category {category}: {size}"

                # create the object
                simulator_obj = DatasetObject(
                    name=f"{category}_{len(self._env.scene.objects)}",
                    category=category,
                    model=model,
                    prim_type=(
                        PrimType.CLOTH if "cloth" in OBJECT_TAXONOMY.get_abilities(obj_synset) else PrimType.RIGID
                    ),
                    **obj_kwargs,
                )
                num_new_obj += 1

                # Load the object into the simulator
                self._env.scene.add_object(simulator_obj)

                # Set these objects to be far-away locations
                simulator_obj.set_position_orientation(
                    position=th.tensor([100.0, 100.0, -100.0]) + th.ones(3) * num_new_obj * 5.0
                )

                self._sampled_objects.add(simulator_obj)
                self._object_scope[obj_inst] = BDDLEntity(bddl_inst=obj_inst, entity=simulator_obj)

        og.sim.play()
        og.sim.stop()

    def _sample_initial_conditions(self):
        """
        Sample initial conditions

        Returns:
            None or str: If successful, returns None. Otherwise, returns an error message
        """
        error_msg, self._inroom_object_scope_filtered_initial = self._sample_conditions(
            self._inroom_object_scope, self._inroom_object_conditions, "initial"
        )
        return error_msg

    def _sample_goal_conditions(self):
        """
        Sample goal conditions

        Returns:
            None or str: If successful, returns None. Otherwise, returns an error message
        """
        activity_goal_conditions = get_goal_conditions(self._activity_conditions, self._backend, self._object_scope)
        ground_goal_state_options = get_ground_goal_state_options(
            self._activity_conditions, self._backend, self._object_scope, activity_goal_conditions
        )
        num_options = ground_goal_state_options.size(0)
        ground_goal_state_options = ground_goal_state_options[random.sample(range(num_options), num_options)]
        log.debug(("number of ground_goal_state_options", len(ground_goal_state_options)))
        num_goal_condition_set_to_test = 10

        goal_condition_success = False
        # Try to fulfill different set of ground goal conditions (maximum num_goal_condition_set_to_test)
        for goal_condition_set in ground_goal_state_options[:num_goal_condition_set_to_test]:
            goal_condition_processed = []
            for condition in goal_condition_set:
                condition, positive = process_single_condition(condition)
                if condition is None:
                    continue
                goal_condition_processed.append((condition, positive))

            error_msg, _ = self._sample_conditions(
                self._inroom_object_scope_filtered_initial, goal_condition_processed, "goal"
            )
            if not error_msg:
                # if one set of goal conditions (and initial conditions) are satisfied, sampling is successful
                goal_condition_success = True
                break

        if not goal_condition_success:
            return error_msg

    def _sample_initial_conditions_final(self, dynamic_scale=False):
        """
        Sample final initial conditions

        Returns:
            None or str: If successful, returns None. Otherwise, returns an error message
        """
        # Sample kinematics first, then particle states, then unary states
        state = og.sim.dump_state(serialized=False)
        for group in ("kinematic", "particle", "unary"):
            log.info(f"Sampling {group} states...")
            if len(self._object_sampling_orders[group]) > 0:
                for cur_batch in self._object_sampling_orders[group]:
                    conditions_to_sample = []
                    for condition, positive in self._object_sampling_conditions[group]:
                        # Sample conditions that involve the current batch of objects
                        child_scope_name = condition.body[0]
                        if child_scope_name in cur_batch:
                            entity = self._object_scope[child_scope_name]
                            conditions_to_sample.append((condition, positive, entity, child_scope_name))

                    # If we're sampling kinematics, sort children based on (a) whether they are cloth or not, and then
                    # (b) their AABB, so that first all rigid objects are sampled before cloth objects, and within each
                    # group the larger objects are sampled first
                    if group == "kinematic":
                        rigid_conditions = [c for c in conditions_to_sample if c[2].prim_type != PrimType.CLOTH]
                        cloth_conditions = [c for c in conditions_to_sample if c[2].prim_type == PrimType.CLOTH]
                        conditions_to_sample = list(
                            reversed(sorted(rigid_conditions, key=lambda x: th.prod(x[2].aabb_extent)))
                        ) + list(reversed(sorted(cloth_conditions, key=lambda x: th.prod(x[2].aabb_extent))))

                    # Sample!
                    for condition, positive, entity, child_scope_name in conditions_to_sample:
                        success = False

                        kwargs = dict()
                        # Reset if we're sampling a kinematic state
                        if condition.STATE_NAME in {"inside", "ontop", "under"}:
                            kwargs["reset_before_sampling"] = True
                        elif condition.STATE_NAME in {"attached"}:
                            kwargs["bypass_alignment_checking"] = True
                            kwargs["check_physics_stability"] = True
                            kwargs["can_joint_break"] = False

                        while True:
                            num_trials = 1
                            for _ in range(num_trials):
                                success = condition.sample(binary_state=positive, **kwargs)
                                log_msg = " ".join(
                                    [
                                        "initial final kinematic condition sampling",
                                        condition.STATE_NAME,
                                        str(condition.body),
                                        str(success),
                                    ]
                                )
                                log.warning(log_msg)
                                if success:
                                    # Update state
                                    state = og.sim.dump_state(serialized=False)
                                    break
                            if success:
                                # After the final round of kinematic sampling, we assign in_rooms to newly imported objects
                                if group == "kinematic":
                                    parent = self._object_scope[condition.body[1]]
                                    entity.in_rooms = deepcopy(parent.in_rooms)

                                # Can terminate immediately
                                break

                            # Can't re-sample non-kinematics or rescale cloth or agent, so in
                            # those cases terminate immediately
                            if (
                                group != "kinematic"
                                or condition.STATE_NAME == "attached"
                                or "agent" in child_scope_name
                                or entity.prim_type == PrimType.CLOTH
                            ):
                                break

                            if dynamic_scale:
                                # If any scales are equal or less than the lower threshold, terminate immediately
                                new_scale = entity.scale - m.DYNAMIC_SCALE_INCREMENT
                                if th.any(new_scale < m.MIN_DYNAMIC_SCALE):
                                    break

                                # Re-scale and re-attempt
                                # Re-scaling is not respected unless sim cycle occurs
                                og.sim.stop()
                                entity.scale = new_scale
                                log.info(
                                    f"Kinematic sampling {condition.STATE_NAME} {condition.body} failed, rescaling obj: {child_scope_name} to {entity.scale}"
                                )
                                og.sim.play()
                                og.sim.load_state(state, serialized=False)
                                og.sim.step_physics()
                            else:
                                break
                        if not success:
                            # Update object registry because we just assigned in_rooms to newly imported objects
                            self._env.scene.object_registry.update(keys=["in_rooms"])
                            return f"Sampleable object conditions failed: {condition.STATE_NAME} {condition.body}"

        # Update object registry because we just assigned in_rooms to newly imported objects
        self._env.scene.object_registry.update(keys=["in_rooms"])

        # One more sim step to make sure the object states are propagated correctly
        # E.g. after sampling Filled.set_value(True), Filled.get_value() will become True only after one step
        og.sim.step()

    def _sample_conditions(self, input_object_scope, conditions, condition_type):
        """
        Sample conditions

        Args:
            input_object_scope (dict):
            conditions (list): List of conditions to filter scope with, where each list entry is
                a tuple of (condition, positive), where @positive is True if the condition has a positive
                evaluation.
            condition_type (str): What type of condition to sample, e.g., "initial"

        Returns:
            None or str: If successful, returns None. Otherwise, returns an error message
        """
        error_msg, problematic_objs = "", []
        while not any(
            th.any(self._object_scope[obj_inst].scale < m.MIN_DYNAMIC_SCALE).item() for obj_inst in problematic_objs
        ):
            filtered_object_scope, problematic_objs = self._filter_object_scope(
                input_object_scope, conditions, condition_type
            )
            error_msg = self._consolidate_room_instance(filtered_object_scope, condition_type)
            if error_msg is None:
                break
            # Re-scaling is not respected unless sim cycle occurs
            og.sim.stop()
            for obj_inst in problematic_objs:
                obj = self._object_scope[obj_inst]
                # If the object's initial condition is attachment, or it's agent or cloth, we can't / shouldn't scale
                # down, so play again and then terminate immediately
                if obj_inst in self._attached_objects or "agent" in obj_inst or obj.prim_type == PrimType.CLOTH:
                    og.sim.play()
                    return error_msg, None
                assert th.all(obj.scale > m.DYNAMIC_SCALE_INCREMENT)
                obj.scale -= m.DYNAMIC_SCALE_INCREMENT
            og.sim.play()

        if error_msg:
            return error_msg, None
        return self._maximum_bipartite_matching(filtered_object_scope, condition_type), filtered_object_scope

    def _maximum_bipartite_matching(self, filtered_object_scope, condition_type):
        """
        Matches objects from @filtered_object_scope to specific room instances it can be
        sampled from

        Args:
            filtered_object_scope (dict): Filtered object scope
            condition_type (str): What type of condition to sample, e.g., "initial"

        Returns:
            None or str: If successful, returns None. Otherwise, returns an error message
        """
        # For each room instance, perform maximum bipartite matching between object instance in scope to simulator objects
        # Left nodes: a list of object instance in scope
        # Right nodes: a list of simulator objects
        # Edges: if the simulator object can support the sampling requirement of ths object instance
        for room_type in filtered_object_scope:
            # The same room instances will be shared across all scene obj in a given room type
            some_obj = list(filtered_object_scope[room_type].keys())[0]
            room_insts = list(filtered_object_scope[room_type][some_obj].keys())
            success = False
            # Loop through each room instance
            for room_inst in room_insts:
                graph = nx.Graph()
                # For this given room instance, gether mapping from obj instance to a list of simulator obj
                obj_inst_to_obj_per_room_inst = {}
                for obj_inst in filtered_object_scope[room_type]:
                    obj_inst_to_obj_per_room_inst[obj_inst] = filtered_object_scope[room_type][obj_inst][room_inst]
                top_nodes = []
                log_msg = "MBM for room instance [{}]".format(room_inst)
                log.debug((log_msg))
                for obj_inst in obj_inst_to_obj_per_room_inst:
                    for obj in obj_inst_to_obj_per_room_inst[obj_inst]:
                        # Create an edge between obj instance and each of the simulator obj that supports sampling
                        graph.add_edge(obj_inst, obj)
                        log_msg = "Adding edge: {} <-> {}".format(obj_inst, obj.name)
                        log.debug((log_msg))
                        top_nodes.append(obj_inst)
                # Need to provide top_nodes that contain all nodes in one bipartite node set
                # The matches will have two items for each match (e.g. A -> B, B -> A)
                matches = nx.bipartite.maximum_matching(graph, top_nodes=top_nodes)
                if len(matches) == 2 * len(obj_inst_to_obj_per_room_inst):
                    log.debug(("Object scope finalized:"))
                    for obj_inst, obj in matches.items():
                        if obj_inst in obj_inst_to_obj_per_room_inst:
                            self._object_scope[obj_inst] = BDDLEntity(bddl_inst=obj_inst, entity=obj)
                            log.debug((obj_inst, obj.name))
                    success = True
                    break
            if not success:
                return "{}: Room type [{}] of scene [{}] do not have enough simulator objects that can successfully sample all the objects needed. This is usually caused by specifying too many object instances in the object scope or the conditions are so stringent that too few simulator objects can satisfy them via sampling.\n".format(
                    condition_type, room_type, self._scene_model
                )
