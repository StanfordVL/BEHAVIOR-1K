from collections import Counter
import copy
import csv
import glob
import json
import os
import pandas as pd
import pathlib
import re
import sys

import bddl
from bddl.activity import Conditions
from bddl.data_generation.tm_submap_params import TM_SUBMAPS_TO_PARAMS
from bddl.parsing import parse_problem, parse_domain
from bddl.trivial_backend import *


# Files

BDDL_DIR = pathlib.Path(__file__).parent

PROBLEM_FILE_DIR = BDDL_DIR / "activity_definitions"
PROPS_TO_SYNS_JSON = BDDL_DIR / "generated_data/properties_to_synsets.json"
SYNS_TO_PROPS_JSON = BDDL_DIR / "generated_data/propagated_annots_canonical.json"
SYNS_TO_PROPS_PARAMS_JSON = BDDL_DIR / "generated_data/propagated_annots_params.json"
CSVS_DIR = "tm_csvs"
TRANSITION_MAP_DIR = BDDL_DIR / "generated_data/transition_map/tm_jsons"

# Constants

*__, domain_predicates = parse_domain("omnigibson")
UNARIES = [
    predicate for predicate, inputs in domain_predicates.items() if len(inputs) == 1
]
BINARIES = [
    predicate for predicate, inputs in domain_predicates.items() if len(inputs) == 2
]

VALID_ATTACHMENTS = set(
    [
        ("address.n.05", "wall_nail.n.01"),
        ("antler.n.01", "wall_nail.n.01"),
        ("bicycle.n.01", "bicycle_rack.n.01"),
        ("bicycle_rack.n.01", "recreational_vehicle.n.01"),
        ("bicycle_rack.n.01", "wall_nail.n.01"),
        ("blackberry.n.01", "scrub.n.01"),
        ("bow.n.08", "wall_nail.n.01"),
        ("broken__light_bulb.n.01", "table_lamp.n.01"),
        ("cabinet_door.n.01", "cabinet_base.n.01"),
        ("clothesline_rope.n.01", "clothesline_pole.n.01"),
        ("cork.n.04", "wine_bottle.n.01"),
        ("curtain_rod.n.01", "wall_nail.n.01"),
        ("dartboard.n.01", "wall_nail.n.01"),
        ("desk_bracket.n.01", "desk_top.n.01"),
        ("desk_leg.n.01", "desk_bracket.n.01"),
        ("digital_camera.n.01", "camera_tripod.n.01"),
        ("dip.n.07", "candlestick.n.01"),
        ("fire_alarm.n.02", "wall_nail.n.01"),
        ("gummed_label.n.01", "license_plate.n.01"),
        ("hanger.n.02", "coatrack.n.01"),
        ("hanger.n.02", "wardrobe.n.01"),
        ("hitch.n.04", "pickup.n.01"),
        ("holly.n.03", "wall_nail.n.01"),
        ("icicle_lights.n.01", "wall_nail.n.01"),
        ("kayak.n.01", "kayak_rack.n.01"),
        ("kayak_rack.n.01", "wall_nail.n.01"),
        ("lens.n.01", "digital_camera.n.01"),
        ("lid.n.02", "hingeless_jar.n.01"),
        ("license_plate.n.01", "car.n.01"),
        ("light_bulb.n.01", "table_lamp.n.01"),
        ("mirror.n.01", "wall_nail.n.01"),
        ("paper_lantern.n.01", "pole.n.01"),
        ("picture_frame.n.01", "wall_nail.n.01"),
        ("pole.n.01", "wall_nail.n.01"),
        ("poster.n.01", "wall_nail.n.01"),
        ("raspberry.n.02", "scrub.n.01"),
        ("bookcase_baseboard.n.01", "bookcase_side.n.01"),
        ("bookcase_shelf.n.01", "bookcase_back.n.01"),
        ("bookcase_side.n.01", "bookcase_back.n.01"),
        ("bookcase_top.n.01", "bookcase_back.n.01"),
        ("skateboard_wheel.n.01", "skateboard_deck.n.01"),
        ("trampoline_leg.n.01", "trampoline_top.n.01"),
        ("trout.n.01", "fishing_rod.n.01"),
        ("webcam.n.02", "monitor.n.04"),
        ("window_blind.n.01", "wall_nail.n.01"),
        ("wind_chime.n.01", "pole.n.01"),
        ("wreath.n.01", "wall_nail.n.01"),
    ]
)

VALID_ROOMS = set()

OBJECT_INSTANCE_RE = r"[A-Za-z-_]+\.n\.[0-9]+_[0-9]+"
OBJECT_CAT_RE = r"[A-Za-z-_]+\.n\.[0-9]+$"
OBJECT_CAT_AND_INST_RE = r"[A-Za-z-_]+\.n\.[0-9]+"
SINGLE_CAT_QUANTS = ["forall", "exists", "forn"]
DOUBLE_CAT_QUANTS = ["forpairs", "fornpairs"]
ROOMS = set(
    [
        "kitchen",
        "dining_room",
        "living_room",
        "utility_room",
        "empty_room",
        "bedroom",
        "bathroom",
        "garden",
        "shared_office",
        "corridor",
        "classroom",
        "grocery_store",
        "computer_lab",
        "playroom",
        "sauna",
        "childs_room",
        "garage",
        "closet",
        "storage_room",
        "entryway",
        "private_office",
        "meeting_room",
        "bar",
        "staircase",
        "spa",
        "television_room",
        "lobby",
        "break_room",
    ]
)
PLACEMENTS = set(
    [
        "ontop",
        "inside",
        "under",
        "overlaid",
        "future",
        "attached",
        "draped",
    ]
)
SUBSTANCE_PLACEMENTS = set(["saturated", "filled", "covered", "insource", "contains"])
FUTURE_PREDICATE = "future"


# Helpers


def _traverse_goal_for_objects(expr, objects=None):
    objects = objects if objects is not None else set()
    # Check that category declarations in quantifiers are really categories, and equal
    if expr[0] in ["forall", "exists", "forpairs"]:
        term, __, cat = expr[1]
        assert (
            term.strip("?") == cat
        ), f"mismatched term and cat declaration: {term}, {cat}"
        assert (
            re.match(OBJECT_CAT_RE, term.strip("?")) is not None
        ), f"non-category term in quantifier declaration: {term}"
        if expr[0] in ["forpairs"]:
            term, __, cat = expr[2]
            assert (
                term.strip("?") == cat
            ), f"mismatched term and cat declaration: {term}, {cat}"
            assert (
                re.match(OBJECT_CAT_RE, term.strip("?")) is not None
            ), f"non-category term in quantifier declaration: {term}"
        _traverse_goal_for_objects(expr[-1], objects=objects)
    if expr[0] in ["forn", "fornpairs"]:
        term, __, cat = expr[2]
        assert (
            term.strip("?") == cat
        ), f"mismatched term and cat declaration: {term}, {cat}"
        assert (
            re.match(OBJECT_CAT_RE, term.strip("?")) is not None
        ), f"non-category term in quantifier declaration: {term}"
        if expr[0] == "fornpairs":
            term, __, cat = expr[3]
            assert (
                term.strip("?") == cat
            ), f"mismatched term and cat declaration: {term}, {cat}"
            assert (
                re.match(OBJECT_CAT_RE, term.strip("?")) is not None
            ), f"non-category term in quantifier declaration: {term}"
        _traverse_goal_for_objects(expr[-1], objects=objects)

    # Check the subexpr for atomic formulae in base case, else recurse
    if type(expr[-1]) is not list:
        for obj in expr[1:]:
            assert (
                re.match(OBJECT_CAT_AND_INST_RE, obj.strip("?")) is not None
            ), f"malformed object term in goal: {obj}"
            objects.add(obj.strip("?"))
    else:
        if expr[0] in ["and", "or"]:
            for subexpr in expr[1:]:
                _traverse_goal_for_objects(subexpr, objects=objects)
        else:
            _traverse_goal_for_objects(expr[-1], objects=objects)


def _get_defn_elements_from_file(activity):
    defn_fn = os.path.join(PROBLEM_FILE_DIR, activity, "problem0.bddl")
    with open(defn_fn, "r") as f:
        __, objects, init, goal = parse_problem(
            activity, 0, "omnigibson", predefined_problem=f.read()
        )
    return activity, objects, init, goal


def _get_objects_from_object_list(objects):
    instances, categories = set(), set()
    for cat, insts in objects.items():
        categories.add(cat)
        for inst in insts:
            instances.add(inst)
    return instances, categories


def _get_instances_in_init(init):
    """
    Take a parsed :init condition and return a set of all instances in it.
    """
    init_insts = set()
    for literal in init:
        formula = literal[1] if literal[0] == "not" else literal
        for inst in formula[1:]:
            assert (re.match(OBJECT_INSTANCE_RE, inst) is not None) or (
                inst in ROOMS
            ), f":init has category: {inst}"
            if inst not in ROOMS:
                init_insts.add(inst)
    return init_insts


def _get_objects_in_goal(goal):
    goal_objects = set()
    goal = ["and"] + goal
    _traverse_goal_for_objects(goal, goal_objects)
    return goal_objects


def _traverse_goal_for_atoms(expr, goal_atoms):
    if all(type(subexpr) == str for subexpr in expr):
        goal_atoms.append(expr)
    elif expr[0] in ["and", "or"]:
        for subexpr in expr[1:]:
            _traverse_goal_for_atoms(subexpr, goal_atoms)
    elif expr[0] in ["forall", "exists", "forn", "forpairs", "fornpairs"]:
        _traverse_goal_for_atoms(expr[-1], goal_atoms)
    elif expr[0] == "imply":
        _traverse_goal_for_atoms(expr[1], goal_atoms)
        _traverse_goal_for_atoms(expr[2], goal_atoms)
    elif expr[0] == "not":
        _traverse_goal_for_atoms(expr[1], goal_atoms)
    else:
        raise ValueError(f"Unhandled logic operator {expr[0]}")


def _get_atoms_in_goal(goal):
    goal_atoms = []
    for goal_expr in goal:
        _traverse_goal_for_atoms(goal_expr, goal_atoms)
    return goal_atoms


def _get_unique_items_from_transition_map():
    obj_set = set()
    for fname in glob.glob(CSVS_DIR):
        with open(fname) as f:
            for row in f:
                first = row.split(",")[0]
                if ".n." in first:
                    obj_set.add(first.rpartition("_")[0])

    obj_set.remove("")

    for obj in obj_set:
        print(obj)


def is_specific_container_synset(synset):
    return (
        "__" in synset
        and "__of__" not in synset
        and "diced__" not in synset
        and "cooked__" not in synset
        and "half__" not in synset
    )


def check_synset_predicate_alignment(atom, syns_to_props):
    if atom[0] == "ontop" and atom[1] == "agent.n.01_1":
        return

    pred, *object_insts = atom
    objects = []
    for object_inst in object_insts:
        syn_match = re.match(OBJECT_CAT_AND_INST_RE, object_inst.strip("?"))
        if syn_match is not None:
            objects.append(syn_match.group())
        elif True:  # object_inst in VALID_ROOMS:    # TODO uncomment when VALID_ROOMS is populated
            if pred == "inroom":
                objects.append(object_inst)
            else:
                raise AssertionError(f"Nonsynset {object_inst} outside inroom")
        else:
            raise AssertionError(f"Invalid room {object_inst}")
    assert (pred in UNARIES) or (pred in BINARIES), f"Invalid predicate: {pred}"
    assert ((pred in UNARIES) and (len(objects) == 1)) or (
        (pred in BINARIES) and (len(objects) == 2)
    ), f"Atom has wrong arity: {atom}"

    if pred == "cooked":
        assert (
            "nonSubstance" in syns_to_props[objects[0]]
        ), f"Inapplicable cooked: {atom}"
        assert "cookable" in syns_to_props[objects[0]], f"Inapplicable cooked: {atom}"
    if pred == "frozen":
        assert (
            "nonSubstance" in syns_to_props[objects[0]]
        ), f"Inapplicable frozen: {atom}"
        assert "freezable" in syns_to_props[objects[0]], f"Inapplicable frozen: {atom}"
    if pred == "closed" or pred == "open":
        assert (
            "rigidBody" in syns_to_props[objects[0]]
        ), f"Inapplicable closed/open: {atom}"
        assert (
            "openable" in syns_to_props[objects[0]]
        ), f"Inapplicable closed/open: {atom}"
    if pred == "folded" or pred == "unfolded":
        # cloth or rope is drapeable
        assert (
            "drapeable" in syns_to_props[objects[0]]
        ), f"Inapplicable folded/unfolded: {atom}"
    if pred == "toggled_on":
        assert (
            "rigidBody" in syns_to_props[objects[0]]
        ), f"Inapplicable toggled_on: {atom}"
        assert (
            "toggleable" in syns_to_props[objects[0]]
        ), f"Inapplicable toggled_on: {atom}"
    if pred == "hot":
        assert "nonSubstance" in syns_to_props[objects[0]], f"Inapplicable hot: {atom}"
        assert "heatable" in syns_to_props[objects[0]], f"Inapplicable hot: {atom}"
    if pred == "on_fire":
        assert (
            "nonSubstance" in syns_to_props[objects[0]]
        ), f"Inapplicable on_fire: {atom}"
        assert "flammable" in syns_to_props[objects[0]], f"Inapplicable on_fire: {atom}"
    if pred == "broken":
        assert "rigidBody" in syns_to_props[objects[0]], f"Inapplicable broken: {atom}"
        assert "breakable" in syns_to_props[objects[0]], f"Inapplicable broken: {atom}"

    # Binaries
    if pred == "saturated":
        assert (
            ("nonSubstance" in syns_to_props[objects[0]])
            and ("particleRemover" in syns_to_props[objects[0]])
            and ("substance" in syns_to_props[objects[1]])
        ), f"Inapplicable saturated: {atom}"
    if pred == "covered":
        assert ("nonSubstance" in syns_to_props[objects[0]]) and (
            "substance" in syns_to_props[objects[1]]
        ), f"Inapplicable covered: {atom}"
        if "drapeable" in syns_to_props[objects[0]]:
            assert (
                "visualSubstance" in syns_to_props[objects[1]]
            ), f"Inapplicable covered: {atom}"
    if pred == "filled":
        assert (
            ("rigidBody" in syns_to_props[objects[0]])
            and ("fillable" in syns_to_props[objects[0]])
            and ("physicalSubstance" in syns_to_props[objects[1]])
        ), f"Inapplicable filled/empty: {atom}"
    if pred == "contains" or pred == "empty":
        assert (
            ("rigidBody" in syns_to_props[objects[0]])
            and ("fillable" in syns_to_props[objects[0]])
            and ("substance" in syns_to_props[objects[1]])
        ), f"Inapplicable contains: {atom}"
    if pred == "ontop":
        assert ("nonSubstance" in syns_to_props[objects[0]]) and (
            "rigidBody" in syns_to_props[objects[1]]
            or "softBody" in syns_to_props[objects[1]]
        ), f"Inapplicable ontop: {atom}"
    if pred == "nextto":
        assert ("nonSubstance" in syns_to_props[objects[0]]) and (
            "nonSubstance" in syns_to_props[objects[1]]
        ), f"Inapplicable nextto: {atom}"
    if pred == "under":
        assert ("nonSubstance" in syns_to_props[objects[0]]) and (
            "rigidBody" in syns_to_props[objects[1]]
            or "softBody" in syns_to_props[objects[1]]
        ), f"Inapplicable under: {atom}"
    if pred == "touching":
        assert ("rigidBody" in syns_to_props[objects[0]]) and (
            "rigidBody" in syns_to_props[objects[1]]
        ), f"Inapplicable touching: {atom}"
    if pred == "inside":
        assert ("nonSubstance" in syns_to_props[objects[0]]) and (
            "rigidBody" in syns_to_props[objects[1]]
            or "softBody" in syns_to_props[objects[1]]
        ), f"Inapplicable inside: {atom}"
        assert "fillable" in syns_to_props[objects[1]], f"Inapplicable inside: {atom}"
    if pred == "overlaid":
        assert ("drapeable" in syns_to_props[objects[0]]) and (
            "rigidBody" in syns_to_props[objects[1]]
        ), f"Inapplicable overlaid: {atom}"
    if pred == "attached":
        assert tuple(objects) in VALID_ATTACHMENTS, f"Inapplicable attached: {atom}"
        assert ("rigidBody" in syns_to_props[objects[0]]) and (
            "rigidBody" in syns_to_props[objects[1]]
        ), f"Inapplicable attached: {atom}"
    if pred == "draped":
        assert ("drapeable" in syns_to_props[objects[0]]) and (
            "rigidBody" in syns_to_props[objects[1]]
        ), f"Inapplicable overlaid: {atom}"
    if pred == "insource":
        assert (
            ("particleSource" in syns_to_props[objects[0]])
            or ("particleApplier" in syns_to_props[objects[0]])
        ) and (
            "substance" in syns_to_props[objects[1]]
        ), f"Inapplicable insource: {atom}"
    if pred == "inroom":
        assert (
            "sceneObject" in syns_to_props[objects[0]]
        ), f"Inapplicable inroom: {atom}"


def check_clashing_transition_rules():
    # Check within each submap
    for submap_name in TM_SUBMAPS_TO_PARAMS:
        with open(
            os.path.join(
                "..",
                "bddl",
                "generated_data",
                "transition_map",
                "tm_jsons",
                submap_name + ".json",
            ),
            "r",
        ) as f:
            submap = json.load(f)
        seen_object_sets = []
        for rule in submap:
            # Get relevant parameters
            rule_name = rule.get("rule_name", "No name")
            input_objects = rule.get("input_synsets", {})
            input_states = rule.get("input_states", {})
            input_states = input_states if input_states is not None else {}
            if submap_name == "heat_cook":
                equipment = set(
                    [
                        list(rule["heat_source"].keys())[0],
                        list(rule["container"].keys())[0],
                    ]
                )
            elif submap_name == "single_toggleable_machine":
                equipment = set([list(rule["machine"].keys())[0]])
            else:
                equipment = set()  # Equivalence will pass trivially when checked, because this rule already clashes
            output_objects = rule.get("output_synsets", {})
            output_states = rule.get("output_states", {})
            output_states = output_states if output_states is not None else {}

            # NOTE doing input_objects.keys, not input_objects.items, because simulator is not actually sensitive to amount. It only checks for category,
            #   so different amounts but same categories still need to result in the same output.

            # Collect all sets of input objects to check subsetting
            input_objects = set(sorted(input_objects.keys(), key=lambda x: x[0]))
            output_objects = set(sorted(output_objects.keys(), key=lambda x: x[0]))
            input_states_strs = set(
                [
                    syns
                    + "@"
                    + ";".join(
                        [
                            f"{pred}:{val}"
                            for pred, val in sorted(states, key=lambda x: x[0])
                        ]
                    )
                    for syns, states in sorted(input_states.items(), key=lambda x: x[0])
                ]
            )
            output_states_strs = set(
                [
                    syns
                    + "@"
                    + ";".join(
                        [
                            f"{pred}:{val}"
                            for pred, val in sorted(states, key=lambda x: x[0])
                        ]
                    )
                    for syns, states in sorted(
                        output_states.items(), key=lambda x: x[0]
                    )
                ]
            )
            for (
                seen_rule_name,
                seen_object_set,
                seen_states_set,
                seen_equipment,
                seen_output_objects,
                seen_output_states,
            ) in seen_object_sets:
                # If we see that our next input objects set is a subset or superset...
                if input_objects.issuperset(seen_object_set) or input_objects.issubset(
                    seen_object_set
                ):
                    # Construct a set of atomic formulae in string format
                    if input_states_strs.issuperset(
                        seen_states_set
                    ) or input_states_strs.issubset(seen_states_set):
                        if equipment == seen_equipment:
                            # At this point, the output needs to be identical
                            if (
                                not output_objects == seen_output_objects
                                or not output_states_strs == seen_output_states
                            ):
                                raise AssertionError(
                                    f"Clashing rules with input objects {rule_name} and {seen_rule_name} in submap {submap_name}."
                                )

            seen_object_sets.append(
                (
                    rule_name,
                    input_objects,
                    input_states_strs,
                    equipment,
                    output_objects,
                    output_states_strs,
                )
            )

    # Check heat_cook rules vs cooking individual items, since those are a potential clash that we know of
    # for activity in os.listdir(PROBLEM_FILE_DIR):
    #     if not os.path.isdir(os.path.join(PROBLEM_FILE_DIR, activity)): continue
    #     __, objects, init, goal = _get_defn_elements_from_file(activity)

    #     # Check for a rigid body ending cooked and starting not cooked/cook state unspecified, or a future clause for a cooked substance
    #     for cooking_rigids:

    # NOTE other heuristics as we discover them


# BDDL tests


def is_water_sourced(objects, init, goal):
    obj_list = []
    for obj_cat in objects:
        obj_list.extend(objects[obj_cat])
    objs = set(obj_list)
    # NOTE Assumes that if water is in goal, it'll be in objects
    if "water.n.06_1" in objs:
        # Find a placement for the water in the init
        for atom in init:
            # match = re.match(r'^\((filled|covered) water\.n\.06_1', atom) is not None
            placed = (atom[0] in set(["filled", "covered"])) and (
                atom[1] == "water.n.06_1"
            )
            # match_insource = re.match(r'^\(insource water\.n\.06_1 sink\.n\.01_[0-9]+\)$', atom) is not None
            insourced = (
                (atom[0] == "insource")
                and (atom[1] == "water.n.06_1")
                and (re.match(r"sink\.n\.01_[0-9]+", atom[2]) is not None)
            )
            if placed:
                # Make sure the container is in the object list
                container = atom[-1]
                if container in objs:
                    print("water was in container")
                    break
                else:
                    # print(container)
                    # print(objs)
                    raise ValueError(
                        "Water is filling/covering something in the init, but that thing is not in the objects section."
                    )
            if insourced:
                # Make sure the sink is in the object list
                sink = atom[-1]
                if sink not in objs:
                    raise ValueError(
                        "Water is insource a sink, but that sink is not in the objects section."
                    )
                # Make sure the sink is in a room
                for atom in init:
                    if atom[0] == "inroom" and atom[1] == sink:
                        print("water was in sink and sink was in room")
                        return True
                else:
                    raise ValueError(
                        "Water is insource a sink and sink is in objects section, but not in an inroom"
                    )
    else:
        print("no water")
    return True


# Checkers


def object_list_correctly_formatted(defn):
    """
    Verify that object list has the following properties for each line:
    - Each ending term matches category regex
    - Penultimate term is `-`
    - First through third-to-last terms match instance regex
    - First through third-to-last terms are of the same category as the ending term
    """
    objects_section = defn.split("(:objects\n")[-1].split("\n    )")[0].split("\n")
    for line in objects_section:
        elements = line.strip().split(" ")
        category = elements[-1]
        assert (
            re.match(OBJECT_CAT_RE, category) is not None
        ), f"Malformed category at end of object section line: {category}"
        assert (
            elements[-2] == "-"
        ), f"There should be a hyphen but instead there is {elements[-2]}"
        for inst in elements[:-2]:
            assert (
                re.match(OBJECT_INSTANCE_RE, inst) is not None
            ), f"Malformed instance {inst}"
            assert (
                category == re.match(OBJECT_CAT_AND_INST_RE, inst).group()
            ), f"Mismatched category and object: {category} and {inst}"


def all_objects_appropriate(objects, init, goal):
    """
    Checks the following:
    1. All objects in :objects are in :init
    2. All objects in :init are in :objects (so basically, the set of objects in :objects and
        the set of objects in :init are equivalent)
    3. The set of object instances and categories in :goal is a subset of the set of object
        instances and categories in :objects
    4. There are no categories in :init
    """
    instances, categories = _get_objects_from_object_list(objects)
    init_insts = _get_instances_in_init(init)

    assert init_insts.issubset(
        instances
    ), f":init has object instances not in :objects: {init_insts.difference(instances)}"
    assert instances.issubset(
        init_insts
    ), f":objects has object instances not in :init: {instances.difference(init_insts)}"

    goal_objects = _get_objects_in_goal(goal)
    assert goal_objects.issubset(
        categories.union(instances)
    ), f":goal has objects not in :objects: {goal_objects.difference(categories.union(instances))}"


def all_objects_placed(init):
    """
    Make sure everything is placed relative to a ROOM, even transitively.
    This is because when I hand-edit, I may accidentally not put every new scene object in an inroom.
    Meanwhile, it's unlikely that TODO fill in whatever else was I thinking of that this function won't check
        but is unlikely lol.
        - Appropriate placements for the object type - I'm unlikely to mess that up if I do say so myself

    NOTE should only be executed AFTER all_objects_appropraite
    """
    insts = _get_instances_in_init(init)
    insts = set([inst for inst in insts if ["future", inst] not in init])

    in_room_check = True
    last_placed = None
    in_room_placed = set()
    while True:
        newly_placed = set()
        for literal in init:
            # Skip not literals
            if literal[0] == "not":
                assert literal[1][0] not in {
                    "attached",
                    "draped",
                    "inside",
                    "nextto",
                    "ontop",
                    "overlaid",
                    "touching",
                    "under",
                }, f"Negative kinematic state in initial states: {literal}"
                continue
            formula = literal
            # Skip future literals
            if formula[0] == "future":
                continue
            inst = None
            substance_placement = False
            if in_room_check:
                # For the first round, check for inroom
                if (formula[0] == "inroom") and (formula[2] in ROOMS):
                    inst = formula[1]
                    in_room_placed.add(inst)
            else:
                # For the following rounds, check for placements w.r.t last placed objects
                if (formula[0] in PLACEMENTS) and (formula[2] in last_placed):
                    inst = formula[1]
                    assert inst not in in_room_placed, f"Object {inst} is placed twice"
                # Or substasnce placements w.r.t last placed objects
                elif (formula[0] in SUBSTANCE_PLACEMENTS) and (
                    formula[1] in last_placed
                ):
                    inst = formula[2]
                    substance_placement = True

            if inst is not None:
                # If it's not a substance placement, we make sure it's only placed once (e.g. we should not place the
                # same eapple on table1 and on table2). If it's a substance placement, it's fine (e.g. we can do stain
                # covering table1 and table2)
                if not substance_placement:
                    assert inst not in newly_placed, f"Object {inst} is placed twice"
                newly_placed.add(inst)

        # If no new objects were placed, we're done
        if len(newly_placed) == 0:
            break

        # Otherwise, we remove the newly placed objects from the list of objects to place and continue
        insts -= newly_placed
        last_placed = newly_placed
        in_room_check = False

    # If there are any objects left, they are unplaced
    assert len(insts) == 0, f"Unplaced object instances: {insts}"


def no_invalid_synsets(objects, init, goal, syns_to_props):
    instances, categories = _get_objects_from_object_list(objects)
    init_insts = _get_instances_in_init(init)
    goal_objects = _get_objects_in_goal(goal)
    object_insts = set(
        [
            re.match(OBJECT_CAT_AND_INST_RE, inst).group()
            for inst in instances.union(init_insts).union(goal_objects)
        ]
    )
    object_terms = object_insts.union(categories)
    for proposed_syn in object_terms:
        assert (proposed_syn in syns_to_props) or (
            proposed_syn == "agent.n.01"
        ), f"Invalid synset: {proposed_syn}"


def no_invalid_predicates(init, goal, domain_predicates):
    atoms = []
    for literal in init:
        atoms.append(literal[1] if literal[0] == "not" else literal)
    atoms.extend(_get_atoms_in_goal(goal))
    for atom in atoms:
        assert atom[0] in domain_predicates, f"Invalid predicate: {atom[0]}"


# Check uncontrolled categories


def future_and_real_present(objects, init, goal):
    init_objects = _get_instances_in_init(init)
    future_objects = set([literal[1] for literal in init if literal[0] == "future"])
    real_objects = set(
        [expression[1].strip("?") for expression in goal if expression[0] == "real"]
    )
    for expression in goal:
        if expression[0] == "or":
            for disjunct in expression[1:]:
                if disjunct[0] == "real":
                    real_objects.add(disjunct[1].strip("?"))
    assert (
        future_objects.issubset(real_objects)
        or (
            ("washer.n.03" in objects)
            and future_objects.difference(real_objects) == set(["water.n.06_1"])
        )
    ), f"{future_objects.difference(real_objects)} in future clauses but not real clauses (and doesn't satisfy washer/water exception)"
    assert real_objects.issubset(
        future_objects.union(init_objects)
    ), f"{real_objects.difference(future_objects)} in real clauses but not future clauses or init"


def no_repeat_object_lines(defn):
    from pprint import pprint

    objects_lines = [
        line.strip()
        for line in defn.split("(:objects")[1].split("(:init")[0].split("\n")[1:-3]
    ]
    seen_cats = set()
    for objects_line in objects_lines:
        if objects_line == ")":
            continue
        *__, __, cat = objects_line.split(" ")
        assert cat not in seen_cats, f"Category repeated: {cat}"
        seen_cats.add(cat)


def no_qmarks_in_init(defn):
    init = defn.split("(:init")[1].split("(:goal")[0]
    assert "?" not in init, "Inappropriate ? in :init."


def no_contradictory_init_atoms(init):
    for literal in init:
        if literal[0] == "not":
            assert (
                literal[1] not in init
            ), f"Contradictory init statements: {literal[1]}"


def no_uncontrolled_category(activity, defn):
    conds = bddl.activity.Conditions(activity, 0, "omnigibson", predefined_problem=defn)
    scope = bddl.activity.get_object_scope(conds)
    bddl.activity.get_initial_conditions(
        conds, TrivialBackend(), scope, generate_ground_options=False
    )
    # Pretend scope has been filled
    for name in scope:
        scope[name] = TrivialGenericObject(name, TrivialSimulator())
    bddl.activity.get_goal_conditions(
        conds, TrivialBackend(), scope, generate_ground_options=False
    )


def agent_present(init):
    for literal in init:
        if (literal[0] == "ontop") and (literal[1] == "agent.n.01_1"):
            break
    else:
        raise AssertionError("Agent not present.")


def problem_name_correct(activity, definition_id=0):
    defn_fn = os.path.join(PROBLEM_FILE_DIR, activity, f"problem{definition_id}.bddl")
    with open(defn_fn, "r") as f:
        problem_name, *__ = parse_problem(
            activity, 0, "omnigibson", predefined_problem=f.read()
        )
    assert (problem_name == f"{activity}-{definition_id}") or (
        problem_name == f"{activity.lower()}-{definition_id}"
    ), f"Wrong problem name '{problem_name}' for activity '{activity}'"


def no_misaligned_synsets_predicates(init, goal, syns_to_props):
    for literal in init:
        init_atom = literal[1] if literal[0] == "not" else literal
        check_synset_predicate_alignment(init_atom, syns_to_props)
    goal_atoms = _get_atoms_in_goal(goal)
    for goal_atom in goal_atoms:
        check_synset_predicate_alignment(goal_atom, syns_to_props)


def no_unnecessary_specific_containers(objects, init, goal, syns_to_props):
    specific_fillable_containers = [
        obj_cat
        for obj_cat in objects.keys()
        if obj_cat != "agent.n.01"
        and "fillable" in syns_to_props[obj_cat]
        and is_specific_container_synset(obj_cat)
    ]

    atoms = []
    for literal in init:
        atoms.append(literal[1] if literal[0] == "not" else literal)
    goal_atoms = [
        [term.strip("?") for term in goal_atom]
        for goal_atom in _get_atoms_in_goal(goal)
    ]
    atoms.extend(goal_atoms)
    fill_atoms = [
        atom
        for atom in atoms
        if (atom[0] in ["filled", "contains", "insource", "inside"])
    ]

    for specific_fillable_container in specific_fillable_containers:
        for atom in fill_atoms:
            # print(atom)
            if (atom[0] in ["filled", "contains", "insource"]) and (
                re.match(OBJECT_CAT_AND_INST_RE, atom[1]).group()
                == specific_fillable_container
            ):
                break
            if (atom[0] == "inside") and (
                re.match(OBJECT_CAT_AND_INST_RE, atom[2]).group()
                == specific_fillable_container
            ):
                break
        else:
            raise AssertionError(
                f"Substance-specific fillable container {specific_fillable_container} that does not fill/contain anything/have anything inside. Switch to container__of version."
            )


def no_substances_with_multiple_instances(objects, syns_to_props):
    for cat, insts in objects.items():
        if "substance" in syns_to_props[cat]:
            assert (
                len(insts) == 1
            ), f"Substance {cat} with {len(insts)} instances instead of 1"


# Transition map tests


def no_missing_required_params(rule, submap):
    param_metadata = TM_SUBMAPS_TO_PARAMS[submap]
    for param in param_metadata:
        assert (
            (not param_metadata[param]["required"]) or (rule[param] is not None)
        ), f"Required param {param} of rule {rule['rule_name']} is required but not present."
    for param, value in rule.items():
        if param_metadata[param]["required"]:
            assert (
                value is not None
            ), f"Required param {param} of rule {rule['rule_name']} is required has no value"


def no_incorrectly_formatted_params(rule, submap):
    param_metadata = TM_SUBMAPS_TO_PARAMS[submap]
    for param, value in rule.items():
        if param_metadata[param]["type"] == "synset" and value is not None:
            assert (
                type(value) == dict
            ), f"Malformed synset-type value for param {param} in rule {rule['rule_name']}"
            for proposed_synset, proposed_integer in value.items():
                assert (
                    re.match(OBJECT_CAT_RE, proposed_synset) is not None
                ), f"Malformed synset {proposed_synset} in param {param} of rule {rule['rule_name']} in submap {submap}"
                assert (
                    type(proposed_integer) == int
                ), f"Malformed integer {proposed_integer} in param {param} of rule {rule['rule_name']}"
        elif param_metadata[param]["type"] == "atom" and value is not None:
            assert (
                type(value) == dict
            ), f"Malformed atom-type value for param {param} in rule {rule['rule_name']} in submap {submap}"
            for proposed_synsets, proposed_predicates_values in value.items():
                for proposed_synset in proposed_synsets.split(","):
                    assert (
                        re.match(OBJECT_CAT_RE, proposed_synset) is not None
                    ), f"Malformed synset {proposed_synset} in param {param} of rule {rule['rule_name']} in submap {submap}"
                for proposed_predicate_value in proposed_predicates_values:
                    assert (
                        len(proposed_predicate_value) == 2
                    ), f"Malformed predicate-value pair {proposed_predicate_value} for param {param} in rule {rule['rule_name']}"
                    predicate, val = proposed_predicate_value
                    assert (
                        type(predicate) == str
                    ), f"Malformed predicate {predicate} for param {param} in rule {rule['rule_name']} in submap {submap}"
                    assert (
                        type(val) == bool
                    ), f"Malformed predicate value {val} for param {param} in rule {rule['rule_name']} in submap {submap}"
        elif param_metadata[param]["type"] == "integer":
            continue
        elif param_metadata[param]["type"] == "string":
            continue
        elif value is None:
            continue
        else:
            raise ValueError(
                f"Unhandled parameter type {param_metadata[param]['type']}"
            )


def no_invalid_synsets_tm(rule, submap, syns_to_props):
    param_metadata = TM_SUBMAPS_TO_PARAMS[submap]
    for param, value in rule.items():
        if (
            param_metadata[param]["type"] == "synset" and value is not None
        ):  # we know the format since we enforced it above
            for proposed_synset in value.keys():
                assert (
                    proposed_synset in syns_to_props
                ), f"Invalid synset: {proposed_synset} in rule {rule['rule_name']} in submap {submap}"
        elif param_metadata[param]["type"] == "atom" and value is not None:
            for proposed_synsets in value.keys():
                for proposed_synset in proposed_synsets.split(","):
                    assert (
                        proposed_synset in syns_to_props
                    ), f"Invalid synset {proposed_synset} in rule {rule['rule_name']} in submap {submap}"
        elif param_metadata[param]["type"] == "integer":
            continue
        elif param_metadata[param]["type"] == "string":
            continue
        elif value is None:
            continue
        else:
            raise ValueError(
                f"Unhandled parameter type {param_metadata[param]['type']}"
            )


def no_invalid_predicates_tm(rule, submap, domain_predicates):
    param_metadata = TM_SUBMAPS_TO_PARAMS[submap]
    for param, value in rule.items():
        if param_metadata[param]["type"] == "synset":
            continue
        elif param_metadata[param]["type"] == "integer":
            continue
        elif param_metadata[param]["type"] == "string":
            continue
        elif value is None:
            continue
        elif param_metadata[param]["type"] == "atom":
            for __, proposed_predicate_values in value.items():
                for proposed_predicate, __ in proposed_predicate_values:
                    assert (
                        proposed_predicate in domain_predicates
                    ), f"Invalid predicate {proposed_predicate} in rule {rule['rule_name']} in submap {submap}"
        else:
            raise ValueError(
                f"Unhandled parameter type {param_metadata[param]['type']}"
            )


def no_misaligned_synsets_predicates_tm(rule, submap, syns_to_props):
    param_metadata = TM_SUBMAPS_TO_PARAMS[submap]
    for param, value in rule.items():
        if param_metadata[param]["type"] == "synset":
            continue
        elif param_metadata[param]["type"] == "integer":
            continue
        elif param_metadata[param]["type"] == "string":
            continue
        elif value is None:
            continue
        elif param_metadata[param]["type"] == "atom":
            for synsets, predicate_vals in value.items():
                for predicate, __ in predicate_vals:
                    synsets = synsets.split(",")
                    check_synset_predicate_alignment(
                        [predicate, *synsets], syns_to_props
                    )
        else:
            raise ValueError(
                f"Unhandled parameter type {param_metadata[param]['type']}"
            )


def no_substances_with_multiple_instances_tm(rule, submap, syns_to_props):
    param_metadata = TM_SUBMAPS_TO_PARAMS[submap]
    for param, value in rule.items():
        if param_metadata[param]["type"] == "synset":
            for synset, num_instances in value.items():
                if "substance" in syns_to_props[synset]:
                    assert (
                        num_instances == 1
                    ), f"Substance {synset} with {num_instances} instances instead of 1 in rule {rule['rule_name']} in submap {submap}"
        elif param_metadata[param]["type"] == "atom":
            continue
        elif param_metadata[param]["type"] == "integer":
            continue
        elif param_metadata[param]["type"] == "string":
            continue
        else:
            raise ValueError(
                f"Unhandled parameter type {param_metadata[param]['type']}"
            )


def no_duplicate_rule_names():
    # Get the JSON files
    json_paths = glob.glob(os.path.join(TRANSITION_MAP_DIR, "*.json"))
    data = []
    for jp in json_paths:
        # Washer rule is a special case
        if "washer" in jp:
            continue
        with open(jp) as f:
            data.append(json.load(f))
    transitions = [rule for rules in data for rule in rules]
    rule_names = [rule["rule_name"] for rule in transitions]
    repeated_rule_names = [k for k, v in Counter(rule_names).items() if v > 1]
    if repeated_rule_names:
        raise ValueError(f"Repeated rule names: {repeated_rule_names}")
