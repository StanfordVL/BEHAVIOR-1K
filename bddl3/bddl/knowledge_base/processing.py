import csv
from collections import defaultdict, Counter
import logging
import json
import glob
import pathlib
import bddl
from bddl.object_taxonomy import ObjectTaxonomy
from bddl.activity import Conditions, get_all_activities, get_instance_count
from bddl.config import get_definition_filename
import tqdm
from bddl.knowledge_base.models import *
from bddl.knowledge_base.knowledgebase import KnowledgeBase
from bddl.knowledge_base.utils import *


BDDL_DIR = pathlib.Path(__file__).parent.parent
GENERATED_DATA_DIR = BDDL_DIR / "generated_data"

logger = logging.getLogger(__name__)


def link_many_to_many(lhs, lhs_field, rhs, rhs_field):
    getattr(lhs, lhs_field).add(rhs)
    getattr(rhs, rhs_field).add(lhs)


def link_many_to_one(child, child_field, parent, parent_field):
    setattr(child, child_field, parent)
    getattr(parent, parent_field).add(child)


def get_or_add(get_fn, add_fn, key):
    obj = get_fn(key)
    if obj is not None:
        return obj, False
    return add_fn(key), True


def tqdm_iter(iterable, verbose, *args, **kwargs):
    if verbose:
        return tqdm.tqdm(iterable, *args, **kwargs)
    else:
        return iterable


# =============================== helper functions ===============================
def preparation(verbose, load_wordnet=False):
    """
    put any preparation work (e.g. sanity check) here
    """
    logger.debug("Running preparation work...")

    if load_wordnet:
        import nltk
        nltk.download("wordnet")

    object_taxonomy = ObjectTaxonomy()

    # sanity check room types are up to date
    room_types_from_model = set([room_type for _, room_type in ROOM_TYPE_CHOICES])
    with open(GENERATED_DATA_DIR / "allowed_room_types.csv", newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        room_types_from_csv = set([row[0] for row in reader][1:])
    assert (
        room_types_from_model == room_types_from_csv
    ), "room types are not up to date with allowed_room_types.csv"

    # get object rename mapping
    object_rename_mapping = {}
    obj_rename_mapping_duplicate_set = set()
    with open(GENERATED_DATA_DIR / "object_renames.csv", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            new_cat = row["New Category"].strip()
            obj_name = row["Object name"].strip()
            # sanity checks
            if obj_name != "":
                assert (
                    len(obj_name.split("-")) == 2
                ), f"{obj_name} should only have one '-'"
                obj_id = obj_name.split("-")[1]
                if obj_id in object_rename_mapping:
                    obj_rename_mapping_duplicate_set.add(obj_id)
                object_rename_mapping[obj_id] = (
                    obj_name,
                    f"{new_cat}-{obj_id}",
                )
        assert (
            len(obj_rename_mapping_duplicate_set) == 0
        ), f"object rename mapping have duplicates: {obj_rename_mapping_duplicate_set}"

    logger.debug("Finished prep work...")
    return object_taxonomy, object_rename_mapping


def post_complete_operation():
    """
    put any post completion work (e.g. update stuff) here
    """
    # logger.debug("Running post completion operations...")
    # self.nuke_unused_synsets()
    pass

def create_synsets(kb, object_taxonomy, verbose, load_wordnet=False):
    """
    create synsets with annotations from propagated_annots_canonical.json and hierarchy from output_hierarchy.json
    """
    logger.debug("Creating synsets...")

    if load_wordnet:
        from nltk.corpus import wordnet as wn
        # Build a set of all valid WordNet synset names upfront for O(1) lookups
        valid_wn_synsets = set()
        for s in wn.all_synsets():
            valid_wn_synsets.add(s.name())
    else:
        valid_wn_synsets = None

    for synset_name in tqdm_iter(object_taxonomy.get_all_synsets(), verbose):
        if valid_wn_synsets is not None:
            synset_is_custom = synset_name not in valid_wn_synsets
            synset_definition = wn.synset(synset_name).definition() if not synset_is_custom else ""
        else:
            synset_is_custom = False
            synset_definition = ""

        synset = kb.get_synset(synset_name)
        created = synset is None
        if created:
            synset = kb.add_synset(
                name=synset_name,
                definition=synset_definition,
                is_custom=synset_is_custom,
            )
        parents = object_taxonomy.get_parents(synset_name)
        for parent in parents:
            parent_obj = kb.get_synset(parent)
            link_many_to_many(synset, "parents", parent_obj, "children")
        cur_ancestors = object_taxonomy.get_ancestors(synset_name)
        for ancestor in sorted(cur_ancestors):
            ancestor_obj = kb.get_synset(ancestor)
            link_many_to_many(synset, "ancestors", ancestor_obj, "descendants")

        # Add any categories
        for category in object_taxonomy.get_categories(synset_name):
            assert not any(
                c.name == category for c in kb.all_categories()
            ), f"Category {category} of {synset_name} already exists!"
            category_obj, _ = get_or_add(
                kb.get_category,
                lambda name: kb.add_category(name=name, synset=synset),
                category,
            )
            link_many_to_one(category_obj, "synset", synset, "categories")

        # Add any particle systems
        for particle_system in object_taxonomy.get_substances(synset_name):
            assert not any(
                ps.name == particle_system for ps in kb.all_particle_systems()
            ), f"Particle system {particle_system} of {synset_name} already exists!"
            ps_obj, _ = get_or_add(
                kb.get_particle_system,
                lambda name: kb.add_particle_system(name=name, synset=synset),
                particle_system,
            )
            link_many_to_one(ps_obj, "synset", synset, "particle_systems")

        # Add any properties
        if created:
            for property_name, params in object_taxonomy.get_abilities(
                synset_name
            ).items():
                property_obj = kb.add_property(
                    synset=synset, name=property_name, parameters=json.dumps(params)
                )
                link_many_to_one(property_obj, "synset", synset, "properties")

def create_objects(kb, object_rename_mapping, verbose):
    """
    Create objects and map to categories (with object inventory)
    """
    logger.debug("Creating objects...")
    # first get Deletion Queue
    deletion_queue = set()
    with open(GENERATED_DATA_DIR / "deletion_queue.csv", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            deletion_queue.add(row["Object"].strip().split("-")[1])
    # then create objects
    with open(GENERATED_DATA_DIR / "object_inventory.json", "r") as f:
        inventory = json.load(f)
        for orig_name, provider in tqdm_iter(inventory["providers"].items(), verbose):
            object_name = orig_name
            orig_id = orig_name.split("-")[1]
            if orig_id in object_rename_mapping:
                from_name, to_name = object_rename_mapping[orig_id]
                assert (
                    orig_name == from_name or orig_name == to_name
                ), f"Object {orig_name} is in the rename mapping with the wrong categories {from_name} -> {to_name}."
                object_name = to_name
            if orig_id in deletion_queue:
                continue

            # Create the object
            orig_category_name = orig_name.split("-")[0]
            obj = kb.add_object(
                name=orig_id,
                original_category_name=orig_category_name,
                provider=provider,
            )

            # Add bounding box info
            if orig_id in inventory["bounding_box_sizes"]:
                obj.bounding_box_size = tuple(
                    inventory["bounding_box_sizes"][orig_id]
                )

            # Add meta link info
            if orig_name in inventory["meta_links"]:
                existing_meta_types = set(inventory["meta_links"][orig_name])
                if "openfillable" in existing_meta_types:
                    existing_meta_types.add("fillable")
                for meta_link in existing_meta_types:
                    meta_link_obj, _ = get_or_add(
                        kb.get_meta_link, kb.add_meta_link, meta_link
                    )
                    link_many_to_many(obj, "meta_links", meta_link_obj, "on_objects")

                if orig_name in inventory["attachment_pairs"]:
                    existing_attachment_pairs = inventory["attachment_pairs"][
                        orig_name
                    ]
                    for gender, pairs in existing_attachment_pairs.items():
                        for pair in pairs:
                            pair_obj, _ = get_or_add(
                                kb.get_attachment_pair, kb.add_attachment_pair, pair
                            )
                            if gender == "F":
                                link_many_to_many(
                                    obj,
                                    "female_attachment_pairs",
                                    pair_obj,
                                    "female_objects",
                                )
                            elif gender == "M":
                                link_many_to_many(
                                    obj,
                                    "male_attachment_pairs",
                                    pair_obj,
                                    "male_objects",
                                )
                            else:
                                raise Exception(
                                    f"Invalid gender {gender} for attachment pair {pair}"
                                )

            # Add the category and/or particle system
            category_name = object_name.split("-")[0]
            particle_system = kb.get_particle_system(category_name)
            category = kb.get_category(category_name)
            assert (
                (category is None) != (particle_system is None)
            ), f"{category_name} should be exactly one of category or particle system"
            if particle_system is not None:
                # If it's a particle system, add it to the object
                link_many_to_one(obj, "particle_system", particle_system, "particles")
            else:
                link_many_to_one(obj, "category", category, "objects")

    # Check that all of the renames have happened
    # TODO: Is this really useful? Doubt it.
    # missing_renames = {final_name for _, final_name in object_rename_mapping.values() if not Object.exists(name=final_name.split("-")[1])}
    # assert len(missing_renames) == 0, f"{missing_renames} do not exist in the database. Did you rename a nonexistent object (or one in the deletion queue)?"
    return deletion_queue

def add_particle_system_parameters(kb):
    with open(GENERATED_DATA_DIR / "substance_hyperparams.csv") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=",", quotechar='"')
        for row in reader:
            # Skip the row if it is marked for pruning
            if int(row["prune"]) == 1:
                continue

            # Get the particle system. It should already exist from the synset stage.
            name = row["substance"]
            particle_system = kb.get_particle_system(name)
            assert (
                particle_system is not None
            ), f"Particle system {name} does not exist in hierarchy."

            # Confirm the synset assignment
            synset_name = row["synset"]
            synset = kb.get_synset(synset_name)
            assert (
                particle_system.synset == synset
            ), f"Particle system {name} is not in the correct synset {synset_name}"

            # Load, and re-dump, the parameters
            params = json.loads(row["hyperparams"])
            particle_system.parameters = json.dumps(params)

def create_scenes(kb, object_rename_mapping, deletion_queue, verbose):
    """
    create scene objects (which stores the room config)
    scene matching to tasks will be generated later when creating task objects
    """
    logger.debug("Creating scenes...")
    with open(GENERATED_DATA_DIR / "combined_room_object_list.json", "r") as f:
        planned_scene_dict = json.load(f)["scenes"]
        for scene_name in tqdm_iter(planned_scene_dict, verbose):
            scene, _ = get_or_add(kb.get_scene, kb.add_scene, scene_name)
            for room_name in planned_scene_dict[scene_name]:
                try:
                    room = kb.add_room(
                        name=room_name,
                        type=room_name.rsplit("_", 1)[0],
                        scene=scene,
                    )
                    link_many_to_one(room, "scene", scene, "rooms")
                except ValueError:
                    raise Exception(
                        f"room {room_name} in {scene.name} (not ready) could not be added - maybe it already exists?"
                    )
                for orig_name, count in planned_scene_dict[scene_name][
                    room_name
                ].items():
                    orig_id = orig_name.split("-")[1]
                    if orig_id not in deletion_queue:
                        if orig_id in object_rename_mapping:
                            from_name, to_name = object_rename_mapping[orig_id]
                            assert (
                                orig_name == from_name or orig_name == to_name
                            ), f"Object {orig_name} is in the rename mapping with the wrong categories {from_name} -> {to_name}."
                        obj = kb.get_object(orig_id)
                        assert (
                        obj is not None
                        ), f"Scene {scene_name} object {orig_id} does not exist in the database."
                        room_object = kb.add_room_object(room=room, object=obj, count=count)
                        link_many_to_one(room_object, "room", room, "roomobjects")
                        link_many_to_one(room_object, "object", obj, "roomobjects")

def create_tasks(kb, verbose, load_wordnet=False):
    """
    create tasks and map to synsets
    """
    logger.debug("Creating tasks...")
    tasks = glob.glob(str(BDDL_DIR / "activity_definitions/*"))
    tasks = [
        (act, inst)
        for act in get_all_activities()
        for inst in range(get_instance_count(act))
    ]
    for act, inst in tqdm_iter(tasks, verbose):
        # Load task definition
        conds = Conditions(act, inst, "behavior-1k")
        synsets = set(
            synset for synset in conds.parsed_objects if synset != "agent.n.01"
        )
        canonicalized_synsets = set(canonicalize(synset) for synset in synsets) if load_wordnet else synsets
        with open(get_definition_filename(act, inst), "r") as f:
            raw_task_definition = "".join(f.readlines())

        initial_conds, goal_conds = get_initial_and_goal_conditions(conds)
        combined_conds = initial_conds + goal_conds

        # Create task object
        task_name = f"{act}-{inst}"
        task = kb.add_task(name=task_name, definition=raw_task_definition)
        for predicate in all_task_predicates(combined_conds):
            pred_obj, _ = get_or_add(
                kb.get_predicate_usage, kb.add_predicate_usage, predicate
            )
            link_many_to_many(task, "uses_predicates", pred_obj, "tasks")

        # add any synset that is not currently in the database
        for synset_name in sorted(canonicalized_synsets):
            is_used_as_non_substance, is_used_as_substance = object_substance_match(
                combined_conds, synset_name
            )
            is_used_as_fillable = object_used_as_fillable(
                combined_conds, synset_name
            )
            # all annotated synsets have been created before, so any newly created synset is illegal
            synset = kb.get_synset(synset_name)
            assert (
                synset is not None
            ), f"Synset {synset_name} used by task {task_name} does not exist in the database."
            synset.is_used_as_substance = (
                synset.is_used_as_substance or is_used_as_substance
            )
            synset.is_used_as_non_substance = (
                synset.is_used_as_non_substance or is_used_as_non_substance
            )
            synset.is_used_as_fillable = (
                synset.is_used_as_fillable or is_used_as_fillable
            )
            synset_used_predicates = object_used_predicates(
                combined_conds, synset_name
            )
            for predicate in synset_used_predicates:
                pred_obj, _ = get_or_add(
                        kb.get_predicate_usage, kb.add_predicate_usage, predicate
                )
                link_many_to_many(synset, "used_in_predicates", pred_obj, "synsets")
            link_many_to_many(task, "synsets", synset, "tasks")

            # If the synset ever shows up as future or real, check validity
            used_as_future_or_real = (
                "future" in synset_used_predicates
                or "real" in synset_used_predicates
            )
            if used_as_future_or_real:
                # Assert that it's used as future in initial and as real in goal
                initial_preds = object_used_predicates(initial_conds, synset_name)
                if "future" not in initial_preds:
                    logger.debug(
                        "Synset %s is not used as future in initial in %s",
                        synset_name,
                        task_name,
                    )
                if "real" in initial_preds:
                    raise ValueError(
                        f"Synset {synset_name} is used as real in initial in {task_name}"
                    )

                goal_preds = object_used_predicates(goal_conds, synset_name)
                if "real" not in goal_preds:
                    logger.debug(
                        "Synset %s is not used as real in goal in %s",
                        synset_name,
                        task_name,
                    )
                if "future" in goal_preds:
                    raise ValueError(
                        f"Synset {synset_name} is used as future in goal in {task_name}"
                    )

                # We only add it if it's used in the initial predicates. Sometimes things will be real()
                # in the goal but they will already exist in the initial, and the real is just being
                # used to say that the object is not entirely used up during the transition.
                if "future" in initial_preds:
                    link_many_to_many(
                        task, "future_synsets", synset, "tasks_using_as_future"
                    )

        # generate room requirements for task
        room_synset_requirements = defaultdict(Counter)  # room[synset] = count
        for cond in leaf_inroom_conds(conds.parsed_initial_conditions, synsets):
            assert len(cond) == 2, f"{task_name}: {str(cond)} not in correct format"
            rr_type = cond[1]
            rr_synset = cond[0]
            room_synset_requirements[rr_type][rr_synset] += 1

        for rr_type, synset_counter in room_synset_requirements.items():
            room_requirement = kb.add_room_requirement(task=task, type=rr_type)
            link_many_to_one(room_requirement, "task", task, "room_requirements")
            for rsr_synset, count in synset_counter.items():
                rsr_synset_obj = kb.get_synset(rsr_synset)
                rsr_obj = kb.add_roomsynset_requirement(
                    room_requirement=room_requirement,
                    synset=rsr_synset_obj,
                    count=count,
                )
                link_many_to_one(
                    rsr_obj,
                    "room_requirement",
                    room_requirement,
                    "roomsynsetrequirements",
                )
                link_many_to_one(
                    rsr_obj, "synset", rsr_synset_obj, "roomsynsetrequirements"
                )

def create_transitions(kb, verbose):
    from bddl.transition_rules import (
        load_cooking_recipes,
        load_machine_recipes,
        load_mixing_recipes,
        load_substance_cooking_recipes,
        load_washer_rule,
    )

    # Load the transition data from JSON
    json_paths = glob.glob(
        str(GENERATED_DATA_DIR / "transition_map/tm_jsons/*.json")
    )
    transitions = []
    for jp in json_paths:
        # This file is in a different format and not relevant.
        if jp.endswith("washer.json"):
            continue
        with open(jp) as f:
            transitions.extend(json.load(f))

    # Create the transition rule objects with synset linkages
    for transition_data in tqdm_iter(transitions, verbose):
        rule_name = transition_data["rule_name"]
        transition = kb.add_transition_rule(name=rule_name)

        # Add the default inputs and outputs
        inputs = set(transition_data["input_synsets"].keys())
        outputs = set(transition_data["output_synsets"].keys())

        # Add the washer rules' washed item both to inputs and outputs
        if "washed_item" in transition_data:
            washed_items = set(transition_data["washed_item"].keys())
            inputs.update(washed_items)
            outputs.update(washed_items)

        assert inputs, f"Transition {transition.name} has no inputs!"
        assert outputs, f"Transition {transition.name} has no outputs!"

        for synset_name in inputs:
            synset = kb.get_synset(synset_name)
            link_many_to_many(
                transition, "input_synsets", synset, "used_by_transition_rules"
            )
        for synset_name in outputs:
            synset = kb.get_synset(synset_name)
            link_many_to_many(
                transition,
                "output_synsets",
                synset,
                "produced_by_transition_rules",
            )
        for auxiliary_synset_type in ["machine", "heat_source", "container"]:
            if auxiliary_synset_type in transition_data:
                machines = transition_data[auxiliary_synset_type]
                for synset_name in machines:
                    machine = kb.get_synset(synset_name)
                    link_many_to_many(
                        transition,
                        "machine_synsets",
                        machine,
                        "machine_in_transition_rules",
                    )

    # Attach typed recipe dataclasses to the transition rules by name
    all_typed_recipes = (
        list(load_cooking_recipes())
        + list(load_mixing_recipes())
        + list(load_machine_recipes())
        + list(load_substance_cooking_recipes())
    )
    for recipe in all_typed_recipes:
        tr = kb.get_transition_rule(recipe.name)
        if tr is not None:
            tr.recipe = recipe

    # Store the washer rule on the KB directly
    kb.washer_rule = load_washer_rule()

def create_complaints(kb):
    with open(GENERATED_DATA_DIR / "complaints.json", "r") as f:
        complaints = json.load(f)

    for complaint in complaints:
        complaint_type_name = complaint["type"]
        complaint_model_id = complaint["object"].split("-")[1]
        complaint_additional_info = complaint["additional_info"]
        complaint_response = complaint["complaint"]

        # Create the relevant complaint type (even if we are processed we want to show all types)
        complaint_type, created = get_or_add(
            kb.get_complaint_type, kb.add_complaint_type, complaint_type_name
        )

        # Skip processed complaints
        if complaint["processed"]:
            continue

        # Check if the model ID exists
        obj = kb.get_object(complaint_model_id)
        if obj is None:
            logger.debug(
                f"Complained object {complaint_model_id} does not exist in the database. Skipping."
            )
            continue

        complaint_obj = kb.add_complaint(
            object=obj,
            complaint_type=complaint_type,
            prompt_additional_info=complaint_additional_info,
            response=complaint_response,
        )
        obj.complaints.add(complaint_obj)
        complaint_type.complaints.add(complaint_obj)

def nuke_unused_synsets(kb):
    # Make repeated passes until we propagate far enough up
    while True:
        removal_names = set()
        for synset in kb.all_synsets():
            # In a given pass, only leaf nodes can be removed
            if synset.children:
                continue

            # If a synset has objects or task relevance, we can't remove it
            if len(synset.matching_objects) != 0:
                continue

            if synset.n_task_required != 0:
                continue

            if synset.used_by_transition_rules:
                continue

            if synset.produced_by_transition_rules:
                continue

            # Otherwise queue it for removal
            removal_names.add(synset.name)

        if removal_names:
            for s in list(kb.all_synsets()):
                if s.name in removal_names:
                    s.delete()
        else:
            break


def build_knowledgebase(verbose=True, load_wordnet=False):
    kb = KnowledgeBase()
    populate_knowledgebase(kb, verbose=verbose, load_wordnet=load_wordnet)
    kb.sort_all()
    return kb


def populate_knowledgebase(kb: KnowledgeBase, verbose=True, load_wordnet=False):
    logger.info("Loading BDDL knowledge base... This may take a few seconds.")
    object_taxonomy, object_rename_mapping = preparation(verbose, load_wordnet=load_wordnet)
    create_synsets(kb, object_taxonomy, verbose, load_wordnet=load_wordnet)
    add_particle_system_parameters(kb)
    deletion_queue = create_objects(kb, object_rename_mapping, verbose)
    create_scenes(kb, object_rename_mapping, deletion_queue, verbose)
    create_tasks(kb, verbose, load_wordnet=load_wordnet)
    create_transitions(kb, verbose)
    create_complaints(kb)
    post_complete_operation()
    kb.sort_all()
    return kb


if __name__ == "__main__":
    import IPython

    build_knowledgebase(verbose=True)
    IPython.embed()
