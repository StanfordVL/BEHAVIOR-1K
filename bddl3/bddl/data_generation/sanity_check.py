import pathlib
import json
import os
from collections.abc import Iterable
from bddl.object_taxonomy import ObjectTaxonomy

TRANSITION_RULE_FOLDER = (
    pathlib.Path(__file__).parents[1] / "generated_data" / "transition_map" / "tm_jsons"
)
SYNSET_KEYS = [
    "machine",
    "container",
    "washed_item",
    "heat_source",
    "input_synsets",
    "output_synsets",
]
MACHINE_KEY = ["machine"]
CONTAINER_KEY = ["container", "machine"]


def sanity_check_object_hierarchy(object_taxonomy):
    leaf_synsets = object_taxonomy.get_leaf_descendants("entity.n.01")
    leaf_substance_synsets = {
        synset
        for synset in leaf_synsets
        if object_taxonomy.has_ability(synset, "substance")
    }
    for s in leaf_substance_synsets:
        substances = object_taxonomy.get_substances(s)
        assert (
            len(substances) == 1
        ), f"Substance synset {s} is mapped to zero or more than one substance: {substances}."

    for s in leaf_synsets:
        abilities = object_taxonomy.get_abilities(s)
        for ability in [
            "particleApplier",
            "particleRemover",
            "particleSource",
            "particleSink",
        ]:
            if ability in abilities:
                ability_params = abilities[ability]
                for substance_synset in ability_params["conditions"]:
                    assert (
                        substance_synset in leaf_substance_synsets
                    ), f"In ParticleModifier annotation, {substance_synset} is not a leaf substance synset."
                    conditions = ability_params["conditions"][substance_synset]
                    if conditions is not None:
                        for condition in conditions:
                            if condition[0] == "saturated":
                                assert (
                                    condition[1] in leaf_substance_synsets
                                ), f"In ParticleModifier annotation, {condition[1]} is not a leaf substance synset."


def sanity_check_transition_rules(object_taxonomy):
    leaf_synsets = object_taxonomy.get_leaf_descendants("entity.n.01")
    valid_keys = set()
    for json_file in os.listdir(TRANSITION_RULE_FOLDER):
        with open(TRANSITION_RULE_FOLDER / json_file, "r") as f:
            transition_rule = json.load(f)
            for rule in transition_rule:
                for key in rule:
                    if key in SYNSET_KEYS:
                        val = rule[key]
                        for s in val:
                            assert (
                                s in leaf_synsets
                            ), f"In transition rule file {json_file}, rule {rule}, {s} is not a leaf synset."

                    if key in MACHINE_KEY:
                        val = rule[key]
                        for s in val:
                            assert object_taxonomy.has_ability(
                                s, "toggleable"
                            ), f"In transition rule file {json_file}, rule {rule}, {val} is not a toggleable machine."
                            assert object_taxonomy.has_ability(
                                s, "fillable"
                            ), f"In transition rule file {json_file}, rule {rule}, {val} is not a fillable machine."

                    if key in CONTAINER_KEY:
                        val = rule[key]
                        for s in val:
                            assert object_taxonomy.has_ability(
                                s, "fillable"
                            ), f"In transition rule file {json_file}, rule {rule}, {val} is not a fillable container."


def sanity_check_transition_rules_washer(object_taxonomy):
    leaf_synsets = object_taxonomy.get_leaf_descendants("entity.n.01")
    with open(TRANSITION_RULE_FOLDER / "washer.json", "r") as f:
        transition_rule = json.load(f)
    for system_synset, conditions in transition_rule.items():
        assert (
            system_synset in leaf_synsets
        ), f"In washer transition rule, {system_synset} is not a leaf synset."
        if conditions is not None:
            for cleanser_synset in conditions:
                assert (
                    cleanser_synset in leaf_synsets
                ), f"In washer transition rule, {cleanser_synset} is not a leaf synset."


def sanity_check():
    object_taxonomy = ObjectTaxonomy()
    # Needs to refresh to use the latest version of the taxonomy
    object_taxonomy.refresh_hierarchy_file()
    sanity_check_object_hierarchy(object_taxonomy)
    sanity_check_transition_rules(object_taxonomy)
    sanity_check_transition_rules_washer(object_taxonomy)


if __name__ == "__main__":
    sanity_check()
