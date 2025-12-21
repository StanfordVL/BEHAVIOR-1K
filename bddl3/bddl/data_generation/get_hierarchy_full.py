"""
Iteration one:
Strict multi-way tree (without no count)
We consider each unique path to a synset "different", with the obvious
problem that each synset can have multiple hypernyms
"""

import collections
import json
import pandas as pd
import pathlib
from nltk.corpus import wordnet as wn

HIERARCHY_OUTPUT_FN = (
    pathlib.Path(__file__).parents[1] / "generated_data" / "output_hierarchy.json"
)
HIERARCHY_PROPERTIES_OUTPUT_FN = (
    pathlib.Path(__file__).parents[1]
    / "generated_data"
    / "output_hierarchy_properties.json"
)
CATEGORY_MAPPING_FN = (
    pathlib.Path(__file__).parents[1] / "generated_data" / "category_mapping.csv"
)
SUBSTANCE_MAPPING_FN = (
    pathlib.Path(__file__).parents[1] / "generated_data" / "substance_hyperparams.csv"
)
SYN_PROP_PARAM_FN = (
    pathlib.Path(__file__).parents[1]
    / "generated_data"
    / "propagated_annots_params.json"
)


def add_igibson_objects(node, synset_to_cat, synset_to_substance):
    """
    Go through the hierarchy and add the words associated with the synsets as attributes.
    """
    if node["name"] in synset_to_cat:
        node["categories"] = sorted(synset_to_cat[node["name"]])
    if node["name"] in synset_to_substance:
        node["substances"] = sorted(synset_to_substance[node["name"]])

    if "children" in node:
        for child_node in node["children"]:
            add_igibson_objects(child_node, synset_to_cat, synset_to_substance)


def add_path(path, hierarchy):
    if not path:
        return
    synset = path[-1]
    if "children" not in hierarchy:
        hierarchy["children"] = []
    for subhierarchy in hierarchy["children"]:
        if subhierarchy["name"] == synset:
            subhierarchy["hasModel"] = True
            add_path(path[:-1], subhierarchy)
            return
    hierarchy["children"].append({"name": synset, "hasModel": True})
    add_path(path[:-1], hierarchy["children"][-1])


def generate_paths(paths, path, synset, syn_prop_dict):
    # Annotated as not custom, or not present (i.e. not leaf, i.e. not custom)
    if (synset not in syn_prop_dict) or (not syn_prop_dict[synset]["is_custom"]):
        hypernyms = [
            x.name() for x in wn.synset(synset).hypernyms()
        ]  # This fails iff a custom synset is incorrectly stored as non-custom
    else:
        hypernyms = syn_prop_dict[synset]["hypernyms"].split(",")

    if not hypernyms:
        paths.append(path)
    else:
        for hypernym in hypernyms:
            generate_paths(paths, path + [hypernym], hypernym, syn_prop_dict)


def add_properties(node, syn_prop_param_dict):
    node["abilities"] = syn_prop_param_dict[node["name"]]
    if "children" in node:
        for child_node in node["children"]:
            add_properties(child_node, syn_prop_param_dict)


# API


def get_hierarchy(syn_prop_dict):
    hierarchy = {"name": "entity.n.01", "children": [], "hasModel": False}
    objects = set()
    for synset in syn_prop_dict.keys():
        # NOTE skip custom synsets without hypernym annotations
        if (
            syn_prop_dict[synset]["is_custom"]
            and syn_prop_dict[synset]["hypernyms"] == ""
        ):
            continue
        objects.add(synset)
        paths = []
        generate_paths(paths, [synset], synset, syn_prop_dict)
        for path in paths:
            add_path(path[:-1], hierarchy)

    # Sort the hierarchy's children by their names
    def _sort_children(node):
        if "children" in node:
            node["children"].sort(key=lambda x: x["name"])
            for child in node["children"]:
                _sort_children(child)

    _sort_children(hierarchy)

    synset_to_cat_raw = pd.read_csv(CATEGORY_MAPPING_FN)[
        ["category", "synset"]
    ].to_dict(orient="records")
    synset_to_cat = collections.defaultdict(list)
    for rec in synset_to_cat_raw:
        synset_to_cat[rec["synset"]].append(rec["category"])

    synset_to_substance_raw = pd.read_csv(SUBSTANCE_MAPPING_FN)[
        ["substance", "synset", "prune"]
    ].to_dict(orient="records")
    synset_to_substance = collections.defaultdict(list)
    for rec in synset_to_substance_raw:
        if int(rec["prune"]) == 1:
            continue
        synset_to_substance[rec["synset"]].append(rec["substance"])

        # If the synset also has a category with the same name, remove it from the category list.
        # It's just a placeholder category for particles of the substance.
        if (
            rec["synset"] in synset_to_cat
            and rec["substance"] in synset_to_cat[rec["synset"]]
        ):
            synset_to_cat[rec["synset"]].remove(rec["substance"])

    add_igibson_objects(hierarchy, synset_to_cat, synset_to_substance)

    with open(HIERARCHY_OUTPUT_FN, "w") as f:
        json.dump(hierarchy, f, indent=2)
    return hierarchy


def create_get_save_hierarchy_with_properties(hierarchy):
    print("Adding params to hierarchy file...")
    with open(SYN_PROP_PARAM_FN) as f:
        syn_prop_param_dict = json.load(f)
    add_properties(hierarchy, syn_prop_param_dict)
    with open(HIERARCHY_PROPERTIES_OUTPUT_FN, "w") as f:
        json.dump(hierarchy, f, indent=2)
    print("Added params to hierarchy file, saved.")
    return hierarchy


if __name__ == "__main__":
    pass

"""
Graph using http://bl.ocks.org/eyaler/10586116#graph.json
"""
