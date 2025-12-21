import pathlib
from bddl.data_generation.get_hierarchy_full import (
    get_hierarchy,
    create_get_save_hierarchy_with_properties,
)
from bddl.data_generation.get_syn_prop_annots_canonical import (
    create_get_save_annots_canonical,
    create_get_save_properties_to_synsets,
    create_get_save_synsets_to_descriptors,
)
from bddl.data_generation.propagate_by_intersection import (
    create_get_save_propagated_canonical,
)
from bddl.data_generation.process_prop_param_annots import (
    create_get_save_propagated_annots_params,
)
from bddl.data_generation.get_implicit_transition_rules import (
    create_get_save_implicit_transition_rules,
)
from bddl.data_generation.get_explicit_transition_rules import (
    create_save_explicit_transition_rules,
)
from bddl.data_generation.sanity_check import sanity_check
import pandas as pd
import csv
import nltk

"""
Inputs:
    ../house_room_info/currently_assigned_activities.json
    synset_masterlist.tsv
    TODO complete
"""
SYN_PROP_DATA_FN = pathlib.Path(__file__).parents[1] / "generated_data" / "synsets.csv"
# # Get owned models


def main():
    nltk.download("wordnet")

    # Get full hierarchy (it's created and saved on import)
    with open(SYN_PROP_DATA_FN, "r") as f:
        syn_prop_dict = {}
        for row in csv.DictReader(f):
            if int(row["prune"]) == 1:
                continue
            syn_prop_dict[row.pop("synset")] = row
    hierarchy = get_hierarchy(syn_prop_dict)
    syn_prop_df = pd.read_csv(SYN_PROP_DATA_FN)

    # Create, get, and save master canonical
    annots_canonical = create_get_save_annots_canonical(syn_prop_dict)
    # Create, get, and save final canonical from hierarchy, master_canonical
    propagated_canonical = create_get_save_propagated_canonical(
        hierarchy, annots_canonical
    )

    # Create and save properties_to_synsets from propagated canonical
    props_to_syns = create_get_save_properties_to_synsets(propagated_canonical)

    # Create and save synsets_to_descriptors from propagated canonical
    create_get_save_synsets_to_descriptors(propagated_canonical)

    # Add parameter info to syns-to-props
    syns_to_param_props = create_get_save_propagated_annots_params(propagated_canonical)

    # Add prop-param info to hierarchy
    create_get_save_hierarchy_with_properties(hierarchy)

    # Create and save implicit transition jsons
    create_get_save_implicit_transition_rules(syns_to_param_props, props_to_syns)

    # Create and save explicit transition jsons
    create_save_explicit_transition_rules(syns_to_param_props)

    # Sanity check object taxonomy and transition rules
    sanity_check()

    # # Create and save activity-specific hierarchies (no getting because that will get complicated)
    # create_save_activity_specific_hierarchies()

    # # Create and save single dict of all activity-specific hierarchies. Since the other script is run before it, the files will be updated
    # create_save_all_activity_hierarchy_dict()


if __name__ == "__main__":
    main()
