import os
from bddl.bddl_verification import *


def verify_tms(): 
    with open(SYNS_TO_PROPS_JSON, "r") as f:
        syns_to_props = json.load(f)
    *__, domain_predicates = parse_domain("omnigibson")
    for submap in TM_SUBMAPS_TO_PARAMS:
        print(submap)
        print()
        with open(os.path.join(TRANSITION_MAP_DIR, submap + ".json"), "r") as f:
            submap_rules = json.load(f)
        for rule in submap_rules:
            no_missing_required_params(rule, submap)
            no_incorrectly_formatted_params(rule, submap)
            no_invalid_synsets_tm(rule, submap, syns_to_props)
            no_invalid_predicates_tm(rule, submap, domain_predicates)
            no_misaligned_synsets_predicates_tm(rule, submap, syns_to_props)
            no_substances_with_multiple_instances_tm(rule, submap, syns_to_props)

    no_duplicate_rule_names()
    

if __name__ == "__main__":
    verify_tms()