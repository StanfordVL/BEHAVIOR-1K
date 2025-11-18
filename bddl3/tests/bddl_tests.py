import json
import os 
import sys
import bddl.bddl_verification as ver
import bddl.parsing as parse


# MAIN 

def verify_definition(activity, syns_to_props, domain_predicates, csv=False):
    defn_fn = os.path.join(ver.PROBLEM_FILE_DIR, activity, "problem0.bddl")
    with open(defn_fn, "r") as f:
        defn = f.read() 
    __, objects, init, goal = ver._get_defn_elements_from_file(activity)
    ver.object_list_correctly_formatted(defn)
    ver.all_objects_appropriate(objects, init, goal)
    ver.all_objects_placed(init)
    ver.future_and_real_present(objects, init, goal)
    ver.no_repeat_object_lines(defn)
    ver.no_qmarks_in_init(defn)
    ver.no_contradictory_init_atoms(init)
    ver.no_invalid_synsets(objects, init, goal, syns_to_props)
    ver.no_invalid_predicates(init, goal, domain_predicates)
    ver.no_uncontrolled_category(activity, defn)
    ver.no_misaligned_synsets_predicates(init, goal, syns_to_props)
    ver.no_unnecessary_specific_containers(objects, init, goal, syns_to_props)
    ver.no_substances_with_multiple_instances(objects, syns_to_props)
    ver.agent_present(init)
    ver.problem_name_correct(activity)


# Master planning sheet
def batch_verify(): 
    with open(ver.SYNS_TO_PROPS_JSON, "r") as f:
        syns_to_props = json.load(f) 
    *__, domain_predicates = parse.parse_domain("omnigibson")
    for activity in sorted(os.listdir(ver.PROBLEM_FILE_DIR)):
        if "-" in activity: continue
        if not os.path.isdir(os.path.join(ver.PROBLEM_FILE_DIR, activity)): continue
        print()
        print(activity)
        verify_definition(activity, syns_to_props, domain_predicates, csv=False)


def main():
    if sys.argv[1] == "verify": 
        verify_definition(sys.argv[2])
    
    elif sys.argv[1] == "batch_verify": 
        batch_verify()


if __name__ == "__main__":
    main()
