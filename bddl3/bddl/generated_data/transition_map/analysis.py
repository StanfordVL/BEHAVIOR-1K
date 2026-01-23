import csv
import os

from tm_submap_params import TM_SUBMAPS_TO_PARAMS

for submap_fn in os.listdir("tm_raw_data_old"):
    with open("tm_raw_data_old/" + submap_fn, "r") as f:
        reader = csv.DictReader(f)
        rules = [rec for rec in reader]
    with open("tm_raw_data/" + submap_fn, "w") as f:
        fieldnames = ["rule_name"] + list(
            TM_SUBMAPS_TO_PARAMS[submap_fn.split(".csv")[0]].keys()
        )
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        updated_rules = []
        for rule in rules:
            updated_rule = {}
            for param, value in rule.items():
                # if param == "input_synsets":
                #     print(value)
                # import sys; sys.exit()
                if param not in fieldnames:
                    continue
                if param == "input_synsets":
                    input_objects = []
                    input_states = []
                    atoms = value.split(" and ")
                    # print(atoms); import sys; sys.exit()
                    for atom in atoms:
                        if "real " in atom:
                            input_objects.append(atom.split(" ")[-1])
                        else:
                            input_states.append(atom)
                    updated_rule["input_synsets"] = " and ".join(input_objects)
                    updated_rule["input_states"] = " and ".join(input_states)
                elif param == "output_synsets":
                    output_objects = []
                    output_states = []
                    atoms = value.split(" and ")
                    for atom in atoms:
                        if "real " in atom:
                            output_objects.append(atom.split(" ")[-1])
                        else:
                            output_states.append(atom)
                    updated_rule["output_synsets"] = " and ".join(output_objects)
                    updated_rule["output_states"] = " and ".join(output_states)
                elif param == "machine":
                    pass
                else:
                    updated_rule[param] = value
            writer.writerow(updated_rule)
