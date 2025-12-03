import json
import csv
import pandas as pd
import pathlib
import os
from collections import Counter
import re

from bddl.data_generation.tm_submap_params import TM_SUBMAPS_TO_PARAMS

SHEETS_DIR = (
    pathlib.Path(__file__).parents[1]
    / "generated_data"
    / "transition_map"
    / "tm_raw_data"
)
JSONS_DIR = (
    pathlib.Path(__file__).parents[1] / "generated_data" / "transition_map" / "tm_jsons"
)

OBJECT_CAT_AND_INST_RE = r"[A-Za-z-_]+\.n\.[0-9]+"


def parse_conditions_entry(unparsed_conditions, syns_to_props):
    # print(f"Parsing: {unparsed_conditions}")
    if unparsed_conditions.isnumeric():
        always_true = bool(int(unparsed_conditions))
        conditions = [] if always_true else None
    else:
        conditions = [
            x for x in unparsed_conditions.lower().split(" or ") if x in syns_to_props
        ]
    return conditions


def sheet_to_json(submap):
    params = TM_SUBMAPS_TO_PARAMS[submap]
    raw_data = pd.read_csv(os.path.join(SHEETS_DIR, submap + ".csv"))[
        list(params.keys()) + ["prune"]
    ].to_json(orient="records")
    data = json.loads(raw_data)
    reformatted_data = []
    for rule in data:
        if int(rule["prune"]) == 1:
            continue
        rule.pop("prune")
        reformatted_rule = {}
        for param, value in rule.items():
            if (
                TM_SUBMAPS_TO_PARAMS[submap][param]["type"] == "synset"
                and value is not None
            ):
                value = value.split(" and ")
                value = Counter(
                    [re.match(OBJECT_CAT_AND_INST_RE, atom).group() for atom in value]
                )
            elif (
                TM_SUBMAPS_TO_PARAMS[submap][param]["type"] == "atom"
                and value is not None
            ):
                value = value.split(" and ")
                unary_atoms = []
                binary_atoms = []
                for atom in value:
                    elements = atom.split(" ")
                    assert len(elements) in range(2, 5), f"Malformed atom {atom}"
                    if len(elements) == 3:
                        if elements[0] == "not":
                            unary_atoms.append(elements)
                        else:
                            binary_atoms.append(elements)
                    elif len(elements) == 2:
                        unary_atoms.append(elements)
                    else:
                        binary_atoms.append(elements)
                reformatted_atoms = {}
                for atom in unary_atoms:
                    synset = re.match(OBJECT_CAT_AND_INST_RE, atom[-1]).group()
                    if synset in reformatted_rule:
                        reformatted_atoms[synset].append((atom[-2], atom[0] != "not"))
                    else:
                        reformatted_atoms[synset] = [(atom[-2], atom[0] != "not")]
                for atom in binary_atoms:
                    synset1, synset2 = (
                        re.match(OBJECT_CAT_AND_INST_RE, atom[-2]).group(),
                        re.match(OBJECT_CAT_AND_INST_RE, atom[-1]).group(),
                    )
                    if f"{synset1},{synset2}" in reformatted_rule:
                        reformatted_atoms[f"{synset1},{synset2}"].append(
                            (atom[-3], atom[0] != "not")
                        )
                    else:
                        reformatted_atoms[f"{synset1},{synset2}"] = [
                            (atom[-3], atom[0] != "not")
                        ]
                value = reformatted_atoms
            elif TM_SUBMAPS_TO_PARAMS[submap][param]["type"] == "string":
                value = value
            elif value is None:
                value = None
            else:
                raise ValueError(
                    f"Unhandled parameter type {TM_SUBMAPS_TO_PARAMS[submap][param]['type']}"
                )
            reformatted_rule[param] = value
        reformatted_data.append(reformatted_rule)

    with open(os.path.join(JSONS_DIR, submap + ".json"), "w") as f:
        json.dump(reformatted_data, f, indent=4)


def sheet_to_json_washer(syns_to_props):
    records = []
    with open(os.path.join(SHEETS_DIR, "washer.csv")) as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    assert len(records) == 1
    record = records[0]

    remover_kwargs = dict()

    # Iterate through all the columns headed by a substance, in no particular order since their ultimate location is a dict
    for dirtiness_substance_synset in [
        key for key in record if re.match(OBJECT_CAT_AND_INST_RE, key) is not None
    ]:
        if dirtiness_substance_synset not in syns_to_props:
            continue
        conditions = parse_conditions_entry(
            record[dirtiness_substance_synset], syns_to_props
        )
        remover_kwargs[dirtiness_substance_synset] = conditions

    with open(os.path.join(JSONS_DIR, "washer.json"), "w") as f:
        json.dump(remover_kwargs, f, indent=4)


def create_save_explicit_transition_rules(syns_to_props):
    print("Creating explicit transition rule jsons...")
    for submap in TM_SUBMAPS_TO_PARAMS:
        sheet_to_json(submap)

    sheet_to_json_washer(syns_to_props)
    print("Created and saved explicit transition rule jsons.")


if __name__ == "__main__":
    for submap in TM_SUBMAPS_TO_PARAMS:
        sheet_to_json(submap)
