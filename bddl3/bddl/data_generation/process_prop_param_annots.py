import csv
from enum import IntEnum
import json
import os
import re
import pandas as pd
import pathlib


OBJECT_INSTANCE_RE = r"[A-Za-z-_]+\.n\.[0-9]+_[0-9]+"
OBJECT_CAT_RE = r"[A-Za-z-_]+\.n\.[0-9]+$"
OBJECT_CAT_AND_INST_RE = r"[A-Za-z-_]+\.n\.[0-9]+"

SYNS_TO_PROPS = (
    pathlib.Path(__file__).parents[1]
    / "generated_data"
    / "propagated_annots_canonical.json"
)
PROP_PARAM_ANNOTS_DIR = (
    pathlib.Path(__file__).parents[1] / "generated_data" / "prop_param_annots"
)
PARAMS_OUTFILE_FN = (
    pathlib.Path(__file__).parents[1]
    / "generated_data"
    / "propagated_annots_params.json"
)
REMOVER_SYNSET_MAPPING = (
    pathlib.Path(__file__).parents[1] / "generated_data" / "remover_synset_mapping.json"
)
LEAF_SYNSETS_FILE = pathlib.Path(__file__).parents[1] / "generated_data" / "synsets.csv"

# Helpers
LEAF_SYNSETS_SET = set(pd.read_csv(LEAF_SYNSETS_FILE)["synset"])


# Specific condition types for applying / removing particles
class ParticleModifyCondition(IntEnum):
    FUNCTION = 0
    SATURATED = 1
    TOGGLEDON = 2
    GRAVITY = 3


PREDICATE_MAPPING = {
    "saturated": ParticleModifyCondition.SATURATED,
    "toggled_on": ParticleModifyCondition.TOGGLEDON,
    "function": ParticleModifyCondition.FUNCTION,
}


def parse_predicate(condition, syns_to_props):
    predicate = condition.split(" ")[0]
    pred_type = PREDICATE_MAPPING[predicate]
    if pred_type == ParticleModifyCondition.SATURATED:
        saturated_synset = condition.split(" ")[1]
        if saturated_synset not in syns_to_props:
            return None
        cond = (predicate, saturated_synset)
    elif pred_type == ParticleModifyCondition.TOGGLEDON:
        cond = (predicate, True)
    elif pred_type == ParticleModifyCondition.FUNCTION:
        raise ValueError("Not supported")
    else:
        raise ValueError(f"Unsupported condition type: {pred_type}")
    return cond


def parse_conditions_entry(unparsed_conditions, syns_to_props):
    # print(f"Parsing: {unparsed_conditions}")
    if unparsed_conditions.isnumeric():
        always_true = bool(int(unparsed_conditions))
        conditions = [] if always_true else None
    else:
        conditions = [
            parse_predicate(cond, syns_to_props)
            for cond in unparsed_conditions.lower().split(" or ")
        ]
        conditions = [cond for cond in conditions if cond is not None]
    return conditions


# particleRemovers


def get_synsets_to_particle_remover_params(syns_to_props):
    synset_cleaning_mapping = {}
    records = []
    with open(os.path.join(PROP_PARAM_ANNOTS_DIR, "particleRemover.csv"), "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    for record in records:
        synset = re.match(OBJECT_CAT_AND_INST_RE, record["synset"])
        assert (
            synset is not None
        ), "prop_param_annots/particleRemover.csv has line without properly formed synset"
        synset = synset.group()

        # Skip washer and dryer
        if "not particleremover" in record["synset"].lower():
            continue

        default_fluid_conditions = parse_conditions_entry(
            record["other liquids"], syns_to_props
        )
        default_visual_conditions = parse_conditions_entry(
            record["other visualSubstances"], syns_to_props
        )
        default_non_fluid_conditions = parse_conditions_entry(
            record["other physicalSubstances"], syns_to_props
        )
        if record["method"] not in {"projection", "adjacency"}:
            raise ValueError(
                f"Synset {record['synset']} prop particleRemover has invalid method {record['method']}"
            )

        remover_kwargs = {
            "conditions": {},
            "default_visual_conditions": default_visual_conditions,
            "default_non_fluid_conditions": default_non_fluid_conditions,
            "default_fluid_conditions": default_fluid_conditions,
            "method": record["method"],
        }

        # Iterate through all the columns headed by a substance, in no particular order since their ultimate location is a dict
        for dirtiness_substance_synset in [
            key for key in record if re.match(OBJECT_CAT_AND_INST_RE, key) is not None
        ]:
            if dirtiness_substance_synset not in syns_to_props:
                continue
            conditions = parse_conditions_entry(
                record[dirtiness_substance_synset], syns_to_props
            )
            remover_kwargs["conditions"][dirtiness_substance_synset] = conditions

        synset_cleaning_mapping[synset] = remover_kwargs

    return synset_cleaning_mapping


# Main parameter method


def create_get_save_propagated_annots_params(syns_to_props):
    print("Processing param annots...")
    synsets_to_particleremover_params = get_synsets_to_particle_remover_params(
        syns_to_props
    )
    for (
        particleremover_syn,
        particleremover_params,
    ) in synsets_to_particleremover_params.items():
        if particleremover_syn not in syns_to_props:
            continue
        assert (
            "particleRemover" in syns_to_props[particleremover_syn]
        ), f"Synset {particleremover_syn} has particleRemover params but is not annotated as a particleRemover"
        syns_to_props[particleremover_syn]["particleRemover"] = particleremover_params

    for prop_fn in os.listdir(PROP_PARAM_ANNOTS_DIR):
        prop = prop_fn.split(".")[0]

        if prop == "particleRemover":
            continue
        else:
            param_annots = pd.read_csv(
                os.path.join(PROP_PARAM_ANNOTS_DIR, prop_fn)
            ).to_dict(orient="records")
            for param_record in param_annots:
                if param_record["synset"] not in syns_to_props:
                    continue

                for param_name, param_value in param_record.items():
                    if param_name == "synset":
                        continue

                    if param_name == "requires_toggled_on" and param_value == 1:
                        assert (
                            "toggleable" in syns_to_props[param_record["synset"]]
                        ), f"Synset {param_record['synset']} has requires_toggled_on param but is not annotated as toggleable"
                    if param_name == "requires_closed" and param_value == 1:
                        assert (
                            "openable" in syns_to_props[param_record["synset"]]
                        ), f"Synset {param_record['synset']} has requires_closed param but is not annotated as openable"

                    # Params that can have NaN values
                    if param_name in [
                        "substance_cooking_derivative_synset",
                        "sliceable_derivative_synset",
                        "uncooked_diceable_derivative_synset",
                        "cooked_diceable_derivative_synset",
                    ]:
                        if not pd.isna(param_value):
                            formatted_param_value = param_value
                            syns_to_props[param_record["synset"]][prop][param_name] = (
                                formatted_param_value
                            )
                        continue

                    # Params that should not have NaN values

                    # NaN values
                    elif pd.isna(param_value):
                        if prop == "coldSource":
                            raise ValueError(
                                f"synset {param_record['synset']} coldSource annotation has NaN value for parameter {param_name}. Either handle NaN or annotate parameter value."
                            )
                        elif prop == "cookable":
                            raise ValueError(
                                f"synset {param_record['synset']} cookable annotation has NaN value for parameter {param_name}. Either handle NaN or annotate parameter value."
                            )
                        elif prop == "flammable":
                            raise ValueError(
                                f"synset {param_record['synset']} flammable annotation has NaN value for parameter {param_name}. Either handle NaN or annotate parameter value."
                            )
                        elif prop == "heatSource":
                            raise ValueError(
                                f"synset {param_record['synset']} heatSource annotation has NaN value for parameter {param_name}. Either handle NaN or annotate parameter value."
                            )
                        elif prop == "particleApplier":
                            raise ValueError(
                                f"synset {param_record['synset']} particleApplier annotation has NaN value for parameter {param_name}. Either handle NaN or annotate parameter value."
                            )
                        elif prop == "particleSink":
                            if param_name == "conditions":
                                formatted_param_value = {}
                            elif param_name == "default_fluid_conditions":
                                formatted_param_value = []
                            elif param_name == "default_non_fluid_conditions":
                                formatted_param_value = []
                            elif param_name == "default_visual_conditions":
                                formatted_param_value = None
                            else:
                                raise ValueError(
                                    f"synset {param_record['synset']} particleSink annotation has NaN value for parameter {param_name}. Either handle NaN or annotate parameter value."
                                )
                        elif prop == "particleSource":
                            raise ValueError(
                                f"synset {param_record['synset']} particleSource annotation has NaN value for parameter {param_name}. Either handle NaN or annotate parameter value."
                            )
                        elif prop == "meltable":
                            raise ValueError(
                                f"synset {param_record['synset']} meltable annotation has NaN value for parameter {param_name}. Either handle NaN or annotate param value."
                            )

                    # `conditions` values - format is keyword1:bool_value1;keyword2:bool_value2;...;keywordN:bool_valueN
                    elif param_name == "conditions":
                        if (prop == "particleApplier") or (prop == "particleSource"):
                            # NOTE assumes that there may be multiple conditions
                            conditions = param_value.split(";")
                            formatted_conditions = []
                            for condition in conditions:
                                if condition == "gravity:True":
                                    formatted_conditions.append(("gravity", True))
                                elif condition == "toggled_on:True":
                                    assert (
                                        "toggleable"
                                        in syns_to_props[param_record["synset"]]
                                    ), f"Synset {param_record['synset']} has particleApplier or particleSource with condition `toggled_on:True` but is not annotated as toggleable"
                                    formatted_conditions.append(("toggled_on", True))
                                else:
                                    raise ValueError(
                                        f"Synset {param_record['synset']} prop {prop} has unhandled condition {condition}"
                                    )
                            formatted_param_value = {
                                param_record["system"]: formatted_conditions
                            }

                        else:
                            raise ValueError(
                                f"prop {prop} not handled for parameter name `conditions`"
                            )

                    # Can skip system since it's just part of handling `conditions`
                    elif param_name == "system":
                        continue

                    # `method` values
                    elif param_name == "method":
                        if (prop == "particleApplier") or (prop == "particleSource"):
                            formatted_param_value = "projection"
                        else:
                            raise ValueError(
                                f"prop {prop} not handled for parameter name `method`"
                            )

                    # Required derivative synsets
                    elif param_name == "meltable_derivative_synset":
                        formatted_param_value = param_value

                    # Float values
                    else:
                        try:
                            formatted_param_value = float(param_value)
                        except ValueError:
                            raise ValueError(
                                f"Synset {param_record['synset']} property {prop} has param {param_name} that is not named `method`, `conditions`, or `system` and is not a NaN or a float. This is unhandled - please check."
                            )

                    assert (
                        prop in syns_to_props[param_record["synset"]]
                    ), f"Synset {param_record['synset']} has prop param {prop} but is not annotated as {prop}"
                    syns_to_props[param_record["synset"]][prop][param_name] = (
                        formatted_param_value
                    )

    with open(PARAMS_OUTFILE_FN, "w") as f:
        json.dump(syns_to_props, f, indent=4)

    print("Params parsed and added to flat and hierarchical files, saved.")
    return syns_to_props


if __name__ == "__main__":
    create_get_save_propagated_annots_params()
