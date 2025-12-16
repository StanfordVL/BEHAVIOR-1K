import os
import pathlib
import time
import gspread
import csv

OUTPUT_ROOT = pathlib.Path(__file__).parents[1] / "generated_data"

ASSETS_SHEET_KEY = "10L8wjNDvr1XYMMHas4IYYP9ZK7TfQHu--Kzoi0qhAe4"
SYNSETS_SHEET_KEY = "1eIQn1HzUJV15nCP4MqsHvrdWAV9VrKoxOqSnQxF0_1A"
SYNSET_PARAMS_SHEET_KEY = "1jQomcQS3DSMLctEOElafCPj_v498gwtb57NC9eNwu-4"
EXPLICIT_TRANSITION_RULES_SHEET_KEY = "1q3MqvnT_bOVbkMit1c7dhbzsOvd3KUQhJGx3aaJiH04"

ALL_SHEETS = [
    (ASSETS_SHEET_KEY, "Object Category Mapping", "category_mapping.csv"),
    (ASSETS_SHEET_KEY, "Allowed Room Types", "allowed_room_types.csv"),
    (ASSETS_SHEET_KEY, "Substance Hyperparams", "substance_hyperparams.csv"),
    (ASSETS_SHEET_KEY, "Object Renames", "object_renames.csv"),
    (ASSETS_SHEET_KEY, "Deletion Queue", "deletion_queue.csv"),
    (SYNSETS_SHEET_KEY, "Synsets", "synsets.csv"),
    (SYNSET_PARAMS_SHEET_KEY, "heatSource", "prop_param_annots/heatSource.csv"),
    (SYNSET_PARAMS_SHEET_KEY, "coldSource", "prop_param_annots/coldSource.csv"),
    (SYNSET_PARAMS_SHEET_KEY, "cookable", "prop_param_annots/cookable.csv"),
    (SYNSET_PARAMS_SHEET_KEY, "flammable", "prop_param_annots/flammable.csv"),
    (
        SYNSET_PARAMS_SHEET_KEY,
        "particleApplier",
        "prop_param_annots/particleApplier.csv",
    ),
    (SYNSET_PARAMS_SHEET_KEY, "particleSource", "prop_param_annots/particleSource.csv"),
    (
        SYNSET_PARAMS_SHEET_KEY,
        "particleRemover",
        "prop_param_annots/particleRemover.csv",
    ),
    (SYNSET_PARAMS_SHEET_KEY, "particleSink", "prop_param_annots/particleSink.csv"),
    (SYNSET_PARAMS_SHEET_KEY, "diceable", "prop_param_annots/diceable.csv"),
    (SYNSET_PARAMS_SHEET_KEY, "sliceable", "prop_param_annots/sliceable.csv"),
    (SYNSET_PARAMS_SHEET_KEY, "meltable", "prop_param_annots/meltable.csv"),
    (SYNSET_PARAMS_SHEET_KEY, "toggleable", "prop_param_annots/toggleable.csv"),
    (
        EXPLICIT_TRANSITION_RULES_SHEET_KEY,
        "heat_cook",
        "transition_map/tm_raw_data/heat_cook.csv",
    ),
    (
        EXPLICIT_TRANSITION_RULES_SHEET_KEY,
        "mixing_stick",
        "transition_map/tm_raw_data/mixing_stick.csv",
    ),
    (
        EXPLICIT_TRANSITION_RULES_SHEET_KEY,
        "single_toggleable_machine",
        "transition_map/tm_raw_data/single_toggleable_machine.csv",
    ),
    (
        EXPLICIT_TRANSITION_RULES_SHEET_KEY,
        "washer",
        "transition_map/tm_raw_data/washer.csv",
    ),
]


def main():
    OUTPUT_ROOT.mkdir(exist_ok=True)
    client = gspread.service_account(
        filename=os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    )
    for key, sheet_name, filename in ALL_SHEETS:
        worksheet = client.open_by_key(key).worksheet(sheet_name)
        with open(OUTPUT_ROOT / filename, "w") as f:
            writer = csv.writer(f)
            writer.writerows(worksheet.get_all_values())

        # Stop Google from complaining.
        time.sleep(10)


if __name__ == "__main__":
    main()
