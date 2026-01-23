import os
import pathlib
import gspread
import csv
import pandas as pd

OUTPUT_ROOT = pathlib.Path(__file__).parents[1] / "metadata"

ASSETS_SHEET_KEY = "10L8wjNDvr1XYMMHas4IYYP9ZK7TfQHu--Kzoi0qhAe4"
SYNSETS_SHEET_KEY = "1eIQn1HzUJV15nCP4MqsHvrdWAV9VrKoxOqSnQxF0_1A"
COMPLAINTS_SHEET_KEY = "1p5SA2Pt44UHcMZsT3IeOHEVPb8TSPkFro_i8bvWodQA"

ALL_SHEETS = [
    (ASSETS_SHEET_KEY, "Object Category Mapping", "category_mapping.csv"),
    (ASSETS_SHEET_KEY, "Allowed Room Types", "allowed_room_types.csv"),
    (ASSETS_SHEET_KEY, "Object Renames", "object_renames.csv"),
    (ASSETS_SHEET_KEY, "Deletion Queue", "deletion_queue.csv"),
    (ASSETS_SHEET_KEY, "Non-Sampleable Categories", "non_sampleable_categories.csv"),
    (ASSETS_SHEET_KEY, "Substance Hyperparams", "substance_hyperparams.csv"),
    (SYNSETS_SHEET_KEY, "Synsets", "synset_property.csv"),
]


def main():
    # Download the sheets
    OUTPUT_ROOT.mkdir(exist_ok=True)
    client = gspread.service_account(
        filename=os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    )
    for key, sheet_name, filename in ALL_SHEETS:
        worksheet = client.open_by_key(key).worksheet(sheet_name)
        with open(OUTPUT_ROOT / filename, "w") as f:
            writer = csv.writer(f)
            writer.writerows(worksheet.get_all_values())

    # Aggregate the complaints
    import pathlib, json

    cad_dir = pathlib.Path("cad")
    files = cad_dir.glob("*/*/complaints.json")
    complaints_by_file = {}
    for fn in files:
        with open(fn) as f:
            target = fn.parent.relative_to(cad_dir).as_posix()
            complaints_by_file[target] = json.load(f)
    next_id = (
        max(c["id"] for complaints in complaints_by_file.values() for c in complaints)
        + 1
    )

    # Upload the complaints
    complaint_sheet = client.open_by_key(COMPLAINTS_SHEET_KEY).worksheet("Complaints")
    df = pd.DataFrame(complaint_sheet.get_all_records())
    df.index += 1
    df.id = pd.to_numeric(df.id, errors="coerce")

    # If there are any new complaints (these would have no ID), assign them IDs, add them to the complaint dicts, and update the sheet
    new_complaints = df[df["id"].isna()]
    indices_to_ids = {}
    for index, c in new_complaints.iterrows():
        # Assign a new ID to the complaint
        this_id = next_id
        next_id += 1
        complaints_by_file[c["file"]].append(
            {
                "id": this_id,
                "object": c["object"],
                "type": c["type"],
                "processed": c["processed"],
                "complaint": c["complaint"],
                "additional_info": c["additional_info"],
            }
        )
        indices_to_ids[index] = this_id

    # Update the complaints sheet with the new IDs
    for i, this_id in indices_to_ids.items():
        df.at[i, "id"] = this_id

    # Check that there are no more NaN IDs
    assert (
        df.id.notna().all()
    ), "There are still NaN IDs in the complaints sheet. Please check."

    # Update the dataframe from the complaint files
    for target, complaints in complaints_by_file.items():
        for c in complaints:
            # Check if a complaint with this ID already exists in the dataframe
            if c["id"] in df.id.values:
                # Update the complaint in the dataframe
                df.at[df[df["id"] == c["id"]].index[0], "processed"] = c["processed"]
                df.at[df[df["id"] == c["id"]].index[0], "complaint"] = c["complaint"]
                df.at[df[df["id"] == c["id"]].index[0], "additional_info"] = c[
                    "additional_info"
                ]
                df.at[df[df["id"] == c["id"]].index[0], "type"] = c["type"]
            else:
                # Add a new complaint to the dataframe. Best to do this in-place rather than creating a new dataframe.
                if c["id"] not in df.id.values:
                    complaint_with_file = {
                        "file": target,
                        "assignee": "",
                        "work done": "",
                    }
                    complaint_with_file.update(c)
                    df = pd.concat(
                        [df, pd.DataFrame([complaint_with_file])], ignore_index=True
                    )
                else:
                    # Update the existing complaint in the dataframe
                    df.at[c["id"], "object"] = c["object"]
                    df.at[c["id"], "processed"] = c["processed"]
                    df.at[c["id"], "complaint"] = c["complaint"]
                    df.at[c["id"], "additional_info"] = c["additional_info"]
                    df.at[c["id"], "type"] = c["type"]

    # Save the JSONs again
    for target, complaints in complaints_by_file.items():
        with open(cad_dir / target / "complaints.json", "w", newline="\n") as f:
            json.dump(complaints, f, indent=4)

    # Update the spreadsheet
    complaint_sheet.update([df.columns.values.tolist()] + df.values.tolist())


if __name__ == "__main__":
    main()
