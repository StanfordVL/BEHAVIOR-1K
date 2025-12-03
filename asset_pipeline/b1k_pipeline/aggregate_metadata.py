import numpy as np

import csv
import json
import traceback

import b1k_pipeline.utils


def main():
    with (
        b1k_pipeline.utils.PipelineFS() as pipeline_fs,
        b1k_pipeline.utils.ParallelZipFS("metadata.zip", write=True) as archive_fs,
    ):
        pipeline_output_dir = pipeline_fs.pipeline_output()
        metadata_in_dir = pipeline_fs.opendir("metadata")
        metadata_out_dir = archive_fs.makedirs("metadata")

        success = True
        error_msgs = []
        warning_msgs = []
        try:
            with pipeline_output_dir.open("object_inventory.json", "r") as f:
                object_inventory = json.load(f)

            with pipeline_output_dir.open("collision_average_volumes.json", "r") as f:
                collision_average_volumes = json.load(f)["average_volumes"]

            # For now, get categories from CSV file
            categories_by_id = {}
            avg_category_specs = {}
            with metadata_in_dir.open("category_mapping.csv", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                for i, row in enumerate(reader):
                    # TODO: Use something more authoritative than this entry
                    # Get the has_system entry to check if the category is a particle system,
                    # and if so, skip it entirely.
                    has_system = row["has_system"].strip().lower()
                    if has_system == "true":
                        continue

                    cat_id = int(row["ID"].strip())
                    category = row["category"].strip()
                    categories_by_id[cat_id] = category

                    assert (
                        category in collision_average_volumes
                    ), f"Category {category} not in collision_average_volumes"

                    volume = collision_average_volumes[category]
                    mass = (
                        float(row["mass (auto)"])
                        if row["mass (auto)"] and row["mass (auto)"] != "#DIV/0!"
                        else None
                    )
                    assert (
                        mass is not None and mass > 0
                    ), f"Invalid mass for category {category}"
                    density = mass / volume if mass and volume else None

                    # The density needs to be temporarily capped at 10000 kg/m^3 - values above that
                    # are likely due to errors in the mass annotation (LLM based) or in the volume
                    # obtained from the collision mesh (infinitesimal dimension).
                    # if density and density > 10000:
                    #     print(
                    #         f"Warning: Density for category {category} is {density} kg/m^3, capping to 10000 kg/m^3"
                    #     )
                    #     density = 10000

                    avg_category_specs[category] = {
                        "mass": mass,
                        "volume": volume,
                        "density": density,
                    }

            # Validate: have we found all categories on the list?
            for cat in avg_category_specs:
                if cat not in categories_by_id.values():
                    error_msgs.append(f"Could not find ID for category {cat}")

                if cat not in avg_category_specs:
                    error_msgs.append(
                        f"Category {cat} not found in avg_category_specs file"
                    )
                else:
                    missing_things = {
                        k for k, v in avg_category_specs[cat].items() if v is None
                    }
                    joined_missing_things = ", ".join(sorted(missing_things))
                    if missing_things & {"enable_ag", "mass"}:  # Errors
                        error_msgs.append(
                            f"Category {cat} missing {joined_missing_things} data"
                        )
                    elif missing_things:
                        warning_msgs.append(
                            f"Category {cat} missing {joined_missing_things} data"
                        )

            # Only continue if no errors are found by now
            assert not error_msgs

            # Fill missing IDs with spaces
            category_ids = [
                categories_by_id[i] if i in categories_by_id else ""
                for i in range(max(categories_by_id.keys()) + 1)
            ]
            category_ids_w_newline = "".join([x + "\n" for x in category_ids])
            with metadata_out_dir.open("categories.txt", "w") as f:
                f.write(category_ids_w_newline)

            # Get the room categories
            with metadata_in_dir.open("allowed_room_types.csv", "r") as f:
                reader = csv.DictReader(f)
                room_categories = "".join(
                    [row["Room Name"].strip() + "\n" for row in reader]
                )
            with metadata_out_dir.open("room_categories.txt", "w") as f:
                f.write(room_categories)

            # Compile the avg_category_specs.json file
            with metadata_out_dir.open("avg_category_specs.json", "w") as f:
                json.dump(avg_category_specs, f, indent=4)

            # Read and dump the non sampleable cats file
            with metadata_in_dir.open("non_sampleable_categories.csv", "r") as f:
                reader = csv.DictReader(f)
                non_sampleable_cats = "".join(
                    [row["synset"].strip() + "\n" for row in reader]
                )
            with metadata_out_dir.open("non_sampleable_categories.txt", "w") as f:
                f.write(non_sampleable_cats)

        except Exception as e:
            success = False
            error_msgs.append(traceback.format_exc())

        if error_msgs:
            print("Errors:")
            print("\n".join(error_msgs))

        if warning_msgs:
            print("Warnings:")
            print("\n".join(warning_msgs))

        with pipeline_fs.pipeline_output().open("aggregate_metadata.json", "w") as f:
            json.dump(
                {
                    "success": success,
                    "error_msgs": error_msgs,
                    "warning_msgs": warning_msgs,
                },
                f,
                indent=4,
            )


if __name__ == "__main__":
    main()
