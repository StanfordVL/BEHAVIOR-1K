import argparse
from collections import Counter, defaultdict
import glob
import json
import os

import yaml
import b1k_pipeline.utils

ALLOWED_PART_TAGS = {
    "subpart",
    "extrapart",
    "connectedpart",
}


def main():
    needed = set()
    providers = defaultdict(list)
    bounding_box_sizes = {}
    meta_links = defaultdict(set)
    attachment_pairs = defaultdict(
        lambda: defaultdict(set)
    )  # attachment_pairs[model_id][F/M] = {attachment_type1, attachment_type2}
    needed_by = defaultdict(list)
    seen_as = defaultdict(set)
    skipped_files = []

    with b1k_pipeline.utils.PipelineFS() as pipeline_fs:
        # Get the providers.
        with pipeline_fs.open("params.yaml", "r") as f:
            params = yaml.load(f, Loader=yaml.SafeLoader)
            targets = params["combined"]

        # Merge the object lists.
        for target in targets:
            with pipeline_fs.target_output(target) as target_output_fs:
                with target_output_fs.open("object_list.json", "r") as f:
                    object_list = json.load(f)

                if not object_list["success"]:
                    skipped_files.append(target)
                    continue

                needed |= set(object_list["needed_objects"])
                for obj in object_list["needed_objects"]:
                    needed_by[obj].append(target)
                for provided in object_list["provided_objects"]:
                    providers[provided].append(target)

                    model_id = provided.split("-")[1]
                    bounding_box_size = object_list["bounding_boxes"][model_id]["0"][
                        "extent"
                    ]
                    bounding_box_sizes[model_id] = [
                        x / 1000.0 for x in bounding_box_size
                    ]
                for obj, links in object_list["meta_links"].items():
                    meta_links[obj].update(links)
                for obj, attachment_dict in object_list["attachment_pairs"].items():
                    for attachment_gender, attachment_types in attachment_dict.items():
                        attachment_pairs[obj][attachment_gender].update(
                            attachment_types
                        )

                # Manually generate pseudo-metalinks for parts
                for name, _, parent in object_list["max_tree"]:
                    if not parent:
                        continue

                    # Check parseable name
                    parsed_name = b1k_pipeline.utils.parse_name(name)
                    if parsed_name is None:
                        continue

                    # Record the seen-as categories
                    seen_as[parsed_name.group("model_id")].add(
                        parsed_name.group("category")
                    )

                    # Get the tags that are on the parent
                    tags_str = parsed_name.group("tag")
                    tags = set()
                    if tags_str:
                        tags = {x[1:] for x in tags_str.split("-") if x}

                    part_tags = tags & ALLOWED_PART_TAGS
                    if not part_tags:
                        continue

                    # If we have part tags, we need to add them to the parent's meta links.
                    # For that, get the parent's name.
                    parsed_parent_name = b1k_pipeline.utils.parse_name(parent)
                    assert (
                        parsed_parent_name
                    ), f"Parent {parent} of {name} is not parseable"
                    parent_category = parsed_parent_name.group("category")
                    parent_model = parsed_parent_name.group("model_id")
                    parent_name = f"{parent_category}-{parent_model}"

                    # Check that the parent shows up in the inventory
                    assert (
                        parent_name in needed_by
                    ), f"Parent {parent_name} of {name} is not in the inventory"

                    # Add the part tags as meta links to the parent
                    meta_links[parent_name].update(part_tags)

        # Check the multiple-provided copies.
        multiple_provided = {k: v for k, v in providers.items() if len(v) > 1}
        single_provider = {k: v[0] for k, v in providers.items()}

        provided_objects = set(single_provider.keys())
        missing_objects = {x.split("-")[1] for x in needed} - {
            x.split("-")[1] for x in provided_objects
        }
        missing_objects &= {
            obj.split("-")[1]
            for obj, needers in needed_by.items()
            if any(t in params["final_scenes"] for t in needers)
        }  # Limit this to stuff that shows up in final scenes

        seen_as_multiple_categories = {
            obj_id: sorted(categories)
            for obj_id, categories in seen_as.items()
            if len(categories) > 1
        }

        id_occurrences = defaultdict(list)
        for obj_name, provider in single_provider.items():
            id_occurrences[obj_name.split("-")[1]].append(provider)
        id_collisions = {
            obj_id: obj_names
            for obj_id, obj_names in id_occurrences.items()
            if len(obj_names) > 1
        }

        success = (
            len(skipped_files) == 0
            and len(multiple_provided) == 0
            and len(missing_objects) == 0
            and len(id_collisions) == 0
            and len(seen_as_multiple_categories) == 0
        )
        results = {
            "success": success,
            "providers": single_provider,
            "needed_by": needed_by,
            "meta_links": {x: sorted(y) for x, y in meta_links.items()},
            "attachment_pairs": {
                k: {kk: sorted(vv) for kk, vv in sorted(v.items())}
                for k, v in sorted(attachment_pairs.items())
            },
            "bounding_box_sizes": bounding_box_sizes,
            "error_skipped_files": sorted(skipped_files),
            "error_multiple_provided": multiple_provided,
            "error_missing_objects": sorted(missing_objects),
            "error_id_collisions": id_collisions,
            "error_seen_as_multiple_categories": seen_as_multiple_categories,
        }
        with pipeline_fs.pipeline_output() as pipeline_output_fs:
            with pipeline_output_fs.open("object_inventory.json", "w") as f:
                json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
