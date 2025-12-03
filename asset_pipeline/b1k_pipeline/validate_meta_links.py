import csv
import json
from bddl.object_taxonomy import ObjectTaxonomy
import tqdm

from b1k_pipeline.utils import get_targets, PipelineFS, PIPELINE_ROOT

OBJECT_TAXONOMY = ObjectTaxonomy()

RENAME_DICT = {}
with open(PIPELINE_ROOT / "metadata/object_renames.csv", "r") as f:
    for row in csv.DictReader(f):
        RENAME_DICT[row["Object name"].strip()] = row["New Category"].strip()

DELETION_QUEUE = set()
with open(PIPELINE_ROOT / "metadata/deletion_queue.csv", "r") as f:
    for row in csv.DictReader(f):
        DELETION_QUEUE.add(row["Object"].strip().split("-")[1])

TASK_REQUIRED_SYNSETS = set()
with open(PIPELINE_ROOT / "metadata/task_required_synsets.csv", "r") as f:
    for row in csv.DictReader(f):
        TASK_REQUIRED_SYNSETS.add(row["Name"].strip())

META_REQUIRING_ABILITIES = {
    "heatSource",
    "coldSource",
    "fillable",
    "toggleable",
    "particleSink",
    "particleSource",
    "particleApplier",
    "particleRemover",
}


def check_meta_links(category, abilities, meta_links):
    errors = []

    # if category in ["walls", "floors", "lawn", "driveway"] and "collision" not in meta_links:
    #     errors.append("Wall/floor/lawn/driveway objects must have a manual collision mesh.")

    if "substance" in abilities:
        return []  # substances don't need any meta links

    for ability in ["heatSource", "coldSource"]:
        if ability in abilities:
            if "requires_toggled_on" not in abilities[ability]:
                errors.append(
                    f"{ability} objects must have requires_toggled_on annotated in BDDL"
                )
            elif (
                abilities[ability]["requires_toggled_on"]
                and "togglebutton" not in meta_links
            ):
                errors.append(
                    f"{ability} objects w/ requires_toggled_on must have a togglebutton"
                )

            if "requires_inside" not in abilities[ability]:
                errors.append(
                    f"{ability} objects must have requires_inside annotated in BDDL"
                )
            elif (
                not abilities[ability]["requires_inside"]
                and "heatsource" not in meta_links
            ):
                errors.append(
                    f"{ability} objects w/o requires_inside must have a heatsource"
                )

    if "fillable" in abilities and "fillable" not in meta_links:
        errors.append("fillable objects must have a fillable volume")

    if "toggleable" in abilities and "togglebutton" not in meta_links:
        errors.append("toggleable objects must have a togglebutton")

    if "particleSink" in abilities and "fluidsink" not in meta_links:
        errors.append("particleSink objects must have a fluidsink")

    particle_pairs = [
        ("particleSource", "fluidsource"),
        ("particleApplier", "particleapplier"),
        ("particleRemover", "particleremover"),
    ]
    for ability, meta_link in particle_pairs:
        if ability in abilities:
            method = 1  # This corresponds to the projection method
            if "method" not in abilities[ability]:
                pass
                # errors.append(f"{ability} parameters did not contain 'method' so we assumed projection.")
            else:
                method = abilities[ability]["method"]
            if method == 1 and meta_link not in meta_links:
                errors.append(
                    f"{ability} objects with projection mode must have a {meta_link}"
                )

    return errors


def main():
    targets = get_targets("combined")

    with PipelineFS() as fs:
        all_errors = {}
        fully_complete_objects = {}
        for target in tqdm.tqdm(targets):
            with fs.target_output(target).open("object_list.json") as f:
                object_list = json.load(f)

            target_errors = []
            for obj in object_list["provided_objects"]:
                if obj.split("-")[1] in DELETION_QUEUE:
                    continue

                # Get the properties of the object
                cat = RENAME_DICT[obj] if obj in RENAME_DICT else obj.split("-")[0]
                syn = OBJECT_TAXONOMY.get_synset_from_category(cat)
                assert syn, f"Could not find synset for object {obj}"
                abilities = OBJECT_TAXONOMY.get_abilities(syn)

                if cat not in fully_complete_objects:
                    fully_complete_objects[cat] = 0

                meta_links = (
                    set(object_list["meta_links"][obj])
                    if obj in object_list["meta_links"]
                    else set()
                )
                errors = check_meta_links(cat, abilities, meta_links)
                if errors:
                    target_errors.append((obj, errors))
                else:
                    # No errors means this object is complete
                    fully_complete_objects[cat] += 1

            if target_errors:
                all_errors[target] = target_errors

        for target, target_errors in all_errors.items():
            print("Errors for target", target)
            for obj, errors in target_errors:
                print("  ", obj)
                for error in errors:
                    print("    ", error)

        print(
            "Total error count:",
            sum(len(target_errors) for target_errors in all_errors.values()),
        )

        # Now check that each leaf-level synset has at least one fully complete object
        print("Task-required synsets that have no fully-complete objects:")
        trs_missing = {}
        for trs in sorted(TASK_REQUIRED_SYNSETS):
            if "substance" in OBJECT_TAXONOMY.get_abilities(trs):
                continue
            descendants = OBJECT_TAXONOMY.get_subtree_categories(trs)
            trs_fully_complete = sum(
                fully_complete_objects[cat]
                for cat in descendants
                if cat in fully_complete_objects
            )
            if trs_fully_complete == 0:
                needed_abilities = sorted(
                    set(OBJECT_TAXONOMY.get_abilities(trs).keys())
                    & META_REQUIRING_ABILITIES
                )
                print(f" - {trs}: {', '.join(needed_abilities)}")
                trs_missing[trs] = needed_abilities

        with fs.pipeline_output().open("validate_meta_links.json", "w") as f:
            json.dump(
                {
                    "target_errors": all_errors,
                    "task_required_synsets_with_no_complete_obj": trs_missing,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
