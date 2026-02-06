# %%
import pathlib, json
import pandas as pd
from tqdm.auto import tqdm
import shutil

# %%
hssd_output_root = pathlib.Path("/fsx-siro/cgokmen/behavior-data2/hssd")
hssd_root = pathlib.Path("/fsx-siro/cgokmen/habitat-data/scene_datasets/hssd-hab")
hssd_models_root = pathlib.Path("/fsx-siro/cgokmen/hssd-models")
hssd_metadata = pd.read_csv(hssd_root / "metadata/hssd_obj_semantics_condensed.csv")
hssd_models = sorted([fn for fn in hssd_models_root.rglob("*.glb") if "filteredSupportSurface" not in fn.name and "collider" not in fn.name])

hssd_mapping = {}
hssd_missing_count = 0
for obj_path in tqdm(hssd_models):
    if obj_path.parts[-2] == "openings":
        category = "openings"
    elif obj_path.parts[-2] == "stages":
        category = "stages"
    else:
        # Check if it exists in the dataframe
        rows = hssd_metadata[hssd_metadata["Object Hash"] == obj_path.stem]
        if not rows.empty:
            category = rows.iloc[0][
                "Semantic Category:\nCONDENSED\n\nThis is an effort to condense the semantic categories by a couple hundred"
            ]
        else:
            print(f"Warning: {obj_path.stem} not found in metadata, defaulting to 'object'")
            category = "object"

    # Sanitize both category and model names to contain only letters (and underscores for category)
    category = "".join(c if c.isalnum() or c == "_" else "_" for c in category.lower())
    model = "hssd" + "".join(c if c.isalnum() else "" for c in obj_path.stem)

    model_root = hssd_output_root / "objects" / category / model
    success_file = model_root / "import.success"
    assert obj_path.stem not in hssd_mapping, f"Filename overlap! {obj_path}"
    if not success_file.exists():
        print(f"Missing success file for {obj_path}: {success_file}")
        hssd_missing_count += 1
        continue
    
    shutil.copyfile(obj_path, model_root / f"{model}.glb")

# (hssd_output_root / "object_name_mapping.json").write_text(json.dumps(hssd_mapping, indent=4))
print(hssd_missing_count, "missing hssd mappings")


# %%
spoc_output_root = pathlib.Path("/fsx-siro/cgokmen/behavior-data2/spoc")
spoc_root = pathlib.Path("/fsx-siro/cgokmen/procthor/assets/2023_07_28")
spoc_annots = json.loads((spoc_root / "annotations.json").read_text())
spoc_models = sorted(spoc_root.glob("assets/*/*.glb"))

spoc_mapping = {}
spoc_missing_count = 0
for obj_path in tqdm(spoc_models):
    if obj_path.stem not in spoc_annots:
        print(f"Skipping {obj_path.stem} as it has no annotations")
        continue
    this_annots = spoc_annots[obj_path.stem]

    # Sanitize both category and model names to contain only letters (and underscores for category)
    category = "".join(c if c.isalnum() or c == "_" else "_" for c in this_annots["category"].lower())
    model = "spoc" + "".join(c if c.isalnum() else "" for c in obj_path.stem.lower())

    model_root = spoc_output_root / "objects" / category / model
    success_file = model_root / "import.success"
    assert obj_path.stem not in spoc_mapping, f"Filename overlap! {obj_path}"
    if not success_file.exists():
        print(f"Missing success file for {obj_path}: {success_file}")
        spoc_missing_count += 1
        continue

    shutil.copyfile(obj_path, model_root / f"{model}.glb")

# (spoc_output_root / "object_name_mapping.json").write_text(json.dumps(spoc_mapping, indent=4))
print(spoc_missing_count, "missing spoc mappings")

# %%
ai2_output_root = pathlib.Path("/fsx-siro/cgokmen/behavior-data2/ai2thor")
ai2_hab_root = pathlib.Path("/fsx-siro/cgokmen/procthor/ai2thor/ai2thor-hab")
ai2_uc_root = pathlib.Path("/fsx-siro/cgokmen/procthor/ai2thor/ai2thorhab-uncompressed")
ai2_categories = pd.read_csv("/fsx-siro/cgokmen/procthor/ai2thor/ai2thor_categories.csv")
model2cat = dict(zip(ai2_categories["Model Name"], ai2_categories["Category"]))

ai2_main_models = set(ai2_uc_root.glob("assets/objects/*.glb"))
print(len(ai2_main_models), "main models")
ai2_stage_models = set(ai2_hab_root.glob("assets/stages/**/*.glb"))
print(len(ai2_stage_models), "stage models")
ai2_models = sorted(ai2_main_models | ai2_stage_models)
print(len(ai2_models), "models")

ai2_mapping = {}
ai2_missing_count = 0
for obj_path in tqdm(ai2_models):
    if obj_path in ai2_stage_models:
        # For stages, the file hierarchy is a bit different.
        stages_dir = ai2_hab_root / "assets" / "stages"
        assert stages_dir in obj_path.parents, f"Stage GLB {obj_path} is not in stages directory"
        # Get the index of stages_dir in the parents list
        idx = obj_path.parents.index(stages_dir)
        # The type is the next parent directory
        stage_type_dir = obj_path.parents[idx - 1]
        # If the filename doesn't already start with the type, rename it
        category = "stages"
        model = obj_path.stem
        if not model.startswith(stage_type_dir.name):
            model = f"{stage_type_dir.name}-{model}"
    else:
        model = obj_path.stem
        category = model2cat[model]

    # Sanitize both category and model names to contain only letters (and underscores for category)
    category = "".join(c if c.isalnum() or c == "_" else "_" for c in category.lower())
    model = "ai2thor" + "".join(c if c.isalnum() else "" for c in obj_path.stem)

    model_root = ai2_output_root / "objects" / category / model
    success_file = model_root / "import.success"
    assert obj_path.stem not in ai2_mapping, f"Filename overlap! {obj_path}"
    if not success_file.exists():
        print(f"Missing success file for {obj_path}: {success_file}")
        ai2_missing_count += 1
        continue

    relpath = obj_path.relative_to(ai2_hab_root) if obj_path in ai2_stage_models else obj_path.relative_to(ai2_uc_root)
    shutil.copyfile(obj_path, model_root / f"{model}.glb")

# (ai2_output_root / "object_name_mapping.json").write_text(json.dumps(ai2_mapping, indent=4))
print(ai2_missing_count, "missing ai2 mappings")

# %%



