import json
import fs.path
from fs.multifs import MultiFS
import numpy as np
import tqdm
import trimesh

import b1k_pipeline.utils


MAX_BBOX = 0.3


def get_fillable_mesh(option, combined_object_and_volume_fs):
    options = {}

    dip_path = "fillable_dip.obj"
    if combined_object_and_volume_fs.exists(dip_path):
        # Find the scale the mesh was generated at
        metadata = json.loads(
            combined_object_and_volume_fs.readtext("misc/metadata.json")
        )
        native_bbox = np.array(metadata["bbox_size"])
        scale = np.minimum(1, MAX_BBOX / np.max(native_bbox))

        dip_mesh = b1k_pipeline.utils.load_mesh(
            combined_object_and_volume_fs, dip_path, force="mesh"
        )
        inv_scale = 1 / scale
        transform = np.diag([inv_scale, inv_scale, inv_scale, 1])
        dip_mesh.apply_transform(transform)
        options["dip"] = dip_mesh

    ray_path = "fillable_ray.obj"
    if combined_object_and_volume_fs.exists(ray_path):
        ray_mesh = b1k_pipeline.utils.load_mesh(
            combined_object_and_volume_fs, ray_path, force="mesh"
        )
        options["ray"] = ray_mesh

    if "dip" in options and "ray" in options:
        # Check if either mesh contains the entire other mesh
        combined_mesh = trimesh.convex.convex_hull(
            np.concatenate([dip_mesh.vertices, ray_mesh.vertices], axis=0)
        )
        options["combined"] = combined_mesh

    assert option in options, f"Option {option} not found in {options.keys()}"
    return options[option]


def aggregate_fillable_volumes():
    with b1k_pipeline.utils.PipelineFS().open(
        "metadata/fillable_assignments.json"
    ) as f:
        data = json.load(f)

    errors = []

    # Process objects one by one
    with (
        b1k_pipeline.utils.ParallelZipFS("objects.zip") as objects_fs,
        b1k_pipeline.utils.PipelineFS().opendir(
            "artifacts/parallels/fillable_volumes"
        ) as fillables_fs,
        b1k_pipeline.utils.ParallelZipFS("fillable_volumes.zip", write=True) as out_fs,
    ):
        input_fs = MultiFS()
        input_fs.add_fs("objects", objects_fs)
        input_fs.add_fs("fillables", fillables_fs)
        objdir_glob = list(objects_fs.glob("objects/*/*/"))
        for item in tqdm.tqdm(objdir_glob):
            # If it doesnt have an annotation, nothing to do.
            model_id = fs.path.parts(item.path)[-1]
            if model_id not in data:
                continue

            # If it does not have a selection annotated, nothing to do.
            selection = data[model_id]
            if selection not in ("combined", "dip", "ray"):
                continue

            # Otherwise let's process the mesh and save it.
            try:
                mesh = get_fillable_mesh(selection, input_fs.opendir(item.path))
                b1k_pipeline.utils.save_mesh(
                    mesh, out_fs.makedirs(item.path), "fillable.obj"
                )
            except Exception as e:
                print("Error processing", item.path, e)
                errors.append(item.path)

    if errors:
        raise ValueError("Some meshes failed to process")


if __name__ == "__main__":
    aggregate_fillable_volumes()
