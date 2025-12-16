import csv
import json
import fs.copy
from fs.multifs import MultiFS
import fs.path

from b1k_pipeline.utils import ParallelZipFS, PipelineFS


def main():
    with (
        PipelineFS() as pipeline_fs,
        ParallelZipFS("objects_usd.zip") as objects_usd_fs,
        ParallelZipFS("objects.zip") as objects_urdf_fs,
    ):
        objects_fs = MultiFS()
        objects_fs.add_fs("objects_usd", objects_usd_fs, priority=0)
        objects_fs.add_fs("objects_urdf", objects_urdf_fs, priority=1)

        with ParallelZipFS("systems.zip", write=True) as out_fs:
            substances = {}
            with pipeline_fs.open("metadata/substance_hyperparams.csv") as csvfile:
                reader = csv.DictReader(csvfile, delimiter=",", quotechar='"')
                for row in reader:
                    name = row["substance"]
                    print(name)
                    assert name not in substances, f"Duplicate substance {name}"
                    params = json.loads(row["hyperparams"])
                    substances[name] = params

            system_root_dir = out_fs.makedir("systems")
            objects_root_dir = objects_fs.opendir("objects")
            for system_name, metadata in substances.items():
                system_dir = system_root_dir.makedir(system_name)

                # Copy over asset files if they exist
                if objects_root_dir.exists(system_name):
                    fs.copy.copy_fs(objects_root_dir.opendir(system_name), system_dir)

                # Dump the metadata
                with system_dir.open("metadata.json", "w") as f:
                    json.dump(metadata, f)

            print("Done processing. Archiving things now.")

            # Save the logs
            with pipeline_fs.pipeline_output().open("generate_systems.json", "w") as f:
                json.dump({"success": True}, f)


if __name__ == "__main__":
    main()
