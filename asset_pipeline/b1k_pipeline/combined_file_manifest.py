import json

import b1k_pipeline.utils


def main():
    targets = b1k_pipeline.utils.get_targets("combined")
    with b1k_pipeline.utils.PipelineFS() as pipeline_fs:
        combined_file_manifest = {}

        # Merge the object lists.
        for target in targets:
            with pipeline_fs.target_output(target) as target_output_fs:
                with target_output_fs.open("file_manifest.json", "r") as f:
                    combined_file_manifest[target] = json.load(f)

        # Write the combined list
        with pipeline_fs.open(
            "artifacts/pipeline/combined_file_manifest.json", "w", newline="\n"
        ) as f:
            json.dump(combined_file_manifest, f, indent=2)


if __name__ == "__main__":
    main()
