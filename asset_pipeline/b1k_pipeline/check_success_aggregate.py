import json
import sys
import b1k_pipeline.utils


def main():
    # Validate file paths.
    failures = []
    target_type = "combined" if len(sys.argv) < 4 else sys.argv[3]
    for target in b1k_pipeline.utils.get_targets(target_type):
        target_path = b1k_pipeline.utils.PIPELINE_ROOT / "cad" / target
        json_file = (target_path / sys.argv[1]).absolute()
        assert (
            json_file.exists() and json_file.suffix == ".json"
        ), f"Can't find {json_file}."
        success_file = (b1k_pipeline.utils.PIPELINE_ROOT / sys.argv[2]).absolute()
        assert (
            not success_file.exists() and success_file.suffix == ".success"
        ), f"Existing or invalid {success_file}."

        # Load the JSON file.
        with open(json_file, "r") as f:
            j = json.load(f)

        if not j["success"]:
            failures.append(target)

    # Create the success file if successful.
    if failures:
        print("Failures found!")
        print("\n".join(failures))
        print(
            "\nTo edit the failed files, copy and paste the below command, and use the Next Failed button on 3ds Max:"
        )
        print("dvc unprotect " + " ".join(f"cad/{x}/processed.max" for x in failures))
        return

    # Otherwise, save the success file
    with open(success_file, "w") as f:
        pass


if __name__ == "__main__":
    main()
