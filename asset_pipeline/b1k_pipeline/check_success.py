import json
import sys
import b1k_pipeline.utils


def main():
    # Validate file paths.
    json_file = (b1k_pipeline.utils.PIPELINE_ROOT / sys.argv[1]).absolute()
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

    # Create the success file if successful.
    if j["success"]:
        with open(success_file, "w") as f:
            pass


if __name__ == "__main__":
    main()
