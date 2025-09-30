import json
import os
import pkgutil
import shutil
from string import Template

import omnigibson
from omnigibson import examples
from omnigibson.utils.asset_utils import download_omnigibson_robot_assets

download_omnigibson_robot_assets()

EXAMPLES_TO_SKIP = [
    "action_primitives.rs_int_example",
    "action_primitives.solve_simple_task",
    "action_primitives.wip_solve_behavior_task",
    "learning.navigation_policy_demo",
    "teleoperation.robot_teleoperate_demo",
    # TODO: Temporarily skip the following examples
    # "object_states.attachment_demo",  # seg fualt??
    # "environments.behavior_env_demo",  # This only works with pre-sampled cached BEHAVIOR activity scene
]


def main():
    examples_list = []
    prefix = examples.__name__ + "."
    for package in pkgutil.walk_packages(examples.__path__, prefix):
        if not package.ispkg:
            examples_list.append(package.name[len(prefix) :])

    current_dir = os.path.dirname(os.path.abspath(__file__))
    tests_of_examples_dir = os.path.join(current_dir, "tests_of_examples")
    shutil.rmtree(tests_of_examples_dir, ignore_errors=True)
    os.makedirs(tests_of_examples_dir, exist_ok=True)

    examples_list = [example for example in examples_list if example not in EXAMPLES_TO_SKIP]

    test_file_names = []

    for example in examples_list:
        template_file_name = os.path.join(omnigibson.__path__[0], "..", "tests", "test_of_example_template.txt")
        with open(template_file_name, "r") as f:
            substitutes = dict()
            substitutes["module"] = example
            name = example.rsplit(".", 1)[-1]
            substitutes["name"] = name
            src = Template(f.read())
            dst = src.substitute(substitutes)
            test_file_name = name + "_test.py"
            test_file_path = os.path.join(tests_of_examples_dir, test_file_name)
            with open(test_file_path, "w") as test_file:
                test_file.write(dst)
            # Add the test file name (without .py extension) to the list
            test_file_names.append(name + "_test")

    # Write the list of test file names to a JSON file
    json_file_path = os.path.join(current_dir, "example_tests.json")
    with open(json_file_path, "w") as json_file:
        json.dump(test_file_names, json_file)


if __name__ == "__main__":
    main()
