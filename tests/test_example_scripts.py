import importlib
import os
import sys
from pathlib import Path

import pytest

current_dir = Path(__file__).parent.absolute()
EXAMPLES_DIR = os.path.normpath(os.path.join(current_dir, "..", "omnigibson", "examples"))


def run_example(module_name, short_exec=True, headless=True):
    sys.path.insert(0, EXAMPLES_DIR)
    module = importlib.import_module(module_name)
    module.main(headless=headless, short_exec=short_exec)
    sys.path.pop(0)


def get_example_files():
    example_files = []
    for root, _, files in os.walk(EXAMPLES_DIR):
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                rel_path = os.path.relpath(os.path.join(root, file), EXAMPLES_DIR)
                module_name = os.path.splitext(rel_path)[0].replace(os.path.sep, ".")
                example_files.append(module_name)
    return example_files


@pytest.mark.parametrize("example_file", get_example_files())
def test_example_script(example_file):
    try:
        run_example(example_file, short_exec=True, headless=True)
    except Exception as e:
        pytest.fail(f"Example {example_file} failed with error: {str(e)}")
