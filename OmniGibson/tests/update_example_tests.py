"""
Updates the example test list in test_examples.py and tests.yml to match the
current set of examples in omnigibson/examples/.

Run this script whenever examples are added or removed:
    python tests/update_example_tests.py

In CI, run with --check to fail if the files would change:
    python tests/update_example_tests.py --check
"""

import argparse
import pkgutil
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
OMNIGIBSON_ROOT = Path(__file__).parent.parent
TEST_EXAMPLES_PY = Path(__file__).parent / "test_examples.py"
TESTS_YML = REPO_ROOT / ".github" / "workflows" / "tests.yml"

# Markers used to delimit the auto-generated sections
EXAMPLES_START = "    # --- BEGIN AUTO-GENERATED EXAMPLES ---"
EXAMPLES_END = "    # --- END AUTO-GENERATED EXAMPLES ---"
INCLUDE_START = "# --- BEGIN AUTO-GENERATED EXAMPLE INCLUDES ---"
INCLUDE_END = "# --- END AUTO-GENERATED EXAMPLE INCLUDES ---"


def discover_examples():
    """Return sorted list of all example module names (e.g. 'objects.draw_bounding_box')."""
    sys.path.insert(0, str(OMNIGIBSON_ROOT))
    from omnigibson import examples

    prefix = examples.__name__ + "."
    return sorted(p.name[len(prefix) :] for p in pkgutil.walk_packages(examples.__path__, prefix) if not p.ispkg)


def get_skip_reasons(content):
    """Parse EXAMPLES_SKIP_REASONS dict from test_examples.py source."""
    match = re.search(r"EXAMPLES_SKIP_REASONS\s*=\s*\{([^}]*)\}", content, re.DOTALL)
    if not match:
        return {}
    skip_reasons = {}
    for line in match.group(1).splitlines():
        m = re.match(r'\s*"([^"]+)"\s*:\s*"([^"]+)"', line)
        if m:
            skip_reasons[m.group(1)] = m.group(2)
    return skip_reasons


def generate_examples_list(examples, skip_reasons):
    """Generate content between (and including) the marker lines."""
    lines = [EXAMPLES_START]
    for name in examples:
        if name not in skip_reasons:
            lines.append(f'    "{name}",')
    lines.append(EXAMPLES_END)
    return "\n".join(lines)


def generate_yml_includes(examples, skip_reasons):
    """Generate the matrix include entries for test_examples."""
    lines = [f"        {INCLUDE_START}"]
    for name in examples:
        if name not in skip_reasons:
            lines.append(f'          - {{test_file: test_examples, example: "{name}"}}')
    lines.append(f"        {INCLUDE_END}")
    return "\n".join(lines)


def replace_section(content, start_marker, end_marker, new_block):
    """Replace from the start marker line to the end marker line (inclusive)."""
    pattern = re.compile(
        rf"^[^\n]*{re.escape(start_marker)}[^\n]*$.*?^[^\n]*{re.escape(end_marker)}[^\n]*$",
        re.MULTILINE | re.DOTALL,
    )
    if not pattern.search(content):
        raise ValueError(f"Could not find markers '{start_marker}' / '{end_marker}'")
    return pattern.sub(new_block, content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Exit with error if files would change")
    args = parser.parse_args()

    all_examples = discover_examples()

    # --- Update test_examples.py ---
    py_content = TEST_EXAMPLES_PY.read_text()
    skip_reasons = get_skip_reasons(py_content)

    new_examples_block = generate_examples_list(all_examples, skip_reasons)
    new_py_content = replace_section(py_content, EXAMPLES_START, EXAMPLES_END, new_examples_block)

    # --- Update tests.yml ---
    yml_content = TESTS_YML.read_text()
    new_includes = generate_yml_includes(all_examples, skip_reasons)
    new_yml_content = replace_section(yml_content, INCLUDE_START, INCLUDE_END, new_includes)

    if args.check:
        changed = []
        if new_py_content != py_content:
            changed.append(str(TEST_EXAMPLES_PY.relative_to(REPO_ROOT)))
        if new_yml_content != yml_content:
            changed.append(str(TESTS_YML.relative_to(REPO_ROOT)))
        if changed:
            print("The following files are out of date. Run `python tests/update_example_tests.py` to fix:")
            for f in changed:
                print(f"  {f}")
            sys.exit(1)
        else:
            print("Example test lists are up to date.")
    else:
        TEST_EXAMPLES_PY.write_text(new_py_content)
        TESTS_YML.write_text(new_yml_content)
        print(f"Updated {TEST_EXAMPLES_PY.relative_to(REPO_ROOT)}")
        print(f"Updated {TESTS_YML.relative_to(REPO_ROOT)}")
        skipped = [e for e in all_examples if e in skip_reasons]
        included = [e for e in all_examples if e not in skip_reasons]
        print(f"  {len(included)} examples included, {len(skipped)} skipped")


if __name__ == "__main__":
    main()
