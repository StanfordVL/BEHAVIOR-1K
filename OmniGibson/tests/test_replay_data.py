import importlib.util
import json
import os
import shutil
import tempfile

import pytest

from omnigibson.macros import gm

_TESTS_DIR = os.path.dirname(__file__)
_JOYLO_REPLAY_SCRIPT = os.path.abspath(os.path.join(_TESTS_DIR, "..", "..", "joylo", "scripts", "replay_data.py"))
_HDF5_PATH = os.path.join(_TESTS_DIR, "data", "vacuuming_floors.hdf5")
_GOLDEN_PATH = os.path.join(_TESTS_DIR, "data", "golden", "vacuuming_floors_qa_results.json")

_DATA_2025 = os.path.join(gm.DATA_PATH, "2025-challenge-task-instances", "metadata", "available_tasks.yaml")
_DATA_2026 = os.path.join(gm.DATA_PATH, "2026-challenge-task-instances", "metadata", "available_tasks.yaml")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(_DATA_2025) and os.path.exists(_DATA_2026)),
    reason="Challenge task instance datasets not present",
)


def _import_replay_data():
    spec = importlib.util.spec_from_file_location("joylo_replay_data", _JOYLO_REPLAY_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_replay_data_vacuuming_floors_qa():
    """Replays vacuuming_floors.hdf5 with QA and checks results match the golden file."""
    import omnigibson as og

    replay_data = _import_replay_data()

    gm.ENABLE_OBJECT_STATES = True

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_hdf5 = os.path.join(tmpdir, "vacuuming_floors.hdf5")
        shutil.copy(_HDF5_PATH, tmp_hdf5)

        replay_data.replay_hdf5_to_video(
            input_path=tmp_hdf5,
            task_name="vacuuming_floors",
            flush_every_n_steps=1000,
            run_qa=True,
            episode_id=0,
        )

        qa_output_path = os.path.join(tmpdir, "vacuuming_floors_qa_results.json")
        assert os.path.exists(qa_output_path), "QA results file was not created"

        with open(qa_output_path) as f:
            actual = json.load(f)

    with open(_GOLDEN_PATH) as f:
        expected = json.load(f)

    assert (
        actual == expected
    ), f"QA results differ from golden.\nActual: {json.dumps(actual, indent=2)}\nExpected: {json.dumps(expected, indent=2)}"

    og.clear()


if __name__ == "__main__":
    test_replay_data_vacuuming_floors_qa()
