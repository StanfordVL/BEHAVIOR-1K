import importlib
import os
from unittest.mock import patch

import pytest

# Must be set before omnigibson is imported so that gm.HEADLESS is True
os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")

# Explicit list of examples to test. In CI each example runs in its own matrix
# job (isolated process), so the Isaac Sim singleton is not an issue. When
# running locally always use -k to run a single example at a time.
EXAMPLES = [
    # --- BEGIN AUTO-GENERATED EXAMPLES ---
    "environments.navigation_env_demo",
    "environments.vector_env_demo",
    "object_states.attachment_demo",
    "object_states.dicing_demo",
    "object_states.folded_unfolded_state_demo",
    "object_states.heat_source_or_sink_demo",
    "object_states.heated_state_demo",
    "object_states.onfire_demo",
    "object_states.overlaid_demo",
    "object_states.particle_applier_remover_demo",
    "object_states.particle_source_sink_demo",
    "object_states.sample_kinematics_demo",
    "object_states.slicing_demo",
    "object_states.temperature_demo",
    "objects.draw_bounding_box",
    "objects.highlight_objects",
    "objects.load_object_selector",
    "objects.view_cloth_configurations",
    "objects.visualize_object",
    "robots.all_robots_visualizer",
    "robots.grasping_mode_example",
    "robots.import_custom_robot",
    "robots.robot_control_example",
    "scenes.scene_selector",
    "scenes.scene_tour_demo",
    "scenes.traversability_map_example",
    "simulator.sim_save_load_example",
    # --- END AUTO-GENERATED EXAMPLES ---
]

# Examples excluded from automated testing
EXAMPLES_TO_SKIP = [
    "action_primitives.rs_int_example",
    "action_primitives.solve_simple_task",
    "action_primitives.wip_solve_behavior_task",
    "environments.behavior_env_demo",  # requires pre-sampled cached BEHAVIOR activity scene
    "learning.navigation_policy_demo",
    "teleoperation.robot_teleoperate_demo",
    "teleoperation.vr_robot_control_demo",  # does not support headless mode
    "teleoperation.vr_scene_tour_demo",  # does not support headless mode
    "robots.curobo_example",  # requires CuRobo and CUDA support
    "objects.import_custom_object",  # CLI conversion tool, requires demo / test asset files
    "object_states.object_state_texture_demo",  # disable temporarily due to contact API bug
]


@pytest.mark.parametrize("example_name", EXAMPLES)
def test_example(example_name, request):
    import click

    module = importlib.import_module(f"omnigibson.examples.{example_name}")
    test_args = request.config.getoption("--test-args", default="")

    if isinstance(module.main, click.BaseCommand):
        from click.testing import CliRunner

        args = test_args.split() if test_args else []
        runner = CliRunner()
        result = runner.invoke(module.main, args, catch_exceptions=False)
        assert result.exit_code == 0, result.output
    else:
        with patch("omnigibson.shutdown"):
            module.main(random_selection=True, headless=True, short_exec=True)
