import subprocess
import sys

import pytest

from omnigibson.utils.asset_utils import download_omnigibson_robot_assets

# Explicit list of examples to test. Each example is run in a subprocess to
# work around Isaac Sim being a singleton (can't be instantiated twice in the
# same process).
EXAMPLES = [
    # --- BEGIN AUTO-GENERATED EXAMPLES ---
    "environments.navigation_env_demo",
    "environments.vector_env_demo",
    "objects.draw_bounding_box",
    "objects.highlight_objects",
    "objects.import_custom_object",
    "objects.load_object_selector",
    "objects.view_cloth_configurations",
    "objects.visualize_object",
    "object_states.attachment_demo",
    "object_states.dicing_demo",
    "object_states.folded_unfolded_state_demo",
    "object_states.heated_state_demo",
    "object_states.heat_source_or_sink_demo",
    "object_states.object_state_texture_demo",
    "object_states.onfire_demo",
    "object_states.overlaid_demo",
    "object_states.particle_applier_remover_demo",
    "object_states.particle_source_sink_demo",
    "object_states.sample_kinematics_demo",
    "object_states.slicing_demo",
    "object_states.temperature_demo",
    "robots.all_robots_visualizer",
    "robots.curobo_example",
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
EXAMPLES_SKIP_REASONS = {
    "action_primitives.rs_int_example": "requires full BEHAVIOR scene setup",
    "action_primitives.solve_simple_task": "requires full BEHAVIOR scene setup",
    "action_primitives.wip_solve_behavior_task": "work in progress",
    "environments.behavior_env_demo": "requires pre-sampled cached BEHAVIOR activity scene",
    "learning.navigation_policy_demo": "requires trained policy checkpoint",
    "teleoperation.robot_teleoperate_demo": "requires teleoperation hardware",
    "teleoperation.vr_robot_control_demo": "requires VR hardware",
    "teleoperation.vr_scene_tour_demo": "requires VR hardware",
}


@pytest.fixture(scope="session", autouse=True)
def download_assets():
    download_omnigibson_robot_assets()


@pytest.mark.parametrize("example_name", EXAMPLES)
def test_example(example_name):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import omnigibson.examples.{example_name} as m; "
            f"m.main(random_selection=True, headless=True, short_exec=True)",
        ],
        timeout=600,
    )
    assert result.returncode == 0, f"Example {example_name} exited with return code {result.returncode}"
