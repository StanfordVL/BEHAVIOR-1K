import numpy as np
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.scene_graphs.graph_builder import visualize_scene_graph
from omnigibson.utils.asset_utils import get_available_g_scenes, get_available_og_scenes
from omnigibson.utils.ui_utils import choose_from_options

# Configure macros for maximum performance
gm.USE_GPU_DYNAMICS = True
gm.ENABLE_FLATCACHE = True
gm.ENABLE_OBJECT_STATES = True
gm.ENABLE_TRANSITION_RULES = True


def main(random_selection=False, headless=False, short_exec=False):
    """
    Prompts the user to select any available interactive scene and loads a turtlebot into it.
    It steps the environment 100 times with random actions sampled from the action space,
    using the Gym interface, resetting it 10 times.
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Choose the scene type to load
    scene_options = {
        "InteractiveTraversableScene": "Procedurally generated scene with fully interactive objects",
        # "StaticTraversableScene": "Monolithic scene mesh with no interactive objects",
    }
    scene_type = "InteractiveTraversableScene"  # choose_from_options(options=scene_options, name="scene type", random_selection=random_selection)

    # Choose the scene model to load
    scenes = get_available_og_scenes() if scene_type == "InteractiveTraversableScene" else get_available_g_scenes()
    scene_model = "house_single_floor" # choose_from_options(options=scenes, name="scene model", random_selection=random_selection)

    cfg = {
        "scene": {
            "type": scene_type,
            "scene_model": scene_model,
        },
        "scene_graph": {
            "egocentric": True,
            "full_obs": True,
            "only_true": True,
            "merge_parallel_edges": False,
        },
        "robots": [
            {"type": "Fetch"}
        ]
    }

    # If the scene type is interactive, also check if we want to quick load or full load the scene
    if scene_type == "InteractiveTraversableScene":
        load_options = {
            "Quick": "Only load the building assets (i.e.: the floors, walls, ceilings)",
            "Full": "Load all interactive objects in the scene",
        }
        load_mode = "Full" # choose_from_options(options=load_options, name="load mode", random_selection=random_selection)
        if load_mode == "Quick":
            cfg["scene"]["load_object_categories"] = ["floors", "walls", "ceilings"]

    # Load the environment
    env = og.Environment(configs=cfg)

    # Allow user to move camera more easily
    if not gm.HEADLESS:
        og.sim.enable_viewer_camera_teleoperation()

    # Run a simple loop and reset periodically
    while True:
        env.step(np.zeros(env.robots[0].action_dim))
    G = env.get_scene_graph()
    # visualize_scene_graph(env.scene, G, True, True)
    print(G.edges)


if __name__ == "__main__":
    main()
