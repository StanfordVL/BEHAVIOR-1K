"""
Generate example top-down segmentation map via renderer
"""
import numpy as np
import omnigibson as og


def main(random_selection=False, headless=False, short_exec=False):
    """
    Highlights visually all object instances of some given category and then removes the highlighting
    It also demonstrates how to apply an action on all instances of objects of a given category
    ONLY WORKS WITH OPTIMIZED RENDERING (not on Mac)
    """
    og.log.info(f"Demo {__file__}\n    " + "*" * 80 + "\n    Description:\n" + main.__doc__ + "*" * 80)

    # Create the scene config to load -- empty scene
    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
        }
    }

    # Create the environment
    env = og.Environment(configs=cfg)

    # Grab all window objects
    windows = og.sim.scene.object_registry("category", "window")

    # Step environment while toggling window highlighting
    i = 0
    highlighted = False
    max_steps = -1 if not short_exec else 1000
    while i != max_steps:
        env.step(np.array([]))
        if i % 50 == 0:
            highlighted = not highlighted
            og.log.info(f"Toggling window highlight to: {highlighted}")
            for window in windows:
                # Note that this property is R/W!
                window.highlighted = highlighted
        i += 1

    # Always close the environment at the end
    env.close()


if __name__ == "__main__":
    main()
