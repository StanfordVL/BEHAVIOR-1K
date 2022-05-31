import logging
from sys import platform

from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.scenes.interactive_traversable_scene import InteractiveIndoorScene
from igibson.simulator import Simulator


def main(random_selection=False, headless=False, short_exec=False):
    """
    Example of randomization of the texture in a scene
    Loads Rs_int (interactive) and randomizes the texture of the objects
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    if platform == "darwin":
        logging.error("Texture randomization currently works only with optimized renderer. Mac OS does not support it")
    settings = MeshRendererSettings(enable_shadow=False, msaa=False)
    s = Simulator(
        mode="gui_interactive" if not headless else "headless",
        image_width=512,
        image_height=512,
        rendering_settings=settings,
    )
    scene = InteractiveIndoorScene(
        "Rs_int",
        # load_object_categories=[],  # To load only the building. Fast
        texture_randomization=True,
        object_randomization=False,
    )
    s.import_scene(scene)

    num_resets = 10 if not short_exec else 2
    num_steps_per_reset = 1000 if not short_exec else 10
    for _ in range(num_resets):
        logging.info("Randomize texture")
        scene.randomize_texture()
        for _ in range(num_steps_per_reset):
            s.step()
    s.disconnect()


if __name__ == "__main__":
    main()
