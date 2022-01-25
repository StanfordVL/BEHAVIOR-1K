"""
    Generate example top-down segmentation map via renderer
"""
import logging

from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.scenes.interactive_traversable_scene import InteractiveIndoorScene
from igibson.simulator import Simulator


def main(random_selection=False, headless=False, short_exec=False):
    """
    Highlights visually all object instances of some given category and then removes the highlighting
    It also demonstrates how to apply an action on all instances of objects of a given category
    ONLY WORKS WITH OPTIMIZED RENDERING (not on Mac)
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    settings = MeshRendererSettings(optimized=True, enable_shadow=True, blend_highlight=True)
    s = Simulator(
        mode="gui_interactive" if not headless else "headless",
        image_width=512,
        image_height=512,
        rendering_settings=settings,
    )
    scene = InteractiveIndoorScene(
        "Rs_int", texture_randomization=False, load_object_categories=["window"], object_randomization=False
    )
    s.import_scene(scene)

    i = 0
    max_steps = -1 if not short_exec else 1000
    while i != max_steps:
        s.step()
        if i % 100 == 0:
            logging.info("Highlighting windows")
            for obj in scene.objects_by_category["window"]:
                obj.highlight()

        if i % 100 == 50:
            logging.info("Deactivating the highlight on windows")
            for obj in scene.objects_by_category["window"]:
                obj.unhighlight()

        i += 1


if __name__ == "__main__":
    main()
