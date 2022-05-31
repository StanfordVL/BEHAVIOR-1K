import logging
from sys import platform

from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.scenes.interactive_traversable_scene import InteractiveIndoorScene
from igibson.simulator import Simulator


def main(random_selection=False, headless=False, short_exec=False):
    """
    Example of partial loading of a scene
    Loads only some objects (by category) and in some room types
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    settings = MeshRendererSettings(enable_shadow=True, msaa=False)
    if platform == "darwin":
        settings.texture_scale = 0.5
    s = Simulator(
        mode="gui_interactive" if not headless else "headless",
        image_width=512,
        image_height=512,
        rendering_settings=settings,
    )
    scene = InteractiveIndoorScene(
        "Rs_int",
        texture_randomization=False,
        object_randomization=False,
        load_object_categories=["swivel_chair"],
        load_room_types=["living_room"],
    )
    s.import_scene(scene)

    max_steps = -1 if not short_exec else 1000
    step = 0
    while step != max_steps:
        s.step()
        step += 1


if __name__ == "__main__":
    main()
