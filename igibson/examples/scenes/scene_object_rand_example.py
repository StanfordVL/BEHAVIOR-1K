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
    settings = MeshRendererSettings(enable_shadow=True, msaa=False)
    if platform == "darwin":
        settings.texture_scale = 0.5
    s = Simulator(
        mode="gui_interactive" if not headless else "headless",
        image_width=512,
        image_height=512,
        rendering_settings=settings,
    )
    load_object_categories = None  # Use the list to load a partial house
    # load_object_categories = [
    #     "bottom_cabinet",
    #     "sofa",
    #     "rocking_chair",
    #     "swivel_chair",
    #     "folding_chair",
    #     "desk",
    #     "dining_table",
    #     "coffee_table",
    #     "breakfast_table",
    # ]  # Use None to load the full house

    num_resets = 10 if not short_exec else 2
    num_steps_per_reset = 1000 if not short_exec else 10
    for i in range(num_resets):
        logging.info("Randomize objects")
        scene = InteractiveIndoorScene(
            "Rs_int",
            texture_randomization=False,
            load_object_categories=load_object_categories,
            object_randomization=True,
            object_randomization_idx=i,
        )
        s.import_scene(scene)
        for _ in range(num_steps_per_reset):
            s.step()
        s.reload()

    s.disconnect()


if __name__ == "__main__":
    main()
