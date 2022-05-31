import logging
import os

import numpy as np

import igibson
from igibson import object_states
from igibson.objects.usd_object import URDFObject
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import download_assets

download_assets()


def main(random_selection=False, headless=False, short_exec=False):
    """
    Demo to use the raycasting-based sampler to load objects onTop and/or inside another
    Loads a cabinet, a microwave open on top of it, and two plates with apples on top, one inside and one on top of the cabinet
    Then loads a shelf and YCB cracker boxes inside of it
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    settings = MeshRendererSettings(enable_shadow=True, msaa=False, optimized=False)
    s = Simulator(
        mode="gui_interactive" if not headless else "headless",
        image_width=512,
        image_height=512,
        rendering_settings=settings,
    )
    scene = EmptyScene(floor_plane_rgba=[0.6, 0.6, 0.6, 1])
    s.import_scene(scene)

    # Set a better viewing direction
    if not headless:
        s.viewer.initial_pos = [-0.2, -1.3, 2.7]
        s.viewer.initial_view_direction = [0.3, 0.8, -0.5]
        s.viewer.reset_viewer()

    try:
        sample_microwave_plates_apples(s)
        sample_boxes_on_shelf(s)

        max_steps = 100 if short_exec else -1
        step = 0
        while step != max_steps:
            s.step()
            step += 1
    finally:
        s.disconnect()


def sample_microwave_plates_apples(simulator):
    cabinet_filename = os.path.join(igibson.assets_path, "models/cabinet2/cabinet_0007.urdf")

    microwave_dir = os.path.join(igibson.ig_dataset_path, "objects/microwave/7128/")
    microwave_filename = os.path.join(microwave_dir, "7128.urdf")

    plate_dir = os.path.join(igibson.ig_dataset_path, "objects/plate/plate_000/")
    plate_filename = os.path.join(plate_dir, "plate_000.urdf")

    apple_dir = os.path.join(igibson.ig_dataset_path, "objects/apple/00_0/")
    apple_filename = os.path.join(apple_dir, "00_0.urdf")

    # Load cabinet, set position manually, and step 100 times
    logging.info("Loading cabinet")
    cabinet = URDFObject(filename=cabinet_filename, category="cabinet", scale=np.array([2.0, 2.0, 2.0]))
    simulator.import_object(cabinet)
    cabinet.set_position([0, 0, 0.5])
    for _ in range(100):
        simulator.step()

    # Load microwave, set position on top of the cabinet, open it, and step 100 times
    logging.info("Loading microwave Open and OnTop of the cabinet")
    microwave = URDFObject(
        filename=microwave_filename, category="microwave", model_path=microwave_dir, scale=np.array([0.5, 0.5, 0.5])
    )
    simulator.import_object(microwave)
    assert microwave.states[object_states.OnTop].set_value(cabinet, True, use_ray_casting_method=True)
    assert microwave.states[object_states.Open].set_value(True)
    logging.info("Microwave loaded and placed")
    for _ in range(100):
        simulator.step()

    logging.info("Loading plates")
    for i in range(3):
        plate = URDFObject(
            filename=plate_filename, category="plate", model_path=plate_dir, bounding_box=[0.25, 0.25, 0.05]
        )
        simulator.import_object(plate)

        # Put the 1st plate in the microwave
        if i == 0:
            logging.info("Loading plate Inside the microwave")
            assert plate.states[object_states.Inside].set_value(microwave, True, use_ray_casting_method=True)
        else:
            logging.info("Loading plate OnTop the microwave")
            assert plate.states[object_states.OnTop].set_value(microwave, True, use_ray_casting_method=True)

        logging.info("Plate %d loaded and placed." % i)
        for _ in range(100):
            simulator.step()

        logging.info("Loading three apples OnTop of the plate")
        for j in range(3):
            apple = URDFObject(filename=apple_filename, category="apple", model_path=apple_dir)
            simulator.import_object(apple)
            assert apple.states[object_states.OnTop].set_value(plate, True, use_ray_casting_method=True)
            logging.info("Apple %d loaded and placed." % j)
            for _ in range(100):
                simulator.step()


def sample_boxes_on_shelf(simulator):
    shelf_dir = os.path.join(igibson.ig_dataset_path, "objects/shelf/1170df5b9512c1d92f6bce2b7e6c12b7/")
    shelf_filename = os.path.join(shelf_dir, "1170df5b9512c1d92f6bce2b7e6c12b7.urdf")

    cracker_box_dir = os.path.join(igibson.ig_dataset_path, "objects/cracker_box/cracker_box_000/")
    cracker_box_filename = os.path.join(cracker_box_dir, "cracker_box_000.urdf")

    shelf = URDFObject(
        filename=shelf_filename, category="shelf", model_path=shelf_dir, bounding_box=np.array([1.0, 0.4, 2.0])
    )
    simulator.import_object(shelf)
    shelf.set_position([2, 2, 1])
    for _ in range(100):
        simulator.step()
    print("Shelf placed")

    for i in range(10):
        cracker_box = URDFObject(
            filename=cracker_box_filename,
            category="cracker_box",
            model_path=cracker_box_dir,
            bounding_box=np.array([0.2, 0.05, 0.3]),
        )
        simulator.import_object(cracker_box)
        cracker_box.states[object_states.Inside].set_value(shelf, True, use_ray_casting_method=True)

        print("Box %d placed." % i)

        for _ in range(100):
            simulator.step()


if __name__ == "__main__":
    main()
