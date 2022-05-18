import numpy as np

import igibson
from igibson import object_states, app, ig_dataset_path
from igibson.objects.dataset_object import DatasetObject
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator_omni import Simulator
from igibson.transition_rules import *


def test_apple_slicing():
    try:
        sim = Simulator()
        scene = EmptyScene()
        sim.import_scene(scene)

        obj_category = "apple"
        obj_model = "00_0"
        name = "apple"

        model_root_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}"
        usd_path = f"{model_root_path}/usd/{obj_model}.usd"

        # Create a dataset object of an apple, but doesn't load it in the simulator.
        whole_obj = DatasetObject(
            prim_path=f"/World/{name}",
            usd_path=usd_path,
            category=obj_category,
            name=f"{name}",
            scale=np.array([10.0, 10.0, 10.0]),
            abilities={"sliceable": {}},
        )

        obj_category = "table_knife"
        obj_model = "1"
        name = "knife"

        model_root_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}"
        usd_path = f"{model_root_path}/usd/{obj_model}.usd"

        # Create a dataset object of an apple, but doesn't load it in the simulator.
        knife = DatasetObject(
            prim_path=f"/World/{name}",
            usd_path=usd_path,
            category=obj_category,
            name=f"{name}",
            scale=np.array([15.0, 15.0, 15.0]),
            abilities={"slicer": {}},
        )

        sim.import_object(whole_obj, auto_initialize=True)
        whole_obj.set_position_orientation(position=np.array([0, 0, 0]))

        sim.import_object(knife, auto_initialize=True)
        knife.set_position_orientation(position=np.array([0, 0, 0]))

        # Need 1 physics step to activate collision meshes for raycasting.
        sim.step(force_playing=True)

        assert not whole_obj.states[object_states.Sliced].get_value()

        for i in range(10):
            sim.step(apply_transitions=True)

        assert sum("apple_part" in obj.name for obj in sim.scene.objects) == 2
    finally:
        app.close()


test_apple_slicing()
