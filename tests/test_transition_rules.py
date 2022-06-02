import numpy as np

from igibson import object_states, app, ig_dataset_path
from igibson.objects.dataset_object import DatasetObject
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator_omni import Simulator
import igibson.transition_rules as transition_rules


def create_dataset_object(category, model, name, scale, abilities=dict()):
    model_root_path = f"{ig_dataset_path}/objects/{category}/{model}"
    usd_path = f"{model_root_path}/usd/{model}.usd"

    # Create a dataset object, but doesn't load it in the simulator.
    dataset_object = DatasetObject(
        prim_path=f"/World/{name}",
        usd_path=usd_path,
        category=category,
        name=f"{name}",
        scale=scale,
        abilities=abilities,
    )
    return dataset_object


def test_apple_slicing():
    try:
        sim = Simulator()
        scene = EmptyScene()
        sim.import_scene(scene)

        apple_obj = create_dataset_object(
            category="apple", model="00_0", name="apple",
            scale=np.array([10.0, 10.0, 10.0]), abilities={"sliceable": {}})

        knife_obj = create_dataset_object(
            category="table_knife", model="1", name="knife",
            scale=np.array([15.0, 15.0, 15.0]), abilities={"slicer": {}})

        sim.import_object(apple_obj, auto_initialize=True)
        apple_obj.set_position_orientation(position=np.array([0, 0, 0]))

        sim.import_object(knife_obj, auto_initialize=True)
        knife_obj.set_position_orientation(position=np.array([0, 0, 0]))

        # Need 1 physics step to activate collision meshes for raycasting.
        sim.step(force_playing=True)

        assert not apple_obj.states[object_states.Sliced].get_value()

        for i in range(10000):
            sim.step(apply_transitions=True)

        assert sum("apple_part" in obj.name for obj in sim.scene.objects) == 2
    finally:
        app.close()


def test_container():
    try:
        sim = Simulator()
        scene = EmptyScene()
        sim.import_scene(scene)

        tea_bag_obj = create_dataset_object(
            category="tea_bag", model="tea_bag_000", name="tea_bag",
            scale=np.array([15.0, 15.0, 15.0]))

        container_obj = create_dataset_object(
            category="bowl", model="68_0", name="bowl",
            scale=np.array([15.0, 15.0, 15.0]))

        apple_obj_attrs = transition_rules.ObjectAttrs(
            category="apple", model="00_0", name="apple",
            scale=np.array([10.0, 10.0, 10.0]))
        container_rule = transition_rules.ContainerRule(
            3, apple_obj_attrs,
            transition_rules.CategoryFilter("bowl"),
            transition_rules.CategoryFilter("tea_bag"))
        transition_rules.DEFAULT_RULES.append(container_rule)

        sim.import_object(tea_bag_obj, auto_initialize=True)
        tea_bag_obj.set_position_orientation(position=np.array([0, 0, 2.0]))

        sim.import_object(container_obj, auto_initialize=True)
        container_obj.set_position_orientation(position=np.array([0, 0, 1.0]))

        # Need 1 physics step to activate collision meshes for raycasting.
        sim.step(force_playing=True)
        for i in range(3000):
            if i % 1000 == 0:
                print(3 - i//1000)
            sim.step(apply_transitions=False)

        for i in range(10000):
            sim.step(apply_transitions=True)
    finally:
        app.close()


def test_container_garbage():
    try:
        sim = Simulator()
        scene = EmptyScene()
        sim.import_scene(scene)

        tea_bag_obj = create_dataset_object(
            category="tea_bag", model="tea_bag_000", name="tea_bag",
            scale=np.array([15.0, 15.0, 15.0]))

        container_obj = create_dataset_object(
            category="bowl", model="68_0", name="bowl",
            scale=np.array([15.0, 15.0, 15.0]))

        knife_obj_attrs = transition_rules.ObjectAttrs(
            category="table_knife", model="1", name="knife",
            scale=np.array([10.0, 10.0, 10.0]))
        garbage_rule = transition_rules.ContainerGarbageRule(
            knife_obj_attrs, transition_rules.CategoryFilter("bowl"))
        transition_rules.DEFAULT_RULES.append(garbage_rule)

        sim.import_object(tea_bag_obj, auto_initialize=True)
        tea_bag_obj.set_position_orientation(position=np.array([0, 0, 2.0]))

        sim.import_object(container_obj, auto_initialize=True)
        container_obj.set_position_orientation(position=np.array([0, 0, 1.0]))

        # Need 1 physics step to activate collision meshes for raycasting.
        sim.step(force_playing=True)
        for i in range(3000):
            if i % 1000 == 0:
                print(3 - i // 1000)
            sim.step(apply_transitions=False)

        for i in range(10000):
            sim.step(apply_transitions=True)
    finally:
        app.close()


# test_apple_slicing()
# test_container()
test_container_garbage()
