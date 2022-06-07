import os

import networkx as nx
import numpy as np


import igibson
from igibson import object_states, app, ig_dataset_path
# from igibson.external.pybullet_tools.utils import Euler, quat_from_euler
from igibson.object_states.factory import get_state_dependency_graph, get_states_by_dependency_order
from igibson.objects.dataset_object import DatasetObject
from igibson.objects.ycb_object import YCBObject
from igibson.objects.primitive_object import PrimitiveObject
from igibson.scenes.empty_scene import EmptyScene
from igibson.sensors.vision_sensor import VisionSensor
from igibson.simulator_omni import Simulator
from igibson.utils.assets_utils import download_assets, get_ig_model_path
from igibson.utils.usd_utils import create_joint
from pxr import Gf

#download_assets()


def test_on_top():
    s = Simulator(mode="headless")

    try:
        scene = EmptyScene()
        s.import_scene(scene)

        cabinet_0007 = os.path.join(igibson.assets_path, "models/cabinet2/usd/cabinet_0007.usd")
        cabinet_0004 = os.path.join(igibson.assets_path, "models/cabinet/usd/cabinet_0004.usd")

        obj1 = DatasetObject(usd_path=cabinet_0007)
        s.import_object(obj1)
        obj1.set_position([0, 0, 0.5])

        obj2 = DatasetObject(usd_path=cabinet_0004)
        s.import_object(obj2)
        obj2.set_position([0, 0, 2])

        obj3 = YCBObject("003_cracker_box")
        s.import_object(obj3)
        obj3.set_position_orientation([0, 0, 1.1], [0, 0, 0, 1])

        # Run simulation for 1000 steps
        for _ in range(1000):
            s.step()

        # Now check that the box is on top of the lower cabinet
        assert obj3.states[object_states.Touching].get_value(obj1)
        assert obj3.states[object_states.OnTop].get_value(obj1)
        assert not obj3.states[object_states.Inside].get_value(obj1)

        # Now check that the box is not on top / touching of the upper cabinet
        assert not obj3.states[object_states.Touching].get_value(obj2)
        assert not obj3.states[object_states.OnTop].get_value(obj2)
        assert not obj3.states[object_states.Inside].get_value(obj2)
    finally:
        s.disconnect()


def test_inside():
    s = Simulator(mode="headless")

    try:
        scene = EmptyScene()
        s.import_scene(scene)

        cabinet_0007 = os.path.join(igibson.assets_path, "models/cabinet2/usd/cabinet_0007.usd")
        cabinet_0004 = os.path.join(igibson.assets_path, "models/cabinet/usd/cabinet_0004.usd")

        obj1 = DatasetObject(usd_path=cabinet_0007)
        s.import_object(obj1)
        obj1.set_position([0, 0, 0.5])

        obj2 = DatasetObject(usd_path=cabinet_0004)
        s.import_object(obj2)
        obj2.set_position([0, 0, 2])

        obj3 = YCBObject("003_cracker_box")
        s.import_object(obj3)
        obj3.set_position_orientation([0, 0, 2.1], [0, 0, 0, 1])

        # Run simulation for 1000 steps
        for _ in range(100):
            s.step()

        # Check that the box is not inside / touching the lower cabinet
        assert not obj3.states[object_states.Touching].get_value(obj1)
        assert not obj3.states[object_states.Inside].get_value(obj1)
        assert not obj3.states[object_states.OnTop].get_value(obj1)

        # Now check that the box is inside / touching the upper cabinet
        assert obj3.states[object_states.Touching].get_value(obj2)
        assert obj3.states[object_states.Inside].get_value(obj2)

        # Open the doors of the cabinet and check that this still holds.
        for joint_id in [0, 1]:
            max_pos = p.getJointInfo(obj2.get_body_ids()[0], joint_id)[9]
            p.resetJointState(obj2.get_body_ids()[0], joint_id, max_pos)
        s.step()
        assert obj3.states[object_states.Touching].get_value(obj2)
        assert obj3.states[object_states.Inside].get_value(obj2)

        # Now rotate the cabinet to see if inside checking still succeeds.
        angles = np.linspace(0, np.pi / 2, 20)
        for angle in angles:
            obj2.set_orientation(quat_from_euler(Euler(yaw=angle)))
            s.step()
            assert obj3.states[object_states.Touching].get_value(obj2)
            assert obj3.states[object_states.Inside].get_value(obj2)
    finally:
        s.disconnect()


def test_open():
    s = Simulator(mode="headless")

    try:
        scene = EmptyScene()
        s.import_scene(scene)

        model_path = os.path.join(igibson.ig_dataset_path, "objects/microwave/7128/usd/7128.usd")
        obj = DatasetObject(
            usd_path=model_path,
            category="microwave",
            name="microwave_1",
            scale=np.array([0.5, 0.5, 0.5]),
            abilities={"openable": {}},
        )
        s.import_object(obj)
        obj.set_position([0, 0, 0.5])

        # --------------------------------------------
        # PART 1: Run with joints at default position.
        # --------------------------------------------
        # Check that the microwave is not open.
        assert not obj.states[object_states.Open].get_value()

        # --------------------------------------------
        # PART 2: Set non-whitelisted joint to the max position
        # --------------------------------------------
        joint_id = 2
        max_pos = p.getJointInfo(obj.get_body_ids()[0], joint_id)[9]
        p.resetJointState(obj.get_body_ids()[0], joint_id, max_pos)
        s.step()

        # Check that the microwave is not open.
        assert not obj.states[object_states.Open].get_value()

        # --------------------------------------------
        # PART 3: Set whitelisted joint to the max position
        # --------------------------------------------
        joint_id = 0
        max_pos = p.getJointInfo(obj.get_body_ids()[0], joint_id)[9]
        p.resetJointState(obj.get_body_ids()[0], joint_id, max_pos)
        s.step()

        # Check that the microwave is open.
        assert obj.states[object_states.Open].get_value()

        # --------------------------------------------
        # PART 4: Now try sampling a closed position.
        # --------------------------------------------
        obj.states[object_states.Open].set_value(False)
        s.step()

        # Check that the microwave is closed.
        assert not obj.states[object_states.Open].get_value()

        # --------------------------------------------
        # PART 5: Finally, sample an open position.
        # --------------------------------------------
        obj.states[object_states.Open].set_value(True)
        s.step()

        # Check that the microwave is open.
        assert obj.states[object_states.Open].get_value()
    finally:
        s.disconnect()


def test_state_graph():
    # Construct the state graph
    G = get_state_dependency_graph()
    assert nx.algorithms.is_directed_acyclic_graph(G), "State dependency graph needs to be a DAG."

    # Get the dependency-sorted list of states.
    ordered_states = get_states_by_dependency_order()
    assert object_states.Inside in ordered_states
    assert object_states.AABB in ordered_states
    assert ordered_states.index(object_states.AABB) < ordered_states.index(
        object_states.Inside
    ), "Each state should be preceded by its deps."

    app.close()


def test_toggle():
    try:
        obj_category = "sink"
        obj_model = "sink_1"
        name = "sink"

        sim = Simulator()
        scene = EmptyScene()
        sim.import_scene(scene)

        model_root_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}"
        usd_path = f"{model_root_path}/usd/{obj_model}.usd"

        sink = DatasetObject(
            prim_path=f"/World/{name}",
            usd_path=usd_path,
            category=obj_category,
            name=name,
            scale=np.array([0.8, 0.8, 0.8]),
            abilities={"toggleable": {}},
        )

        sim.import_object(sink, auto_initialize=True)
        sink.set_position(np.array([1, 1, 0.8]))

        assert object_states.ToggledOn in sink.states

        sim.step(force_playing=True)

        for i in range(1000000):
            sim.step()
            # Every other second, toggle on/off
            if i % 1000 == 0:
                if (i // 1000) % 2 == 1:
                    sink.states[object_states.ToggledOn].set_value(True)
                    print("On!")
                else:
                    sink.states[object_states.ToggledOn].set_value(False)
                    print("Off!")

    finally:
        app.close()


def test_dirty():
    try:
        obj_category = "sink"
        obj_model = "sink_1"
        name = "sink"

        sim = Simulator()
        scene = EmptyScene()
        sim.import_scene(scene)

        model_root_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}"
        usd_path = f"{model_root_path}/usd/{obj_model}.usd"

        sink = DatasetObject(
            prim_path=f"/World/{name}",
            usd_path=usd_path,
            category=obj_category,
            name=name,
            scale=np.array([0.8, 0.8, 0.8]),
            abilities={"dustyable": {}},
        )

        sim.import_object(sink, auto_initialize=False)
        #sink.set_position([1, 1, 0.8])
        assert object_states.Dusty in sink.states

        for i in range(10):
            sim.step()

    finally:
        app.close()


def test_burnt():
    try:
        obj_category = "sink"
        obj_model = "sink_1"
        name = "sink"

        sim = Simulator()
        scene = EmptyScene()
        sim.import_scene(scene)

        model_root_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}"
        usd_path = f"{model_root_path}/usd/{obj_model}.usd"

        sink = DatasetObject(
            prim_path=f"/World/{name}",
            usd_path=usd_path,
            category=obj_category,
            name=name,
            scale=np.array([0.8, 0.8, 0.8]),
            abilities={"burnable": {}},
        )

        sim.import_object(sink, auto_initialize=False)
        #sink.set_position([1, 1, 0.8])
        assert object_states.Burnt in sink.states

        for i in range(10):
            sim.step()

    finally:
        app.close()

def test_cooked():
    try:
        obj_category = "sink"
        obj_model = "sink_1"
        name = "sink"

        sim = Simulator()
        scene = EmptyScene()
        sim.import_scene(scene)

        model_root_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}"
        usd_path = f"{model_root_path}/usd/{obj_model}.usd"

        sink = DatasetObject(
            prim_path=f"/World/{name}",
            usd_path=usd_path,
            category=obj_category,
            name=name,
            scale=np.array([0.8, 0.8, 0.8]),
            abilities={"cookable": {}},
        )

        sim.import_object(sink, auto_initialize=False)
        #sink.set_position([1, 1, 0.8])
        assert object_states.Cooked in sink.states

        for i in range(10):
            sim.step()

    finally:
        app.close()


def test_frozen():
    try:
        obj_category = "sink"
        obj_model = "sink_1"
        name = "sink"

        sim = Simulator()
        scene = EmptyScene()
        sim.import_scene(scene)

        model_root_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}"
        usd_path = f"{model_root_path}/usd/{obj_model}.usd"

        sink = DatasetObject(
            prim_path=f"/World/{name}",
            usd_path=usd_path,
            category=obj_category,
            name=name,
            scale=np.array([0.8, 0.8, 0.8]),
            abilities={"freezable": {}},
        )

        sim.import_object(sink, auto_initialize=False)
        #sink.set_position([1, 1, 0.8])
        assert object_states.Frozen in sink.states

        for i in range(10):
            sim.step()

    finally:
        app.close()


def test_vertical_adjacency():
    try:
        obj_category = "sink"
        obj_model = "sink_1"
        name = "sink"

        sim = Simulator()
        scene = EmptyScene()
        sim.import_scene(scene)

        model_root_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}"
        usd_path = f"{model_root_path}/usd/{obj_model}.usd"

        sink_1 = DatasetObject(
            prim_path=f"/World/{name}_1",
            usd_path=usd_path,
            category=obj_category,
            name=f"{name}_1",
            scale=np.array([0.8, 0.8, 0.8]),
            abilities={},
        )

        sink_2 = DatasetObject(
            prim_path=f"/World/{name}_2",
            usd_path=usd_path,
            category=obj_category,
            name=f"{name}_2",
            scale=np.array([0.8, 0.8, 0.8]),
            abilities={},
        )

        sink_3 = DatasetObject(
            prim_path=f"/World/{name}_3",
            usd_path=usd_path,
            category=obj_category,
            name=f"{name}_3",
            scale=np.array([0.8, 0.8, 0.8]),
            abilities={},
        )

        sim.import_object(sink_1, auto_initialize=False)
        sink_1.set_position_orientation(position=np.array([0, 0, 1]))

        sim.import_object(sink_2, auto_initialize=False)
        sink_2.set_position_orientation(position=np.array([0, 0, 3]))
        
        sim.import_object(sink_3, auto_initialize=False)
        sink_3.set_position_orientation(position=np.array([0, 0, 5]))

        # needs 1 physics step to activate collision meshes for raycasting
        sim.step(force_playing=True)
        sim.pause()

        # adjacency = sink_1.states[object_states.VerticalAdjacency].get_value()
        # print(adjacency)

        # adjacency = sink_2.states[object_states.VerticalAdjacency].get_value()
        # print(adjacency)

        # adjacency = sink_3.states[object_states.VerticalAdjacency].get_value()
        # print(adjacency)

        assert sink_1.states[object_states.Under].get_value(sink_2)
        assert sink_1.states[object_states.Under].get_value(sink_3)

        assert sink_2.states[object_states.OnTop].get_value(sink_1)
        assert sink_2.states[object_states.Under].get_value(sink_3)

        assert sink_3.states[object_states.OnTop].get_value(sink_1)
        assert sink_3.states[object_states.OnTop].get_value(sink_2)
        
        for i in range(1000000):
            sim.step()

    finally:
        app.close()


def test_horizontal_adjacency():
    try:
        obj_category = "sink"
        obj_model = "sink_1"
        name = "sink"

        sim = Simulator()
        scene = EmptyScene()
        sim.import_scene(scene)

        model_root_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}"
        usd_path = f"{model_root_path}/usd/{obj_model}.usd"

        sink_1 = DatasetObject(
            prim_path=f"/World/{name}_1",
            usd_path=usd_path,
            category=obj_category,
            name=f"{name}_1",
            scale=np.array([0.8, 0.8, 0.8]),
            abilities={},
        )

        sink_2 = DatasetObject(
            prim_path=f"/World/{name}_2",
            usd_path=usd_path,
            category=obj_category,
            name=f"{name}_2",
            scale=np.array([0.8, 0.8, 0.8]),
            abilities={},
        )

        sink_3 = DatasetObject(
            prim_path=f"/World/{name}_3",
            usd_path=usd_path,
            category=obj_category,
            name=f"{name}_3",
            scale=np.array([0.8, 0.8, 0.8]),
            abilities={},
        )

        sim.import_object(sink_1, auto_initialize=False)
        sink_1.set_position_orientation(position=np.array([0, 0, 1]))

        sim.import_object(sink_2, auto_initialize=False)
        sink_2.set_position_orientation(position=np.array([2, 0, 1]))
        
        sim.import_object(sink_3, auto_initialize=False)
        sink_3.set_position_orientation(position=np.array([0, 1, 1]))

        # needs 1 physics step to activate collision meshes for raycasting
        sim.step(force_playing=True)
        sim.pause()

        # adjacency = sink_1.states[object_states.HorizontalAdjacency].get_value()
        # print(adjacency)

        # adjacency = sink_2.states[object_states.HorizontalAdjacency].get_value()
        # print(adjacency)

        # adjacency = sink_3.states[object_states.HorizontalAdjacency].get_value()
        # print(adjacency)

        assert sink_1.states[object_states.NextTo].get_value(sink_2)
        assert sink_1.states[object_states.NextTo].get_value(sink_3)

        assert sink_2.states[object_states.NextTo].get_value(sink_1)
        assert sink_2.states[object_states.NextTo].get_value(sink_3)

        assert sink_3.states[object_states.NextTo].get_value(sink_1)
        assert sink_3.states[object_states.NextTo].get_value(sink_2)
        
        for i in range(1000000):
            sim.step()

    finally:
        app.close()


def test_inside():
    try:
        obj_category = "sink"
        obj_model = "sink_1"
        name = "sink"

        sim = Simulator()
        scene = EmptyScene()
        sim.import_scene(scene)

        model_root_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}"
        usd_path = f"{model_root_path}/usd/{obj_model}.usd"

        sink_1 = DatasetObject(
            prim_path=f"/World/{name}_1",
            usd_path=usd_path,
            category=obj_category,
            name=f"{name}_1",
            scale=np.array([0.8, 0.8, 0.8]),
            abilities={},
        )

        sink_2 = DatasetObject(
            prim_path=f"/World/{name}_2",
            usd_path=usd_path,
            category=obj_category,
            name=f"{name}_2",
            scale=np.array([0.2, 0.2, 0.2]),
            abilities={},
        )

        sim.import_object(sink_1, auto_initialize=False)
        sink_1.set_position_orientation(position=np.array([0, 0, 1]))

        sim.import_object(sink_2, auto_initialize=False)
        sink_2.set_position_orientation(position=np.array([0, 0, 1]))
        

        # needs 1 physics step to activate collision meshes for raycasting
        sim.step(force_playing=True)
        sim.pause()

        # horizontal_adjacency = sink_2.states[object_states.HorizontalAdjacency].get_value()
        # print(horizontal_adjacency)

        # vertical_adjacency = sink_2.states[object_states.VerticalAdjacency].get_value()
        # print(vertical_adjacency)

        assert sink_2.states[object_states.Inside].get_value(sink_1)
        
        for i in range(1000000):
            sim.step()

    finally:
        app.close()

def test_heat_source():
    try:
        obj_category = "stove"
        obj_model = "101908"
        name = "stove"

        sim = Simulator()
        scene = EmptyScene()
        sim.import_scene(scene)

        model_root_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}"
        usd_path = f"{model_root_path}/usd/{obj_model}.usd"

        stove = DatasetObject(
            prim_path=f"/World/{name}",
            usd_path=usd_path,
            category=obj_category,
            name=f"{name}",
            scale=np.array([0.5, 0.5, 0.5]),
            abilities={"heatSource": {}},
        )

        sim.import_object(stove, auto_initialize=True)
        stove.set_position_orientation(position=np.array([0, 0, 0.5]))
        

        # needs 1 physics step to activate collision meshes for raycasting
        sim.step(force_playing=True)
        sim.pause()

        heat_source = stove.states[object_states.HeatSourceOrSink]
        heat_source_state, heat_source_position = heat_source.get_value()
        print(heat_source_state, heat_source_position)

        assert heat_source_state
        
        for i in range(1000000):
            sim.step()

    finally:
        app.close()


def test_temperature():
    try:
        obj_category = "apple"
        obj_model = "00_0"
        name = "apple"

        sim = Simulator()
        scene = EmptyScene()
        sim.import_scene(scene)

        model_root_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}"
        usd_path = f"{model_root_path}/usd/{obj_model}.usd"

        apple = DatasetObject(
            prim_path=f"/World/{name}",
            usd_path=usd_path,
            category=obj_category,
            name=f"{name}",
            scale=np.array([1.0, 1.0, 1.0]),
            abilities={"freezable": {}, "cookable": {}, "burnable": {}},
        )

        sim.import_object(apple, auto_initialize=False)
        apple.set_position_orientation(position=np.array([0, 0, 0.2]))

        # Manually increase the temperature of the apple
        apple.states[object_states.Burnt].burn_temperature = 200
        for i in range(-10, 100):
            temp = i * 5
            print("Apple temperature: {} degrees Celsius".format(temp))
            apple.states[object_states.Temperature].set_value(temp)
            print("Frozen(Apple)? {}".format(apple.states[object_states.Frozen].get_value()))
            print("Cooked(Apple)? {}".format(apple.states[object_states.Cooked].get_value()))
            print("Burnt(Apple)? {}".format(apple.states[object_states.Burnt].get_value()))
            for j in range(10):
                sim.step()
    
    finally:
        app.close()

def test_touching():
    try:
        obj_category = "apple"
        obj_model = "00_0"
        name = "apple"

        sim = Simulator()
        scene = EmptyScene()
        sim.import_scene(scene)

        model_root_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}"
        usd_path = f"{model_root_path}/usd/{obj_model}.usd"

        apple_1 = DatasetObject(
            prim_path=f"/World/{name}_1",
            usd_path=usd_path,
            category=obj_category,
            name=f"{name}_1",
            scale=np.array([10.0, 10.0, 10.0]),
            abilities={},
        )

        apple_2 = DatasetObject(
            prim_path=f"/World/{name}_2",
            usd_path=usd_path,
            category=obj_category,
            name=f"{name}_2",
            scale=np.array([10.0, 10.0, 10.0]),
            abilities={},
        )

        sim.import_object(apple_1, auto_initialize=True)
        apple_1.set_position_orientation(position=np.array([0, 0, 0]))

        sim.import_object(apple_2, auto_initialize=True)
        apple_2.set_position_orientation(position=np.array([0.5, 0, 0]))

        # needs 1 physics step to activate collision meshes for raycasting
        sim.step(force_playing=True)
        # sim.pause()

        # take enough steps for the apples to separate and sit tangent to each other
        for i in range(100):
            sim.step()

        assert apple_1.states[object_states.Touching].get_value(apple_2)
        assert apple_2.states[object_states.Touching].get_value(apple_1)
        
        for i in range(1000000):
            sim.step()

            # touching = apple_1.states[object_states.Touching].get_value(apple_2)
            # print("apple 1 is touching apple 2:", touching)
            # # floor = apple_1.states[object_states.OnFloor].get_value(apple_1.room_floor)
            # # print("apple 1 is on the floor:", floor)

            # touching = apple_2.states[object_states.Touching].get_value(apple_1)
            # print("apple 2 is touching apple 1:", touching)
            # # floor = apple_2.states[object_states.OnFloor].get_value(apple_2.room_floor)
            # # print("apple 2 is on the floor:", floor)
            # print()

    finally:
        app.close()

def test_open():
    try:
        obj_category = "microwave"
        obj_model = "7128"
        name = "microwave"

        sim = Simulator()
        scene = EmptyScene()
        sim.import_scene(scene)

        model_root_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}"
        usd_path = f"{model_root_path}/usd/{obj_model}.usd"

        microwave = DatasetObject(
            prim_path=f"/World/{name}",
            usd_path=usd_path,
            category=obj_category,
            name=f"{name}",
            scale=np.array([0.5, 0.5, 0.5]),
            abilities={"openable":{}},
        )

        sim.import_object(microwave, auto_initialize=True)
        microwave.set_position_orientation(position=np.array([0, 0, 0.5]))
        

        # needs 1 physics step to activate collision meshes for raycasting
        sim.step(force_playing=True)
        #sim.pause()
        
        for i in range(1000000):
            sim.step()
            
            # every second, alternate between opening and closing the door
            if i % 1000 == 0:
                if (i // 1000) % 2 == 0:
                    print("opening microwave")
                    microwave.states[object_states.Open].set_value(new_value=True, fully=True)
                elif (i // 1000) % 2 == 1:
                    print("closing microwave")
                    microwave.states[object_states.Open].set_value(new_value=False, fully=True)
                is_open = microwave.states[object_states.Open].get_value()
                print(is_open)

    finally:
        app.close()

def test_demo():
    try:
        sim = Simulator()
        scene = EmptyScene()
        sim.import_scene(scene)

        obj_category = "apple"
        obj_model = "00_0"
        name = "apple"

        model_root_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}"
        usd_path = f"{model_root_path}/usd/{obj_model}.usd"

        apple = DatasetObject(
            prim_path=f"/World/{name}",
            usd_path=usd_path,
            category=obj_category,
            name=f"{name}",
            scale=np.array([2.0, 2.0, 2.0]),
            abilities={"freezable": {}, "cookable": {}, "burnable": {}},
        )

        obj_category = "stove"
        obj_model = "101908"
        name = "stove"

        model_root_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}"
        usd_path = f"{model_root_path}/usd/{obj_model}.usd"

        stove = DatasetObject(
            prim_path=f"/World/{name}",
            usd_path=usd_path,
            category=obj_category,
            name=f"{name}",
            scale=np.array([0.5, 0.5, 0.5]),
            abilities={"heatSource": {
                            "temperature": 250,
                            "heating_rate": 0.2,
                            "distance_threshold": 0.5,
                        }, "openable": {}, "toggleable": {}
            },
        )

        sim.import_object(apple, auto_initialize=True)
        apple.set_position_orientation(position=np.array([0, -1, 0]))
        
        sim.import_object(stove, auto_initialize=True)
        stove.set_position_orientation(position=np.array([0, 0, 0]))

        # needs 1 physics step to activate collision meshes for raycasting
        sim.step(force_playing=True)
        # sim.pause()
        
        # setup apple
        apple.states[object_states.Temperature].set_value(-50)
        apple.states[object_states.Burnt].burn_temperature = 200
        
        # setup stove
        stove.states[object_states.ToggledOn].set_value(True)
        print("Stove is ToggledOn:", stove.states[object_states.ToggledOn].get_value())

        heat_source_state, heat_source_position = stove.states[object_states.HeatSourceOrSink].get_value()
        print("Stove is a HeatSource:", heat_source_state)
        
        for i in range(1000000):
            sim.step()

            if i % 100 == 0:
                print("Apple is...\n\tTouching: %r\n\tOnTop: %r\n\tInside: %r\n\tTemperature: %.2f\n\tFrozen: %r\n\tCooked: %r\n\tBurnt: %r\n"
                    % (
                        apple.states[object_states.Touching].get_value(stove),
                        apple.states[object_states.OnTop].get_value(stove),
                        apple.states[object_states.Inside].get_value(stove),
                        apple.states[object_states.Temperature].get_value(),
                        apple.states[object_states.Frozen].get_value(),
                        apple.states[object_states.Cooked].get_value(),
                        apple.states[object_states.Burnt].get_value(),
                    )
                )

    finally:
        app.close()

def test_sliced():
    try:
        obj_category = "apple"
        obj_model = "00_0"
        name = "apple"

        sim = Simulator()
        scene = EmptyScene()
        sim.import_scene(scene)

        model_root_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}"
        usd_path = f"{model_root_path}/usd/{obj_model}.usd"

        # Create an dataset object of an apple, but doesn't load it in the simulator
        whole_obj = DatasetObject(
            prim_path=f"/World/{name}",
            usd_path=usd_path,
            category=obj_category,
            name=f"{name}",
            scale=np.array([10.0, 10.0, 10.0]),
            abilities={"sliceable":{}},
        )

        sim.import_object(whole_obj, auto_initialize=True)
        whole_obj.set_position_orientation(position=np.array([0, 0, 0]))

        # needs 1 physics step to activate collision meshes for raycasting
        sim.step(force_playing=True)
        
        # Let the apple get stable
        print("Countdown to slice...")
        for i in range(3000):
            if i % 1000 == 0:
                print(3 - i//1000)
            sim.step()
        
        # # Save the initial state.
        # initial_state = sim.dump_state(serialized=True)

        print("Slicing the apple...")
        
        assert not whole_obj.states[object_states.Sliced].get_value()
        whole_obj.states[object_states.Sliced].set_value(True)
        assert whole_obj.states[object_states.Sliced].get_value()

        print("Success!")
        
        for i in range(1000000):
            sim.step()

        # print("Restoring the state")
        # # Restore the state
        # sim.load_state(initial_state)

        # The apple should become whole again

    finally:
        app.close()

def test_slicer():
    try:
        obj_category = "apple"
        obj_model = "00_0"
        name = "apple"

        sim = Simulator()
        scene = EmptyScene()
        sim.import_scene(scene)

        model_root_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}"
        usd_path = f"{model_root_path}/usd/{obj_model}.usd"

        # Create an dataset object of an apple, but doesn't load it in the simulator
        whole_obj = DatasetObject(
            prim_path=f"/World/{name}",
            usd_path=usd_path,
            category=obj_category,
            name=f"{name}",
            scale=np.array([10.0, 10.0, 10.0]),
            abilities={"sliceable":{}},
        )

        obj_category = "table_knife"
        obj_model = "1"
        name = "knife"

        model_root_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}"
        usd_path = f"{model_root_path}/usd/{obj_model}.usd"

        # Create an dataset object of an apple, but doesn't load it in the simulator
        knife = DatasetObject(
            prim_path=f"/World/{name}",
            usd_path=usd_path,
            category=obj_category,
            name=f"{name}",
            scale=np.array([15.0, 15.0, 15.0]),
            abilities={"slicer":{}},
        )

        sim.import_object(whole_obj, auto_initialize=True)
        whole_obj.set_position_orientation(position=np.array([0, 0, 0]))

        sim.import_object(knife, auto_initialize=True)
        knife.set_position_orientation(position=np.array([0, 1, 0]))

        # needs 1 physics step to activate collision meshes for raycasting
        sim.step(force_playing=True)
        
        assert not whole_obj.states[object_states.Sliced].get_value()
        
        for i in range(1000000):
            sim.step()

    finally:
        app.close()


def test_water_source_sink():
    try:
        obj_category = "sink"
        obj_model = "sink_2"
        name = "sink"

        sim = Simulator()
        scene = EmptyScene()
        sim.import_scene(scene)

        # Overwrite physics
        sim.stop()
        pc = sim.get_physics_context()
        pc.enable_gpu_dynamics(True)
        pc.set_broadphase_type("GPU")
        sim.play()
        sim.stop()

        model_root_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}"
        usd_path = f"{model_root_path}/usd/{obj_model}.usd"

        # Create object
        obj = DatasetObject(
            prim_path=f"/World/{name}",
            usd_path=usd_path,
            category=obj_category,
            name=f"{name}",
            scale=np.array([1.0, 1.0, 1.0]),
            abilities={
                "waterSource": {},
                "waterSink": {},
                "toggleable": {},
            },
        )


        sim.import_object(obj)
        obj.set_position(np.array([0,0,0.35]))
        sim.play()

        # Create camera to track sick location correctly
        cam = VisionSensor(
            prim_path="/World/viewer_camera",
            name="camera",
            image_height=720,
            image_width=1280,
            modalities="rgb",
            viewport_name="Viewport",
        )
        sim.step()
        cam.initialize()
        cam.set_position_orientation(position=np.array([0.00551, -1.5846, 2.39234]), orientation=np.array([0.3657, 0.00318, 0.0081, 0.93069]))

        # Toggle water source on
        obj.states[object_states.ToggledOn].set_value(True)
        ws = obj.states[object_states.WaterSource]
        sim.stop()
        sim.play()
        print(f"WaterSource and WaterSink are active!")

        for i in range(1000):
            sim.step()

            if i % 50 == 0:
                print(f"Number of water particles active: {ws.n_particles_per_group * len(ws.fluid_groups)}")

    finally:
        app.close()


def test_water_filled():
    try:
        obj_category = "cup"
        obj_model = "cup_000"
        name = "cup"

        sim = Simulator()
        scene = EmptyScene(floor_plane_visible=True)
        sim.import_scene(scene)

        # Overwrite physics
        sim.stop()
        pc = sim.get_physics_context()
        pc.enable_gpu_dynamics(True)
        pc.set_broadphase_type("GPU")
        sim.play()
        sim.stop()

        model_root_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}"
        usd_path = f"{model_root_path}/usd/{obj_model}.usd"

        # Create water source
        water_spout = PrimitiveObject(
            prim_path="/World/water_spout",
            primitive_type="Sphere",
            name="water_spout",
            visual_only=True,
            visible=False,
            abilities={
                "waterSource": {},
                "toggleable": {},
            }
        )
        sim.import_object(water_spout)

        # Define appropriate metalinks for this spout
        for metalink_name in ("toggle_button_link", "water_source_link"):
            metalink_prim_path = f"{water_spout.prim_path}/{metalink_name}"
            sim.stage.DefinePrim(metalink_prim_path, "Xform")
            # Add fixed joint between new link and root link
            create_joint(
                prim_path=f"{water_spout.prim_path}/{metalink_name}_joint",
                joint_type="FixedJoint",
                body0=water_spout.root_link.prim_path,
                body1=metalink_prim_path,
                enabled=True,
                stage=sim.stage,
            )

        # Update some internal states via post-load again
        water_spout._post_load()

        # Create cup object
        obj = DatasetObject(
            prim_path=f"/World/{name}",
            usd_path=usd_path,
            category=obj_category,
            name=f"{name}",
            scale=0.3,
            abilities={
                "fillable": {"fluid": "Water"},
            },
        )
        sim.import_object(obj)

        # Define appropriate container link and mesh for this object
        container_prim_path = f"{obj.prim_path}/container_link"
        sim.stage.DefinePrim(container_prim_path, "Xform")
        # Add fixed joint between new link and root link
        create_joint(
            prim_path=f"{obj.prim_path}/container_link_joint",
            joint_type="FixedJoint",
            body0=obj.root_link.prim_path,
            body1=container_prim_path,
            enabled=True,
            stage=sim.stage,
        )
        # Add mesh to symbolize the actual volume to be filled
        sim.stage.DefinePrim(f"{container_prim_path}/container_cube", "Cube")

        # Update some internal states via post-load again, and make sure our dummy link has low mass
        obj._post_load()
        obj.links["container_link"].mass = 0.0001
        container_mesh = obj.links["container_link"].visual_meshes["container_cube"]

        container_mesh.set_attribute("xformOp:translate", Gf.Vec3d(0, 0.56657, 0.15427))
        container_mesh.scale = np.array([0.12, 0.12, 0.15])

        obj.set_position(np.array([0,0,0]))
        sim.play()

        # Hide visibility of all toggle markers for water spout and container volume for cup
        for link in water_spout.states[object_states.ToggledOn].visual_marker_on.links.values():
            link.visible = False
        for link in water_spout.states[object_states.ToggledOn].visual_marker_off.links.values():
            link.visible = False
        obj.links["container_link"].visible = False

        # Create camera to track cup location correctly
        cam = VisionSensor(
            prim_path="/World/viewer_camera",
            name="camera",
            image_height=720,
            image_width=1280,
            modalities="rgb",
            viewport_name="Viewport",
        )
        sim.step()
        cam.initialize()
        cam.set_position_orientation(position=np.array([-0.76926,-0.22523,1.29085]), orientation=np.array([0.27383,-0.17394,-0.50717,0.79846]))
        sim.stop()

        # Move the water source to be above the cup above its center of mass
        base_link = obj.links["base_link"]
        water_pos = base_link.get_position() + np.array(base_link.get_attribute("physics:centerOfMass")) * base_link.get_world_scale() + np.array([0, 0, 0.2])
        water_spout.set_position(water_pos)

        # Toggle water source on
        water_spout.states[object_states.ToggledOn].set_value(True)
        ws = water_spout.states[object_states.WaterSource]

        sim.play()
        print(f"Cup is filled: {obj.states[object_states.Filled].get_value()}")

        for i in range(400):
            if i == 300:
                water_spout.states[object_states.ToggledOn].set_value(False)
            sim.step()

        cup_is_filled = obj.states[object_states.Filled].get_value()
        print(f"Cup is filled: {cup_is_filled}")
        assert cup_is_filled

    finally:
        app.close()


def test_attachment():
    try:
        obj_category = "apple"
        obj_model = "00_0"
        name = "apple"

        sim = Simulator()
        scene = EmptyScene()
        sim.import_scene(scene)

        model_root_path = f"{ig_dataset_path}/objects/{obj_category}/{obj_model}"
        usd_path = f"{model_root_path}/usd/{obj_model}.usd"

        apple_1 = DatasetObject(
            prim_path=f"/World/{name}_1",
            usd_path=usd_path,
            category=obj_category,
            name=f"{name}_1",
            scale=np.array([10.0, 10.0, 10.0]),
            abilities={"sticky": {}},
        )

        apple_2 = DatasetObject(
            prim_path=f"/World/{name}_2",
            usd_path=usd_path,
            category=obj_category,
            name=f"{name}_2",
            scale=np.array([10.0, 10.0, 10.0]),
            abilities={},
        )

        sim.import_object(apple_1, auto_initialize=True)
        apple_1.set_position_orientation(position=np.array([0, 0, 0]))

        sim.import_object(apple_2, auto_initialize=True)
        apple_2.set_position_orientation(position=np.array([1.0, 0, 0]))

        # needs 1 physics step to activate collision meshes for raycasting
        sim.step(force_playing=True)
        # sim.pause()

        # apple_1.states[object_states.MagneticAttachment].set_value(apple_2, True)

        # assert apple_1.states[object_states.MagneticAttachment].get_value(apple_2)

        for i in range(1000000):
            sim.step()

    finally:
        app.close()


## WORKS
#test_state_graph()
#test_dirty()
#test_burnt()
#test_cooked()
#test_frozen()
#test_vertical_adjacency()
#test_horizontal_adjacency()
#test_inside()
#test_heat_source()
#test_temperature()
#test_touching()
#test_open()
#test_toggle()
#test_sliced()
#test_slicer()
#test_water_source_sink()
#test_attachment()

# test_demo()

## BROKEN
