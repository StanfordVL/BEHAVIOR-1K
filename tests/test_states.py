import os

import networkx as nx
import numpy as np


import igibson
from igibson import object_states, app, ig_dataset_path
# from igibson.external.pybullet_tools.utils import Euler, quat_from_euler
from igibson.object_states.factory import get_state_dependency_graph, get_states_by_dependency_order
from igibson.objects.dataset_object import DatasetObject
from igibson.objects.ycb_object import YCBObject
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator_omni import Simulator
from igibson.utils.assets_utils import download_assets, get_ig_model_path

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

        sim.import_object(sink, auto_initialize=False)
        sink.states[object_states.ToggledOn].set_value(True)
        #sink.set_position([1, 1, 0.8])
        assert object_states.ToggledOn in sink.states

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


def test_water_source():
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
            abilities={"waterSource": {}, "toggleable": {}},
        )

        sim.import_object(sink, auto_initialize=False)
        sink.states[object_states.ToggledOn].set_value(True)
        #sink.set_position([1, 1, 0.8])
        assert object_states.WaterSource in sink.states

        for i in range(10):
            sim.step()

        # Check that we have some loaded particles here.
        assert (
            sink.states[object_states.WaterSource].water_stream.get_active_particles()[0].get_body_ids()[0] is not None
        )

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


## WORKS
#test_state_graph()
#test_dirty()
#test_burnt()
#test_cooked()
test_frozen()

## BROKEN
#test_toggle()
#test_water_source()
