import os

import networkx as nx
import numpy as np


import igibson
from igibson import object_states
from igibson.external.pybullet_tools.utils import Euler, quat_from_euler
from igibson.object_states.factory import get_state_dependency_graph, get_states_by_dependency_order
from igibson.objects.usd_object import ArticulatedObject, URDFObject
from igibson.objects.ycb_object import YCBObject
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import download_assets, get_ig_model_path

download_assets()


def test_on_top():
    s = Simulator(mode="headless")

    try:
        scene = EmptyScene()
        s.import_scene(scene)

        cabinet_0007 = os.path.join(igibson.assets_path, "models/cabinet2/cabinet_0007.urdf")
        cabinet_0004 = os.path.join(igibson.assets_path, "models/cabinet/cabinet_0004.urdf")

        obj1 = ArticulatedObject(filename=cabinet_0007)
        s.import_object(obj1)
        obj1.set_position([0, 0, 0.5])

        obj2 = ArticulatedObject(filename=cabinet_0004)
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

        cabinet_0007 = os.path.join(igibson.assets_path, "models/cabinet2/cabinet_0007.urdf")
        cabinet_0004 = os.path.join(igibson.assets_path, "models/cabinet/cabinet_0004.urdf")

        obj1 = ArticulatedObject(filename=cabinet_0007)
        s.import_object(obj1)
        obj1.set_position([0, 0, 0.5])

        obj2 = ArticulatedObject(filename=cabinet_0004)
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

        microwave_dir = os.path.join(igibson.ig_dataset_path, "objects/microwave/7128/")
        microwave_filename = os.path.join(microwave_dir, "7128.urdf")
        obj = URDFObject(
            filename=microwave_filename,
            category="microwave",
            model_path=microwave_dir,
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


def test_toggle():
    s = Simulator(mode="headless")

    try:
        scene = EmptyScene()
        s.import_scene(scene)
        model_path = os.path.join(get_ig_model_path("sink", "sink_1"), "sink_1.urdf")

        sink = URDFObject(
            filename=model_path,
            category="sink",
            name="sink_1",
            scale=np.array([0.8, 0.8, 0.8]),
            abilities={"toggleable": {}},
        )

        s.import_object(sink)
        sink.set_position([1, 1, 0.8])
        assert object_states.ToggledOn in sink.states

    finally:
        s.disconnect()


def test_dirty():
    s = Simulator(mode="headless")

    try:
        scene = EmptyScene()
        s.import_scene(scene)
        model_path = os.path.join(get_ig_model_path("sink", "sink_1"), "sink_1.urdf")

        sink = URDFObject(
            filename=model_path,
            category="sink",
            name="sink_1",
            scale=np.array([0.8, 0.8, 0.8]),
            abilities={"dustyable": {}},
        )

        s.import_object(sink)
        sink.set_position([1, 1, 0.8])
        assert object_states.Dusty in sink.states

        for i in range(10):
            s.step()

    finally:
        s.disconnect()


def test_water_source():
    s = Simulator(mode="headless")

    try:
        scene = EmptyScene()
        s.import_scene(scene)
        model_path = os.path.join(get_ig_model_path("sink", "sink_1"), "sink_1.urdf")

        sink = URDFObject(
            filename=model_path,
            category="sink",
            name="sink_1",
            scale=np.array([0.8, 0.8, 0.8]),
            abilities={"waterSource": {}, "toggleable": {}},
        )

        s.import_object(sink)
        sink.states[object_states.ToggledOn].set_value(True)
        sink.set_position([1, 1, 0.8])
        assert object_states.WaterSource in sink.states

        for i in range(2):
            s.step()

        # Check that we have some loaded particles here.
        assert (
            sink.states[object_states.WaterSource].water_stream.get_active_particles()[0].get_body_ids()[0] is not None
        )
    finally:
        s.disconnect()
