import os

import numpy as np
from IPython import embed

import igibson
from igibson.object_states import *
from igibson.objects.usd_object import URDFObject
from igibson.objects.multi_object_wrappers import ObjectGrouper, ObjectMultiplexer
from igibson.robots.fetch import Fetch
from igibson.scenes.interactive_traversable_scene import InteractiveIndoorScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_ig_model_path

CABINET_POS = np.array([100, 100, 100])
CABINET_JOINT = {
    "bottom_cabinet_0_joint_1": (0.1, 0.5),
    "bottom_cabinet_0_joint_2": (0.2, 0.6),
    "bottom_cabinet_0_joint_3": (0.3, 0.7),
    "bottom_cabinet_0_joint_4": (0.4, 0.8),
}
FETCH_POS = np.array([0.5, -2.5, 0])
FETCH_JOINT = np.array([1.0, 0.5])


def test_saving():
    s = Simulator(mode="headless", use_pb_gui=False)

    scene = InteractiveIndoorScene(
        "Rs_int",
        # load_object_categories=["breakfast_table", "bed"],
    )
    s.import_scene(scene)

    # Change kinematic of existing objects
    scene.objects_by_name["bottom_cabinet_0"].set_position(CABINET_POS)
    scene.objects_by_name["bottom_cabinet_0"].set_joint_states(CABINET_JOINT)

    # Change non-kinematic states of existing object
    scene.objects_by_name["pot_plant_1"].states[Soaked].set_value(True)
    scene.objects_by_name["floor_lamp_3"].states[ToggledOn].set_value(True)

    # Save
    scene.save(urdf_path="changed_state.urdf", pybullet_filename="changed_state.bullet")

    # Add a new object
    model_path = os.path.join(get_ig_model_path("apple", "00_0"), "00_0.urdf")
    simulator_obj = URDFObject(model_path, name="00_0", category="apple", scale=np.array([1.0, 1.0, 1.0]))
    whole_object = simulator_obj
    object_parts = []
    for i, part in enumerate(simulator_obj.metadata["object_parts"]):
        category = part["category"]
        model = part["model"]
        # Scale the offset accordingly
        part_pos = part["pos"] * whole_object.scale
        part_orn = part["orn"]
        model_path = get_ig_model_path(category, model)
        filename = os.path.join(model_path, model + ".urdf")
        obj_name = whole_object.name + "_part_{}".format(i)
        simulator_obj_part = URDFObject(
            filename,
            name=obj_name,
            category=category,
            model_path=model_path,
            scale=whole_object.scale,
        )
        object_parts.append((simulator_obj_part, (part_pos, part_orn)))
    grouped_obj_parts = ObjectGrouper(object_parts)
    apple = ObjectMultiplexer(whole_object.name + "_multiplexer", [whole_object, grouped_obj_parts], 0)
    s.import_object(apple)
    apple.set_position([0, 0, 1])

    # Change its default non-kinematic state
    apple.states[Sliced].set_value(True)

    # Import agents
    fetch = Fetch(name="agent_0")
    s.import_object(fetch)
    fetch.set_position(FETCH_POS)
    fetch.joints["head_tilt_joint"].reset_state(*FETCH_JOINT)

    # Save
    scene.save(urdf_path="changed_structure.urdf", pybullet_filename="changed_structure.bullet")

    s.disconnect()


def test_loading_state_with_bullet_file():
    s = Simulator(mode="headless", use_pb_gui=False)
    scene = InteractiveIndoorScene("Rs_int")
    s.import_scene(scene)
    scene.restore(urdf_path="changed_state.urdf", pybullet_filename="changed_state.bullet")
    # print("load_state")
    # embed()
    assert np.array_equal(scene.objects_by_name["bottom_cabinet_0"].get_position(), CABINET_POS)
    joint_states = scene.objects_by_name["bottom_cabinet_0"].get_joint_states()
    for key in joint_states:
        assert np.array_equal(np.array(joint_states[key]), np.array(CABINET_JOINT[key]))
    assert scene.objects_by_name["pot_plant_1"].states[Soaked].get_value()
    assert scene.objects_by_name["floor_lamp_3"].states[ToggledOn].get_value()
    s.disconnect()


def test_loading_state_without_bullet_file():
    s = Simulator(mode="headless", use_pb_gui=False)
    scene = InteractiveIndoorScene("Rs_int")
    s.import_scene(scene)
    scene.restore(urdf_path="changed_state.urdf")
    # print("load_state")
    # embed()
    assert np.array_equal(scene.objects_by_name["bottom_cabinet_0"].get_position(), CABINET_POS)
    joint_states = scene.objects_by_name["bottom_cabinet_0"].get_joint_states()
    for key in joint_states:
        assert np.array_equal(np.array(joint_states[key]), np.array(CABINET_JOINT[key]))
    assert scene.objects_by_name["pot_plant_1"].states[Soaked].get_value()
    assert scene.objects_by_name["floor_lamp_3"].states[ToggledOn].get_value()
    s.disconnect()


def test_loading_state_with_sliceable():
    s = Simulator(mode="headless", use_pb_gui=False)
    scene = InteractiveIndoorScene(
        "Rs_int",
        urdf_path="changed_structure.urdf",
    )
    s.import_scene(scene)
    scene.restore(urdf_path="changed_structure.urdf")
    # print("load_state")
    # embed()
    assert np.array_equal(scene.objects_by_name["bottom_cabinet_0"].get_position(), CABINET_POS)
    joint_states = scene.objects_by_name["bottom_cabinet_0"].get_joint_states()
    for key in joint_states:
        assert np.array_equal(np.array(joint_states[key]), np.array(CABINET_JOINT[key]))
    assert scene.objects_by_name["pot_plant_1"].states[Soaked].get_value()
    assert scene.objects_by_name["floor_lamp_3"].states[ToggledOn].get_value()
    assert scene.objects_by_name["00_0_multiplexer"].states[Sliced].get_value()
    assert np.array_equal(scene.objects_by_name["agent_0"].get_position(), FETCH_POS)
    assert np.array_equal(
        np.array(scene.objects_by_name["agent_0"].joints["head_tilt_joint"].get_state()[:2]), FETCH_JOINT
    )
    s.disconnect()


def test_loading_structure_with_bullet_file():
    s = Simulator(mode="headless", use_pb_gui=False)
    scene = InteractiveIndoorScene(
        "Rs_int",
        urdf_path="changed_structure.urdf",
        pybullet_filename="changed_structure.bullet",
    )
    s.import_scene(scene)
    # print("load_structure")
    # embed()
    assert np.array_equal(scene.objects_by_name["bottom_cabinet_0"].get_position(), CABINET_POS)
    joint_states = scene.objects_by_name["bottom_cabinet_0"].get_joint_states()
    for key in joint_states:
        assert np.array_equal(np.array(joint_states[key]), np.array(CABINET_JOINT[key]))
    assert scene.objects_by_name["pot_plant_1"].states[Soaked].get_value()
    assert scene.objects_by_name["floor_lamp_3"].states[ToggledOn].get_value()
    assert scene.objects_by_name["00_0_multiplexer"].states[Sliced].get_value()
    assert np.array_equal(scene.objects_by_name["agent_0"].get_position(), FETCH_POS)
    assert np.array_equal(
        np.array(scene.objects_by_name["agent_0"].joints["head_tilt_joint"].get_state()[:2]), FETCH_JOINT
    )
    s.disconnect()


def test_loading_structure_without_bullet_file():
    s = Simulator(mode="headless", use_pb_gui=False)
    scene = InteractiveIndoorScene(
        "Rs_int",
        urdf_path="changed_structure.urdf",
    )
    s.import_scene(scene)
    # print("load_structure")
    # embed()
    assert np.array_equal(scene.objects_by_name["bottom_cabinet_0"].get_position(), CABINET_POS)
    joint_states = scene.objects_by_name["bottom_cabinet_0"].get_joint_states()
    for key in joint_states:
        assert np.array_equal(np.array(joint_states[key]), np.array(CABINET_JOINT[key]))
    assert scene.objects_by_name["pot_plant_1"].states[Soaked].get_value()
    assert scene.objects_by_name["floor_lamp_3"].states[ToggledOn].get_value()
    assert scene.objects_by_name["00_0_multiplexer"].states[Sliced].get_value()
    assert np.array_equal(scene.objects_by_name["agent_0"].get_position(), FETCH_POS)
    assert np.array_equal(
        np.array(scene.objects_by_name["agent_0"].joints["head_tilt_joint"].get_state()[:2]), FETCH_JOINT
    )
    s.disconnect()


# if __name__ == "__main__":
#     test_saving()
#     test_loading_state_with_bullet_file()
#     test_loading_state_without_bullet_file()
#     test_loading_state_with_sliceable()
#     test_loading_structure_with_bullet_file()
#     test_loading_structure_without_bullet_file()
