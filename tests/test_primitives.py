import numpy as np
import pytest
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, StarterSemanticActionPrimitiveSet
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject

def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        env.step(action)

def set_start_pose(robot):
    reset_pose_tiago = np.array([
        -1.78029833e-04,  3.20231302e-05, -1.85759447e-07, -1.16488536e-07,
        4.55182843e-08,  2.36128806e-04,  1.50000000e-01,  9.40000000e-01,
        -1.10000000e+00,  0.00000000e+00, -0.90000000e+00,  1.47000000e+00,
        0.00000000e+00,  2.10000000e+00,  2.71000000e+00,  1.50000000e+00,
        1.71000000e+00,  1.30000000e+00, -1.57000000e+00, -1.40000000e+00,
        1.39000000e+00,  0.00000000e+00,  0.00000000e+00,  4.50000000e-02,
        4.50000000e-02,  4.50000000e-02,  4.50000000e-02,
    ])
    robot.set_joint_positions(reset_pose_tiago)
    og.sim.step()

def primitive_tester(load_object_categories, objects, primitives, primitives_args):
    cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "load_object_categories": load_object_categories,
        },
        "robots": [
            {
                "type": "Tiago",
                "obs_modalities": ["scan", "rgb", "depth"],
                "scale": 1.0,
                "self_collisions": True,
                "action_normalize": False,
                "action_type": "continuous",
                "grasping_mode": "sticky",
                "rigid_trunk": False,
                "default_arm_pose": "diagonal30",
                "default_trunk_offset": 0.365,
                "controller_config": {
                    "base": {
                        "name": "JointController",
                        "motor_type": "velocity"
                    },
                    "arm_left": {
                        "name": "JointController",
                        "motor_type": "position",
                        "command_input_limits": None,
                        "command_output_limits": None, 
                        "use_delta_commands": False
                    },
                    "arm_right": {
                        "name": "JointController",
                        "motor_type": "position",
                        "command_input_limits": None,
                        "command_output_limits": None, 
                        "use_delta_commands": False
                    },
                    "gripper_left": {
                        "name": "JointController",
                        "motor_type": "position",
                        "command_input_limits": [-1, 1],
                        "command_output_limits": None,
                        "use_delta_commands": True
                    },
                    "gripper_right": {
                        "name": "JointController",
                        "motor_type": "position",
                        "command_input_limits": [-1, 1],
                        "command_output_limits": None,
                        "use_delta_commands": True
                    },
                    "camera": {
                        "name": "JointController",
                        "motor_type": "position",
                        "command_input_limits": None,
                        "command_output_limits": None,
                        "use_delta_commands": False
                    }
                }
            }
        ],
    }

    # Make sure sim is stopped
    og.sim.stop()

    # Make sure GPU dynamics are enabled (GPU dynamics needed for cloth) and no flatcache
    gm.ENABLE_OBJECT_STATES = True
    gm.USE_GPU_DYNAMICS = False
    gm.ENABLE_FLATCACHE = False

    # Create the environment
    env = og.Environment(configs=cfg)
    robot = env.robots[0]
    env.reset()

    for obj in objects:
        og.sim.import_object(obj['object'])
        obj['object'].set_position_orientation(obj['position'], obj['orientation'])
        og.sim.step()

    controller = StarterSemanticActionPrimitives(env)
    set_start_pose(robot)
    for primitive, args in zip(primitives, primitives_args):
        try:
            execute_controller(controller.apply_ref(primitive, *args), env)
        except:
            og.sim.clear()
            return False

    # Clear the sim
    og.sim.clear()
    return True

@pytest.mark.skip(reason="primitives are broken")
def test_navigate():
    categories = ["floors", "ceilings", "walls"]

    objects = []
    obj_1 = {
        "object": DatasetObject(
            name="cologne",
            category="bottle_of_cologne",
            model="lyipur"
        ),
        "position": [-0.3, -0.8, 0.5],
        "orientation": [0, 0, 0, 1]
    }
    objects.append(obj_1)

    primitives = [StarterSemanticActionPrimitiveSet.NAVIGATE_TO]
    primitives_args = [(obj_1['object'],)]    

    assert primitive_tester(categories, objects, primitives, primitives_args)

@pytest.mark.skip(reason="primitives are broken")
def test_grasp():
    categories = ["floors", "ceilings", "walls", "coffee_table"]

    objects = []
    obj_1 = {
        "object": DatasetObject(
            name="cologne",
            category="bottle_of_cologne",
            model="lyipur"
        ),
        "position": [-0.3, -0.8, 0.5],
        "orientation": [0, 0, 0, 1]
    }
    objects.append(obj_1)

    primitives = [StarterSemanticActionPrimitiveSet.GRASP]
    primitives_args = [(obj_1['object'],)]    

    assert primitive_tester(categories, objects, primitives, primitives_args)

@pytest.mark.skip(reason="primitives are broken")
def test_place():
    categories = ["floors", "ceilings", "walls", "coffee_table"]

    objects = []
    obj_1 = {
        "object": DatasetObject(
                name="table",
                category="breakfast_table",
                model="rjgmmy",
                scale=[0.3, 0.3, 0.3]
            ),
        "position": [-0.7, 0.5, 0.2],
        "orientation": [0, 0, 0, 1]
    }
    obj_2 = {
        "object": DatasetObject(
            name="cologne",
            category="bottle_of_cologne",
            model="lyipur"
        ),
        "position": [-0.3, -0.8, 0.5],
        "orientation": [0, 0, 0, 1]
    }
    objects.append(obj_1)
    objects.append(obj_2)

    primitives = [StarterSemanticActionPrimitiveSet.GRASP, StarterSemanticActionPrimitiveSet.PLACE_ON_TOP]
    primitives_args = [(obj_2['object'],), (obj_1['object'],)]    

    assert primitive_tester(categories, objects, primitives, primitives_args)

@pytest.mark.skip(reason="primitives are broken")
def test_open_prismatic():
    categories = ["floors"]

    objects = []
    obj_1 = {
        "object": DatasetObject(
            name="bottom_cabinet",
            category="bottom_cabinet",
            model="bamfsz",
            scale=[0.7, 0.7, 0.7]
        ),
        "position": [-1.2, -0.4, 0.5],
        "orientation": [0, 0, 0, 1]
    }
    objects.append(obj_1)

    primitives = [StarterSemanticActionPrimitiveSet.OPEN]
    primitives_args = [(obj_1['object'],)]    

    assert primitive_tester(categories, objects, primitives, primitives_args)

@pytest.mark.skip(reason="primitives are broken")
def test_open_revolute():
    categories = ["floors"]

    objects = []
    obj_1 = {
        "object": DatasetObject(
            name="fridge",
            category="fridge",
            model="dszchb",
            scale=[0.7, 0.7, 0.7]
        ),
        "position": [-1.2, -0.4, 0.5],
        "orientation": [0, 0, 0, 1]
    }
    objects.append(obj_1)

    primitives = [StarterSemanticActionPrimitiveSet.OPEN]
    primitives_args = [(obj_1['object'],)]    

    assert primitive_tester(categories, objects, primitives, primitives_args)