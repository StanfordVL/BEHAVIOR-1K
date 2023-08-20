import omnigibson as og
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives, StarterSemanticActionPrimitiveSet
import omnigibson.utils.transform_utils as T
from omnigibson.objects.dataset_object import DatasetObject

def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        env.step(action)

# def main():
#     # Load the config
#     config_filename = os.path.join(og.example_config_path, "homeboy.yaml")
#     config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

#     # config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls", "coffee_table"]

#     # Load the environment
#     env = og.Environment(configs=config)
#     scene = env.scene
#     robot = env.robots[0]

#     robot._links['base_link'].mass = 10000

#     # Allow user to move camera more easily
#     og.sim.enable_viewer_camera_teleoperation()

#     # table = DatasetObject(
#     #     name="table",
#     #     category="breakfast_table",
#     #     model="rjgmmy",
#     # )
#     # og.sim.import_object(table)
#     # table.set_position([1.0, 1.0, 0.58])

#     # grasp_obj = DatasetObject(
#     #     name="potato",
#     #     category="bottle_of_cologne",
#     #     model="lyipur",
#     # )

#     # og.sim.import_object(grasp_obj)
#     # grasp_obj.set_position([-0.3, -0.8, 0.5])
#     # og.sim.step()

#     controller = StarterSemanticActionPrimitives(None, scene, robot)

#     def test_grasp():
#         grasp_obj, = scene.object_registry("category", "bottle_of_vodka")
#         execute_controller(controller.apply_ref(StarterSemanticActionPrimitiveSet.GRASP, grasp_obj), env)

#     def test_place():
#         box, = scene.object_registry("category", "storage_box")
#         execute_controller(controller.apply_ref(StarterSemanticActionPrimitiveSet.PLACE_INSIDE, box), env)

#     # Work more reliably
#     # IPython.embed()
#     # test_navigate_to_obj()
#     # test_grasp_no_navigation()
#     # test_grasp_replay_and_place()

#     og.sim.step()

#     # Don't work as reliably
#     test_grasp()
#     test_place()

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
                        "use_delta_commands": True,
                        "use_single_command": True
                    },
                    "gripper_right": {
                        "name": "JointController",
                        "motor_type": "position",
                        "command_input_limits": [-1, 1],
                        "command_output_limits": None,
                        "use_delta_commands": True,
                        "use_single_command": True
                    },
                    "camera": {
                        "name": "JointController",
                        "motor_type": "velocity",
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
    env = og.Environment(configs=cfg, action_timestep=1 / 60., physics_timestep=1 / 60.)
    scene = env.scene
    robot = env.robots[0]
    env.reset()

    for obj in objects:
        og.sim.import_object(obj['object'])
        obj['object'].set_position(obj['position'])
        og.sim.step()

    controller = StarterSemanticActionPrimitives(None, scene, robot)
    from IPython import embed; embed()
    for primitive, args in zip(primitives, primitives_args):
        try:
            print(*args)
            from IPython import embed; embed()
            execute_controller(controller.apply_ref(primitive, *args), env)
        except:
            og.sim.clear()
            return False

    # Clear the sim
    og.sim.clear()
    return True


def test_grasp():
    categories = ["floors", "ceilings", "walls", "coffee_table"]

    objects = []
    obj_1 = {
        "object": DatasetObject(
                name="table",
                category="breakfast_table",
                model="rjgmmy",
                scale=[0.3, 0.3, 0.3]
            ),
        "position": [-0.7, 0.5, 0.2]
    }
    obj_2 = {
        "object": DatasetObject(
            name="cologne",
            category="cologne",
            model="lyipur",
            scale=[0.01, 0.01, 0.01]
        ),
        "position": [-0.3, -0.8, 0.5]
    }
    objects.append(obj_1)
    objects.append(obj_2)

    primitives = [StarterSemanticActionPrimitiveSet.GRASP]
    primitives_args = [(obj_2['object'],)]    

    success = primitive_tester(categories, objects, primitives, primitives_args)
    print("Grasp: ", success)

if __name__ == "__main__":
    test_grasp()