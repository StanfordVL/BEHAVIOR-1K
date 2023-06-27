import yaml
import numpy as np

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.object_states import ContactBodies
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.utils.motion_planning_utils import detect_robot_collision
from pxr import PhysxSchema


# Load the config
config_filename = "test.yaml"
config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls"]

# Load the environment
env = og.Environment(configs=config)
scene = env.scene
robot = env.robots[0]

# Allow user to move camera more easily
og.sim.enable_viewer_camera_teleoperation()

table = DatasetObject(
    name="table",
    category="breakfast_table",
    model="rjgmmy",
    scale = 0.3
)
og.sim.import_object(table)
table.set_position([1.0, 1.0, 0.58])
og.sim.step()

def pause(time):
    for _ in range(int(time*100)):
        # og.sim.render()
        og.sim.step()

def _get_robot_pose_from_2d_pose(pose_2d):
        pos = np.array([pose_2d[0], pose_2d[1], 0.05])
        orn = T.euler2quat([0, 0, pose_2d[2]])
        return pos, orn

def detect_robot_collision(robot, filter_objs=[]):
    filter_objects = [o.name for o in filter_objs] + ["floor", "potato"]
    obj_in_hand = obj_in_hand = robot._ag_obj_in_hand[robot.default_arm] 
    if obj_in_hand is not None:
        filter_objects.append(obj_in_hand.name)
    # collision_objects = list(filter(lambda obj : "floor" not in obj.name, robot.states[ContactBodies].get_value()))
    collision_objects = robot.states[ContactBodies].get_value()
    filtered_collision_objects = []
    for col_obj in collision_objects:
        if not any([f in col_obj.name for f in filter_objects]):
            filtered_collision_objects.append(col_obj)
    # print("-----")
    # print(filtered_collision_objects)
    # for f in filtered_collision_objects:
    #     if obj_in_hand is not None:
    #         print(obj_in_hand.name)
    #     print(f.name)
    return len(filtered_collision_objects) > 0

class UndoableContext(object):
    def __init__(self):
        pass

    def __enter__(self):
        self.state = og.sim.dump_state(serialized=False)
        og.sim._physics_context.set_gravity(value=0.0)
        for obj in og.sim.scene.objects:
            for link in obj.links.values():
                PhysxSchema.PhysxRigidBodyAPI(link.prim).GetSolveContactAttr().Set(False)
            obj.keep_still()

    def __exit__(self, *args):
        og.sim._physics_context.set_gravity(value=-9.81)
        for obj in og.sim.scene.objects:
            for link in obj.links.values():
                PhysxSchema.PhysxRigidBodyAPI(link.prim).GetSolveContactAttr().Set(True)
        og.sim.step()
        og.sim.step()
        og.sim.load_state(self.state, serialized=False)    

while True:
    with UndoableContext():
        pose_2d = [1.21189, 0.625961, 1.87304]
        pose = _get_robot_pose_from_2d_pose(pose_2d)
        robot.set_position_orientation(*pose)
        og.sim.step()
        print(detect_robot_collision(robot))
        print("-------")
        pause(2)
