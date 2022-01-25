"""
Developed by Caelen Garrett in pybullet-planning repository (https://github.com/caelan/pybullet-planning)
and adapted by iGibson team.
"""
from __future__ import print_function

import colorsys
import json
import math
import os
import pickle
import platform
import numpy as np

import random
import sys
import time
import datetime
from collections import defaultdict, deque, namedtuple
from itertools import product, combinations, count

from .transformations import quaternion_from_matrix, unit_vector

from igibson.external.motion.motion_planners.rrt_connect import birrt, direct_path
from igibson.external.motion.motion_planners.rrt_star import rrt_star
from igibson.external.motion.motion_planners.lazy_prm import lazy_prm_replan_loop
from igibson.external.motion.motion_planners.rrt import rrt
from igibson.external.motion.motion_planners.smoothing import optimize_path
from igibson.utils.constants import OccupancyGridState
#from ..motion.motion_planners.rrt_connect import birrt, direct_path
import cv2
import logging

# from future_builtins import map, filter
# from builtins import input # TODO - use future
try:
    user_input = raw_input
except NameError:
    user_input = input

INF = np.inf
PI = np.pi
CIRCULAR_LIMITS = -PI, PI
UNBOUNDED_LIMITS = -INF, INF
DEFAULT_TIME_STEP = 1./240.  # seconds

#####################################

DRAKE_PATH = 'models/drake/'

# Models

# Robots
MODEL_DIRECTORY = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, 'models/'))
ROOMBA_URDF = 'models/turtlebot/roomba.urdf'
TURTLEBOT_URDF = 'models/turtlebot/turtlebot_holonomic.urdf'
DRAKE_IIWA_URDF = 'models/drake/iiwa_description/urdf/iiwa14_polytope_collision.urdf'
# wsg_50 | wsg_50_mesh_visual | wsg_50_mesh_collision
WSG_50_URDF = 'models/drake/wsg_50_description/urdf/wsg_50_mesh_visual.urdf'
#SCHUNK_URDF = 'models/drake/wsg_50_description/sdf/schunk_wsg_50.sdf'
PANDA_HAND_URDF = "models/franka_description/robots/hand.urdf"
PANDA_ARM_URDF = "models/franka_description/robots/panda_arm_hand.urdf"

# Pybullet Robots
#PYBULLET_DIRECTORY = add_data_path()
KUKA_IIWA_URDF = "kuka_iiwa/model.urdf"
KUKA_IIWA_GRIPPER_SDF = "kuka_iiwa/kuka_with_gripper.sdf"
R2D2_URDF = "r2d2.urdf"
MINITAUR_URDF = "quadruped/minitaur.urdf"
HUMANOID_MJCF = "mjcf/humanoid.xml"
HUSKY_URDF = "husky/husky.urdf"
RACECAR_URDF = 'racecar/racecar.urdf'  # racecar_differential.urdf
PR2_GRIPPER = 'pr2_gripper.urdf'
# wsg50_one_motor_gripper | wsg50_one_motor_gripper_new
WSG_GRIPPER = 'gripper/wsg50_one_motor_gripper.sdf'

# Pybullet Objects
KIVA_SHELF_SDF = "kiva_shelf/model.sdf"
FLOOR_URDF = 'plane.urdf'
TABLE_URDF = 'table'

# Objects
SMALL_BLOCK_URDF = "models/drake/objects/block_for_pick_and_place.urdf"
BLOCK_URDF = "models/drake/objects/block_for_pick_and_place_mid_size.urdf"
SINK_URDF = 'models/sink.urdf'
STOVE_URDF = 'models/stove.urdf'

#####################################

# I/O

SEPARATOR = '\n' + 50*'-' + '\n'

# def inf_generator():
#    return iter(int, 1)

inf_generator = count


def print_separator(n=50):
    print('\n' + n*'-' + '\n')


def is_remote():
    return 'SSH_CONNECTION' in os.environ


def is_darwin():  # TODO: change loading accordingly
    return platform.system() == 'Darwin'  # platform.release()
    # return sys.platform == 'darwin'


def read(filename):
    with open(filename, 'r') as f:
        return f.read()


def write(filename, string):
    with open(filename, 'w') as f:
        f.write(string)


def read_pickle(filename):
    # Can sometimes read pickle3 from python2 by calling twice
    # Can possibly read pickle2 from python3 by using encoding='latin1'
    with open(filename, 'rb') as f:
        return pickle.load(f)


def write_pickle(filename, data):  # NOTE - cannot pickle lambda or nested functions
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def read_json(path):
    return json.loads(read(path))


def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)


def safe_remove(p):
    if os.path.exists(p):
        os.remove(p)


def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)


def safe_zip(sequence1, sequence2):
    assert len(sequence1) == len(sequence2)
    return zip(sequence1, sequence2)


def clip(value, min_value=-INF, max_value=+INF):
    return min(max(min_value, value), max_value)


def randomize(iterable):  # TODO: bisect
    sequence = list(iterable)
    random.shuffle(sequence)
    return sequence


def get_random_seed():
    return random.getstate()[1][0]


def get_numpy_seed():
    return np.random.get_state()[1][0]


def set_random_seed(seed):
    if seed is not None:
        random.seed(seed)


def set_numpy_seed(seed):
    # These generators are different and independent
    if seed is not None:
        np.random.seed(seed % (2**32))
        #print('Seed:', seed)


def get_date():
    return datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')


def implies(p1, p2):
    return not p1 or p2

#####################################

# https://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python/14797594#14797594
# https://stackoverflow.com/questions/4178614/suppressing-output-of-module-calling-outside-library
# https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python/22434262#22434262


class HideOutput(object):
    '''
    A context manager that block stdout for its scope, usage:

    with HideOutput():
        os.system('ls -l')
    '''
    DEFAULT_ENABLE = True

    def __init__(self, enable=None):
        if enable is None:
            enable = self.DEFAULT_ENABLE
        self.enable = enable
        if not self.enable:
            return
        sys.stdout.flush()
        self._origstdout = sys.stdout
        self._oldstdout_fno = os.dup(sys.stdout.fileno())
        self._devnull = os.open(os.devnull, os.O_WRONLY)

    def __enter__(self):
        if not self.enable:
            return
        self._newstdout = os.dup(1)
        os.dup2(self._devnull, 1)
        os.close(self._devnull)
        sys.stdout = os.fdopen(self._newstdout, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enable:
            return
        sys.stdout.close()
        sys.stdout = self._origstdout
        sys.stdout.flush()
        os.dup2(self._oldstdout_fno, 1)
        os.close(self._oldstdout_fno)  # Added

#####################################

# Savers

# TODO: contextlib


class Saver(object):
    def restore(self):
        raise NotImplementedError()

    def __enter__(self):
        # TODO: move the saving to enter?
        pass

    def __exit__(self, type, value, traceback):
        self.restore()


class ClientSaver(Saver):
    def __init__(self, new_client=None):
        self.client = CLIENT
        if new_client is not None:
            set_client(new_client)

    def restore(self):
        set_client(self.client)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.client)


class VideoSaver(Saver):
    def __init__(self, path):
        self.path = path
        if path is None:
            self.log_id = None
        else:
            name, ext = os.path.splitext(path)
            assert ext == '.mp4'
            # STATE_LOGGING_PROFILE_TIMINGS, STATE_LOGGING_ALL_COMMANDS
            # p.submitProfileTiming("pythontest")
            self.log_id = p.startStateLogging(
                p.STATE_LOGGING_VIDEO_MP4, fileName=path, physicsClientId=CLIENT)

    def restore(self):
        if self.log_id is not None:
            p.stopStateLogging(self.log_id)

#####################################


class PoseSaver(Saver):
    def __init__(self, body):
        self.body = body
        self.pose = get_pose(self.body)
        self.velocity = get_velocity(self.body)

    def apply_mapping(self, mapping):
        self.body = mapping.get(self.body, self.body)

    def restore(self):
        set_pose(self.body, self.pose)
        set_velocity(self.body, *self.velocity)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.body)


class ConfSaver(Saver):
    def __init__(self, body):  # , joints):
        self.body = body
        self.conf = get_configuration(body)

    def apply_mapping(self, mapping):
        self.body = mapping.get(self.body, self.body)

    def restore(self):
        set_configuration(self.body, self.conf)

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.body)

#####################################


class BodySaver(Saver):
    def __init__(self, body):  # , pose=None):
        # if pose is None:
        #    pose = get_pose(body)
        self.body = body
        self.pose_saver = PoseSaver(body)
        self.conf_saver = ConfSaver(body)
        self.savers = [self.pose_saver, self.conf_saver]
        # TODO: store velocities

    def apply_mapping(self, mapping):
        for saver in self.savers:
            saver.apply_mapping(mapping)

    def restore(self):
        for saver in self.savers:
            saver.restore()

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.body)


class WorldSaver(Saver):
    def __init__(self):
        self.body_savers = [BodySaver(body) for body in get_bodies()]
        # TODO: add/remove new bodies

    def restore(self):
        for body_saver in self.body_savers:
            body_saver.restore()

#####################################

# Simulation


CLIENTS = {}
CLIENT = 0


def get_client(client=None):
    if client is None:
        return CLIENT
    return client


def set_client(client):
    global CLIENT
    CLIENT = client


ModelInfo = namedtuple('URDFInfo', ['name', 'path', 'fixed_base', 'scale'])

INFO_FROM_BODY = {}


def get_model_info(body):
    key = (CLIENT, body)
    return INFO_FROM_BODY.get(key, None)


def get_urdf_flags(cache=False, cylinder=False):
    # by default, Bullet disables self-collision
    # URDF_INITIALIZE_SAT_FEATURES
    # URDF_ENABLE_CACHED_GRAPHICS_SHAPES seems to help
    # but URDF_INITIALIZE_SAT_FEATURES does not (might need to be provided a mesh)
    # flags = p.URDF_INITIALIZE_SAT_FEATURES | p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
    flags = 0
    if cache:
        flags |= p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
    if cylinder:
        flags |= p.URDF_USE_IMPLICIT_CYLINDER
    return flags


def load_pybullet(filename, fixed_base=False, scale=1., **kwargs):
    # fixed_base=False implies infinite base mass
    with LockRenderer():
        if filename.endswith('.urdf'):
            flags = get_urdf_flags(**kwargs)
            body = p.loadURDF(filename, useFixedBase=fixed_base, flags=flags,
                              globalScaling=scale, physicsClientId=CLIENT)
        elif filename.endswith('.sdf'):
            body = p.loadSDF(filename, physicsClientId=CLIENT)
        elif filename.endswith('.xml'):
            body = p.loadMJCF(filename, physicsClientId=CLIENT)
        elif filename.endswith('.bullet'):
            body = p.loadBullet(filename, physicsClientId=CLIENT)
        elif filename.endswith('.obj'):
            # TODO: fixed_base => mass = 0?
            body = create_obj(filename, scale=scale, **kwargs)
        else:
            raise ValueError(filename)
    INFO_FROM_BODY[CLIENT, body] = ModelInfo(None, filename, fixed_base, scale)
    return body


def set_caching(cache):
    p.setPhysicsEngineParameter(
        enableFileCaching=int(cache), physicsClientId=CLIENT)


def load_model_info(info):
    # TODO: disable file caching to reuse old filenames
    # p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=CLIENT)
    if info.path.endswith('.urdf'):
        return load_pybullet(info.path, fixed_base=info.fixed_base, scale=info.scale)
    if info.path.endswith('.obj'):
        mass = STATIC_MASS if info.fixed_base else 1.
        return create_obj(info.path, mass=mass, scale=info.scale)
    raise NotImplementedError(info.path)


URDF_FLAGS = [p.URDF_USE_INERTIA_FROM_FILE,
              p.URDF_USE_SELF_COLLISION,
              p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
              p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS]


def get_model_path(rel_path):  # TODO: add to search path
    directory = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(directory, '..', rel_path)


def load_model(rel_path, pose=None, **kwargs):
    # TODO: error with loadURDF when loading MESH visual and CYLINDER collision
    abs_path = get_model_path(rel_path)
    add_data_path()
    # with LockRenderer():
    body = load_pybullet(abs_path, **kwargs)
    if pose is not None:
        set_pose(body, pose)
    return body

#####################################

# class World(object):
#     def __init__(self, client):
#         self.client = client
#         self.bodies = {}
#     def activate(self):
#         set_client(self.client)
#     def load(self, path, name=None, fixed_base=False, scale=1.):
#         body = p.loadURDF(path, useFixedBase=fixed_base, physicsClientId=self.client)
#         self.bodies[body] = URDFInfo(name, path, fixed_base, scale)
#         return body
#     def remove(self, body):
#         del self.bodies[body]
#         return p.removeBody(body, physicsClientId=self.client)
#     def reset(self):
#         p.resetSimulation(physicsClientId=self.client)
#         self.bodies = {}
#     # TODO: with statement
#     def copy(self):
#         raise NotImplementedError()
#     def __repr__(self):
#         return '{}({})'.format(self.__class__.__name__, len(self.bodies))

#####################################


def elapsed_time(start_time):
    return time.time() - start_time


MouseEvent = namedtuple('MouseEvent', [
                        'eventType', 'mousePosX', 'mousePosY', 'buttonIndex', 'buttonState'])


def get_mouse_events():
    return list(MouseEvent(*event) for event in p.getMouseEvents(physicsClientId=CLIENT))


def update_viewer():
    # https://docs.python.org/2/library/select.html
    # events = p.getKeyboardEvents() # TODO: only works when the viewer is in focus
    get_mouse_events()
    # for k, v in keys.items():
    #    #p.KEY_IS_DOWN, p.KEY_WAS_RELEASED, p.KEY_WAS_TRIGGERED
    #    if (k == p.B3G_RETURN) and (v & p.KEY_WAS_TRIGGERED):
    #        return
    # time.sleep(1e-3) # Doesn't work
    # disable_gravity()


def wait_for_duration(duration):  # , dt=0):
    t0 = time.time()
    while elapsed_time(t0) <= duration:
        update_viewer()


def simulate_for_duration(duration):
    dt = get_time_step()
    for i in range(int(duration / dt)):
        step_simulation()


def get_time_step():
    # {'gravityAccelerationX', 'useRealTimeSimulation', 'gravityAccelerationZ', 'numSolverIterations',
    # 'gravityAccelerationY', 'numSubSteps', 'fixedTimeStep'}
    return p.getPhysicsEngineParameters(physicsClientId=CLIENT)['fixedTimeStep']


def enable_separating_axis_test():
    p.setPhysicsEngineParameter(enableSAT=1, physicsClientId=CLIENT)
    # p.setCollisionFilterPair()
    # p.setCollisionFilterGroupMask()
    # p.setInternalSimFlags()
    # enableFileCaching: Set to 0 to disable file caching, such as .obj wavefront file loading
    # p.getAPIVersion() # TODO: check that API is up-to-date
    # p.isNumpyEnabled()


def simulate_for_sim_duration(sim_duration, real_dt=0, frequency=INF):
    t0 = time.time()
    sim_dt = get_time_step()
    sim_time = 0
    last_print = 0
    while sim_time < sim_duration:
        if frequency < (sim_time - last_print):
            print('Sim time: {:.3f} | Real time: {:.3f}'.format(
                sim_time, elapsed_time(t0)))
            last_print = sim_time
        step_simulation()
        sim_time += sim_dt
        time.sleep(real_dt)


def wait_for_user(message='Press enter to continue'):
    if is_darwin():
        # OS X doesn't multi-thread the OpenGL visualizer
        # wait_for_interrupt()
        return threaded_input(message)
    return user_input(message)


def is_unlocked():
    return CLIENTS[CLIENT] is True


def wait_if_unlocked(*args, **kwargs):
    if is_unlocked():
        wait_for_user(*args, **kwargs)


def wait_for_interrupt(max_time=np.inf):
    """
    Hold Ctrl to move the camera as well as zoom
    """
    print('Press Ctrl-C to continue')
    try:
        wait_for_duration(max_time)
    except KeyboardInterrupt:
        pass
    finally:
        print()


def disable_viewer():
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, False, physicsClientId=CLIENT)
    p.configureDebugVisualizer(
        p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, False, physicsClientId=CLIENT)
    p.configureDebugVisualizer(
        p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, False, physicsClientId=CLIENT)
    p.configureDebugVisualizer(
        p.COV_ENABLE_RGB_BUFFER_PREVIEW, False, physicsClientId=CLIENT)
    #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, False, physicsClientId=CLIENT)
    #p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, True, physicsClientId=CLIENT)
    #p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, False, physicsClientId=CLIENT)
    #p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, True, physicsClientId=CLIENT)
    #p.COV_ENABLE_MOUSE_PICKING, p.COV_ENABLE_KEYBOARD_SHORTCUTS


def set_renderer(enable):
    client = CLIENT
    if not has_gui(client):
        return
    CLIENTS[client] = enable
    p.configureDebugVisualizer(
        p.COV_ENABLE_RENDERING, int(enable), physicsClientId=client)


class LockRenderer(Saver):
    # disabling rendering temporary makes adding objects faster
    def __init__(self, lock=True):
        self.client = CLIENT
        self.state = CLIENTS[self.client]
        # skip if the visualizer isn't active
        if has_gui(self.client) and lock:
            set_renderer(enable=False)

    def restore(self):
        if not has_gui(self.client):
            return
        assert self.state is not None
        if self.state != CLIENTS[self.client]:
            set_renderer(enable=self.state)


def connect(use_gui=True, shadows=True):
    # Shared Memory: execute the physics simulation and rendering in a separate process
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/vrminitaur.py#L7
    # make sure to compile pybullet with PYBULLET_USE_NUMPY enabled
    if use_gui and not is_darwin() and ('DISPLAY' not in os.environ):
        use_gui = False
        print('No display detected!')
    method = p.GUI if use_gui else p.DIRECT
    with HideOutput():
        # options="--width=1024 --height=768"
        #  --window_backend=2 --render_device=0'
        sim_id = p.connect(method)
        #sim_id = p.connect(p.GUI, options="--opengl2") if use_gui else p.connect(p.DIRECT)
    assert 0 <= sim_id
    #sim_id2 = p.connect(p.SHARED_MEMORY)
    #print(sim_id, sim_id2)
    CLIENTS[sim_id] = True if use_gui else None
    if use_gui:
        # p.COV_ENABLE_PLANAR_REFLECTION
        # p.COV_ENABLE_SINGLE_STEP_RENDERING
        p.configureDebugVisualizer(
            p.COV_ENABLE_GUI, False, physicsClientId=sim_id)
        p.configureDebugVisualizer(
            p.COV_ENABLE_TINY_RENDERER, False, physicsClientId=sim_id)
        p.configureDebugVisualizer(
            p.COV_ENABLE_RGB_BUFFER_PREVIEW, False, physicsClientId=sim_id)
        p.configureDebugVisualizer(
            p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, False, physicsClientId=sim_id)
        p.configureDebugVisualizer(
            p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, False, physicsClientId=sim_id)
        p.configureDebugVisualizer(
            p.COV_ENABLE_SHADOWS, shadows, physicsClientId=sim_id)

    # you can also use GUI mode, for faster OpenGL rendering (instead of TinyRender CPU)
    # visualizer_options = {
    #    p.COV_ENABLE_WIREFRAME: 1,
    #    p.COV_ENABLE_SHADOWS: 0,
    #    p.COV_ENABLE_RENDERING: 0,
    #    p.COV_ENABLE_TINY_RENDERER: 1,
    #    p.COV_ENABLE_RGB_BUFFER_PREVIEW: 0,
    #    p.COV_ENABLE_DEPTH_BUFFER_PREVIEW: 0,
    #    p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW: 0,
    #    p.COV_ENABLE_VR_RENDER_CONTROLLERS: 0,
    #    p.COV_ENABLE_VR_PICKING: 0,
    #    p.COV_ENABLE_VR_TELEPORTING: 0,
    # }
    # for pair in visualizer_options.items():
    #    p.configureDebugVisualizer(*pair)
    return sim_id


def threaded_input(*args, **kwargs):
    # OS X doesn't multi-thread the OpenGL visualizer
    # http://openrave.org/docs/0.8.2/_modules/openravepy/misc/#SetViewerUserThread
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/userData.py
    # https://github.com/bulletphysics/bullet3/tree/master/examples/ExampleBrowser
    #from pybullet_utils import bullet_client
    #from pybullet_utils.bullet_client import BulletClient
    # server = bullet_client.BulletClient(connection_mode=p.SHARED_MEMORY_SERVER) # GUI_SERVER
    #sim_id = p.connect(p.GUI)
    # print(dir(server))
    #client = bullet_client.BulletClient(connection_mode=p.SHARED_MEMORY)
    #sim_id = p.connect(p.SHARED_MEMORY)

    #threading = __import__('threading')
    import threading
    data = []
    thread = threading.Thread(target=lambda: data.append(
        user_input(*args, **kwargs)), args=[])
    thread.start()
    # threading.enumerate()
    #thread_id = 0
    # for tid, tobj in threading._active.items():
    #    if tobj is thread:
    #        thread_id = tid
    #        break
    try:
        while thread.is_alive():
            update_viewer()
    finally:
        thread.join()
    return data[-1]


def disconnect():
    # TODO: change CLIENT?
    if CLIENT in CLIENTS:
        del CLIENTS[CLIENT]
    with HideOutput():
        return p.disconnect(physicsClientId=CLIENT)


def is_connected():
    return p.getConnectionInfo(physicsClientId=CLIENT)['isConnected']


def get_connection(client=None):
    return p.getConnectionInfo(physicsClientId=get_client(client))['connectionMethod']


def has_gui(client=None):
    return get_connection(get_client(client)) == p.GUI


def get_data_path():
    import pybullet_data
    return pybullet_data.getDataPath()


def add_data_path(data_path=None):
    if data_path is None:
        data_path = get_data_path()
    p.setAdditionalSearchPath(data_path)
    return data_path


GRAVITY = 9.8


def enable_gravity():
    p.setGravity(0, 0, -GRAVITY, physicsClientId=CLIENT)


def disable_gravity():
    p.setGravity(0, 0, 0, physicsClientId=CLIENT)


def step_simulation():
    p.stepSimulation(physicsClientId=CLIENT)


def set_real_time(real_time):
    p.setRealTimeSimulation(int(real_time), physicsClientId=CLIENT)


def enable_real_time():
    set_real_time(True)


def disable_real_time():
    set_real_time(False)


def update_state():
    # TODO: this doesn't seem to automatically update still
    disable_gravity()
    # step_simulation()
    # for body in get_bodies():
    #    for link in get_links(body):
    #        # if set to 1 (or True), the Cartesian world position/orientation
    #        # will be recomputed using forward kinematics.
    #        get_link_state(body, link)
    # for body in get_bodies():
    #    get_pose(body)
    #    for joint in get_joints(body):
    #        get_joint_position(body, joint)
    # p.getKeyboardEvents()
    # p.getMouseEvents()


def reset_simulation():
    p.resetSimulation(physicsClientId=CLIENT)


CameraInfo = namedtuple('CameraInfo', ['width', 'height', 'viewMatrix', 'projectionMatrix', 'cameraUp', 'cameraForward',
                                       'horizontal', 'vertical', 'yaw', 'pitch', 'dist', 'target'])


def get_camera():
    return CameraInfo(*p.getDebugVisualizerCamera(physicsClientId=CLIENT))


def set_camera(yaw, pitch, distance, target_position=np.zeros(3)):
    p.resetDebugVisualizerCamera(
        distance, yaw, pitch, target_position, physicsClientId=CLIENT)


def get_pitch(point):
    dx, dy, dz = point
    return np.math.atan2(dz, np.sqrt(dx ** 2 + dy ** 2))


def get_yaw(point):
    dx, dy, dz = point
    return np.math.atan2(dy, dx)


def set_camera_pose(camera_point, target_point=np.zeros(3)):
    delta_point = np.array(target_point) - np.array(camera_point)
    distance = np.linalg.norm(delta_point)
    yaw = get_yaw(delta_point) - np.pi/2  # TODO: hack
    pitch = get_pitch(delta_point)
    p.resetDebugVisualizerCamera(distance, math.degrees(yaw), math.degrees(pitch),
                                 target_point, physicsClientId=CLIENT)


def set_camera_pose2(world_from_camera, distance=2):
    target_camera = np.array([0, 0, distance])
    target_world = tform_point(world_from_camera, target_camera)
    camera_world = point_from_pose(world_from_camera)
    set_camera_pose(camera_world, target_world)
    #roll, pitch, yaw = euler_from_quat(quat_from_pose(world_from_camera))
    # TODO: assert that roll is about zero?
    # p.resetDebugVisualizerCamera(cameraDistance=distance, cameraYaw=math.degrees(yaw), cameraPitch=math.degrees(-pitch),
    #                             cameraTargetPosition=target_world, physicsClientId=CLIENT)


CameraImage = namedtuple(
    'CameraImage', ['rgbPixels', 'depthPixels', 'segmentationMaskBuffer'])


def demask_pixel(pixel):
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/segmask_linkindex.py
    # Not needed when p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX is not enabled
    # if 0 <= pixel:
    #    return None
    # Returns a large value when undefined
    body = pixel & ((1 << 24) - 1)
    link = (pixel >> 24) - 1
    return body, link


def save_image(filename, rgba):
    import scipy.misc
    if filename.endswith('.jpg'):
        scipy.misc.imsave(filename, rgba[:, :, :3])
    elif filename.endswith('.png'):
        scipy.misc.imsave(filename, rgba)  # (480, 640, 4)
        # scipy.misc.toimage(image_array, cmin=0.0, cmax=...).save('outfile.jpg')
    else:
        raise ValueError(filename)
    print('Saved image at {}'.format(filename))


def get_projection_matrix(width, height, vertical_fov, near, far):
    """
    OpenGL projection matrix
    :param width: 
    :param height: 
    :param vertical_fov: vertical field of view in radians
    :param near: 
    :param far: 
    :return: 
    """
    # http://www.songho.ca/opengl/gl_projectionmatrix.html
    # http://www.songho.ca/opengl/gl_transform.html#matrix
    # https://www.edmundoptics.fr/resources/application-notes/imaging/understanding-focal-length-and-field-of-view/
    # gluPerspective() requires only 4 parameters; vertical field of view (FOV),
    # the aspect ratio of width to height and the distances to near and far clipping planes.
    aspect = float(width) / height
    fov_degrees = math.degrees(vertical_fov)
    projection_matrix = p.computeProjectionMatrixFOV(fov=fov_degrees, aspect=aspect,
                                                     nearVal=near, farVal=far, physicsClientId=CLIENT)
    # projection_matrix = p.computeProjectionMatrix(0, width, height, 0, near, far, physicsClientId=CLIENT)
    return projection_matrix
    # return np.reshape(projection_matrix, [4, 4])


RED = (1, 0, 0, 1)
GREEN = (0, 1, 0, 1)
BLUE = (0, 0, 1, 1)
BLACK = (0, 0, 0, 1)
WHITE = (1, 1, 1, 1)
BROWN = (0.396, 0.263, 0.129, 1)
TAN = (0.824, 0.706, 0.549, 1)
GREY = (0.5, 0.5, 0.5, 1)
YELLOW = (1, 1, 0, 1)

COLOR_FROM_NAME = {
    'red': RED,
    'green': GREEN,
    'blue': BLUE,
    'white': WHITE,
    'grey': GREY,
    'black': BLACK,
}


def apply_alpha(color, alpha=1.0):
    return tuple(color[:3]) + (alpha,)


def spaced_colors(n, s=1, v=1):
    return [colorsys.hsv_to_rgb(h, s, v) for h in np.linspace(0, 1, n, endpoint=False)]


def image_from_segmented(segmented, color_from_body=None):
    if color_from_body is None:
        bodies = get_bodies()
        color_from_body = dict(zip(bodies, spaced_colors(len(bodies))))
    image = np.zeros(segmented.shape[:2] + (3,))
    for r in range(segmented.shape[0]):
        for c in range(segmented.shape[1]):
            body, link = segmented[r, c, :]
            image[r, c, :] = color_from_body.get(body, (0, 0, 0))
    return image


def get_image(camera_pos, target_pos, width=640, height=480, vertical_fov=60.0, near=0.02, far=5.0,
              segment=False, segment_links=False):
    # computeViewMatrixFromYawPitchRoll
    view_matrix = p.computeViewMatrix(cameraEyePosition=camera_pos, cameraTargetPosition=target_pos,
                                      cameraUpVector=[0, 0, 1], physicsClientId=CLIENT)
    projection_matrix = get_projection_matrix(
        width, height, vertical_fov, near, far)
    if segment:
        if segment_links:
            flags = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
        else:
            flags = 0
    else:
        flags = p.ER_NO_SEGMENTATION_MASK
    image = CameraImage(*p.getCameraImage(width, height, viewMatrix=view_matrix,
                                          projectionMatrix=projection_matrix,
                                          shadow=False,
                                          flags=flags,
                                          renderer=p.ER_TINY_RENDERER,  # p.ER_BULLET_HARDWARE_OPENGL
                                          physicsClientId=CLIENT)[2:])
    depth = far * near / (far - (far - near) * image.depthPixels)
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/pointCloudFromCameraImage.py
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/getCameraImageTest.py
    segmented = None
    if segment:
        segmented = np.zeros(image.segmentationMaskBuffer.shape + (2,))
        for r in range(segmented.shape[0]):
            for c in range(segmented.shape[1]):
                pixel = image.segmentationMaskBuffer[r, c]
                segmented[r, c, :] = demask_pixel(pixel)
    return CameraImage(image.rgbPixels, depth, segmented)


def set_default_camera():
    set_camera(160, -35, 2.5, Point())


def save_state():
    return p.saveState(physicsClientId=CLIENT)


def restore_state(state_id):
    p.restoreState(stateId=state_id, physicsClientId=CLIENT)


def save_bullet(filename):
    p.saveBullet(filename, physicsClientId=CLIENT)


def restore_bullet(filename):
    p.restoreState(fileName=filename, physicsClientId=CLIENT)

#####################################

# Geometry

#Pose = namedtuple('Pose', ['position', 'orientation'])


def Point(x=0., y=0., z=0.):
    return np.array([x, y, z])


def Euler(roll=0., pitch=0., yaw=0.):
    return np.array([roll, pitch, yaw])


def Pose(point=None, euler=None):
    point = Point() if point is None else point
    euler = Euler() if euler is None else euler
    return (point, quat_from_euler(euler))

# def Pose2d(x=0., y=0., yaw=0.):
#    return np.array([x, y, yaw])


def invert(pose):
    (point, quat) = pose
    return p.invertTransform(point, quat)


def multiply(*poses):
    pose = poses[0]
    for next_pose in poses[1:]:
        pose = p.multiplyTransforms(pose[0], pose[1], *next_pose)
    return pose


def invert_quat(quat):
    pose = (unit_point(), quat)
    return quat_from_pose(invert(pose))


def multiply_quats(*quats):
    return quat_from_pose(multiply(*[(unit_point(), quat) for quat in quats]))


def unit_from_theta(theta):
    return np.array([np.cos(theta), np.sin(theta)])


def quat_from_euler(euler):
    return p.getQuaternionFromEuler(euler)


def euler_from_quat(quat):
    return p.getEulerFromQuaternion(quat)


def unit_point():
    return (0., 0., 0.)


def unit_quat():
    return quat_from_euler([0, 0, 0])  # [X,Y,Z,W]


def quat_from_axis_angle(axis, angle):  # axis-angle
    # return get_unit_vector(np.append(vec, [angle]))
    return np.append(math.sin(angle/2) * get_unit_vector(axis), [math.cos(angle / 2)])


def unit_pose():
    return (unit_point(), unit_quat())


def get_length(vec, norm=2):
    return np.linalg.norm(vec, ord=norm)


def get_difference(p1, p2):
    return np.array(p2) - np.array(p1)


def get_distance(p1, p2, **kwargs):
    return get_length(get_difference(p1, p2), **kwargs)


def angle_between(vec1, vec2):
    return np.math.acos(np.dot(vec1, vec2) / (get_length(vec1) * get_length(vec2)))


def get_angle(q1, q2):
    dx, dy = np.array(q2[:2]) - np.array(q1[:2])
    return np.math.atan2(dy, dx)


def get_unit_vector(vec):
    norm = get_length(vec)
    if norm == 0:
        return vec
    return np.array(vec) / norm


def z_rotation(theta):
    return quat_from_euler([0, 0, theta])


def matrix_from_quat(quat):
    return np.array(p.getMatrixFromQuaternion(quat, physicsClientId=CLIENT)).reshape(3, 3)


def quat_from_matrix(mat):
    matrix = np.eye(4)
    matrix[:3, :3] = mat
    return quaternion_from_matrix(matrix)


def point_from_tform(tform):
    return np.array(tform)[:3, 3]


def matrix_from_tform(tform):
    return np.array(tform)[:3, :3]


def point_from_pose(pose):
    return pose[0]


def quat_from_pose(pose):
    return pose[1]


def tform_from_pose(pose):
    (point, quat) = pose
    tform = np.eye(4)
    tform[:3, 3] = point
    tform[:3, :3] = matrix_from_quat(quat)
    return tform


def pose_from_tform(tform):
    return point_from_tform(tform), quat_from_matrix(matrix_from_tform(tform))


def wrap_angle(theta, lower=-np.pi):  # [-np.pi, np.pi)
    return (theta - lower) % (2 * np.pi) + lower


def circular_difference(theta2, theta1):
    return wrap_angle(theta2 - theta1)


def base_values_from_pose(pose, tolerance=1e-3):
    (point, quat) = pose
    x, y, _ = point
    roll, pitch, yaw = euler_from_quat(quat)
    #assert (abs(roll) < tolerance) and (abs(pitch) < tolerance)
    return (x, y, yaw)


pose2d_from_pose = base_values_from_pose


def pose_from_base_values(base_values, default_pose=unit_pose()):
    x, y, yaw = base_values
    _, _, z = point_from_pose(default_pose)
    roll, pitch, _ = euler_from_quat(quat_from_pose(default_pose))
    return (x, y, z), quat_from_euler([roll, pitch, yaw])


def quat_angle_between(quat0, quat1):  # quaternion_slerp
    # p.computeViewMatrixFromYawPitchRoll()
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    d = clip(np.dot(q0, q1), min_value=-1., max_value=+1.)
    angle = math.acos(d)
    # TODO: angle_between
    #delta = p.getDifferenceQuaternion(quat0, quat1)
    #angle = math.acos(delta[-1])
    return angle


def all_between(lower_limits, values, upper_limits):
    assert len(lower_limits) == len(values)
    assert len(values) == len(upper_limits)
    return np.less_equal(lower_limits, values).all() and \
        np.less_equal(values, upper_limits).all()

def get_child_frame_pose(parent_bid, parent_link, child_bid, child_link):
    # TODO(mjlbach):Mostly shared with BRRobot, can be made a util

    # Different pos/orn calculations for base/links
    if child_link == -1:
        body_pos, body_orn = p.getBasePositionAndOrientation(child_bid)
    else:
        body_pos, body_orn = p.getLinkState(child_bid, child_link)[:2]

    # Get inverse world transform of body frame
    inv_body_pos, inv_body_orn = p.invertTransform(body_pos, body_orn)
    link_state = p.getLinkState(parent_bid, parent_link)
    link_pos = link_state[0]
    link_orn = link_state[1]
    # B * T = P -> T = (B-1)P, where B is body transform, T is target transform and P is palm transform
    child_frame_pos, child_frame_orn = p.multiplyTransforms(inv_body_pos, inv_body_orn, link_pos, link_orn)

    return child_frame_pos, child_frame_orn
#####################################

# Bodies


def get_bodies():
    return [p.getBodyUniqueId(i, physicsClientId=CLIENT)
            for i in range(p.getNumBodies(physicsClientId=CLIENT))]


BodyInfo = namedtuple('BodyInfo', ['base_name', 'body_name'])


def get_body_info(body):
    return BodyInfo(*p.getBodyInfo(body, physicsClientId=CLIENT))


def get_base_name(body):
    return get_body_info(body).base_name.decode(encoding='UTF-8')


def get_body_name(body):
    return get_body_info(body).body_name.decode(encoding='UTF-8')


def get_name(body):
    name = get_body_name(body)
    if name == '':
        name = 'body'
    return '{}{}'.format(name, int(body))


def has_body(name):
    try:
        body_from_name(name)
    except ValueError:
        return False
    return True


def body_from_name(name):
    for body in get_bodies():
        if get_body_name(body) == name:
            return body
    raise ValueError(name)


def remove_body(body):
    if (CLIENT, body) in INFO_FROM_BODY:
        del INFO_FROM_BODY[CLIENT, body]
    return p.removeBody(body, physicsClientId=CLIENT)


def get_pose(body):
    return p.getBasePositionAndOrientation(body, physicsClientId=CLIENT)
    # return np.concatenate([point, quat])


def get_point(body):
    return get_pose(body)[0]


def get_quat(body):
    return get_pose(body)[1]  # [x,y,z,w]


def get_euler(body):
    return euler_from_quat(get_quat(body))


def get_base_values(body):
    return base_values_from_pose(get_pose(body))


def set_pose(body, pose):
    (point, quat) = pose
    p.resetBasePositionAndOrientation(
        body, point, quat, physicsClientId=CLIENT)


def set_point(body, point):
    set_pose(body, (point, get_quat(body)))


def set_quat(body, quat):
    set_pose(body, (get_point(body), quat))


def set_euler(body, euler):
    set_quat(body, quat_from_euler(euler))


def pose_from_pose2d(pose2d):
    x, y, theta = pose2d
    return Pose(Point(x=x, y=y), Euler(yaw=theta))


def set_base_values(body, values):
    _, _, z = get_point(body)
    x, y, theta = values
    set_point(body, (x, y, z))
    set_quat(body, z_rotation(theta))


def set_base_values_with_z(body, values, z):
    x, y, theta = values
    set_point(body, (x, y, z))
    set_quat(body, z_rotation(theta))


def get_velocity(body):
    linear, angular = p.getBaseVelocity(body, physicsClientId=CLIENT)
    return linear, angular  # [x,y,z], [wx,wy,wz]


def set_velocity(body, linear=None, angular=None):
    if linear is not None:
        p.resetBaseVelocity(body, linearVelocity=linear,
                            physicsClientId=CLIENT)
    if angular is not None:
        p.resetBaseVelocity(body, angularVelocity=angular,
                            physicsClientId=CLIENT)


def is_rigid_body(body):
    for joint in get_joints(body):
        if is_movable(body, joint):
            return False
    return True


def is_fixed_base(body):
    return get_mass(body) == STATIC_MASS


def dump_body(body):
    print('Body id: {} | Name: {} | Rigid: {} | Fixed: {}'.format(
        body, get_body_name(body), is_rigid_body(body), is_fixed_base(body)))
    for joint in get_joints(body):
        if is_movable(body, joint):
            print('Joint id: {} | Name: {} | Type: {} | Circular: {} | Limits: {}'.format(
                joint, get_joint_name(
                    body, joint), JOINT_TYPES[get_joint_type(body, joint)],
                is_circular(body, joint), get_joint_limits(body, joint)))
    link = -1
    print('Link id: {} | Name: {} | Mass: {} | Collision: {} | Visual: {}'.format(
        link, get_base_name(body), get_mass(body),
        len(get_collision_data(body, link)), -1))  # len(get_visual_data(body, link))))
    for link in get_links(body):
        joint = parent_joint_from_link(link)
        joint_name = JOINT_TYPES[get_joint_type(body, joint)] if is_fixed(
            body, joint) else get_joint_name(body, joint)
        print('Link id: {} | Name: {} | Joint: {} | Parent: {} | Mass: {} | Collision: {} | Visual: {}'.format(
            link, get_link_name(body, link), joint_name,
            get_link_name(body, get_link_parent(
                body, link)), get_mass(body, link),
            len(get_collision_data(body, link)), -1))  # len(get_visual_data(body, link))))
        #print(get_joint_parent_frame(body, link))
        #print(map(get_data_geometry, get_visual_data(body, link)))
        #print(map(get_data_geometry, get_collision_data(body, link)))


def dump_world():
    for body in get_bodies():
        dump_body(body)
        print()

#####################################

# Joints


JOINT_TYPES = {
    p.JOINT_REVOLUTE: 'revolute',  # 0
    p.JOINT_PRISMATIC: 'prismatic',  # 1
    p.JOINT_SPHERICAL: 'spherical',  # 2
    p.JOINT_PLANAR: 'planar',  # 3
    p.JOINT_FIXED: 'fixed',  # 4
    p.JOINT_POINT2POINT: 'point2point',  # 5
    p.JOINT_GEAR: 'gear',  # 6
}


def get_num_joints(body):
    return p.getNumJoints(body, physicsClientId=CLIENT)


def get_joints(body):
    return list(range(get_num_joints(body)))


def get_joint(body, joint_or_name):
    if type(joint_or_name) is str:
        return joint_from_name(body, joint_or_name)
    return joint_or_name


JointInfo = namedtuple('JointInfo', ['jointIndex', 'jointName', 'jointType',
                                     'qIndex', 'uIndex', 'flags',
                                     'jointDamping', 'jointFriction', 'jointLowerLimit', 'jointUpperLimit',
                                     'jointMaxForce', 'jointMaxVelocity', 'linkName', 'jointAxis',
                                     'parentFramePos', 'parentFrameOrn', 'parentIndex'])


def get_joint_info(body, joint):
    return JointInfo(*p.getJointInfo(body, joint, physicsClientId=CLIENT))


def get_joint_name(body, joint):
    return get_joint_info(body, joint).jointName  # .decode('UTF-8')


def get_joint_names(body, joints):
    return [get_joint_name(body, joint) for joint in joints]


def joint_from_name(body, name):
    for joint in get_joints(body):
        joint_name = get_joint_name(body, joint)
        if type(joint_name) == bytes:
            joint_name = joint_name.decode()
        if joint_name == name:
            return joint
    raise ValueError(body, name)


def has_joint(body, name):
    try:
        joint_from_name(body, name)
    except ValueError:
        return False
    return True


def joints_from_names(body, names):
    return tuple(joint_from_name(body, name) for name in names)


JointState = namedtuple('JointState', ['jointPosition', 'jointVelocity',
                                       'jointReactionForces', 'appliedJointMotorTorque'])


def get_joint_state(body, joint):
    return JointState(*p.getJointState(body, joint, physicsClientId=CLIENT))


def get_joint_position(body, joint):
    return get_joint_state(body, joint).jointPosition


def get_joint_velocity(body, joint):
    return get_joint_state(body, joint).jointVelocity


def get_joint_reaction_force(body, joint):
    return get_joint_state(body, joint).jointReactionForces


def get_joint_torque(body, joint):
    return get_joint_state(body, joint).appliedJointMotorTorque


def get_joint_positions(body, joints):  # joints=None):
    return tuple(get_joint_position(body, joint) for joint in joints)


def get_joint_velocities(body, joints):
    return tuple(get_joint_velocity(body, joint) for joint in joints)


def set_joint_position(body, joint, value):
    p.resetJointState(body, joint, value, targetVelocity=0,
                      physicsClientId=CLIENT)


def set_joint_positions(body, joints, values):
    assert len(joints) == len(values)
    for joint, value in zip(joints, values):
        set_joint_position(body, joint, value)


def get_configuration(body):
    return get_joint_positions(body, get_movable_joints(body))


def set_configuration(body, values):
    set_joint_positions(body, get_movable_joints(body), values)


def get_full_configuration(body):
    # Cannot alter fixed joints
    return get_joint_positions(body, get_joints(body))


def get_labeled_configuration(body):
    movable_joints = get_movable_joints(body)
    return dict(zip(get_joint_names(body, movable_joints),
                    get_joint_positions(body, movable_joints)))


def get_joint_type(body, joint):
    return get_joint_info(body, joint).jointType


def is_fixed(body, joint):
    return get_joint_type(body, joint) == p.JOINT_FIXED


def is_movable(body, joint):
    return not is_fixed(body, joint)


def prune_fixed_joints(body, joints):
    return [joint for joint in joints if is_movable(body, joint)]


def get_movable_joints(body):  # 45 / 87 on pr2
    return prune_fixed_joints(body, get_joints(body))


def joint_from_movable(body, index):
    return get_joints(body)[index]


def movable_from_joints(body, joints):
    movable_from_original = {o: m for m,
                             o in enumerate(get_movable_joints(body))}
    return [movable_from_original[joint] for joint in joints]


def is_circular(body, joint):
    joint_info = get_joint_info(body, joint)
    if joint_info.jointType == p.JOINT_FIXED:
        return False
    return joint_info.jointUpperLimit < joint_info.jointLowerLimit


def get_joint_limits(body, joint):
    # TODO: make a version for several joints?
    if is_circular(body, joint):
        # TODO: return UNBOUNDED_LIMITS
        return CIRCULAR_LIMITS
    joint_info = get_joint_info(body, joint)
    return joint_info.jointLowerLimit, joint_info.jointUpperLimit


def get_min_limit(body, joint):
    # TODO: rename to min_position
    return get_joint_limits(body, joint)[0]


def get_min_limits(body, joints):
    return [get_min_limit(body, joint) for joint in joints]


def get_max_limit(body, joint):
    return get_joint_limits(body, joint)[1]


def get_max_limits(body, joints):
    return [get_max_limit(body, joint) for joint in joints]


def get_max_velocity(body, joint):
    return get_joint_info(body, joint).jointMaxVelocity


def get_max_force(body, joint):
    return get_joint_info(body, joint).jointMaxForce


def get_joint_q_index(body, joint):
    return get_joint_info(body, joint).qIndex


def get_joint_v_index(body, joint):
    return get_joint_info(body, joint).uIndex


def get_joint_axis(body, joint):
    return get_joint_info(body, joint).jointAxis


def get_joint_parent_frame(body, joint):
    joint_info = get_joint_info(body, joint)
    return joint_info.parentFramePos, joint_info.parentFrameOrn


def violates_limit(body, joint, value):
    if is_circular(body, joint):
        return False
    lower, upper = get_joint_limits(body, joint)
    return (value < lower) or (upper < value)


def violates_limits(body, joints, values):
    return any(violates_limit(body, joint, value) for joint, value in zip(joints, values))


def wrap_position(body, joint, position):
    if is_circular(body, joint):
        return wrap_angle(position)
    return position


def wrap_positions(body, joints, positions):
    assert len(joints) == len(positions)
    return [wrap_position(body, joint, position)
            for joint, position in zip(joints, positions)]


def get_custom_limits(body, joints, custom_limits={}, circular_limits=UNBOUNDED_LIMITS):
    joint_limits = []
    for joint in joints:
        if joint in custom_limits:
            joint_limits.append(custom_limits[joint])
        elif is_circular(body, joint):
            joint_limits.append(circular_limits)
        else:
            joint_limits.append(get_joint_limits(body, joint))
    return zip(*joint_limits)

#####################################

# Links


BASE_LINK = -1
STATIC_MASS = 0

get_num_links = get_num_joints
get_links = get_joints  # Does not include BASE_LINK


def child_link_from_joint(joint):
    return joint  # link


def parent_joint_from_link(link):
    return link  # joint


def get_all_links(body):
    return [BASE_LINK] + list(get_links(body))


def get_link_name(body, link):
    if link == BASE_LINK:
        return get_base_name(body)
    return get_joint_info(body, link).linkName.decode('UTF-8')


def get_link_parent(body, link):
    if link == BASE_LINK:
        return None
    return get_joint_info(body, link).parentIndex


parent_link_from_joint = get_link_parent


def link_from_name(body, name):
    if name == get_base_name(body):
        return BASE_LINK
    for link in get_joints(body):
        if get_link_name(body, link) == name:
            return link
    raise ValueError("Could not find link %s for body %d" % (name, body))


def has_link(body, name):
    try:
        link_from_name(body, name)
    except ValueError:
        return False
    return True


def get_link_position_from_name(body_id, name):
    try:
        link_id = link_from_name(body_id, name)
    except ValueError:
        return None

    return get_link_state(body_id, link_id).linkWorldPosition


LinkState = namedtuple('LinkState', ['linkWorldPosition', 'linkWorldOrientation',
                                     'localInertialFramePosition', 'localInertialFrameOrientation',
                                     'worldLinkFramePosition', 'worldLinkFrameOrientation'])


def get_link_state(body, link, kinematics=True, velocity=True):
    # TODO: the defaults are set to False?
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/pybullet.c
    return LinkState(*p.getLinkState(body, link,  # computeLinkVelocity=velocity, computeForwardKinematics=kinematics,
                                     physicsClientId=CLIENT))


def get_com_pose(body, link):  # COM = center of mass
    link_state = get_link_state(body, link)
    return link_state.linkWorldPosition, link_state.linkWorldOrientation


def get_link_inertial_pose(body, link):
    link_state = get_link_state(body, link)
    return link_state.localInertialFramePosition, link_state.localInertialFrameOrientation


def get_link_pose(body, link):
    if link == BASE_LINK:
        return get_pose(body)
    # if set to 1 (or True), the Cartesian world position/orientation will be recomputed using forward kinematics.
    # , kinematics=True, velocity=False)
    link_state = get_link_state(body, link)
    return link_state.worldLinkFramePosition, link_state.worldLinkFrameOrientation


def get_relative_pose(body, link1, link2=BASE_LINK):
    world_from_link1 = get_link_pose(body, link1)
    world_from_link2 = get_link_pose(body, link2)
    link2_from_link1 = multiply(invert(world_from_link2), world_from_link1)
    return link2_from_link1

#####################################


def get_all_link_parents(body):
    return {link: get_link_parent(body, link) for link in get_links(body)}


def get_all_link_children(body):
    children = {}
    for child, parent in get_all_link_parents(body).items():
        if parent not in children:
            children[parent] = []
        children[parent].append(child)
    return children


def get_link_children(body, link):
    children = get_all_link_children(body)
    return children.get(link, [])


def get_link_ancestors(body, link):
    # Returns in order of depth
    # Does not include link
    parent = get_link_parent(body, link)
    if parent is None:
        return []
    return get_link_ancestors(body, parent) + [parent]


def get_joint_ancestors(body, joint):
    link = child_link_from_joint(joint)
    return get_link_ancestors(body, link) + [link]


def get_movable_joint_ancestors(body, link):
    return prune_fixed_joints(body, get_joint_ancestors(body, link))


def get_joint_descendants(body, link):
    return list(map(parent_joint_from_link, get_link_descendants(body, link)))


def get_movable_joint_descendants(body, link):
    return prune_fixed_joints(body, get_joint_descendants(body, link))


def get_link_descendants(body, link, test=lambda l: True):
    descendants = []
    for child in get_link_children(body, link):
        if test(child):
            descendants.append(child)
            descendants.extend(get_link_descendants(body, child, test=test))
    return descendants


def get_link_subtree(body, link, **kwargs):
    return [link] + get_link_descendants(body, link, **kwargs)


def are_links_adjacent(body, link1, link2):
    return (get_link_parent(body, link1) == link2) or \
           (get_link_parent(body, link2) == link1)


def get_adjacent_links(body):
    adjacent = set()
    for link in get_links(body):
        parent = get_link_parent(body, link)
        adjacent.add((link, parent))
        #adjacent.add((parent, link))
    return adjacent


def get_adjacent_fixed_links(body):
    return list(filter(lambda item: not is_movable(body, item[0]),
                       get_adjacent_links(body)))


def get_fixed_links(body):
    edges = defaultdict(list)
    for link, parent in get_adjacent_fixed_links(body):
        edges[link].append(parent)
        edges[parent].append(link)
    visited = set()
    fixed = set()
    for initial_link in get_links(body):
        if initial_link in visited:
            continue
        cluster = [initial_link]
        queue = deque([initial_link])
        visited.add(initial_link)
        while queue:
            for next_link in edges[queue.popleft()]:
                if next_link not in visited:
                    cluster.append(next_link)
                    queue.append(next_link)
                    visited.add(next_link)
        fixed.update(product(cluster, cluster))
    return fixed

#####################################


DynamicsInfo = namedtuple('DynamicsInfo', ['mass', 'lateral_friction',
                                           'local_inertia_diagonal', 'local_inertial_pos',  'local_inertial_orn',
                                           'restitution', 'rolling_friction', 'spinning_friction',
                                           'contact_damping', 'contact_stiffness'])


def get_dynamics_info(body, link=BASE_LINK):
    return DynamicsInfo(*p.getDynamicsInfo(body, link, physicsClientId=CLIENT))


get_link_info = get_dynamics_info


def get_mass(body, link=BASE_LINK):
    # TOOD: get full mass
    return get_dynamics_info(body, link).mass


def set_dynamics(body, link=BASE_LINK, **kwargs):
    # TODO: iterate over all links
    p.changeDynamics(body, link, physicsClientId=CLIENT, **kwargs)


def set_mass(body, mass, link=BASE_LINK):
    set_dynamics(body, link=link, mass=mass)


def set_static(body):
    for link in get_all_links(body):
        set_mass(body, mass=STATIC_MASS, link=link)


def set_all_static():
    # TODO: mass saver
    disable_gravity()
    for body in get_bodies():
        set_static(body)


def get_joint_inertial_pose(body, joint):
    dynamics_info = get_dynamics_info(body, joint)
    return dynamics_info.local_inertial_pos, dynamics_info.local_inertial_orn


def get_local_link_pose(body, joint):
    parent_joint = parent_link_from_joint(body, joint)

    #world_child = get_link_pose(body, joint)
    #world_parent = get_link_pose(body, parent_joint)
    # return multiply(invert(world_parent), world_child)
    # return multiply(world_child, invert(world_parent))

    # https://github.com/bulletphysics/bullet3/blob/9c9ac6cba8118544808889664326fd6f06d9eeba/examples/pybullet/gym/pybullet_utils/urdfEditor.py#L169
    parent_com = get_joint_parent_frame(body, joint)
    tmp_pose = invert(
        multiply(get_joint_inertial_pose(body, joint), parent_com))
    parent_inertia = get_joint_inertial_pose(body, parent_joint)
    # return multiply(parent_inertia, tmp_pose) # TODO: why is this wrong...
    _, orn = multiply(parent_inertia, tmp_pose)
    pos, _ = multiply(parent_inertia, Pose(parent_com[0]))
    return (pos, orn)

#####################################

# Shapes


SHAPE_TYPES = {
    p.GEOM_SPHERE: 'sphere',  # 2
    p.GEOM_BOX: 'box',  # 3
    p.GEOM_CYLINDER: 'cylinder',  # 4
    p.GEOM_MESH: 'mesh',  # 5
    p.GEOM_PLANE: 'plane',  # 6
    p.GEOM_CAPSULE: 'capsule',  # 7
    # p.GEOM_FORCE_CONCAVE_TRIMESH
}

# TODO: clean this up to avoid repeated work


def get_box_geometry(width, length, height):
    return {
        'shapeType': p.GEOM_BOX,
        'halfExtents': [width/2., length/2., height/2.]
    }


def get_cylinder_geometry(radius, height):
    return {
        'shapeType': p.GEOM_CYLINDER,
        'radius': radius,
        'length': height,
    }


def get_sphere_geometry(radius):
    return {
        'shapeType': p.GEOM_SPHERE,
        'radius': radius,
    }


def get_capsule_geometry(radius, height):
    return {
        'shapeType': p.GEOM_CAPSULE,
        'radius': radius,
        'length': height,
    }


def get_plane_geometry(normal):
    return {
        'shapeType': p.GEOM_PLANE,
        'planeNormal': normal,
    }


def get_mesh_geometry(path, scale=1.):
    return {
        'shapeType': p.GEOM_MESH,
        'fileName': path,
        'meshScale': scale*np.ones(3),
    }


NULL_ID = -1


def create_collision_shape(geometry, pose=unit_pose()):
    point, quat = pose
    collision_args = {
        'collisionFramePosition': point,
        'collisionFrameOrientation': quat,
        'physicsClientId': CLIENT,
    }
    collision_args.update(geometry)
    if 'length' in collision_args:
        # TODO: pybullet bug visual => length, collision => height
        collision_args['height'] = collision_args['length']
        del collision_args['length']
    return p.createCollisionShape(**collision_args)


def create_visual_shape(geometry, pose=unit_pose(), color=(1, 0, 0, 1), specular=None):
    if (color is None):  # or not has_gui():
        return NULL_ID
    point, quat = pose
    visual_args = {
        'rgbaColor': color,
        'visualFramePosition': point,
        'visualFrameOrientation': quat,
        'physicsClientId': CLIENT,
    }
    visual_args.update(geometry)
    if specular is not None:
        visual_args['specularColor'] = specular
    return p.createVisualShape(**visual_args)


def create_shape(geometry, pose=unit_pose(), collision=True, **kwargs):
    collision_id = create_collision_shape(
        geometry, pose=pose) if collision else NULL_ID
    visual_id = create_visual_shape(geometry, pose=pose, **kwargs)
    return collision_id, visual_id


def plural(word):
    exceptions = {'radius': 'radii'}
    if word in exceptions:
        return exceptions[word]
    if word.endswith('s'):
        return word
    return word + 's'


def create_shape_array(geoms, poses, colors=None):
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/pybullet.c
    # createCollisionShape: height
    # createVisualShape: length
    # createCollisionShapeArray: lengths
    # createVisualShapeArray: lengths
    mega_geom = defaultdict(list)
    for geom in geoms:
        extended_geom = get_default_geometry()
        extended_geom.update(geom)
        #extended_geom = geom.copy()
        for key, value in extended_geom.items():
            mega_geom[plural(key)].append(value)

    collision_args = mega_geom.copy()
    for (point, quat) in poses:
        collision_args['collisionFramePositions'].append(point)
        collision_args['collisionFrameOrientations'].append(quat)
    collision_id = p.createCollisionShapeArray(
        physicsClientId=CLIENT, **collision_args)
    if (colors is None):  # or not has_gui():
        return collision_id, NULL_ID

    visual_args = mega_geom.copy()
    for (point, quat), color in zip(poses, colors):
        # TODO: color doesn't seem to work correctly here
        visual_args['rgbaColors'].append(color)
        visual_args['visualFramePositions'].append(point)
        visual_args['visualFrameOrientations'].append(quat)
    visual_id = p.createVisualShapeArray(physicsClientId=CLIENT, **visual_args)
    return collision_id, visual_id

#####################################


def create_body(collision_id=-1, visual_id=-1, mass=STATIC_MASS):
    return p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_id,
                             baseVisualShapeIndex=visual_id, physicsClientId=CLIENT)


def create_box(w, l, h, mass=STATIC_MASS, color=(1, 0, 0, 1)):
    collision_id, visual_id = create_shape(
        get_box_geometry(w, l, h), color=color)
    return create_body(collision_id, visual_id, mass=mass)
    # basePosition | baseOrientation
    # linkCollisionShapeIndices | linkVisualShapeIndices


def create_cylinder(radius, height, mass=STATIC_MASS, color=(0, 0, 1, 1)):
    collision_id, visual_id = create_shape(
        get_cylinder_geometry(radius, height), color=color)
    return create_body(collision_id, visual_id, mass=mass)


def create_capsule(radius, height, mass=STATIC_MASS, color=(0, 0, 1, 1)):
    collision_id, visual_id = create_shape(
        get_capsule_geometry(radius, height), color=color)
    return create_body(collision_id, visual_id, mass=mass)


def create_sphere(radius, mass=STATIC_MASS, color=(0, 0, 1, 1)):
    collision_id, visual_id = create_shape(
        get_sphere_geometry(radius), color=color)
    return create_body(collision_id, visual_id, mass=mass)


def create_plane(normal=[0, 0, 1], mass=STATIC_MASS, color=(0, 0, 0, 1)):
    # color seems to be ignored in favor of a texture
    collision_id, visual_id = create_shape(
        get_plane_geometry(normal), color=color)
    return create_body(collision_id, visual_id, mass=mass)


def create_obj(path, scale=1., mass=STATIC_MASS, collision=True, color=(0.5, 0.5, 0.5, 1)):
    collision_id, visual_id = create_shape(get_mesh_geometry(
        path, scale=scale), collision=collision, color=color)
    body = create_body(collision_id, visual_id, mass=mass)
    fixed_base = (mass == STATIC_MASS)
    INFO_FROM_BODY[CLIENT, body] = ModelInfo(
        None, path, fixed_base, scale)  # TODO: store geometry info instead?
    return body


Mesh = namedtuple('Mesh', ['vertices', 'faces'])
mesh_count = count()
TEMP_DIR = 'temp/'


def create_mesh(mesh, under=True, **kwargs):
    # http://people.sc.fsu.edu/~jburkardt/data/obj/obj.html
    # TODO: read OFF / WRL / OBJ files
    # TODO: maintain dict to file
    ensure_dir(TEMP_DIR)
    path = os.path.join(TEMP_DIR, 'mesh{}.obj'.format(next(mesh_count)))
    write(path, obj_file_from_mesh(mesh, under=under))
    return create_obj(path, **kwargs)
    # safe_remove(path) # TODO: removing might delete mesh?

#####################################


VisualShapeData = namedtuple('VisualShapeData', ['objectUniqueId', 'linkIndex',
                                                 'visualGeometryType', 'dimensions', 'meshAssetFileName',
                                                 'localVisualFrame_position', 'localVisualFrame_orientation',
                                                 'rgbaColor'])  # 'textureUniqueId'

UNKNOWN_FILE = 'unknown_file'


def visual_shape_from_data(data, client=None):
    client = get_client(client)
    if (data.visualGeometryType == p.GEOM_MESH) and (data.meshAssetFileName == UNKNOWN_FILE):
        return -1
    # visualFramePosition: translational offset of the visual shape with respect to the link
    # visualFrameOrientation: rotational offset (quaternion x,y,z,w) of the visual shape with respect to the link frame
    #inertial_pose = get_joint_inertial_pose(data.objectUniqueId, data.linkIndex)
    #point, quat = multiply(invert(inertial_pose), pose)
    point, quat = get_data_pose(data)
    return p.createVisualShape(shapeType=data.visualGeometryType,
                               radius=get_data_radius(data),
                               halfExtents=np.array(get_data_extents(data))/2,
                               # TODO: pybullet bug
                               length=get_data_height(data),
                               fileName=data.meshAssetFileName,
                               meshScale=get_data_scale(data),
                               planeNormal=get_data_normal(data),
                               rgbaColor=data.rgbaColor,
                               # specularColor=,
                               visualFramePosition=point,
                               visualFrameOrientation=quat,
                               physicsClientId=client)


def get_visual_data(body, link=BASE_LINK):
    visual_data = [VisualShapeData(
        *tup) for tup in p.getVisualShapeData(body, physicsClientId=CLIENT)]
    return list(filter(lambda d: d.linkIndex == link, visual_data))


# object_unique_id and linkIndex seem to be noise
CollisionShapeData = namedtuple('CollisionShapeData', ['object_unique_id', 'linkIndex',
                                                       'geometry_type', 'dimensions', 'filename',
                                                       'local_frame_pos', 'local_frame_orn'])


def collision_shape_from_data(data, body, link, client=None):
    client = get_client(client)
    if (data.geometry_type == p.GEOM_MESH) and (data.filename == UNKNOWN_FILE):
        return -1
    pose = multiply(get_joint_inertial_pose(body, link), get_data_pose(data))
    point, quat = pose
    # TODO: the visual data seems affected by the collision data
    return p.createCollisionShape(shapeType=data.geometry_type,
                                  radius=get_data_radius(data),
                                  # halfExtents=get_data_extents(data.geometry_type, data.dimensions),
                                  halfExtents=np.array(
                                      get_data_extents(data)) / 2,
                                  height=get_data_height(data),
                                  fileName=data.filename.decode(
                                      encoding='UTF-8'),
                                  meshScale=get_data_scale(data),
                                  planeNormal=get_data_normal(data),
                                  flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
                                  collisionFramePosition=point,
                                  collisionFrameOrientation=quat,
                                  physicsClientId=client)
    # return p.createCollisionShapeArray()


def clone_visual_shape(body, link, client=None):
    client = get_client(client)
    # if not has_gui(client):
    #    return -1
    visual_data = get_visual_data(body, link)
    if not visual_data:
        return -1
    assert (len(visual_data) == 1)
    return visual_shape_from_data(visual_data[0], client)


def clone_collision_shape(body, link, client=None):
    client = get_client(client)
    collision_data = get_collision_data(body, link)
    if not collision_data:
        return -1
    assert (len(collision_data) == 1)
    # TODO: can do CollisionArray
    return collision_shape_from_data(collision_data[0], body, link, client)


def clone_body(body, links=None, collision=True, visual=True, client=None):
    # TODO: names are not retained
    # TODO: error with createMultiBody link poses on PR2
    # localVisualFrame_position: position of local visual frame, relative to link/joint frame
    # localVisualFrame orientation: orientation of local visual frame relative to link/joint frame
    # parentFramePos: joint position in parent frame
    # parentFrameOrn: joint orientation in parent frame
    client = get_client(client)  # client is the new client for the body
    if links is None:
        links = get_links(body)
    #movable_joints = [joint for joint in links if is_movable(body, joint)]
    new_from_original = {}
    base_link = get_link_parent(body, links[0]) if links else BASE_LINK
    new_from_original[base_link] = -1

    masses = []
    collision_shapes = []
    visual_shapes = []
    positions = []  # list of local link positions, with respect to parent
    orientations = []  # list of local link orientations, w.r.t. parent
    inertial_positions = []  # list of local inertial frame pos. in link frame
    inertial_orientations = []  # list of local inertial frame orn. in link frame
    parent_indices = []
    joint_types = []
    joint_axes = []
    for i, link in enumerate(links):
        new_from_original[link] = i
        joint_info = get_joint_info(body, link)
        dynamics_info = get_dynamics_info(body, link)
        masses.append(dynamics_info.mass)
        collision_shapes.append(clone_collision_shape(
            body, link, client) if collision else -1)
        visual_shapes.append(clone_visual_shape(
            body, link, client) if visual else -1)
        point, quat = get_local_link_pose(body, link)
        positions.append(point)
        orientations.append(quat)
        inertial_positions.append(dynamics_info.local_inertial_pos)
        inertial_orientations.append(dynamics_info.local_inertial_orn)
        # TODO: need the increment to work
        parent_indices.append(new_from_original[joint_info.parentIndex] + 1)
        joint_types.append(joint_info.jointType)
        joint_axes.append(joint_info.jointAxis)
    # https://github.com/bulletphysics/bullet3/blob/9c9ac6cba8118544808889664326fd6f06d9eeba/examples/pybullet/gym/pybullet_utils/urdfEditor.py#L169

    base_dynamics_info = get_dynamics_info(body, base_link)
    base_point, base_quat = get_link_pose(body, base_link)
    new_body = p.createMultiBody(baseMass=base_dynamics_info.mass,
                                 baseCollisionShapeIndex=clone_collision_shape(
                                     body, base_link, client) if collision else -1,
                                 baseVisualShapeIndex=clone_visual_shape(
                                     body, base_link, client) if visual else -1,
                                 basePosition=base_point,
                                 baseOrientation=base_quat,
                                 baseInertialFramePosition=base_dynamics_info.local_inertial_pos,
                                 baseInertialFrameOrientation=base_dynamics_info.local_inertial_orn,
                                 linkMasses=masses,
                                 linkCollisionShapeIndices=collision_shapes,
                                 linkVisualShapeIndices=visual_shapes,
                                 linkPositions=positions,
                                 linkOrientations=orientations,
                                 linkInertialFramePositions=inertial_positions,
                                 linkInertialFrameOrientations=inertial_orientations,
                                 linkParentIndices=parent_indices,
                                 linkJointTypes=joint_types,
                                 linkJointAxis=joint_axes,
                                 physicsClientId=client)
    # set_configuration(new_body, get_joint_positions(body, movable_joints)) # Need to use correct client
    for joint, value in zip(range(len(links)), get_joint_positions(body, links)):
        # TODO: check if movable?
        p.resetJointState(new_body, joint, value,
                          targetVelocity=0, physicsClientId=client)
    return new_body


def clone_world(client=None, exclude=[]):
    visual = has_gui(client)
    mapping = {}
    for body in get_bodies():
        if body not in exclude:
            new_body = clone_body(body, collision=True,
                                  visual=visual, client=client)
            mapping[body] = new_body
    return mapping

#####################################


def get_collision_data(body, link=BASE_LINK):
    # TODO: try catch
    return [CollisionShapeData(*tup) for tup in p.getCollisionShapeData(body, link, physicsClientId=CLIENT)]


def get_data_type(data):
    return data.geometry_type if isinstance(data, CollisionShapeData) else data.visualGeometryType


def get_data_filename(data):
    return (data.filename if isinstance(data, CollisionShapeData)
            else data.meshAssetFileName).decode(encoding='UTF-8')


def get_data_pose(data):
    if isinstance(data, CollisionShapeData):
        return (data.local_frame_pos, data.local_frame_orn)
    return (data.localVisualFrame_position, data.localVisualFrame_orientation)


def get_default_geometry():
    return {
        'halfExtents': DEFAULT_EXTENTS,
        'radius': DEFAULT_RADIUS,
        'length': DEFAULT_HEIGHT,  # 'height'
        'fileName': DEFAULT_MESH,
        'meshScale': DEFAULT_SCALE,
        'planeNormal': DEFAULT_NORMAL,
    }


DEFAULT_MESH = ''

DEFAULT_EXTENTS = [1, 1, 1]


def get_data_extents(data):
    """
    depends on geometry type:
    for GEOM_BOX: extents,
    for GEOM_SPHERE dimensions[0] = radius,
    for GEOM_CAPSULE and GEOM_CYLINDER, dimensions[0] = height (length), dimensions[1] = radius.
    For GEOM_MESH, dimensions is the scaling factor.
    :return:
    """
    geometry_type = get_data_type(data)
    dimensions = data.dimensions
    if geometry_type == p.GEOM_BOX:
        return dimensions
    return DEFAULT_EXTENTS


DEFAULT_RADIUS = 0.5


def get_data_radius(data):
    geometry_type = get_data_type(data)
    dimensions = data.dimensions
    if geometry_type == p.GEOM_SPHERE:
        return dimensions[0]
    if geometry_type in (p.GEOM_CYLINDER, p.GEOM_CAPSULE):
        return dimensions[1]
    return DEFAULT_RADIUS


DEFAULT_HEIGHT = 1


def get_data_height(data):
    geometry_type = get_data_type(data)
    dimensions = data.dimensions
    if geometry_type in (p.GEOM_CYLINDER, p.GEOM_CAPSULE):
        return dimensions[0]
    return DEFAULT_HEIGHT


DEFAULT_SCALE = [1, 1, 1]


def get_data_scale(data):
    geometry_type = get_data_type(data)
    dimensions = data.dimensions
    if geometry_type == p.GEOM_MESH:
        return dimensions
    return DEFAULT_SCALE


DEFAULT_NORMAL = [0, 0, 1]


def get_data_normal(data):
    geometry_type = get_data_type(data)
    dimensions = data.dimensions
    if geometry_type == p.GEOM_PLANE:
        return dimensions
    return DEFAULT_NORMAL


def get_data_geometry(data):
    geometry_type = get_data_type(data)
    if geometry_type == p.GEOM_SPHERE:
        parameters = [get_data_radius(data)]
    elif geometry_type == p.GEOM_BOX:
        parameters = [get_data_extents(data)]
    elif geometry_type in (p.GEOM_CYLINDER, p.GEOM_CAPSULE):
        parameters = [get_data_height(data), get_data_radius(data)]
    elif geometry_type == p.GEOM_MESH:
        parameters = [get_data_filename(data), get_data_scale(data)]
    elif geometry_type == p.GEOM_PLANE:
        parameters = [get_data_extents(data)]
    else:
        raise ValueError(geometry_type)
    return SHAPE_TYPES[geometry_type], parameters


def set_color(body, color, link=BASE_LINK, shape_index=-1):
    """
    Experimental for internal use, recommended ignore shapeIndex or leave it -1.
    Intention was to let you pick a specific shape index to modify,
    since URDF (and SDF etc) can have more than 1 visual shape per link.
    This shapeIndex matches the list ordering returned by getVisualShapeData.
    :param body:
    :param link:
    :param shape_index:
    :return:
    """
    # specularColor
    return p.changeVisualShape(body, link, shapeIndex=shape_index, rgbaColor=color,
                               # textureUniqueId=None, specularColor=None,
                               physicsClientId=CLIENT)

#####################################

# Bounding box


AABB = namedtuple('AABB', ['lower', 'upper'])


def aabb_from_points(points):
    return AABB(np.min(points, axis=0), np.max(points, axis=0))


def aabb_union(aabbs):
    return aabb_from_points(np.vstack([aabb for aabb in aabbs]))


def aabb_overlap(aabb1, aabb2):
    lower1, upper1 = aabb1
    lower2, upper2 = aabb2
    return np.less_equal(lower1, upper2).all() and \
        np.less_equal(lower2, upper1).all()


def get_subtree_aabb(body, root_link=BASE_LINK):
    return aabb_union(get_aabb(body, link) for link in get_link_subtree(body, root_link))


def get_aabbs(body):
    return [get_aabb(body, link=link) for link in get_all_links(body)]


def get_aabb(body, link=None):
    # Note that the query is conservative and may return additional objects that don't have actual AABB overlap.
    # This happens because the acceleration structures have some heuristic that enlarges the AABBs a bit
    # (extra margin and extruded along the velocity vector).
    # Contact points with distance exceeding this threshold are not processed by the LCP solver.
    # AABBs are extended by this number. Defaults to 0.02 in Bullet 2.x
    #p.setPhysicsEngineParameter(contactBreakingThreshold=0.0, physicsClientId=CLIENT)
    if link is None:
        aabb = aabb_union(get_aabbs(body))
    else:
        aabb = p.getAABB(body, linkIndex=link, physicsClientId=CLIENT)
    return aabb


get_lower_upper = get_aabb


def get_aabb_center(aabb):
    lower, upper = aabb
    return (np.array(lower) + np.array(upper)) / 2.


def get_aabb_extent(aabb):
    lower, upper = aabb
    return np.array(upper) - np.array(lower)


def get_center_extent(body, **kwargs):
    aabb = get_aabb(body, **kwargs)
    return get_aabb_center(aabb), get_aabb_extent(aabb)


def aabb2d_from_aabb(aabb):
    (lower, upper) = aabb
    return lower[:2], upper[:2]


def aabb_contains_aabb(contained, container):
    lower1, upper1 = contained
    lower2, upper2 = container
    return np.less_equal(lower2, lower1).all() and \
        np.less_equal(upper1, upper2).all()
    # return np.all(lower2 <= lower1) and np.all(upper1 <= upper2)


def aabb_contains_point(point, container):
    lower, upper = container
    return np.less_equal(lower, point).all() and \
        np.less_equal(point, upper).all()
    # return np.all(lower <= point) and np.all(point <= upper)


def get_bodies_in_region(aabb):
    (lower, upper) = aabb
    return p.getOverlappingObjects(lower, upper, physicsClientId=CLIENT)


def get_aabb_volume(aabb):
    return np.prod(get_aabb_extent(aabb))


def get_aabb_area(aabb):
    return np.prod(get_aabb_extent(aabb2d_from_aabb(aabb)))

#####################################

# AABB approximation


def vertices_from_data(data):
    geometry_type = get_data_type(data)
    # if geometry_type == p.GEOM_SPHERE:
    #    parameters = [get_data_radius(data)]
    if geometry_type == p.GEOM_BOX:
        extents = np.array(get_data_extents(data))
        aabb = AABB(-extents/2., +extents/2.)
        vertices = get_aabb_vertices(aabb)
    elif geometry_type in (p.GEOM_CYLINDER, p.GEOM_CAPSULE):
        # TODO: p.URDF_USE_IMPLICIT_CYLINDER
        radius, height = get_data_radius(data), get_data_height(data)
        extents = np.array([2*radius, 2*radius, height])
        aabb = AABB(-extents/2., +extents/2.)
        vertices = get_aabb_vertices(aabb)
    elif geometry_type == p.GEOM_SPHERE:
        radius = get_data_radius(data)
        half_extents = radius*np.ones(3)
        aabb = AABB(-half_extents, +half_extents)
        vertices = get_aabb_vertices(aabb)
    elif geometry_type == p.GEOM_MESH:
        filename, scale = get_data_filename(data), get_data_scale(data)
        if filename == UNKNOWN_FILE:
            raise RuntimeError(filename)
        mesh = read_obj(filename, decompose=False)
        vertices = [scale*np.array(vertex) for vertex in mesh.vertices]
        # TODO: could compute AABB here for improved speed at the cost of being conservative
    # elif geometry_type == p.GEOM_PLANE:
    #   parameters = [get_data_extents(data)]
    else:
        raise NotImplementedError(geometry_type)
    return apply_affine(get_data_pose(data), vertices)


def vertices_from_link(body, link):
    # In local frame
    vertices = []
    # TODO: requires the viewer to be active
    # for data in get_visual_data(body, link):
    #    vertices.extend(vertices_from_data(data))
    # Pybullet creates multiple collision elements (with unknown_file) when noncovex
    for data in get_collision_data(body, link):
        vertices.extend(vertices_from_data(data))
    return vertices


OBJ_MESH_CACHE = {}


def vertices_from_rigid(body, link=BASE_LINK):
    assert implies(link == BASE_LINK, get_num_links(body) == 0)
    try:
        vertices = vertices_from_link(body, link)
    except RuntimeError:
        info = get_model_info(body)
        assert info is not None
        _, ext = os.path.splitext(info.path)
        if ext == '.obj':
            if info.path not in OBJ_MESH_CACHE:
                OBJ_MESH_CACHE[info.path] = read_obj(
                    info.path, decompose=False)
            mesh = OBJ_MESH_CACHE[info.path]
            vertices = mesh.vertices
        else:
            raise NotImplementedError(ext)
    return vertices


def approximate_as_prism(body, body_pose=unit_pose(), **kwargs):
    # TODO: make it just orientation
    vertices = apply_affine(body_pose, vertices_from_rigid(body, **kwargs))
    aabb = aabb_from_points(vertices)
    return get_aabb_center(aabb), get_aabb_extent(aabb)
    # with PoseSaver(body):
    #    set_pose(body, body_pose)
    #    set_velocity(body, linear=np.zeros(3), angular=np.zeros(3))
    #    return get_center_extent(body, **kwargs)


def approximate_as_cylinder(body, **kwargs):
    center, (width, length, height) = approximate_as_prism(body, **kwargs)
    diameter = (width + length) / 2  # TODO: check that these are close
    return center, (diameter, height)

#####################################

# Collision


#MAX_DISTANCE = 1e-3
MAX_DISTANCE = 0.


def set_all_collisions(body_id, enabled=1):
    body_link_idxs = [-1] + [i for i in range(p.getNumJoints(body_id))]

    for col_id in range(p.getNumBodies()):
        col_link_idxs = [-1] + [i for i in range(p.getNumJoints(col_id))]
        for body_link_idx in body_link_idxs:
            for col_link_idx in col_link_idxs:
                p.setCollisionFilterPair(body_id, col_id, body_link_idx, col_link_idx, enabled)

def set_coll_filter(target_id, body_id, body_links, enable):
    # TODO(mjlbach): mostly shared with behavior robot, can be factored out
    """
    Sets collision filters for body - to enable or disable them
    :param target_id: physics body to enable/disable collisions with
    :param body_id: physics body to enable/disable collisions from
    :param body_links: physics links on body to enable/disable collisions from
    :param enable: whether to enable/disable collisions
    """
    target_link_idxs = [-1] + [i for i in range(p.getNumJoints(target_id))]

    for body_link_idx in body_links:
        for target_link_idx in target_link_idxs:
            p.setCollisionFilterPair(body_id, target_id, body_link_idx, target_link_idx, 1 if enable else 0)

def contact_collision():
    step_simulation()
    return len(p.getContactPoints(physicsClientId=CLIENT)) != 0


ContactResult = namedtuple('ContactResult', ['contactFlag', 'bodyUniqueIdA', 'bodyUniqueIdB',
                                             'linkIndexA', 'linkIndexB', 'positionOnA', 'positionOnB',
                                             'contactNormalOnB', 'contactDistance', 'normalForce'])


def pairwise_link_collision(body1, link1, body2, link2=BASE_LINK, max_distance=MAX_DISTANCE):  # 10000
    return len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance,
                                  linkIndexA=link1, linkIndexB=link2,
                                  physicsClientId=CLIENT)) != 0  # getContactPoints


def flatten_links(body, links=None):
    if links is None:
        links = get_all_links(body)
    return {(body, frozenset([link])) for link in links}


def expand_links(body):
    body, links = body if isinstance(body, tuple) else (body, None)
    if links is None:
        links = get_all_links(body)
    return body, links


def any_link_pair_collision(body1, links1, body2, links2=None, **kwargs):
    # TODO: this likely isn't needed anymore
    if links1 is None:
        links1 = get_all_links(body1)
    if links2 is None:
        links2 = get_all_links(body2)
    for link1, link2 in product(links1, links2):
        if (body1 == body2) and (link1 == link2):
            continue
        if pairwise_link_collision(body1, link1, body2, link2, **kwargs):
            #print('body {} link {} body {} link {}'.format(body1, get_link_name(body1, link1), body2, get_link_name(body2, link2)))
            return True
    return False


def body_collision(body1, body2, max_distance=MAX_DISTANCE):  # 10000
    # TODO: confirm that this doesn't just check the base link
    
    #for i in range(p.getNumJoints(body1)):
    #    for j in range(p.getNumJoints(body2)):
    #        #if len(p.getContactPoints(body1, body2, i, j)) > 0:
    #            #print('body {} {} collide with body {} {}'.format(body1, i, body2, j))

    return len(p.getClosestPoints(bodyA=body1, bodyB=body2, distance=max_distance,
                                  physicsClientId=CLIENT)) != 0  # getContactPoints`


def pairwise_collision(body1, body2, **kwargs):
    if isinstance(body1, tuple) or isinstance(body2, tuple):
        body1, links1 = expand_links(body1)
        body2, links2 = expand_links(body2)
        return any_link_pair_collision(body1, links1, body2, links2, **kwargs)
    print("{} {} {}".format(body1, body2, body_collision(body1, body2, **kwargs)))
    return body_collision(body1, body2, **kwargs)

# def single_collision(body, max_distance=1e-3):
#    return len(p.getClosestPoints(body, max_distance=max_distance)) != 0


def single_collision(body1, **kwargs):
    for body2 in get_bodies():
        if (body1 != body2) and pairwise_collision(body1, body2, **kwargs):
            return True
    return False


def link_pairs_collision(body1, links1, body2, links2=None, **kwargs):
    if links2 is None:
        links2 = get_all_links(body2)
    for link1, link2 in product(links1, links2):
        if (body1 == body2) and (link1 == link2):
            continue
        if pairwise_link_collision(body1, link1, body2, link2, **kwargs):
            return True
    return False

#####################################


Ray = namedtuple('Ray', ['start', 'end'])


def get_ray(ray):
    start, end = ray
    return np.array(end) - np.array(start)


RayResult = namedtuple('RayResult', ['objectUniqueId', 'linkIndex',
                                     'hit_fraction', 'hit_position', 'hit_normal'])


def ray_collision(ray):
    # TODO: be careful to disable gravity and set static masses for everything
    step_simulation()  # Needed for some reason
    start, end = ray
    result, = p.rayTest(start, end, physicsClientId=CLIENT)
    # TODO: assign hit_position to be the end?
    return RayResult(*result)


def batch_ray_collision(rays, threads=1):
    assert 1 <= threads <= p.MAX_RAY_INTERSECTION_BATCH_SIZE
    if not rays:
        return []
    step_simulation()  # Needed for some reason
    ray_starts = [start for start, _ in rays]
    ray_ends = [end for _, end in rays]
    return [RayResult(*tup) for tup in p.rayTestBatch(
        ray_starts, ray_ends,
        numThreads=threads,
        # parentObjectUniqueId=
        # parentLinkIndex=
        physicsClientId=CLIENT)]

#####################################

# Joint motion planning


def uniform_generator(d):
    while True:
        yield np.random.uniform(size=d)


def halton_generator(d):
    import ghalton
    seed = random.randint(0, 1000)
    #sequencer = ghalton.Halton(d)
    sequencer = ghalton.GeneralizedHalton(d, seed)
    # sequencer.reset()
    while True:
        [weights] = sequencer.get(1)
        yield np.array(weights)


def unit_generator(d, use_halton=False):
    if use_halton:
        try:
            import ghalton
        except ImportError:
            print('ghalton is not installed (https://pypi.org/project/ghalton/)')
            use_halton = False
    return halton_generator(d) if use_halton else uniform_generator(d)


def interval_generator(lower, upper, **kwargs):
    assert len(lower) == len(upper)
    assert np.less(lower, upper).all()  # TODO: equality
    extents = np.array(upper) - np.array(lower)
    for scale in unit_generator(len(lower), **kwargs):
        point = np.array(lower) + scale*extents
        yield point


def get_sample_fn(body, joints, custom_limits={}, **kwargs):
    lower_limits, upper_limits = get_custom_limits(
        body, joints, custom_limits, circular_limits=CIRCULAR_LIMITS)
    generator = interval_generator(lower_limits, upper_limits, **kwargs)

    def fn():
        return tuple(next(generator))
    return fn


def get_halton_sample_fn(body, joints, **kwargs):
    return get_sample_fn(body, joints, use_halton=True, **kwargs)


def get_difference_fn(body, joints):
    circular_joints = [is_circular(body, joint) for joint in joints]

    def fn(q2, q1):
        return tuple(circular_difference(value2, value1) if circular else (value2 - value1)
                     for circular, value2, value1 in zip(circular_joints, q2, q1))
    return fn


def get_distance_fn(body, joints, weights=None):  # , norm=2):
    # TODO: use the energy resulting from the mass matrix here?
    if weights is None:
        weights = 1*np.ones(len(joints))  # TODO: use velocities here
    difference_fn = get_difference_fn(body, joints)

    def fn(q1, q2):
        diff = np.array(difference_fn(q2, q1))
        return np.sqrt(np.dot(weights, diff * diff))
        # return np.linalg.norm(np.multiply(weights * diff), ord=norm)
    return fn


def get_refine_fn(body, joints, num_steps=0):
    difference_fn = get_difference_fn(body, joints)
    num_steps = num_steps + 1

    def fn(q1, q2):
        q = q1
        for i in range(num_steps):
            positions = (1. / (num_steps - i)) * \
                np.array(difference_fn(q2, q)) + q
            q = tuple(positions)
            #q = tuple(wrap_positions(body, joints, positions))
            yield q
    return fn


def refine_path(body, joints, waypoints, num_steps):
    refine_fn = get_refine_fn(body, joints, num_steps)
    refined_path = []
    for v1, v2 in zip(waypoints, waypoints[1:]):
        refined_path += list(refine_fn(v1, v2))
    return refined_path


DEFAULT_RESOLUTION = 0.05


def get_extend_fn(body, joints, resolutions=None, norm=2):
    # norm = 1, 2, INF
    if resolutions is None:
        resolutions = DEFAULT_RESOLUTION*np.ones(len(joints))
    difference_fn = get_difference_fn(body, joints)

    def fn(q1, q2):
        #steps = int(np.max(np.abs(np.divide(difference_fn(q2, q1), resolutions))))
        steps = int(np.linalg.norm(
            np.divide(difference_fn(q2, q1), resolutions), ord=norm))
        refine_fn = get_refine_fn(body, joints, num_steps=steps)
        return refine_fn(q1, q2)
    return fn


def remove_redundant(path, tolerance=1e-3):
    assert path
    new_path = [path[0]]
    for conf in path[1:]:
        difference = np.array(new_path[-1]) - np.array(conf)
        if not np.allclose(np.zeros(len(difference)), difference, atol=tolerance, rtol=0):
            new_path.append(conf)
    return new_path


def waypoints_from_path(path, tolerance=1e-3):
    path = remove_redundant(path, tolerance=tolerance)
    if len(path) < 2:
        return path

    def difference_fn(q2, q1): return np.array(q2) - np.array(q1)
    #difference_fn = get_difference_fn(body, joints)

    waypoints = [path[0]]
    last_conf = path[1]
    last_difference = get_unit_vector(difference_fn(last_conf, waypoints[-1]))
    for conf in path[2:]:
        difference = get_unit_vector(difference_fn(conf, waypoints[-1]))
        if not np.allclose(last_difference, difference, atol=tolerance, rtol=0):
            waypoints.append(last_conf)
            difference = get_unit_vector(difference_fn(conf, waypoints[-1]))
        last_conf = conf
        last_difference = difference
    waypoints.append(last_conf)
    return waypoints


def adjust_path(robot, joints, path):
    start_positions = get_joint_positions(robot, joints)
    difference_fn = get_difference_fn(robot, joints)
    differences = [difference_fn(q2, q1) for q1, q2 in zip(path, path[1:])]
    adjusted_path = [np.array(start_positions)]
    for difference in differences:
        adjusted_path.append(adjusted_path[-1] + difference)
    return adjusted_path


def get_moving_links(body, joints):
    moving_links = set()
    for joint in joints:
        link = child_link_from_joint(joint)
        if link not in moving_links:
            moving_links.update(get_link_subtree(body, link))
    return list(moving_links)


def get_moving_pairs(body, moving_joints):
    """
    Check all fixed and moving pairs
    Do not check all fixed and fixed pairs
    Check all moving pairs with a common
    """
    moving_links = get_moving_links(body, moving_joints)
    for link1, link2 in combinations(moving_links, 2):
        ancestors1 = set(get_joint_ancestors(body, link1)) & set(moving_joints)
        ancestors2 = set(get_joint_ancestors(body, link2)) & set(moving_joints)
        if ancestors1 != ancestors2:
            yield link1, link2


def get_self_link_pairs(body, joints, disabled_collisions=set(), only_moving=True):
    moving_links = get_moving_links(body, joints)
    fixed_links = list(set(get_links(body)) - set(moving_links))
    check_link_pairs = list(product(moving_links, fixed_links))
    if only_moving:
        check_link_pairs.extend(get_moving_pairs(body, joints))
    else:
        check_link_pairs.extend(combinations(moving_links, 2))
    check_link_pairs = list(
        filter(lambda pair: not are_links_adjacent(body, *pair), check_link_pairs))
    check_link_pairs = list(filter(lambda pair: (pair not in disabled_collisions) and
                                                (pair[::-1] not in disabled_collisions), check_link_pairs))
    return check_link_pairs


# TODO: convert most of these to keyword arguments
def get_collision_fn(body, joints, obstacles, attachments, self_collisions, disabled_collisions,
                     custom_limits={}, allow_collision_links=[], **kwargs):

    # Pair of links within the robot that need to be checked for self-collisions
    # Pairs in the disabled_collisions list are excluded
    # If the flag self_collisions is False, we do not check for any self collisions by setting this list to be empty
    check_link_pairs = get_self_link_pairs(body, joints, disabled_collisions) if self_collisions else []

    # List of links that move on the robot and that should be checked for collisions with the obstacles
    # We do not check for collisions of the links that are allowed (in the allow_collision_links list)
    moving_links = frozenset([item for item in get_moving_links(body, joints) if not item in allow_collision_links])

    # TODO: This is a fetch specific change
    attached_bodies = [attachment.child for attachment in attachments]
    moving_bodies = [(body, moving_links)] + attached_bodies
    #moving_bodies = [body] + [attachment.child for attachment in attachments]
    # + list(combinations(moving_bodies, 2))

    # Pairs to check for collisions
    check_body_pairs = list(product(moving_bodies, obstacles))
    lower_limits, upper_limits = get_custom_limits(body, joints, custom_limits)

    # TODO: maybe prune the link adjacent to the robot
    # TODO: test self collision with the holding
    def collision_fn(q):
        if not all_between(lower_limits, q, upper_limits):
            pass
            # print(lower_limits, q, upper_limits)
            # print('Joint limits violated')
            # return True
        set_joint_positions(body, joints, q)
        for attachment in attachments:
            attachment.assign()

        # Check for self collisions
        for link1, link2 in check_link_pairs:
            # Self-collisions should not have the max_distance parameter
            # , **kwargs):
            if pairwise_link_collision(body, link1, body, link2):
                return True

        # Check for collisions of the moving bodies and the obstacles
        for body1, body2 in check_body_pairs:
            if pairwise_collision(body1, body2, **kwargs):
                # print('body collision between {} and {}'.format(body1, body2))
                # if isinstance(body1, tuple):
                #     body1a, links1a = expand_links(body1)
                #     print(get_body_name(body1a))
                # else:
                #     print(get_body_name(body1))
                # if isinstance(body2, tuple):
                #     body2a, links2a = expand_links(body2)
                #     print(get_body_name(body2a))
                # else:
                #     print(get_body_name(body2))
                return True
        return False
    return collision_fn


def plan_waypoints_joint_motion(body, joints, waypoints, start_conf=None, obstacles=[], attachments=[],
                                self_collisions=True, disabled_collisions=set(),
                                resolutions=None, custom_limits={}, max_distance=MAX_DISTANCE):
    extend_fn = get_extend_fn(body, joints, resolutions=resolutions)
    collision_fn = get_collision_fn(body, joints, obstacles, attachments, self_collisions, disabled_collisions,
                                    custom_limits=custom_limits, max_distance=max_distance)
    if start_conf is None:
        start_conf = get_joint_positions(body, joints)
    else:
        assert len(start_conf) == len(joints)

    for i, waypoint in enumerate([start_conf] + list(waypoints)):
        if collision_fn(waypoint):
            #print("Warning: waypoint configuration {}/{} is in collision".format(i, len(waypoints)))
            return None
    path = [start_conf]
    for waypoint in waypoints:
        assert len(joints) == len(waypoint)
        for q in extend_fn(path[-1], waypoint):
            if collision_fn(q):
                return None
            path.append(q)
    return path


def plan_direct_joint_motion(body, joints, end_conf, **kwargs):
    return plan_waypoints_joint_motion(body, joints, [end_conf], **kwargs)


def check_initial_end(start_conf, end_conf, collision_fn):
    if collision_fn(start_conf):
        # print("Warning: initial configuration is in collision")
        return False
    if collision_fn(end_conf):
        # print("Warning: end configuration is in collision")
        return False
    return True


def plan_joint_motion(body, joints, end_conf, obstacles=[], attachments=[],
                      self_collisions=True, disabled_collisions=set(),
                      weights=None, resolutions=None, max_distance=MAX_DISTANCE, custom_limits={}, algorithm='birrt', allow_collision_links=[], **kwargs):

    assert len(joints) == len(end_conf)
    sample_fn = get_sample_fn(body, joints, custom_limits=custom_limits)
    distance_fn = get_distance_fn(body, joints, weights=weights)
    extend_fn = get_extend_fn(body, joints, resolutions=resolutions)
    collision_fn = get_collision_fn(body, joints, obstacles, attachments, self_collisions, disabled_collisions,
                                    custom_limits=custom_limits, max_distance=max_distance, allow_collision_links=allow_collision_links)

    start_conf = get_joint_positions(body, joints)

    if not check_initial_end(start_conf, end_conf, collision_fn):
        return None
    if algorithm == 'direct':
        return direct_path(start_conf, end_conf, extend_fn, collision_fn)
    elif algorithm == 'birrt':
        return birrt(start_conf, end_conf, distance_fn,
                     sample_fn, extend_fn, collision_fn, **kwargs)
    elif algorithm == 'rrt_star':
        return rrt_star(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn, max_iterations=5000, **kwargs)
    elif algorithm == 'rrt':
        return rrt(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn, iterations=500, **kwargs)
    elif algorithm == 'lazy_prm':
        return lazy_prm_replan_loop(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn, [500, 2000, 5000], **kwargs)
    else:
        return None


def plan_lazy_prm(start_conf, end_conf, sample_fn, extend_fn, collision_fn, **kwargs):
    # TODO: cost metric based on total robot movement (encouraging greater distances possibly)
    from motion_planners.lazy_prm import lazy_prm
    path, samples, edges, colliding_vertices, colliding_edges = lazy_prm(
        start_conf, end_conf, sample_fn, extend_fn, collision_fn, num_samples=200, **kwargs)
    if path is None:
        return path

    #lower, upper = get_custom_limits(body, joints, circular_limits=CIRCULAR_LIMITS)
    def draw_fn(q):  # TODO: draw edges instead of vertices
        return np.append(q[:2], [1e-3])
        # return np.array([1, 1, 0.25])*(q + np.array([0., 0., np.pi]))
    handles = []
    for q1, q2 in zip(path, path[1:]):
        handles.append(add_line(draw_fn(q1), draw_fn(q2), color=(0, 1, 0)))
    for i1, i2 in edges:
        color = (0, 0, 1)
        if any(colliding_vertices.get(i, False) for i in (i1, i2)) or colliding_vertices.get((i1, i2), False):
            color = (1, 0, 0)
        elif not colliding_vertices.get((i1, i2), True):
            color = (0, 0, 0)
        handles.append(
            add_line(draw_fn(samples[i1]), draw_fn(samples[i2]), color=color))
    wait_for_user()
    return path

#####################################


def get_closest_angle_fn(body, joints, linear_weight=1., angular_weight=1., reversible=True):
    assert len(joints) == 3
    linear_extend_fn = get_distance_fn(
        body, joints[:2], weights=linear_weight*np.ones(2))
    angular_extend_fn = get_distance_fn(
        body, joints[2:], weights=[angular_weight])

    def closest_fn(q1, q2):
        angle_and_distance = []
        for direction in [0, PI] if reversible else [PI]:
            angle = get_angle(q1[:2], q2[:2]) + direction
            distance = angular_extend_fn(q1[2:], [angle]) \
                + linear_extend_fn(q1[:2], q2[:2]) \
                + angular_extend_fn([angle], q2[2:])
            angle_and_distance.append((angle, distance))
        return min(angle_and_distance, key=lambda pair: pair[1])
    return closest_fn


def get_nonholonomic_distance_fn(body, joints, weights=None, **kwargs):
    assert weights is None
    closest_angle_fn = get_closest_angle_fn(body, joints, **kwargs)

    def distance_fn(q1, q2):
        _, distance = closest_angle_fn(q1, q2)
        return distance
    return distance_fn


def get_nonholonomic_extend_fn(body, joints, resolutions=None, **kwargs):
    assert resolutions is None
    assert len(joints) == 3
    linear_extend_fn = get_extend_fn(body, joints[:2])
    angular_extend_fn = get_extend_fn(body, joints[2:])
    closest_angle_fn = get_closest_angle_fn(body, joints, **kwargs)

    def extend_fn(q1, q2):
        angle, _ = closest_angle_fn(q1, q2)
        path = []
        for aq in angular_extend_fn(q1[2:], [angle]):
            path.append(np.append(q1[:2], aq))
        for lq in linear_extend_fn(q1[:2], q2[:2]):
            path.append(np.append(lq, [angle]))
        for aq in angular_extend_fn([angle], q2[2:]):
            path.append(np.append(q2[:2], aq))
        return path
    return extend_fn


def plan_nonholonomic_motion(body, joints, end_conf, obstacles=[], attachments=[],
                             self_collisions=True, disabled_collisions=set(),
                             weights=None, resolutions=None, reversible=True,
                             max_distance=MAX_DISTANCE, custom_limits={}, **kwargs):

    assert len(joints) == len(end_conf)
    sample_fn = get_sample_fn(body, joints, custom_limits=custom_limits)
    distance_fn = get_nonholonomic_distance_fn(
        body, joints, weights=weights, reversible=reversible)
    extend_fn = get_nonholonomic_extend_fn(
        body, joints, resolutions=resolutions, reversible=reversible)
    collision_fn = get_collision_fn(body, joints, obstacles, attachments,
                                    self_collisions, disabled_collisions,
                                    custom_limits=custom_limits, max_distance=max_distance)

    start_conf = get_joint_positions(body, joints)
    if not check_initial_end(start_conf, end_conf, collision_fn):
        return None
    return birrt(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn, **kwargs)

#####################################

# SE(2) pose motion planning


def get_base_difference_fn():
    def fn(q2, q1):
        dx, dy = np.array(q2[:2]) - np.array(q1[:2])
        dtheta = circular_difference(q2[2], q1[2])
        return (dx, dy, dtheta)
    return fn


def get_base_distance_fn(weights=1*np.ones(3)):
    difference_fn = get_base_difference_fn()

    def fn(q1, q2):
        difference = np.array(difference_fn(q2, q1))
        return np.sqrt(np.dot(weights, difference * difference))
    return fn


def plan_base_motion(body, end_conf, base_limits, obstacles=[], direct=False,
                     weights=1*np.ones(3), resolutions=0.05*np.ones(3),
                     max_distance=MAX_DISTANCE, **kwargs):
    def sample_fn():
        x, y = np.random.uniform(*base_limits)
        theta = np.random.uniform(*CIRCULAR_LIMITS)
        return (x, y, theta)

    difference_fn = get_base_difference_fn()
    distance_fn = get_base_distance_fn(weights=weights)

    def extend_fn(q1, q2):
        target_theta = np.arctan2(q2[1]-q1[1], q2[0]-q1[0])

        n1 = int(np.abs(circular_difference(
            target_theta, q1[2]) / resolutions[2])) + 1
        n3 = int(np.abs(circular_difference(
            q2[2], target_theta) / resolutions[2])) + 1
        steps2 = np.abs(np.divide(difference_fn(q2, q1), resolutions))
        n2 = int(np.max(steps2)) + 1

        for i in range(n1):
            q = (i / (n1)) * \
                np.array(difference_fn(
                    (q1[0], q1[1], target_theta), q1)) + np.array(q1)
            q = tuple(q)
            yield q

        for i in range(n2):
            q = (i / (n2)) * np.array(difference_fn((q2[0], q2[1], target_theta),
                                                    (q1[0], q1[1], target_theta))) + np.array((q1[0], q1[1], target_theta))
            q = tuple(q)
            yield q

        for i in range(n3):
            q = (i / (n3)) * np.array(difference_fn(q2,
                                                    (q2[0], q2[1], target_theta))) + np.array((q2[0], q2[1], target_theta))
            q = tuple(q)
            yield q

    def collision_fn(q):
        # TODO: update this function
        set_base_values(body, q)
        return any(pairwise_collision(body, obs, max_distance=max_distance) for obs in obstacles)

    start_conf = get_base_values(body)
    if collision_fn(start_conf):
        #print("Warning: initial configuration is in collision")
        return None
    if collision_fn(end_conf):
        #print("Warning: end configuration is in collision")
        return None
    if direct:
        return direct_path(start_conf, end_conf, extend_fn, collision_fn)
    return birrt(start_conf, end_conf, distance_fn,
                 sample_fn, extend_fn, collision_fn, **kwargs)


def plan_base_motion_2d(body,
                        end_conf,
                        base_limits,
                        map_2d,
                        occupancy_range,
                        grid_resolution,
                        robot_footprint_radius_in_map,
                        obstacles=[],
                        weights=1 * np.ones(3),
                        resolutions=0.05 * np.ones(3),
                        max_distance=MAX_DISTANCE,
                        min_goal_dist=-0.02,
                        algorithm='birrt',
                        optimize_iter=0,
                        visualize_planning=True,
                        visualize_result=True,
                        metric2map=None,
                        flip_vertically=False,
                        use_pb_for_collisions=False,
                        debugging_prints = False,
                        upsampling_factor = 4,
                        **kwargs):
    """
    Performs motion planning for a robot base in 2D
    :param body: PyBullet id of the robot
    :param end_conf: End configuration, 3D vector of the goal (x, y, theta)
    :param base_limits: Limits to sample new configurations
    :param map_2d: Map to check collisions for a sampled location
    :param occupancy_range: size of the occupancy map in metric units (assumed square), e.g., meters from left to right
           borders of the occupancy map
    :param grid_resolution: resolution/size of the occupancy grid image (pixels, expected to be squared)
           together with the occupancy_range provides the size of each pixel in metric units:
           occupancy_range/grid_resolution [m/pixel]
    :param robot_footprint_radius_in_map: Number of pixels that the robot base occupies in the occupancy map
    :param obstacles: list of obstacles to check for collisions if using pybullet as collision checker
    :param weights: Weights to compute the distance between base configurations:
            d_12 = sqrt(w1*(x1-x2)^2 + w2*(y1-y2)^2 + w3*(theta1-theta2)^2)
    :param resolutions: step-size to extend between points
    :param max_distance: maximum distance to check collisions if using pybullet as collision checker
    :param min_goal_dist: minimum distance between the current robot location and the goal for doing planning
    :param algorithm: planning algorithm to use
    :param optimize_iter: iterations of the optimizer to run after a path has been found, to smooth it out
    :param visualize_planning: boolean. To visualize the planning process (tested with birrt only)
    :param visualize_result: To visualize the planning results (tested with birrt only)
    :param metric2map: Function to map points from metric space into pixels on the map
    :param flip_vertically: If the image needs to be flipped (for global maps)
    :param use_pb_for_collisions:  If pybullet is used to check for collisions. If not, we use the local or global 2D
           traversable map
    :param debugging_prints: Boolean to activate debugging prints
    :param upsampling_factor: Upscaling factor to enlarge the maps
    :param kwargs:
    :return: Path (if found)
    """
    # Sampling function. Generates sample configurations within the limits
    def sample_fn():
        x, y = np.random.uniform(*base_limits)
        theta = np.random.uniform(*CIRCULAR_LIMITS)
        return (x, y, theta)

    # Function that measures the difference between two base configuration.
    # It returns a 3D vector: diff in x, y, and theta
    difference_fn = get_base_difference_fn()
    # Function that measures the distance between two base configurations. It returns
    # d_12 = sqrt(w1*(x1-x2)^2 + w2*(y1-y2)^2 + w3*(theta1-theta2)^2)
    # diff theta is computed using circular difference
    distance_fn = get_base_distance_fn(weights=weights)

    # Function that creates candidates of points to extend a path from q1 towards q2
    def extend_fn(q1, q2):
        target_theta = np.arctan2(q2[1] - q1[1], q2[0] - q1[0]) # Amount of rotation requested

        n1 = int(np.abs(circular_difference(target_theta, q1[2]) / resolutions[2])) + 1
        n3 = int(np.abs(circular_difference(q2[2], target_theta) / resolutions[2])) + 1
        steps2 = np.abs(np.divide(difference_fn(q2, q1), resolutions))
        n2 = int(np.max(steps2)) + 1

        # First interpolate between the initial point with initial orientation, and target orientation
        for i in range(n1+1):
            q = (i / (n1)) * np.array(difference_fn((q1[0], q1[1], target_theta), q1)) + np.array(q1)
            q = tuple(q)
            yield q

        # Then interpolate between initial point and goal point, keeping the target orientation
        for i in range(n2):
            q = (i / (n2)) * np.array(
                difference_fn((q2[0], q2[1], target_theta), (q1[0], q1[1], target_theta))) + np.array(
                (q1[0], q1[1], target_theta))
            q = tuple(q)
            yield q

        # Finally, interpolate between the final point with target orientation and final point with final orientation
        for i in range(n3+1):
            q = (i / (n3)) * np.array(difference_fn(q2, (q2[0], q2[1], target_theta))) + np.array(
                (q2[0], q2[1], target_theta))
            q = tuple(q)
            yield q

    # Get the start configuration: x, y, theta of the robot base
    start_conf = get_base_values(body)

    # Do not plan for goals that are very close by
    # This makes it impossible to "plan" for pure rotations. Use a negative min_goal_dist to allow pure rotations
    if np.abs(start_conf[0] - end_conf[0]) < min_goal_dist and np.abs(start_conf[1] - end_conf[1]) < min_goal_dist:
        print("Warning: goal is too close to the initial position. Returning")
        return None

    def transform_point_to_occupancy_map(q):
        if metric2map is None: # a local occupancy map is used
            # Vector connecting robot location and the query confs. in metric absolute space
            delta = np.array(q)[:2] - np.array(start_conf)[:2]

            # Initial orientation of the base
            theta = start_conf[2]
            # X axis of the robot given the initial orientation (x is forward)
            x_dir = np.array([np.sin(theta), -np.cos(theta)])
            # Y axis of the robot given the initial orientation (y is to the right of the robot)
            y_dir = np.array([np.cos(theta), np.sin(theta)])

            # Place the query configuration in robot's reference frame (metric robot-relative space)
            end_in_start_frame = [x_dir.dot(delta), y_dir.dot(delta)]

            # Find the corresponding pixel for the query in the occupancy map, assuming the robot is at the center
            # (image-space)
            pts = np.array(end_in_start_frame) * (grid_resolution/occupancy_range) + grid_resolution / 2
            pts = pts.astype(np.int32)
        else:
            pts = metric2map(np.array(q[0:2]))

        if debugging_prints:
            print("original point {} and in image: {}".format(q, pts))
        return pts

    # Draw the initial situation of the planning problem: src, dst and occupancy map
    if visualize_planning or visualize_result:
        planning_map_2d = cv2.cvtColor(map_2d, cv2.COLOR_GRAY2RGB)
        origin_in_image = transform_point_to_occupancy_map(start_conf)
        goal_in_image = transform_point_to_occupancy_map(end_conf)
        cv2.circle(planning_map_2d, [origin_in_image[1], origin_in_image[0]], radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(planning_map_2d, [goal_in_image[1], goal_in_image[0]], radius=3, color=(255, 0, 255), thickness=-1)
        if visualize_planning:
            cv2.namedWindow("Planning Problem")
            planning_map_2d_upsampled = cv2.resize(planning_map_2d, None, None, upsampling_factor, upsampling_factor, cv2.INTER_NEAREST)
            # We need to flip vertically if we use the global map. Points are already in the right frame
            if flip_vertically:
                planning_map_2d_upsampled = cv2.flip(planning_map_2d_upsampled, 0)
            cv2.imshow("Planning Problem", planning_map_2d_upsampled)
            cv2.waitKey(1)

    # Auxiliary function to draw two points and the line connecting them, to draw paths
    def draw_path(q1, q2, color, image=None):
        pt1 = transform_point_to_occupancy_map(q1)
        pt2 = transform_point_to_occupancy_map(q2)
        if image is None:
            cv2.circle(planning_map_2d, [pt2[1], pt2[0]], radius=1, color=color, thickness=-1)
            cv2.line(planning_map_2d, [pt1[1], pt1[0]], [pt2[1], pt2[0]], color, thickness=1, lineType=8)
            planning_map_2d_upsampled = cv2.resize(planning_map_2d, None, None, upsampling_factor, upsampling_factor, cv2.INTER_NEAREST)
            # We need to flip vertically if we use the global map. Points are already in the right frame
            if flip_vertically:
                planning_map_2d_upsampled = cv2.flip(planning_map_2d_upsampled, 0)
            cv2.imshow("Planning Problem", planning_map_2d_upsampled)
            cv2.waitKey(1)
        else:
            cv2.line(image, [pt1[1], pt1[0]], [pt2[1], pt2[0]], color, thickness=1, lineType=8)

    # Auxiliary function to draw a point
    def draw_point(q1, color, not_in_image=False, radius=1):
        if not_in_image:
            q = transform_point_to_occupancy_map(q1)
        else:
            q = q1
        cv2.circle(planning_map_2d, (q[1], q[0]), radius=radius, color=color, thickness=-1)
        planning_map_2d_upsampled = cv2.resize(planning_map_2d, None, None, upsampling_factor, upsampling_factor, cv2.INTER_NEAREST)
        # We need to flip vertically if we use the global map. Points are already in the right frame
        if flip_vertically:
            planning_map_2d_upsampled = cv2.flip(planning_map_2d_upsampled, 0)
        cv2.imshow("Planning Problem", planning_map_2d_upsampled)
        cv2.waitKey(1)

    # Function to check collisions for a given configuration q
    def collision_fn(q):
        # TODO: update this function
        pts = transform_point_to_occupancy_map(q)

        if visualize_planning:
            draw_point(pts, (0, 155, 155))
            cv2.waitKey(1)

        if use_pb_for_collisions:
            set_base_values(body, q)
            in_collision = any(pairwise_collision(body, obs, max_distance=max_distance) for obs in obstacles)
            set_base_values(body, start_conf)   # Reset the original configuration
        else:   # Use local or global map
            # If the point is outside of the map/occupancy map, then we return "collision"
            if pts[0] < robot_footprint_radius_in_map or pts[1] < robot_footprint_radius_in_map \
                or pts[0] > grid_resolution - robot_footprint_radius_in_map - 1 or pts[
                    1] > grid_resolution - robot_footprint_radius_in_map - 1:
                return True

            # We create a mask using the robot size (assumed base is circular) to check collisions around a point
            mask = np.zeros((robot_footprint_radius_in_map * 2 + 1,
                             robot_footprint_radius_in_map * 2 + 1))
            cv2.circle(mask, (robot_footprint_radius_in_map, robot_footprint_radius_in_map), robot_footprint_radius_in_map,
                       1, -1)
            mask = mask.astype(np.bool)

            # Check if all the points where the shifted mask of the robot base overlaps with the occupancy map are
            # marked as FREESPACE, and return false if not
            map_2d_around_robot = map_2d[pts[0] - robot_footprint_radius_in_map : pts[0] + robot_footprint_radius_in_map + 1,
                         pts[1] - robot_footprint_radius_in_map : pts[1] + robot_footprint_radius_in_map + 1]

            in_collision = not np.all(map_2d_around_robot[mask] == OccupancyGridState.FREESPACE)

        if visualize_planning:
            planning_map_2d_cpy = planning_map_2d[
                                  pts[0] - robot_footprint_radius_in_map: pts[0] + robot_footprint_radius_in_map + 1,
                                  pts[1] - robot_footprint_radius_in_map: pts[1] + robot_footprint_radius_in_map + 1].copy()
            draw_point(pts, 1, radius=robot_footprint_radius_in_map)
            cv2.waitKey(10) #Extra wait to visualize better the process

        if debugging_prints:
            print("In collision? {}".format(in_collision))
        if visualize_planning:
            planning_map_2d[pts[0] - robot_footprint_radius_in_map: pts[0] + robot_footprint_radius_in_map + 1, pts[1] - robot_footprint_radius_in_map: pts[1] + robot_footprint_radius_in_map + 1] = planning_map_2d_cpy
            planning_map_2d_upsampled = cv2.resize(planning_map_2d, None, None, upsampling_factor, upsampling_factor, cv2.INTER_NEAREST)
            # We need to flip vertically if we use the global map. Points are already in the right frame
            if flip_vertically:
                planning_map_2d_upsampled = cv2.flip(planning_map_2d_upsampled, 0)
            cv2.imshow("Planning Problem", planning_map_2d_upsampled)
            cv2.waitKey(10)

        return in_collision

    # Do not plan if the initial pose is in collision
    if collision_fn(start_conf):
        if debugging_prints:
            print("Warning: initial configuration is in collision")
        return None
    # Do not plan if the final pose is in collision
    if collision_fn(end_conf):
        if debugging_prints:
            print("Warning: end configuration is in collision")
        return None
    
    if algorithm == 'direct':
        path = direct_path(start_conf, end_conf, extend_fn, collision_fn)
    elif algorithm == 'birrt':
        path = birrt(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn, draw_path=[None, draw_path][visualize_planning], draw_point=[None, draw_point][visualize_planning], **kwargs)
    elif algorithm == 'rrt_star':
        path = rrt_star(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn, max_iterations=5000, **kwargs)
    elif algorithm == 'rrt':
        path = rrt(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn, iterations=5000, **kwargs)
    elif algorithm == 'lazy_prm':
        path = lazy_prm_replan_loop(start_conf, end_conf, distance_fn, sample_fn, extend_fn, collision_fn, [250, 500, 1000, 2000, 4000, 4000], **kwargs)
    else:
        path = None

    if optimize_iter > 0 and path is not None:
        logging.info("Optimizing the path found")
        path = optimize_path(path, extend_fn, collision_fn, iterations=optimize_iter)

    if visualize_result and path is not None:
        cv2.namedWindow("Resulting Plan")
        result_map_2d = cv2.cvtColor(map_2d, cv2.COLOR_GRAY2RGB)
        origin_in_image = transform_point_to_occupancy_map(start_conf)
        goal_in_image = transform_point_to_occupancy_map(end_conf)
        cv2.circle(result_map_2d, [origin_in_image[1], origin_in_image[0]], radius=3, color=(0, 0, 255), thickness=-1)
        cv2.circle(result_map_2d, [goal_in_image[1], goal_in_image[0]], radius=3, color=(255, 0, 255), thickness=-1)
        for idx in range(len(path)-1):
            draw_path(path[idx], path[idx+1], (0,0,155), image=result_map_2d)
        result_map_2d_scaledup = cv2.resize(result_map_2d, None, None, upsampling_factor, upsampling_factor, cv2.INTER_NEAREST)
        # We need to flip vertically if we use the global map. Points are already in the right frame
        if flip_vertically:
            result_map_2d_scaledup = cv2.flip(result_map_2d_scaledup, 0)
        cv2.imshow("Resulting Plan", result_map_2d_scaledup)
        cv2.waitKey(10)

    return path

#####################################

# Placements


def stable_z_on_aabb(body, aabb):
    center, extent = get_center_extent(body)
    _, upper = aabb
    return (upper + extent/2 + (get_point(body) - center))[2]


def stable_z(body, surface, surface_link=None):
    return stable_z_on_aabb(body, get_aabb(surface, link=surface_link))


def is_placed_on_aabb(body, bottom_aabb, above_epsilon=1e-2, below_epsilon=0.0):
    assert (0 <= above_epsilon) and (0 <= below_epsilon)
    top_aabb = get_aabb(body)  # TODO: approximate_as_prism
    top_z_min = top_aabb[0][2]
    bottom_z_max = bottom_aabb[1][2]
    return ((bottom_z_max - below_epsilon) <= top_z_min <= (bottom_z_max + above_epsilon)) and \
           (aabb_contains_aabb(aabb2d_from_aabb(top_aabb), aabb2d_from_aabb(bottom_aabb)))


def is_placement(body, surface, **kwargs):
    return is_placed_on_aabb(body, get_aabb(surface), **kwargs)


def is_center_on_aabb(body, bottom_aabb, above_epsilon=1e-2, below_epsilon=0.0):
    # TODO: compute AABB in origin
    # TODO: use center of mass?
    assert (0 <= above_epsilon) and (0 <= below_epsilon)
    center, extent = get_center_extent(body)  # TODO: approximate_as_prism
    base_center = center - np.array([0, 0, extent[2]])/2
    top_z_min = base_center[2]
    bottom_z_max = bottom_aabb[1][2]
    return ((bottom_z_max - abs(below_epsilon)) <= top_z_min <= (bottom_z_max + abs(above_epsilon))) and \
           (aabb_contains_point(
               base_center[:2], aabb2d_from_aabb(bottom_aabb)))


def is_center_stable(body, surface, **kwargs):
    return is_center_on_aabb(body, get_aabb(surface), **kwargs)


def sample_placement_on_aabb(top_body, bottom_aabb, top_pose=unit_pose(),
                             percent=1.0, max_attempts=50, epsilon=1e-3):
    # TODO: transform into the coordinate system of the bottom
    # TODO: maybe I should instead just require that already in correct frame
    for _ in range(max_attempts):
        theta = np.random.uniform(*CIRCULAR_LIMITS)
        rotation = Euler(yaw=theta)
        set_pose(top_body, multiply(Pose(euler=rotation), top_pose))
        center, extent = get_center_extent(top_body)
        lower = (np.array(bottom_aabb[0]) + percent*extent/2)[:2]
        upper = (np.array(bottom_aabb[1]) - percent*extent/2)[:2]
        if np.less(upper, lower).any():
            continue
        x, y = np.random.uniform(lower, upper)
        z = (bottom_aabb[1] + extent/2.)[2] + epsilon
        point = np.array([x, y, z]) + (get_point(top_body) - center)
        pose = multiply(Pose(point, rotation), top_pose)
        set_pose(top_body, pose)
        return pose
    return None


def sample_placement(top_body, bottom_body, bottom_link=None, **kwargs):
    bottom_aabb = get_aabb(bottom_body, link=bottom_link)
    return sample_placement_on_aabb(top_body, bottom_aabb, **kwargs)

#####################################

# Reachability


def sample_reachable_base(robot, point, reachable_range=(0.25, 1.0)):
    radius = np.random.uniform(*reachable_range)
    x, y = radius*unit_from_theta(np.random.uniform(-np.pi, np.pi)) + point[:2]
    yaw = np.random.uniform(*CIRCULAR_LIMITS)
    base_values = (x, y, yaw)
    #set_base_values(robot, base_values)
    return base_values


def uniform_pose_generator(robot, gripper_pose, **kwargs):
    point = point_from_pose(gripper_pose)
    while True:
        base_values = sample_reachable_base(robot, point, **kwargs)
        if base_values is None:
            break
        yield base_values
        #set_base_values(robot, base_values)
        # yield get_pose(robot)

#####################################

# Constraints - applies forces when not satisfied


def get_constraints():
    """
    getConstraintUniqueId will take a serial index in range 0..getNumConstraints,  and reports the constraint unique id.
    Note that the constraint unique ids may not be contiguous, since you may remove constraints.
    """
    return [p.getConstraintUniqueId(i, physicsClientId=CLIENT)
            for i in range(p.getNumConstraints(physicsClientId=CLIENT))]

def get_constraint_violation(cid):
    (
        parent_body,
        parent_link,
        child_body,
        child_link,
        _,
        _,
        joint_position_parent,
        joint_position_child,
    ) = p.getConstraintInfo(cid)[:8]

    if parent_link == -1:
        parent_link_pos, parent_link_orn = p.getBasePositionAndOrientation(parent_body)
    else:
        parent_link_pos, parent_link_orn = p.getLinkState(parent_body, parent_link)[:2]

    if child_link == -1:
        child_link_pos, child_link_orn = p.getBasePositionAndOrientation(child_body)
    else:
        child_link_pos, child_link_orn = p.getLinkState(child_body, child_link)[:2]

    joint_pos_in_parent_world = p.multiplyTransforms(
        parent_link_pos, parent_link_orn, joint_position_parent, [0, 0, 0, 1]
    )[0]
    joint_pos_in_child_world = p.multiplyTransforms(
        child_link_pos, child_link_orn, joint_position_child, [0, 0, 0, 1]
    )[0]

    diff = np.linalg.norm(np.array(joint_pos_in_parent_world) - np.array(joint_pos_in_child_world))
    return diff


def add_p2p_constraint(body, body_link, robot, robot_link, max_force=None):
    if body_link == -1:
        body_pose = get_pose(body)
    else:
        body_pose = get_com_pose(body, link=body_link)
    #end_effector_pose = get_link_pose(robot, robot_link)
    end_effector_pose = get_com_pose(robot, robot_link)
    grasp_pose = multiply(invert(end_effector_pose), body_pose)
    point, quat = grasp_pose
    # TODO: can I do this when I'm not adjacent?
    # joint axis in local frame (ignored for JOINT_FIXED)
    # return p.createConstraint(robot, robot_link, body, body_link,
    #                          p.JOINT_FIXED, jointAxis=unit_point(),
    #                          parentFramePosition=unit_point(),
    #                          childFramePosition=point,
    #                          parentFrameOrientation=unit_quat(),
    #                          childFrameOrientation=quat)
    constraint = p.createConstraint(robot, robot_link, body, body_link,  # Both seem to work
                                    p.JOINT_POINT2POINT, jointAxis=unit_point(),
                                    parentFramePosition=point,
                                    childFramePosition=unit_point(),
                                    parentFrameOrientation=quat,
                                    childFrameOrientation=unit_quat(),
                                    physicsClientId=CLIENT)
    if max_force is not None:
        p.changeConstraint(constraint, maxForce=max_force,
                           physicsClientId=CLIENT)
    return constraint


def remove_constraint(constraint):
    p.removeConstraint(constraint, physicsClientId=CLIENT)


ConstraintInfo = namedtuple('ConstraintInfo', ['parentBodyUniqueId', 'parentJointIndex',
                                               'childBodyUniqueId', 'childLinkIndex', 'constraintType',
                                               'jointAxis', 'jointPivotInParent', 'jointPivotInChild',
                                               'jointFrameOrientationParent', 'jointFrameOrientationChild', 'maxAppliedForce'])


def get_constraint_info(constraint):  # getConstraintState
    # TODO: four additional arguments
    return ConstraintInfo(*p.getConstraintInfo(constraint, physicsClientId=CLIENT)[:11])


def get_fixed_constraints():
    fixed_constraints = []
    for constraint in get_constraints():
        constraint_info = get_constraint_info(constraint)
        if constraint_info.constraintType == p.JOINT_FIXED:
            fixed_constraints.append(constraint)
    return fixed_constraints


def add_fixed_constraint(body, robot, robot_link, max_force=None):
    body_link = BASE_LINK
    body_pose = get_pose(body)
    #body_pose = get_com_pose(body, link=body_link)
    #end_effector_pose = get_link_pose(robot, robot_link)
    end_effector_pose = get_com_pose(robot, robot_link)
    grasp_pose = multiply(invert(end_effector_pose), body_pose)
    point, quat = grasp_pose
    # TODO: can I do this when I'm not adjacent?
    # joint axis in local frame (ignored for JOINT_FIXED)
    # return p.createConstraint(robot, robot_link, body, body_link,
    #                          p.JOINT_FIXED, jointAxis=unit_point(),
    #                          parentFramePosition=unit_point(),
    #                          childFramePosition=point,
    #                          parentFrameOrientation=unit_quat(),
    #                          childFrameOrientation=quat)
    constraint = p.createConstraint(robot, robot_link, body, body_link,  # Both seem to work
                                    p.JOINT_FIXED, jointAxis=unit_point(),
                                    parentFramePosition=point,
                                    childFramePosition=unit_point(),
                                    parentFrameOrientation=quat,
                                    childFrameOrientation=unit_quat(),
                                    physicsClientId=CLIENT)
    if max_force is not None:
        p.changeConstraint(constraint, maxForce=max_force,
                           physicsClientId=CLIENT)
    return constraint


def remove_fixed_constraint(body, robot, robot_link):
    for constraint in get_fixed_constraints():
        constraint_info = get_constraint_info(constraint)
        if (body == constraint_info.childBodyUniqueId) and \
                (BASE_LINK == constraint_info.childLinkIndex) and \
                (robot == constraint_info.parentBodyUniqueId) and \
                (robot_link == constraint_info.parentJointIndex):
            remove_constraint(constraint)

#####################################

# Grasps


GraspInfo = namedtuple('GraspInfo', ['get_grasps', 'approach_pose'])


class Attachment(object):
    def __init__(self, parent, parent_link, grasp_pose, child):
        self.parent = parent
        self.parent_link = parent_link
        self.grasp_pose = grasp_pose
        self.child = child
        # self.child_link = child_link # child_link=BASE_LINK

    @property
    def bodies(self):
        return flatten_links(self.child) | flatten_links(self.parent, get_link_subtree(
            self.parent, self.parent_link))

    def assign(self):
        parent_link_pose = get_link_pose(self.parent, self.parent_link)
        child_pose = body_from_end_effector(parent_link_pose, self.grasp_pose)
        set_pose(self.child, child_pose)
        return child_pose

    def apply_mapping(self, mapping):
        self.parent = mapping.get(self.parent, self.parent)
        self.child = mapping.get(self.child, self.child)

    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, self.parent, self.child)


def create_attachment(parent, parent_link, child):
    parent_link_pose = get_link_pose(parent, parent_link)
    child_pose = get_pose(child)
    grasp_pose = multiply(invert(parent_link_pose), child_pose)
    return Attachment(parent, parent_link, grasp_pose, child)


def body_from_end_effector(end_effector_pose, grasp_pose):
    """
    world_from_parent * parent_from_child = world_from_child
    """
    return multiply(end_effector_pose, grasp_pose)


def end_effector_from_body(body_pose, grasp_pose):
    """
    grasp_pose: the body's pose in gripper's frame

    world_from_child * (parent_from_child)^(-1) = world_from_parent
    (parent: gripper, child: body to be grasped)

    Pose_{world,gripper} = Pose_{world,block}*Pose_{block,gripper}
                         = Pose_{world,block}*(Pose_{gripper,block})^{-1}
    """
    return multiply(body_pose, invert(grasp_pose))


def approach_from_grasp(approach_pose, end_effector_pose):
    return multiply(approach_pose, end_effector_pose)


def get_grasp_pose(constraint):
    """
    Grasps are parent_from_child
    """
    constraint_info = get_constraint_info(constraint)
    assert(constraint_info.constraintType == p.JOINT_FIXED)
    joint_from_parent = (constraint_info.jointPivotInParent,
                         constraint_info.jointFrameOrientationParent)
    joint_from_child = (constraint_info.jointPivotInChild,
                        constraint_info.jointFrameOrientationChild)
    return multiply(invert(joint_from_parent), joint_from_child)

#####################################

# Control


def control_joint(body, joint, value):
    return p.setJointMotorControl2(bodyUniqueId=body,
                                   jointIndex=joint,
                                   controlMode=p.POSITION_CONTROL,
                                   targetPosition=value,
                                   targetVelocity=0.0,
                                   maxVelocity=get_max_velocity(body, joint),
                                   force=get_max_force(body, joint),
                                   physicsClientId=CLIENT)


def control_joints(body, joints, positions):
    # TODO: the whole PR2 seems to jitter
    #kp = 1.0
    #kv = 0.3
    forces = [get_max_force(body, joint) * 100 for joint in joints]
    #forces = [5000]*len(joints)
    #forces = [20000]*len(joints)
    return p.setJointMotorControlArray(body, joints, p.POSITION_CONTROL,
                                       targetPositions=positions,
                                       targetVelocities=[0.0] * len(joints),
                                       physicsClientId=CLIENT, forces=forces)  # ,
    #positionGains=[kp] * len(joints),
    # velocityGains=[kv] * len(joints),)
    # forces=forces)


def joint_controller(body, joints, target, tolerance=1e-3):
    assert(len(joints) == len(target))
    positions = get_joint_positions(body, joints)
    while not np.allclose(positions, target, atol=tolerance, rtol=0):
        control_joints(body, joints, target)
        yield positions
        positions = get_joint_positions(body, joints)


def joint_controller_hold(body, joints, target, **kwargs):
    """
    Keeps other joints in place
    """
    movable_joints = get_movable_joints(body)
    conf = list(get_joint_positions(body, movable_joints))
    for joint, value in zip(movable_from_joints(body, joints), target):
        conf[joint] = value
    return joint_controller(body, movable_joints, conf, **kwargs)


def joint_controller_hold2(body, joints, positions, velocities=None,
                           tolerance=1e-2 * np.pi, position_gain=0.05, velocity_gain=0.01):
    """
    Keeps other joints in place
    """
    # TODO: velocity_gain causes the PR2 to oscillate
    if velocities is None:
        velocities = [0.] * len(positions)
    movable_joints = get_movable_joints(body)
    target_positions = list(get_joint_positions(body, movable_joints))
    target_velocities = [0.] * len(movable_joints)
    movable_from_original = {o: m for m, o in enumerate(movable_joints)}
    #print(list(positions), list(velocities))
    for joint, position, velocity in zip(joints, positions, velocities):
        target_positions[movable_from_original[joint]] = position
        target_velocities[movable_from_original[joint]] = velocity
    # return joint_controller(body, movable_joints, conf)
    current_conf = get_joint_positions(body, movable_joints)
    #forces = [get_max_force(body, joint) for joint in movable_joints]
    while not np.allclose(current_conf, target_positions, atol=tolerance, rtol=0):
        # TODO: only enforce velocity constraints at end
        p.setJointMotorControlArray(body, movable_joints, p.POSITION_CONTROL,
                                    targetPositions=target_positions,
                                    # targetVelocities=target_velocities,
                                    positionGains=[
                                        position_gain] * len(movable_joints),
                                    #velocityGains=[velocity_gain] * len(movable_joints),
                                    # forces=forces,
                                    physicsClientId=CLIENT)
        yield current_conf
        current_conf = get_joint_positions(body, movable_joints)


def trajectory_controller(body, joints, path, **kwargs):
    for target in path:
        for positions in joint_controller(body, joints, target, **kwargs):
            yield positions


# Allow option to sleep rather than yield?
def simulate_controller(controller, max_time=np.inf):
    sim_dt = get_time_step()
    sim_time = 0.0
    for _ in controller:
        if max_time < sim_time:
            break
        step_simulation()
        sim_time += sim_dt
        yield sim_time


def velocity_control_joints(body, joints, velocities):
    #kv = 0.3
    return p.setJointMotorControlArray(body, joints, p.VELOCITY_CONTROL,
                                       targetVelocities=velocities,
                                       physicsClientId=CLIENT)  # ,
    # velocityGains=[kv] * len(joints),)
    # forces=forces)

#####################################


def compute_jacobian(robot, link, positions=None):
    joints = get_movable_joints(robot)
    if positions is None:
        positions = get_joint_positions(robot, joints)
    assert len(joints) == len(positions)
    velocities = [0.0] * len(positions)
    accelerations = [0.0] * len(positions)
    translate, rotate = p.calculateJacobian(robot, link, unit_point(), positions,
                                            velocities, accelerations, physicsClientId=CLIENT)
    #movable_from_joints(robot, joints)
    return list(zip(*translate)), list(zip(*rotate))  # len(joints) x 3


def compute_joint_weights(robot, num=100):
    # http://openrave.org/docs/0.6.6/_modules/openravepy/databases/linkstatistics/#LinkStatisticsModel
    # TODO: use velocities instead
    start_time = time.time()
    joints = get_movable_joints(robot)
    sample_fn = get_sample_fn(robot, joints)
    weighted_jacobian = np.zeros(len(joints))
    links = list(get_links(robot))
    # links = {l for j in joints for l in get_link_descendants(self.robot, j)}
    masses = [get_mass(robot, link) for link in links]  # Volume, AABB volume
    total_mass = sum(masses)
    for _ in range(num):
        conf = sample_fn()
        for mass, link in zip(masses, links):
            translate, rotate = compute_jacobian(robot, link, conf)
            weighted_jacobian += np.array([mass * np.linalg.norm(vec)
                                           for vec in translate]) / total_mass
    weighted_jacobian /= num
    # print(list(weighted_jacobian))
    # print(time.time() - start_time)
    return weighted_jacobian

#####################################


def inverse_kinematics_helper(robot, link, target_pose, null_space=None):
    (target_point, target_quat) = target_pose
    assert target_point is not None
    if null_space is not None:
        assert target_quat is not None
        lower, upper, ranges, rest = null_space

        kinematic_conf = p.calculateInverseKinematics(robot, link, target_point,
                                                      lowerLimits=lower, upperLimits=upper, jointRanges=ranges, restPoses=rest,
                                                      physicsClientId=CLIENT)
    elif target_quat is None:
        #ikSolver = p.IK_DLS or p.IK_SDLS
        kinematic_conf = p.calculateInverseKinematics(robot, link, target_point,
                                                      #lowerLimits=ll, upperLimits=ul, jointRanges=jr, restPoses=rp, jointDamping=jd,
                                                      # solver=ikSolver, maxNumIterations=-1, residualThreshold=-1,
                                                      physicsClientId=CLIENT)
    else:
        kinematic_conf = p.calculateInverseKinematics(
            robot, link, target_point, target_quat, physicsClientId=CLIENT)
    if (kinematic_conf is None) or any(map(math.isnan, kinematic_conf)):
        return None
    return kinematic_conf


def is_pose_close(pose, target_pose, pos_tolerance=1e-3, ori_tolerance=1e-3*np.pi):
    (point, quat) = pose
    (target_point, target_quat) = target_pose
    if (target_point is not None) and not np.allclose(point, target_point, atol=pos_tolerance, rtol=0):
        return False
    if (target_quat is not None) and not np.allclose(quat, target_quat, atol=ori_tolerance, rtol=0):
        # TODO: account for quaternion redundancy
        return False
    return True


def inverse_kinematics(robot, link, target_pose, max_iterations=200, custom_limits={}, **kwargs):
    movable_joints = get_movable_joints(robot)
    for iterations in range(max_iterations):
        # TODO: stop is no progress
        # TODO: stop if collision or invalid joint limits
        kinematic_conf = inverse_kinematics_helper(robot, link, target_pose)
        if kinematic_conf is None:
            return None
        set_joint_positions(robot, movable_joints, kinematic_conf)
        if is_pose_close(get_link_pose(robot, link), target_pose, **kwargs):
            break
    else:
        return None
    lower_limits, upper_limits = get_custom_limits(
        robot, movable_joints, custom_limits)
    if not all_between(lower_limits, kinematic_conf, upper_limits):
        return None
    return kinematic_conf

#####################################


def get_position_waypoints(start_point, direction, quat, step_size=0.01):
    distance = get_length(direction)
    unit_direction = get_unit_vector(direction)
    for t in np.arange(0, distance, step_size):
        point = start_point + t*unit_direction
        yield (point, quat)
    yield (start_point + direction, quat)


def get_quaternion_waypoints(point, start_quat, end_quat, step_size=np.pi/16):
    angle = quat_angle_between(start_quat, end_quat)
    for t in np.arange(0, angle, step_size):
        fraction = t/angle
        quat = p.getQuaternionSlerp(
            start_quat, end_quat, interpolationFraction=fraction)
        #quat = quaternion_slerp(start_quat, end_quat, fraction=fraction)
        yield (point, quat)
    yield (point, end_quat)


def interpolate_poses(pose1, pose2, pos_step_size=0.01, ori_step_size=np.pi/16):
    pos1, quat1 = pose1
    pos2, quat2 = pose2
    num_steps = int(math.ceil(max(get_distance(pos1, pos2)/pos_step_size,
                                  quat_angle_between(quat1, quat2)/ori_step_size)))
    for i in range(num_steps):
        fraction = float(i) / num_steps
        pos = (1-fraction)*np.array(pos1) + fraction*np.array(pos2)
        quat = p.getQuaternionSlerp(
            quat1, quat2, interpolationFraction=fraction)
        #quat = quaternion_slerp(quat1, quat2, fraction=fraction)
        yield (pos, quat)
    yield pose2

# def workspace_trajectory(robot, link, start_point, direction, quat, **kwargs):
#     # TODO: pushing example
#     # TODO: just use current configuration?
#     # TODO: check collisions?
#     # TODO: lower intermediate tolerance
#     traj = []
#     for pose in get_cartesian_waypoints(start_point, direction, quat):
#         conf = inverse_kinematics(robot, link, pose, **kwargs)
#         if conf is None:
#             return None
#         traj.append(conf)
#     return traj

#####################################


NullSpace = namedtuple('Nullspace', ['lower', 'upper', 'range', 'rest'])


def get_null_space(robot, joints, custom_limits={}):
    rest_positions = get_joint_positions(robot, joints)
    lower, upper = get_custom_limits(robot, joints, custom_limits)
    lower = np.maximum(lower, -10*np.ones(len(joints)))
    upper = np.minimum(upper, +10*np.ones(len(joints)))
    joint_ranges = 10*np.ones(len(joints))
    return NullSpace(list(lower), list(upper), list(joint_ranges), list(rest_positions))


def plan_cartesian_motion(robot, first_joint, target_link, waypoint_poses,
                          max_iterations=200, custom_limits={}, **kwargs):
    # TODO: fix stationary joints
    # TODO: pass in set of movable joints and take least common ancestor
    # TODO: update with most recent bullet updates
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/inverse_kinematics.py
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/inverse_kinematics_husky_kuka.py
    # TODO: plan a path without needing to following intermediate waypoints

    lower_limits, upper_limits = get_custom_limits(
        robot, get_movable_joints(robot), custom_limits)
    selected_links = get_link_subtree(
        robot, first_joint)  # TODO: child_link_from_joint?
    selected_movable_joints = prune_fixed_joints(robot, selected_links)
    assert(target_link in selected_links)
    selected_target_link = selected_links.index(target_link)
    sub_robot = clone_body(robot, links=selected_links,
                           visual=False, collision=False)  # TODO: joint limits
    sub_movable_joints = get_movable_joints(sub_robot)
    #null_space = get_null_space(robot, selected_movable_joints, custom_limits=custom_limits)
    null_space = None

    solutions = []
    for target_pose in waypoint_poses:
        for iteration in range(max_iterations):
            sub_kinematic_conf = inverse_kinematics_helper(
                sub_robot, selected_target_link, target_pose, null_space=null_space)
            if sub_kinematic_conf is None:
                remove_body(sub_robot)
                return None
            set_joint_positions(
                sub_robot, sub_movable_joints, sub_kinematic_conf)
            if is_pose_close(get_link_pose(sub_robot, selected_target_link), target_pose, **kwargs):
                set_joint_positions(
                    robot, selected_movable_joints, sub_kinematic_conf)
                kinematic_conf = get_configuration(robot)
                if not all_between(lower_limits, kinematic_conf, upper_limits):
                    #movable_joints = get_movable_joints(robot)
                    # print([(get_joint_name(robot, j), l, v, u) for j, l, v, u in
                    #       zip(movable_joints, lower_limits, kinematic_conf, upper_limits) if not (l <= v <= u)])
                    #print("Limits violated")
                    # wait_for_user()
                    remove_body(sub_robot)
                    return None
                #print("IK iterations:", iteration)
                solutions.append(kinematic_conf)
                break
        else:
            remove_body(sub_robot)
            return None
    remove_body(sub_robot)
    return solutions


def sub_inverse_kinematics(robot, first_joint, target_link, target_pose, **kwargs):
    solutions = plan_cartesian_motion(
        robot, first_joint, target_link, [target_pose], **kwargs)
    if solutions:
        return solutions[0]
    return None

#####################################


def get_lifetime(lifetime):
    if lifetime is None:
        return 0
    return lifetime


def add_debug_parameter():
    # TODO: make a slider that controls the step in the trajectory
    # TODO: could store a list of savers
    #targetVelocitySlider = p.addUserDebugParameter("wheelVelocity", -10, 10, 0)
    #maxForce = p.readUserDebugParameter(maxForceSlider)
    raise NotImplementedError()


def add_text(text, position=(0, 0, 0), color=(0, 0, 0), lifetime=None, parent=-1, parent_link=BASE_LINK):
    return p.addUserDebugText(str(text), textPosition=position, textColorRGB=color[:3],  # textSize=1,
                              lifeTime=get_lifetime(lifetime), parentObjectUniqueId=parent, parentLinkIndex=parent_link,
                              physicsClientId=CLIENT)


def add_line(start, end, color=(0, 0, 0), width=1, lifetime=None, parent=-1, parent_link=BASE_LINK):
    return p.addUserDebugLine(start, end, lineColorRGB=color[:3], lineWidth=width,
                              lifeTime=get_lifetime(lifetime), parentObjectUniqueId=parent, parentLinkIndex=parent_link,
                              physicsClientId=CLIENT)


def remove_debug(debug):
    p.removeUserDebugItem(debug, physicsClientId=CLIENT)


remove_handle = remove_debug


def remove_handles(handles):
    for handle in handles:
        remove_debug(handle)


def remove_all_debug():
    p.removeAllUserDebugItems(physicsClientId=CLIENT)


def add_body_name(body, name=None, **kwargs):
    if name is None:
        name = get_name(body)
    with PoseSaver(body):
        set_pose(body, unit_pose())
        lower, upper = get_aabb(body)
    #position = (0, 0, upper[2])
    position = upper
    # removeUserDebugItem
    return add_text(name, position=position, parent=body, **kwargs)


def add_segments(points, closed=False, **kwargs):
    lines = []
    for v1, v2 in zip(points, points[1:]):
        lines.append(add_line(v1, v2, **kwargs))
    if closed:
        lines.append(add_line(points[-1], points[0], **kwargs))
    return lines


def draw_link_name(body, link=BASE_LINK):
    return add_text(get_link_name(body, link), position=(0, 0.2, 0),
                    parent=body, parent_link=link)


def draw_pose(pose, length=0.1, **kwargs):
    origin_world = tform_point(pose, np.zeros(3))
    handles = []
    for k in range(3):
        axis = np.zeros(3)
        axis[k] = 1
        axis_world = tform_point(pose, length*axis)
        handles.append(
            add_line(origin_world, axis_world, color=axis, **kwargs))
    return handles


def draw_base_limits(limits, z=1e-2, **kwargs):
    lower, upper = limits
    vertices = [(lower[0], lower[1], z), (lower[0], upper[1], z),
                (upper[0], upper[1], z), (upper[0], lower[1], z)]
    return add_segments(vertices, closed=True, **kwargs)


def draw_circle(center, radius, n=24, **kwargs):
    vertices = []
    for i in range(n):
        theta = i*2*math.pi/n
        unit = np.append(unit_from_theta(theta), [0])
        vertices.append(center+radius*unit)
    return add_segments(vertices, closed=True, **kwargs)


def get_aabb_vertices(aabb):
    d = len(aabb[0])
    return [tuple(aabb[i[k]][k] for k in range(d))
            for i in product(range(len(aabb)), repeat=d)]


def draw_aabb(aabb, **kwargs):
    d = len(aabb[0])
    vertices = list(product(range(len(aabb)), repeat=d))
    lines = []
    for i1, i2 in combinations(vertices, 2):
        if sum(i1[k] != i2[k] for k in range(d)) == 1:
            p1 = [aabb[i1[k]][k] for k in range(d)]
            p2 = [aabb[i2[k]][k] for k in range(d)]
            lines.append(add_line(p1, p2, **kwargs))
    return lines


def draw_point(point, size=0.01, **kwargs):
    lines = []
    for i in range(len(point)):
        axis = np.zeros(len(point))
        axis[i] = 1.0
        p1 = np.array(point) - size/2 * axis
        p2 = np.array(point) + size/2 * axis
        lines.append(add_line(p1, p2, **kwargs))
    return lines
    #extent = size * np.ones(len(point)) / 2
    #aabb = np.array(point) - extent, np.array(point) + extent
    # return draw_aabb(aabb, **kwargs)


def get_face_edges(face):
    # return list(combinations(face, 2))
    return list(zip(face, face[1:] + face[:1]))


def draw_mesh(mesh, **kwargs):
    verts, faces = mesh
    lines = []
    for face in faces:
        for i1, i2 in get_face_edges(face):
            lines.append(add_line(verts[i1], verts[i2], **kwargs))
    return lines


def draw_ray(ray, ray_result=None, visible_color=GREEN, occluded_color=RED, **kwargs):
    if ray_result is None:
        return [add_line(ray.start, ray.end, color=visible_color, **kwargs)]
    if ray_result.objectUniqueId == -1:
        hit_position = ray.end
    else:
        hit_position = ray_result.hit_position
    return [
        add_line(ray.start, hit_position, color=visible_color, **kwargs),
        add_line(hit_position, ray.end, color=occluded_color, **kwargs),
    ]

#####################################

# Polygonal surfaces


def create_rectangular_surface(width, length):
    # TODO: unify with rectangular_mesh
    extents = np.array([width, length, 0]) / 2.
    unit_corners = [(-1, -1), (+1, -1), (+1, +1), (-1, +1)]
    return [np.append(c, 0) * extents for c in unit_corners]


def is_point_in_polygon(point, polygon):
    sign = None
    for i in range(len(polygon)):
        v1, v2 = np.array(polygon[i - 1][:2]), np.array(polygon[i][:2])
        delta = v2 - v1
        normal = np.array([-delta[1], delta[0]])
        dist = normal.dot(point[:2] - v1)
        if i == 0:  # TODO: equality?
            sign = np.sign(dist)
        elif np.sign(dist) != sign:
            return False
    return True


def distance_from_segment(x1, y1, x2, y2, x3, y3):  # x3,y3 is the point
    # https://stackoverflow.com/questions/10983872/distance-from-a-point-to-a-polygon
    # https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
    px = x2 - x1
    py = y2 - y1
    norm = px*px + py*py
    u = ((x3 - x1) * px + (y3 - y1) * py) / float(norm)
    if u > 1:
        u = 1
    elif u < 0:
        u = 0
    x = x1 + u * px
    y = y1 + u * py
    dx = x - x3
    dy = y - y3
    return math.sqrt(dx*dx + dy*dy)


def tform_point(affine, point):
    return point_from_pose(multiply(affine, Pose(point=point)))


def apply_affine(affine, points):
    return [tform_point(affine, p) for p in points]


def is_mesh_on_surface(polygon, world_from_surface, mesh, world_from_mesh, epsilon=1e-2):
    surface_from_mesh = multiply(invert(world_from_surface), world_from_mesh)
    points_surface = apply_affine(surface_from_mesh, mesh.vertices)
    min_z = np.min(points_surface[:, 2])
    return (abs(min_z) < epsilon) and \
        all(is_point_in_polygon(p, polygon) for p in points_surface)


def is_point_on_surface(polygon, world_from_surface, point_world):
    [point_surface] = apply_affine(invert(world_from_surface), [point_world])
    return is_point_in_polygon(point_surface, polygon[::-1])


def sample_polygon_tform(polygon, points):
    min_z = np.min(points[:, 2])
    aabb_min = np.min(polygon, axis=0)
    aabb_max = np.max(polygon, axis=0)
    while True:
        x = np.random.uniform(aabb_min[0], aabb_max[0])
        y = np.random.uniform(aabb_min[1], aabb_max[1])
        theta = np.random.uniform(0, 2 * np.pi)
        point = Point(x, y, -min_z)
        quat = Euler(yaw=theta)
        surface_from_origin = Pose(point, quat)
        yield surface_from_origin
        # if all(is_point_in_polygon(p, polygon) for p in apply_affine(surface_from_origin, points)):
        #  yield surface_from_origin


def sample_surface_pose(polygon, world_from_surface, mesh):
    for surface_from_origin in sample_polygon_tform(polygon, mesh.vertices):
        world_from_mesh = multiply(world_from_surface, surface_from_origin)
        if is_mesh_on_surface(polygon, world_from_surface, mesh, world_from_mesh):
            yield world_from_mesh

#####################################

# Sampling edges


def sample_categorical(categories):
    from bisect import bisect
    names = categories.keys()
    cutoffs = np.cumsum([categories[name]
                         for name in names])/sum(categories.values())
    return names[bisect(cutoffs, np.random.random())]


def sample_edge_point(polygon, radius):
    edges = zip(polygon, polygon[-1:] + polygon[:-1])
    edge_weights = {i: max(get_length(v2 - v1) - 2 * radius, 0)
                    for i, (v1, v2) in enumerate(edges)}
    # TODO: fail if no options
    while True:
        index = sample_categorical(edge_weights)
        v1, v2 = edges[index]
        t = np.random.uniform(radius, get_length(v2 - v1) - 2 * radius)
        yield t * get_unit_vector(v2 - v1) + v1


def get_closest_edge_point(polygon, point):
    # TODO: always pick perpendicular to the edge
    edges = zip(polygon, polygon[-1:] + polygon[:-1])
    best = None
    for v1, v2 in edges:
        proj = (v2 - v1)[:2].dot((point - v1)[:2])
        if proj <= 0:
            closest = v1
        elif get_length((v2 - v1)[:2]) <= proj:
            closest = v2
        else:
            closest = proj * get_unit_vector((v2 - v1))
        if (best is None) or (get_length((point - closest)[:2]) < get_length((point - best)[:2])):
            best = closest
    return best


def sample_edge_pose(polygon, world_from_surface, mesh):
    radius = max(get_length(v[:2]) for v in mesh.vertices)
    origin_from_base = Pose(Point(z=p.min(mesh.vertices[:, 2])))
    for point in sample_edge_point(polygon, radius):
        theta = np.random.uniform(0, 2 * np.pi)
        surface_from_origin = Pose(point, Euler(yaw=theta))
        yield multiply(world_from_surface, surface_from_origin, origin_from_base)

#####################################

# Convex Hulls


def convex_hull(points):
    from scipy.spatial import ConvexHull
    # TODO: cKDTree is faster, but KDTree can do all pairs closest
    hull = ConvexHull(points)
    new_indices = {i: ni for ni, i in enumerate(hull.vertices)}
    vertices = hull.points[hull.vertices, :]
    faces = np.vectorize(lambda i: new_indices[i])(hull.simplices)
    return Mesh(vertices.tolist(), faces.tolist())


def convex_signed_area(vertices):
    if len(vertices) < 3:
        return 0.
    vertices = [np.array(v[:2]) for v in vertices]
    segments = safe_zip(vertices, vertices[1:] + vertices[:1])
    return sum(np.cross(v1, v2) for v1, v2 in segments) / 2.


def convex_area(vertices):
    return abs(convex_signed_area(vertices))


def convex_centroid(vertices):
    # TODO: also applies to non-overlapping polygons
    vertices = [np.array(v[:2]) for v in vertices]
    segments = list(safe_zip(vertices, vertices[1:] + vertices[:1]))
    return sum((v1 + v2)*np.cross(v1, v2) for v1, v2 in segments) \
        / (6.*convex_signed_area(vertices))


def mesh_from_points(points):
    vertices, indices = convex_hull(points)
    new_indices = []
    for triplet in indices:
        centroid = np.average(vertices[triplet], axis=0)
        v1, v2, v3 = vertices[triplet]
        normal = np.cross(v3 - v1, v2 - v1)
        if normal.dot(centroid) > 0:
            # if normal.dot(centroid) < 0:
            triplet = triplet[::-1]
        new_indices.append(tuple(triplet))
    return Mesh(vertices.tolist(), new_indices)


def rectangular_mesh(width, length):
    # TODO: 2.5d polygon
    extents = np.array([width, length, 0])/2.
    unit_corners = [(-1, -1), (+1, -1), (+1, +1), (-1, +1)]
    vertices = [np.append(c, [0])*extents for c in unit_corners]
    faces = [(0, 1, 2), (2, 3, 0)]
    return Mesh(vertices, faces)


def tform_mesh(affine, mesh):
    return Mesh(apply_affine(affine, mesh.vertices), mesh.faces)


def grow_polygon(vertices, radius, n=8):
    vertices2d = [vertex[:2] for vertex in vertices]
    if not vertices2d:
        return []
    points = []
    for vertex in convex_hull(vertices2d).vertices:
        points.append(vertex)
        if 0 < radius:
            for theta in np.linspace(0, 2*PI, num=n, endpoint=False):
                points.append(vertex + radius*unit_from_theta(theta))
    return convex_hull(points).vertices

#####################################

# Mesh & Pointcloud Files


def obj_file_from_mesh(mesh, under=True):
    """
    Creates a *.obj mesh string
    :param mesh: tuple of list of vertices and list of faces
    :return: *.obj mesh string
    """
    vertices, faces = mesh
    s = 'g Mesh\n'  # TODO: string writer
    for v in vertices:
        assert(len(v) == 3)
        s += '\nv {}'.format(' '.join(map(str, v)))
    for f in faces:
        # assert(len(f) == 3) # Not necessarily true
        f = [i+1 for i in f]  # Assumes mesh is indexed from zero
        s += '\nf {}'.format(' '.join(map(str, f)))
        if under:
            s += '\nf {}'.format(' '.join(map(str, reversed(f))))
    return s


def get_connected_components(vertices, edges):
    undirected_edges = defaultdict(set)
    for v1, v2 in edges:
        undirected_edges[v1].add(v2)
        undirected_edges[v2].add(v1)
    clusters = []
    processed = set()
    for v0 in vertices:
        if v0 in processed:
            continue
        processed.add(v0)
        cluster = {v0}
        queue = deque([v0])
        while queue:
            v1 = queue.popleft()
            for v2 in (undirected_edges[v1] - processed):
                processed.add(v2)
                cluster.add(v2)
                queue.append(v2)
        if cluster:  # preserves order
            clusters.append(frozenset(cluster))
    return clusters


def read_obj(path, decompose=True):
    mesh = Mesh([], [])
    meshes = {}
    vertices = []
    faces = []
    for line in read(path).split('\n'):
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == 'o':
            name = tokens[1]
            mesh = Mesh([], [])
            meshes[name] = mesh
        elif tokens[0] == 'v':
            vertex = tuple(map(float, tokens[1:4]))
            vertices.append(vertex)
        elif tokens[0] in ('vn', 's'):
            pass
        elif tokens[0] == 'f':
            face = tuple(int(token.split('/')[0]) - 1 for token in tokens[1:])
            faces.append(face)
            mesh.faces.append(face)
    if not decompose:
        return Mesh(vertices, faces)
    # if not meshes:
    #    # TODO: ensure this still works if no objects
    #    meshes[None] = mesh
    #new_meshes = {}
    # TODO: make each triangle a separate object
    for name, mesh in meshes.items():
        indices = sorted({i for face in mesh.faces for i in face})
        mesh.vertices[:] = [vertices[i] for i in indices]
        new_index_from_old = {i2: i1 for i1, i2 in enumerate(indices)}
        mesh.faces[:] = [tuple(new_index_from_old[i1]
                               for i1 in face) for face in mesh.faces]
        #edges = {edge for face in mesh.faces for edge in get_face_edges(face)}
        # for k, cluster in enumerate(get_connected_components(indices, edges)):
        #    new_name = '{}#{}'.format(name, k)
        #    new_indices = sorted(cluster)
        #    new_vertices = [vertices[i] for i in new_indices]
        #    new_index_from_old = {i2: i1 for i1, i2 in enumerate(new_indices)}
        #    new_faces = [tuple(new_index_from_old[i1] for i1 in face)
        #                 for face in mesh.faces if set(face) <= cluster]
        #    new_meshes[new_name] = Mesh(new_vertices, new_faces)
    return meshes


def transform_obj_file(obj_string, transformation):
    new_lines = []
    for line in obj_string.split('\n'):
        tokens = line.split()
        if not tokens or (tokens[0] != 'v'):
            new_lines.append(line)
            continue
        vertex = list(map(float, tokens[1:]))
        transformed_vertex = transformation.dot(vertex)
        new_lines.append('v {}'.format(' '.join(map(str, transformed_vertex))))
    return '\n'.join(new_lines)


def read_mesh_off(path, scale=1.0):
    """
    Reads a *.off mesh file
    :param path: path to the *.off mesh file
    :return: tuple of list of vertices and list of faces
    """
    with open(path) as f:
        assert (f.readline().split()[0] == 'OFF'), 'Not OFF file'
        nv, nf, ne = [int(x) for x in f.readline().split()]
        verts = [tuple(scale * float(v) for v in f.readline().split())
                 for _ in range(nv)]
        faces = [tuple(map(int, f.readline().split()[1:])) for _ in range(nf)]
        return Mesh(verts, faces)


def read_pcd_file(path):
    """
    Reads a *.pcd pointcloud file
    :param path: path to the *.pcd pointcloud file
    :return: list of points
    """
    with open(path) as f:
        data = f.readline().split()
        num_points = 0
        while data[0] != 'DATA':
            if data[0] == 'POINTS':
                num_points = int(data[1])
            data = f.readline().split()
            continue
        return [tuple(map(float, f.readline().split())) for _ in range(num_points)]


def is_collision_free(body_a, link_a_list,
                      body_b=None, link_b_list=None):
    """
    :param body_a: body id of body A
    :param link_a_list: link ids of body A that that of interest
    :param body_b: body id of body B (optional)
    :param link_b_list: link ids of body B that are of interest (optional)
    :return: whether the bodies and links of interest are collision-free
    """
    if body_b is None:
        for link_a in link_a_list:
            contact_pts = p.getContactPoints(
                bodyA=body_a, linkIndexA=link_a)
            if len(contact_pts) > 0:
                return False
    elif link_b_list is None:
        for link_a in link_a_list:
            contact_pts = p.getContactPoints(
                bodyA=body_a, bodyB=body_b, linkIndexA=link_a)
            if len(contact_pts) > 0:
                return False
    else:
        for link_a in link_a_list:
            for link_b in link_b_list:
                contact_pts = p.getContactPoints(
                    bodyA=body_a, bodyB=body_b,
                    linkIndexA=link_a, linkIndexB=link_b)
                if len(contact_pts) > 0:
                    return False
    return True

# TODO: factor out things that don't depend on pybullet

#####################################

# https://github.com/kohterai/OBJ-Parser


"""
def readWrl(filename, name='wrlObj', scale=1.0, color='black'):
    def readOneObj():
        vl = []
        while True:
            line = fl.readline()
            split = line.split(',')
            if len(split) != 2:
                break
            split = split[0].split()
            if len(split) == 3:
                vl.append(np.array([scale*float(x) for x in split]+[1.0]))
            else:
                break
        print '    verts', len(vl),
        verts = np.vstack(vl).T
        while line.split()[0] != 'coordIndex':
            line = fl.readline()
        line = fl.readline()
        faces = []
        while True:
            line = fl.readline()
            split = line.split(',')
            if len(split) > 3:
                faces.append(np.array([int(x) for x in split[:3]]))
            else:
                break
        print 'faces', len(faces)
        return Prim(verts, faces, hu.Pose(0,0,0,0), None,
                    name=name+str(len(prims)))

    fl = open(filename)
    assert fl.readline().split()[0] == '#VRML', 'Not VRML file?'
    prims = []
    while True:
        line = fl.readline()
        if not line: break
        split = line.split()
        if not split or split[0] != 'point':
            continue
        else:
            print 'Object', len(prims)
            prims.append(readOneObj())
    # Have one "part" so that shadows are simpler
    part = Shape(prims, None, name=name+'_part')
    # Keep color only in top entry.
    return Shape([part], None, name=name, color=color)
"""
