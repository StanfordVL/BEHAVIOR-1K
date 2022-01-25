import atexit
import json
import multiprocessing
import os
import sys
import time
import traceback
import uuid

import bddl
import numpy as np

from bddl.parsing import construct_full_bddl
from bddl.utils import UncontrolledCategoryError, UnsupportedPredicateError
from flask import Flask, Response, render_template, request
from flask_apscheduler import APScheduler
from flask_cors import CORS

import igibson
from igibson.envs.igibson_env import iGibsonEnv
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.simulator import Simulator
from igibson.utils.utils import parse_config, restoreState

interactive = True
NUM_REQUIRED_SUCCESSFUL_SCENES = 3
PERIODIC_CLEANUP_TASK_ID = "interval-task-id"
PERIODIC_CLEANUP_INTERVAL = 3600


class ProcessPyEnvironment(object):
    """Step a single env in a separate process for lock free paralellism."""

    # Message types for communication via the pipe.
    _READY = 1
    _ACCESS = 2
    _CALL = 3
    _RESULT = 4
    _EXCEPTION = 5
    _CLOSE = 6

    def __init__(self, env_constructor):
        self._env_constructor = env_constructor

        self.last_active_time = time.time()

    def start(self):
        """Start the process."""
        self._conn, conn = multiprocessing.Pipe()
        self._process = multiprocessing.Process(target=self._worker, args=(conn, self._env_constructor))
        atexit.register(self.close)
        self.last_active_time = time.time()
        self._process.start()
        result = self._conn.recv()
        self.last_active_time = time.time()
        if isinstance(result, Exception):
            self._conn.close()
            self._process.join(5)
            raise result
        assert result is self._READY, result

    def __getattr__(self, name):
        """Request an attribute from the environment.
        Note that this involves communication with the external process, so it can
        be slow.

        :param name: attribute to access.
        :return: value of the attribute.
        """
        print("gettinng", name)
        self._conn.send((self._ACCESS, name))
        return self._receive()

    def call(self, name, *args, **kwargs):
        """Asynchronously call a method of the external environment.

        :param name: name of the method to call.
        :param args: positional arguments to forward to the method.
        :param kwargs: keyword arguments to forward to the method.
        :return: promise object that blocks and provides the return value when called.
        """
        payload = name, args, kwargs
        self._conn.send((self._CALL, payload))
        return self._receive

    def close(self):
        """Send a close message to the external process and join it."""
        try:
            self._conn.send((self._CLOSE, None))
            self._conn.close()
        except IOError:
            # The connection was already closed.
            pass
        self._process.join(5)

    def step(self, action, blocking=True):
        """Step the environment.

        :param action: the action to apply to the environment.
        :param blocking: whether to wait for the result.
        :return: (next_obs, reward, done, info) tuple when blocking, otherwise callable that returns that tuple
        """
        promise = self.call("step", action)
        if blocking:
            return promise()
        else:
            return promise

    def reset(self, blocking=True):
        """Reset the environment.

        :param blocking: whether to wait for the result.
        :return: next_obs when blocking, otherwise callable that returns next_obs
        """
        promise = self.call("reset")
        if blocking:
            return promise()
        else:
            return promise

    def sample(self, behavior_activity, bddl, blocking=True):
        """Run a sampling in the environment

        :param bddl (str): the bddl being sampled in the environment
        :param blocking (bool): whether to wait for the result
        :return (bool, str): (success, feedback) from the sampling process
        """
        self.last_active_time = time.time()
        promise = self.call("sample", behavior_activity, bddl)
        self.last_active_time = time.time()
        if blocking:
            return promise()
        else:
            return promise

    def _receive(self):
        """Wait for a message from the worker process and return its payload.

        :raise Exception: an exception was raised inside the worker process.
        :raise KeyError: the reveived message is of an unknown type.

        :return: payload object of the message.
        """
        message, payload = self._conn.recv()
        # Re-raise exceptions in the main process.
        if message == self._EXCEPTION:
            stacktrace = payload
            raise Exception(stacktrace)
        if message == self._RESULT:
            return payload
        self.close()
        raise KeyError("Received message of unexpected type {}".format(message))

    def _worker(self, conn, env_constructor):
        """The process waits for actions and sends back environment results.

        :param conn: connection for communication to the main process.
        :param env_constructor: env_constructor for the OpenAI Gym environment.


        :raise KeyError: when receiving a message of unknown type.
        """
        try:
            np.random.seed()
            env = env_constructor()
            conn.send(self._READY)  # Ready.
            while True:
                try:
                    # Only block for short times to have keyboard exceptions be raised.
                    if not conn.poll(0.1):
                        continue
                    message, payload = conn.recv()
                except (EOFError, KeyboardInterrupt):
                    break
                if message == self._ACCESS:
                    name = payload
                    result = getattr(env, name)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CALL:
                    name, args, kwargs = payload
                    if name == "step" or name == "reset" or name == "sample":
                        result = getattr(env, name)(*args, **kwargs)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CLOSE:
                    getattr(env, "close")()
                    assert payload is None
                    break
                raise KeyError("Received message of unknown type {}".format(message))
        except Exception:  # pylint: disable=broad-except
            etype, evalue, tb = sys.exc_info()
            stacktrace = "".join(traceback.format_exception(etype, evalue, tb))
            message = "Error in environment process: {}".format(stacktrace)
            conn.send((self._EXCEPTION, stacktrace))
        finally:
            conn.close()


class ToyEnv(object):
    def __init__(self):
        config = parse_config(os.path.join(igibson.example_config_path, "turtlebot_demo.yaml"))
        hdr_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_02.hdr")
        hdr_texture2 = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_03.hdr")
        light_modulation_map_filename = os.path.join(
            igibson.ig_dataset_path, "scenes", "Rs_int", "layout", "floor_lighttype_0.png"
        )
        background_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "urban_street_01.jpg")

        settings = MeshRendererSettings(enable_shadow=False, enable_pbr=False)

        self.s = Simulator(mode="headless", image_width=400, image_height=400, rendering_settings=settings)
        scene = StaticIndoorScene("Rs")
        self.s.import_scene(scene)

    def step(self, a):
        self.s.step()
        frame = self.s.renderer.render_robot_cameras(modes=("rgb"))[0]
        return frame

    def close(self):
        self.s.disconnect()


class ToyEnvInt(object):
    def __init__(self, scene="Rs_int"):
        config_file = os.path.join(igibson.example_config_path, "behavior_vr.yaml")
        env_config = parse_config(config_file)
        env_config["scene_id"] = scene
        env_config["task"] = "trivial"
        env_config["task_id"] = 0
        env_config["online_sampling"] = True
        env_config["load_clutter"] = False
        settings = MeshRendererSettings(texture_scale=0.01)
        self.env = iGibsonEnv(config_file=env_config, mode="headless", rendering_settings=settings)
        self.state_id = p.saveState()
        self.num_body_ids = p.getNumBodies()
        self.num_particle_systems = len(self.env.simulator.particle_systems)

    def step(self, a):
        pass

    def restore_scene(self):
        for sim_obj in self.env.task.newly_added_objects:
            self.env.scene.remove_object(sim_obj)

        self.env.simulator.particle_systems = self.env.simulator.particle_systems[: self.num_particle_systems]

        for body_id in range(self.num_body_ids, p.getNumBodies()):
            p.removeBody(body_id)

        restoreState(self.state_id)

    def sample(self, behavior_activity, bddl):
        try:
            self.env.task.update_problem(behavior_activity, 0, predefined_problem=bddl)
        except UncontrolledCategoryError:
            accept_scene = False
            feedback = "Goal state has uncontrolled categories."
            return accept_scene, feedback
        except UnsupportedPredicateError as e:
            accept_scene = False
            feedback = f"We don't yet support the [{e.predicate}] adjective for any objects. We will soon!"
            return accept_scene, feedback
        except AssertionError as message:
            accept_scene = False
            if message == "No ground goal options":
                feedback = (
                    "The goal conditions are logically impossible (there is no solution). Check for a contradiction (e.g. asking for the floor to be stained and not stained at the same time).",
                )
            else:
                feedback = "Let Sanjana know there was an indeterminate assertion error during problem update."
            return accept_scene, feedback

        try:
            accept_scene, feedback = self.env.task.initialize(self.env)
        except AssertionError as message:
            accept_scene = False
            feedback = f"Let Sanjana know there was an assertion error during scene checking/sampling: {str(message)}"
            self.restore_scene()
            return accept_scene, feedback

        self.restore_scene()
        return accept_scene, feedback

    def close(self):
        self.env.close()


class iGFlask(Flask):
    def __init__(self, args, **kwargs):
        super(iGFlask, self).__init__(args, **kwargs)
        self.action = {}
        self.envs = {}
        self.envs_inception_time = {}
        self.envs_last_use_time = {}

    def cleanup(self):
        # TODO change this to allow people to make the conditions
        for k, v in self.envs_inception_time.items():
            if time.time() - v > 200:
                # clean up an old environment
                self.stop_env(k)

    def prepare_env(self, uuid, scene):
        # self.cleanup()
        def env_constructor():
            if interactive:
                return ToyEnvInt(scene=scene)
            else:
                return ToyEnv()

        self.envs[uuid] = ProcessPyEnvironment(env_constructor)
        self.envs[uuid].start()
        self.envs_inception_time[uuid] = time.time()
        self.envs_last_use_time[uuid] = time.time()

    def periodic_cleanup(self, periodic_cleanup_interval=PERIODIC_CLEANUP_INTERVAL):
        stale_uids = []
        for uid, env in self.envs.items():
            last_active_time = env.last_active_time
            print(f"uuid: {uid}, time since last use: {int(time.time() - last_active_time)}")
            if time.time() - last_active_time > periodic_cleanup_interval:
                print("stale uuid:", uid)
                stale_uids.append(uid)
        for stale_uid in stale_uids:
            self.stop_env(stale_uid)

    # TODO how to update envs_last_use_time? Maybe make it a field in
    #   the ToyEnvInt or ProcessPyEnv and update it while some call is running?

    def stop_env(self, uuid):
        self.envs[uuid].close()
        del self.envs[uuid]
        del self.envs_inception_time[uuid]
        del self.envs_last_use_time[uuid]


app = iGFlask(__name__)
CORS(app)
scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()


########### REQUEST HANDLERS ###########


@app.route("/", methods=["POST"])
def index():
    id = uuid.uuid4()
    return render_template("index.html", uuid=id)


@app.route("/setup", methods=["POST"])
def setup():
    """Set up the three environments when requested by annotation React app"""
    scenes = json.loads(request.data)  # TODO check what this looks like
    ids = [str(uuid.uuid4()) for __ in range(len(scenes))]
    scenes_ids = list(zip(scenes, ids))
    for scene, unique_id in scenes_ids:
        # TODO uncomment when basic infra is done
        app.prepare_env(unique_id, scene)
        print(f"Instantiated {scene} with uuid {unique_id}")

    return Response(json.dumps({"scenes_ids": scenes_ids}))

    # TODO need to send uuids to the upcoming POSTs somehow, I guess in the response


@app.route("/check_sampling", methods=["POST"])
def check_sampling():
    """Check BDDL sent by React app in all three relevant scenes

    :return (Response): response indicating success of sampling in all three
                         scenes, feedback given from each
    """
    # Prepare data
    data = json.loads(request.data)
    behavior_activity = data["activityName"]
    init_state = data["initialConditions"]
    goal_state = data["goalConditions"]
    object_list = data["objectList"]
    # bddl = init_state + goal_state + object_list        # TODO fix using existing utils
    bddl = construct_full_bddl(behavior_activity, "feasibility_check", object_list, init_state, goal_state)
    scenes_ids = data["scenes_ids"]
    ids = [unique_id for scene, unique_id in scenes_ids]
    scenes = [scene for scene, unique_id in scenes_ids]

    # Try sampling
    num_successful_scenes = 0
    feedback_instances = []
    new_scenes_ids = scenes_ids[:]
    for i, info in enumerate(scenes_ids):
        scene, unique_id = info
        # If this scene's environment has been cleaned, make a new one
        if unique_id not in app.envs:
            new_unique_id = str(uuid.uuid4())
            new_scenes_ids[i] = (scene, new_unique_id)
            # TODO is this asynchronous with the sample call?
            app.prepare_env(new_unique_id, scene)
            print(f"Instantiated {scene} with {new_unique_id} because previous version was cleaned up")
        else:
            new_unique_id = unique_id
        success, feedback = app.envs[new_unique_id].sample(behavior_activity, bddl)
        if success:
            num_successful_scenes += 1
        feedback_instances.append(feedback)

    success = num_successful_scenes >= min(NUM_REQUIRED_SUCCESSFUL_SCENES, len(ids))
    full_feedback = feedback_instances

    return Response(json.dumps({"success": success, "feedback": full_feedback, "scenes_ids": new_scenes_ids}))


@app.route("/teardown", methods=["POST"])
def teardown():
    """Tear down the three environments created for this instance of the React app"""
    data = json.loads(request.data)
    scenes_ids = data["scenes_ids"]
    for scene, unique_id in scenes_ids:
        if unique_id in app.envs:
            app.stop_env(unique_id)
            print(f"uuid {unique_id} stopped")

    # TODO need anything else?
    return Response(json.dumps({"success": True}))


########### PERIODIC CLEANUP ###########


def periodic_cleanup():
    app.periodic_cleanup()


scheduler.add_job(id=PERIODIC_CLEANUP_TASK_ID, func=periodic_cleanup, seconds=5, trigger="interval")


if __name__ == "__main__":
    port = int(sys.argv[1])
    # app.run(host="0.0.0.0", port=port, debug=True)
    app.run(host="0.0.0.0", port=port)
