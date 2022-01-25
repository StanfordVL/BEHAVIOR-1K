import atexit
import base64
import binascii
import multiprocessing
import os
import sys
import time
import traceback
import uuid
from io import BytesIO

import numpy as np
from flask import Flask, Response, render_template, request
from PIL import Image

import igibson
from igibson.objects.ycb_object import YCBObject
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.robots.fetch import Fetch
from igibson.robots.turtlebot import Turtlebot
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.scenes.interactive_traversable_scene import InteractiveIndoorScene
from igibson.simulator import Simulator
from igibson.utils.utils import parse_config

interactive = True


def pil_image_to_base64(pil_image):
    buf = BytesIO()
    pil_image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue())


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

    def start(self):
        """Start the process."""
        self._conn, conn = multiprocessing.Pipe()
        self._process = multiprocessing.Process(target=self._worker, args=(conn, self._env_constructor))
        atexit.register(self.close)
        self._process.start()
        result = self._conn.recv()
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
                    if name == "step" or name == "reset":
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
    """
    ToyEnv is an example environment that wraps around the simulator. It doesn't follow
    OpenAI gym interface, and only has step and close function. It works with static
    mesh scenes.
    """

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
        self.robot = Turtlebot(config)
        self.s.import_robot(self.robot)

        for _ in range(5):
            obj = YCBObject("003_cracker_box")
            self.s.import_object(obj)
            obj.set_position_orientation(np.random.uniform(low=0, high=2, size=3), [0, 0, 0, 1])
        print(self.s.renderer.instances)

    def step(self, a):
        # run simulation for one step and get an rgb frame
        self.robot.apply_action(a)
        self.s.step()
        frame = self.s.renderer.render_robot_cameras(modes=("rgb"))[0]
        return frame

    def close(self):
        # tear down the simulation
        self.s.disconnect()


class ToyEnvInt(object):
    """
    Same with ToyEnv, but works with interactive scenes.
    """

    def __init__(self, robot="turtlebot", scene="Rs_int"):
        config = parse_config(os.path.join(igibson.example_config_path, "turtlebot_demo.yaml"))
        hdr_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_02.hdr")
        hdr_texture2 = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_03.hdr")
        light_modulation_map_filename = os.path.join(
            igibson.ig_dataset_path, "scenes", "Rs_int", "layout", "floor_lighttype_0.png"
        )
        background_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "urban_street_01.jpg")

        scene = InteractiveIndoorScene(scene, texture_randomization=False, object_randomization=False)
        # scene._set_first_n_objects(5)
        scene.open_all_doors()

        settings = MeshRendererSettings(
            env_texture_filename=hdr_texture,
            env_texture_filename2=hdr_texture2,
            env_texture_filename3=background_texture,
            light_modulation_map_filename=light_modulation_map_filename,
            enable_shadow=True,
            msaa=True,
            light_dimming_factor=1.0,
            optimized=True,
        )

        self.s = Simulator(mode="headless", image_width=400, image_height=400, rendering_settings=settings)
        self.s.import_scene(scene)

        if robot == "turtlebot":
            self.robot = Turtlebot(config)
        else:
            self.robot = Fetch(config)

        self.s.import_robot(self.robot)

        for _ in range(5):
            obj = YCBObject("003_cracker_box")
            self.s.import_object(obj)
            obj.set_position_orientation(np.random.uniform(low=0, high=2, size=3), [0, 0, 0, 1])
        print(self.s.renderer.instances)

    def step(self, a):
        # run simulation for one step and get an rgb frame
        action = np.zeros(self.robot.action_space.shape)
        # for some reason, the wheel velocity of Fetch Robot needs to be reversed.
        if isinstance(self.robot, Turtlebot):
            action[0] = a[0]
            action[1] = a[1]
        else:
            action[1] = a[0]
            action[0] = a[1]

        self.robot.apply_action(action)
        self.s.step()
        frame = self.s.renderer.render_robot_cameras(modes=("rgb"))[0]
        return frame

    def close(self):
        # tear down the simulation
        self.s.disconnect()


class iGFlask(Flask):
    """
    iGFlask is a Flask app that handles environment creation and teardown.
    """

    def __init__(self, args, **kwargs):
        super(iGFlask, self).__init__(args, **kwargs)
        self.action = {}  # map uuid to input action
        self.envs = {}  # map uuid to environment instance
        self.envs_inception_time = {}  # map uuid to environment start time

    def cleanup(self):
        """
        Routine clean up, this function tries to find any environment that idles for more
        than 200s and stops it.
        """
        print(self.envs)
        for k, v in self.envs_inception_time.items():
            if time.time() - v > 200:
                # clean up an old environment
                self.stop_env(k)

    def prepare_env(self, uuid, robot, scene):
        """
        This function creates an Env (ToyEnv or ToyEnvInt) in a subprocess.
        """
        self.cleanup()

        def env_constructor():
            if interactive:
                return ToyEnvInt(robot=robot, scene=scene)
            else:
                return ToyEnv()

        self.envs[uuid] = ProcessPyEnvironment(env_constructor)
        self.envs[uuid].start()
        self.envs_inception_time[uuid] = time.time()

    def stop_env(self, uuid):
        # stop an environment (ToyEnv or ToyEnvInt) that lives in a subprocess.
        self.envs[uuid].close()
        del self.envs[uuid]
        del self.envs_inception_time[uuid]


app = iGFlask(__name__)


@app.route("/")
def index():
    id = uuid.uuid4()
    return render_template("index.html", uuid=id)


@app.route("/demo")
def demo():
    args = request.args
    id = uuid.uuid4()
    robot = args["robot"]
    scene = args["scene"]
    return render_template("demo.html", uuid=id, robot=robot, scene=scene)


"""
gen is a utility function that generate an image based on user id
    and user input (robot and scene), and send it back to the user.
    The images are played quickly so it becomes a video.
"""


def gen(app, unique_id, robot, scene):
    image = np.array(Image.open("templates/loading.jpg").resize((400, 400))).astype(np.uint8)
    loading_frame = pil_image_to_base64(Image.fromarray(image))
    loading_frame = binascii.a2b_base64(loading_frame)

    image = np.array(Image.open("templates/waiting.jpg").resize((400, 400))).astype(np.uint8)
    waiting_frame = pil_image_to_base64(Image.fromarray(image))
    waiting_frame = binascii.a2b_base64(waiting_frame)

    image = np.array(Image.open("templates/finished.jpg").resize((400, 400))).astype(np.uint8)
    finished_frame = pil_image_to_base64(Image.fromarray(image))
    finished_frame = binascii.a2b_base64(finished_frame)
    id = unique_id
    if len(app.envs) < 3:
        # if number of envs is smaller than 3, then create an environment and provide to the user
        for i in range(5):
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + loading_frame + b"\r\n\r\n")
        app.prepare_env(id, robot, scene)
        try:
            start_time = time.time()
            if interactive:
                timeout = 200
            else:
                timeout = 30
            while time.time() - start_time < timeout:
                # If the environment is still valid (before it should be destroyed)
                # generate a frame from the Env and supply to the user.
                frame = app.envs[id].step(app.action[id])
                frame = (frame[:, :, :3] * 255).astype(np.uint8)
                frame = pil_image_to_base64(Image.fromarray(frame))
                frame = binascii.a2b_base64(frame)
                yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n\r\n")
        except:
            pass
        finally:
            # if timeouts, stop the environment, and show an text prompt image telling
            # the user the simulation has finished
            app.stop_env(id)
        for i in range(5):
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + finished_frame + b"\r\n\r\n")
    else:
        # If number of envs is >= 3, then let the user wait
        for i in range(5):
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + waiting_frame + b"\r\n\r\n")


@app.route("/video_feed", methods=["POST", "GET"])
def video_feed():
    unique_id = request.args["uuid"]
    if "robot" in request.args.keys():
        robot = request.args["robot"]
    if "scene" in request.args.keys():
        scene = request.args["scene"]
    print(unique_id)
    if request.method == "POST":
        key = request.args["key"]
        if key == "w":
            app.action[unique_id] = [1, 1]
        if key == "s":
            app.action[unique_id] = [-1, -1]
        if key == "d":
            app.action[unique_id] = [0.3, -0.3]
        if key == "a":
            app.action[unique_id] = [-0.3, 0.3]
        if key == "f":
            app.action[unique_id] = [0, 0]
        return ""
    else:
        app.action[unique_id] = [0, 0]
        return Response(gen(app, unique_id, robot, scene), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    port = int(sys.argv[1])
    app.run(host="0.0.0.0", port=port)
