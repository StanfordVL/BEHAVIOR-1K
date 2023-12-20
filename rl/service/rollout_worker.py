from concurrent.futures import ThreadPoolExecutor
import inspect
from multiprocessing import Queue
import pickle
import threading
from typing import Optional
import grpc
import environment_pb2
import environment_pb2_grpc

import gymnasium as gym

def _unwrap_wrapper(env: gym.Env, wrapper_class: str) -> Optional[gym.Wrapper]:
    """
    Retrieve a ``VecEnvWrapper`` object by recursively searching.

    :param env: Environment to unwrap
    :param wrapper_class: Wrapper to look for
    :return: Environment unwrapped till ``wrapper_class`` if it has been wrapped with it
    """
    env_tmp = env
    while isinstance(env_tmp, gym.Wrapper):
        this_wrapper_class = type(env_tmp)
        class_str = ".".join([this_wrapper_class.__module__, this_wrapper_class.__qualname__])
        if class_str == wrapper_class:
            return env_tmp
        env_tmp = env_tmp.env
    return None


def _is_wrapped(env: gym.Env, wrapper_class: str) -> bool:
    """
    Check if a given environment has been wrapped with a given wrapper.

    :param env: Environment to check
    :param wrapper_class: Wrapper class to look for
    :return: True if environment has been wrapped with ``wrapper_class``.
    """
    return _unwrap_wrapper(env, wrapper_class) is not None


class EnvironmentServicerReal(environment_pb2_grpc.EnvironmentService):
    def __init__(self, env) -> None:
        self._env = env

    @property
    def env(self):
        # This function can only be used on the main thread
        assert threading.current_thread() == threading.main_thread(), (
            "You must only call `env` from the main thread."
        )
        return self._env

    def Step(self, request, unused_context):
        action = pickle.loads(request.action)
        print("Action:", action, type(action))
        assert self.env.action_space.contains(action), "Action must be contained in action space."
        observation, reward, terminated, truncated, info = self.env.step(action)

        return environment_pb2.StepResponse(
            observation=pickle.dumps(observation),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=pickle.dumps(info)
        )

    def Reset(self, request, unused_context):
        seed = request.seed if request.HasField("seed") else None
        maybe_options = {"options": pickle.loads(request.options)} if request.options else {}
        observation, reset_info = self.env.reset(seed=seed, **maybe_options)
        
        return environment_pb2.ResetResponse(
            observation=pickle.dumps(observation),
            reset_info=pickle.dumps(reset_info)
        )

    def Render(self, request, unused_context):
        image = self.env.render()
        return environment_pb2.RenderResponse(render_data=pickle.dumps(image))

    def Close(self, request, unused_context):
        self.env.close()
        return environment_pb2.CloseResponse()

    def GetSpaces(self, request, unused_context):
        return environment_pb2.GetSpacesResponse(
            observation_space=pickle.dumps(self.env.observation_space),
            action_space=pickle.dumps(self.env.action_space)
        )

    def EnvMethod(self, request, unused_context):
        method_name = request.method_name
        args, kwargs = pickle.loads(request.arguments)
        method = getattr(self.env, method_name)
        result = method(*args, **kwargs)
        return environment_pb2.EnvMethodResponse(result=pickle.dumps(result))

    def GetAttr(self, request, unused_context):
        attr = request.attribute_name
        result = getattr(self.env, attr)
        return environment_pb2.GetAttrResponse(attribute_value=pickle.dumps(result))

    def SetAttr(self, request, unused_context):
        attr = request.attribute_name
        val = pickle.loads(request.attribute_value)
        setattr(self.env, attr, val)
        return environment_pb2.SetAttrResponse()

    def IsWrapped(self, request, unused_context):
        is_it_wrapped = _is_wrapped(self.env, request.wrapper_type)
        return environment_pb2.IsWrappedResponse(is_wrapped=is_it_wrapped)
    
class EnvironmentServicerOnThread(environment_pb2_grpc.EnvironmentService):
    def __init__(self, request_queue, response_queue) -> None:
        self.request_queue : Queue = request_queue
        self.response_queue : Queue = response_queue

    def Step(self, request, unused_context):
        self.request_queue.put(request)
        resp = self.response_queue.get()
        assert isinstance(resp, environment_pb2.StepResponse)
        return resp

    def Reset(self, request, unused_context):
        self.request_queue.put(request)
        resp = self.response_queue.get()
        assert isinstance(resp, environment_pb2.ResetResponse)
        return resp

    def Render(self, request, unused_context):
        self.request_queue.put(request)
        resp = self.response_queue.get()
        assert isinstance(resp, environment_pb2.RenderResponse)
        return resp

    def Close(self, request, unused_context):
        self.request_queue.put(request)
        resp = self.response_queue.get()
        assert isinstance(resp, environment_pb2.CloseResponse)
        return resp

    def GetSpaces(self, request, unused_context):
        self.request_queue.put(request)
        resp = self.response_queue.get()
        assert isinstance(resp, environment_pb2.GetSpacesResponse)
        return resp

    def EnvMethod(self, request, unused_context):
        self.request_queue.put(request)
        resp = self.response_queue.get()
        assert isinstance(resp, environment_pb2.EnvMethodResponse)
        return resp

    def GetAttr(self, request, unused_context):
        self.request_queue.put(request)
        resp = self.response_queue.get()
        assert isinstance(resp, environment_pb2.GetAttrResponse)
        return resp

    def SetAttr(self, request, unused_context):
        self.request_queue.put(request)
        resp = self.response_queue.get()
        assert isinstance(resp, environment_pb2.SetAttrResponse)
        return resp

    def IsWrapped(self, request, unused_context):
        self.request_queue.put(request)
        resp = self.response_queue.get()
        assert isinstance(resp, environment_pb2.IsWrappedResponse)
        return resp
    
def register(local_addr, learner_addr):
    channel = grpc.insecure_channel(learner_addr)
    stub = environment_pb2_grpc.EnvironmentRegistrationServiceStub(channel)
    request = environment_pb2.RegisterEnvironmentRequest(
        ip=local_addr.split(":")[0],
        port=int(local_addr.split(":")[1])
    )
    response = stub.RegisterEnvironment(request)
    return response.success


def serve(env, local_addr, learner_addr):
    request_queue = Queue()
    response_queue = Queue()

    server = grpc.server(ThreadPoolExecutor(max_workers=4))
    environment_pb2_grpc.add_EnvironmentServiceServicer_to_server(
        EnvironmentServicerOnThread(request_queue, response_queue), server
    )
    server.add_insecure_port(local_addr)
    server.start()
    print("Launched env server.")

    # With our server started, let's get registered.
    print(f"Registering env {local_addr} with learner {learner_addr}.")
    success = register(local_addr, learner_addr)
    assert success, "Failed to register environment with learner."
    print("Registered successfully.")

    # Repeatedly feed commands from queue into the servicer
    servicer = EnvironmentServicerReal(env)
    while True:
        req = request_queue.get()
        assert request_queue.empty(), "Request queue should be empty."
        assert response_queue.empty(), "Response queue should be empty."

        if isinstance(req, environment_pb2.StepRequest):
            resp = servicer.Step(req, None)
        elif isinstance(req, environment_pb2.ResetRequest):
            resp = servicer.Reset(req, None)
        elif isinstance(req, environment_pb2.RenderRequest):
            resp = servicer.Render(req, None)
        elif isinstance(req, environment_pb2.CloseRequest):
            resp = servicer.Close(req, None)
        elif isinstance(req, environment_pb2.GetSpacesRequest):
            resp = servicer.GetSpaces(req, None)
        elif isinstance(req, environment_pb2.EnvMethodRequest):
            resp = servicer.EnvMethod(req, None)
        elif isinstance(req, environment_pb2.GetAttrRequest):
            resp = servicer.GetAttr(req, None)
        elif isinstance(req, environment_pb2.SetAttrRequest):
            resp = servicer.SetAttr(req, None)
        elif isinstance(req, environment_pb2.IsWrappedRequest):
            resp = servicer.IsWrapped(req, None)
        else:
            raise ValueError(f"Unknown request type: {type(req)}")
        
        response_queue.put(resp)
