import asyncio
from concurrent.futures import ThreadPoolExecutor
import inspect
import pickle
from typing import Optional
import grpc
import environment_pb2
import environment_pb2_grpc

import gymnasium as gym
import gymnasium.wrappers

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

env_closed = asyncio.Event()

class EnvironmentServicer(environment_pb2_grpc.EnvironmentService):
    def __init__(self, env) -> None:
        self.env = env

    def Step(self, request, unused_context):
        action = pickle.loads(request.action)
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
        env_closed.set()
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
    
def register(local_addr, learner_addr):
    channel = grpc.insecure_channel(learner_addr)
    stub = environment_pb2_grpc.EnvironmentRegistrationServiceStub(channel)
    request = environment_pb2.RegisterEnvironmentRequest(
        ip=local_addr.split(":")[0],
        port=int(local_addr.split(":")[1])
    )
    response = stub.RegisterEnvironment(request)
    return response.success

async def serve(env, local_addr, learner_addr):
    server = grpc.aio.server()
    environment_pb2_grpc.add_EnvironmentServiceServicer_to_server(
        EnvironmentServicer(env), server
    )
    server.add_insecure_port(local_addr)
    await server.start()

    # With our server started, let's get registered.
    executor = ThreadPoolExecutor()
    success = await asyncio.get_running_loop().run_in_executor(executor, register, local_addr, learner_addr)
    assert success, "Failed to register environment with learner."
     
    # Return when either is true
    _, pending = await asyncio.wait(
        [env_closed.wait(), server.wait_for_termination()], return_when=asyncio.FIRST_COMPLETED)
    [t.cancel() for t in pending]