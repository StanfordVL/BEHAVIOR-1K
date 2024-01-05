import pickle
import grpc
from telegym.protos import environment_pb2
from telegym.protos import environment_pb2_grpc

import gymnasium as gym

class GRPCClientEnv(gym.Env):
  def __init__(self, url):
    super().__init__()

    self.url = url
    self.channel = grpc.insecure_channel(url)
    self._stub = environment_pb2_grpc.EnvironmentServiceStub(self.channel)
    self.observation_space, self.action_space = self._get_spaces()

  @property
  def stub(self):
    # TODO: Reestablish connection if it's down.
    return self._stub

  def step(self, action):
    response = self.stub.Step(environment_pb2.StepRequest(action=pickle.dumps(action)))
    obs = pickle.loads(response.observation)
    reward = response.reward
    truncated = response.truncated
    terminated = response.terminated
    info = pickle.loads(response.info)
    done = terminated or truncated
    info["TimeLimit.truncated"] = truncated and not terminated

    reset_infos = {}
    if done:
      info["terminal_observation"] = obs
      obs, reset_infos = self.reset()

    return obs, reward, done, info, reset_infos
  
  def reset(self, seed=None, options=None):
    request = environment_pb2.ResetRequest()
    if seed is not None:
      request.seed = seed
    if options is not None:
      request.options = pickle.dumps(options)
    response = self.stub.Reset(request)
    return pickle.loads(response.observation), pickle.loads(response.reset_info)
  
  def render(self):
    request = environment_pb2.RenderRequest()
    response = self.stub.Render(request)
    return pickle.loads(response.render_data)
  
  def close(self):
    request = environment_pb2.CloseRequest()
    self.stub.Close(request)

  def _get_spaces(self):  
    request = environment_pb2.GetSpacesRequest()
    response = self.stub.GetSpaces(request)
    return pickle.loads(response.observation_space), pickle.loads(response.action_space)
  
  def env_method(self, method_name, *args, **kwargs):
    request = environment_pb2.EnvMethodRequest(
      method_name=method_name,
      arguments=pickle.dumps((args, kwargs))
    )
    response = self.stub.EnvMethod(request)
    return pickle.loads(response.result)
  
  def get_attr(self, attr_name):
    request = environment_pb2.GetAttrRequest(attribute_name=attr_name)
    response = self.stub.GetAttr(request)
    return pickle.loads(response.attribute_value)
  
  def set_attr(self, attr_name, value):
    request = environment_pb2.SetAttrRequest(
      attribute_name=attr_name,
      attribute_value=pickle.dumps(value)
    )
    self.stub.SetAttr(request)

  def is_wrapped(self, wrapper_type):
    wrapper_type_str = ".".join([wrapper_type.__module__, wrapper_type.__qualname__])
    request = environment_pb2.IsWrappedRequest(wrapper_type=wrapper_type_str)
    response = self.stub.IsWrapped(request)
    return response.is_wrapped