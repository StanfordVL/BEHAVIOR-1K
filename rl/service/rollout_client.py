import pickle
import grpc
import environment_pb2
import environment_pb2_grpc

import gymnasium as gym

class GRPCEnv(gym.Env):
  def __init__(self, url):
    self.url = url
    self.channel = grpc.insecure_channel(url)
    self.stub = environment_pb2_grpc.EnvironmentStub(self.channel)
    self.observation_space, self.action_space = self._get_spaces()

  def step(self, action):
    request = environment_pb2.StepRequest(action=pickle.dumps(action))
    response = self.stub.Step(request)
    return pickle.loads(response.observation), response.reward, response.done, pickle.loads(response.info)
  
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
    response = self.stub.Close(request)
    self.channel.close()

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

  def is_wrapped(self, wrapper_type):
    request = environment_pb2.IsWrappedRequest(wrapper_class=wrapper_type)
    response = self.stub.IsWrapped(request)
    return response.is_wrapped