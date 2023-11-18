import pickle
import grpc
import environment_pb2
import environment_pb2_grpc

from stable_baselines3.common.env_util import is_wrapped

class EnvironmentServicer(environment_pb2_grpc.EnvironmentServicer):
    def __init__(self, env) -> None:
        self.env = env

    def Step(self, request, unused_context):
        action = pickle.loads(request.action)
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        info["TimeLimit.truncated"] = truncated and not terminated

        if done:
            info["terminal_observation"] = observation
            observation, reset_info = self.env.reset()

        return environment_pb2.StepResponse(
            observation=pickle.dumps(observation),
            reward=reward,
            done=done,
            info=pickle.dumps(info),
            reset_info=pickle.dumps(reset_info)
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
        wrapper_type = request.wrapper_type
        is_wrapped = hasattr(self.env, wrapper_type)  # Assuming is_wrapped is implemented as hasattr
        return environment_pb2.IsWrappedResponse(is_wrapped=is_wrapped)
    
async def serve(env):
    server = grpc.aio.server()
    environment_pb2_grpc.add_EnvironmentServicer_to_server(
        EnvironmentServicer(env), server
    )
    server.add_insecure_port("[::]:50051")
    await server.start()
    await server.wait_for_termination()