import pickle
import grpc
import environment_pb2
import environment_pb2_grpc

from stable_baselines3.common.env_util import is_wrapped

class EnvironmentServicer(environment_pb2_grpc.EnvironmentServicer):
    def __init__(self, env) -> None:
        self.env = env

    def ManageEnvironment(
        self, request: environment_pb2.EnvironmentRequest, unused_context
    ) -> environment_pb2.EnvironmentResponse:
        response = environment_pb2.EnvironmentResponse()

        if request.WhichOneOf("command") == "step":
            action = pickle.loads(request.step.action)
            observation, reward, terminated, truncated, info = self.env.step(action)
            # convert to SB3 VecEnv api
            done = terminated or truncated
            info["TimeLimit.truncated"] = truncated and not terminated
            if done:
                # save final observation where user can get it, then reset
                info["terminal_observation"] = observation
                observation, reset_info = self._env.reset()
            subresponse = response.step_response
            subresponse.observation = pickle.dumps(observation)
            subresponse.reward = reward
            subresponse.done = done
            subresponse.info = pickle.dumps(info)
            subresponse.reset_info = pickle.dumps(reset_info)
        elif request.WhichOneOf("command") == "reset":
            seed = request.reset.seed
            maybe_options = {"options": pickle.loads(request.reset.options)} if request.reset.options else {}
            observation, reset_info = self.env.reset(seed=seed, **maybe_options)
            subresponse = response.reset_response
            subresponse.observation = pickle.dumps(observation)
            subresponse.reset_info = pickle.dumps(reset_info)
        elif request.WhichOneOf("command") == "render":
            image = self.env.render()
            subresponse = response.render_response
            subresponse.render_data = pickle.dumps(image)
        elif request.WhichOneOf("command") == "close":
            self._env.close()
            response.close_response.SetInParent()
        elif request.WhichOneOf("command") == "get_spaces":
            subresponse = response.get_spaces_response
            subresponse.observation_space = pickle.dumps(self.env.observation_space)
            subresponse.action_space = pickle.dumps(self.env.action_space)
        elif request.WhichOneOf("command") == "env_method":
            method_name = request.env_method.method_name
            args, kwargs = pickle.arguments(request.env_method.args)
            method = getattr(self.env, method_name)
            result = method(*args, **kwargs)
            subresponse = response.env_method_response
            subresponse.result = pickle.dumps(result)
        elif request.WhichOneOf("command") == "get_attr":
            attr = request.get_attr.attribute_name
            result = getattr(self.env, attr)
            subresponse = response.get_attr_response
            subresponse.attribute_value = pickle.dumps(result)
        elif request.WhichOneOf("command") == "set_attr":
            attr = request.get_attr.attribute_name
            val = pickle.loads(request.set_attr.attribute_value)
            result = setattr(self.env, attr, val)
            response.set_attr_response.SetInParent()
        elif request.WhichOneOf("command") == "is_wrapped":
            wrapper_type = request.is_wrapped.wrapper_type
            result = is_wrapped(self.env, wrapper_type)
            subresponse = response.is_wrapped_response
            subresponse.is_wrapped = result
        else:
            raise NotImplementedError(f"Invalid request is not implemented in the worker")
        
        return response
    
async def serve(env):
    server = grpc.aio.server()
    environment_pb2_grpc.add_EnvironmentServicer_to_server(
        EnvironmentServicer(env), server
    )
    server.add_insecure_port("[::]:50051")
    await server.start()
    await server.wait_for_termination()