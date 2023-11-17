import grpc
import environment_pb2
import environment_pb2_grpc

class EnvironmentServicer(environment_pb2_grpc.EnvironmentServicer):
    def __init__(self, env) -> None:
        self.env = env

    def ManageEnvironment(
        self, request: environment_pb2.EnvironmentRequest, unused_context
    ) -> environment_pb2.EnvironmentResponse:
        if cmd == "step":
            observation, reward, terminated, truncated, info = env.step(data)
            # convert to SB3 VecEnv api
            done = terminated or truncated
            info["TimeLimit.truncated"] = truncated and not terminated
            if done:
                # save final observation where user can get it, then reset
                info["terminal_observation"] = observation
                observation, reset_info = env.reset()
            remote.send((observation, reward, done, info, reset_info))
        elif cmd == "reset":
            maybe_options = {"options": data[1]} if data[1] else {}
            observation, reset_info = env.reset(seed=data[0], **maybe_options)
            remote.send((observation, reset_info))
        elif cmd == "render":
            remote.send(env.render())
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_spaces":
            remote.send((env.observation_space, env.action_space))
        elif cmd == "env_method":
            method = getattr(env, data[0])
            remote.send(method(*data[1], **data[2]))
        elif cmd == "get_attr":
            remote.send(getattr(env, data))
        elif cmd == "set_attr":
            remote.send(setattr(env, data[0], data[1]))  # type: ignore[func-returns-value]
        elif cmd == "is_wrapped":
            remote.send(is_wrapped(env, data))
        else:
            raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
    
async def serve(env):
    server = grpc.aio.server()
    environment_pb2_grpc.add_EnvironmentServicer_to_server(
        EnvironmentServicer(env), server
    )
    server.add_insecure_port("[::]:50051")
    await server.start()
    await server.wait_for_termination()