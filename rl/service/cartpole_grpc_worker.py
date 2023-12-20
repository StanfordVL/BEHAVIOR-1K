import asyncio

import gymnasium as gym

from grpc_server import serve_env_over_grpc


async def main(local_addr, learner_addr):
    env = gym.make("PongNoFrameskip-v4", render_mode='rgb_array')

    # Now start servicing!
    await serve_env_over_grpc(env, local_addr, learner_addr)

if __name__ == "__main__":
    import sys
    local_port = int(sys.argv[1])
    asyncio.get_event_loop().run_until_complete(main("localhost:" + str(local_port), "localhost:50051"))