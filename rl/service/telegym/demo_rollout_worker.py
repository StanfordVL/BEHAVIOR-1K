import asyncio

import gymnasium as gym

from telegym.grpc_server import serve_env_over_grpc


async def main(local_addr, learner_addr):
    env = gym.make("CartPole-v1", render_mode='rgb_array')

    # Now start servicing!
    await serve_env_over_grpc(env, local_addr, learner_addr)

if __name__ == "__main__":
    import sys, socket

    # Obtain an unused port
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    local_port = s.getsockname()[1]
    s.close()

    asyncio.get_event_loop().run_until_complete(main("localhost:" + str(local_port), sys.argv[1]))
