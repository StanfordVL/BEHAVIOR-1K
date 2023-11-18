import asyncio

import gymnasium as gym

from rollout_worker import serve


async def main(local_addr, learner_addr):
    env = gym.make("CartPole-v1", render_mode='rgb_array')

    # Now start servicing!
    await serve(env, local_addr, learner_addr)

if __name__ == "__main__":
    import sys
    local_port = int(sys.argv[1])
    asyncio.get_event_loop().run_until_complete(main("localhost:" + str(local_port), "localhost:50051"))