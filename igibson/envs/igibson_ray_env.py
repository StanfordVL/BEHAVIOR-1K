# TODO: Refactor!

# """
# Example showing how to wrap the iGibson class using ray for low-level environment control.
# Multiple environments are only supported on Linux. If issues arise, please ensure torch/numpy
# are installed *without* MKL support.
# """
# import argparse
# import time
#
# import ray
#
# from igibson.envs.igibson_env import iGibsonEnv
#
# ray.init()
#
#
# @ray.remote
# class iGibsonRayEnv(iGibsonEnv):
#     def sample_action_space(self):
#         return self.action_space.sample()
#
#     def get_current_step(self):
#         return self.current_step
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", "-c", help="which config file to use [default: use yaml files in examples/configs]")
#     parser.add_argument(
#         "--mode",
#         "-m",
#         choices=["headless", "headless_tensor", "gui_interactive", "gui_non_interactive"],
#         default="headless",
#         help="which mode for simulation (default: headless)",
#     )
#     args = parser.parse_args()
#
#     env = iGibsonRayEnv.remote(
#         config_file=args.config, mode=args.mode, action_timestep=1.0 / 10.0, physics_timestep=1.0 / 40.0
#     )
#
#     step_time_list = []
#     for episode in range(100):
#         print("Episode: {}".format(episode))
#         start = time.time()
#         env.reset.remote()
#         for _ in range(100):  # 10 seconds
#             # This is unnecessarily slow, if you can avoid calling ray.get and directly pass
#             # the handle to your actor
#             action = ray.get(env.sample_action_space.remote())
#             state, reward, done, _ = ray.get(env.step.remote(action))
#             print("reward", reward)
#             if done:
#                 break
#         print(
#             "Episode finished after {} timesteps, took {} seconds.".format(
#                 ray.get(env.get_current_step.remote()), time.time() - start
#             )
#         )
#     env.remote.close()
