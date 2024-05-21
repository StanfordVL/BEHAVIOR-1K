import os
from time import time

import gym
import psutil
from pynvml.smi import nvidia_smi

import omnigibson as og


class ProfilingEnv(og.Environment):
    def step(self, action):
        try:
            start = time()
            # If the action is not a dictionary, convert into a dictionary
            if not isinstance(action, dict) and not isinstance(action, gym.spaces.Dict):
                action_dict = dict()
                idx = 0
                for robot in self.robots:
                    action_dim = robot.action_dim
                    action_dict[robot.name] = action[idx : idx + action_dim]
                    idx += action_dim
            else:
                # Our inputted action is the action dictionary
                action_dict = action

            # Iterate over all robots and apply actions
            for robot in self.robots:
                robot.apply_action(action_dict[robot.name])

            # Run simulation step
            sim_start = time()
            if len(og.sim._objects_to_initialize) > 0:
                og.sim.render()
            super(type(og.sim), og.sim).step(render=True)
            omni_time = (time() - sim_start) * 1e3

            # Additionally run non physics things
            og.sim._non_physics_step()

            # Grab observations
            obs, obs_info = self.get_obs()

            # Step the scene graph builder if necessary
            if self._scene_graph_builder is not None:
                self._scene_graph_builder.step(self.scene)

            # Grab reward, done, and info, and populate with internal info
            reward, done, info = self.task.step(self, action)
            self._populate_info(info)

            if done and self._automatic_reset:
                # Add lost observation to our information dict, and reset
                info["last_observation"] = obs
                info["last_observation_info"] = obs_info
                obs, obs_info = self.reset()

            # Increment step
            self._current_step += 1

            # collect profiling data
            total_frame_time = (time() - start) * 1e3
            og_time = total_frame_time - omni_time
            # memory usage in GB
            memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024**3
            # VRAM usage in GB
            for gpu in nvidia_smi.getInstance().DeviceQuery()["gpu"]:
                found = False
                for process in gpu["processes"]:
                    if process["pid"] == os.getpid():
                        vram_usage = process["used_memory"] / 1024
                        found = True
                        break
                if found:
                    break

            ret = [total_frame_time, omni_time, og_time, memory_usage, vram_usage]
            if self._current_step % 100 == 0:
                print(
                    "total time: {:.3f} ms, Omni time: {:.3f} ms, OG time: {:.3f} ms, memory: {:.3f} GB, vram: {:.3f} GB.".format(
                        *ret
                    )
                )

            return obs, reward, done, info, ret
        except:
            raise ValueError(
                f"Failed to execute environment step {self._current_step} in episode {self._current_episode}"
            )
