import os
from time import time

import gymnasium as gym
import psutil

import omnigibson as og
import omnigibson.utils.pynvml_utils as pynvml


# Method copied from: https://github.com/wandb/wandb/blob/main/wandb/sdk/internal/system/assets/gpu.py
def gpu_in_use_by_this_process(gpu_handle: "GPUHandle", pid: int) -> bool:
    if psutil is None:
        return False

    try:
        base_process = psutil.Process(pid=pid)
    except psutil.NoSuchProcess:
        # do not report any gpu metrics if the base process cant be found
        return False

    our_processes = base_process.children(recursive=True)
    our_processes.append(base_process)

    our_pids = {process.pid for process in our_processes}

    compute_pids = {process.pid for process in pynvml.nvmlDeviceGetComputeRunningProcesses(gpu_handle)}  # type: ignore
    graphics_pids = {
        process.pid for process in pynvml.nvmlDeviceGetGraphicsRunningProcesses(gpu_handle)  # type: ignore
    }

    pids_using_device = compute_pids | graphics_pids

    return len(pids_using_device & our_pids) > 0


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
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                found = False
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                if gpu_in_use_by_this_process(handle, os.getpid()):
                    vram_usage = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**3
                    found = True
                    break
                if found:
                    break
            pynvml.nvmlShutdown()

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
