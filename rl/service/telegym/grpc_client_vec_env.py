import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import threading
import time
from typing import Any, List, Type

import gymnasium as gym

import grpc
import numpy as np
import environment_pb2
import environment_pb2_grpc
from grpc_client_env import GRPCClientEnv

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env import DummyVecEnv

class EnvironmentRegistrationServicer(environment_pb2_grpc.EnvironmentRegistrationService):
    def __init__(self, n_workers):
        self.envs = [None] * n_workers
        self.completed = asyncio.Event()

    def RegisterEnvironment(self, request, unused_context):
        for i, env in enumerate(self.envs):
            if env is None:
                address = request.ip + ":" + str(request.port)
                self.envs[i] = GRPCClientEnv(address)

                remaining = sum(1 for x in self.envs if x is None)
                print(f"Registering worker at {address}, {remaining} more workers needed.")

                # Confirm us as done if we're the last worker
                if not remaining:
                    print("Registered all workers.")
                    self.completed.set()

                return environment_pb2.RegisterEnvironmentResponse(success=True)
        return environment_pb2.RegisterEnvironmentResponse(success=False)

    async def await_workers(self):
        print(f"Waiting for {len(self.envs)} workers to connect.")
        await self.completed.wait()
        return list(self.envs)

async def _register_workers(local_addr, n_envs):
    # Start the registration server
    registration_server = grpc.aio.server()
    registration_servicer = EnvironmentRegistrationServicer(n_envs)
    environment_pb2_grpc.add_EnvironmentRegistrationServiceServicer_to_server(registration_servicer, registration_server)
    registration_server.add_insecure_port(local_addr)
    print(f"Launching registration server at {local_addr}.")
    await registration_server.start()

    # Await the workers
    return await registration_servicer.await_workers()

class GRPCClientVecEnv(DummyVecEnv):
    def __init__(self, local_addr, n_envs):
        self.waiting = False

        # Register workers
        envs = asyncio.get_event_loop().run_until_complete(_register_workers(local_addr, n_envs))

        super().__init__([(lambda x=x: x) for x in envs])

    def step_async(self, actions: np.ndarray) -> None:
        for env, action in zip(self.envs, actions):
            env.step_async(action)
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        # Avoid circular imports
        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx] = self.envs[env_idx].step_wait()
            # convert to SB3 VecEnv api
            self.buf_dones[env_idx] = terminated or truncated
            # See https://github.com/openai/gym/issues/3102
            # Gym 0.26 introduces a breaking change
            self.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated

            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [env_i.get_attr(attr_name) for env_i in target_envs]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            env_i.set_attr(attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [env_i.env_method(method_name, *method_args, **method_kwargs) for env_i in target_envs]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_envs = self._get_target_envs(indices)
        return [env_i.is_wrapped(wrapper_class) for env_i in target_envs]