import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any, List, Optional, Sequence, Type
import warnings

import gymnasium as gym

import grpc
import numpy as np
from telegym.protos import environment_pb2
from telegym.protos import environment_pb2_grpc
from telegym.grpc_client_env import GRPCClientEnv

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env import DummyVecEnv

class EnvironmentRegistrationServicer(environment_pb2_grpc.EnvironmentRegistrationService):
    def __init__(self, n_workers):
        self.envs = [None] * n_workers
        self.completed = asyncio.Event()

    def RegisterEnvironment(self, request, context):
        for i, env in enumerate(self.envs):
            if env is None:
                identity = context.peer().split(":")
                assert len(identity) == 3 and identity[0] == "ipv4", f"Identity {context.peer()} isn't valid ipv4 identity"
                ip = identity[1]
                address = ip + ":" + str(request.port)
                print(f"Start registration of {context.peer()}")
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

        # Register multithreaded executor
        self._executor = ThreadPoolExecutor(max_workers=n_envs)

        super().__init__([(lambda x=x: x) for x in envs])

    def step_wait(self) -> VecEnvStepReturn:
        step = lambda env_i, act_i: env_i.step(act_i)
        for env_idx, result in enumerate(self._executor.map(step, self.envs, self.actions)):
            obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx], self.reset_infos[env_idx] = result
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))
    
    def reset(self):
        reset = lambda env_i, seed_i, opts_i: env_i.reset(seed=seed_i, **opts_i)

        maybe_options = [{"options": self._options[env_idx]} if self._options[env_idx] else {} for env_idx in range(self.num_envs)]
        for env_idx, (obs, reset_info) in enumerate(self._executor.map(reset, self.envs, self._seeds, maybe_options)):
            self.reset_infos[env_idx] = reset_info
            self._save_obs(env_idx, obs)

        self._reset_seeds()
        self._reset_options()
        return self._obs_from_buf()
    
    def get_images(self):
        if self.render_mode != "rgb_array":
            warnings.warn(
                f"The render mode is {self.render_mode}, but this method assumes it is `rgb_array` to obtain images."
            )
            return [None for _ in self.envs]
        render = lambda env_i: env_i.render()
        return list(self._executor.map(render, self.envs))

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        get_attr = lambda env_i: env_i.get_attr(attr_name)
        return list(self._executor.map(get_attr, target_envs))

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        set_attr = lambda env_i: env_i.set_attr(attr_name, value)
        return list(self._executor.map(set_attr, target_envs))

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        env_method = lambda env_i: env_i.env_method(*method_args, **method_kwargs)
        return list(self._executor.map(env_method, target_envs))

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_envs = self._get_target_envs(indices)
        is_wrapped = lambda env_i: env_i.is_wrapped(wrapper_class)
        return list(self._executor.map(is_wrapped, target_envs))
    
    def close(self):
        close = lambda env_i: env_i.close()
        return list(self._executor.map(close, self.envs))
