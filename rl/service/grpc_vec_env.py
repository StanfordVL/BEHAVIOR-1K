from concurrent.futures import ThreadPoolExecutor
import threading
import time
from typing import List

import grpc
import environment_pb2
import environment_pb2_grpc

from stable_baselines3.common.vec_env import VecEnv

class EnvironmentRegistrationServicer(environment_pb2_grpc.EnvironmentRegistrationServicer):
    def __init__(self, n_workers):
        self.addresses = [None] * n_workers
        self.channels = [None] * n_workers
        self.stubs = [None] * n_workers
        self.lock = threading.Lock()

    def RegisterEnvironment(self, request, unused_context):
        with self.lock:
            for i, env in enumerate(self.addresses):
                if env is None:
                    self.addresses[i] = request.ip + ":" + str(request.port)
                    self.channels[i] = grpc.insecure_channel(self.addresses[i])
                    self.stubs[i] = environment_pb2_grpc.EnvironmentStub(self.channels[i])
                    return environment_pb2.EnvironmentRegistrationResponse(success=True)
            return environment_pb2.EnvironmentRegistrationResponse(success=False)
        
    def remove_worker(self, i):
        assert self.lock.locked()
        self.addresses[i] = None
        self.channels[i] = None
        self.stubs[i] = None

    def await_workers(self):
        while True:
            with self.lock:
                if all([x is not None for x in self.addresses]):
                    return
            time.sleep(3)

class GRPCVecEnv(VecEnv):
    def __init__(self, n_envs):
        self.waiting = False
        self.closed = False

        # Start the registration server
        self.registration_server = grpc.server(ThreadPoolExecutor(max_workers=2))
        self.registration_servicer = EnvironmentRegistrationServicer(n_envs)
        environment_pb2_grpc.add_EnvironmentRegistrationServiceServicer_to_server(self.registration_servicer, self.registration_server)
        self.registration_server.add_insecure_port("[::]:50051")
        self.registration_server.start()

        # Await the workers
        self.registration_servicer.await_workers()

        # TODO: Fix this call.
        resp = self._send_command(0, environment_pb2.EnvironmentRequest(get_spaces=environment_pb2.GetSpacesCommand()))
        observation_space, action_space = None, None  # TODO: Fix this!

        super().__init__(n_envs, observation_space, action_space)

    def _send_command(self, i, request):
        # First wait for all workers to be available.
        self.registration_servicer.await_workers()

        # Get the stub
        with self.registration_servicer.lock:
            try:
                stub = self.registration_servicer.stubs[i]
                return stub.ManageEnvironment(request)
            except:  # TODO: Narrow this exception down.
                self.registration_servicer.remove_worker(i)
                raise

    def step_async(self, actions: np.ndarray) -> None:
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos, self.reset_infos = zip(*results)
        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos

    def reset(self) -> VecEnvObs:
        for env_idx, remote in enumerate(self.remotes):
            remote.send(("reset", (self._seeds[env_idx], self._options[env_idx])))
        results = [remote.recv() for remote in self.remotes]
        obs, self.reset_infos = zip(*results)
        # Seeds and options are only used once
        self._reset_seeds()
        self._reset_options()
        return _flatten_obs(obs, self.observation_space)

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        if self.render_mode != "rgb_array":
            warnings.warn(
                f"The render mode is {self.render_mode}, but this method assumes it is `rgb_array` to obtain images."
            )
            return [None for _ in self.remotes]
        for pipe in self.remotes:
            # gather render return from subprocesses
            pipe.send(("render", None))
        outputs = [pipe.recv() for pipe in self.remotes]
        return outputs

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("is_wrapped", wrapper_class))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices: VecEnvIndices) -> List[Any]:
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.

        :param indices: refers to indices of envs.
        :return: Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]