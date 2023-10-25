from ray.rllib.algorithms.algorithm import Algorithm
import logging
import math
import gymnasium as gym
import h5py

import numpy as np
import random
from typing import List, Optional, TYPE_CHECKING, Union
from urllib.parse import urlparse
from PIL import Image
import omnigibson as og

from omnigibson.envs.rl_env import RLEnv

try:
    from smart_open import smart_open
except ImportError:
    smart_open = None

from ray.rllib.offline.input_reader import InputReader
from ray.rllib.offline.io_context import IOContext
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import (
    DEFAULT_POLICY_ID,
    MultiAgentBatch,
    SampleBatch,
    concat_samples,
    convert_ma_batch_to_sample_batch,
)
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.models.preprocessors import DictFlatteningPreprocessor
from ray.rllib.utils.annotations import override, PublicAPI, DeveloperAPI
from ray.rllib.utils.compression import unpack_if_needed
from ray.rllib.utils.spaces.space_utils import clip_action, normalize_action
from ray.rllib.utils.typing import Any, FileType, SampleBatchType

from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.offline import (
    InputReader,
    IOContext,
    JsonReader,
    ShuffledInput,
)

if TYPE_CHECKING:
    from ray.rllib.evaluation import RolloutWorker

logger = logging.getLogger(__name__)

WINDOWS_DRIVES = [chr(i) for i in range(ord("c"), ord("z") + 1)]

@PublicAPI
class CustomReader(InputReader):
    """Reader object that loads experiences.

    The input files will be read from in random order.
    """

    @PublicAPI
    def __init__(
        self, inputs: Union[str, List[str]], ioctx: Optional[IOContext] = None
    ):
        """Initializes a JsonReader instance.

        Args:
            inputs: Either a glob expression for files, e.g. `/tmp/**/*.json`,
                or a list of single file paths or URIs, e.g.,
                ["s3://bucket/file.json", "s3://bucket/file2.json"].
            ioctx: Current IO context object or None.
        """
        self.ioctx = ioctx or IOContext()
        self.default_policy = self.policy_map = None
        self.batch_size = 1
        if self.ioctx:
            self.batch_size = self.ioctx.config.get("train_batch_size", 1)
            num_workers = self.ioctx.config.get("num_workers", 0)
            if num_workers:
                self.batch_size = max(math.ceil(self.batch_size / num_workers), 1)

        if self.ioctx.worker is not None:
            self.policy_map = self.ioctx.worker.policy_map
            self.default_policy = self.policy_map.get(DEFAULT_POLICY_ID)

        self.batch_builder = SampleBatchBuilder() 
        self.files = list(inputs)
        self.cur_file = None
        self.cur_file_path = None
        self.cur_file_index = 0
        observation_space = gym.spaces.Dict({
            'proprio': gym.spaces.Box(-np.inf, np.inf, (65,), np.float64), 
            'robot0:eyes_Camera_sensor_depth_linear': gym.spaces.Box(0.0, np.inf, (224, 224), np.float32), 
            # 'robot0:eyes_Camera_sensor_rgb': gym.spaces.Box(0, 255, (224, 224, 3), np.uint8), 
            'robot0:eyes_Camera_sensor_seg_instance': gym.spaces.Box(0, 1024, (224, 224), np.uint32), 
            'robot0:eyes_Camera_sensor_seg_semantic': gym.spaces.Box(0, 4096, (224, 224), np.uint32)
        })
        # from IPython import embed; embed()
        self.prep = DictFlatteningPreprocessor(observation_space)

    @override(InputReader)
    def next(self) -> SampleBatchType:
        count = 0
        while count < self.batch_size:
            line = self._next_line()
            t = line['t']
            eps_id = line['id'].decode('utf-8')

            prev_proprio = line['prev_proprio']
            prev_obs_rgb = np.array(Image.open(f'{self.cur_file_path}/eyes_Camera_sensor_rgb/{eps_id}_{t-1}.jpeg'))
            prev_obs_depth = np.array(Image.open(f'{self.cur_file_path}/eyes_Camera_sensor_depth_linear/{eps_id}_{t-1}.png'))
            prev_obs_seg_instance = np.array(Image.open(f'{self.cur_file_path}/eyes_Camera_sensor_seg_instance/{eps_id}_{t-1}.png'))
            prev_obs_seg_semantic = np.array(Image.open(f'{self.cur_file_path}/eyes_Camera_sensor_seg_semantic/{eps_id}_{t-1}.png'))
            prev_obs = {
                # 'robot0:eyes_Camera_sensor_rgb': prev_obs_rgb,
                'robot0:eyes_Camera_sensor_depth_linear': prev_obs_depth,
                'robot0:eyes_Camera_sensor_seg_instance': prev_obs_seg_instance,
                'robot0:eyes_Camera_sensor_seg_semantic': prev_obs_seg_semantic,
                'proprio': prev_proprio,   
            }

            proprio = line['proprio']
            obs_rgb = np.array(Image.open(f'{self.cur_file_path}/eyes_Camera_sensor_rgb/{eps_id}_{t}.jpeg'))
            obs_depth = np.array(Image.open(f'{self.cur_file_path}/eyes_Camera_sensor_depth_linear/{eps_id}_{t}.png'))
            obs_seg_instance = np.array(Image.open(f'{self.cur_file_path}/eyes_Camera_sensor_seg_instance/{eps_id}_{t}.png'))
            obs_seg_semantic = np.array(Image.open(f'{self.cur_file_path}/eyes_Camera_sensor_seg_semantic/{eps_id}_{t}.png'))
            obs = {
                # 'robot0:eyes_Camera_sensor_rgb': obs_rgb,
                'robot0:eyes_Camera_sensor_depth_linear': obs_depth,
                'robot0:eyes_Camera_sensor_seg_instance': obs_seg_instance,
                'robot0:eyes_Camera_sensor_seg_semantic': obs_seg_semantic,
                'proprio': proprio,   
            }


            self.batch_builder.add_values(
                t=t,
                eps_id=eps_id,
                agent_index=0,
                obs=self.prep.transform(prev_obs),
                actions=line['action'],
                action_prob=1.0,  # put the true action probability here
                action_logp=0.0,
                rewards=line['reward'],
                prev_actions=line['prev_action'],
                prev_rewards=line['prev_reward'],
                terminateds=line['done'],
                truncateds=line['truncated'],
                infos=None,
                new_obs=self.prep.transform(obs),
            )
            count += 1
        batch = self.batch_builder.build_and_reset() 
        # batch = self._postprocess_if_needed(batch)
        return batch

    def _postprocess_if_needed(self, batch: SampleBatchType) -> SampleBatchType:
        if not self.ioctx.config.get("postprocess_inputs"):
            return batch

        batch = convert_ma_batch_to_sample_batch(batch)

        if isinstance(batch, SampleBatch):
            out = []
            for sub_batch in batch.split_by_episode():
                out.append(self.default_policy.postprocess_trajectory(sub_batch))
            return concat_samples(out)
        else:
            # TODO(ekl) this is trickier since the alignments between agent
            #  trajectories in the episode are not available any more.
            raise NotImplementedError(
                "Postprocessing of multi-agent data not implemented yet."
            )

    def _next_line(self) -> str:
        if not self.cur_file or self.cur_file_index >= len(self.cur_file['data_group']['ids']):
            self.cur_file = self._next_file()
            self.cur_file_index = 0
        
        t = self.cur_file['data_group']['timesteps'][self.cur_file_index]
        if t == -1:
            self.cur_file_index += 1
            return self._next_line()
        
        line = {}
        line['t'] = self.cur_file['data_group']['timesteps'][self.cur_file_index]
        line['id'] = self.cur_file['data_group']['ids'][self.cur_file_index]
        line['action'] = self.cur_file['data_group']['actions'][self.cur_file_index]
        line['reward'] = self.cur_file['data_group']['rewards'][self.cur_file_index]
        line['truncated'] = self.cur_file['data_group']['truncated'][self.cur_file_index]
        line['done'] = self.cur_file['data_group']['done'][self.cur_file_index]
        line['proprio'] = self.cur_file['data_group']['proprio'][self.cur_file_index]
        line['prev_proprio'] = self.cur_file['data_group']['proprio'][self.cur_file_index - 1]
        line['prev_action'] = self.cur_file['data_group']['actions'][self.cur_file_index - 1]
        line['prev_reward'] = self.cur_file['data_group']['rewards'][self.cur_file_index - 1]
        self.cur_file_index += 1
        return line

        # if not line:
        #     if hasattr(self.cur_file, "close"):  # legacy smart_open impls
        #         self.cur_file.close()
        #     self.cur_file = self._next_file()
        #     self.cur_file_index = 0
        #     line = self.cur_file.readline()
        #     if not line:
        #         logger.debug("Ignoring empty file {}".format(self.cur_file))
        # if not line:
        #     raise ValueError(
        #         "Failed to read next line from files: {}".format(self.files)
        #     )
        # return line

    def _next_file(self) -> FileType:
        # If this is the first time, we open a file, make sure all workers
        # start with a different one if possible.
        if self.cur_file is None and self.ioctx.worker is not None:
            idx = self.ioctx.worker.worker_index
            total = self.ioctx.worker.num_workers or 1
            path = self.files[round((len(self.files) - 1) * (idx / total))]
        # After the first file, pick all others randomly.
        else:
            path = random.choice(self.files)
        self.cur_file_path = path
        return h5py.File(path + "/data.h5", 'r')

DIST_COEFF = 0.1
GRASP_REWARD = 0.3

cfg = {
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": "Rs_int",
            "load_object_categories": ["floors", "coffee_table"],
        },
        "robots": [
            {
                "type": "Tiago",
                "obs_modalities": ["rgb", "depth_linear", "seg_instance", "seg_semantic", "proprio"],
                "proprio_obs": ["robot_pose", "joint_qpos", "joint_qvel", "eef_left_pos", "eef_left_quat", "grasp_left"],
                "scale": 1.0,
                "self_collisions": True,
                "action_normalize": False,
                "action_type": "continuous",
                "grasping_mode": "sticky",
                "rigid_trunk": False,
                "default_arm_pose": "diagonal30",
                "default_trunk_offset": 0.365,
                "sensor_config": {
                    "VisionSensor": {
                        "modalities": ["rgb", "depth_linear", "seg_instance", "seg_semantic"],
                        "sensor_kwargs": {
                            "image_width": 224,
                            "image_height": 224
                        }
                    }
                },
                "controller_config": {
                    "base": {
                        "name": "JointController",
                        "motor_type": "velocity"
                    },
                    "arm_left": {
                        "name": "InverseKinematicsController",
                        "motor_type": "velocity",
                        "command_input_limits": None,
                        "command_output_limits": None,
                        "mode": "pose_absolute_ori", 
                        "kv": 3.0
                    },
                    # "arm_left": {
                    #     "name": "JointController",
                    #     "motor_type": "position",
                    #     "command_input_limits": None,
                    #     "command_output_limits": None, 
                    #     "use_delta_commands": False
                    # },
                    "arm_right": {
                        "name": "JointController",
                        "motor_type": "position",
                        "command_input_limits": None,
                        "command_output_limits": None, 
                        "use_delta_commands": False
                    },
                    "gripper_left": {
                        "name": "JointController",
                        "motor_type": "position",
                        "command_input_limits": [-1, 1],
                        "command_output_limits": None,
                        "use_delta_commands": True,
                        "use_single_command": True
                    },
                    "gripper_right": {
                        "name": "JointController",
                        "motor_type": "position",
                        "command_input_limits": [-1, 1],
                        "command_output_limits": None,
                        "use_delta_commands": True,
                        "use_single_command": True
                    },
                    "camera": {
                        "name": "JointController",
                        "motor_type": "position",
                        "command_input_limits": None,
                        "command_output_limits": None,
                        "use_delta_commands": False
                    }
                }
            }
        ],
        "task": {
            "type": "GraspTask",
            "obj_name": "cologne",
            "termination_config": {
                "max_steps": 100000,
            },
            "reward_config": {
                "r_dist_coeff": DIST_COEFF,
                "r_grasp": GRASP_REWARD
            }
        },
        "objects": [
            {
                "type": "DatasetObject",
                "name": "cologne",
                "category": "bottle_of_cologne",
                "model": "lyipur",
                "position": [-0.3, -0.8, 0.5],
            },
        ]
    }

env_config = {
    "cfg": cfg,
    "reset_positions": [],
    "action_space_controllers": ["base", "camera", "arm_left", "gripper_left"]
}

env = RLEnv(env_config)

register_env("my_env", lambda config: env)
# Use the Algorithm's `from_checkpoint` utility to get a new algo instance
# that has the exact same state as the old one, from which the checkpoint was
# created in the first place:
algo = Algorithm.from_checkpoint("./checkpoints")
obs = env.reset()
del obs['robot0']['robot0:eyes_Camera_sensor_rgb']

for i in range(100):
    # from IPython import embed; embed()
    # transform_obs = DictFlatteningPreprocessor(observation_space).transform(obs['robot0'])
    # from IPython import embed; embed()
    action = algo.compute_single_action(obs['robot0'])
    action = env.transform_policy_action(action)
    obs, reward, done, truncated, info = env.step(action)
    del obs['robot0']['robot0:eyes_Camera_sensor_rgb']