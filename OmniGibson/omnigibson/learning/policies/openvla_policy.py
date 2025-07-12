import torch
import tensorflow as tf
from typing import Union, Tuple
import copy
import numpy as np
from collections import deque
from PIL import Image
from omnigibson.learning.policies.policy_base import BasePolicy

RESIZE_SIZE = 224

def generate_prop_state(proprio_data):
    base_qpos = proprio_data[:,246:249] # 3
    trunk_qpos = proprio_data[:,238:242] # 4
    arm_left_qpos = proprio_data[:,158:165] #  7
    arm_right_qpos = proprio_data[:,198:205] #  7
    left_gripper_width = proprio_data[:,194:196].sum(axis=-1)[:,None] # 1
    right_gripper_width = proprio_data[:,234:236].sum(axis=-1)[:,None] # 1
    
    prop_state = np.concatenate((base_qpos, trunk_qpos, arm_left_qpos, arm_right_qpos, left_gripper_width, right_gripper_width), axis=-1) # 23
    return prop_state

def resize_image_for_openvla(img: np.ndarray, resize_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    """
    Resize an image to match openvla-oft policy's expected input size.
    """
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    return np.array(
        Image.fromarray(img).resize(resize_size, resample=Image.LANCZOS)
    )

class OpenVLA(BasePolicy):
    """
    OpenVLA-OFT policy from Kim et al. https://arxiv.org/pdf/2502.19645
    """

    def __init__(
        self,
        host: str,
        port: int,
        text_prompt : str,
        control_mode : str = "temporal_ensemble",
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._resize_obs = resize_image_for_openvla
        
        # server endpoint for action generation from OpenVLA server
        self.policy_endpoint = f"http://{host}:{port}/act"

        self.text_prompt = text_prompt
        self.control_mode = control_mode
        self.action_queue = deque([],maxlen=10)
        self.last_action = np.zeros((10, 21), dtype=np.float64)

        self.replan_interval = 10             # K: replan every 10 steps
        self.max_len = 50                     # how long the policy sequences are
        self.temporal_ensemble_max = 5        # max number of sequences to ensemble
        self.step_counter = 0

    def reset(self) -> None:
        self.action_queue = deque([],maxlen=10)
        self.last_action = np.zeros((10, 23), dtype=np.float64)
        self.step_counter = 0

    def process_obs(self, obs: dict) -> dict:
        """
        Prepare observation for policy input.
        """
        prop_state = generate_prop_state(obs["robot_r1::proprio"][None])
        img_obs = torch.stack([
            obs["robot_r1::robot_r1:zed_link:Camera:0::rgb"][None,...,:3],
            obs["robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"][None,...,:3],
            obs["robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"][None,...,:3],
        ], axis=1)

        processed_obs = {
            "observation": img_obs,  # Shape: (1, 3, H, W, C)
            "proprio": prop_state,
            "prompt": self.text_prompt,
        }
        return processed_obs

    def get_action_from_server(self, obs: dict) -> torch.Tensor:
        import requests
        response = requests.post(
            self.policy_endpoint,
            json=obs,
        )
        return response.json()

    def forward(self, obs: dict, *args, **kwargs) -> torch.Tensor:
        """
        Model input expected: 
            📌 Key: full_image
            Type: ndarray
            Dtype: uint8
            Shape: (224, 224, 3)

            📌 Key: left_wrist_image
            Type: ndarray
            Dtype: uint8
            Shape: (224, 224, 3)

            📌 Key: right_wrist_image
            Type: ndarray
            Dtype: uint8
            Shape: (224, 224, 3)

            📌 Key: state
            Type: ndarray
            Dtype: float64
            Shape: (23,)

            📌 Key: instruction
            Type: str
            Value: do something
        
        Model will output:
            📌 Key: actions
            Type: ndarray
            Dtype: float64
            Shape: (10, 23)
        """
        input_obs = self.process_obs(obs)

        if self.control_mode == 'receeding_temporal':
            return self._act_receeding_temporal(input_obs)

        if self.control_mode == 'receeding_horizon':
            if len(self.action_queue) > 0:
                # pop the first action in the queue
                final_action = self.action_queue.popleft()[None]
                return final_action[...,:23]

        nbatch = copy.deepcopy(input_obs)

        # (B, T, num_cameras, H, W, C) -> (B, num_cameras, H, W, C), only use last observation 
        nbatch["observation"] = nbatch["observation"][:, -1] 
        # (B, NUM_CAM, H, W, C) --> (B, NUM_CAM, C, H, W), permute if pytorch
        if nbatch["observation"].shape[-1] != 3: 
            nbatch["observation"] = np.transpose(nbatch["observation"], (0, 1, 3, 4, 2))

        # nbatch["proprio"], (B, T, 16), where B=1
        joint_positions = nbatch["proprio"][0, -1]
        batch = {
                "full_image": self._resize_obs(nbatch["observation"][0, 0].numpy(), RESIZE_SIZE),
                "left_wrist_image": self._resize_obs(nbatch["observation"][0, 1].numpy(), RESIZE_SIZE),
                "right_wrist_image": self._resize_obs(nbatch["observation"][0, 2].numpy(), RESIZE_SIZE),
                "state": joint_positions,
                "instruction": self.text_prompt,
        }

        try:
            action = self.get_action_from_server(batch)
            self.last_action = action
        except:
            action = self.last_action
            print("Error in action prediction, using last action")
        
        # action shape: (10, 23), joint_positions shape: (23,)
        target_joint_positions = action.copy()

        if self.control_mode == 'receeding_horizon':
            self.action_queue = deque([a for a in target_joint_positions[:self.max_len]])
            final_action = self.action_queue.popleft()[None]

        # # temporal emsemble start
        elif self.control_mode == 'temporal_ensemble':
            new_actions = deque(target_joint_positions)
            self.action_queue.append(new_actions)
            actions_current_timestep = np.empty((len(self.action_queue), target_joint_positions.shape[1]))

            # k = 0.01
            k = 0.005
            for i, q in enumerate(self.action_queue):
                actions_current_timestep[i] = q.popleft()

            exp_weights = np.exp(k * np.arange(actions_current_timestep.shape[0]))
            exp_weights = exp_weights / exp_weights.sum()

            final_action = (actions_current_timestep * exp_weights[:, None]).sum(axis=0)
            final_action[-8] = target_joint_positions[0,-8]
            final_action[-1] = target_joint_positions[0,-1]
            final_action = final_action[None]
        else:
            final_action = target_joint_positions

        return final_action[...,:23]  # return only the first 23 joints, which are the robot's joints (base, trunk, arms, grippers)

    def _act_receeding_temporal(self, input_obs):
        # Step 1: check if we should re-run policy
        if self.step_counter % self.replan_interval == 0:
            # Run policy every K steps
            nbatch = copy.deepcopy(input_obs)
            if nbatch["observation"].shape[-1] != 3:
                nbatch["observation"] = np.transpose(nbatch["observation"], (0, 1, 3, 4, 2))

            joint_positions = nbatch["proprio"][0]
            batch = {
                "full_image": self._resize_obs(nbatch["observation"][0, 0].numpy(), RESIZE_SIZE),
                "left_wrist_image": self._resize_obs(nbatch["observation"][0, 1].numpy(), RESIZE_SIZE),
                "right_wrist_image": self._resize_obs(nbatch["observation"][0, 2].numpy(), RESIZE_SIZE),
                "state": joint_positions,
                "instruction": self.text_prompt,
            }

            try:
                action = self.get_action_from_server(batch)
                print(f"action: {action.shape}")
                self.last_action = action
            except:
                action = self.last_action
                print("Error in action prediction, using last action")

            target_joint_positions = action.copy()

            # Add this sequence to action queue
            new_seq = deque([a for a in target_joint_positions[:self.max_len]])
            self.action_queue.append(new_seq)

            # Optional: limit memory
            while len(self.action_queue) > self.temporal_ensemble_max:
                self.action_queue.popleft()

        # Step 2: Smooth across current step from all stored sequences
        if len(self.action_queue) == 0:
            raise ValueError("Action queue empty in receeding_temporal mode.")

        actions_current_timestep = np.empty((len(self.action_queue), self.action_queue[0][0].shape[0]))

        for i in range(len(self.action_queue)):
            actions_current_timestep[i] = self.action_queue[i].popleft()

        # Drop exhausted sequences
        self.action_queue = deque([q for q in self.action_queue if len(q) > 0])

        # Apply temporal ensemble
        k = 0.005
        exp_weights = np.exp(k * np.arange(actions_current_timestep.shape[0]))
        exp_weights = exp_weights / exp_weights.sum()

        final_action = (actions_current_timestep * exp_weights[:, None]).sum(axis=0)

        # Preserve grippers from most recent rollout
        final_action[-8] = actions_current_timestep[0, -8]
        final_action[-1] = actions_current_timestep[0, -1]
        final_action = final_action[None]

        self.step_counter += 1

        return final_action[...,:23]

