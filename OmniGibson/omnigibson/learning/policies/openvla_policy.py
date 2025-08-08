import torch
from typing import Union, Tuple
import copy
import numpy as np
import requests
from collections import deque
from PIL import Image
from omnigibson.learning.policies.policy_base import BasePolicy
from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES, ACTION_QPOS_INDICES


RESIZE_SIZE = 224
ACTION_DIM = 23
LEFT_GRIPPER_IDX = ACTION_QPOS_INDICES["R1Pro"]["left_gripper"]
RIGHT_GRIPPER_IDX = ACTION_QPOS_INDICES["R1Pro"]["right_gripper"]


def generate_prop_state(proprio_data):
    return np.concatenate(
        [
            proprio_data[:, PROPRIOCEPTION_INDICES["R1Pro"]["base_qpos"]],  # 3
            proprio_data[:, PROPRIOCEPTION_INDICES["R1Pro"]["trunk_qpos"]],  # 4
            proprio_data[:, PROPRIOCEPTION_INDICES["R1Pro"]["arm_left_qpos"]],  #  7
            proprio_data[:, PROPRIOCEPTION_INDICES["R1Pro"]["arm_right_qpos"]],  #  7
            proprio_data[:, PROPRIOCEPTION_INDICES["R1Pro"]["gripper_left_qpos"]].sum(axis=-1)[:, None],  # 1
            proprio_data[:, PROPRIOCEPTION_INDICES["R1Pro"]["gripper_right_qpos"]].sum(axis=-1)[:, None],  # 1
        ],
        axis=-1,
    )  # (:,23)


def resize_image_for_openvla(img: np.ndarray, size: Union[int, Tuple[int, int]]) -> np.ndarray:
    """
    Resize an image to match openvla-oft policy's expected input size.
    """
    if isinstance(size, int):
        size = (size, size)
    return np.array(Image.fromarray(img).resize(size, resample=Image.LANCZOS))


class OpenVLA(BasePolicy):
    """
    OpenVLA-OFT policy from Kim et al. https://arxiv.org/pdf/2502.19645
    """

    def __init__(
        self,
        *args,
        host: str,
        port: int,
        text_prompt: str,
        control_mode: str = "temporal_ensemble",
        robot_type: str = "R1Pro",
        **kwargs,
    ) -> None:
        import json_numpy

        json_numpy.patch()
        super().__init__(robot_type=robot_type, *args, **kwargs)
        # server endpoint for action generation from OpenVLA server
        self.policy_endpoint = f"http://{host}:{port}/act"
        self._resize_obs = resize_image_for_openvla

        self.text_prompt = text_prompt
        self.control_mode = control_mode
        self.action_queue = deque([], maxlen=10)
        self.last_action = np.zeros((10, ACTION_DIM), dtype=np.float64)

        self.replan_interval = 10  # K: replan every 10 steps
        self.max_len = 50  # how long the policy sequences are
        self.temporal_ensemble_max = 5  # max number of sequences to ensemble
        self.step_counter = 0

    def reset(self) -> None:
        self.action_queue = deque([], maxlen=10)
        self.last_action = np.zeros((10, ACTION_DIM), dtype=np.float64)
        self.step_counter = 0

    def process_obs(self, obs: dict, control_mode: str) -> dict:
        """
        Prepare observation for policy input.
        """
        prop_state = generate_prop_state(obs["robot_r1::proprio"][None])
        img_obs = torch.stack(
            [
                obs["robot_r1::robot_r1:zed_link:Camera:0::rgb"][None, ..., :3],
                obs["robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"][None, ..., :3],
                obs["robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"][None, ..., :3],
            ],
            axis=1,
        )  # (1, 3, H, W, C)

        # Only use last observation and proprio if not receeding_temporal
        if control_mode != "receeding_temporal":
            img_obs = img_obs[:, -1]  # (B, T, num_cameras, H, W, C) -> (B, num_cameras, H, W, C)
            prop_state = prop_state[:, -1]  # (B, T, 23) -> (B, 23)

        if img_obs.shape[-1] != 3:  # Channel last
            img_obs = np.transpose(img_obs, (0, 1, 3, 4, 2))

        processed_obs = {
            "full_image": self._resize_obs(img_obs[0, 0].numpy(), RESIZE_SIZE),
            "left_wrist_image": self._resize_obs(img_obs[0, 1].numpy(), RESIZE_SIZE),
            "right_wrist_image": self._resize_obs(img_obs[0, 2].numpy(), RESIZE_SIZE),
            "state": prop_state[0],
            "instruction": self.text_prompt,
        }

        return processed_obs

    def get_action_from_server(self, obs: dict) -> torch.Tensor:
        """
        Get action from OpenVLA server, update self.last_action if successful
        """
        try:
            response = requests.post(
                self.policy_endpoint,
                json=obs,
            )
            action = response.json()  # list of np.ndarray
            if action != "error":
                self.last_action = action
            else:
                print(f"Error in action prediction, using last action")
        except:
            print(f"Error in action prediction, using last action")
        return self.last_action

    def forward(self, obs: dict, *args, **kwargs) -> torch.Tensor:
        """
        Model input expected: (finetuned on R1Pro)
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
        input_obs = self.process_obs(obs, self.control_mode)

        if self.control_mode == "receeding_temporal":
            return self._act_receeding_temporal(input_obs)

        if self.control_mode == "receeding_horizon":
            if len(self.action_queue) > 0:
                # pop the first action in the queue
                final_action = self.action_queue.popleft()[None]
                return final_action[..., :ACTION_DIM]

        # Get action from OpenVLA server
        batch = copy.deepcopy(input_obs)
        action = self.get_action_from_server(batch)

        # action shape: (10, 23), joint_positions shape: (23,)
        target_joint_positions = action.copy()

        if self.control_mode == "receeding_horizon":
            self.action_queue = deque([a for a in target_joint_positions[: self.max_len]])
            final_action = self.action_queue.popleft()[None]

        # # temporal emsemble start
        elif self.control_mode == "temporal_ensemble":
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
            final_action[LEFT_GRIPPER_IDX] = target_joint_positions[0, LEFT_GRIPPER_IDX]
            final_action[RIGHT_GRIPPER_IDX] = target_joint_positions[0, RIGHT_GRIPPER_IDX]
            final_action = final_action[None]
        else:
            final_action = target_joint_positions

        return final_action[
            ..., :ACTION_DIM
        ]  # return only the first 23 joints, which are the robot's joints (base, trunk, arms, grippers)

    def _act_receeding_temporal(self, input_obs):
        # Step 1: check if we should re-run policy
        if self.step_counter % self.replan_interval == 0:
            # Run policy every K steps
            batch = copy.deepcopy(input_obs)

            # Get action from OpenVLA server
            action = self.get_action_from_server(batch)
            target_joint_positions = action.copy()

            # Add this sequence to action queue
            new_seq = deque([a for a in target_joint_positions[: self.max_len]])
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
        final_action[LEFT_GRIPPER_IDX] = actions_current_timestep[0, LEFT_GRIPPER_IDX]
        final_action[RIGHT_GRIPPER_IDX] = actions_current_timestep[0, RIGHT_GRIPPER_IDX]
        final_action = final_action[None]

        self.step_counter += 1

        return final_action[..., :ACTION_DIM]
