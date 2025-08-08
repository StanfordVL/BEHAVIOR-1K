import numpy as np
from collections import deque
import logging
import copy
import torch
from omnigibson.learning.policies.policy_base import BasePolicy
from omnigibson.learning.utils.network_utils import WebsocketClientPolicy

RESIZE_SIZE = 224


def generate_prop_state(proprio_data):
    base_qvel = proprio_data[:, 246:249]  # 3
    trunk_qpos = proprio_data[:, 238:242]  # 4
    arm_left_qpos = proprio_data[:, 158:165]  #  7
    arm_right_qpos = proprio_data[:, 198:205]  #  7
    left_gripper_width = proprio_data[:, 194:196].sum(axis=-1)[:, None]  # 1
    right_gripper_width = proprio_data[:, 234:236].sum(axis=-1)[:, None]  # 1

    prop_state = np.concatenate(
        (base_qvel, trunk_qpos, arm_left_qpos, arm_right_qpos, left_gripper_width, right_gripper_width), axis=-1
    )  # 23
    return prop_state


class OpenPi(BasePolicy):
    """
    Pi-0 policy from Physical Intelligence https://www.physicalintelligence.company/download/pi0.pdf
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
        """
        Args:
            host (str): Host address of the OpenPi server.
            port (int): Port number of the OpenPi server.
            text_prompt (str): Text prompt to guide the policy's actions.
            control_mode (str): Control mode for the policy. Options are 'receeding_temporal', 'receeding_horizon', or 'temporal_ensemble'.
        """
        super().__init__(robot_type=robot_type, *args, **kwargs)
        from openpi_client.image_tools import resize_with_pad

        self._resize_with_pad = resize_with_pad
        # Create a trained policy.
        self.policy = WebsocketClientPolicy(
            host=host,
            port=port,
        )
        logging.info(f"Server metadata: {self.policy.get_server_metadata()}")
        self.text_prompt = text_prompt
        self.control_mode = control_mode
        self.action_queue = deque([], maxlen=10)
        self.last_action = np.zeros((10, 21), dtype=np.float64)
        self.max_len = 8

        self.replan_interval = 10  # K: replan every 10 steps
        self.max_len = 50  # how long the policy sequences are
        self.temporal_ensemble_max = 5  # max number of sequences to ensemble
        self.step_counter = 0

    def reset(self) -> None:
        self.action_queue = deque([], maxlen=10)
        self.last_action = np.zeros((10, 21), dtype=np.float64)
        self.step_counter = 0

    def process_obs(self, obs: dict) -> dict:
        """
        Process the observation dictionary to match the expected input format for the model.
        """
        prop_state = generate_prop_state(obs["robot_r1::proprio"][None])
        img_obs = torch.stack(
            [
                obs["robot_r1::robot_r1:zed_link:Camera:0::rgb"][None, ..., :3],
                obs["robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"][None, ..., :3],
                obs["robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"][None, ..., :3],
            ],
            axis=1,
        )
        processed_obs = {
            "observation": img_obs,  # Shape: (1, 3, H, W, C)
            "proprio": prop_state,
            "prompt": self.text_prompt,
        }
        return processed_obs

    def forward(self, obs: dict, *args, **kwargs) -> torch.Tensor:
        # TODO reformat data into the correct format for the model
        # TODO: communicate with justin that we are using numpy to pass the data. Also we are passing in uint8 for images
        """
        Model input expected:
            📌 Key: observation/exterior_image_1_left
            Type: ndarray
            Dtype: uint8
            Shape: (224, 224, 3)

            📌 Key: observation/exterior_image_2_left
            Type: ndarray
            Dtype: uint8
            Shape: (224, 224, 3)

            📌 Key: observation/joint_position
            Type: ndarray
            Dtype: float64
            Shape: (16,)

            📌 Key: prompt
            Type: str
            Value: do something

        Model will output:
            📌 Key: actions
            Type: ndarray
            Dtype: float64
            Shape: (10, 16)
        """
        input_obs = self.process_obs(obs)

        if self.control_mode == "receeding_temporal":
            return self._act_receeding_temporal(input_obs)

        if self.control_mode == "receeding_horizon":
            if len(self.action_queue) > 0:
                # pop the first action in the queue
                final_action = self.action_queue.popleft()[None]
                return final_action[..., :23]

        nbatch = copy.deepcopy(input_obs)
        # update nbatch observation (B, T, num_cameras, H, W, C) -> (B, num_cameras, H, W, C)
        nbatch["observation"] = nbatch["observation"][:, -1]  # only use the last observation step
        if nbatch["observation"].shape[-1] != 3:
            # make B, num_cameras, H, W, C  from B, num_cameras, C, H, W
            # permute if pytorch
            nbatch["observation"] = np.transpose(nbatch["observation"], (0, 1, 3, 4, 2))

        # nbatch["proprio"] is B, T, 16, where B=1
        joint_positions = nbatch["proprio"][0, -1]
        batch = {
            "observation/egocentric_camera": self._resize_with_pad(
                nbatch["observation"][0, 0], RESIZE_SIZE, RESIZE_SIZE
            ),
            "observation/wrist_image_left": self._resize_with_pad(
                nbatch["observation"][0, 1], RESIZE_SIZE, RESIZE_SIZE
            ),
            "observation/wrist_image_right": self._resize_with_pad(
                nbatch["observation"][0, 2], RESIZE_SIZE, RESIZE_SIZE
            ),
            "observation/joint_position": joint_positions,
            "prompt": self.text_prompt,
        }
        try:
            action = self.policy.act(batch)
            self.last_action = action
        except:
            action = self.last_action
            print("Error in action prediction, using last action")
        # convert to absolute action and append gripper command
        # action["actions"] shape: (10, 21), joint_positions shape: (21,)
        # Need to broadcast joint_positions to match action sequence length
        target_joint_positions = action["actions"].copy()

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
            final_action[-8] = target_joint_positions[0, -8]
            final_action[-1] = target_joint_positions[0, -1]
            final_action = final_action[None]
        else:
            final_action = target_joint_positions

        return final_action[
            ..., :23
        ]  # return only the first 23 joints, which are the robot's joints (base, trunk, arms, grippers)

    def _act_receeding_temporal(self, input_obs):
        # Step 1: check if we should re-run policy
        if self.step_counter % self.replan_interval == 0:
            # Run policy every K steps
            nbatch = copy.deepcopy(input_obs)
            if nbatch["observation"].shape[-1] != 3:
                nbatch["observation"] = np.transpose(nbatch["observation"], (0, 1, 3, 4, 2))

            joint_positions = nbatch["proprio"][0]
            batch = {
                "observation/egocentric_camera": self._resize_with_pad(
                    nbatch["observation"][0, 0].numpy(), RESIZE_SIZE, RESIZE_SIZE
                ),
                "observation/wrist_image_left": self._resize_with_pad(
                    nbatch["observation"][0, 1].numpy(), RESIZE_SIZE, RESIZE_SIZE
                ),
                "observation/wrist_image_right": self._resize_with_pad(
                    nbatch["observation"][0, 2].numpy(), RESIZE_SIZE, RESIZE_SIZE
                ),
                "observation/joint_position": joint_positions,
                "prompt": self.text_prompt,
            }

            try:
                action = self.policy.act(batch)
                self.last_action = action
            except:
                action = self.last_action
                print("Error in action prediction, using last action")

            target_joint_positions = action["actions"].copy()

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
        final_action[-8] = actions_current_timestep[0, -8]
        final_action[-1] = actions_current_timestep[0, -1]
        final_action = final_action[None]

        self.step_counter += 1

        return final_action[..., :23]
