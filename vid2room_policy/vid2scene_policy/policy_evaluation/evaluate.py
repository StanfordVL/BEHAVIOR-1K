import sys
sys.path.insert(0, "/home/yalcintr/workspace/lerobot/src")

import argparse
import os
import json
import numpy as np
import torch as th
import yaml
from pathlib import Path

import omnigibson as og
from omnigibson.macros import gm

gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = True


def load_policy(checkpoint_path, policy_type, device="cuda"):
    pretrained_path = Path(checkpoint_path) / "pretrained_model"

    if policy_type == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy
        policy = ACTPolicy.from_pretrained(str(pretrained_path))
    else:
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        policy = DiffusionPolicy.from_pretrained(str(pretrained_path))

    policy.to(device)
    policy.eval()
    return policy


def prepare_batch(obs, env, config, device="cuda"):
    batch = {}
    robot = env.robots[0]
    proprio_dict = robot._get_proprioception_dict()
    state = proprio_dict["joint_qpos"]
    if isinstance(state, th.Tensor):
        state = state.cpu().numpy()
    batch["observation.state"] = th.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)

    robot_key = list(obs.keys())[0]
    robot_obs = obs[robot_key]

    env_imgs = []
    for sensor_key in robot_obs:
        sensor_data = robot_obs[sensor_key]
        if isinstance(sensor_data, dict) and "rgb" in sensor_data:
            img = sensor_data["rgb"]
            if isinstance(img, th.Tensor):
                img = img.cpu().numpy()
            if img.shape[-1] == 4:
                img = img[..., :3]
            img = img.astype(np.float32) / 255.0
            env_imgs.append(th.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device))
        elif isinstance(sensor_data, (np.ndarray, th.Tensor)):
            img = sensor_data
            if isinstance(img, th.Tensor):
                img = img.cpu().numpy()
            if len(img.shape) == 3 and img.shape[-1] in [3, 4]:
                if img.shape[-1] == 4:
                    img = img[..., :3]
                img = img.astype(np.float32) / 255.0
                env_imgs.append(th.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device))

    for i, key in enumerate(config.image_features):
        if i < len(env_imgs):
            batch[key] = env_imgs[i]
        elif len(env_imgs) > 0:
            batch[key] = env_imgs[0]

    return batch


def run_evaluation(policy, env, num_steps=1000, save_dir="rollouts"):
    os.makedirs(save_dir, exist_ok=True)
    device = policy.config.device

    obs, _ = env.reset()
    policy.reset()

    rollout_data = {"actions": [], "rewards": []}

    for step in range(num_steps):
        batch = prepare_batch(obs, env, policy.config, device)

        with th.no_grad():
            action = policy.select_action(batch)

        action_np = action.squeeze(0).cpu().numpy()
        obs, reward, done, truncated, info = env.step(action_np)

        rollout_data["actions"].append(action_np.tolist())
        rollout_data["rewards"].append(float(reward))

        if step % 100 == 0:
            print(f"Step {step}/{num_steps}, reward: {reward}")

        if done:
            print(f"Episode done at step {step}")
            obs, _ = env.reset()
            policy.reset()

    save_path = os.path.join(save_dir, "rollout.json")
    with open(save_path, "w") as f:
        json.dump(rollout_data, f)
    print(f"Saved rollout to {save_path}")

    return rollout_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_architecture", type=str, required=True, choices=["act", "diffusion"])
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default="rollouts")
    args = parser.parse_args()

    policy = load_policy(args.checkpoint_path, args.policy_architecture, args.device)

    config_filename = os.path.join(og.example_config_path, "stretch_vid2scene.yaml")
    og_config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    og_config["scene"]["load_object_categories"] = ["floors", "walls", "ceilings"]

    env = og.Environment(configs=og_config)
    run_evaluation(policy, env, args.num_steps, args.save_dir)
    og.shutdown()


if __name__ == "__main__":
    main()
