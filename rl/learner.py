import argparse
import subprocess
import uuid
import torch as th

from stable_baselines3 import PPO
from dask.distributed import Client

class PolicyWrapper():
    def __init__(self):
        self.PPO = PPO('MlpPolicy', 'CartPole-v1', verbose=1, device='cuda')

    def _read_and_concatanate_rollouts(self, rollout_paths):
        pass

    def _collect_rollouts(self, rollout_paths):
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.PPO.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()

        while n_steps < n_rollout_steps:

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            self._update_info_buffer(infos)
            n_steps += 1

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)


    def _learn(self):

        total_timesteps, callback = self.PPO._setup_learn(
            total_timesteps,
            None,
            True,
            "OnPolicyAlgorithm",
            False,
        )

        self._train()


    def _train(self, rollout_paths):
        # Switch to train mode (this affects batch norm / dropout)
        self.PPO.policy.set_training_mode(True)
        # Update optimizer learning rate
        self.PPO._update_learning_rate(self.PPO.policy.optimizer)
        # Compute current clip range
        clip_range = self.PPO.clip_range(self.PPO._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.PPO.clip_range_vf is not None:
            clip_range_vf = self.PPO.clip_range_vf(self.PPO._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.PPO.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.PPO.rollout_buffer.get(self.PPO.batch_size):
                actions = rollout_data.actions

                # Re-sample the noise matrix because the log_std has changed
                if self.PPO.use_sde:
                    self.PPO.policy.reset_noise(self.PPO.batch_size)

                values, log_prob, entropy = self.PPO.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.PPO.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.PPO.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.PPO.ent_coef * entropy_loss + self.PPO.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.PPO.target_kl is not None and approx_kl_div > 1.5 * self.PPO.target_kl:
                    continue_training = False
                    if self.PPO.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.PPO.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.PPO.policy.parameters(), self.PPO.max_grad_norm)
                self.PPO.policy.optimizer.step()

            self.PPO._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.PPO.rollout_buffer.values.flatten(), self.PPO.rollout_buffer.returns.flatten())

        # Logs
        self.PPO.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.PPO.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.PPO.logger.record("train/value_loss", np.mean(value_losses))
        self.PPO.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.PPO.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.PPO.logger.record("train/loss", loss.item())
        self.PPO.logger.record("train/explained_variance", explained_var)
        if hasattr(self.PPO.policy, "log_std"):
            self.PPO.logger.record("train/std", th.exp(self.PPO.policy.log_std).mean().item())

        self.PPO.logger.record("train/n_updates", self.PPO._n_updates, exclude="tensorboard")
        self.PPO.logger.record("train/clip_range", clip_range)
        if self.PPO.clip_range_vf is not None:
            self.PPO.logger.record("train/clip_range_vf", clip_range_vf)


def run_rollouts(policy_path, max_rollouts_per_worker):
    rollouts_uuid = uuid.uuid4().hex
    rollouts_path = './rollouts/rollouts_{}.hdf5'.format(rollouts_uuid)

    cmd = " ".join(['OMNIGIBSON_HEADLESS=1', 'python', '-m', "learner", policy_path, rollouts_path, max_rollouts_per_worker])
    subprocess.call(cmd, shell=True)

    return rollouts_path

def main(scheduler_route, num_workers, max_rollouts):
    c = Client(scheduler_route)
    policy = PolicyWrapper()
    rollout_files = []
    while True:
        # train policy from rollouts
        policy.train(rollout_files)
        
        # save policy
        uuid = uuid.uuid4().hex
        policy_path = './policies/policy_{}'.format(uuid)
        policy.save(policy_path)

        max_rollouts_per_worker = round(max_rollouts / num_workers)

        # now call the runner with the new policy
        futures = [c.submit(run_rollouts, policy_path, max_rollouts_per_worker) for _ in range(num_workers)]
        
        # wait for all the futures
        rollout_files = [f.result() for f in futures]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run learner")
    parser.add_argument("scheduler_route")
    parser.add_argument("num_workers")
    parser.add_argument("max_rollouts")
    
    args = parser.parse_args()
    main(args.scheduler_route, args.num_workers, args.max_rollouts)
