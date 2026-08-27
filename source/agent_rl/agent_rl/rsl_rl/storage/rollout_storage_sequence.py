# =============================================================================
# This file contains code derived from the following third-party projects.
#
#   [rsl_rl / legged_gym]
#     License  : BSD-3-Clause
#     Copyright: Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
#
#   [fudan_rl_wheel_leg]
#     URL      : https://github.com/yly-true/fudan_rl_wheel_leg
#     说明     : 序列版 RolloutStorage 移植自该仓库；额外存储 observation_history 与
#                next_observations，供 encoder（隐式线速度估计器）的独立优化使用。
#
# =============================================================================

"""ActorCriticSequence 专用的 RolloutStorage。

相对 stock RolloutStorage 多存两块：
- ``observation_history``：展平的本体观测历史，用于 encoder。
- ``next_observations``：下一步本体观测（此处未参与损失，保留以对齐 fudan 接口）。

``privileged_observations`` 存的是**已拼接 latent** 的 critic 观测（拼接在算法侧完成）。
不含 RNN / hidden-state 逻辑（ActorCriticSequence 非循环网络）。
"""

from __future__ import annotations

import torch


class RolloutStorageSequence:
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.next_observations = None
            self.observation_history = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        obs_history_shape,
        actions_shape,
        device="cpu",
    ):
        self.device = device

        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape

        # Core
        self.observations = torch.zeros(
            num_transitions_per_env, num_envs, *obs_shape, device=self.device
        )
        self.next_observations = torch.zeros(
            num_transitions_per_env, num_envs, *obs_shape, device=self.device
        )
        self.observation_history = torch.zeros(
            num_transitions_per_env, num_envs, *obs_history_shape, device=self.device
        )
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(
                num_transitions_per_env,
                num_envs,
                *privileged_obs_shape,
                device=self.device,
            )
        else:
            self.privileged_observations = None
        self.rewards = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.actions = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )
        self.dones = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        ).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.values = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.returns = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.advantages = torch.zeros(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.mu = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )
        self.sigma = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        self.observation_history[self.step].copy_(transition.observation_history)
        self.next_observations[self.step].copy_(transition.next_observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = (
                self.rewards[step]
                + next_is_not_terminal * gamma * next_values
                - self.values[step]
            )
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std() + 1e-8
        )

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (
                flat_dones.new_tensor([-1], dtype=torch.int64),
                flat_dones.nonzero(as_tuple=False)[:, 0],
            )
        )
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size, requires_grad=False, device=self.device
        )

        observations = self.observations.flatten(0, 1)
        observations_history = self.observation_history.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                obs_history_batch = observations_history[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                yield (
                    obs_batch,
                    obs_history_batch,
                    critic_observations_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                )

    def encoder_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size, requires_grad=False, device=self.device
        )

        next_observations = self.next_observations.flatten(0, 1)
        critic_observations = self.privileged_observations.flatten(0, 1)
        obs_history = self.observation_history.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                next_obs_batch = next_observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                obs_history_batch = obs_history[batch_idx]
                yield next_obs_batch, critic_observations_batch, obs_history_batch
