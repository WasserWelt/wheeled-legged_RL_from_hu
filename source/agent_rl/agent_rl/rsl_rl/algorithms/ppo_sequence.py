# =============================================================================
# This file contains code derived from the following third-party projects.
#
#   [rsl_rl / legged_gym]
#     License  : BSD-3-Clause
#     Copyright: Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
#
#   [fudan_rl_wheel_leg]
#     URL      : https://github.com/yly-true/fudan_rl_wheel_leg
#     说明     : 序列版 PPO 移植自该仓库；在标准 PPO 之外用独立 extra_optimizer 训练
#                encoder，使 latent[:, :3] 逼近基座线速度（critic_obs[:, :3]）。
#
# =============================================================================

"""ActorCriticSequence 专用 PPO（fudan 移植，适配 from_hu 的 dict 观测约定）。

与 stock PPO 的差异：
1. **双优化器**：``optimizer`` 训练 actor+critic+std；``extra_optimizer`` 单独训练 encoder。
2. **critic 拼 latent**：在 ``act`` 里把 ``latent`` 拼到 critic 观测后再评估价值，
   故 module 的 ``num_critic_obs`` = 基础特权维 + latent_dim。
3. **encoder 监督**：``extra`` 损失 = MSE(latent[:, :3], critic_obs[:, :3])，
   critic_obs 前 3 维必须是基座线速度（由 env 的观测布局保证）。

观测以 dict 形式传入（键 ``policy`` / ``policy_hist`` / ``critic``），
与 from_hu 的 RslRlVecEnvWrapper / 自定义 runner 约定一致。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from agent_rl.rsl_rl.storage.rollout_storage_sequence import RolloutStorageSequence


class PPOSequence:
    def __init__(
        self,
        policy,
        num_learning_epochs=5,
        num_mini_batches=4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        learning_rate=1e-3,
        extra_learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="adaptive",
        desired_kl=0.005,
        device="cpu",
        **kwargs,
    ):
        if kwargs:
            print(
                "PPOSequence.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.policy = policy
        self.policy.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(
            [
                {"params": self.policy.actor.parameters()},
                {"params": self.policy.critic.parameters()},
                {"params": self.policy.std},
            ],
            lr=learning_rate,
        )
        # encoder（隐式线速度估计器）单独优化
        self.extra_optimizer = optim.Adam(
            [{"params": self.policy.encoder.parameters()}],
            lr=extra_learning_rate,
        )
        self.transition = RolloutStorageSequence.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        obs_history_shape,
        action_shape,
    ):
        self.storage = RolloutStorageSequence(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            obs_history_shape,
            action_shape,
            self.device,
        )

    def test_mode(self):
        self.policy.eval()

    def train_mode(self):
        self.policy.train()

    # ------------------------------------------------------------------ #
    # dict 观测的解包助手
    # ------------------------------------------------------------------ #
    @staticmethod
    def _split_obs(obs_dict):
        obs = obs_dict["policy"]
        obs_history = obs_dict["policy_hist"]
        critic_obs = obs_dict.get("critic", obs)
        return obs, obs_history, critic_obs

    def act(self, obs_dict):
        obs, obs_history, critic_obs = self._split_obs(obs_dict)
        # 采样动作（同时计算 latent）
        self.transition.actions = self.policy.act(obs, obs_history).detach()
        latent = self.policy.get_latent().detach()
        # critic 观测拼上 latent 再评估价值（fudan 做法）；latent detach，存入 storage 的
        # critic 观测不携带 encoder 计算图（encoder 由 extra_optimizer 在 update 里单独训练）
        critic_obs = torch.cat((critic_obs, latent), dim=-1)
        self.transition.values = self.policy.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # 记录 step 前的观测
        self.transition.observations = obs.clone()
        self.transition.observation_history = obs_history.clone()
        self.transition.critic_observations = critic_obs.clone()
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, next_obs_dict):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # 超时自举（bootstrap on time outs）
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )
        # 记录 transition
        self.transition.next_observations = next_obs_dict["policy"]
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_obs_dict):
        _, last_obs_history, last_critic_obs = self._split_obs(last_obs_dict)
        last_critic_obs = torch.cat(
            (last_critic_obs, self.policy.encode(last_obs_history)), dim=-1
        )
        last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_kl = 0
        num_updates = 0

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )
        for (
            obs_batch,
            obs_history_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in generator:
            self.policy.act(obs_batch, obs_history_batch)
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(critic_obs_batch)
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            # KL / 自适应学习率
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                    + (
                        torch.square(old_sigma_batch)
                        + torch.square(old_mu_batch - mu_batch)
                    )
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = torch.mean(kl)

                if self.desired_kl is not None and self.schedule == "adaptive":
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_kl += kl_mean.item()
            num_updates += 1

        # ---- encoder（隐式线速度估计器）单独更新 ----
        num_updates_extra = 0
        mean_extra_loss = 0
        generator = self.storage.encoder_mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        )
        for next_obs_batch, critic_obs_batch, obs_history_batch in generator:
            latent_batch = self.policy.encode(obs_history_batch)
            vel_est_loss = (
                (latent_batch[:, :3] - critic_obs_batch[:, :3]).pow(2).mean()
            )
            if self.policy.latent_dim > 3:
                obs_denoise_loss = (
                    (
                        latent_batch[:, 3 : self.policy.latent_dim]
                        - critic_obs_batch[:, 3 : self.policy.latent_dim]
                    )
                    .pow(2)
                    .mean()
                )
                extra_loss = vel_est_loss + obs_denoise_loss
            else:
                extra_loss = vel_est_loss

            self.extra_optimizer.zero_grad()
            extra_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.1)
            self.extra_optimizer.step()

            mean_extra_loss += extra_loss.item()
            num_updates_extra += 1

        mean_value_loss /= max(num_updates, 1)
        mean_surrogate_loss /= max(num_updates, 1)
        mean_kl /= max(num_updates, 1)
        if num_updates_extra > 0:
            mean_extra_loss /= num_updates_extra
        self.storage.clear()

        return {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "encoder": mean_extra_loss,
            "mean_kl": mean_kl,
        }
