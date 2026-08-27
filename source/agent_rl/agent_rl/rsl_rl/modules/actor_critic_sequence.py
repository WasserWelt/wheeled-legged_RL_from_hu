# =============================================================================
# This file contains code derived from the following third-party projects.
#
#   [rsl_rl / legged_gym]
#     License  : BSD-3-Clause
#     Copyright: Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
#
#   [fudan_rl_wheel_leg]
#     URL      : https://github.com/yly-true/fudan_rl_wheel_leg
#     说明     : ActorCriticSequence（历史 encoder → latent，latent 作为隐式基座
#                线速度估计器，对 actor 输入做 detach）忠实移植自该仓库 jump 分支。
#
# =============================================================================

"""带历史序列 encoder 的 Actor-Critic（fudan 移植）。

结构：
- ``encoder``：把展平的本体观测历史 (num_encoder_obs = hist_len * num_obs) 编码成
  ``latent_dim`` 维 latent；其前 3 维在算法侧被监督为基座线速度（隐式估计器）。
- ``actor``：输入 ``cat(obs, latent.detach())``，latent 对 actor **detach**，梯度不回传 encoder。
- ``critic``：输入 ``critic_obs``（算法侧会把 latent 拼到 critic_obs 后再送入，
  故 ``num_critic_obs`` 需 = 基础特权维 + latent_dim）。

encoder 由 PPO 的独立 ``extra_optimizer`` 训练（见 ``PPOSequence``）。
"""

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCriticSequence(nn.Module):
    is_recurrent = False
    is_sequence = True

    def __init__(
        self,
        num_obs,
        num_critic_obs,
        num_actions,
        num_encoder_obs,
        latent_dim,
        encoder_hidden_dims=[128, 64],
        actor_hidden_dims=[128, 64, 32],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
        orthogonal_init=False,
        init_noise_std=0.5,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticSequence.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.orthogonal_init = orthogonal_init
        self.latent_dim = latent_dim

        activation = get_activation(activation)

        # Encoder: 展平历史观测 -> latent
        encoder_layers = []
        encoder_layers.append(nn.Linear(num_encoder_obs, encoder_hidden_dims[0]))
        if self.orthogonal_init:
            torch.nn.init.orthogonal_(encoder_layers[-1].weight, np.sqrt(2))
        encoder_layers.append(activation)
        for l in range(len(encoder_hidden_dims)):
            if l == len(encoder_hidden_dims) - 1:
                encoder_layers.append(nn.Linear(encoder_hidden_dims[l], self.latent_dim))
                if self.orthogonal_init:
                    torch.nn.init.orthogonal_(encoder_layers[-1].weight, 0.01)
                    torch.nn.init.constant_(encoder_layers[-1].bias, 0.0)
            else:
                encoder_layers.append(
                    nn.Linear(encoder_hidden_dims[l], encoder_hidden_dims[l + 1])
                )
                if self.orthogonal_init:
                    torch.nn.init.orthogonal_(encoder_layers[-1].weight, np.sqrt(2))
                    torch.nn.init.constant_(encoder_layers[-1].bias, 0.0)
                encoder_layers.append(activation)
        self.encoder = nn.Sequential(*encoder_layers)

        # Policy: cat(obs, latent) -> action mean
        actor_layers = []
        actor_layers.append(nn.Linear(num_obs + self.latent_dim, actor_hidden_dims[0]))
        if self.orthogonal_init:
            torch.nn.init.orthogonal_(actor_layers[-1].weight, np.sqrt(2))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
                if self.orthogonal_init:
                    torch.nn.init.orthogonal_(actor_layers[-1].weight, 0.01)
                    torch.nn.init.constant_(actor_layers[-1].bias, 0.0)
            else:
                actor_layers.append(
                    nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1])
                )
                if self.orthogonal_init:
                    torch.nn.init.orthogonal_(actor_layers[-1].weight, np.sqrt(2))
                    torch.nn.init.constant_(actor_layers[-1].bias, 0.0)
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function: critic_obs(+latent, 由算法侧拼接) -> value
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
                if self.orthogonal_init:
                    torch.nn.init.orthogonal_(critic_layers[-1].weight, 0.01)
                    torch.nn.init.constant_(critic_layers[-1].bias, 0.0)
            else:
                critic_layers.append(
                    nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1])
                )
                if self.orthogonal_init:
                    torch.nn.init.orthogonal_(critic_layers[-1].weight, np.sqrt(2))
                    torch.nn.init.constant_(critic_layers[-1].bias, 0.0)
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Encoder MLP: {self.encoder}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(
                mod for mod in sequential if isinstance(mod, nn.Linear)
            )
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, observation_history):
        self.latent = self.encoder(observation_history)
        # latent 对 actor detach：actor 的梯度不流入 encoder，encoder 由 extra_optimizer 单独训练
        mean = self.actor(torch.cat((observations, self.latent.detach()), dim=-1))
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, observation_history, **kwargs):
        self.update_distribution(observations, observation_history)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_latent(self, **kwargs):
        return self.latent

    def act_inference(self, observations, observation_history):
        self.latent = self.encoder(observation_history)
        actions_mean = self.actor(torch.cat((observations, self.latent), dim=-1))
        return actions_mean, self.latent

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def encode(self, observation_history, **kwargs):
        latent = self.encoder(observation_history)
        return latent


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
