# =============================================================================
# This file contains code derived from the following third-party projects.
#
#   [rsl_rl / legged_gym]
#     License  : BSD-3-Clause
#     Copyright: Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
#
#   [fudan_rl_wheel_leg]
#     URL      : https://github.com/yly-true/fudan_rl_wheel_leg
#     说明     : 序列版 runner 移植自该仓库 jump 分支的 OnPolicyRunner；负责把 latent_dim
#                加到 critic 维度、用展平历史观测喂 encoder，走 PPOSequence。
#
# =============================================================================

"""ActorCriticSequence + PPOSequence 专用 runner（fudan 移植，dict 观测约定）。

与 DreamWaQ runner 一致的接线方式（``object.__init__`` 旁路、从 env 的
``num_observations`` dict 取 policy / policy_hist / critic 维度、自定义 save/load），
差异在于：
- critic 维度 = ``num_privileged_obs + latent_dim``（latent 在算法侧拼进 critic 观测）。
- encoder 输入维度 = ``num_observations['policy_hist']``（= hist_len × policy 维度）。
- 显式向 ``PPOSequence.init_storage`` 传 obs / critic / history / action 四个 shape。
- checkpoint 额外存 ``extra_optimizer_state_dict``（encoder 优化器）。
"""

from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque

import rsl_rl
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import store_code_state

from agent_rl.rsl_rl.env import VecEnv
from agent_rl.rsl_rl.algorithms import PPOSequence
from agent_rl.rsl_rl.modules import ActorCriticSequence


def _sequence_tensorboard_tag(key: str) -> str:
    """Route legacy environment keys into compact TensorBoard groups."""
    prefixes = (
        ("Episode/Reward/", "RewardTerms/"),
        ("Episode/Reset/", "Termination/Count/"),
        ("Episode/FDU_L0Boundary/", "FDU/L0/Episode/"),
    )
    for source, target in prefixes:
        if key.startswith(source):
            return target + key[len(source) :]
    if "/" in key:
        return key
    return "EpisodeSummary/" + key


_SEQUENCE_CONSOLE_KEYS = {
    "Episode/Reset/terminate",
    "Episode/Reset/time_out",
    "Termination/Count/orientation",
    "Termination/Count/contact",
    "Termination/Count/numerical_safety",
    "Termination/Count/terrain_boundary",
    "Episode/FDU_L0Boundary/affected_env_fraction",
}


class OnPolicySequenceRunner(OnPolicyRunner):
    """带历史序列 encoder 的 on-policy runner。"""

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        object.__init__(self)

        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        # check if multi-gpu is enabled
        self._configure_multi_gpu()

        # ---- 维度推断 ----
        num_obs = self.env.num_observations["policy"]
        if self.env.num_privileged_obs is not None:
            base_critic_obs = self.env.num_privileged_obs
        else:
            base_critic_obs = num_obs
        # 历史观测（展平）维度：encoder 输入
        num_encoder_obs = self.env.num_observations.get(
            "policy_hist",
            self.env.unwrapped.cfg.num_obs_hist * num_obs,
        )
        latent_dim = self.policy_cfg["latent_dim"]
        # critic 观测在算法侧会拼上 latent，故 module 的 num_critic_obs 要含 latent_dim
        num_critic_obs = base_critic_obs + latent_dim

        actor_critic_class = eval(self.policy_cfg.pop("class_name"))  # ActorCriticSequence
        policy: ActorCriticSequence = actor_critic_class(
            num_obs,
            num_critic_obs,
            self.env.num_actions,
            num_encoder_obs,
            **self.policy_cfg,
        ).to(self.device)
        alg_class = eval(self.alg_cfg.pop("class_name"))  # PPOSequence
        self.alg: PPOSequence = alg_class(policy, device=self.device, **self.alg_cfg)

        # store training configuration
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.empirical_normalization = self.cfg["empirical_normalization"]
        self.experiment_name = self.cfg["experiment_name"]
        self.save_interval = self.cfg["save_interval"]
        self.save_best_after = self.cfg.get("save_best_after", self.save_interval)
        self.save_best_interval = self.cfg.get("save_best_interval", 0)
        self.save_best_epoch = 0

        # init storage：显式传 obs / critic(含 latent) / history / action 的 shape
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_critic_obs],
            [num_encoder_obs],
            [self.env.num_actions],
        )

        # We only log from the process with rank 0 (main process)
        self.disable_logs = self.is_distributed and self.gpu_global_rank != 0

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from agent_rl.rsl_rl.utils.wandb_utils import WandbSummaryWriterExt as WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")

        obs_dict, extras = self.env.reset()
        # Randomize after reset: DirectRLEnv.reset() clears episode_length_buf.
        # Fudan's runner uses this ordering so the first rollout is genuinely
        # staggered instead of synchronizing all episode completions.
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs_dict = {k: v.to(self.device) for k, v in obs_dict.items()}
        self.alg.policy.train()

        # Book keeping
        ep_infos = []
        cur_score = best_score = -float("inf")
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs_dict)
                    obs_dict, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs_dict = {k: v.to(self.device) for k, v in obs_dict.items()}
                    rewards, dones = rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, extras, obs_dict)

                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(obs_dict)

            # Update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Logging info and save checkpoint
            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if len(rewbuffer) > max(0.5 * rewbuffer.maxlen, 1):
                    cur_score = statistics.mean(rewbuffer)
                if (
                    it > self.save_best_after
                    and it > self.save_best_epoch + self.save_best_interval
                    and cur_score > best_score
                ):
                    best_score = cur_score
                    self.save_best_epoch = it
                    self.save(
                        os.path.join(self.log_dir, f"{self.experiment_name}.pt"),
                        infos={"score": best_score},
                        is_best=True,
                    )
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"), infos={"score": cur_score})

            ep_infos.clear()
            if it == start_iter and not self.disable_logs:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        # Save the final model after training.
        if self.log_dir is not None and not self.disable_logs:
            self.save(
                os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"),
                infos={"score": cur_score},
            )

    def log(self, locs, width=80, pad=35):
        """Log WYW training with compact terminal output and split TB groups."""
        collection_size = self.num_steps_per_env * self.env.num_envs * getattr(
            self, "gpu_world_size", 1
        )
        self.tot_timesteps += collection_size
        iteration_time = locs["collection_time"] + locs["learn_time"]
        self.tot_time += iteration_time

        episode_values: dict[str, torch.Tensor] = {}
        if locs["ep_infos"]:
            all_keys = sorted({key for info in locs["ep_infos"] for key in info})
            for key in all_keys:
                values = []
                for info in locs["ep_infos"]:
                    if key not in info:
                        continue
                    value = info[key]
                    value = value if isinstance(value, torch.Tensor) else torch.tensor(float(value))
                    values.append(value.detach().float().reshape(-1).to(self.device))
                if not values:
                    continue
                mean_value = torch.cat(values).mean()
                episode_values[key] = mean_value
                self.writer.add_scalar(_sequence_tensorboard_tag(key), mean_value, locs["it"])

        loss_dict = locs.get("loss_dict") or {}
        for key, value in loss_dict.items():
            scalar = value.item() if isinstance(value, torch.Tensor) else float(value)
            if key.startswith("encoder_"):
                _, metric = key.split("encoder_", 1)
                self.writer.add_scalar("Encoder/" + metric.replace("_", "/"), scalar, locs["it"])
            else:
                self.writer.add_scalar(f"Loss/{key}", scalar, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])

        mean_std = self.alg.policy.action_std.mean()
        fps = int(collection_size / iteration_time) if iteration_time > 0 else 0
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection_time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        if locs["rewbuffer"]:
            mean_reward = statistics.mean(locs["rewbuffer"])
            mean_length = statistics.mean(locs["lenbuffer"])
            self.writer.add_scalar("Train/mean_reward", mean_reward, locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", mean_length, locs["it"])
            if self.logger_type != "wandb":
                self.writer.add_scalar("Train/mean_reward/time", mean_reward, self.tot_time)
                self.writer.add_scalar("Train/mean_episode_length/time", mean_length, self.tot_time)

        tot_iter = locs.get("tot_iter", locs.get("num_learning_iterations", 0))
        title = f" \033[1m Learning iteration {locs['it']}/{tot_iter} \033[0m "
        lines = [
            "#" * width,
            title.center(width, " "),
            "",
            f"{'Computation:':>{pad}} {fps} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)",
            f"{'Mean action noise std:':>{pad}} {mean_std.item():.2f}",
        ]
        for key in ("value_function", "surrogate", "encoder", "mean_kl"):
            if key in loss_dict:
                lines.append(f"{f'Mean {key} loss:':>{pad}} {float(loss_dict[key]):.4f}")
        if locs["rewbuffer"]:
            lines.extend(
                (
                    f"{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}",
                    f"{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}",
                )
            )
        # Keep the terminal useful for reward shaping: TensorBoard gets the
        # compact ``RewardTerms/*`` group, while stdout prints every reward
        # term reported by the environment.
        for key in sorted(episode_values):
            if key.startswith("Episode/Reward/"):
                lines.append(
                    f"{f'{_sequence_tensorboard_tag(key)}:':>{pad}} {episode_values[key]:.4f}"
                )
        for key in sorted(_SEQUENCE_CONSOLE_KEYS):
            if key in episode_values:
                lines.append(f"{f'{_sequence_tensorboard_tag(key)}:':>{pad}} {episode_values[key]:.4f}")

        start_iter = locs.get("start_iter", 0)
        completed_iterations = max(locs["it"] - start_iter + 1, 1)
        remaining_iterations = max(tot_iter - locs["it"] - 1, 0)
        eta_seconds = self.tot_time / completed_iterations * remaining_iterations
        lines.extend(
            (
                "-" * width,
                f"{'Total timesteps:':>{pad}} {self.tot_timesteps}",
                f"{'Iteration time:':>{pad}} {iteration_time:.2f}s",
                f"{'Time elapsed:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(self.tot_time))}",
                f"{'ETA:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(eta_seconds))}",
            )
        )
        print("\n".join(lines) + "\n")

    def save(self, path, infos=None, is_best=False):
        torch.save(
            {
                "model_state_dict": self.alg.policy.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "extra_optimizer_state_dict": self.alg.extra_optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs and is_best:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = False, load_iteration: bool = False):
        loaded_dict = torch.load(path, weights_only=False, map_location=self.device)
        self.alg.policy.load_state_dict(loaded_dict["model_state_dict"], strict=False)
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            if "extra_optimizer_state_dict" in loaded_dict:
                self.alg.extra_optimizer.load_state_dict(loaded_dict["extra_optimizer_state_dict"])
        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.alg.policy.eval()
        if device is not None:
            self.alg.policy.to(device)

        def inference_policy(obs_dict):
            obs, obs_history, _ = self.alg._split_obs(obs_dict)
            actions, _ = self.alg.policy.act_inference(obs, obs_history)
            return actions

        return inference_policy
