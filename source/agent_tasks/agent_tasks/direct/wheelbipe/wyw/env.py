# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Authors:
#     Wasser_welt
# =============================================================================

"""wyw 任务族环境（fudan 设计移植，from_hu 本体）。

采用「扩展点」策略而非整体重写基类：
- 复用 :class:`WheelbipeV14Env` 的全部物理 / 命令 / 复位副作用（``_get_observations``
  里大量有状态副作用喂给下一步的 reward，必须保留）。
- ``_get_observations``：先调 ``super()`` 触发全部副作用，再用 fudan 布局的张量覆写
  ``policy`` / ``critic`` / ``policy_hist`` 三个键。base_lin_vel 放 critic 前 3 维，供
  ``PPOSequence`` 的 encoder 做隐式线速度监督。
- ``_postprocess_reward_terms``：先调 ``super()``，Jump 任务再注入 fudan 涌现式跳跃项
  （基类只对 ``cfg.rewards`` 中列出的键求和，多余 ``rew_*`` 自动丢弃，无污染）。
- ``_get_dones``：直接复用基类（已含接触 / 倾倒 / 数值安全 / 超时终止）。

自维护缓冲（不依赖基类的 obs_history）：
- ``_wyw_obs_hist``：(N, T=5, 25) 的 fudan policy 历史，滚动写入，供 encoder。
"""

from __future__ import annotations

from typing import Sequence

import torch

from agent_tasks.direct.wheelbipe.wheelbipe_V14.env import WheelbipeV14Env

from . import wyw_constants as C


class WheelbipeWywEnv(WheelbipeV14Env):
    """fudan 风格 Actor/Critic/Obs/Reward 的 wyw 环境（from_hu 本体）。"""

    # ------------------------------------------------------------------ #
    # 缓冲初始化 / 复位
    # ------------------------------------------------------------------ #
    def _ensure_wyw_buffers(self) -> None:
        """惰性创建自维护缓冲（super().__init__ 期间的首次 reset 可能早于本子类字段就绪）。"""
        if getattr(self, "_wyw_buffers_ready", False):
            return
        self._wyw_obs_hist = torch.zeros(
            self.num_envs,
            C.WYW_NUM_OBS_HIST,
            C.WYW_POLICY_OBS_DIM,
            dtype=torch.float,
            device=self.device,
        )
        # critic 特权项索引 / 基准：base_link 体索引 + 未随机化的默认关节位（用于 DR 偏差项）。
        base_idx, _ = self.robot.find_bodies("base_link")
        self._wyw_base_body_idx = int(base_idx[0]) if len(base_idx) else 0
        self._wyw_nominal_default_dof = self.robot.data.default_joint_pos[:, self._actuate_idx].clone()
        self._wyw_buffers_ready = True

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        self._ensure_wyw_buffers()
        if env_ids is None:
            self._wyw_obs_hist.zero_()
        else:
            self._wyw_obs_hist[env_ids] = 0.0

    # ------------------------------------------------------------------ #
    # 观测组装（fudan 布局）
    # ------------------------------------------------------------------ #
    def _get_wyw_command_block(self) -> torch.Tensor:
        """命令块 [vx, yaw_rate, height]，各自缩放。"""
        vx = self.command[:, 0:1] * self.cfg.wyw_lin_vel_scale
        yaw = self.command[:, 2:3] * self.cfg.wyw_cmd_ang_vel_scale
        height = self._get_observation_height_cmd().unsqueeze(-1) * self.cfg.wyw_height_cmd_scale
        return torch.cat([vx, yaw, height], dim=-1)

    def _build_wyw_policy_obs(self) -> torch.Tensor:
        """fudan actor 观测（25 维），用基类算好的延迟 / 带噪副本。"""
        leg = C.WYW_LEG_DIM
        cmd = self._get_wyw_command_block()
        obs = torch.cat(
            [
                self.obs_root_ang_vel_b * self.cfg.wyw_ang_vel_scale,        # 3
                self.obs_projected_gravity_b * self.cfg.wyw_proj_gravity_scale,  # 3
                cmd,                                                         # 3
                self.obs_joint_pos[:, :leg] * self.cfg.wyw_joint_pos_scale,  # 4
                self.obs_joint_vel * self.cfg.wyw_dof_vel_scale,             # 6
                self._actions,                                               # 6 (原始动作，fudan obs 不缩放)
            ],
            dim=-1,
        )
        return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_wyw_priv_dr_obs(self) -> torch.Tensor:
        """fudan critic 尾部的域随机化特权块（12 维，未做 obs 缩放，与 fudan 一致）。

        组成：base_mass_dev(1) + base_com(3) + default_dof_delta(6) + friction(1) + restitution(1)。
        对应 fudan 的 (base_mass-mean) / base_com / (default_dof_pos-raw) / friction_coef /
        restitution_coef。from_hu 若某项未随机化（如 default_dof）则该段恒为 0，仅占位保维度。
        """
        n = self.num_envs
        base_idx = self._wyw_base_body_idx

        # base_mass 偏差 = 当前质量 − 默认（未随机化）质量
        masses = self._get_body_masses_tensor()
        default_mass = getattr(self.robot.data, "default_mass", None)
        if masses is not None and default_mass is not None:
            default_mass = torch.as_tensor(default_mass, dtype=torch.float, device=self.device)
            if default_mass.shape != masses.shape:
                default_mass = default_mass.expand_as(masses)
            base_mass_dev = (masses - default_mass)[:, base_idx].reshape(n, 1)
        else:
            base_mass_dev = torch.zeros(n, 1, dtype=torch.float, device=self.device)

        # base_com（体坐标系下质心，DR 会平移）
        body_com = getattr(self.robot.data, "body_com_pos_b", None)
        if body_com is not None:
            base_com = body_com[:, base_idx, :].reshape(n, 3)
        else:
            base_com = torch.zeros(n, 3, dtype=torch.float, device=self.device)

        # default_dof 偏差（from_hu 通常不随机化 → 0；捕获一次 nominal 后作差，随机化则自动有效）
        default_dof_delta = (
            self.robot.data.default_joint_pos[:, self._actuate_idx] - self._wyw_nominal_default_dof
        )

        # 摩擦 / 恢复系数：material_properties = [static_friction, dynamic_friction, restitution]
        material = self._get_body_material_tensor()
        if material is not None and material.ndim >= 3 and material.shape[-1] >= 3:
            friction = material[..., 0].mean(dim=1, keepdim=True)
            restitution = material[..., 2].mean(dim=1, keepdim=True)
        else:
            friction = torch.zeros(n, 1, dtype=torch.float, device=self.device)
            restitution = torch.zeros(n, 1, dtype=torch.float, device=self.device)

        dr = torch.cat(
            [base_mass_dev, base_com, default_dof_delta, friction, restitution],  # 1+3+6+1+1=12
            dim=-1,
        )
        return torch.nan_to_num(dr, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_wyw_critic_obs(self, policy_obs: torch.Tensor) -> torch.Tensor:
        """fudan critic 特权观测（141 维，拼 latent 前）。base_lin_vel 必须是前 3 维。

        逐段对齐 fudan ``privileged_obs_buf``：
        base_lin_vel(3) + obs_buf(25=policy 本体) + prev_actions(6) + before_prev_actions(6)
        + joint_acc(6) + heights(77) + torque(6) + DR 特权(12)。
        """
        base_lin_vel = self.robot.data.root_lin_vel_b * self.cfg.wyw_lin_vel_scale  # 3 (encoder 监督目标)
        joint_acc = self.robot.data.joint_acc[:, self._actuate_idx]
        torque = self.robot.data.applied_torque[:, self._actuate_idx]
        heights = self._get_scan_dot_obs()  # 77（已 ×height_scale，_pad_flat_features 保证 = n_scan）
        critic = torch.cat(
            [
                base_lin_vel,                                                # 3
                policy_obs,                                                  # 25 (= fudan obs_buf)
                self._previous_actions,                                      # 6  (last_actions[:,:,0]，不缩放)
                self._before_previous_actions,                               # 6  (last_actions[:,:,1]，不缩放)
                joint_acc * self.cfg.wyw_joint_acc_scale,                    # 6
                heights,                                                     # 77
                torque * self.cfg.wyw_torque_scale,                          # 6
                self._build_wyw_priv_dr_obs(),                               # 12
            ],
            dim=-1,
        )
        return torch.nan_to_num(critic, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_observations(self) -> dict:
        # 先跑基类：触发全部有状态副作用（obs 延迟副本、命令刷新、地面高度估计、历史 deque 等）
        observations = super()._get_observations()
        self._ensure_wyw_buffers()

        policy = self._build_wyw_policy_obs()
        # 滚动历史：index 0 最旧、-1 最新（与基类 Sim2Sim 约定一致）
        self._wyw_obs_hist = torch.roll(self._wyw_obs_hist, shifts=-1, dims=1)
        self._wyw_obs_hist[:, -1] = policy
        policy_hist = self._wyw_obs_hist.reshape(self.num_envs, -1)

        observations["policy"] = policy
        observations["policy_hist"] = policy_hist
        observations["critic"] = self._build_wyw_critic_obs(policy)
        # 清理基类可能残留的、维度与 wyw 不一致的 critic 相关键，避免下游误用
        observations.pop("prev_critic", None)
        observations.pop("critic_hist", None)
        return observations

    # ------------------------------------------------------------------ #
    # 跳跃奖励注入（Jump 任务）
    # ------------------------------------------------------------------ #
    def _compute_wyw_jump_terms(self) -> dict[str, torch.Tensor]:
        """fudan 涌现式跳跃奖励项（均为 per-step 速率，后续统一 ×step_dt）。"""
        zeros = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # 腿长 L0（缓存已在 _get_rewards 开头刷新）
        _, _, wheel_pos_heading_b, _, _ = self._get_root_quat_inv_and_wheel_pos_b()
        if wheel_pos_heading_b.shape[1] >= 2:
            l0_left = torch.norm(wheel_pos_heading_b[:, 0], dim=-1)
            l0_right = torch.norm(wheel_pos_heading_b[:, 1], dim=-1)
        else:
            l0_left = l0_right = zeros

        # 车轮接触峰值 → 滞空 / 触地判定
        net_forces = self.contact_sensor.data.net_forces_w_history
        peaks = self._get_wheel_contact_force_peaks(net_forces)
        if peaks.shape[1] >= 1:
            not_contact = peaks < C.WYW_FLIGHT_CONTACT_FORCE
            in_flight = torch.all(not_contact, dim=1)
            any_contact = torch.any(peaks > C.WYW_FLIGHT_CONTACT_FORCE, dim=1)
        else:
            in_flight = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            any_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        in_flight_f = in_flight.float()

        root_z = self.robot.data.root_pos_w[:, 2]
        vz = self.robot.data.root_lin_vel_w[:, 2]

        # 滞空期望机身高度
        base_height_flight = torch.exp(
            -torch.abs(root_z - C.WYW_BASE_HEIGHT_FLIGHT) * 6.0
        ) * in_flight_f
        # 滞空收腿
        leg_tuck = torch.exp(
            -(torch.abs(l0_left - C.WYW_L0_TUCK) + torch.abs(l0_right - C.WYW_L0_TUCK)) * 4.0
        ) * in_flight_f
        # 触地蹬伸（有轮触地且正在向上加速）
        takeoff_mask = (any_contact & (vz > C.WYW_TAKEOFF_VZ)).float()
        takeoff_extend = torch.exp(
            -(torch.abs(l0_left - C.WYW_L0_EXTEND) + torch.abs(l0_right - C.WYW_L0_EXTEND)) * 4.0
        ) * takeoff_mask
        # 滞空正竖直速度 / 平坦滞空奖励 / 高度加权滞空
        line_z = torch.clamp(vz, min=0.0) * in_flight_f
        flight = in_flight_f
        encourage_jump = torch.clamp(root_z, min=0.0, max=C.WYW_AIRTIME_HEIGHT_CLIP) * in_flight_f

        return {
            "base_height_flight": base_height_flight,
            "leg_tuck": leg_tuck,
            "takeoff_extend": takeoff_extend,
            "line_z": line_z,
            "flight": flight,
            "encourage_jump": encourage_jump,
        }

    def _postprocess_reward_terms(self, reward_terms: dict) -> dict:
        reward_terms = super()._postprocess_reward_terms(reward_terms)
        if bool(getattr(self.cfg, "wyw_jump_enabled", False)):
            reward_terms.update(self._compute_wyw_jump_terms())
        return reward_terms
