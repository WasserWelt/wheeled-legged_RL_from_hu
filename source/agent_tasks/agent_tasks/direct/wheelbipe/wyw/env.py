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

"""wyw task family on the FDU closed-chain parallel-linkage robot.

采用「扩展点」策略而非整体重写基类：
- 复用 :class:`Wheelbipe25V3Env` 的物理 / 命令 / 复位副作用（``_get_observations``
  里大量有状态副作用喂给下一步的 reward，必须保留）。
- ``_get_observations``：先调 ``super()`` 触发全部副作用，再用 fudan 布局的张量覆写
  ``policy`` / ``critic`` / ``policy_hist`` 三个键。base_lin_vel 放 critic 前 3 维，供
  ``PPOSequence`` 的 encoder 做隐式线速度监督。
- ``_get_rewards``：使用独立的 Fudan 精确公式和 term 集合，同时保留基类命令刷新、
  episode logging、数值保护和逐项裁剪契约。
- ``_get_dones``：使用 Fudan 连续倾倒判定，同时保留数值安全和超时终止。

自维护缓冲（不依赖基类的 obs_history）：
- ``_wyw_obs_hist``：(N, T=5, 25) 的 fudan policy 历史，滚动写入，供 encoder。
"""

from __future__ import annotations

from typing import Sequence

import torch

from agent_tasks.direct.wheelbipe.wheelbipe25_v3.env import Wheelbipe25V3Env

from . import wyw_constants as C
from .fdu_mapping import (
    POLICY_JOINT_NAMES,
    compute_fdu_equivalent_leg_state,
    project_fdu_leg_targets,
    update_buggy_fudan_airtime,
)


class WheelbipeWywEnv(Wheelbipe25V3Env):
    """Fudan task semantics with direct control of four entity drive bars."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        policy_indices = []
        for name in POLICY_JOINT_NAMES:
            indices, _ = self.robot.find_joints(name)
            if len(indices) != 1:
                raise RuntimeError(f"Expected exactly one FDU joint named {name!r}, got {indices}")
            policy_indices.append(int(indices[0]))
        if len(set(policy_indices)) != C.WYW_ACTION_DIM:
            raise RuntimeError(f"FDU policy joint mapping contains duplicates: {policy_indices}")

        self._wyw_policy_joint_idx = policy_indices
        self._wyw_leg_joint_idx = [policy_indices[i] for i in C.WYW_LEG_ACTION_IDS]
        self._wyw_wheel_joint_idx = [policy_indices[i] for i in C.WYW_WHEEL_ACTION_IDS]
        self._legs_act_idx = list(self._wyw_leg_joint_idx)
        self._wheel_idx = list(self._wyw_wheel_joint_idx)
        self._actuate_idx = list(self._wyw_policy_joint_idx)
        self._leg_action_dim = len(self._legs_act_idx)
        self._wheel_action_dim = len(self._wheel_idx)
        self._actuated_joint_count = len(self._actuate_idx)
        self.max_wheel_vel = float(self.cfg.max_wheel_vel)

        self._left_wheel_link_idx, _ = self.robot.find_bodies("l_wheel_Link")
        self._right_wheel_link_idx, _ = self.robot.find_bodies("r_wheel_Link")
        left_hip, _ = self.robot.find_bodies("lf0_Link")
        right_hip, _ = self.robot.find_bodies("rf0_Link")
        if len(left_hip) != 1 or len(right_hip) != 1:
            raise RuntimeError(f"Expected one FDU hip link per side, got left={left_hip}, right={right_hip}")
        self._wyw_hip_link_idx = [int(left_hip[0]), int(right_hip[0])]
        self._wheel_link_idx = list(self._left_wheel_link_idx) + list(self._right_wheel_link_idx)
        self._wheel_link_count = len(self._wheel_link_idx)
        self._desired_contact_link_idx = self._find_contact_sensor_indices(["[lr]_wheel_Link"])
        self._reset_contact_link_idx = self._find_contact_sensor_indices(["base_link_del"])
        if bool(getattr(self.cfg, "wyw_jump_enabled", False)):
            self._undesired_contact_link_idx = self._find_contact_sensor_indices(
                ["base_link_del", "[lr]f[01]_Link", "[lr]2[0-3]_Link"]
            )
        else:
            # Fudan Plane has penalize_contacts_on=[]; keep the configured
            # collision reward term for logging compatibility, but its raw
            # value must remain exactly zero.
            self._undesired_contact_link_idx = []
        self._invalidate_step_caches()
        self.static_priv_obs = self._get_static_priv_obs()
        self._wyw_buffers_ready = False
        self._ensure_wyw_buffers()
        print(f"[WYW:FDU] policy joint order: {list(zip(POLICY_JOINT_NAMES, policy_indices))}")

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._invalidate_step_caches()
        self.finish_init.fill_(True)
        self.start_reset.fill_(True)
        self._actions = actions.clone()
        if self.use_action_low_pass_filter:
            self._actions = self._low_pass_action_filter(self._actions)
        self.last_actions.copy_(self._actions)
        self.leg_actions = (
            self.robot.data.default_joint_pos[:, self._wyw_leg_joint_idx]
            + self.leg_action_scale * self._actions[:, C.WYW_LEG_ACTION_IDS]
        )
        self.wheel_actions = self.wheel_action_scale * self._actions[:, C.WYW_WHEEL_ACTION_IDS]

    def _apply_action(self) -> None:
        desired = self.leg_actions
        leg_targets = torch.stack(
            project_fdu_leg_targets(
                desired[:, 0], desired[:, 1], desired[:, 2], desired[:, 3],
                min_length=self.cfg.wyw_safe_l0_range[0],
                max_length=self.cfg.wyw_safe_l0_range[1],
                max_abs_theta=self.cfg.wyw_safe_theta0_abs,
            ),
            dim=-1,
        )
        wheel_targets = torch.clamp(self.wheel_actions, -self.max_wheel_vel, self.max_wheel_vel)
        self.robot.set_joint_position_target(leg_targets, joint_ids=self._wyw_leg_joint_idx)
        self.robot.set_joint_velocity_target(wheel_targets, joint_ids=self._wyw_wheel_joint_idx)
        self._update_obs()

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
        self._wyw_history_needs_fill = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._wyw_last_wheel_contacts = torch.zeros(
            self.num_envs, 2, dtype=torch.bool, device=self.device
        )
        self._wyw_base_air_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # critic 特权项索引 / 基准：FDU root + 未随机化的默认关节位。
        base_idx, _ = self.robot.find_bodies("base_link_del")
        self._wyw_base_body_idx = int(base_idx[0]) if len(base_idx) else 0
        self._wyw_nominal_default_dof = self.robot.data.default_joint_pos[:, self._actuate_idx].clone()
        self._wyw_command_ranges_x = torch.tensor(
            self.cfg.commands.ranges.lin_vel_x, dtype=torch.float, device=self.device
        ).repeat(self.num_envs, 1)
        self._wyw_buffers_ready = True

    def _reset_idx(self, env_ids: Sequence[int] | None):
        reset_env_ids = self._as_env_ids_tensor(env_ids)
        self._update_fdu_rough_curriculum(reset_env_ids)
        super()._reset_idx(env_ids)
        self._ensure_wyw_buffers()
        self._wyw_obs_hist[reset_env_ids] = 0.0
        self._wyw_history_needs_fill[reset_env_ids] = True
        self._sample_wyw_lin_vel_command(reset_env_ids)
        self._clear_termination_duration_buffers(
            reset_env_ids,
            counter_attr="_wyw_orientation_termination_counter",
            raw_attr="_wyw_orientation_termination_raw_buf",
        )
        # Deliberately do not clear _wyw_base_air_time or the previous-contact
        # filter. The trained Fudan Jump implementation carries both across reset.

    # ------------------------------------------------------------------ #
    # 观测组装（fudan 布局）
    # ------------------------------------------------------------------ #
    def _get_wyw_command_block(self) -> torch.Tensor:
        """命令块 [vx, yaw_rate, height]，各自缩放。"""
        vx = self.command[:, 0:1] * self.cfg.wyw_lin_vel_scale
        yaw = self.command[:, 2:3] * self.cfg.wyw_cmd_ang_vel_scale
        height = self._get_observation_height_cmd().unsqueeze(-1) * self.cfg.wyw_height_cmd_scale
        return torch.cat([vx, yaw, height], dim=-1)

    def _build_wyw_policy_obs(self, *, noisy: bool) -> torch.Tensor:
        """Build the 25-D Fudan proprioception, optionally adding actor-only noise."""
        cmd = self._get_wyw_command_block()
        policy_pos = self.obs_joint_pos[:, : C.WYW_ACTION_DIM]
        policy_vel = self.obs_joint_vel[:, : C.WYW_ACTION_DIM]
        ang_vel = self.obs_root_ang_vel_b
        gravity = self.obs_projected_gravity_b
        if noisy and self.use_self_obs_noise:
            ang_vel = self.self_obs_noise["root_ang_vel_b"](ang_vel)
            gravity = self.self_obs_noise["projected_gravity_b"](gravity)
            policy_pos = self.self_obs_noise["joint_pos"](policy_pos)
            leg_vel = self.self_obs_noise["leg_joint_vel"](policy_vel[:, C.WYW_LEG_ACTION_IDS])
            wheel_vel = self.self_obs_noise["wheel_joint_vel"](policy_vel[:, C.WYW_WHEEL_ACTION_IDS])
            policy_vel = policy_vel.clone()
            policy_vel[:, C.WYW_LEG_ACTION_IDS] = leg_vel
            policy_vel[:, C.WYW_WHEEL_ACTION_IDS] = wheel_vel
        obs = torch.cat(
            [
                ang_vel * self.cfg.wyw_ang_vel_scale,                         # 3
                gravity * self.cfg.wyw_proj_gravity_scale,                    # 3
                cmd,                                                         # 3
                policy_pos[:, C.WYW_LEG_ACTION_IDS] * self.cfg.wyw_joint_pos_scale,  # 4
                policy_vel * self.cfg.wyw_dof_vel_scale,                     # 6
                self._actions,                                               # 6 (原始动作，fudan obs 不缩放)
            ],
            dim=-1,
        )
        return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_wyw_priv_dr_obs(self) -> torch.Tensor:
        """fudan critic 尾部的域随机化特权块（12 维，未做 obs 缩放，与 fudan 一致）。

        组成：centered_base_mass(1) + sampled_com_offset(3) + default_dof_delta(6)
        + friction(1) + restitution(1)。对应 fudan 的 (base_mass-mean) / base_com /
        (default_dof_pos-raw) / friction_coef /
        restitution_coef。FDU 若某项未随机化（如 default_dof）则该段恒为 0，仅占位保维度。
        """
        n = self.num_envs
        # Fudan privilege records the sampled base addition before inertia scaling.
        base_mass_sample = getattr(self, "_wyw_base_mass_dev_sample", None)
        if base_mass_sample is not None:
            base_mass_dev = base_mass_sample.reshape(n, 1)
        else:
            base_mass_dev = torch.zeros(n, 1, dtype=torch.float, device=self.device)

        # base_com（体坐标系下质心，DR 会平移）
        base_com_sample = getattr(self, "_wyw_base_com_sample", None)
        if base_com_sample is not None:
            base_com = base_com_sample.reshape(n, 3)
        else:
            base_com = torch.zeros(n, 3, dtype=torch.float, device=self.device)

        # default_dof 偏差（FDU 通常不随机化 → 0；捕获一次 nominal 后作差，随机化则自动有效）
        default_dof_delta = getattr(self, "_wyw_default_dof_delta_sample", None)
        if default_dof_delta is None:
            default_dof_delta = self.robot.data.default_joint_pos[:, self._actuate_idx] - self._wyw_nominal_default_dof

        # 摩擦 / 恢复系数：material_properties = [static_friction, dynamic_friction, restitution]
        friction_sample = getattr(self, "_wyw_friction_sample", None)
        restitution_sample = getattr(self, "_wyw_restitution_sample", None)
        if friction_sample is not None and restitution_sample is not None:
            friction = friction_sample.reshape(n, 1)
            restitution = restitution_sample.reshape(n, 1)
        else:
            friction = torch.zeros(n, 1, dtype=torch.float, device=self.device)
            restitution = torch.zeros(n, 1, dtype=torch.float, device=self.device)

        dr = torch.cat(
            [base_mass_dev, base_com, default_dof_delta, friction, restitution],  # 1+3+6+1+1=12
            dim=-1,
        )
        return torch.nan_to_num(dr, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_wyw_critic_obs(
        self,
        policy_obs: torch.Tensor,
        previous_actions: torch.Tensor | None = None,
        before_previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """fudan critic 特权观测（141 维，拼 latent 前）。base_lin_vel 必须是前 3 维。

        逐段对齐 fudan ``privileged_obs_buf``：
        base_lin_vel(3) + obs_buf(25=policy 本体) + prev_actions(6) + before_prev_actions(6)
        + joint_acc(6) + heights(77) + torque(6) + DR 特权(12)。
        """
        base_lin_vel = self.robot.data.root_lin_vel_b * self.cfg.wyw_lin_vel_scale  # 3 (encoder 监督目标)
        joint_acc = self.robot.data.joint_acc[:, self._actuate_idx]
        torque = self.robot.data.applied_torque[:, self._actuate_idx]
        heights = self._get_fdu_height_scan_obs()
        previous_actions = self._previous_actions if previous_actions is None else previous_actions
        before_previous_actions = (
            self._before_previous_actions if before_previous_actions is None else before_previous_actions
        )
        critic = torch.cat(
            [
                base_lin_vel,                                                # 3
                policy_obs,                                                  # 25 (= fudan obs_buf)
                previous_actions,                                            # 6  (a_{t-1}，不缩放)
                before_previous_actions,                                     # 6  (a_{t-2}，不缩放)
                joint_acc * self.cfg.wyw_joint_acc_scale,                    # 6
                heights,                                                     # 77
                torque * self.cfg.wyw_torque_scale,                          # 6
                self._build_wyw_priv_dr_obs(),                               # 12
            ],
            dim=-1,
        )
        return torch.nan_to_num(critic, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_fdu_height_scan_obs(self) -> torch.Tensor:
        """Return Fudan's clip(root_z - 0.5 - terrain_z) * height_scale scan."""
        ray_hits = getattr(getattr(self.dot_scanner, "data", None), "ray_hits_w", None)
        if ray_hits is None:
            return torch.zeros(self.num_envs, C.WYW_N_SCAN, device=self.device)
        relative = self.robot.data.root_pos_w[:, 2:3] - 0.5 - ray_hits[..., 2]
        relative = self._pad_flat_features(relative, C.WYW_N_SCAN)
        return torch.nan_to_num(torch.clamp(relative, -1.0, 1.0) * self.cfg.height_scale)

    def _get_fdu_base_height(self) -> torch.Tensor:
        """Average base-to-local-ground height over the same 77 Fudan samples."""
        ray_hits = getattr(getattr(self.dot_scanner, "data", None), "ray_hits_w", None)
        if ray_hits is None:
            return self.robot.data.root_pos_w[:, 2] - self.ground_z_est
        terrain_z = ray_hits[..., 2]
        finite = torch.isfinite(terrain_z)
        mean_z = terrain_z.masked_fill(~finite, 0.0).sum(dim=-1) / finite.sum(dim=-1).clamp_min(1)
        mean_z = torch.where(finite.any(dim=-1), mean_z, self.ground_z_est)
        return self.robot.data.root_pos_w[:, 2] - mean_z

    def _get_observations(self) -> dict:
        # The V3 base advances its two action-history tensors inside
        # _get_observations(). Snapshot them first so the WYW critic retains
        # Fudan's [a_{t-1}, a_{t-2}] contract while policy still sees a_t.
        previous_actions = self._previous_actions.clone()
        before_previous_actions = self._before_previous_actions.clone()
        # 先跑基类：触发全部有状态副作用（obs 延迟副本、命令刷新、地面高度估计、历史 deque 等）
        observations = super()._get_observations()
        self._ensure_wyw_buffers()

        clean_policy = self._build_wyw_policy_obs(noisy=False)
        policy = self._build_wyw_policy_obs(noisy=True)
        # 滚动历史：index 0 最旧、-1 最新（与基类 Sim2Sim 约定一致）
        self._wyw_obs_hist = torch.roll(self._wyw_obs_hist, shifts=-1, dims=1)
        self._wyw_obs_hist[:, -1] = policy
        if self._wyw_history_needs_fill.any():
            fill_ids = self._wyw_history_needs_fill.nonzero(as_tuple=False).flatten()
            self._wyw_obs_hist[fill_ids] = policy[fill_ids].unsqueeze(1)
            self._wyw_history_needs_fill[fill_ids] = False
        policy_hist = self._wyw_obs_hist.reshape(self.num_envs, -1)

        observations["policy"] = policy
        observations["policy_hist"] = policy_hist
        observations["critic"] = self._build_wyw_critic_obs(
            clean_policy, previous_actions, before_previous_actions
        )
        # 清理基类可能残留的、维度与 wyw 不一致的 critic 相关键，避免下游误用
        observations.pop("prev_critic", None)
        observations.pop("critic_hist", None)
        if policy.shape[-1] != C.WYW_POLICY_OBS_DIM:
            raise RuntimeError(f"WYW policy observation has shape {policy.shape}, expected last dim 25")
        if policy_hist.shape[-1] != C.WYW_POLICY_HIST_DIM:
            raise RuntimeError(f"WYW policy history has shape {policy_hist.shape}, expected last dim 125")
        if observations["critic"].shape[-1] != C.WYW_CRITIC_DIM:
            raise RuntimeError(
                f"WYW critic observation has shape {observations['critic'].shape}, expected last dim 141"
            )
        return observations

    # ------------------------------------------------------------------ #
    # 跳跃奖励注入（Jump 任务）
    # ------------------------------------------------------------------ #
    def _get_wyw_virtual_leg_geometry(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return wheel poses plus analytic kite ``L0/theta0``.

        The specified FDU body is an equivalent kite mechanism driven by four
        entity bars.  Computing geometry from the actual four joint angles is
        deterministic and avoids solver/link-pose lag; the independent PhysX
        wheel positions remain available as the first return value for scans.
        """
        _, wheel_pos_b, _, _, _ = self._get_root_quat_inv_and_wheel_pos_b()
        q = self.robot.data.joint_pos[:, self._wyw_leg_joint_idx]
        left_l0, left_theta, right_l0, right_theta = compute_fdu_equivalent_leg_state(
            q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        )
        lengths = torch.stack((left_l0, right_l0), dim=-1).clamp_min(1.0e-6)
        angles = torch.stack((left_theta, right_theta), dim=-1)
        return wheel_pos_b, lengths, angles

    def _compute_wyw_jump_terms(self, leg_lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        """fudan 涌现式跳跃奖励项（均为 per-step 速率，后续统一 ×step_dt）。"""
        zeros = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        l0_left = leg_lengths[:, 0]
        l0_right = leg_lengths[:, 1]

        # Fudan check_jump: current world-z force OR previous policy frame.
        net_forces = self.contact_sensor.data.net_forces_w
        if net_forces is not None and len(self._desired_contact_link_idx) == 2:
            contact_now = (
                net_forces[:, self._desired_contact_link_idx, 2] > self.cfg.wyw_flight_contact_force
            )
            contact_filt = contact_now | self._wyw_last_wheel_contacts
            self._wyw_last_wheel_contacts.copy_(contact_now)
            in_flight = torch.all(~contact_filt, dim=1)
            any_contact = torch.any(contact_filt, dim=1)
        else:
            in_flight = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            any_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        in_flight_f = in_flight.float()

        root_z = self.robot.data.root_pos_w[:, 2]
        vz = self.robot.data.root_lin_vel_w[:, 2]

        # 滞空期望机身高度
        base_height_flight = torch.exp(
            -torch.abs(root_z - self.cfg.wyw_base_height_flight) * 6.0
        ) * in_flight_f
        # 滞空收腿
        leg_tuck = torch.exp(
            -(torch.abs(l0_left - self.cfg.wyw_l0_tuck) + torch.abs(l0_right - self.cfg.wyw_l0_tuck)) * 4.0
        ) * in_flight_f
        # 触地蹬伸（有轮触地且正在向上加速）
        takeoff_mask = (any_contact & (vz > self.cfg.wyw_takeoff_vz)).float()
        takeoff_extend = torch.exp(
            -(torch.abs(l0_left - self.cfg.wyw_l0_extend) + torch.abs(l0_right - self.cfg.wyw_l0_extend)) * 4.0
        ) * takeoff_mask
        # 滞空正竖直速度 / 平坦滞空奖励 / 高度加权滞空
        line_z = torch.clamp(vz, min=0.0) * in_flight_f
        flight = in_flight_f
        encourage_jump, self._wyw_base_air_time = update_buggy_fudan_airtime(
            self._wyw_base_air_time, in_flight, root_z, vz, self.step_dt
        )

        return {
            "base_height_flight": base_height_flight,
            "leg_tuck": leg_tuck,
            "takeoff_extend": takeoff_extend,
            "line_z": line_z,
            "flight": flight,
            "encourage_jump": encourage_jump,
        }

    def _compute_fdu_reward_terms(self) -> dict[str, torch.Tensor]:
        """Compute the named Fudan reward terms without V3 shaping gates."""
        self._update_ground_height_estimate()
        wheel_pos_b, leg_lengths, leg_angles = self._get_wyw_virtual_leg_geometry()
        left_len, right_len = leg_lengths[:, 0], leg_lengths[:, 1]
        left_theta, right_theta = leg_angles[:, 0], leg_angles[:, 1]

        sigma = max(float(getattr(self.cfg, "tracking_sigma", 0.25)), 1.0e-6)
        lin_err = torch.square(self.command[:, 0] - self.robot.data.root_lin_vel_b[:, 0])
        ang_err = torch.square(self.command[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        lin_scale = 2.0 if bool(getattr(self.cfg, "wyw_jump_enabled", False)) else 1.0
        lin_track = torch.exp(-lin_err / sigma) * lin_scale
        lin_enhance = (torch.exp(-lin_err / (10.0 * sigma)) - 1.0) * lin_scale

        observed_height = self._get_fdu_base_height()
        height_cmd = self._get_observation_height_cmd()
        height = torch.exp(-torch.square(observed_height - height_cmd) / 0.001)

        qdot = self.robot.data.joint_vel[:, self._actuate_idx]
        qddot = self.robot.data.joint_acc[:, self._actuate_idx]
        tau = self.robot.data.applied_torque[:, self._actuate_idx]
        action_delta = self._actions - self._previous_actions
        action_second = self._actions - 2.0 * self._previous_actions + self._before_previous_actions

        hard_limits = self.robot.data.joint_pos_limits[:, self._wyw_leg_joint_idx]
        centers = 0.5 * (hard_limits[..., 0] + hard_limits[..., 1])
        half_ranges = 0.5 * (hard_limits[..., 1] - hard_limits[..., 0])
        soft_lower = centers - 0.97 * half_ranges
        soft_upper = centers + 0.97 * half_ranges
        positions = self.robot.data.joint_pos[:, self._wyw_leg_joint_idx]
        pos_limit_penalty = torch.sum(
            torch.clamp(soft_lower - positions, min=0.0)
            + torch.clamp(positions - soft_upper, min=0.0),
            dim=-1,
        )

        contact_forces = getattr(self.contact_sensor.data, "net_forces_w", None)
        if contact_forces is None or not self._undesired_contact_link_idx:
            collision = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        else:
            selected = contact_forces[:, self._undesired_contact_link_idx]
            collision = (torch.linalg.vector_norm(selected, dim=-1) > 0.1).float().sum(dim=-1)

        terms = {
            "tracking_lin_vel": lin_track,
            "tracking_lin_vel_enhance": lin_enhance,
            "tracking_ang_vel": torch.exp(-ang_err / sigma),
            "tracking_ang_vel_enhance": torch.exp(-ang_err / (10.0 * sigma)) - 1.0,
            "base_height": height,
            "nominal_state": torch.square(left_theta - right_theta),
            "lin_vel_z": torch.square(self.robot.data.root_lin_vel_b[:, 2]),
            "ang_vel_xy": torch.square(self.robot.data.root_ang_vel_b[:, :2]).sum(dim=-1),
            "orientation": torch.square(self.robot.data.projected_gravity_b[:, :2]).sum(dim=-1),
            "dof_vel": torch.square(qdot[:, C.WYW_LEG_ACTION_IDS]).sum(dim=-1),
            "dof_acc": torch.square(qddot).sum(dim=-1),
            "torques": torch.square(tau).sum(dim=-1),
            "action_rate": torch.square(action_delta).sum(dim=-1),
            "action_smooth": torch.square(action_second[:, C.WYW_LEG_ACTION_IDS]).sum(dim=-1),
            "collision": collision,
            "dof_pos_limits": pos_limit_penalty,
        }
        if bool(getattr(self.cfg, "wyw_jump_enabled", False)):
            terms.update(self._compute_wyw_jump_terms(leg_lengths))
            terms["pen_theta_no0"] = torch.square(torch.stack((left_theta, right_theta), dim=-1)).sum(dim=-1)
            terms["nominal_state"] = torch.square(left_theta - right_theta) + 10.0 * torch.square(left_len - right_len)
            terms.pop("base_height", None)
            terms.pop("lin_vel_z", None)
            terms.pop("dof_vel", None)
            terms.pop("dof_acc", None)
            terms.pop("action_smooth", None)
            terms.pop("dof_pos_limits", None)
            terms.pop("tracking_ang_vel_enhance", None)
        return terms

    def _postprocess_reward_terms(self, reward_terms: dict) -> dict:
        return reward_terms

    def _get_rewards(self) -> torch.Tensor:
        """Aggregate the exact WYW/Fudan terms using the base bookkeeping contract."""
        # Fudan resamples commands in _post_physics_step_callback before
        # termination/reward computation, so a boundary step is scored against
        # the newly sampled command. DirectRLEnv does not own a command manager;
        # advance ours here to preserve that ordering.
        self._advance_fdu_commands()
        reward_terms = self._postprocess_reward_terms(self._compute_fdu_reward_terms())
        rewards = {
            key: self.cfg.rewards[key] * reward_terms.get(key, torch.zeros(self.num_envs, device=self.device)) * self.step_dt
            for key in self.cfg.rewards
        }
        clip_single = getattr(self.cfg, "clip_single_reward", None)
        if clip_single is not None:
            bound = abs(float(clip_single)) * self.step_dt
            rewards = {key: torch.clamp(value, -bound, bound) for key, value in rewards.items()}
        if bool(getattr(self.cfg, "only_positive_rewards", False)) and rewards:
            total = torch.stack(list(rewards.values()), dim=-1).sum(dim=-1)
            rewards[next(iter(rewards))] += torch.clamp_min(-total, 0.0)
        bad = self._numerical_safety_reset_buf
        for key, value in rewards.items():
            value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
            rewards[key] = torch.where(bad, torch.zeros_like(value), value)
        self._last_reward_terms = {key: value.detach() for key, value in rewards.items()}
        for key, value in rewards.items():
            self._episode_sums.setdefault(key, torch.zeros(self.num_envs, device=self.device))
            self._episode_sums[key] += value
        total_reward = torch.stack([rewards[key] for key in self.cfg.rewards], dim=-1).sum(dim=-1)
        self._last_total_reward = total_reward.detach()
        return torch.nan_to_num(total_reward, nan=0.0, posinf=0.0, neginf=0.0)

    def _advance_fdu_commands(self) -> None:
        sync_cmd_iteration = getattr(self, "_sync_command_generator_training_iteration", None)
        if sync_cmd_iteration is not None:
            sync_cmd_iteration()
        self.command_generator.compute(self.step_dt)
        self._resample_custom_cmd(self.command_generator.command_counter)
        self.command = self.command_generator.command.clone()
        self._on_command_updated()
        self._apply_predefined_reset_air_command_limits()

    def _sample_wyw_lin_vel_command(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0 or not hasattr(self, "_wyw_command_ranges_x"):
            return
        ranges = self._wyw_command_ranges_x[env_ids]
        values = ranges[:, 0] + torch.rand(env_ids.numel(), device=self.device) * (ranges[:, 1] - ranges[:, 0])
        self.command_generator.vel_command_b[env_ids, 0] = values

    def _resample_custom_cmd(self, command_counter: torch.Tensor):
        resampled = super()._resample_custom_cmd(command_counter)
        self._sample_wyw_lin_vel_command(resampled)
        return resampled

    def _update_fdu_rough_curriculum(self, env_ids: torch.Tensor) -> None:
        """Apply Fudan's episode-performance terrain/command curriculum on reset."""
        if not bool(getattr(self.cfg, "wyw_rough_curriculum_enabled", False)):
            return
        if env_ids.numel() == 0 or not getattr(self, "_wyw_buffers_ready", False):
            return
        terrain = self.terrain
        if getattr(terrain, "terrain_origins", None) is None:
            return

        origins = terrain.env_origins[env_ids]
        distance = torch.linalg.vector_norm(self.robot.data.root_pos_w[env_ids, :2] - origins[:, :2], dim=-1)
        tracking_sum = self._episode_sums.get("tracking_lin_vel")
        if tracking_sum is None:
            return
        tracking_rate = tracking_sum[env_ids] / max(float(self.max_episode_length_s), 1.0e-6)
        move_up = distance > (float(terrain.cfg.terrain_generator.size[0]) / 4.0)
        move_down = (tracking_rate < 0.4) & (~move_up)

        old_levels = terrain.terrain_levels[env_ids].clone()
        candidate_levels = old_levels + move_up.long() - move_down.long()
        success = candidate_levels >= int(terrain.max_terrain_level)
        failure = candidate_levels < 0
        terrain.update_env_origins(env_ids, move_up, move_down)

        # Fudan narrows failed ranges towards [-1, 1]. Successful terrains
        # expand by 0.05, with an additional 0.45 for basic terrain classes.
        failed_ids = env_ids[failure]
        if failed_ids.numel() > 0:
            self._wyw_command_ranges_x[failed_ids, 0] = torch.clamp(
                self._wyw_command_ranges_x[failed_ids, 0] + 0.25, min=-2.5, max=-1.0
            )
            self._wyw_command_ranges_x[failed_ids, 1] = torch.clamp(
                self._wyw_command_ranges_x[failed_ids, 1] - 0.25, min=1.0, max=2.5
            )

        successful_ids = env_ids[success & (tracking_rate > 0.7)]
        if successful_ids.numel() > 0:
            terrain_types = terrain.terrain_types[successful_ids]
            # Reproduce Fudan's executable index sets exactly. Its variable
            # names for the two stair ranges are reversed relative to the
            # generated geometry: basic={0:12,14:18}, advanced={12:14,18:20}.
            basic = (terrain_types < 12) | ((terrain_types >= 14) & (terrain_types < 18))
            delta = torch.where(basic, 0.5, 0.05)
            max_abs = torch.where(basic, 2.5, 1.5)
            self._wyw_command_ranges_x[successful_ids, 0] = torch.maximum(
                self._wyw_command_ranges_x[successful_ids, 0] - delta, -max_abs
            )
            self._wyw_command_ranges_x[successful_ids, 1] = torch.minimum(
                self._wyw_command_ranges_x[successful_ids, 1] + delta, max_abs
            )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Use Fudan's persisted tilt failure while retaining numerical safety."""
        _, time_out = super()._get_dones()
        time_out = time_out | self._get_rough_terrain_boundary_time_out()
        immediate_terminate = self._numerical_safety_reset_buf.clone()
        bad_orientation = self.robot.data.projected_gravity_b[:, 2] > -0.1
        orientation_terminate = self._apply_termination_duration(
            bad_orientation,
            counter_attr="_wyw_orientation_termination_counter",
            raw_attr="_wyw_orientation_termination_raw_buf",
        )
        terminate = immediate_terminate | orientation_terminate
        if self.cfg.play is True and not bool(getattr(self.cfg, "play_keep_done_reset", False)):
            terminate.zero_()
        return terminate, time_out

    def _get_rough_terrain_boundary_time_out(self) -> torch.Tensor:
        """Classify leaving a generated rough tile as timeout, never as failure."""
        terrain_cfg = getattr(self.cfg, "terrain", None)
        terrain_gen = getattr(terrain_cfg, "terrain_generator", None)
        if terrain_gen is None or getattr(terrain_cfg, "terrain_type", "plane") == "plane":
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        size = getattr(terrain_gen, "size", None)
        if size is None or len(size) < 2:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        half_x = 0.5 * float(getattr(terrain_gen, "num_rows", 0)) * float(size[0])
        half_y = 0.5 * float(getattr(terrain_gen, "num_cols", 0)) * float(size[1])
        cfg = getattr(self.cfg, "rough_terrain_boundary_reset_cfg", {})
        if bool(cfg.get("use_inner_terrain_area", False)):
            pass
        else:
            border = float(getattr(terrain_gen, "border_width", 0.0))
            half_x += border
            half_y += border
        margin = max(float(cfg.get("margin", 0.5)), 0.0)
        half_x = max(half_x - margin, 0.0)
        half_y = max(half_y - margin, 0.0)
        root = self.robot.data.root_pos_w
        return (torch.abs(root[:, 0]) > half_x) | (torch.abs(root[:, 1]) > half_y)
