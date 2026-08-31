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
- ``_get_dones``：使用 Fudan 接触/倾倒共享的连续失败判定，同时保留数值安全和超时终止。

自维护缓冲（不依赖基类的 obs_history）：
- ``_wyw_obs_hist``：(N, T=5, 25) 的 fudan policy 历史，滚动写入，供 encoder。
"""

from __future__ import annotations

from typing import Sequence

import torch
from isaaclab.utils.math import quat_apply, quat_inv

from agent_tasks.direct.wheelbipe.wheelbipe25_v3.env import Wheelbipe25V3Env

from . import wyw_constants as C
from .fdu_mapping import (
    POLICY_JOINT_NAMES,
    compute_fdu_equivalent_leg_state,
    update_buggy_fudan_airtime,
)
from .fdu_semantics import (
    aggregate_fdu_rewards,
    build_fdu_critic_observation,
    build_fdu_dr_privilege,
    build_fdu_policy_observation,
    compute_fdu_collision_count,
    compute_fdu_failure_termination_reward,
    compute_fdu_jump_reward_terms,
    compute_fdu_plane_reward_terms,
    compute_fdu_flat_command_curriculum_transition,
    compute_fdu_rough_curriculum_transition,
    filter_fdu_wheel_contacts,
    get_due_fdu_flat_command_curriculum_step,
    sample_uniform_command_from_ranges,
    update_fdu_observation_history,
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
        failure_patterns = list(self.cfg.wyw_failure_contact_body_patterns)
        self._reset_contact_link_idx = self._find_contact_sensor_indices(failure_patterns)
        if not self._reset_contact_link_idx:
            raise RuntimeError(
                "FDU failure/collision contact mapping matched no sensor bodies: "
                f"patterns={failure_patterns}, available={self.contact_sensor.body_names}"
            )
        self._undesired_contact_link_idx = list(self._reset_contact_link_idx)
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
        wheel_targets = torch.clamp(self.wheel_actions, -self.max_wheel_vel, self.max_wheel_vel)
        # Match Fudan's direct joint-target semantics; L0/theta0 remain diagnostics only.
        self.robot.set_joint_position_target(self.leg_actions, joint_ids=self._wyw_leg_joint_idx)
        self.robot.set_joint_velocity_target(wheel_targets, joint_ids=self._wyw_wheel_joint_idx)
        self._update_obs()
        # The simulator refreshes body poses after _apply_action. From the
        # second substep onward this call samples the preceding fresh physics
        # result; _compute_fdu_reward_terms samples the final substep.
        substep_index = int(getattr(self, "_sim_step_counter", 0)) % int(self.cfg.decimation)
        if getattr(self, "_wyw_buffers_ready", False) and substep_index != 1:
            self._update_wyw_l0_stability_monitor(self._get_wyw_measured_leg_lengths())

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
        self._wyw_l0_boundary_episode_samples = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        # critic 特权项索引 / 基准：FDU root + 未随机化的默认关节位。
        base_idx, _ = self.robot.find_bodies("base_link_del")
        self._wyw_base_body_idx = int(base_idx[0]) if len(base_idx) else 0
        self._wyw_nominal_default_dof = self.robot.data.default_joint_pos[:, self._actuate_idx].clone()
        self._wyw_command_ranges_x = torch.tensor(
            self.cfg.commands.ranges.lin_vel_x, dtype=torch.float, device=self.device
        ).repeat(self.num_envs, 1)
        self._wyw_flat_curriculum_last_step = 0
        self._wyw_flat_curriculum_pending_log = None
        self._wyw_buffers_ready = True

    def _reset_idx(self, env_ids: Sequence[int] | None):
        reset_env_ids = self._as_env_ids_tensor(env_ids)
        # Snapshot the terminal-step attribution before the base reset mutates
        # robot/contact state.  These masks are populated in ``_get_dones`` and
        # are diagnostics only: they do not alter the done contract.
        done_reason_masks = {}
        for name in (
            "orientation",
            "contact",
            "persistent_failure",
            "numerical_safety",
            "terrain_boundary",
        ):
            mask = getattr(self, f"_wyw_done_reason_{name}", None)
            if mask is not None and reset_env_ids.numel() > 0:
                done_reason_masks[name] = mask[reset_env_ids].clone()
        terminal_contact_by_body = None
        terminal_contact_names: list[str] = []
        contact_body_mask = getattr(self, "_wyw_done_contact_body_mask", None)
        if contact_body_mask is not None and reset_env_ids.numel() > 0:
            terminal_contact_by_body = contact_body_mask[reset_env_ids].clone()
            sensor_body_names = list(getattr(self.contact_sensor, "body_names", []))
            terminal_contact_names = [sensor_body_names[index] for index in self._reset_contact_link_idx]
        monitor_ready = getattr(self, "_wyw_buffers_ready", False) and hasattr(
            self, "_wyw_l0_boundary_episode_samples"
        )
        if monitor_ready and reset_env_ids.numel() > 0:
            episode_boundary_samples = self._wyw_l0_boundary_episode_samples[reset_env_ids].clone()
        self._update_fdu_flat_command_curriculum(reset_env_ids)
        self._update_fdu_rough_curriculum(reset_env_ids)
        super()._reset_idx(env_ids)
        self._ensure_wyw_buffers()
        if done_reason_masks:
            log = self.extras.setdefault("log", {})
            reset_count = max(int(reset_env_ids.numel()), 1)
            for name, mask in done_reason_masks.items():
                count = int(mask.count_nonzero().item())
                log[f"Termination/Count/{name}"] = count
                log[f"Termination/Fraction/{name}"] = count / reset_count
            if terminal_contact_by_body is not None:
                for body_index, body_name in enumerate(terminal_contact_names):
                    log[f"Termination/ContactBody/{body_name}"] = int(
                        terminal_contact_by_body[:, body_index].count_nonzero().item()
                    )
        if self._wyw_flat_curriculum_pending_log is not None:
            self.extras.setdefault("log", {}).update(self._wyw_flat_curriculum_pending_log)
            self._wyw_flat_curriculum_pending_log = None
        if monitor_ready and reset_env_ids.numel() > 0:
            log = self.extras.setdefault("log", {})
            log["Episode/FDU_L0Boundary/affected_env_fraction"] = float(
                torch.mean((episode_boundary_samples > 0).float()).item()
            )
            self._wyw_l0_boundary_episode_samples[reset_env_ids] = 0
        self._wyw_obs_hist[reset_env_ids] = 0.0
        self._wyw_history_needs_fill[reset_env_ids] = True
        self._sample_wyw_lin_vel_command(reset_env_ids)
        self._clear_termination_duration_buffers(
            reset_env_ids,
            counter_attr="_wyw_failure_termination_counter",
            raw_attr="_wyw_failure_termination_raw_buf",
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
        return build_fdu_policy_observation(
            base_ang_vel=ang_vel,
            projected_gravity=gravity,
            command_block=cmd,
            leg_pos_deviation=policy_pos[:, C.WYW_LEG_ACTION_IDS],
            policy_joint_vel=policy_vel,
            actions=self._actions,
            ang_vel_scale=self.cfg.wyw_ang_vel_scale,
            projected_gravity_scale=self.cfg.wyw_proj_gravity_scale,
            joint_pos_scale=self.cfg.wyw_joint_pos_scale,
            dof_vel_scale=self.cfg.wyw_dof_vel_scale,
        )

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

        return build_fdu_dr_privilege(
            centered_base_mass=base_mass_dev,
            base_com_offset=base_com,
            default_dof_delta=default_dof_delta,
            friction=friction,
            restitution=restitution,
        )

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
        return build_fdu_critic_observation(
            scaled_base_lin_vel=base_lin_vel,
            clean_policy_observation=policy_obs,
            previous_actions=previous_actions,
            before_previous_actions=before_previous_actions,
            scaled_joint_acc=joint_acc * self.cfg.wyw_joint_acc_scale,
            height_scan=heights,
            scaled_torque=torque * self.cfg.wyw_torque_scale,
            dr_privilege=self._build_wyw_priv_dr_obs(),
        )

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
        self._wyw_obs_hist, self._wyw_history_needs_fill = update_fdu_observation_history(
            self._wyw_obs_hist, policy, self._wyw_history_needs_fill
        )
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

    def _get_wyw_measured_leg_lengths(self) -> torch.Tensor:
        """Measure hip-link to wheel-link planar length from PhysX body poses.

        This intentionally uses the same physical definition as the accepted
        drop diagnostics. It can therefore detect solver compression or a loop
        limit cycle that an analytic joint-angle reconstruction may hide.
        """
        # Do not use the policy-step wheel cache here: this monitor is called
        # at every physics substep and must observe each freshly simulated pose.
        root_quat_inv = quat_inv(self.robot.data.root_quat_w)
        wheel_rel_pos_w = (
            self.robot.data.body_pos_w[:, self._wheel_link_idx]
            - self.robot.data.root_pos_w.unsqueeze(1)
        )
        wheel_pos_b = quat_apply(
            root_quat_inv.unsqueeze(1).expand(-1, len(self._wheel_link_idx), -1),
            wheel_rel_pos_w,
        )
        hip_rel_pos_w = (
            self.robot.data.body_pos_w[:, self._wyw_hip_link_idx]
            - self.robot.data.root_pos_w.unsqueeze(1)
        )
        hip_pos_b = quat_apply(
            root_quat_inv.unsqueeze(1).expand(-1, len(self._wyw_hip_link_idx), -1),
            hip_rel_pos_w,
        )
        delta = wheel_pos_b - hip_pos_b
        return torch.linalg.vector_norm(delta[..., (0, 2)], dim=-1)

    def _update_wyw_l0_stability_monitor(self, measured_l0: torch.Tensor) -> None:
        """Retain only whether each episode ever crossed the L0 boundary."""
        if not bool(getattr(self.cfg, "wyw_l0_stability_monitor_enabled", True)):
            return
        threshold = float(self.cfg.wyw_l0_stability_boundary_m)
        per_env_min = measured_l0.min(dim=-1).values
        active = per_env_min <= threshold
        self._wyw_l0_boundary_episode_samples += active.long()

    def _compute_wyw_jump_terms(self, leg_lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        """fudan 涌现式跳跃奖励项（均为 per-step 速率，后续统一 ×step_dt）。"""
        # Fudan check_jump: current world-z force OR previous policy frame.
        net_forces = self.contact_sensor.data.net_forces_w
        if net_forces is not None and len(self._desired_contact_link_idx) == 2:
            contact_now = (
                net_forces[:, self._desired_contact_link_idx, 2] > self.cfg.wyw_flight_contact_force
            )
            in_flight, any_contact, next_contacts = filter_fdu_wheel_contacts(
                contact_now, self._wyw_last_wheel_contacts
            )
            self._wyw_last_wheel_contacts.copy_(next_contacts)
        else:
            in_flight = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            any_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        root_z = self.robot.data.root_pos_w[:, 2]
        vz = self.robot.data.root_lin_vel_w[:, 2]
        terms, self._wyw_base_air_time = compute_fdu_jump_reward_terms(
            leg_lengths=leg_lengths,
            in_flight=in_flight,
            any_contact=any_contact,
            root_z=root_z,
            root_vz=vz,
            base_air_time=self._wyw_base_air_time,
            step_dt=self.step_dt,
            l0_tuck=self.cfg.wyw_l0_tuck,
            l0_extend=self.cfg.wyw_l0_extend,
            base_height_flight=self.cfg.wyw_base_height_flight,
            takeoff_vz=self.cfg.wyw_takeoff_vz,
            airtime_update=update_buggy_fudan_airtime,
        )
        return terms

    def _compute_fdu_reward_terms(self) -> dict[str, torch.Tensor]:
        """Compute the named Fudan reward terms without V3 shaping gates."""
        self._update_ground_height_estimate()
        wheel_pos_b, leg_lengths, leg_angles = self._get_wyw_virtual_leg_geometry()
        # Capture the fifth/final 500 Hz result, which becomes available only
        # after DirectRLEnv finishes the decimation loop.
        self._update_wyw_l0_stability_monitor(self._get_wyw_measured_leg_lengths())
        left_len, right_len = leg_lengths[:, 0], leg_lengths[:, 1]
        left_theta, right_theta = leg_angles[:, 0], leg_angles[:, 1]

        observed_height = self._get_fdu_base_height()
        height_cmd = self._get_observation_height_cmd()

        qdot = self.robot.data.joint_vel[:, self._actuate_idx]
        qddot = self.robot.data.joint_acc[:, self._actuate_idx]
        tau = self.robot.data.applied_torque[:, self._actuate_idx]

        hard_limits = self.robot.data.joint_pos_limits[:, self._wyw_leg_joint_idx]
        centers = 0.5 * (hard_limits[..., 0] + hard_limits[..., 1])
        half_ranges = 0.5 * (hard_limits[..., 1] - hard_limits[..., 0])
        soft_lower = centers - 0.97 * half_ranges
        soft_upper = centers + 0.97 * half_ranges
        positions = self.robot.data.joint_pos[:, self._wyw_leg_joint_idx]

        contact_forces = getattr(self.contact_sensor.data, "net_forces_w", None)
        if contact_forces is None or not self._undesired_contact_link_idx:
            collision = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        else:
            collision = compute_fdu_collision_count(
                contact_forces[:, self._undesired_contact_link_idx],
                float(self.cfg.wyw_collision_contact_force),
            )

        terms = compute_fdu_plane_reward_terms(
            command_vx=self.command[:, 0],
            command_yaw=self.command[:, 2],
            base_lin_vel=self.robot.data.root_lin_vel_b,
            base_ang_vel=self.robot.data.root_ang_vel_b,
            projected_gravity=self.robot.data.projected_gravity_b,
            observed_height=observed_height,
            height_command=height_cmd,
            left_l0=left_len,
            right_l0=right_len,
            left_theta=left_theta,
            right_theta=right_theta,
            joint_vel=qdot,
            joint_acc=qddot,
            applied_torque=tau,
            actions=self._actions,
            previous_actions=self._previous_actions,
            before_previous_actions=self._before_previous_actions,
            leg_positions=positions,
            leg_soft_lower=soft_lower,
            leg_soft_upper=soft_upper,
            collision_count=collision,
            tracking_sigma=self.cfg.tracking_sigma,
            jump=bool(getattr(self.cfg, "wyw_jump_enabled", False)),
        )
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
        total_reward, rewards = aggregate_fdu_rewards(
            reward_terms,
            self.cfg.rewards,
            step_dt=self.step_dt,
            clip_single_reward=getattr(self.cfg, "clip_single_reward", None),
            only_positive_rewards=bool(getattr(self.cfg, "only_positive_rewards", False)),
            invalid_mask=self._numerical_safety_reset_buf,
        )
        # Match Fudan's terminal-reward ordering: add the one-shot failure
        # penalty after positive/term clipping. Timeouts, terrain boundaries,
        # and numerical-safety resets intentionally do not enter this mask.
        failure_termination = getattr(
            self,
            "_wyw_failure_termination_reward_mask",
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        )
        termination_reward = compute_fdu_failure_termination_reward(
            failure_termination,
            self.cfg.rewards.get("termination", 0.0),
            self.step_dt,
        )
        rewards["termination"] = termination_reward
        total_reward = total_reward + termination_reward
        self._last_reward_terms = {key: value.detach() for key, value in rewards.items()}
        for key, value in rewards.items():
            self._episode_sums.setdefault(key, torch.zeros(self.num_envs, device=self.device))
            self._episode_sums[key] += value
        self._last_total_reward = total_reward.detach()
        return total_reward

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
        values = sample_uniform_command_from_ranges(
            ranges, torch.rand(env_ids.numel(), device=self.device)
        )
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
        old_levels = terrain.terrain_levels[env_ids].clone()
        move_up, move_down, _success, updated_ranges = compute_fdu_rough_curriculum_transition(
            old_levels=old_levels,
            terrain_types=terrain.terrain_types[env_ids],
            distance=distance,
            tracking_rate=tracking_rate,
            command_ranges_x=self._wyw_command_ranges_x[env_ids],
            terrain_length=float(terrain.cfg.terrain_generator.size[0]),
            max_terrain_level=int(terrain.max_terrain_level),
        )
        terrain.update_env_origins(env_ids, move_up, move_down)
        self._wyw_command_ranges_x[env_ids] = updated_ranges

    def _update_fdu_flat_command_curriculum(self, env_ids: torch.Tensor) -> None:
        """Consume each reached Fudan Plane vx-curriculum cadence on the next reset."""
        if not bool(getattr(self.cfg, "wyw_flat_command_curriculum_enabled", False)):
            return
        if env_ids.numel() == 0 or not getattr(self, "_wyw_buffers_ready", False):
            return
        interval = max(int(self.cfg.wyw_command_curriculum_interval_steps), 1)
        step = int(getattr(self, "common_step_counter", 0))
        due_step = get_due_fdu_flat_command_curriculum_step(
            current_step=step,
            interval=interval,
            last_consumed_step=self._wyw_flat_curriculum_last_step,
        )
        if due_step is None:
            return
        tracking_lin = self._episode_sums.get("tracking_lin_vel")
        tracking_yaw = self._episode_sums.get("tracking_ang_vel")
        if tracking_lin is None or tracking_yaw is None:
            return
        duration = max(float(self.max_episode_length_s), 1.0e-6)
        mean_lin_rate = torch.mean(tracking_lin[env_ids]) / duration
        mean_yaw_rate = torch.mean(tracking_yaw[env_ids]) / duration
        expanded, updated = compute_fdu_flat_command_curriculum_transition(
            command_ranges_x=self._wyw_command_ranges_x,
            mean_tracking_lin_rate=mean_lin_rate,
            mean_tracking_yaw_rate=mean_yaw_rate,
            lin_threshold=float(self.cfg.wyw_command_curriculum_lin_threshold),
            yaw_threshold=float(self.cfg.wyw_command_curriculum_yaw_threshold),
            expansion_step=float(self.cfg.wyw_command_curriculum_step),
            max_abs=float(self.cfg.wyw_command_curriculum_max_abs),
        )
        self._wyw_command_ranges_x.copy_(updated)
        self._wyw_flat_curriculum_last_step = due_step
        self._wyw_flat_curriculum_pending_log = {
            "Curriculum/FDUFlat/cadence_step": due_step,
            "Curriculum/FDUFlat/consumed_at_step": step,
            "Curriculum/FDUFlat/mean_tracking_lin_rate": float(mean_lin_rate.item()),
            "Curriculum/FDUFlat/mean_tracking_yaw_rate": float(mean_yaw_rate.item()),
            "Curriculum/FDUFlat/vx_min": float(updated[0, 0].item()),
            "Curriculum/FDUFlat/vx_max": float(updated[0, 1].item()),
            "Curriculum/FDUFlat/expanded": int(expanded),
        }

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Use Fudan's shared persisted contact/tilt failure while retaining numerical safety."""
        _, time_out = super()._get_dones()
        boundary_terminate = self._get_rough_terrain_boundary_termination()
        immediate_terminate = self._numerical_safety_reset_buf.clone()
        contact_forces = getattr(self.contact_sensor.data, "net_forces_w", None)
        if contact_forces is None:
            bad_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            bad_contact_by_body = torch.zeros(
                self.num_envs,
                len(self._reset_contact_link_idx),
                dtype=torch.bool,
                device=self.device,
            )
        else:
            failure_forces = contact_forces[:, self._reset_contact_link_idx]
            bad_contact_by_body = (
                torch.linalg.vector_norm(failure_forces, dim=-1)
                > float(self.cfg.wyw_failure_contact_force)
            )
            bad_contact = torch.any(bad_contact_by_body, dim=-1)
        bad_orientation = self.robot.data.projected_gravity_b[:, 2] > -0.1
        persistent_failure = self._apply_termination_duration(
            bad_orientation | bad_contact,
            counter_attr="_wyw_failure_termination_counter",
            raw_attr="_wyw_failure_termination_raw_buf",
        )
        # Fudan resets an out-of-bounds environment immediately and does not
        # classify that reset as an episode-length timeout. In particular, it
        # must not enter the one-second contact/tilt persistence counter.
        terminate = immediate_terminate | persistent_failure | boundary_terminate
        # Latched for reset-time attribution.  ``orientation`` and ``contact``
        # are the raw conditions on the terminal step; ``persistent_failure``
        # is the actual one-second persisted decision used by the environment.
        self._wyw_done_reason_orientation = bad_orientation
        self._wyw_done_reason_contact = bad_contact
        self._wyw_done_reason_persistent_failure = persistent_failure
        self._wyw_done_reason_numerical_safety = immediate_terminate
        self._wyw_done_reason_terrain_boundary = boundary_terminate
        self._wyw_done_contact_body_mask = bad_contact_by_body
        if self.cfg.play is True and not bool(getattr(self.cfg, "play_keep_done_reset", False)):
            terminate.zero_()
        self._wyw_failure_termination_reward_mask = (
            persistent_failure
            & terminate
            & ~time_out
            & ~immediate_terminate
            & ~boundary_terminate
        )
        return terminate, time_out

    def _get_rough_terrain_boundary_termination(self) -> torch.Tensor:
        """Return immediate failure termination outside the configured rough terrain area."""
        cfg = getattr(self.cfg, "rough_terrain_boundary_reset_cfg", {})
        if not bool(cfg.get("enabled", False)):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        terrain_cfg = getattr(self.cfg, "terrain", None)
        terrain_gen = getattr(terrain_cfg, "terrain_generator", None)
        if terrain_gen is None or getattr(terrain_cfg, "terrain_type", "plane") == "plane":
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        size = getattr(terrain_gen, "size", None)
        if size is None or len(size) < 2:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        half_x = 0.5 * float(getattr(terrain_gen, "num_rows", 0)) * float(size[0])
        half_y = 0.5 * float(getattr(terrain_gen, "num_cols", 0)) * float(size[1])
        if bool(cfg.get("use_inner_terrain_area", False)):
            pass
        else:
            border = float(getattr(terrain_gen, "border_width", 0.0))
            half_x += border
            half_y += border
        margin = max(float(cfg.get("margin", 1.0)), 0.0)
        half_x = max(half_x - margin, 0.0)
        half_y = max(half_y - margin, 0.0)
        root = self.robot.data.root_pos_w
        return (torch.abs(root[:, 0]) > half_x) | (torch.abs(root[:, 1]) > half_y)
