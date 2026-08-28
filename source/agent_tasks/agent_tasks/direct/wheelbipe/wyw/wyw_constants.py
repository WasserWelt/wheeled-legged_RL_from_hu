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

"""WYW tensor-layout constants for the FDU closed-chain robot.

设计意图（对齐 fudan_rl_wheel_leg）：
- Actor 观测 25 维、Critic 特权观测拼 latent 前为 141 维（= fudan 原始 privileged_obs）、
  5 帧历史 → encoder 输入 125 维。
- Actor/Critic/Obs/网络/PPO 超参在 Flat / Rough / Jump 三个任务间共享
  （唯一例外：Jump 的 ``wyw_lin_vel_scale=3.0``，Flat/Rough 为 2.0，见 env_cfg）。
维度约定（FDU 闭链并联本体，动作 6 维 = 4 实体驱动杆位置 + 2 轮速）：
- 腿部驱动关节 ``leg_dim = 4``，轮 2，``_actuate_idx`` 合计 6。
- Policy(25) = ang_vel(3) + proj_gravity(3) + cmd[vx,yaw,height](3)
               + leg_pos_dev(4) + dof_vel(6) + actions(6)
- Critic(141) = base_lin_vel(3) + obs_buf(25=policy 本体) + prev_actions(6)
                + before_prev_actions(6) + joint_acc(6) + heights(77)
                + torque(6) + base_mass_dev(1) + base_com(3)
                + default_dof_delta(6) + friction(1) + restitution(1)
  与 fudan ``privileged_obs_buf`` 逐段一致（plane 与 jump 版组成完全相同）。
  其中 base_lin_vel 必须是前 3 维，供 encoder 的隐式线速度监督
  （PPOSequence: MSE(latent[:, :3], critic_obs[:, :3])）。
  注：critic 只在训练用（不导出部署），故 heights 的 77 个采样点空间排序无需与 fudan 逐点对齐。
"""

from __future__ import annotations

# ---------------------------------------------------------------------------- #
# 观测 / 网络维度
# ---------------------------------------------------------------------------- #
WYW_LEG_DIM = 4                 # 腿部实体驱动杆数
WYW_ACTION_DIM = 6              # 4 腿 + 2 轮
WYW_LEG_ACTION_IDS = [0, 1, 3, 4]
WYW_WHEEL_ACTION_IDS = [2, 5]
WYW_POLICY_OBS_DIM = 25         # actor 观测维（见文件头布局）
WYW_NUM_OBS_HIST = 5            # encoder 历史帧数
WYW_POLICY_HIST_DIM = WYW_POLICY_OBS_DIM * WYW_NUM_OBS_HIST  # 125
WYW_N_SCAN = 77                 # 地形高度扫描点数（fudan 11×7；dot_scanner size=(1.0,0.6) res=0.1）
WYW_HEIGHT_SCALE = 5.0          # 高度扫描 obs 缩放（fudan height_measurements=5.0）
WYW_CRITIC_DIM = 141            # 拼 latent 前的特权观测维（= fudan privileged_obs，见文件头布局）
WYW_LATENT_DIM = 3              # encoder latent（隐式基座线速度估计）

# ---------------------------------------------------------------------------- #
# 观测缩放（obs_scales）**不在此文件**。
#
# 按 IsaacLab DirectRLEnv 与本仓库基类（wheelbipe25_v3/env_cfg.py 的 lin_vel_scale /
# ang_vel_scale / ... 字段）的惯例，所有 obs 缩放作为 **configclass 字段** 定义在
# ``env_cfg.py`` 的 ``WheelbipeWywFlatEnvCfg`` 上（``wyw_*_scale``），好处：
#   1) 随 ``params/env.yaml`` 自动落盘，便于复现 / 审计 / 对齐部署端；
#   2) 可按任务覆写（如 Jump 把 ``wyw_lin_vel_scale`` 从 2.0 改成 3.0）；
#   3) 与基类 / IsaacLab 编码风格一致（scale 是配置，不是几何常量）。
# 本文件只保留网络/观测维度和不可配置的字段索引。几何目标、阈值和奖励权重
# 均是任务配置，定义在 env_cfg.py 中以便写入 params/env.yaml。
