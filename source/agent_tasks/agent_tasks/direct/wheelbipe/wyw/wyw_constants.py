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

"""wyw 任务族的几何 / 观测 / 奖励常量（fudan 设计移植，from_hu 机器人本体）。

设计意图（对齐 fudan_rl_wheel_leg）：
- Actor 观测 25 维、Critic 特权观测拼 latent 前为 141 维（= fudan 原始 privileged_obs）、
  5 帧历史 → encoder 输入 125 维。
- Actor/Critic/Obs/网络/PPO 超参在 Flat / Rough / Jump 三个任务间共享
  （唯一例外：Jump 的 ``wyw_lin_vel_scale=3.0``，Flat/Rough 为 2.0，见 env_cfg）。
- ``WYW_ROBOT`` 开关切换本体相关的几何常量：``"from_hu"``（首版，USD 仍在改）或 ``"fudan"``。

维度约定（from_hu Wheelbipe_V14_2，动作 6 维 = 4 腿关节位置 + 2 轮速）：
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
# 本体开关：首版用 from_hu（fudan USD 仍在修改，见 docs/intention.md）
# ---------------------------------------------------------------------------- #
WYW_ROBOT = "from_hu"  # "from_hu" | "fudan"

# ---------------------------------------------------------------------------- #
# 观测 / 网络维度
# ---------------------------------------------------------------------------- #
WYW_LEG_DIM = 4                 # 腿部驱动关节数（from_hu V14）
WYW_ACTION_DIM = 6              # 4 腿 + 2 轮
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
# 本文件只保留：网络/观测**维度**、**几何**目标（L0、滞空高度）、物理**阈值**、跳跃**奖励权重**。
#
# ---------------------------------------------------------------------------- #
# 跳跃（jump）几何常量：随本体切换
#   L0 = 腿长 = ||wheel_pos_heading_b||（每条腿轮心到基座在随航向水平系下的距离）
# ---------------------------------------------------------------------------- #
_JUMP_GEOMETRY = {
    "from_hu": dict(
        L0_TUCK=0.16,            # 收腿目标腿长（滞空收腿）
        L0_EXTEND=0.31,          # 蹬伸目标腿长（起跳蹬地）
        BASE_HEIGHT_FLIGHT=0.60,  # 滞空期望机身高度（顶点附近）
    ),
    "fudan": dict(
        L0_TUCK=0.16,
        L0_EXTEND=0.31,
        BASE_HEIGHT_FLIGHT=0.65,
    ),
}

_geom = _JUMP_GEOMETRY[WYW_ROBOT]
WYW_L0_TUCK = _geom["L0_TUCK"]
WYW_L0_EXTEND = _geom["L0_EXTEND"]
WYW_BASE_HEIGHT_FLIGHT = _geom["BASE_HEIGHT_FLIGHT"]

# 跳跃触发 / 接触阈值（与本体无关）
WYW_TAKEOFF_VZ = 0.15           # 判定"正在蹬伸起跳"的最小竖直速度
WYW_FLIGHT_CONTACT_FORCE = 1.0  # 车轮离地判定的接触力阈值（N）
WYW_FALL_CONTACT_FORCE = 10.0   # 摔倒判定接触力（当前复用基类 _get_dones，未直接使用）

# 空中滞空累积器的高度裁剪（encourage_jump）
WYW_AIRTIME_HEIGHT_CLIP = 0.5

# ---------------------------------------------------------------------------- #
# 跳跃奖励权重（对齐 fudan；仅 Jump 任务的 cfg.rewards 引用）
# ---------------------------------------------------------------------------- #
WYW_JUMP_REWARD_WEIGHTS = dict(
    base_height_flight=6.0,
    leg_tuck=1.7,
    takeoff_extend=0.5,
    line_z=6.0,
    flight=0.15,
    encourage_jump=1.0,
)
