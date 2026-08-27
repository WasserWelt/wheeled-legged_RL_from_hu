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
- Actor 观测 25 维、Critic 特权观测拼 latent 前为 46 维、5 帧历史 → encoder 输入 125 维。
- Actor/Critic/Obs/网络/PPO 超参在 Flat / Rough / Jump 三个任务间共享。
- ``WYW_ROBOT`` 开关切换本体相关的几何常量：``"from_hu"``（首版，USD 仍在改）或 ``"fudan"``。

维度约定（from_hu Wheelbipe_V14_2，动作 6 维 = 4 腿关节位置 + 2 轮速）：
- 腿部驱动关节 ``leg_dim = 4``，轮 2，``_actuate_idx`` 合计 6。
- Policy(25) = ang_vel(3) + proj_gravity(3) + cmd[vx,yaw,height](3)
               + leg_pos_dev(4) + dof_vel(6) + actions(6)
- Critic(46) = base_lin_vel(3) + ang_vel(3) + proj_gravity(3) + cmd(3)
               + leg_pos_dev(4) + dof_vel(6) + actions(6) + prev_actions(6)
               + joint_acc(6) + torque(6)
  其中 base_lin_vel 必须是前 3 维，供 encoder 的隐式线速度监督
  （PPOSequence: MSE(latent[:, :3], critic_obs[:, :3])）。
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
WYW_CRITIC_DIM = 46             # 拼 latent 前的特权观测维（见文件头布局）
WYW_LATENT_DIM = 3              # encoder latent（隐式基座线速度估计）

# ---------------------------------------------------------------------------- #
# 观测缩放（对齐 fudan obs_scales）
# ---------------------------------------------------------------------------- #
WYW_ANG_VEL_SCALE = 0.25        # 机身角速度
WYW_DOF_VEL_SCALE = 0.05        # 关节速度
WYW_LIN_VEL_SCALE = 2.0         # 基座线速度（命令 vx + critic base_lin_vel）
WYW_CMD_ANG_VEL_SCALE = 0.25    # 偏航速度命令
WYW_HEIGHT_CMD_SCALE = 1.0      # 高度命令
WYW_PROJ_GRAVITY_SCALE = 1.0
WYW_JOINT_POS_SCALE = 1.0
WYW_ACTION_SCALE = 1.0
WYW_JOINT_ACC_SCALE = 0.0025    # critic 专用，抑制量级
WYW_TORQUE_SCALE = 0.05         # critic 专用

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
