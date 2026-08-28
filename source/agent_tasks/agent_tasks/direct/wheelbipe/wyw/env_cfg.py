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

"""wyw FDU 闭链并联本体环境配置（Flat / Rough / Jump + Play）。

- 本体使用指定 ``infantry_V2.urdf`` 转换出的 ``Wheelbipe_FDU_CFG``。
- 观测走 fudan 25/125/141 布局：在 ``__post_init__`` 末尾（``_apply_wyw_common``）强制把
  ``observation_space`` / ``state_space`` 设为 **int**（25 / 141，**不是 dict**——传 dict 会被
  stock ``DirectRLEnv._configure_gym_env_spaces`` 整体嵌进 ``["policy"]`` 丢掉 policy_hist 键），
  环境侧 ``_get_observations`` 再覆写实际返回张量（policy/policy_hist/critic 三键）。
- decimation=2 → 100Hz 策略（from_hu V14 默认 decimation=4=50Hz）。
- 命令范围收敛到 fudan：vx±2.1、yaw±2.0，关闭 spin/dash 特殊模式。
- Flat/Rough 复用 V14 的 locomotion ``rewards``；Jump 追加 fudan 涌现式跳跃项。
"""

from __future__ import annotations

import copy
from collections import OrderedDict

from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

import agent_tasks.manager.mdp.isaaclab as mdp
from agent_tasks.direct.wheelbipe.wheelbipe25_v3.env_cfg import Wheelbipe25v3FlatEnvCfg, EventCfg
from agent_tasks.direct.wheelbipe.wheelbipe_V14.cfg_utils import _apply_v14_rough_runtime_cfg
from agent_world.assets.wheelbipe_fdu import Wheelbipe_FDU_CFG
from agent_world import AssetPath

from . import wyw_constants as C


# ---------------------------------------------------------------------------- #
# 工具：把 fudan 观测/命令/控制频率强制写到一个已 __post_init__ 过的 cfg 上
# ---------------------------------------------------------------------------- #
def _apply_wyw_common(cfg) -> None:
    """在 super().__post_init__() 之后，强制 fudan 的 obs 形状 / 命令 / 100Hz。"""
    # 100Hz 策略
    cfg.decimation = 2

    # fudan 观测形状。stock DirectRLEnv._configure_gym_env_spaces 会把
    # observation_space 原样交给 spec_to_gym_space —— 传 dict 会被整体嵌套进
    # single_observation_space["policy"]（各子键 flatdim 相加），从而丢失 policy_hist
    # 顶层键。故这里用 int：
    #   single_observation_space = {policy: Box(25), critic: Box(46)}
    #   → 自定义 wrapper: num_observations={policy:25, critic:46}, num_privileged_obs=46
    #   → runner 对 policy_hist 回落到 num_obs_hist * num_obs = 5*25 = 125（=encoder 输入）。
    cfg.observation_space = C.WYW_POLICY_OBS_DIM
    cfg.state_space = C.WYW_CRITIC_DIM
    cfg.num_obs_hist = C.WYW_NUM_OBS_HIST

    # fudan critic 含 11×7=77 维地形高度扫描（privileged）。启用基类 dot_scanner，
    # 并把网格改成 fudan 尺寸：x∈[-0.5,0.5] 步 0.1（11 点）、y∈[-0.3,0.3] 步 0.1（7 点）。
    # 三任务（含 plane 的 Flat/Jump）都挂扫描器——plane 上读到近平地（clip 后≈0），
    # rough 上读到真实地形起伏。_get_scan_dot_obs 用 _pad_flat_features 截/补到 n_scan=77。
    cfg.enable_scan_dot = True
    cfg.n_scan = C.WYW_N_SCAN
    cfg.height_scale = C.WYW_HEIGHT_SCALE
    cfg.dot_scanner = copy.deepcopy(cfg.dot_scanner)
    cfg.dot_scanner.pattern_cfg.resolution = 0.1
    cfg.dot_scanner.pattern_cfg.size = (1.0, 0.6)
    cfg.height_scanner = copy.deepcopy(cfg.height_scanner)
    cfg.height_scanner.prim_path = "/World/envs/env_.*/Robot/base_link_del"
    cfg.dot_scanner.prim_path = "/World/envs/env_.*/Robot/base_link_del"

    # 命令：收敛到 fudan locomotion 范围，关闭 spin/dash 特殊模式
    ranges = getattr(cfg.commands, "ranges", None)
    if ranges is not None:
        ranges.lin_vel_x = (-2.1, 2.1)
        ranges.lin_vel_y = (0.0, 0.0)
        ranges.ang_vel_z = (-2.0, 2.0)
    special_modes = getattr(cfg.commands, "special_modes", None)
    if special_modes:
        for mode in special_modes.values():
            mode.rel_envs = 0.0

    # 高度命令区间锁死为当前 FDU 任务配置（rough helper 可能改写，故这里强制回来）
    cfg.height_range = [0.20, 0.42]
    if cfg.terrain.terrain_type == "plane":
        cfg.terrain.terrain_type = "usd"
        cfg.terrain.usd_path = f"{AssetPath}/usd_files/flat_ground.usda"


@configclass
class FduEventCfg(EventCfg):
    """V3 domain randomization retargeted to exact FDU entity names."""

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link_del"),
            "mass_distribution_params": (0.9, 1.2),
            "operation": "scale",
        },
    )
    add_leg_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["[lr]f[01]_Link", "[lr]2[0-3]_Link"]),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    add_wheel_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="[lr]_wheel_Link"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link_del"),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.04, 0.04), "z": (-0.02, 0.02)},
        },
    )
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="[lr]_wheel_Link"),
            "static_friction_range": (0.5, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    base_external_force_torque_xyz = EventTerm(
        func=mdp.apply_external_force_torque_xyz,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link_del"),
            "force_range": ((-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)),
            "torque_range": ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
        },
    )


@configclass
class WheelbipeWywFlatEnvCfg(Wheelbipe25v3FlatEnvCfg):
    """wyw Flat：平地 + fudan locomotion 奖励 + ActorCriticSequence。"""

    # 关闭基类 7 维 ctrl_mode_obs（我们完全自定义 obs 布局）
    ctrl_mode_obs_enabled = False
    curriculum = None
    use_frame_stack = False
    num_obs_hist = C.WYW_NUM_OBS_HIST
    num_privileged_obs_hist = 1

    # 声明（会在 __post_init__ 末尾再强制一次，防止基类重算覆盖）
    observation_space = C.WYW_POLICY_OBS_DIM
    state_space = C.WYW_CRITIC_DIM
    events = FduEventCfg()
    robot_cfg: ArticulationCfg = Wheelbipe_FDU_CFG.replace(prim_path="/World/envs/env_.*/Robot").copy()
    legs_act_name = robot_cfg.actuators["legs_act"].joint_names_expr
    legs_inact_name = robot_cfg.actuators["legs_inact"].joint_names_expr
    wheel_name = robot_cfg.actuators["wheel"].joint_names_expr
    use_spring = False
    use_leg_random_start = False
    use_joint_vel_random_start = False
    use_predefined_leg_random_start = False
    use_obs_delay = False
    use_act_delay = False
    leg_action_scale = 0.5
    wheel_vel_action_scale = 10.0
    max_wheel_vel = 60.0
    termination_duration_enabled = True
    termination_duration_steps = 100

    # ------------------------------------------------------------------ #
    # 观测缩放（obs_scales）—— 按 IsaacLab / wheelbipe25_v3 风格作为 configclass 字段。
    # 好处：随 params/env.yaml 落盘、可按任务覆写、与基类风格一致（scale 是配置非常量）。
    # env.py 通过 self.cfg.wyw_*_scale 读取。⚠️ 与部署端逐位一致（对齐 fudan obs_scales）。
    # 注意：obs 里的 action 段**不缩放**（fudan obs 直接用原始 actions / last_actions，
    #       scale=1.0 等于无操作），故不设 wyw_action_scale 字段，env.py 直接用 self._actions。
    #       env 级动作输出缩放是另一个字段 action_scale=0.25（第 4 节），与 obs 无关。
    # ------------------------------------------------------------------ #
    wyw_ang_vel_scale = 0.25        # 机身角速度
    wyw_dof_vel_scale = 0.05        # 关节速度
    wyw_lin_vel_scale = 2.0         # 命令 vx + critic base_lin_vel
    wyw_cmd_ang_vel_scale = 0.25    # 偏航命令
    wyw_height_cmd_scale = 1.0      # 高度命令
    wyw_proj_gravity_scale = 1.0    # 投影重力
    wyw_joint_pos_scale = 1.0       # 腿关节位置 / 偏差
    wyw_joint_acc_scale = 0.0025    # critic 专用
    wyw_torque_scale = 0.05         # critic 专用

    # 跳跃奖励注入开关（Flat/Rough 关闭）
    wyw_jump_enabled = False

    def __post_init__(self):
        super().__post_init__()
        _apply_wyw_common(self)


@configclass
class WheelbipeWywRoughEnvCfg(WheelbipeWywFlatEnvCfg):
    """wyw Rough：trimesh 地形 + 课程，obs/reward/网络与 Flat 共享。"""

    rough_terrain_generator_cfg = copy.deepcopy(mdp.RM_ROTATION_TERRAINS_CFG_99)
    rough_terrain_boundary_reset_cfg = {
        "enabled": True,
        "margin": 0.5,
        "use_inner_terrain_area": False,
    }

    def __post_init__(self):
        super().__post_init__()
        # 复用 V14 的 rough 运行时配置（swap 到 generator 地形、启用高度扫描/状态机、课程）
        _apply_v14_rough_runtime_cfg(self)
        # rough helper 可能改动扫描/状态机，但不动 obs dict；再强制一次 wyw 形状与频率
        _apply_wyw_common(self)


@configclass
class WheelbipeWywJumpEnvCfg(WheelbipeWywFlatEnvCfg):
    """wyw Jump：平地 + locomotion + fudan 涌现式跳跃奖励（无显式起跳状态机）。"""

    wyw_jump_enabled = True

    # fudan jump 变体的 lin_vel obs_scale = 3.0（plane 版为 2.0）。该字段同时驱动
    # 命令 vx 缩放与 critic base_lin_vel 缩放（也即 encoder 监督目标 = base_lin_vel×3.0），
    # 与 fudan commands_scale[0]==obs_scales.lin_vel 的耦合一致。
    wyw_lin_vel_scale = 3.0

    def __post_init__(self):
        super().__post_init__()
        self.robot_cfg = copy.deepcopy(self.robot_cfg)
        self.robot_cfg.actuators["legs_act"].stiffness = 6.0
        self.robot_cfg.actuators["legs_act"].damping = 0.5
        # locomotion + jump 奖励项合并（基类只对 cfg.rewards 中列出的键求和）
        jump_rewards = OrderedDict(self.rewards)
        jump_rewards.update(C.WYW_JUMP_REWARD_WEIGHTS)
        self.rewards = jump_rewards


# ---------------------------------------------------------------------------- #
# Play 变体（少环境、关课程、关随机化事件）
# ---------------------------------------------------------------------------- #
@configclass
class WheelbipeWywFlatEnvCfg_Play(WheelbipeWywFlatEnvCfg):
    events = FduEventCfg()
    curriculum = None


@configclass
class WheelbipeWywRoughEnvCfg_Play(WheelbipeWywRoughEnvCfg):
    events = FduEventCfg()
    curriculum = None


@configclass
class WheelbipeWywJumpEnvCfg_Play(WheelbipeWywJumpEnvCfg):
    events = FduEventCfg()
    curriculum = None
