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
- 物理 500Hz（dt=0.002）、decimation=5 → 100Hz 策略；闭链求解固定 16/6 iterations。
- 命令范围收敛到 fudan：vx±2.1、yaw±2.0，关闭 spin/dash 特殊模式。
- Flat/Rough/Jump 使用各自严格的 Fudan reward term 集合，不混入 V3/V14 shaping。
"""

from __future__ import annotations

import copy
from collections import OrderedDict

import torch
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import NoiseModelCfg, UniformNoiseCfg

import agent_tasks.manager.mdp.isaaclab as mdp
from agent_tasks.direct.wheelbipe.wheelbipe25_v3.env_cfg import Wheelbipe25v3FlatEnvCfg, EventCfg
from agent_world.assets.wheelbipe_fdu import Wheelbipe_FDU_CFG
from agent_world import AssetPath

from . import wyw_constants as C
from .fdu_mapping import POLICY_JOINT_NAMES
from .rough_cfg import FDU_ROUGH_TERRAIN_CFG


FDU_PLANE_REWARDS = OrderedDict(
    tracking_lin_vel=1.0,
    tracking_lin_vel_enhance=1.0,
    tracking_ang_vel=1.0,
    base_height=1.0,
    nominal_state=-1.0,
    lin_vel_z=-1.0,
    ang_vel_xy=-0.05,
    orientation=-15.0,
    dof_vel=-5.0e-5,
    dof_acc=-3.0e-7,
    torques=-1.0e-3,
    action_rate=-0.3,
    action_smooth=-0.3,
    collision=-1.0,
    dof_pos_limits=-1.0,
)

FDU_JUMP_REWARDS = OrderedDict(
    tracking_lin_vel=1.0,
    tracking_lin_vel_enhance=1.0,
    tracking_ang_vel=1.0,
    flight=0.15,
    encourage_jump=1.0,
    base_height_flight=6.0,
    leg_tuck=1.7,
    takeoff_extend=0.5,
    line_z=6.0,
    pen_theta_no0=-2.0,
    action_rate=-0.04,
    torques=-5.0e-5,
    orientation=-25.0,
    ang_vel_xy=-0.10,
    nominal_state=-1.0,
    collision=-1.0,
)


def randomize_fdu_mass_inertia(
    env,
    env_ids,
    added_base_mass_range: tuple[float, float],
    body_scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Apply Fudan's base-mass addition followed by per-body mass/inertia scaling."""
    asset = env.scene[asset_cfg.name]
    ids = torch.arange(env.scene.num_envs, device="cpu") if env_ids is None else env_ids.cpu()
    body_ids = torch.arange(asset.num_bodies, dtype=torch.long, device="cpu")
    masses = asset.data.default_mass.cpu().clone()
    inertias = asset.data.default_inertia.cpu().clone()
    base_add = torch.empty(len(ids), device="cpu").uniform_(*added_base_mass_range)
    scales = torch.empty(len(ids), asset.num_bodies, device="cpu").uniform_(*body_scale_range)
    masses[ids, 0] = masses[ids, 0] + base_add
    centered_base_mass = base_add - base_add.mean()
    env._wyw_base_mass_dev_sample = torch.zeros(env.scene.num_envs, device=asset.device)
    env._wyw_base_mass_dev_sample[ids.to(asset.device)] = centered_base_mass.to(asset.device)
    masses[ids[:, None], body_ids] *= scales
    inertias[ids[:, None], body_ids] *= scales.unsqueeze(-1)
    asset.root_physx_view.set_masses(masses, ids)
    asset.root_physx_view.set_inertias(inertias, ids)


def randomize_fdu_material(
    env,
    env_ids,
    friction_range: tuple[float, float],
    restitution_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Assign one friction and restitution sample to every robot shape per env."""
    asset = env.scene[asset_cfg.name]
    ids = torch.arange(env.scene.num_envs, device="cpu") if env_ids is None else env_ids.cpu()
    props = asset.root_physx_view.get_material_properties()
    friction = torch.empty(len(ids), device="cpu").uniform_(*friction_range)
    restitution = torch.empty(len(ids), device="cpu").uniform_(*restitution_range)
    props[ids, :, 0] = friction[:, None]
    props[ids, :, 1] = friction[:, None]
    props[ids, :, 2] = restitution[:, None]
    asset.root_physx_view.set_material_properties(props, ids)
    env._wyw_friction_sample = torch.zeros(env.scene.num_envs, device=asset.device)
    env._wyw_restitution_sample = torch.zeros(env.scene.num_envs, device=asset.device)
    env._wyw_friction_sample[ids.to(asset.device)] = friction.to(asset.device)
    env._wyw_restitution_sample[ids.to(asset.device)] = restitution.to(asset.device)


def randomize_fdu_base_com(
    env,
    env_ids,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Add and retain Fudan's sampled base COM offset."""
    asset = env.scene[asset_cfg.name]
    ids = torch.arange(env.scene.num_envs, device="cpu") if env_ids is None else env_ids.cpu()
    body_ids, _ = asset.find_bodies("base_link_del")
    if len(body_ids) != 1:
        raise RuntimeError(f"Expected one base_link_del, got {body_ids}")
    bounds = torch.tensor([com_range[axis] for axis in ("x", "y", "z")], device="cpu")
    samples = torch.empty(len(ids), 3, device="cpu").uniform_(0.0, 1.0)
    samples = bounds[:, 0] + samples * (bounds[:, 1] - bounds[:, 0])
    coms = asset.root_physx_view.get_coms().clone()
    coms[ids, body_ids[0], :3] += samples
    asset.root_physx_view.set_coms(coms, ids)
    env._wyw_base_com_sample = torch.zeros(env.scene.num_envs, 3, device=asset.device)
    env._wyw_base_com_sample[ids.to(asset.device)] = samples.to(asset.device)


def randomize_fdu_default_joint_pos(
    env,
    env_ids,
    offset_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Randomize the six policy-joint defaults and retain their critic privilege."""
    asset = env.scene[asset_cfg.name]
    ids = torch.arange(env.scene.num_envs, device=asset.device) if env_ids is None else env_ids.to(asset.device)
    joint_ids = []
    for name in POLICY_JOINT_NAMES:
        indices, _ = asset.find_joints(name)
        if len(indices) != 1:
            raise RuntimeError(f"Expected one policy joint named {name!r}, got {indices}")
        joint_ids.append(int(indices[0]))
    joint_ids = torch.tensor(joint_ids, dtype=torch.long, device=asset.device)
    offsets = torch.empty(len(ids), len(joint_ids), device=asset.device).uniform_(*offset_range)
    asset.data.default_joint_pos[ids[:, None], joint_ids] += offsets
    env._wyw_default_dof_delta_sample = torch.zeros(env.scene.num_envs, 6, device=asset.device)
    env._wyw_default_dof_delta_sample[ids] = offsets


# ---------------------------------------------------------------------------- #
# 工具：把 fudan 观测/命令/控制频率强制写到一个已 __post_init__ 过的 cfg 上
# ---------------------------------------------------------------------------- #
def _apply_wyw_common(cfg) -> None:
    """在 super().__post_init__() 之后，强制 fudan 的 obs 形状 / 命令 / 100Hz。"""
    # FDU maximal-coordinate loop constraints need 500 Hz at the accepted
    # training workspace. Keep the Fudan policy period at 100 Hz.
    cfg.sim = copy.deepcopy(cfg.sim)
    cfg.sim.dt = 0.002
    cfg.decimation = 5
    cfg.sim.render_interval = cfg.decimation
    cfg.robot_cfg = copy.deepcopy(cfg.robot_cfg)
    cfg.robot_cfg.spawn.articulation_props.solver_position_iteration_count = 16
    cfg.robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = 6
    cfg.sim.physics_material = copy.deepcopy(cfg.sim.physics_material)
    cfg.sim.physics_material.friction_combine_mode = "average"
    cfg.sim.physics_material.restitution_combine_mode = "average"
    cfg.sim.physics_material.static_friction = 0.5
    cfg.sim.physics_material.dynamic_friction = 0.5
    cfg.sim.physics_material.restitution = 0.5
    cfg.terrain.physics_material = copy.deepcopy(cfg.terrain.physics_material)
    cfg.terrain.physics_material.friction_combine_mode = "average"
    cfg.terrain.physics_material.restitution_combine_mode = "average"
    cfg.terrain.physics_material.static_friction = 0.5
    cfg.terrain.physics_material.dynamic_friction = 0.5
    cfg.terrain.physics_material.restitution = 0.5

    # fudan 观测形状。stock DirectRLEnv._configure_gym_env_spaces 会把
    # observation_space 原样交给 spec_to_gym_space —— 传 dict 会被整体嵌套进
    # single_observation_space["policy"]（各子键 flatdim 相加），从而丢失 policy_hist
    # 顶层键。故这里用 int：
    #   single_observation_space = {policy: Box(25), critic: Box(141)}
    #   → 自定义 wrapper: num_observations={policy:25, critic:141}, num_privileged_obs=141
    #   → runner 对 policy_hist 回落到 num_obs_hist * num_obs = 5*25 = 125（=encoder 输入）。
    cfg.observation_space = C.WYW_POLICY_OBS_DIM
    cfg.state_space = C.WYW_CRITIC_DIM
    cfg.num_obs_hist = C.WYW_NUM_OBS_HIST

    # The Fudan baseline has no V3/V14 runtime state machines. Disable the
    # manager at its top-level gate for every WYW task, rather than relying on
    # each currently-known child machine also happening to be disabled.
    cfg.enable_state_machines = False
    cfg.airborne_state_machine_cfg = copy.deepcopy(cfg.airborne_state_machine_cfg)
    cfg.airborne_state_machine_cfg["enabled"] = False
    cfg.wheel_forward_scan_cfg = copy.deepcopy(cfg.wheel_forward_scan_cfg)
    cfg.wheel_forward_scan_cfg["enabled"] = False

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
    if getattr(cfg, "right_wheel_height_scanner", None) is not None:
        cfg.right_wheel_height_scanner = copy.deepcopy(cfg.right_wheel_height_scanner)
        cfg.right_wheel_height_scanner.prim_path = "/World/envs/env_.*/Robot/r_wheel_Link"
    if getattr(cfg, "left_wheel_height_scanner", None) is not None:
        cfg.left_wheel_height_scanner = copy.deepcopy(cfg.left_wheel_height_scanner)
        cfg.left_wheel_height_scanner.prim_path = "/World/envs/env_.*/Robot/l_wheel_Link"

    # 命令：直接采样 yaw-rate；各任务在自己的 __post_init__ 中设置 vx 和周期。
    cfg.commands = copy.deepcopy(cfg.commands)
    cfg.commands.heading_command = False
    cfg.commands.rel_heading_envs = 0.0
    cfg.commands.rel_standing_envs = 0.0
    cfg.commands.resampling_time_range = (5.0, 5.0)
    cfg.commands.debug_vis = bool(getattr(cfg, "play", False))
    ranges = getattr(cfg.commands, "ranges", None)
    if ranges is not None:
        ranges.lin_vel_x = (-2.0, 2.0)
        ranges.lin_vel_y = (0.0, 0.0)
        ranges.ang_vel_z = (-2.0, 2.0)

    # 高度命令区间锁死为当前 FDU 任务配置（rough helper 可能改写，故这里强制回来）。
    # This is a task command/reward range; physical entity targets remain
    # independently projected into the calibrated L0/theta0 workspace.
    cfg.height_range = [0.15, 0.30]
    if cfg.terrain.terrain_type == "plane":
        cfg.terrain.terrain_type = "usd"
        cfg.terrain.usd_path = f"{AssetPath}/usd_files/flat_ground.usda"


@configclass
class FduEventCfg(EventCfg):
    """V3 domain randomization retargeted to exact FDU entity names."""

    mass_inertia = EventTerm(
        func=randomize_fdu_mass_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "added_base_mass_range": (-1.0, 2.0),
            "body_scale_range": (0.9, 1.1),
        },
    )
    add_base_mass = None
    add_leg_mass = None
    add_wheel_mass = None
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform_vel_b,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {
                axis: (-0.5, 0.5) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
            },
        },
    )
    base_com = EventTerm(
        func=randomize_fdu_base_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link_del"),
            "com_range": {axis: (-0.02, 0.02) for axis in ("x", "y", "z")},
        },
    )
    physics_material = EventTerm(
        func=randomize_fdu_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "friction_range": (0.6, 1.4),
            "restitution_range": (0.6, 1.0),
        },
    )
    default_joint_pos = EventTerm(
        func=randomize_fdu_default_joint_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(POLICY_JOINT_NAMES)),
            "offset_range": (-0.03, 0.03),
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(POLICY_JOINT_NAMES)),
            "stiffness_distribution_params": (0.95, 1.05),
            "damping_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )
    motor_torque = EventTerm(
        func=mdp.randomize_actuator_effort_output,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(POLICY_JOINT_NAMES)),
            # 40 N*m is a hard ceiling; domain randomization only weakens it.
            "effort_scale_distribution_params": (0.95, 1.0),
        },
    )
    push_robot = None
    base_external_force_torque_xyz = None


@configclass
class FduJumpEventCfg(FduEventCfg):
    mass_inertia = EventTerm(
        func=randomize_fdu_mass_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "added_base_mass_range": (-2.0, 3.0),
            "body_scale_range": (0.8, 1.2),
        },
    )
    base_com = EventTerm(
        func=randomize_fdu_base_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link_del"),
            "com_range": {axis: (-0.05, 0.05) for axis in ("x", "y", "z")},
        },
    )
    physics_material = EventTerm(
        func=randomize_fdu_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "friction_range": (0.1, 2.0),
            "restitution_range": (0.5, 1.0),
        },
    )
    default_joint_pos = EventTerm(
        func=randomize_fdu_default_joint_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(POLICY_JOINT_NAMES)),
            "offset_range": (-0.05, 0.05),
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(POLICY_JOINT_NAMES)),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    motor_torque = EventTerm(
        func=mdp.randomize_actuator_effort_output,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(POLICY_JOINT_NAMES)),
            # 40 N*m is a hard ceiling; domain randomization only weakens it.
            "effort_scale_distribution_params": (0.9, 1.0),
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5)}},
    )


@configclass
class FduRoughEventCfg(FduEventCfg):
    """Fudan rough reset: random XY within one metre of the terrain tile origin."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform_vel_b,
        mode="reset",
        params={
            "pose_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
            "velocity_range": {
                axis: (-0.5, 0.5) for axis in ("x", "y", "z", "roll", "pitch", "yaw")
            },
        },
    )


@configclass
class FduPlayEventCfg(FduEventCfg):
    base_com = None
    physics_material = None
    robot_joint_stiffness_and_damping = None
    mass_inertia = None
    default_joint_pos = None
    motor_torque = None


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
    self_obs_noise_cfg = {
        "root_ang_vel_b": NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.2, n_max=0.2)),
        "projected_gravity_b": NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.05, n_max=0.05)),
        "joint_pos": NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-0.02, n_max=0.02)),
        "leg_joint_vel": NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-1.5, n_max=1.5)),
        "wheel_joint_vel": NoiseModelCfg(noise_cfg=UniformNoiseCfg(n_min=-1.5, n_max=1.5)),
    }
    leg_action_scale = 0.5
    wheel_vel_action_scale = 10.0
    # Conservative wheel-output cap for the M3508+C620 with the P13.71
    # gearbox. It is derived from the P19 load curve, whose speed falls from
    # 500 rpm at zero load to about 450 rpm at 4.5 N m before ratio conversion.
    max_wheel_vel = 60.0
    # Persist the active action/reward/termination contract in params/env.yaml
    # for run auditing. Checkpoint compatibility remains an operator decision.
    wyw_training_semantics_version = "fdu_flat_p0_direct_bars_v1"
    termination_duration_enabled = True
    termination_duration_steps = 100
    # Fudan Plane treats all non-wheel leg links plus the base as the
    # rf/lf/base collision and failure-contact set.
    wyw_failure_contact_body_patterns = [
        "base_link_del",
        "[lr]f[01]_Link",
        "[lr]2[0-3]_Link",
    ]
    wyw_collision_contact_force = 0.1
    wyw_failure_contact_force = 10.0
    # Plane command curriculum: at the 20 s global cadence, expand vx by 0.1
    # after mean linear/yaw tracking rates exceed 0.70/0.56, capped at +/-2.5.
    wyw_flat_command_curriculum_enabled = True
    wyw_command_curriculum_interval_steps = 2000
    wyw_command_curriculum_lin_threshold = 0.7
    wyw_command_curriculum_yaw_threshold = 0.56
    wyw_command_curriculum_step = 0.1
    wyw_command_curriculum_max_abs = 2.5
    clip_single_reward = 1.0
    only_positive_rewards = False
    rewards = copy.deepcopy(FDU_PLANE_REWARDS)

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
    tracking_sigma = 0.25
    # This is a diagnostic boundary, not an action clamp or termination.
    # The 500 Hz / 16/6 scans showed persistent loop limit cycles below it.
    wyw_l0_stability_monitor_enabled = True
    wyw_l0_stability_boundary_m = 0.14
    wyw_l0_stability_check_interval_steps = 10
    wyw_l0_stability_warning_interval_steps = 1000
    wyw_l0_stability_log_max_env_ids = 16
    wyw_l0_tuck = 0.23
    wyw_l0_extend = 0.31
    wyw_base_height_flight = 0.65
    wyw_takeoff_vz = 0.15
    wyw_flight_contact_force = 1.0

    # 跳跃奖励注入开关（Flat/Rough 关闭）
    wyw_jump_enabled = False
    wyw_rough_curriculum_enabled = False

    def __post_init__(self):
        super().__post_init__()
        _apply_wyw_common(self)
        self.scene.num_envs = 4096
        self.commands.ranges.lin_vel_x = (-2.0, 2.0)
        self.commands.resampling_time_range = (5.0, 5.0)


@configclass
class WheelbipeWywRoughEnvCfg(WheelbipeWywFlatEnvCfg):
    """wyw Rough：trimesh 地形 + 课程，obs/reward/网络与 Flat 共享。"""

    events = FduRoughEventCfg()
    wyw_flat_command_curriculum_enabled = False
    wyw_rough_curriculum_enabled = True
    rough_terrain_generator_cfg = copy.deepcopy(FDU_ROUGH_TERRAIN_CFG)
    rough_terrain_boundary_reset_cfg = {
        "enabled": True,
        "margin": 1.0,
        "use_inner_terrain_area": False,
    }

    def __post_init__(self):
        super().__post_init__()
        # 这里只复用 generator 作为当前的 rough 几何载体，不引入 V14 的
        # airborne/forward-scan 状态机、特殊命令覆盖或 iteration 课程。
        self.terrain = copy.deepcopy(self.terrain)
        self.terrain.terrain_type = "generator"
        self.terrain.terrain_generator = copy.deepcopy(self.rough_terrain_generator_cfg)
        self.terrain.terrain_generator.curriculum = True
        self.terrain.max_init_terrain_level = 5
        self.enable_state_machines = False
        self.airborne_state_machine_cfg = copy.deepcopy(self.airborne_state_machine_cfg)
        self.airborne_state_machine_cfg["enabled"] = False
        self.wheel_forward_scan_cfg = copy.deepcopy(self.wheel_forward_scan_cfg)
        self.wheel_forward_scan_cfg["enabled"] = False
        self.terrain_command_overrides = {}
        _apply_wyw_common(self)
        self.commands.ranges.lin_vel_x = (-2.0, 2.0)
        self.commands.resampling_time_range = (5.0, 5.0)


@configclass
class WheelbipeWywJumpEnvCfg(WheelbipeWywFlatEnvCfg):
    """wyw Jump：平地 + locomotion + fudan 涌现式跳跃奖励（无显式起跳状态机）。"""

    wyw_jump_enabled = True
    wyw_flat_command_curriculum_enabled = False
    events = FduJumpEventCfg()

    # fudan jump 变体的 lin_vel obs_scale = 3.0（plane 版为 2.0）。该字段同时驱动
    # 命令 vx 缩放与 critic base_lin_vel 缩放（也即 encoder 监督目标 = base_lin_vel×3.0），
    # 与 fudan commands_scale[0]==obs_scales.lin_vel 的耦合一致。
    wyw_lin_vel_scale = 3.0
    clip_single_reward = 2.5

    def __post_init__(self):
        super().__post_init__()
        self.robot_cfg = copy.deepcopy(self.robot_cfg)
        self.robot_cfg.actuators["legs_act"].stiffness = 6.0
        self.robot_cfg.actuators["legs_act"].damping = 0.5
        self.robot_cfg.actuators["wheel"].effort_limit = 50.0
        self.rewards = copy.deepcopy(FDU_JUMP_REWARDS)
        self.scene.num_envs = 4096
        self.commands.ranges.lin_vel_x = (-2.1, 2.1)
        self.commands.resampling_time_range = (20.0, 20.0)


# ---------------------------------------------------------------------------- #
# Play 变体（少环境、关课程、关随机化事件）
# ---------------------------------------------------------------------------- #
@configclass
class WheelbipeWywFlatEnvCfg_Play(WheelbipeWywFlatEnvCfg):
    events = FduPlayEventCfg()
    curriculum = None
    play = True
    wyw_flat_command_curriculum_enabled = False
    self_obs_noise_cfg = None

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.height_scanner.debug_vis = False


@configclass
class WheelbipeWywRoughEnvCfg_Play(WheelbipeWywRoughEnvCfg):
    events = FduPlayEventCfg()
    curriculum = None
    play = True
    self_obs_noise_cfg = None

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.height_scanner.debug_vis = False


@configclass
class WheelbipeWywJumpEnvCfg_Play(WheelbipeWywJumpEnvCfg):
    events = FduPlayEventCfg()
    curriculum = None
    play = True
    self_obs_noise_cfg = None

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.height_scanner.debug_vis = False
