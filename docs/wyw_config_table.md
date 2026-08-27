# wyw 任务族配置核对表

> **用途**：人工核对 wyw 环境的全部关键配置。
> **数据来源（权威优先）**：
> 1. `logs/rsl_rl/wheelbipe_v14_wyw_flat_direct/2026-08-27_20-33-42/params/{env,agent}.yaml`
>    —— 这是 **Flat 任务实际实例化后、所有继承已解析** 的快照（最权威）。
> 2. `source/agent_tasks/agent_tasks/direct/wheelbipe/wyw/wyw_constants.py`（wyw 特有常量宏）
> 3. `source/agent_world/agent_world/assets/wheelbipe_V14_2.py`（机器人 CFG 源）
> 4. `source/agent_tasks/.../wyw/env_cfg.py`、`env.py`（Rough/Jump 差异、obs 组装、jump 奖励）
>
> 表中 `WYW_ROBOT="from_hu"`（首版）。⚠️ = 需重点核对 / 与文档或部署契约可能不一致处。
> 生成日期：2026-08-27。
>
> **本次同步（2026-08-27 第二轮，工作区未暂存改动）**：
> 1. **每任务独立 runner cfg**：新增 `WheelbipeWywRoughPPORunnerCfg` / `WheelbipeWywJumpPPORunnerCfg`
>    （均继承 `WheelbipeWywPPORunnerCfg`，仅 `experiment_name` 不同）；`wyw/__init__.py` 里
>    Rough/Jump 注册改指向各自 runner（`_RUNNER_ROUGH` / `_RUNNER_JUMP`），Flat 用 `_RUNNER_FLAT`。
>    → 第 1 节、§13-3 已更新（此前是"计划中"，现已落地）。
> 2. **height_range 强制锁定**：`_apply_wyw_common` 末尾新增 `cfg.height_range = [0.20, 0.42]`
>    （防止 rough helper 改成 [0.2,0.3]）。→ 第 5、12 节已更新。
> 3. **obs 缩放常量搬家（✅ 已补齐为 configclass 字段）**：obs 缩放从 `wyw_constants.py` 常量
>    迁移为 `env_cfg.py` 中 `WheelbipeWywFlatEnvCfg` 的 `wyw_*_scale` configclass 字段（IsaacLab /
>    `wheelbipe25_v3` 风格），`env.py` 改读 `self.cfg.wyw_*_scale`。→ 第 3 节已改写。
>    （注：本轮首次交付时该重构只做了一半会崩，现已补齐 env_cfg 字段定义 + env.py 读取。）

---

## 0. 任务注册总览（6 个）

| 任务 id | env_cfg 类 | 地形 | jump 奖励 | 说明 |
|---|---|---|---|---|
| `Robotics-Wheelbipe-V14-wyw-Flat-v1` | `WheelbipeWywFlatEnvCfg` | plane | ✗ | 平地 locomotion |
| `Robotics-Wheelbipe-V14-wyw-Flat-Play-v1` | `..._Play` | plane | ✗ | Play（关课程/关随机化事件） |
| `Robotics-Wheelbipe-V14-wyw-Rough-v1` | `WheelbipeWywRoughEnvCfg` | trimesh (RM_ROTATION_TERRAINS_CFG_99) | ✗ | 粗糙地形 + 课程 |
| `Robotics-Wheelbipe-V14-wyw-Rough-Play-v1` | `..._Play` | trimesh | ✗ | Play |
| `Robotics-Wheelbipe-V14-wyw-Jump-v1` | `WheelbipeWywJumpEnvCfg` | plane | ✓ 6 项 | 平地 + 涌现式跳跃 |
| `Robotics-Wheelbipe-V14-wyw-Jump-Play-v1` | `..._Play` | plane | ✓ 6 项 | Play |

三个主任务**共享** Actor/Critic/Obs/网络/PPO 超参。`entry_point` 均为 `WheelbipeWywEnv`；
`rsl_rl_cfg_entry_point` 均为 `WheelbipeWywPPORunnerCfg`。

---

## 1. 网络与 PPO 超参（来源：agent.yaml）

| 项 | 值 | 备注 |
|---|---|---|
| `runner_class` | `OnPolicySequenceRunner` | 实际生效的 runner（`class_name: OnPolicyRunner` 是占位默认，被 runner_class 覆盖） |
| `policy.class_name` | `ActorCriticSequence` | |
| `algorithm.class_name` | `PPOSequence` | 双优化器（主 + encoder extra_optimizer） |
| `latent_dim` | 3 | encoder 输出 = 隐式基座线速度估计 |
| `init_noise_std` | 0.5 | |
| `encoder_hidden_dims` | [128, 64] | 输入 125 → latent 3 |
| `actor_hidden_dims` | [128, 64, 32] | 输入 = 25 + latent 3 = 28 |
| `critic_hidden_dims` | [256, 128, 64] | 输入 = 46 + latent 3 = 49 |
| `activation` | elu | |
| `num_steps_per_env` | 48 | rollout 长度 |
| `num_learning_epochs` | 5 | |
| `num_mini_batches` | 4 | |
| `learning_rate` | 0.001 | 主优化器（adaptive KL） |
| `extra_learning_rate` | 0.001 | encoder 优化器 |
| `schedule` | adaptive | |
| `desired_kl` | 0.005 | |
| `gamma` | 0.99 | |
| `lam` | 0.95 | |
| `clip_param` | 0.2 | |
| `entropy_coef` | 0.01 | |
| `value_loss_coef` | 1.0 | |
| `max_grad_norm` | 1.0 | |
| `experiment_name` | Flat=`wheelbipe_v14_wyw_flat_direct` / Rough=`..._rough_direct` / Jump=`..._jump_direct` | ✅ 三任务日志目录已分开。本轮已落地：`WheelbipeWywRoughPPORunnerCfg` / `WheelbipeWywJumpPPORunnerCfg` 均继承 `WheelbipeWywPPORunnerCfg`（后者继承 `SequencePPORunnerCfg`），**仅** `experiment_name` 不同、其余超参完全继承共享。`wyw/__init__.py` 中 Flat/Rough/Jump 分别指向 `_RUNNER_FLAT`/`_RUNNER_ROUGH`/`_RUNNER_JUMP`；Play 变体复用对应主任务 runner（同 name），便于 play.py 加载 checkpoint。 |

**encoder 监督**：`PPOSequence` 每次更新加 `MSE(latent[:, :3], critic_obs[:, :3])`，
即 latent 前 3 维回归 `base_lin_vel × 2.0`。

---

## 2. 观测维度与布局（来源：wyw_constants.py + env.py + env.yaml）

| 项 | 值 |
|---|---|
| `observation_space` (policy) | **25** (int) |
| `state_space` (critic) | **46** (int) |
| `num_obs_hist` | 5 |
| encoder 输入 (policy_hist) | 125 = 5 × 25（runner 回落推导，非 obs dict 键） |
| `num_privileged_obs_hist` | 1 |
| `action_space` | 6 |
| `ctrl_mode_obs_enabled` | False（关闭基类 7 维 ctrl_mode 块） |
| `use_frame_stack` | False |

### Policy 观测（25 维，`_build_wyw_policy_obs`）

| 段 | 维 | 缩放 | 源 |
|---|---|---|---|
| 机身角速度 `obs_root_ang_vel_b` | 3 | ×0.25 | 基类延迟/带噪副本 |
| 投影重力 `obs_projected_gravity_b` | 3 | ×1.0 | 同上 |
| 命令 `[vx, yaw, height]` | 3 | vx×2.0 / yaw×0.25 / height×1.0 | `_get_wyw_command_block` |
| 腿关节位置 `obs_joint_pos[:, :4]` | 4 | ×1.0 | 前 4 = 驱动腿关节 |
| 关节速度 `obs_joint_vel` | 6 | ×0.05 | 4 腿 + 2 轮 |
| 上一步动作 `_actions` | 6 | ×1.0 | |

### Critic 特权观测（46 维，`_build_wyw_critic_obs`）

| 段 | 维 | 缩放 | 备注 |
|---|---|---|---|
| **基座线速度 `root_lin_vel_b`** | **3** | **×2.0** | **必须是前 3 维**（encoder 监督目标） |
| 机身角速度 `root_ang_vel_b` | 3 | ×0.25 | |
| 投影重力 `projected_gravity_b` | 3 | ×1.0 | |
| 命令 `[vx, yaw, height]` | 3 | 同 policy | |
| 腿关节偏差 `(joint_pos-default)[_legs_act_idx]` | 4 | ×1.0 | |
| 关节速度 `joint_vel[_actuate_idx]` | 6 | ×0.05 | |
| 当前动作 `_actions` | 6 | ×1.0 | |
| 前前步动作 `_before_previous_actions` | 6 | ×1.0 | |
| 关节加速度 `joint_acc[_actuate_idx]` | 6 | ×0.0025 | critic 专用 |
| 关节力矩 `applied_torque[_actuate_idx]` | 6 | ×0.05 | critic 专用 |

> ⚠️ Critic **不含**地形高度扫描、域随机化特权参数（base_mass/com/friction 等）。原计划估算
> ≈141，as-built 精简为 46。若后续需要 sim2real 鲁棒性可再追加（并同步 `state_space`）。

---

## 3. 观测缩放（obs_scales）——configclass 字段（env_cfg.py）

> **本轮改动（已完成）**：obs 缩放**从 `wyw_constants.py` 常量迁移为 `env_cfg.py` 的 configclass
> 字段**，挂在 `WheelbipeWywFlatEnvCfg` 上（Rough/Jump 继承），字段名 `wyw_*_scale`；`env.py`
> 通过 `self.cfg.wyw_*_scale` 读取。采用 IsaacLab / `wheelbipe25_v3`（`lin_vel_scale=1.0` …）的
> 编码风格。好处：① 随 `params/env.yaml` 自动落盘，便于复现 / 审计 / 对齐部署端；② 可按任务覆写
> （如 Jump 单独把 `wyw_lin_vel_scale` 调成 3.0）；③ scale 是配置而非几何常量。
> `wyw_constants.py` 只保留：网络/观测**维度**、**几何**目标（L0、滞空高度）、物理**阈值**、跳跃**奖励权重**。

| configclass 字段（`WheelbipeWywFlatEnvCfg`） | 值 | 用途 |
|---|---|---|
| `wyw_ang_vel_scale` | 0.25 | 机身角速度 |
| `wyw_dof_vel_scale` | 0.05 | 关节速度 |
| `wyw_lin_vel_scale` | 2.0 | 命令 vx + critic base_lin_vel |
| `wyw_cmd_ang_vel_scale` | 0.25 | 偏航命令 |
| `wyw_height_cmd_scale` | **1.0** | 高度命令 ⚠️ 见第 5 节注意点 |
| `wyw_proj_gravity_scale` | 1.0 | |
| `wyw_joint_pos_scale` | 1.0 | |
| `wyw_action_scale` | 1.0 | **obs 里** action 段的缩放（≠ env 级动作输出缩放 `action_scale=0.25`，见第 4 节） |
| `wyw_joint_acc_scale` | 0.0025 | critic |
| `wyw_torque_scale` | 0.05 | critic |

> ⚠️ 命名注意：obs action 段缩放叫 `wyw_action_scale`（=1.0），与基类 env 级动作输出缩放
> `action_scale`（=0.25，第 4 节）**是不同字段**，勿混淆。

---

## 4. 仿真 / 控制频率 / 动作（env.yaml）

| 项 | 值 | 备注 |
|---|---|---|
| `sim.dt` | 0.005 | 物理 200 Hz |
| `decimation` | 2 | → policy 100 Hz（policy dt=0.01s）✅ 对齐 fudan |
| `sim.render_interval` | 4 | |
| `sim.gravity` | (0, 0, −9.81) | |
| `episode_length_s` | 20.0 | |
| `action_space` | 6 | 4 腿关节位置 + 2 轮速 |
| **`action_scale`** | **0.25** | ⚠️ **env 级动作输出缩放**（基类），与 obs 里的 `WYW_ACTION_SCALE=1.0` 是**两回事**。⚠️ 需核对：docs/intention.md 部署契约写"腿 0.5 / 轮 10"——请确认腿/轮是否分别再映射，以及是否与该 0.25 一致。 |

---

## 5. 命令（env.yaml，`commands`）

| 项 | 值 | 备注 |
|---|---|---|
| `class_type` | `SpecialModeUniformVelocityCommand` | |
| `ranges.lin_vel_x` | (−2.1, 2.1) | ✅ fudan 范围 |
| `ranges.lin_vel_y` | (0.0, 0.0) | ✅ 无侧向 |
| `ranges.ang_vel_z` | (−2.0, 2.0) | ✅ |
| `ranges.heading` | (−π, π) | |
| `heading_command` | True | |
| `heading_control_stiffness` | 5.0 | |
| `rel_standing_envs` | 0.1 | 10% 站立指令 |
| `rel_heading_envs` | 0.5 | |
| `resampling_time_range` | (5.0, 15.0) s | |
| `special_modes[0/1/2].rel_envs` | **0.0 / 0.0 / 0.0** | ✅ spin/dash 全关闭（`_apply_wyw_common`） |

### 高度命令（env.yaml）

| 项 | 值 | 备注 |
|---|---|---|
| `default_height_cmd` | 0.22 | from_hu 骑行高度默认 |
| `height_range` | [0.20, 0.42] | ✅ from_hu 骑行高度量纲（非 fudan 站高）。本轮起在 `_apply_wyw_common` 末尾**强制锁定** `cfg.height_range=[0.20,0.42]`，防止 rough helper（`_apply_v14_rough_runtime_cfg`）改成 [0.2,0.3]。 |
| `use_absolute_height` | True | |
| `use_leg_length_as_height` | False | |
| `height_command_special_modes_cfg.enabled` | False | 正弦/阶跃高度命令关闭 |
| `obs_input_scale_cfg.height_cmd`（基类 obs 机制） | 5.0 | ⚠️ **wyw 未用基类 obs 机制**；wyw 里 height obs 实际缩放 = `WYW_HEIGHT_CMD_SCALE = 1.0`。请确认部署端 height 缩放与训练一致（1.0 而非 5.0）。 |

---

## 6. 终止条件（env.yaml，复用基类 `_get_dones`）

| 项 | 值 | 备注 |
|---|---|---|
| `termination_roll_deg` | 40.0 | |roll| 超过则终止 |
| `termination_pitch_deg` | 40.0 | |pitch| 超过则终止 |
| `fail_to_terminal_time_s` | (未设) | ⚠️ wyw 未设 fudan 式 fail_buf 宽限，直接用基类接触/倾倒/超时终止 |
| `termination`（奖励项） | −200.0 | 终止惩罚 |

---

## 7. Reset 初始状态随机化（env.yaml）—— “劈叉”来源

| 项 | 值 | 备注 |
|---|---|---|
| `use_leg_random_start` | **True** | 每次 reset 随机腿姿 |
| `leg_length_range` | **[0.13, 0.32]** m | 左右腿**独立**采样腿长 L0 |
| `leg_angle_range` | **[−π/2, 0.75π]** = **[−90°, +135°]** | 左右腿**独立**摆角 → 视觉“劈叉” |
| `wheel_angle_range` | [−2π, 2π] | 轮转角随机 |
| `use_joint_vel_random_start` | True | |
| `leg_joint_vel_range` | [−π/2, π/2] | 腿关节初速度 |
| `wheel_joint_vel_range` | [−50, 50] | 轮初速度 |

> 这是**故意的鲁棒性随机化**，非 bug。Play 变体用 `EventCfgV14_Play`（随机化事件减弱）。
> 若目视调试想要固定站姿，可临时设 `use_leg_random_start=False` / `use_joint_vel_random_start=False`。

---

## 8. Locomotion 奖励（env.yaml，Flat/Rough/Jump 共用）

> as-built 决定：**沿用 from_hu V14 的 rewards OrderedDict**（不是 fudan 的项名）。
> 基类只对下表列出的键求和；权重 0 的项等于关闭但仍被记录。

| 奖励项 | 权重 |
|---|---|
| `termination` | −200.0 |
| `track_lin_vel_xy` | 1.0 |
| `track_lin_vel_xy_square` | −1.0 |
| `track_ang_vel_z` | 1.0 |
| `track_ang_vel_z_square` | −1.0 |
| `track_height_exp_tight` | 1.0 |
| `track_height_square` | −1.0 |
| `stand_still_lin_vel` | −1.0 |
| `lin_vel_z` | −0.5 |
| `ang_vel_xy` | −0.05 |
| `flat_orientation_y_v` | −2.0 |
| `flat_orientation_y_exp` | 1.0 |
| `flat_orientation_x_v` | −2.0 |
| `flat_orientation_x_exp` | 1.0 |
| `action_smoothness_leg` | −0.05 |
| `action_smoothness_wheel` | −0.01 |
| `action_rate` | −0.01 |
| `leg_joint_acc` | −5e-07 |
| `leg_joint_vel` | −0.005 |
| `joint_torque` | −0.0001 |
| `wheel_acc` | −1e-08 |
| `wheel_vel` | −1e-05 |
| `wheel_power` | −0.0001 |
| `no_fork` | −1.0 |
| `no_fork_square` | −1.0 |
| `undesired_contact` | −2.0 |

（权重为 0 的项：`leg_joint_pair_pos_diff`、`wheel_air_spin`、`stand_still`、`track_lin_vel_xy_tight`、
`track_height_exp`、`track_height_exp_soft`、`track_height_exp_both_wheels_contact`、
`flat_orientation_x/y`、`no_fork_exp`、`no_fork_z_exp` —— 已列在表内但当前关闭。）

---

## 9. Jump 奖励（仅 Jump 任务，wyw_constants.py + env.py `_compute_wyw_jump_terms`）

| 奖励项 | 权重 | 公式（per-step 速率，后 ×step_dt） | 门控 |
|---|---|---|---|
| `base_height_flight` | 6.0 | `exp(−|root_z − 0.60|·6)` | in_flight |
| `leg_tuck` | 1.7 | `exp(−(|L0ₗ−0.16|+|L0ᵣ−0.16|)·4)` | in_flight |
| `takeoff_extend` | 0.5 | `exp(−(|L0ₗ−0.31|+|L0ᵣ−0.31|)·4)` | any_contact & vz>0.15 |
| `line_z` | 6.0 | `clamp(vz, min=0)` | in_flight |
| `flight` | 0.15 | 常数 1 | in_flight |
| `encourage_jump` | 1.0 | `clamp(root_z, 0, 0.5)` | in_flight |

**门控判定**（`_get_wheel_contact_force_peaks`）：
- `in_flight` = 所有轮接触力峰值 < `WYW_FLIGHT_CONTACT_FORCE = 1.0` N
- `any_contact` = 任一轮峰值 > 1.0 N
- `WYW_TAKEOFF_VZ = 0.15` m/s（判定“正在蹬伸起跳”）

> Jump 任务 `wyw_jump_enabled=True`；`__post_init__` 把上表 6 项合并进 `cfg.rewards`，
> 与第 8 节 locomotion 奖励**叠加**。`jump_takeoff_state_machine` 保持禁用（涌现式，非状态机）。

---

## 10. Jump 几何常量（随 `WYW_ROBOT` 切换，wyw_constants.py）

| 常量 | from_hu（当前） | fudan（换 USD 后） | 说明 |
|---|---|---|---|
| `WYW_L0_TUCK` | 0.16 | 0.16 | 收腿目标腿长 |
| `WYW_L0_EXTEND` | 0.31 | 0.31 | 蹬伸目标腿长 |
| `WYW_BASE_HEIGHT_FLIGHT` | **0.60** | 0.65 | ⚠️ 滞空期望机身高度（from_hu 为待实测初值） |
| `WYW_TAKEOFF_VZ` | 0.15 | 0.15 | 与本体无关 |
| `WYW_FLIGHT_CONTACT_FORCE` | 1.0 | 1.0 | 离地接触力阈值 (N) |
| `WYW_FALL_CONTACT_FORCE` | 10.0 | 10.0 | （当前复用基类 _get_dones，未直接使用） |
| `WYW_AIRTIME_HEIGHT_CLIP` | 0.5 | 0.5 | encourage_jump 高度裁剪 |

> L0 = `‖wheel_pos_heading_b‖`，从仿真实际连杆位姿读取，与杆长无关、换 USD 自动正确。

---

## 11. 机器人执行器与初始位姿（源：wheelbipe_V14_2.py，`Wheelbipe_V14_2_CFG`）

**初始位姿**：`pos = (0, 0, 0.38)`；所有 `joint_pos = 0.0`（USD 静态初值，reset 时被第 7 节随机化覆盖）。

| 执行器组 | 关节 | stiffness (Kp) | damping (Kd) | effort_limit | velocity_limit |
|---|---|---|---|---|---|
| `legs_act` (IdealPD) | `*_rear1_joint`, `*_front1_joint` | **60.0** | 2.0 | 40.0 N·m | 17 |
| `legs_inact` (IdealPD, 被动) | `*_rear2/front2/front3/front4/spring1/guide_joint` | 0.0 | 0.01 | 50.0 | 300.0 |
| `wheel` (IdealPD) | `*_wheel_joint` | 0.0 | 0.2 | 5.0 | 60.0 |
| `spring` (IdealPD) | `*_spring2_joint` | 0.0 | (见源文件) | — | — |

> ⚠️ **起跳能力风险**：`legs_act` Kp=60、effort=40 N·m，比 fudan jump 用的软增益（Kp≈6）硬得多。
> Jump 任务可能蹬不起来，或需单独调软腿 PD / 换 nominal 站姿。见 docs/intention.md 风险项。
> ⚠️ **右后连杆碰撞网格警告**：view_robot 日志出现
> `right_rear2_link/collisions/mesh_0` non-existent path（不影响运行，疑似 USD 结构问题）。

---

## 12. 三任务差异对比

| 项 | Flat | Rough | Jump |
|---|---|---|---|
| 地形 | plane | trimesh `RM_ROTATION_TERRAINS_CFG_99` | plane |
| 地形边界复位 | ✗ | ✓ (margin 0.5) | ✗ |
| 课程 / 高度扫描 / 状态机 | 关 | 经 `_apply_v14_rough_runtime_cfg` 开启 | 关 |
| `wyw_jump_enabled` | False | False | **True** |
| 奖励 | locomotion（第 8 节） | locomotion | locomotion + jump 6 项（第 9 节） |
| Obs / Critic / 网络 / PPO | 完全共享 | 共享 | 共享 |
| decimation / 命令范围 / **height_range** | `_apply_wyw_common` 强制一致（decimation=2、obs/critic int 形状、命令范围、**height_range=[0.20,0.42]**） | 同（rough helper 后再强制一次） | 同 |

> Rough 的 `__post_init__` 顺序：`super()` → `_apply_v14_rough_runtime_cfg(self)` →
> **再次** `_apply_wyw_common(self)`（防止 rough helper 覆盖 obs 形状 / 频率 / 命令）。

---

## 13. ⚠️ 需重点人工核对清单

1. **动作输出缩放**：env `action_scale = 0.25`。核对腿/轮是否各自再映射（部署契约称腿 0.5 / 轮 10），
   训练与部署是否一致。（obs 里的 `wyw_action_scale=1.0` 是另一回事，勿混淆。）
2. **height 命令 obs 缩放**：wyw 用 `wyw_height_cmd_scale=1.0`，基类 obs 机制是 5.0。确认部署端一致。
3. ✅ **experiment_name 已分开**（flat/rough/jump 三个目录）；PPO 超参三者共享（继承同一 `SequencePPORunnerCfg`）。本表 PPO 数值仍取自 Flat dump，如需可跑 Rough/Jump 各 3-iter 交叉核对。
4. **起跳能力**：legs_act Kp=60 是否够软，Jump 是否需要单独 PD / nominal 站姿。
5. **`WYW_BASE_HEIGHT_FLIGHT=0.60`**（from_hu 待实测）是否合理，换 fudan USD 后改 0.65。
6. **critic 46 维是否够用**：暂无高度扫描 / DR 特权项，粗糙地形或 sim2real 是否需要补。
7. **`right_rear2_link` 碰撞网格**警告的根因（USD 层面）。
