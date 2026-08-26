# Wheelbipe V14 —— RL 核心 MDP 文档

本文档整理轮腿机器人（wheelbipe）活跃任务家族 `wheelbipe_V14` 的强化学习 MDP 定义，
用于快速查阅动作、观测、奖励、终止、事件、命令等核心项，以及网络结构与训练超参。

范式：**非对称 Actor-Critic + 特权观测 + estimator/teacher-student**，用 PPO 家族
（PPO / DreamWaQ / HIMLoco / NP3O）在 Isaac Sim 大规模并行仿真里端到端学习关节控制。

- 控制频率：**50 Hz**（`sim.dt = 1/200`，`decimation = 4`）
- 关键文件：
  - 任务注册：`source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_V14/__init__.py`
  - V14 环境配置：`source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_V14/env_cfg.py`
  - 常量/缩放/裁剪：`source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_V14/cfg_utils.py`
  - V14 环境逻辑：`source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe_V14/env.py`
  - 基类环境（obs/action/reward/done 实现）：`source/agent_tasks/agent_tasks/direct/wheelbipe/wheelbipe25_v3/env.py`
  - PPO/agent 配置：`source/agent_tasks/agent_tasks/direct/wheelbipe/agents/rsl_rl_ppo_cfg.py`

> 说明：V14 的 `env.py` 继承自 V13 → wheelbipe25_v3，动作/观测/终止的底层实现多在 `wheelbipe25_v3/env.py`。

---

## 1. 动作空间 (Action) — 6 维

| 项 | 值 | 位置 |
|---|---|---|
| 动作维度 `action_space` | **6**（腿 4 + 轮 2） | `wheelbipe25_v3/env_cfg.py:183`；组装 `env.py:2613-2614` |
| 腿部动作 (4) | 4 个腿主动关节（`.*_front1_joint`, `.*_rear1_joint` 左右各一）的**目标位置** | `env.py:2774-2779, 2835` |
| 轮子动作 (2) | 左右 `.*_wheel_joint` 的**目标角速度**（速度控制） | `env.py:2837` |
| 腿动作缩放 `leg_action_scale` | 0.5，命令 = `0.5*a + default_joint_pos` | `wheelbipe25_v3/env_cfg.py:343`；`env.py:2779` |
| 轮子控制模式 `use_wheel_vel_control` | True（速度控制） | V14 `env_cfg.py:568` |
| 轮速缩放 `wheel_vel_action_scale` | 10.0（rad/s per unit） | `v3/env_cfg.py:409`；`env.py:2031, 2837` |
| 轮速上限 `max_wheel_vel` | 100.0 × 1.5 | `v3/env_cfg.py:410`；`env.py:2032` |
| 轮子最大力矩 `max_wheel_torque` | 20.0 N·m（V14 覆盖） | V14 `env_cfg.py:1087` |
| 动作延迟随机化 `act_delay_cfg` | leg/wheel 各 [1,3] 步，`use_act_delay=True` | V14 `env_cfg.py:520-528` |
| 低通动作滤波 | 关闭（`use_action_low_pass_filter=False`） | `v3/env_cfg.py:413` |

> 云台（gimbal）不属于 policy 动作：pitch 固定位置、yaw 用速度/heading-PD 单独控制（V14 `env.py:1055-1098`）。

---

## 2. 观测空间 (Observation)

观测在 `wheelbipe25_v3/env.py:_get_observations()`（行 3100+）手工拼接。
基础维度常量：`policy = 28`、`privileged/critic = 32`（`cfg_utils.py:25-26`）；V14 实际 critic = **71**（`env_cfg.py:792-793`）。

### Actor / policy 观测（base 28 维，顺序见 `env.py:3198-3214`）

| 分量 | 维度 | 说明/缩放 |
|---|---|---|
| command (vx, vy, wz) | 3 | 速度命令，scale 各 1.0 |
| height_cmd | 1 | 目标机身高度，scale 5.0 |
| root_ang_vel_b | 3 | 机身角速度，scale 0.5 |
| projected_gravity_b | 3 | 投影重力 |
| joint_pos (相对 default) | 6 | 4 腿 + 2 轮；V14 `mute_wheel_pos_obs=True` 轮位置置 0（`env.py:3143`） |
| joint_vel_leg | 4 | scale 0.1 |
| joint_vel_wheel | 2 | scale 0.1 |
| actions (上一步) | 6 | |
| **小计** | **28** | 与 `V14_BASE_POLICY_OBS_DIM=28` 一致 |
| + ctrl_mode_obs | +7 | V14 额外附加，使 `num_single_obs=35` |

`ctrl_mode_obs`（7 维）布局：`("normal","stair","slope","recover","jump","height_target","state_time")`
（`env_cfg.py:593-603`，实现 `wheelbipe25_v3/env.py:1769`；v2 gimbal 模式会改写这 7 维，V14 `env.py:1173`）。
`task_flag` 默认关闭（dim 0）。

### Critic / privileged 观测（V14 = 71 维，`env.py:3262-3295`）

= policy 前 28 维（不含 ctrl_mode） + 特权项：

| 额外分量 | 维度 |
|---|---|
| root_lin_vel_b（特权真值线速度） | 3 |
| obs_height | 1 |
| ctrl_mode_obs（critic_extra） | 7 |
| privileged_extra_obs | 39 |

`privileged_extra_obs`（39 维，`env.py:4667-4780`）组成：
joint_stiffness(6) + joint_damping(6) + joint_torque(6) + obs_delay_steps(4) + act_delay_steps(2)
+ wheel_body_lin_vel(2×3=6) + wheel_contact_state(2)【动态 32】
+ body_mass_scale(1) + body_material(2×3=6)【静态 7】。

### 观测历史 / 噪声 / 延迟

- 主任务：`use_frame_stack=False`, `num_obs_hist=1`, `num_privileged_obs_hist=1`（V14 `env_cfg.py:740-742`）——**无堆叠**。
- 观测延迟随机化：`obs_delay_cfg`（ang_vel/gravity/joint_pos/joint_vel 各 [1,4]），`obs_history_len=10`，`use_obs_delay=True`（`env_cfg.py:504-518`）。
- estimator 历史：DreamWaQ `POLICY_HIST=5`、HIM `POLICY_HIST=5`、NP3O `POLICY_HIST=10`（`cfg_utils.py:51-58`）。
- 观测裁剪/缩放：`V14_BASIC_OBS_CLIP` / `V14_EXTRA_OBS_SCALE`（`cfg_utils.py:75-149`），`obs_input_scale_enabled=True`。
- 观测噪声 `self_obs_noise_cfg`（`env_cfg.py:532-541`）：ang_vel ±0.25、gravity ±0.05、joint_pos ±0.025、leg_joint_vel ±0.5、wheel_joint_vel ±1.0。

---

## 3. 奖励项 (Rewards)

权重来自 `WheelbipeV14FlatEnvCfg.rewards` OrderedDict（`env_cfg.py:869-914`）；
奖励函数实现于 `wheelbipe25_v3/env.py:_get_rewards()`（行 3474+）。

| Reward term | 权重 | 类型/说明 |
|---|---|---|
| termination | **-200.0** | 终止惩罚 |
| track_lin_vel_xy | +1.0 | 线速度跟踪 exp，sigma 0.5 |
| track_lin_vel_xy_square | -1.0 | 线速度误差平方 |
| track_ang_vel_z | +1.0 | 角速度跟踪 exp，sigma 0.25 |
| track_ang_vel_z_square | -1.0 | 角速度误差平方 |
| track_height_exp_tight | +1.0 | 高度跟踪 exp（紧），sigma 0.025 |
| track_height_square | -1.0 | 高度误差平方 |
| track_height_exp / _soft / _both_wheels_contact | 0.0 | 关闭 |
| flat_orientation_y_exp | +1.0 | 俯仰水平 exp，sigma 0.02 |
| flat_orientation_x_exp | +1.0 | 横滚水平 exp，sigma 0.01 |
| flat_orientation_y_v / flat_orientation_x_v | -2.0 / -2.0 | 姿态角速度惩罚 |
| flat_orientation_y / flat_orientation_x | 0.0 | 关闭 |
| lin_vel_z | -0.5 | 垂直速度惩罚 |
| ang_vel_xy | -0.05 | roll/pitch 角速度惩罚 |
| joint_torque | -1e-4 | 关节力矩平方 |
| leg_joint_acc | -5e-7 | 腿关节加速度 |
| leg_joint_vel | -5e-3 | 腿关节速度 |
| leg_joint_pair_pos_diff | 0.0 | 左右腿对称 |
| wheel_acc | -1e-8 | 轮加速度 |
| wheel_vel | -1e-5 | 轮速度 |
| wheel_power | -1e-4 | 轮功率 |
| wheel_air_spin | 0.0 | 空转 |
| action_rate | -0.01 | 动作变化率 |
| action_smoothness_leg | -0.05 | 腿动作二阶差分 |
| action_smoothness_wheel | -0.01 | 轮动作二阶差分 |
| stand_still_lin_vel | -1.0 | 站立时线速度惩罚 |
| stand_still | 0.0 | 关闭 |
| no_fork | -1.0 | 腿"叉开"惩罚 |
| no_fork_square | -1.0 | 平方版 |
| no_fork_exp / no_fork_z_exp | 0.0 | 关闭 |
| undesired_contact | -2.0 | 非轮部位接触惩罚 |

变体额外奖励：
- v2 gimbal 自旋平移：`gimbal_spin_track_lin_heading=5.0` 等（`env_cfg.py:1311-1319`）
- v1 腾空落地：`airborne_air_wheel_zero_torque_exp=20.0`, `airborne_precontact_wheel_directional_speed=10.0` 等（`env_cfg.py:1455-1666`）
- Rough：`V14_ROUGH_NP3O_BARLOW_PLUS_PRIV_REWARDS`（`cfg_utils.py:770-814`，如 `track_lin_vel_xy=5.0`, `termination=-500`）

---

## 4. 终止条件 (Terminations)

实现于 `wheelbipe25_v3/env.py:_get_dones()`（行 4911-5028）。

| 条件 | 阈值 | 位置 |
|---|---|---|
| 超时 time_out | `episode_length_s=20.0`s → 1000 步（dt=0.02） | `v3/env_cfg.py:177`；`env.py:4915` |
| 非法接触倒下 | reset 链接（base/gimbal/guide 等）接触力 > 1.0 N | `env.py:4913-4916, 134-145` |
| 姿态倒下 roll/pitch | `termination_roll_deg=40`, `termination_pitch_deg=40` | `v3/env_cfg.py:178-179`；`env.py:4994-5003` |
| 数值安全（立即） | joint_vel>500, ang_vel>200, lin_vel>100, 或 NaN/Inf | `env.py:4934-4964` |
| 观测异常（立即） | raw obs 超阈值/非有限 | `env.py:4971-4992` |
| 朝向偏差终止（默认关） | `reset_heading_target_terminate_enabled=False` | V14 `env_cfg.py:591` |
| 地形边界（记为 time_out） | margin 0.5 | V14 `env.py:541-572`（Rough 启用） |
| 连续终止 `termination_duration` | `enabled=True`, `steps=20`（数值安全类不受此限） | V14 `env_cfg.py:589-590`；`env.py:4863-4895` |

---

## 5. 事件 / 域随机化 (Events)

`EventCfgV14`（`env_cfg.py:34-354`，继承基类 `EventCfg` `v3/env_cfg.py:36-170`）；
函数实现在 `manager/mdp/isaaclab/events.py`。

**startup（质量/惯量/COM/摩擦）**
- add_base_mass：base 质量 scale (0.9, 1.3)
- add_leg_mass：各腿 + 云台 scale (0.9, 1.1)
- add_wheel_mass：轮 scale (0.9, 1.1)
- base_com：质心偏移 x±0.04, y±0.02, z±0.02
- 材质摩擦：base/guide/轮（轮 static 0.5-1.2, dynamic 0.4-1.0, restitution 0.02-0.2）
- 关节摩擦：leg_front/rear、wheel、legs_inact、gimbal
- base_inertia / wheels_inertia：定义后置 `None`（关闭）

**reset**
- robot_joint_stiffness_and_damping：刚度/阻尼 scale (0.75, 1.25)
- spring_damping：弹簧 scale
- leg_effort_noise / wheel_effort_noise：输出力矩扰动 scale (0.8-1.1)/(0.9-1.1)
- reset_base：roll/pitch ±0.15, yaw ±π（`reset_root_state_uniform_vel_b`）
- 预定义重置：`predefined_reset_ground`（`cfg_utils.py:198-226`）；v1 有 `predefined_reset_air`（`env_cfg.py:1488-1521`）

**interval（推力/外力，继承基类）**
- push_robot：每 5-10s，速度扰动 x/y ±0.25（`v3/env_cfg.py:142-150`）
- base_external_force_torque_xyz：每 5-10s，力 ±10 N 各轴、扭矩 ±1 N·m（`v3/env_cfg.py:161-170`）

---

## 6. 命令 (Commands)

V14 使用 `SpecialModeUniformVelocityCommandCfg`（`env_cfg.py:975-1045`；实现 `manager/mdp/isaaclab/commands.py:192,608`）。
命令向量本身 3 维 (vx, vy, wz)，机身高度为独立命令。

| 分量 | 采样范围 | 说明 |
|---|---|---|
| lin_vel_x | (-2.7, 2.7) m/s | `env_cfg.py:989` |
| lin_vel_y | (0.0, 0.0) | 固定 0 |
| ang_vel_z | (-2π, 2π) rad/s | `env_cfg.py:991` |
| heading | (-π, π) | heading_command=True, stiffness=5.0 |
| base height (height_cmd) | [0.20, 0.42]，默认 0.22 | `env_cfg.py:502, 564` |
| 重采样时间 | (5.0, 15.0) s | `env_cfg.py:977` |
| 站立环境比例 rel_standing_envs | 0.1 | |
| heading 环境比例 rel_heading_envs | 0.5 | |

特殊模式（按训练迭代门控启用）：
- `spin_low`（ang_vel 2π~3.25π，iter≥3000）
- `spin_mid`（3.25π~4.5π，iter≥4000）
- `dash`（lin_vel ±2~3，iter≥2000）
- `gimbal_spin_translate`（v2 增加，小陀螺自旋+平移）

命令随地形的空间型变化由 `manager/mdp/terrain/terrain_command_manager.py` 的 `TerrainCommandManager`
按机器人所踩瓦片覆盖（如楼梯地形强制特定 lin_vel_x、`disable_special_mode=True`）。

---

## 7. 特殊姿态 / 特殊动作

框架：可组合的运行时**状态机栈** `WheelbipeStateMachineManager`（`state_machines/manager.py:26`），
`from_env()` 按各 cfg 的 `enabled` 开关把 Airborne / JumpTakeoff / StepUp / Stair 依次入栈。
抽象层在 `manager/mdp/state_machine/{base,manager}.py`，机器人专用状态机在
`direct/wheelbipe/state_machines/`。状态机通过 hook 修改
[奖励缩放 / 高度目标&参考 / 有效命令 / 终止 mask] 来塑造 policy 行为。

**关键原则：没有开环脚本硬切姿态，最终动作始终由 policy 输出。**
状态机栈（airborne/jump/step_up/stair）主要靠改奖励和高度目标引导、仅少量改命令/终止；
小陀螺则由 command generator 的 special_mode 直接强改命令 + 观测 + 奖励。

| 特殊姿态 | 触发条件 | 交互方式 | 实现 |
|---|---|---|---|
| **腾空 airborne** | 机身高度 > 阈值 且 两轮心离地 且 地形允许、非楼梯（可要求持续时长）；或被 jump 交棒强制进入 | 改高度奖励参考(轮位置构造) + 抬高目标 + 落地二次曲线轨迹参考 + 大量空中专属奖励(收腿/零轮力矩/软着陆/防滑) | `state_machines/airborne.py`（进入 `:740-789`，退出 `:865-881`，奖励 `:983-1292`） |
| **跳跃 jump** | idle + request(flag/随机概率) + 稳定条件 + cooldown 结束 + 地形允许 + jump permission | 三相 IDLE→PUSH→TUCK 弹道；由 peak_height 反解 push 加速度/release 速度/tuck 时机；**保留 XY/height 命令不变，靠奖励引导**；训练期可施加可衰减外力/写 root 速度辅助起跳；退出交棒 airborne | `state_machines/jump_takeoff.py`（相位 `:32`，弹道 `:213`，触发 `:813`，assist `:342-454`） |
| **上台阶 step_up** | 轮前射线扫描时间差分高度 ∈ [step_min, step_max]（> wall_height 判墙） | 把 height_cmd 加 bias 并 hold；判墙则把该 episode 转 time_out 而非 terminate | `state_machines/step_up.py`（检测 `:52-76`，有效高度 `:28`，墙 `:99`） |
| **上楼梯 stair** | 近/远轮前扫描检测台阶且地形允许 | 采样目标 height_cmd(≈0.37~0.40) 覆盖有效高度；完整"进入→成功/失败/超时"状态；成功/失败/进展/快速成功等奖励；与 airborne 互斥 | `state_machines/stair.py`（进入 `:181-205`，成功 `:260-278`，失败 `:280-298`，奖励 `:328`） |
| **小陀螺 gimbal_spin_translate** | command generator 的 special_mode 激活（`env.py:929`） | **改命令 + 改观测 + 改奖励三者都做**：整机绕 z 高速自转，云台 yaw 靠 PD 保持朝向不动；在"云台系"采平移速度→投影回机身系写入 `command[:,0:2]`；抑制常规 track_lin_vel 奖励，改用云台系跟踪奖励 | `wheelbipe_V14/env.py`（采样 `:818`，应用命令 `:957`，观测 `:1173`，奖励 `:988`，云台 PD `:906`；配置 `env_cfg.py:1203`） |
| **蹲下/站立/调高** | 连续 height_cmd（命令第 3 维，范围 0.20~0.42） | 无独立姿态状态机；靠高度跟踪奖励 + 站立死区(`stand_still_deadzone` + `stand_still_lin_vel` 惩罚)实现连续蹲/站/静止 | `wheelbipe25_v3/env.py:_get_effective_height_cmd :1251` |

状态机在环境主循环的挂接点（`wheelbipe25_v3/env.py`）：
`on_observation_step :808`、`get_height_reward_reference_height :906`、`get_effective_height_cmd :1255`、
`get_height_reward_target_height :1737`、`apply_reward_term_scales :4067`、`apply_done_masks :5014`、
`on_command_updated/apply_command_overrides :5642/:5645`、`on_reset :5124`、构建栈 `_build_state_machine_manager :1914`。

交棒/互斥关系：jump 退出可置 `airborne_force_enter_request` → airborne 接管；stair 激活会清除 airborne 状态（二者互斥）。
jump 的物理辅助力是可衰减的训练脚手架，最终策略不依赖它。

另有两类"纯自旋/冲刺"velocity special_mode（不带云台平移投影）：`spin_low`/`spin_mid`（高 ang_vel_z）、`dash`（高速前冲），见第 6 节命令表。

### 7.1 触发机制：是"命令(cmd)"触发的吗？

**结论：分两类，大部分特殊姿态不是由速度命令向量 `command[:, :3]` 触发的。**

| 姿态 | 触发来源 | 是否 cmd 触发 |
|---|---|---|
| **腾空 airborne** | **物理检测**：机身相对地面高度 + 两轮离地间隙超阈值（`airborne.py:740-789`）；或 jump 退出时 `airborne_force_enter_request` 强制进入 | 否（被动感知触发） |
| **上台阶 step_up** | **前向射线扫描检测**：轮前方地面高度差 ∈ [step_min, step_max]（`step_up.py:52-76`） | 否（地形感知触发） |
| **上楼梯 stair** | **前向射线扫描检测**：近/远轮前检测到台阶且地形允许（`stair.py:181-205`） | 否（地形感知触发） |
| **跳跃 jump** | **内部 request 标志**（不是速度命令）：`jump_takeoff_request`。trigger `mode` 可为 `flag`（外部置位）/`random`（按 `probability_per_step` 或 `rate_per_s` 随机置位）/`manual`（`jump_takeoff.py:771-784`） | 否（独立 request 系统） |
| **小陀螺 gimbal_spin_translate** | **命令生成器的 special_mode**：由 command generator 采样 `special_mode_id` 决定哪些 env 进入该模式，按训练迭代窗口门控（`env.py:706-716, 929-933`） | **是（命令 special_mode 驱动）** |
| **纯自旋/冲刺 spin_low/spin_mid/dash** | 同上，命令 special_mode（只改 ang_vel_z / lin_vel_x 命令范围） | **是（命令 special_mode 驱动）** |
| **蹲下/站立/调高** | 连续 height_cmd（命令第 3 维）+ special height wave | 是（属于命令本身） |

要点：
- **airborne / step_up / stair 是"环境反应型"**——靠机身/轮子的物理状态与地形扫描检测触发，与用户下发的速度命令无关。
- **jump 有一套独立于 special_mode 的触发系统**：`request 标志 → permission 许可层 → 稳定性/冷却/地形门控`（见 7.2），不占用命令 special_mode 桶。jump 明确**保留当前 XY/yaw/height 命令不变**，只靠奖励引导起跳。
- **只有 spin/dash/gimbal_spin_translate 这类才是真正"由命令 special_mode 触发"**，因为它们本质就是命令层面的模式切换。

### 7.2 跳跃 (jump_takeoff) 完整细节

实现：`state_machines/jump_takeoff.py`（`JumpTakeoffStateMachine`）。这是一个**弹道式起跳辅助状态机**，
显式保留 XY/yaw 速度命令与 height_cmd 不变（类注释 `:22-28`），起跳完全靠弹道参考 + 奖励引导，policy 自己学关节动作。

**三相状态机**（`PHASE_IDLE=0 / PHASE_PUSH=1 / PHASE_TUCK=2`，`:32-34`）：

1. **触发进入 PUSH**（`on_command_updated`, `:743-822`）。`enter_mask` 需同时满足（`:813-819`）：
   - `idle`（当前空闲）
   - `request_mask`：来自 `jump_takeoff_request` 标志（flag）或随机置位（random 模式）
   - `stable_mask`：`episode_length ≥ min_episode_time_s`；若 `require_tracking=True`，还要求当前高度误差 ≤ `height_error_max` 且线速度跟踪误差 ≤ `lin_vel_error_max`（`:797-811`）
   - `cooldown ≤ 0`（距上次跳跃已过冷却，`:735`）
   - 地形允许（`allowed_terrain_names` / `not_allowed_terrain_names`, `:79-90`）
   - **许可层 mask**（若 `jump_takeoff_permission_cfg.enabled`）与 `~disabled_mask`（`:786-795`）

2. **弹道参考解算**（`_sample_ballistic_peak` + `_get_ballistic_reference`, `:492-572, 213-267`）：
   - 采样 `peak_height`（峰高）：支持 curriculum（随迭代 `iteration_start→end` 线性推进范围，`:497-521`）和分 bin 概率采样（`peak_height_bins`, `:456-490`）；默认范围 (0.50, 0.70)。
   - 由 peak_height 反解：`release_vel_z`（离地速度）、`push_time`（蹬伸时长）、`push_accel`（蹬伸加速度）、`peak_time`、`duration`。两种时序模式：`dynamic_push_time`（默认，按 push_distance/release_vel 算）或 `fixed_tuck_time`（固定收腿时刻反解）。
   - 生成分段参考轨迹：PUSH 段 `h = push_start + ½·a·t²`、`v = a·t`；飞行段 `h = release_h + v₀t − ½gt²`、`v = v₀ − gt`（`:254-267`）。

3. **PUSH → TUCK 切换**（`:686-722`）：满足任一即收腿——
   - 机身相对高度 ≥ `tuck_start_height`（默认=release_height），或轮已离地（`_get_wheel_air_mask`），或（可选）达到 `tuck_start_height_ratio·peak`（自然触发）
   - 相位时间 ≥ `tuck_start_time = push_time + offset`（时间触发）

4. **退出（回到 IDLE）**（`:724-738`）：相位时间 ≥ `duration`。退出时：
   - 若 `exit.enter_airborne_on_exit=True` → 置 `airborne_force_enter_request` **交棒给 airborne 状态机**接管落地
   - 设置 `cooldown_time = trigger.cooldown_s`，清 request/assist

**训练辅助（curriculum assist，`:342-454, 675-684`）**——可衰减的脚手架，最终策略不依赖：
- `type="force"`：起跳时对 base_link 施加竖直外力 `set_external_force_and_torque`（力 = `force_z + gain·(ref_release_vel − 当前vz)`，可衰减）
- `type="velocity"`：直接写 root z 速度 `write_root_velocity_to_sim`（在 push_start 或 tuck_start 时刻，`set` 或 `set_if_lower` 模式）
- 概率/力度按训练迭代 `probability_start→end`、`force_scale_start→end` 线性衰减到 0（`:132-166`）

**奖励塑造**（`apply_reward_term_scales`, `:824-1044`）：按相位（active/push/tuck 及 pre_release/post_release/push_accel/rise_to_tuck 窗口）缩放已有奖励，并叠加多种 addition 项，如：
`takeoff_vel_z`（起跳竖直速度）、`upward_accel_z`（上冲加速度）、`push_max_vel_z`、`push_release_vel_z_threshold`（达到/超过 release 速度奖励）、`release_vel_z_shortfall`（速度不足惩罚）、`release/trajectory_vel_z_tracking_exp`（速度轨迹跟踪）、`peak_height_tracking_exp`（峰高跟踪，退出时结算）、`push/trajectory_height_tracking_exp`（高度轨迹跟踪）、`leg_retraction`/`leg_length_tracking_exp`/`wheel_height_below_base_exp`（收腿）、`wheel_airtime` 等（类型分派见 `:864-1037`）。

**可视化**：PUSH/TUCK 相位各对应一个 marker 小球（`apply_visual_marker_state`, `:1094-1119`）。

---

## 8. 网络结构

非对称 Actor-Critic：actor 只用本体（非特权）观测（+历史/估计量），critic 用特权观测。

| 算法 | actor 输入 | critic 输入 | 隐藏层 | 备注 |
|---|---|---|---|---|
| PPO（默认） | 35（28 本体 + 7 ctrl_mode），无堆叠 | ≈71 特权 | [256,128,64] | ELU，init_noise_std=1.0 |
| DreamWaQ | policy(28) + CENet code(~16) | 32 特权 | enc[256,128,64]/dec[64,128,256] | CENet=VAE，从 policy_hist(28×5) 编码 latent+速度 |
| HIMLoco | policy + estimator(vel + 16维latent) | 特权 | [512,256,128] | estimator 从 policy_hist(28×5) 前向(no_grad) |
| NP3O | BarlowTwins：scan_encoder + history_encoder(28×10) + priv_encoder | 内部切 prop/scan/priv 三段 | [512,256,128], scan[128,64,32] | 额外 cost 网络(MLP+Softplus, 5维)，EmpiricalNormalization |

---

## 9. 训练超参 (PPO / RSL-RL，默认 Flat-v0)

配置：`agents/rsl_rl_ppo_cfg.py`。

| 超参 | 值 | 超参 | 值 |
|---|---|---|---|
| num_steps_per_env | 24 | learning_rate | 1e-4（adaptive schedule） |
| max_iterations | 20000 | gamma | 0.99 |
| num_envs（训练 CLI） | 4096（cfg 默认 32） | lam (GAE) | 0.95 |
| num_learning_epochs | 5 | desired_kl | 0.01 |
| num_mini_batches | 4 | clip_param | 0.2 |
| value_loss_coef | 4.0 | entropy_coef | 0.005 |
| use_clipped_value_loss | True | max_grad_norm | 1.0 |
| init_noise_std | 1.0 | 激活函数 | ELU |
| 网络 hidden | [256,128,64] | save_interval | 500 |
| sim.dt / decimation | (1/200) / 4 → **50 Hz** | empirical_normalization | False |
| experiment_name | wheelbipe_v14_2_flat_direct | | |

各算法变体 max_iterations：PPO 20000 / DreamWaQ 10000 / HIM 5000 / NP3O 3000。

NP3O 约束相关：num_costs=5，cost_value_loss_coef=0.1，cost_viol_loss_coef=0.1，dagger_update_freq=20。
