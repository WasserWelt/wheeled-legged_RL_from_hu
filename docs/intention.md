# 计划：向 wheeled-legged_RL_from_hu 新增 fudan 风格的 "wyw" Task 族

> **状态：已实现并通过冒烟验证（2026-08-27）。** 下方"实现结果（as-built）"记录最终落地
> 架构与两处关键偏离原计划的决策；其后的原始计划正文保留作设计记录，但凡与"as-built"冲突处
> **以 as-built 为准**。

## 实现结果（as-built）

最终落地采用**「扩展点」策略**，而非原计划的"整体重写三个 MDP 方法"。核心权衡：
`WheelbipeV14Env._get_observations` 内含大量喂给下一步 reward 的**有状态副作用**
（obs 延迟/带噪副本、命令刷新、地面高度估计、prev-action 滚动、历史 deque），整体重写会
遗漏这些副作用。故改为"先 `super()` 触发全部副作用，再覆写观测/注入奖励"。

**落地文件（均带 SCUTRobotLab MIT 头、中文注释）：**
- `direct/wheelbipe/wyw/wyw_constants.py` —— 几何/目标/缩放常数宏，顶部 `WYW_ROBOT` 开关
  （`"from_hu"` | `"fudan"`），第一版 = `"from_hu"`。维度：actor **25** / critic **46** /
  encoder 输入 **125**（=5×25）/ latent **3**。
- `direct/wheelbipe/wyw/env.py` —— `WheelbipeWywEnv(WheelbipeV14Env)`，扩展点式：
  - `_get_observations`：`super()` 后覆写 `policy`(25)/`policy_hist`(125)/`critic`(46) 三键，
    并 `pop` 掉基类残留的 `prev_critic`/`critic_hist`。自维护 `_wyw_obs_hist`(N,5,25) 滚动缓冲。
    base_lin_vel×2.0 放 critic **前 3 维**供 encoder 监督。
  - `_postprocess_reward_terms`（**基类提供的钩子**）：`super()` 后，若 `cfg.wyw_jump_enabled`
    则 `update` 注入 6 个 fudan 涌现式跳跃项。**不重写** `_get_rewards`。
  - **不重写** `_get_dones`：直接复用基类（已含接触/倾倒/数值安全/超时终止）。
- `direct/wheelbipe/wyw/env_cfg.py` —— Flat/Rough/Jump（+ 各 Play）配置类。
- `direct/wheelbipe/wyw/__init__.py` —— 6 个 `gym.register`。
- `agent_rl/.../rsl_rl/{modules/actor_critic_sequence.py, algorithms/ppo_sequence.py,
  runners/on_policy_sequence_runner.py, storage/rollout_storage_sequence.py}` + 各 `__init__` 导出。
- 编辑：`agent_tasks/__init__.py`（family import）、`agents/rsl_rl_ppo_cfg.py`
  （`SequencePPORunnerCfg` + `WheelbipeWywPPORunnerCfg`）。

**两处关键偏离原计划：**

1. **`observation_space`/`state_space` 必须是 int，不能是 dict。** 原计划设想传 dict 观测布局，
   但 stock `DirectRLEnv._configure_gym_env_spaces`（`isaaclab/envs/direct_rl_env.py`）把
   `observation_space` 原样交给 `spec_to_gym_space`——**传 dict 会被整体嵌套进
   `single_observation_space["policy"]`**（各子键 flatdim 相加 = 196），从而丢失 `policy_hist`
   顶层键，导致 encoder 输入维度算错（曾报
   `mat1 and mat2 shapes cannot be multiplied (64x125 and 980x128)`，980=5×196）。
   **修复**：`observation_space=25`(int)、`state_space=46`(int)、`num_obs_hist=5`。于是
   `single_observation_space={policy:Box(25), critic:Box(46)}` →
   自定义 wrapper `num_observations={policy:25, critic:46}`、`num_privileged_obs=46` →
   runner 对 `policy_hist` **回落**到 `num_obs_hist*num_obs = 5*25 = 125`（=encoder 输入）。
   源码树中**没有**任何对 `_configure_gym_env_spaces` 的覆盖。

2. **critic = 46，不是 ~141。** 原计划担心 from_hu 高度扫描网格维度未知，估算 critic≈141。
   最终 critic 观测**不含**高度扫描/域随机化特权参数，只拼：base_lin_vel×2.0(3, encoder 监督) +
   ang_vel×0.25(3) + proj_gravity(3) + cmd(3) + 腿关节偏差(4) + dof_vel×0.05(6) + actions(6) +
   before_prev_actions(6) + joint_acc×0.0025(6) + torque×0.05(6) = **46**。

**奖励注入无污染的机制**：基类 `_get_rewards` 用 `locals()` 反射收集 `rew_*` → 调
`_postprocess_reward_terms` 钩子 → ×step_dt → **只对 `cfg.rewards` OrderedDict 中列出的键求和**
（缺失键 `.get` 默认 0）。故 Flat/Rough 的 `cfg.rewards` 不列 jump 项 → 即便注入也自动丢弃；
Jump 的 `WheelbipeWywJumpEnvCfg.__post_init__` 把 `WYW_JUMP_REWARD_WEIGHTS` 合并进 `rewards`
→ 6 项生效。冒烟验证已确认 6 项仅在 Jump 任务出现、Flat 中计数为 0。

**冒烟验证结果**：`list_envs.py` 列出全部 6 个 wyw 任务；Flat/Rough/Jump 各 3 iter 训练成功，
encoder 速度估计 loss 均下降（Flat 0.47→0.15、Rough 0.19→0.06、Jump 0.47→0.16），无 shape 报错。

## 背景

在 `wheeled-legged_RL_from_hu`(Isaac Sim + Isaac Lab **DirectRLEnv** + RSL-RL)中
新增一套 RL Task。其**设计**(地形方案、Actor/Critic 架构、观测、奖励、终止条件,尤其
是**涌现式 jump** 行为)移植自 fudan 的 IsaacGym 仓库 `fudan_rl_wheel_leg`
(legged_gym + 内置 rsl_rl)。**机器人**沿用 from_hu 原仓库的机器人
(`Wheelbipe_V14_2_CFG`):6 维动作 = 4 个腿关节位置目标 + 2 个轮速度目标,另有被动弹簧
连杆与 2 自由度云台,云台**不**进入动作向量。由于机器人不同,即使照搬设计意图,动作/
观测的**细节维度**也会与 fudan 不一样。

fudan 并没有注册独立的 rough/jump 任务,而是在同一个 env 上按次修改配置来跑不同"任务"。
它实际训练了**三种形态**:Flat(平地 plane)、Rough(trimesh 斜坡/粗糙/台阶,课程开启)、
Jump(平地;跳跃完全靠奖励塑形 + 基于接触力的腾空检测**学出来**,而**不是**靠地形,也
**不是**靠显式相位状态机)。

### 已确认决策(与用户确认)
- **Jump = 忠实移植 fudan 涌现式**奖励;**不启用** from_hu 里现成但休眠的
  `JumpTakeoffStateMachine`(弹道/分相位)——行为必须由 `root_vel_z` + 轮接触腾空检测
  相关的奖励涌现出来。
- **Actor/Critic = 移植 fudan `ActorCriticSequence`**(历史 encoder → 3 维 latent,作为
  隐式基座线速度估计器监督训练,latent 对 actor 做 detach)。三个 Task 共享。
- **观测(fudan 风格 25 actor / ~141 critic)**、**PPO 超参(fudan)** 与网络,三个 Task
  **共享**。
- **控制频率对齐 fudan = 100Hz**:wyw 的 `sim dt=1/200`、`decimation=2`(from_hu V14 默认是
  decimation 4 = 50Hz),使 policy dt = 0.01s 与 fudan 一致。
- **未来切换到 fudan 机器人模型**:fudan 与 hu 连杆构型一致、仅杆长不同;后续会把机器人 USD 换成
  fudan 模型使杆长一致。因此**所有几何常数/奖励目标用宏(具名常量)集中写死**,换模型时只改一处。
- **第一版训练在 from_hu USD 上跑(方案 A)**:fudan USD 仍在修改中。故 `wyw_constants.py` 初值
  按 from_hu 机器人填,`WYW_ROBOT = "from_hu"`;fudan USD 就绪后切一组常量(或 `WYW_ROBOT="fudan"`)
  即可。所幸 from_hu 的 L0 量程 `[0.13,0.32]` 与 fudan 操作区间 `[0.16,0.31]` 近乎重合(同构型),
  `L0_TUCK=0.16`/`L0_EXTEND=0.31` 在 from_hu 上直接可用;**过渡期主要需调 apex 高度**(fudan 站高
  ≈0.18→apex 0.65;from_hu 站高≈0.35,量纲不同 → 初值 `BASE_HEIGHT_FLIGHT≈0.60` 待实测微调)。
- **注册三个 Task**,名称含 `wyw` + 形态 + `v1`:
  - `Robotics-Wheelbipe-V14-wyw-Flat-v1`  → 平地 `plane`
  - `Robotics-Wheelbipe-V14-wyw-Rough-v1` → `trimesh`,课程开启,
    `terrain_proportions = [0.2, 0.2, 0.2, 0.1, 0.2, 0.1]`
  - `Robotics-Wheelbipe-V14-wyw-Jump-v1`  → 平地 `plane` + 涌现式 jump 奖励
  - 按仓库惯例补充对应的 `-Play-v1` 变体。
- Flat/Rough 使用 fudan **运动(locomotion)** 奖励;Jump = 运动奖励 **+** 额外的 jump 奖励。

### 关于跳跃的两个澄清(训练涌现 vs 真机切换)
理解这两点才能保证 wyw-Jump 训出来的策略是"能用"的:

1. **训练侧:跳跃是"涌现"的,不是命令触发的。** fudan 的 jump 网络输入命令只有
   `[vx, yaw, height]`,**没有**"现在跳"这一位;那套跳跃奖励(`base_height_flight` /
   `leg_tuck` / `takeoff_extend` / `encourage_jump` …)常开,只被腾空检测门控,于是网络被训成
   **"只要能跳就反复跳"**。→ wyw-Jump 同样不引入跳跃命令位,靠奖励涌现。
2. **真机侧:跳跃是靠"切换到独立的 jump policy"打开的。** fudan 真机上 stable/spin/upstairs/
   jump 是 **4 个并列的独立 ONNX + 各自参数表**,由 `ChassisPolicy::Jump` 整体切换(换模型、
   换 PD 增益、换 default_dof_pos、清空 125 维历史)。→ 部署时"进入跳跃模式"= 切到 wyw-Jump
   策略,而不是在通用策略里加动作。
3. **行进间跳跃 = 同一个 jump policy + 持续送速度命令。** 不需要单独的"行进跳"策略。
   fudan 的 jump 任务在训练时**同时打开运动跟踪奖励**(`tracking_lin_vel` /
   `tracking_lin_vel_enhance` / `tracking_ang_vel`)且随机采样 `vx∈[-2.1,2.1]`、
   `yaw∈[-2,2]`,所以网络学会"边按 vx 前进边跳"。真机给 `vx>0` 就边走边跳,`vx=0` 就原地跳。
   → **wyw-Jump 必须保留运动跟踪奖励 + 同样的随机速度命令范围**,否则只会原地跳、不具备
   行进间跳跃能力。

## 总体架构

复用现有的继承与注册机制,新增一层 `wyw` 配置和一套新的 RL module/algorithm/runner,
尽量不去 fork 那个 5700 行的基类 env。

- **Env 类**:新增子类 `WheelbipeWywEnv(WheelbipeV14Env)`。**继承只为复用机械管线**:
  场景/传感器/动作下发/**云台保持**(fudan 机器人也带云台且不进动作,这段必须保留)。
  **fudan 的"设计"以整体覆盖三个 MDP 方法的方式注入**(见下"代码组织")。
- **Env 配置**:新建 `wyw/env_cfg.py`,包含 `WheelbipeWywFlatEnvCfg`、`...RoughEnvCfg`、
  `...JumpEnvCfg`(+ Play),继承现有 V14 flat 配置并覆盖地形、`rewards` OrderedDict、
  观测布局与网络挂接。
- **RL**:在 `agent_rl` 新增 `ActorCriticSequence` module + `PPOSequence` algorithm +
  runner,通过 `class_name`/`runner_class` 字符串选择。**不采用 DreamWaQ/HIM 的算法**,仅把它们
  作为"本仓库如何组织一个自定义 (module + algorithm + runner) 三元组"的**接线范式**参考
  (CLAUDE.md 已说明每个算法都是这样的三元组)。

### 代码组织(避免 fudan 设计与 from_hu 机制交叉,防止"两者乱")
基类 `Wheelbipe25v3Env`(5700 行)含大量 fudan 没有的东西:云台、小陀螺 spin/translate、
多个状态机(airborne / jump_takeoff / stair)、多种 height 模式、`ctrl_mode_obs`,以及靠
`locals()` 反射 + `cfg.rewards` OrderedDict 收集的几十个奖励项。若"深继承 + 逐字段禁用",
两处最易乱:(a) 奖励——某个 from_hu 遗留项还留在 OrderedDict 里就会静默生效,污染 fudan 涌现式
奖励且难以发现;(b) 观测——用配置驱动的 block 机制去凑 fudan 精确的 25/141 会互相打架。

**〔as-built 修订〕** 原计划设想"整体重写三个 MDP 方法"以隔离 fudan 与 from_hu 机制。实际落地
改用**扩展点策略**(见顶部"实现结果"),因为 `_get_observations` 内含大量喂给下一步 reward 的
有状态副作用,整体重写会丢失它们。最终:
- `_get_observations`:先 `super()` 触发全部副作用,**再覆写** `policy`(25)/`policy_hist`(125)/
  `critic`(46) 三键并 `pop` 掉残留 critic 键。观测污染由此避免——三键被显式覆写,`ctrl_mode_obs`
  等 block 不进入这三键。
- **不重写** `_get_rewards`:改用基类 `_postprocess_reward_terms` 钩子注入 jump 项;奖励污染由
  `cfg.rewards` OrderedDict 白名单机制天然避免(未列出的键 `.get` 默认 0,详见顶部)。
- **不重写** `_get_dones`:直接复用基类的接触/倾倒/数值安全/超时终止(见下第 4 步修订)。
- 其余(`_pre_physics_step`/`_apply_action`/云台/场景/传感器)沿用 V14 基类,不重写。

### 几何常数用宏(具名常量)集中定义
fudan 奖励里有多处几何量(如虚拟腿长 L0)。fudan 与 hu **连杆构型一致、仅杆长不同**,且后续会
把机器人 USD 换成 fudan 模型使杆长一致。故新建 `wyw/wyw_constants.py` 把**所有几何常数与奖励
目标写成具名常量(宏)**,换模型时只改这一处:
- 顶部一个开关 `WYW_ROBOT`(`"from_hu"` | `"fudan"`)选择整组常量。**第一版 = `"from_hu"`**。
- 杆长 `L1 / L2 / OFFSET`:from_hu = `links_length=[0.1134,0.135,0.210]`;fudan = `l1=0.175`、
  `l2=0.208`、`offset=0.0`。(L0 走 `wheel_pos_heading_b` 时用不到杆长,仅解析 FK 备用。)
- L0 目标:`L0_TUCK=0.16`、`L0_EXTEND=0.31`(两机器人量程近乎重合,共用);apex
  `BASE_HEIGHT_FLIGHT`:from_hu ≈`0.60`(待实测)/ fudan =`0.65`;门控阈值
  `TAKEOFF_VZ=0.15`、`FLIGHT_CONTACT_FORCE=1.0`、`FALL_CONTACT_FORCE=10.0` 等(与机器人无关,共用)。
- **L0 的计算**推荐直接复用 from_hu 现成的 `torch.norm(wheel_pos_heading_b, dim=-1)`——它从
  仿真实际连杆位姿读,**与杆长无关、自动正确**,换上 fudan USD 后即为 fudan 的 L0。若要与
  fudan **逐位一致**,可改用移植的解析 `forward_kinematics(θ1,θ2,L1,L2)`(用上面的杆长宏)。
- 因最终就是 fudan 机器人,**这些目标值(0.65/0.16/0.31)作为宏直接采用 fudan 值**;仅当在换
  USD 之前先跑当前 from_hu 机器人时,才临时按其腿长量程缩放这些宏(见"待确认风险")。

## 实施步骤

### 1. 将 `ActorCriticSequence` 移植进 `agent_rl`(模板:DreamWaQ / HIM)
参照 `source/agent_rl/agent_rl/rsl_rl/{modules,algorithms,runners}/` 中 DreamWaQ/HIM
的速度估计变体实现(把它们当结构模板阅读——它们已经实现了"encoder + 估计器 + 把估计
输出拼到 critic 观测、并喂历史观测的自定义 runner")。

新建:
- `source/agent_rl/agent_rl/rsl_rl/modules/actor_critic_sequence.py`
  - Encoder:`Linear(history_len*num_obs → 128 → 64 → latent_dim=3)`,ELU,`history_len=5`。
    latent 在拼入 actor 输入前**做 detach**。
  - Actor MLP `[128,64,32]`,输入 `num_obs + latent_dim`。
  - Critic MLP `[256,128,64]`,输入 `num_critic_obs + latent_dim`。
  - `init_noise_std=0.5`,ELU,可学习 std。
- `source/agent_rl/agent_rl/rsl_rl/algorithms/ppo_sequence.py`
  - 继承 PPO;新增 `extra_optimizer`(仅优化 encoder 参数,lr `1e-3`);每次更新额外加一项
    MSE loss:`latent[:, :3]` 与 `critic_obs[:, :3]`(= base_lin_vel × scale)——隐式基座
    线速度估计。PPO 主体:gamma .99、lam .95、clip .2、entropy .01、desired_kl .005
    (adaptive)、lr 1e-3、5 epochs、4 minibatches。
- `source/agent_rl/agent_rl/rsl_rl/runners/on_policy_sequence_runner.py`
  - 维护历史观测缓冲(长度 5),做编码,把 3 维 latent 拼到 critic 观测上,走
    `ppo_sequence`。历史与特权/critic 观测如何从 env wrapper
    (`agent_rl.rsl_rl.env.RslRlVecEnvWrapper`)取得,严格照 DreamWaQ/HIM 的 runner。
- 在各 `__init__` 中导出新类,使 `scripts/rsl_rl/train.py` 里基于字符串的 `eval(...)`
  能解析到。

### 2. 观测拼装(fudan 25 actor / **46** critic)〔as-built:critic=46,非 ~141〕
**〔as-built 关键〕`observation_space`/`state_space` 用 int(25/46),不能用 dict**——否则 stock
`_configure_gym_env_spaces` 会把 dict 嵌套进 `["policy"]` 丢掉 `policy_hist` 键(详见顶部"偏离
1")。观测在 `WheelbipeWywEnv._get_observations` 里按下面布局手工拼装,常数集中在
`wyw_constants.py`,与部署端逐位一致:

- **Actor 观测 = 25**:基座 `ang_vel`(3)、`projected_gravity`(3)、
  命令 `[vx, wz, height_cmd]`(3)、腿关节位置 − 默认(4 个腿关节)、
  `dof_vel`(6 = 4 腿 + 2 轮)、`prev_actions`(6)。**不含基座线速度**(由 encoder 估计)。
  fudan 忠实变体**不加** `ctrl_mode_obs` 块。
- **观测历史** 长度 5 → encoder 输入 `5×25 = 125`。〔as-built〕基类无此布局的历史缓冲,故在
  env 子类里**自维护** `_wyw_obs_hist`(N,5,25) 滚动写入;runner 侧 `policy_hist` 维度经 int
  space + wrapper 回落到 `num_obs_hist*num_obs=125`(不再需要把 `policy_hist` 作为 obs dict 键)。
- **Critic / 特权观测(as-built = 46,非 ~141)**:`root_lin_vel_b`×2.0(3,**必须是前 3 维**,
  encoder 监督目标) + `root_ang_vel_b`×0.25(3) + `projected_gravity_b`(3) + cmd(3) +
  腿关节偏差(4) + `dof_vel`×0.05(6) + `actions`(6) + `before_prev_actions`(6) +
  `joint_acc`×0.0025(6) + `applied_torque`×0.05(6) = **46**。
  〔说明〕原计划设想拼入地形高度扫描 + 域随机化特权参数(估算 ≈141),最终**未采用**——先跑通
  fudan 序列架构的最小可用特权集;高度扫描/DR 特权项作为后续可选增强。

### 3. 奖励〔as-built:不重写 `_get_rewards`,改用 `_postprocess_reward_terms` 钩子注入〕
**〔as-built 修订〕** 原计划设想整体重写 `_get_rewards`。最终**复用**基类的
`locals()` 反射 + `cfg.rewards` OrderedDict 白名单机制,因为该机制**天然无污染**:基类只对
`cfg.rewards` 中列出的键求和(缺失键默认 0)。故:
- Flat/Rough 直接沿用 V14 的 locomotion `rewards`(基类自动丢弃未列出的项)。
- Jump 项通过覆写 `_postprocess_reward_terms(reward_terms)` 钩子注入(`super()` 后
  `update` 6 项),并在 `WheelbipeWywJumpEnvCfg.__post_init__` 里把 `WYW_JUMP_REWARD_WEIGHTS`
  合并进 `self.rewards`——只有 Jump 任务的白名单含这些键,故仅 Jump 生效。
几何常数与目标从 `wyw_constants.py` 引用。下方 fudan 权重清单为**设计参照**;实际 Flat/Rough
运动权重沿用 from_hu V14 现成项。

- **运动(Flat/Rough/Jump)** —— fudan 权重:`tracking_lin_vel`(exp,σ0.25,权重 1.0)、
  `tracking_lin_vel_enhance`(1.0,负偏置的锐化整形)、`tracking_ang_vel`(1.0)、
  `orientation`(−25)、`ang_vel_xy`(−0.1)、`action_rate`(−0.04)、`torques`(−5e-5)、
  `collision`(−1)、`nominal_state`(左右腿对称,−1.0)、`pen_theta_no0`(腿竖直,−2.0)。
  映射到 from_hu 已有等价项(`rew_flat_orientation*`、`rew_ang_vel_xy`、
  `rew_undesired_contact`、腿角度项)。
- **数值目标 = fudan 值,写成 `wyw_constants.py` 宏**(`BASE_HEIGHT_FLIGHT=0.65`、
  `L0_TUCK=0.16`、`L0_EXTEND=0.31` …)。因最终采用 fudan 机器人模型,这些值直接成立。**仅当在
  换 fudan USD 之前先跑当前 from_hu 机器人时**(from_hu L0 量程 `[0.13,0.32]`、站高 0.35m、骑行高
  `[0.20,0.42]`),才需临时按其腿长量程缩放这些宏,换回 fudan USD 后改回 fudan 值即可。
- **仅 Jump**(`_compute_wyw_jump_terms`,均为 per-step 速率,后续统一 ×step_dt),复用
  `_get_root_quat_inv_and_wheel_pos_b`(取 L0 = `norm(wheel_pos_heading_b)`)与
  `_get_wheel_contact_force_peaks`(腾空/触地判定)。门控:`in_flight` = 所有轮峰值 <
  `FLIGHT_CONTACT_FORCE(1.0)`;`any_contact` = 任一轮峰值 > 1.0。〔as-built 权重见
  `WYW_JUMP_REWARD_WEIGHTS`〕
  - `base_height_flight`(6.0):`exp(−|root_z − BASE_HEIGHT_FLIGHT(0.60)|·6)` × in_flight。
  - `leg_tuck`(1.7):`exp(−(|L0ₗ−0.16|+|L0ᵣ−0.16|)·4)` × in_flight。
  - `takeoff_extend`(0.5):`exp(−(|L0ₗ−0.31|+|L0ᵣ−0.31|)·4)` × (any_contact & vz>0.15)。
  - `line_z`(6.0):`clamp(vz, min=0)` × in_flight。
  - `flight`(0.15):in_flight 固定 bonus。
  - `encourage_jump`(1.0):〔as-built 简化〕`clamp(root_z, 0, 0.5)` × in_flight
    (连续高度加权,非原计划的"触地结算式滞空累加器"——避免额外的跨步状态缓冲)。
  - **不要**设置 `jump_takeoff_state_machine_cfg` / 保持其禁用。

### 4. 终止〔as-built:不覆写 `_get_dones`,复用基类〕
**〔as-built 修订〕** 原计划设想覆写 `_get_dones` 实现 fudan `fail_buf`(1 秒宽限)。最终
**直接复用基类** `_get_dones`——from_hu V14 已含接触/倾倒/数值安全/超时终止(Rough 另含地形
边界复位),对 wyw locomotion + jump 已足够,无需引入独立的 fail_buf。若后续需要 fudan 式宽限
计数,再在子类补一个 `fail_buf` 覆写即可(接口已就位)。

### 5. 三个 env 配置类 + 地形
`source/agent_tasks/agent_tasks/direct/wheelbipe/wyw/env_cfg.py`:
- `WheelbipeWywFlatEnvCfg(WheelbipeV14FlatEnvCfg)`:地形 = plane;〔as-built〕类属性
  `ctrl_mode_obs_enabled=False`、`observation_space=25`(int)、`state_space=46`(int)、
  `num_obs_hist=5`、`wyw_jump_enabled=False`;`__post_init__` 末尾调 `_apply_wyw_common(self)`
  强制 decimation=2、int obs/critic 形状、命令范围。命令 `[vx, wz, height]`:速度区间用 fudan
  (vx ±2.1、wz ±2、lin_vel_y=0),关闭 spin/dash 特殊模式(`special_modes[*].rel_envs=0`)。
  〔待实测〕height 命令沿用 from_hu 骑行高度量纲(非 fudan 站高量纲)。
- `WheelbipeWywRoughEnvCfg(WheelbipeWywFlatEnvCfg)`:`__post_init__` 把地形换成 trimesh
  生成器,`terrain_proportions=[0.2,0.2,0.2,0.1,0.2,0.1]`,课程开启(参照现有
  `WheelbipeV14RoughEnvCfg` 的地形替换写法)。
- `WheelbipeWywJumpEnvCfg(WheelbipeWywFlatEnvCfg)`:plane 地形;在 `rewards` OrderedDict 上
  追加第 3 步的 jump 项。
- `*PlayEnvCfg` 变体(更少环境数,终止按惯例)。

### 6. PPO / runner 配置
在 `source/agent_tasks/agent_tasks/direct/wheelbipe/agents/rsl_rl_ppo_cfg.py` 新增:
`WheelbipeWywPPORunnerCfg`,采用 fudan 超参(`num_steps_per_env=48`;`max_iterations` 之后由
用户定;`init_noise_std=0.5`,actor `[128,64,32]`,critic `[256,128,64]`,ELU),并把
`class_name`/`runner_class` 字符串指向新增的
`ActorCriticSequence` / `PPOSequence` / `OnPolicySequenceRunner`。

### 7. 注册 + 接线
- 新建 family 目录 `source/agent_tasks/agent_tasks/direct/wheelbipe/wyw/`,其 `__init__.py`
  内放 `gym.register()`(3 个 Task + Play 变体),`env_cfg_entry_point` 指向新 cfg 类,
  `rsl_rl_cfg_entry_point` 指向 `WheelbipeWywPPORunnerCfg`。`entry_point` = `WheelbipeV14Env`
  (若确需子类则用 `WheelbipeWywEnv`)。
- 在 `source/agent_tasks/agent_tasks/__init__.py` 加上该 family 的 import,使注册在导入时触发。

### 8. 部署契约(sim2real,来自 fudan_WheelLeg_RLdeploy)
真机侧**不复现**奖励/腾空检测/终止逻辑——那些是训练产物;跳跃能力已"烧"进网络权重。真机只
复刻:观测构建 + 双输入网络推理 + 力矩映射。训练时必须锁死下列约束,否则 sim2real 会错位:

- **观测布局与缩放**必须与部署端逐位一致(gyro×0.25、投影重力×1.0、指令外部缩放
  `[vx,yaw,height]`、腿关节偏差×1.0、关节角速度×0.05、上步动作×1.0)。
- **双输入 ONNX 导出契约**:`forward(obs[25], obs_history[125]) → actions[6]`,即
  `latent=encoder(obs_history); action=actor(cat(obs, latent))`。导出脚本需按此签名(参考 from_hu
  `agent_rl` 现有的 `utils/exporter*.py` 与 fudan `export_onnx`)。真机 base_lin_vel 不测,靠 encoder 估。
- **动作→力矩**:腿=位置环 `pos_ref=action*0.5`、轮=速度环 `vel_ref=action*10`,与训练
  `action_scale`(腿 0.5 / 轮 10)必须一致。
- **控制频率(已定)**:对齐 fudan **100Hz**。wyw 设 `sim dt=1/200`、`decimation=2`
  (from_hu V14 默认 decimation 4=50Hz),policy dt=0.01s。真机执行层 500Hz、推理每 5 周期一次,
  与 fudan 一致。所有按 dt 缩放的奖励常数(`base_air_time += dt·…`)据此复核。
- **per-task 控制参数可不同**:fudan 的 jump 用了比 stable 更"软"的腿 PD(Kp 6 / Kd 0.5 vs 15 / 1.0)
  和不同的 nominal 关节位(0.2/0.4 vs −0.23/−0.65)才蹬得动。本项目机器人配置沿用 from_hu,但需注意
  **from_hu 腿刚度较高(legs_act stiffness 60),Jump 任务可能需要单独调软腿 PD / 换 nominal 站姿**才能起跳。

## 关键文件
- 新建:`.../direct/wheelbipe/wyw/{__init__.py, env_cfg.py, env.py, wyw_constants.py}`
  (`env.py` = `WheelbipeWywEnv`,〔as-built〕扩展点式:覆写 `_get_observations`(super 后覆写三键)
  + `_postprocess_reward_terms`(注入 jump 项),**不重写** `_get_rewards`/`_get_dones`;
  `wyw_constants.py` = 几何/目标/缩放常数宏)
- 新建:`source/agent_rl/agent_rl/rsl_rl/modules/actor_critic_sequence.py`、
  `.../algorithms/ppo_sequence.py`、`.../runners/on_policy_sequence_runner.py`、
  `.../storage/rollout_storage_sequence.py`
- 编辑:`agent_tasks/__init__.py`(family import)、
  `.../wheelbipe/agents/rsl_rl_ppo_cfg.py`(新增 RunnerCfg)、agent_rl 包 `__init__` 导出。
- 参考(只读):`.../wheelbipe25_v3/env.py`(观测/奖励/`_get_dones`/L0 + 接触辅助函数)、
  `.../wheelbipe_V14/env_cfg.py`(Rough 地形替换写法)、agent_rl 里的 DreamWaQ/HIM
  (移植模板)、fudan 的 `actor_critic_sequence.py` / `ppo.py` / `on_policy_runner.py`。

## 验证
1. ✅ `python scripts/list_envs.py` —— 6 个 `wyw` 任务(3 主 + 3 Play)全部出现,无报错。
2. ⬜ `python scripts/view_robot.py --task=Robotics-Wheelbipe-V14-wyw-Flat-v1 --num_envs=1 --device=cpu` —— 机器人生成(视觉检查,**尚未跑**)。
3. ✅ 冒烟训练(短,3 iter,`--num_envs=64 --device=cuda:0 --headless`) —— ActorCriticSequence
   runner 正常加载,obs 25 / critic 46 / encoder 输入 125 / latent 3,无 shape 报错;encoder
   速度估计 loss 下降(Flat 0.47→0.15、value_function 0.86→0.62、mean_kl≈0.008)。
4. ✅ `-Rough-v1`(地形构建 + 课程,encoder loss 0.19→0.06)与 `-Jump-v1`(encoder loss
   0.47→0.16)冒烟训练成功;6 个 jump `rew_*` 项**仅在 Jump 任务**被记录/生效,Flat 中计数为 0。
5. ⬜ 用 Jump 短 checkpoint 跑 `play.py`,肉眼确认涌现的跳跃行为(**尚未跑**,需较长训练后)。

## 待确认风险(★=仍未闭合,✅=已在实现/验证中解决)

**仍未闭合(需较长训练 / 换 USD 时处理):**
- **★ 几何目标当前用 from_hu 值,换 USD 时切换**:`WYW_ROBOT="from_hu"` 下 `BASE_HEIGHT_FLIGHT=0.60`
  (fudan=0.65)、`L0_TUCK=0.16`/`L0_EXTEND=0.31`(两机型近乎重合,共用)。fudan USD 就绪后切
  `WYW_ROBOT="fudan"` 即用 fudan 值。apex 0.60 为待实测初值。
- **★ 起跳能力 / 腿 PD 刚度**:from_hu `legs_act` stiffness 60、effort 40N·m,比 fudan 软增益
  (Kp 6)硬得多;Jump 任务可能蹬不起来,需要单独调软腿 PD 或换 nominal 站姿(与部署 jumpParams
  思路一致)。**冒烟训练只验证了管线跑通,是否真能起跳需较长训练 + play.py 目视确认(验证步骤 5)。**
- **★ 特权观测暂未含 DR / 高度扫描项**:as-built critic=46 只含本体 + 动作历史,未拼 base_mass/
  base_com/friction/restitution/default_dof 偏差与地形高度扫描。若 sim2real 或粗糙地形需要,再作
  为增强追加(并同步 `state_space` int 值)。

**已解决:**
- ✅ **观测关节选择**:obs 的腿关节偏差(4)取 `_legs_act_idx`、dof_vel/actions/torque 等取
  `_actuate_idx`(驱动 4 腿 + 2 轮),已排除被动连杆/弹簧/云台。
- ✅ **Critic/state_space 维度**:不再担心 141——as-built 明确为 **int 46**(见"偏离 2")。
- ✅ **历史缓冲**:基类无此布局历史,已在 env 子类自维护 `_wyw_obs_hist`(N,5,25);runner 侧
  `policy_hist` 经 int space 回落到 `num_obs_hist*num_obs=125`。
- ✅ **自定义 runner 的观测接线**:`OnPolicySequenceRunner` + `RslRlVecEnvWrapper` 从
  `single_observation_space.spaces` 建 `num_observations`,`num_privileged_obs=num_observations["critic"]`;
  冒烟训练确认 obs/critic/encoder 维度全部对齐。
- ✅ **奖励名拼写 / 污染**:jump 6 项与 `WYW_JUMP_REWARD_WEIGHTS` 键逐一核对;`cfg.rewards`
  白名单机制确保仅 Jump 生效(Flat 计数 0 已验证)。
- ✅ 新文件保留 SCUTRobotLab MIT 头;注释用中文,遵循仓库惯例。
