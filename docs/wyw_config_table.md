# WYW FDU 当前配置表

> 更新时间：2026-08-31。本文是当前代码的配置快照，不是设计草案。
> 源码（尤其是 `wyw/env_cfg.py`、`wyw/env.py`、`wyw/fdu_semantics.py`、
> `agent_world/assets/wheelbipe_fdu.py`）是最终权威；本表用于人工审计和复现实验。
> “源码默认值”和“云端本次训练覆盖值”分开记录，避免把一次实验参数误当成永久配置。

## 任务注册

| Task ID                                      | 环境配置                         | 地形                  | 奖励实现    |
| -------------------------------------------- | -------------------------------- | --------------------- | ----------- |
| `Robotics-Wheelbipe-FDU-wyw-Flat-v1`       | `WheelbipeWywFlatEnvCfg`       | flat USD              | Fudan Plane |
| `Robotics-Wheelbipe-FDU-wyw-Flat-Play-v1`  | `WheelbipeWywFlatEnvCfg_Play`  | flat USD              | Fudan Plane |
| `Robotics-Wheelbipe-FDU-wyw-Rough-v1`      | `WheelbipeWywRoughEnvCfg`      | FDU trimesh generator | Fudan Plane |
| `Robotics-Wheelbipe-FDU-wyw-Rough-Play-v1` | `WheelbipeWywRoughEnvCfg_Play` | FDU trimesh generator | Fudan Plane |
| `Robotics-Wheelbipe-FDU-wyw-Jump-v1`       | `WheelbipeWywJumpEnvCfg`       | flat USD              | Fudan Jump  |
| `Robotics-Wheelbipe-FDU-wyw-Jump-Play-v1`  | `WheelbipeWywJumpEnvCfg_Play`  | flat USD              | Fudan Jump  |

Flat、Rough、Jump 都禁用 V3/V14 状态机、special spin/dash 命令和气弹簧；Jump 的跳跃行为由
Fudan reward 和接触/腾空计算产生。

## 本体与闭链

| 项             | 当前值                                                                                                  |
| -------------- | ------------------------------------------------------------------------------------------------------- |
| 权威模型       | `robot_models/fdu_infantry_V4_mujoco/meshes/infantry_V2.urdf` 转换的 `Wheelbipe_FDU_CFG`            |
| 根体           | `base_link_del`；14 DOF、15 rigid bodies                                                              |
| 闭链实现       | USD 中 4 个 spherical loop constraint，标记`physics:excludeFromArticulation`，由 PhysX 约束求解器闭合 |
| 自碰撞         | `enabled_self_collisions=False`（生产配置和已接受跌落测试均关闭）                                     |
| 几何常量       | `L1=0.17472 m`、`L2=0.208 m`；使用 `fdu_mapping.py` 的解析筝形求解器                              |
| 被动机械范围   | `L0≈[0.09495, 0.34149] m`；仅三角形几何上限为 `0.38272 m`                                          |
| 动作工作区保护 | 无；不按`L0/theta0` 投影或裁剪                                                                        |
| 驱动杆         | `lf0 + l20`、`rf0 + r20` 四个实体关节，策略直接控制                                                 |
| 关节顺序       | `lf0_Joint, l20_Joint, l_wheel_Joint, rf0_Joint, r20_Joint, r_wheel_Joint`                            |

四个腿动作直接按 `q_target = q_default + 0.5 * action` 发送给实体驱动杆，不再计算
`(L0, theta0)` 后投影或逆解。`L0/theta0` 只用于 reward、几何诊断和稳定边界日志。轮动作是
速度目标 `10 * action`。策略输出仍为 6 维：四个实体驱动杆位置目标和两个轮速度目标。
runner 侧 Flat/Rough 将策略动作裁剪到 `[-1,1]`；Jump 为兼容既有 Fudan checkpoint，仍使用
`[-100,100]` 的宽裁剪范围。环境侧随后再对轮速度目标裁剪到 `±60 rad/s`。

## 执行器与限制

| 执行器                   | Kp |   Kd | effort limit | velocity limit |
| ------------------------ | -: | ---: | -----------: | -------------: |
| 四个驱动杆（Flat/Rough） | 20 |    1 |       40 N m |       30 rad/s |
| 四个驱动杆（Jump）       |  6 |  0.5 |       40 N m |       30 rad/s |
| 被动闭链关节             |  0 | 0.01 |       50 N m |      300 rad/s |
| 轮（Flat/Rough）         |  0 |  0.2 |        5 N m |       60 rad/s |
| 轮（Jump）               |  0 |  0.2 |       50 N m |       60 rad/s |

轮速度目标缩放为 `10`，运行时 clamp 和 PhysX `velocity_limit_sim` 都是 `60 rad/s`。
这是当前速度目标 adapter 的安全边界；Fudan 原始控制器本身采用轮 torque 控制，不应把
`60 rad/s` 描述成原始 Fudan URDF 的速度限位。四个驱动杆的 40 N m 是当前硬上限，域随机化
只会按比例降低输出（Flat/Rough `[0.95,1.0]`，Jump `[0.9,1.0]`）。

## 时间、环境和求解器

| 项                                    |                         当前值 |
| ------------------------------------- | -----------------------------: |
| physics`dt`                         |          `0.002 s`（500 Hz） |
| decimation                            |                          `5` |
| policy 频率                           |                     `100 Hz` |
| solver position / velocity iterations |                     `16 / 6` |
| episode length                        |  `20 s`（2000 policy steps） |
| 训练环境数                            |                       `4096` |
| Play 默认环境数                       |         `50`（CLI 可以覆盖） |
| self-collision                        |                           关闭 |
| 气弹簧/tendon                         | `use_spring=False`，尚未加入 |

## 命令与 reset

| 项                                           | Flat/Rough                     | Jump                           |
| -------------------------------------------- | ------------------------------ | ------------------------------ |
| `vx`                                       | 初始 `[-0.5, 0.5] m/s`       | `[-2.1, 2.1] m/s`            |
| `vy`                                       | `[0, 0] m/s`                 | `[0, 0] m/s`                 |
| yaw rate                                     | `[-2.0, 2.0] rad/s`          | `[-2.0, 2.0] rad/s`          |
| 高度命令                                     | `[0.15, 0.30] m`             | `[0.15, 0.30] m`             |
| resampling period                            | `5 s`                        | `20 s`                       |
| heading command                              | `False`（直接采样 yaw rate） | `False`（直接采样 yaw rate） |
| `rel_heading_envs` / `rel_standing_envs` | `0 / 0`                      | `0 / 0`                      |

所有任务 reset 时线速度、角速度各轴均在 `[-0.5,0.5]` 随机化。Rough 额外将 reset
XY 位置随机化到 `[-1,1] m`。Jump 每 5 s 施加一次 XY 速度 push，范围 `[-1.5,1.5] m/s`。
Play 关闭 startup 随机化、push 和观测噪声，只保留 reset event；默认仍保留 episode timeout。
Flat 每 2000 个 policy steps（20 s）检查一次课程条件：参与 reset 的环境平均线速度/偏航跟踪率
分别超过 `0.70/0.56` 时，将所有环境的 `vx` 上下界各扩展 `0.1 m/s`，最大到 `±2.5 m/s`。
Rough 的 `vx` 也从 `[-0.5,0.5] m/s` 开始，但由地形课程按环境分别更新。

## 观测契约

| 张量           | 维度 | 段顺序                                                                                                 |
| -------------- | ---: | ------------------------------------------------------------------------------------------------------ |
| policy         |   25 | `ang_vel(3), projected_gravity(3), command(vx,yaw,height)(3), leg_pos_dev(4), dof_vel(6), action(6)` |
| policy history |  125 | 最近`5 x 25` 帧，reset 后用当前帧填满                                                                |
| critic         |  141 | `base_lin_vel(3), clean_policy(25), a[t-1](6), a[t-2](6), dof_acc(6), scan(77), torque(6), DR(12)`   |
| encoder latent |    3 | 由历史 policy 观测估计隐式 base linear velocity                                                        |
| action         |    6 | 四驱动杆位置目标 + 左右轮速度目标                                                                      |

观测缩放：`ang_vel=0.25`、`dof_vel=0.05`、Flat/Rough `lin_vel=2`、Jump `lin_vel=3`、
yaw command `0.25`、height command `1`、gravity `1`、joint position `1`、critic joint
acceleration `0.0025`、critic torque `0.05`、height scan `5`。action 段不缩放。

训练 actor 噪声为：角速度 `±0.2`、投影重力 `±0.05`、关节位置 `±0.02`、腿/轮速度
`±1.5`；critic 的 policy 段使用 clean 副本。obs delay 和 action delay 均关闭；Play 噪声关闭。

## 奖励

Flat/Rough 的权重（raw term 乘 weight 乘 `step_dt=0.01` 后，再按单项裁剪）如下：

| term                        |  weight | 含义 |
| --------------------------- | ------: | ---- |
| `termination`               |    -500 | 姿态或指定机身/腿部 link 接触连续失败 1 s 后的一次性终止惩罚；在单项裁剪之后加入 |
| `tracking_lin_vel`          |       1 | 在重力对齐 yaw frame 跟踪 x 向速度；乘 `clamp(-gz,0,0.7)/0.7` 姿态 gate |
| `tracking_lin_vel_enhance`  |       1 | 更宽容的线速度跟踪塑形；完全跟踪时为 0，有误差时为负 |
| `tracking_ang_vel`          |       1 | 跟踪世界 z 轴偏航角速度；乘 `clamp(-gz,0,0.7)/0.7` 姿态 gate |
| `base_height`               |       1 | 跟踪基座高度命令；乘 `clamp(-gz,0,0.7)/0.7` 姿态 gate |
| `upright_orientation`       |       1 | 正向直立奖励 `exp(-(gx²+gy²)/0.025)`，仅 `gz<0` 生效；10° 约 0.30，侧翻和倒置为 0 |
| `nominal_state`             |      -1 | 惩罚左右虚拟腿角度 `theta` 不对称 |
| `lin_vel_z`                 |      -1 | 惩罚基座竖直速度 |
| `ang_vel_xy`                |   -0.05 | 惩罚基座 roll/pitch 角速度 |
| `orientation`               |     -15 | 惩罚投影重力的 x/y 分量，即机身倾斜 |
| `dof_vel`                   |   -5e-5 | 惩罚四个腿部驱动杆的关节速度平方和 |
| `dof_acc`                   |   -3e-7 | 惩罚六个受控关节的加速度平方和 |
| `torques`                   |   -1e-3 | 惩罚六个受控关节的施加力矩平方和 |
| `action_rate`               |    -0.3 | 惩罚六维动作的一阶差分 `a[t]-a[t-1]` |
| `action_smooth`             |    -0.3 | 惩罚四维腿动作的二阶差分 `a[t]-2a[t-1]+a[t-2]` |
| `collision`                 |      -1 | 统计 base 和非轮腿部 link 上超过 `0.1 N` 的接触个数 |
| `dof_pos_limits`            |      -1 | 惩罚四个腿部驱动杆超出 97% 软关节范围的距离 |

Jump 与 Flat/Rough 不同的奖励项如下；未列出的 `tracking_ang_vel` 和 `collision` 权重及含义相同：

| term                       | weight | 与 Flat/Rough 的差异或含义 |
| -------------------------- | -----: | -------------------------- |
| `termination`              |   -200 | 同类一次性持续失败终止惩罚，但幅值较小 |
| `tracking_lin_vel`         |      1 | 权重相同，raw term 额外乘 `2` |
| `tracking_lin_vel_enhance` |      1 | 权重相同，raw term 额外乘 `2` |
| `flight`                   |   0.15 | 奖励两轮均离地 |
| `encourage_jump`           |      1 | 使用 Fudan 兼容的腾空计时逻辑鼓励跳跃 |
| `base_height_flight`       |      6 | 腾空时奖励基座接近 `0.65 m` |
| `leg_tuck`                 |    1.7 | 腾空时奖励两腿收至 `L0=0.23 m` |
| `takeoff_extend`           |    0.5 | 有轮接触且向上起跳时奖励两腿伸至 `L0=0.31 m` |
| `line_z`                   |      6 | 腾空时奖励正的世界系竖直速度 |
| `pen_theta_no0`            |     -2 | 惩罚左右虚拟腿角度各自偏离 0 |
| `action_rate`              |  -0.04 | 动作一阶差分惩罚弱于 Flat/Rough |
| `torques`                  |  -5e-5 | 力矩惩罚弱于 Flat/Rough |
| `orientation`              |    -25 | 倾斜惩罚强于 Flat/Rough |
| `ang_vel_xy`               |   -0.1 | roll/pitch 角速度惩罚强于 Flat/Rough |
| `nominal_state`            |     -1 | 除 `theta` 不对称外，再惩罚 `10*(L0_left-L0_right)^2` |

Jump 不使用 Flat/Rough 的 `base_height`、`upright_orientation`、`lin_vel_z`、`dof_vel`、
`dof_acc`、`action_smooth` 和 `dof_pos_limits`。

关键 raw 公式（见 `fdu_semantics.py`）：

- 速度跟踪：`exp(-squared_error / 0.25)`；enhance 使用 `exp(-squared_error / 2.5) - 1`。
- Jump 的线速度跟踪两项额外乘 `2`。
- 高度：`exp(-square(height - command) / 0.001) * clamp(-gz,0,0.7)/0.7`；姿态惩罚为
  `gx^2 + gy^2`；独立的窄直立
  奖励为 `exp(-(gx^2+gy^2)/0.025) * (gz<0)`。Flat/Rough 的正向速度跟踪项另乘
  RobotLab gate：`clamp(-gz, 0, 0.7) / 0.7`。
- `action_smooth` 使用 `a[t] - 2*a[t-1] + a[t-2]` 的腿部四维。
- Jump 腾空项由两轮接触过滤器决定；`leg_tuck` 目标 `L0=0.23 m`，起跳伸展目标
  `L0=0.31 m`，飞行基座高度 `0.65 m`，起跳竖直速度阈值 `0.15 m/s`，接触阈值 `1 N`。
- Jump `nominal_state` 包含 theta 不对称项以及 `10*(L0_left-L0_right)^2`。

Flat/Rough `clip_single_reward=1.0`，Jump 为 `2.5`，所以单项最终步长裁剪分别为
`±0.01` 和 `±0.025`。`only_positive_rewards=False`。姿态/接触持续失败终止会在单项裁剪之后
额外加入：Flat/Rough 为 `-500 * 0.01 = -5.0`，Jump 为 `-200 * 0.01 = -2.0`；timeout、
Rough 地形越界和数值安全重置不加该惩罚。三个任务的 collision 都检查映射到的 base/非轮腿部
link 是否存在大于 `0.1 N` 的接触。当前代码有意保留 Fudan 的 `base_air_time *= ~in_flight` 跨 reset
状态 bug，后续修正版应作为独立消融实验。

## 终止与稳定性诊断

- 数值安全条件（NaN/Inf 等）立即终止。
- `projected_gravity_b[:,2] > -0.1` 连续 `100` 个 policy steps 才终止（1 s 持续倾倒）。
- Play 默认将普通 termination 清零（可用 `play_keep_done_reset=True` 恢复），timeout 仍有效。
- Rough 离开生成地形边界会立即终止，但不计作 timeout，也不触发一次性失败终止惩罚。
- 500 Hz 每 physics step 采样等效腿长；`L0 <= 0.14 m` 只累计当前 episode 的越界样本，
  reset 时记录受影响环境比例。它是精简后的诊断指标，不是动作 clamp 或终止条件。

## 域随机化

| 项                          | Flat/Rough      | Jump           |
| --------------------------- | --------------- | -------------- |
| base mass addition          | `[-1,2] kg`   | `[-2,3] kg`  |
| 每刚体 mass/inertia scale   | `[0.9,1.1]`   | `[0.8,1.2]`  |
| base COM xyz                | `±0.02 m`    | `±0.05 m`   |
| robot friction              | `[0.6,1.4]`   | `[0.1,2.0]`  |
| robot restitution           | `[0.6,1.0]`   | `[0.5,1.0]`  |
| default policy-joint offset | `±0.03 rad`  | `±0.05 rad` |
| Kp/Kd scale                 | `[0.95,1.05]` | `[0.9,1.1]`  |
| effort output scale         | `[0.95,1.0]`  | `[0.9,1.0]`  |

每个环境独立采样并把 base mass deviation、COM、default joint delta、friction、restitution
写入 critic 的 12 维 privilege。仿真/地面材质 friction/restitution 默认均为 `0.5/0.5`，
combine mode 为 `average`。

## Rough 地形与课程

地形生成器：size `8 x 8 m`、border `25 m`、`10` rows、`20` cols、horizontal scale `0.1 m`、
vertical scale `0.005 m`、curriculum 开启。子地形比例为：flat `.20`、smooth slope up/down
`.10/.10`、rough slope up/down `.10/.10`、stairs down `.10`、stairs up `.20`、obstacles `.10`。
rough slope 叠加随机高度，noise step `0.005 m`，范围随 difficulty 为 `±(0.05+0.05*difficulty) m`；
stairs 高度 `0.05--0.23 m`，宽 `0.7 m`；obstacle 高度 `0.05--0.15 m`。

reset 时若距 terrain origin 超过 terrain length 的四分之一（`2 m`）则升级；否则 tracking rate
低于 `.4` 降级。达到最高等级且 tracking rate 大于 `.7` 时扩展命令范围：基础地形每次 `0.5`
（上限 `|vx|=2.5`），困难地形每次 `0.05`（上限 `|vx|=1.5`）。失败回退时命令范围按 `0.25`
收缩，边界为 `[-2.5,-1.0]` 到 `[1.0,2.5]`。

## PPO

Flat/Rough/Jump 共享 `WheelbipeWywPPORunnerCfg` 的网络和 PPO 超参；experiment name 分开，
Jump 另行放宽 runner 动作裁剪范围：

| 项                                      | 值                                                                     |
| --------------------------------------- | ---------------------------------------------------------------------- |
| runner / policy / algorithm             | `OnPolicySequenceRunner` / `ActorCriticSequence` / `PPOSequence` |
| rollout                                 | `48` steps/env                                                       |
| source runner`max_iterations`         | `20000`                                                              |
| save interval                           | `500`                                                                |
| actor / critic / encoder                | `[128,64,32]` / `[256,128,64]` / `[128,64]`                      |
| latent / activation / init std          | `3` / ELU / `0.5`                                                  |
| orthogonal init                         | `False`                                                              |
| value coef / clipped value / clip param | `1` / `True` / `.2`                                              |
| entropy / epochs / minibatches          | `.01` / `5` / `4`                                                |
| policy LR / encoder LR                  | `1e-3` / `1e-4`                                                    |
| encoder loss                            | Smooth L1（Huber delta `1.0`），排除 terminal 样本                    |
| runner action clip                      | Flat/Rough `±1`；Jump `±100`                                        |
| schedule / gamma / lambda               | adaptive /`.99` / `.95`                                            |
| desired KL / max grad norm              | `.005` / `1`                                                       |

本次云端 Flat 训练使用 launch CLI 将 `max_iterations` 覆盖为 `5000`，环境数 `4096`、seed
`42`；这不是源码 runner 默认值。该任务在移除动作投影前启动，checkpoint 属于旧动作语义，不能作为
当前直接四杆目标配置的续训或 Play 验收基线。

## Play 与验证状态

Play 配置默认 50 envs、无观测噪声和 startup DR；`scripts/rsl_rl/play.py` 可用 `--num_envs`、
`--max_steps`、`--video` 覆盖。可视化/视频脚本输出应放在 `docs/fdu_validation/video/`。

已接受的几何报告为 [`geometry/fdu_calibration_report.json`](./fdu_validation/geometry/fdu_calibration_report.json)，
验证索引见 [`fdu_validation/README.md`](./fdu_validation/README.md)。当前直接动作路径已通过 1-env、CPU、
500 Hz Flat reset/step smoke，pure-tensor mapping/reward tests 为 20 passed。已有长时 reset/command 和
64-env、3-iteration GPU smoke 均使用旧投影动作语义，只能作为历史对照；Flat/Rough/Jump 必须重新执行
GPU smoke 后才能恢复验收状态。旧 Jump 运行在未加入气弹簧时仍是物理失败，最终 iteration 有 58/64
个环境进入 L0 稳定边界。短腿跌落视频和报告保留在 `fdu_validation/video/` 与 `drop/`，800 Hz A/B
仅作诊断，不改变当前生产配置。

## 已知限制

1. 闭链由最大坐标约束求解，剧烈运动或短腿时仍可能出现数值抖动；`L0` 边界日志用于追查，
   不是修复。当前生产折中是 `500 Hz + 16/6`。
2. 气弹簧/tendon 尚未建模，Jump 不能视为部署前的物理验收结果。
3. 轮速度 adapter 的 60 rad/s 和四驱动杆 40 N m 是当前仿真保护值；真实电机电流、减速比、
   编码器零点和方向仍需硬件辨识。
