# wyw 任务族迁移设计（intention_v3）

> 状态：迁移基线已实现；下文同时记录目标契约、已完成验收和未完成的长训/几何标定项。
>
> 本文取代 `intention_v2.md` 作为 wyw 任务族的设计权威；`intention_v2.md` 保留为历史记录。
> [`wyw_config_table.md`](./wyw_config_table.md) 是 2026-08-27 的 as-built 快照，只用于比较当前实现与
> 本文目标之间的差距，不能覆盖本文定义。

## 0. 范围、依据与设计原则

wyw 包含 Flat、Rough、Jump 及其 Play 变体。迁移目标是：

1. 机器人完全使用指定本体：`robot_models/fdu_infantry_V4_mujoco/meshes/infantry_V2.urdf`，包括
   `base_link_del.STL`、云台相关实体、全部腿/轮 STL、质量、质心和惯量；工程配置为 `Wheelbipe_FDU_CFG`。
2. 与机器人结构无关的训练语义精确对齐 Fudan，包括时序、公式、裁剪顺序和随机量含义，而不只是
   奖励项名称或近似效果相似。
3. 与机器人结构绑定的量按语义映射到 FDU 闭链本体，并通过具名配置显式记录，不依赖隐含数组顺序。
4. 使用 Isaac Lab 的 `DirectRLEnv`、`configclass` 和事件项风格实现，不复制 legged_gym 框架代码。

Fudan 权威源为以下工作树，版本 `403f391c481566560293f6fbf68aa93bdceb583a`：

- `fudan_rl_wheel_leg/plane/wheel_legged_gym/envs/base/{legged_robot.py,legged_robot_config.py}`
- `fudan_rl_wheel_leg/jump/wheel_legged_gym/envs/base/{legged_robot.py,legged_robot_config.py}`
- Jump 已训练快照（只用于确认实际训练过的 reward 实现，不作为整套 config 权威）：
  `fudan_rl_wheel_leg/jump/logs/wheel_legged/p60.50.2目前测试下来最好的/`

若本文与上述版本的可执行代码冲突，先写 golden test 复现 Fudan 行为，再更新本文；不能直接用当前
Isaac Lab 实现或历史训练 YAML 猜测目标行为。

已训练 Jump 快照仍包含 §7.4 的 air-time bug，因此该行为属于基线兼容范围；但快照还把
`added_mass_range` 改为 `[-2, 23]` 并启用了 action delay，这两项不随 bug 一并迁入，本文仍以当前
base config 的 `[-2, 3]` 和 `randomize_action_delay=False` 为目标。

### 0.1 已确认的有意差异

以下值不照抄 Fudan，属于已确认的 FDU 本体适配：

| 项               |                         wyw 目标 | 原因                                      |
| ---------------- | -------------------------------: | ----------------------------------------- |
| 机器人           |          `Wheelbipe_FDU_CFG` | 指定 `infantry_V2.urdf` 的闭链并联本体      |
| 高度命令范围     |               `[0.20, 0.42]` m | FDU 几何量纲                              |
| 默认高度命令     |                       `0.22` m | 已确认                                    |
| 高度命令观测缩放 |                          `1.0` | 已确认；Fudan 此处为`5.0`               |
| 腿部 PD 初值     |                      Kp=20、Kd=1 | FDU 仿真控制初值；Jump 覆写为 Kp=6、Kd=0.5 |
| 动作顺序         | `lf0,l20,l_wheel,rf0,r20,r_wheel` | 用户已确认；当前无部署兼容约束            |

滞空目标高度、收腿/伸腿目标长度仍需 FDU 几何标定，见 §12。除此之外，不允许为了复用现成
V3/V14 奖励或命令机制而改变 Fudan 语义。

### 0.2 本体转换和已完成标定

- URDF 清理脚本为 `robot_models/fdu_infantry_V4_mujoco/build_infantry_closed_urdf.py`，输入只允许是
  `meshes/infantry_V2.urdf`；不能替换成同目录 `urdf/infantry_V2.urdf`。清理只修复导入所需的零关节
  限位、`r21_Link` 的关节命名和导出为零的 `l20_Link` 惯量，不改 `base_link_del` 的质量/质心/惯量。
- USD 转换后由 `add_loop_joints.py` 添加四个球铰闭链约束，并将其排除出 reduced-coordinate articulation；
  四个驱动杆仍是实际 articulation DOF，不存在 dummy spring DOF。
- CPU 标定报告见 [`fdu_calibration_report.json`](./fdu_calibration_report.json)：14 DOF、15 刚体、根质量
  11.718 kg、10 步静置状态有限。报告中的 actuator natural order 仅用于审计，策略顺序始终按 §2.1 的
  具名映射。
- `effort_limit`、PD 和动作缩放是仿真控制假设，不等于真实电机额定参数；真实驱动器的力矩常数、减速比、
  电流限幅和编码器方向仍需硬件辨识后单独写入配置版本。

## 1. 架构和配置所有权

```text
DirectRLEnv
 └─ Wheelbipe25V3Env                  共享仿真基础设施
     ├─ WheelbipeV13Env -> WheelbipeV14Env
     └─ WheelbipeWywEnv               wyw 独立任务族
```

- `WheelbipeWywEnv` 直接继承 `Wheelbipe25V3Env`，不得继承 V13/V14 任务类；机器人由 `Wheelbipe_FDU_CFG` 提供。
- Flat、Rough、Jump 配置均直接继承 WYW base config；Jump 不继承 Flat，Rough 也不继承 V14 config。
- 可以复用 V3 的关节发现、控制器、接触传感器、扫描器和数值安全设施，但任务级观测、复位、终止、
  命令、课程和奖励聚合由 WYW 覆写。
- Rough 所需的边界复位和地形课程写在 WYW 自有模块中，不复制 V14 的 `TerrainCommandManager`、
  `SpecialMode`、云台旋转或按训练迭代数升级地形的逻辑。
- 所有影响行为或张量契约的数值必须是 `configclass` 字段并写入 `params/env.yaml`。禁止用导入时全局
  `WYW_ROBOT` 分支构造配置，也禁止把机器人选择、几何目标、阈值藏在模块级条件中。
- 模块常量只保存真正不随任务实例变化的维度、字段名和默认顺序；任务可调值放入 config。

云台不属于动作或观测关节。复位时回到资产默认状态；其质量、STL 和惯量来自指定 URDF。WYW 不引入
云台动作。

## 2. 动作、关节和机体映射

### 2.1 唯一动作顺序

动作、完整关节速度、关节加速度和力矩的 6 维顺序固定为：

| 索引 | WYW 关节               | 控制方式 | Fudan 语义        |
| ---: | ---------------------- | -------- | ----------------- |
|    0 | `lf0_Joint`          | 位置     | left front drive bar  |
|    1 | `l20_Joint`          | 位置     | left rear drive bar   |
|    2 | `l_wheel_Joint`      | 速度     | left wheel            |
|    3 | `rf0_Joint`          | 位置     | right front drive bar |
|    4 | `r20_Joint`          | 位置     | right rear drive bar  |
|    5 | `r_wheel_Joint`      | 速度     | right wheel           |

对应的语义索引必须在初始化时由精确名称解析并断言：

```text
LEG_ACTION_IDS   = [0, 1, 3, 4]
WHEEL_ACTION_IDS = [2, 5]
LEFT_ACTION_IDS  = [0, 1, 2]
RIGHT_ACTION_IDS = [3, 4, 5]
```

不得依赖 `find_joints()`、USD 或正则匹配返回的自然顺序。环境初始化后必须断言六个名称各匹配一次、
无重复、动作数为 6，并在日志中打印最终映射。所有 leg-only 奖励使用 `LEG_ACTION_IDS`，不能再写
`[:4]` 或 `[0:2, 3:5]` 之类与布局偶然耦合的切片。

动作值是网络原始输出。腿目标为 `default_joint_pos + 0.5 * action`，轮目标速度为 `10.0 * action`
rad/s；二者使用语义 mask 分别生成。Fudan 原训练和指定闭链 MuJoCo sim2sim 都没有对 `vel_ref` 做 wheel
target speed clamp：它们以速度误差经 wheel damping 生成力矩，再裁剪轮力矩。原 Isaac Gym URDF 的
`1500 rad/s` 和指定 `infantry_V2.urdf` 对全部关节统一写入的 `1000 rad/s` 都不是策略工作轮速标定，不能作为
WYW 的运行时上限。当前 WYW 仍采用 Isaac Lab velocity-target adapter，因此显式设 `60 rad/s`：0.06 m 轮半径下，
Fudan 最大前进命令 2.1 m/s 对应 35 rad/s；最大偏航命令 2 rad/s、半轮距约 0.1877 m 再贡献约 6.3 rad/s，
极端组合约 41.3 rad/s，60 rad/s 留有约 45% 裕量。FDU asset 的 `velocity_limit`、`velocity_limit_sim`、WYW
配置 `max_wheel_vel` 和运行时 clamp 必须同时保持为 `60 rad/s`。这是 adapter 兼容边界，不是由
`clip_actions=100` 推算的 Fudan 硬限制。

轮力矩上限按**训练**配置取值：Flat/Rough 为 `5 N·m`，Jump 为 `50 N·m`。这里不能混用后续阶段的
限制：闭链 MuJoCo 常规 sim2sim 代码把轮力矩裁到 `5 N·m`，真实部署又把 Jump 单独裁到 `4 N·m`；
后两者都是策略训练完成后的执行保护，不是当前 Isaac Lab Jump 训练上限。

### 2.2 虚拟腿几何

Fudan 的 `L0`、`theta0` 是腿根到轮中心的虚拟杆长度和机身系摆角。FDU 为五连杆闭链，WYW 从
仿真实际轮连杆位姿计算，不照搬二连杆解析式：

```python
wheel_from_hip_b = wheel_pos_b - hip_offset_b
L0 = norm(wheel_from_hip_b[..., [0, 2]], dim=-1)
theta0 = atan2(-wheel_from_hip_b[..., 0], -wheel_from_hip_b[..., 2])
```

- `wheel_pos_b` 必须由完整 root quaternion 变换到机身系，不能使用只去 yaw 的 heading frame。
- `hip_offset_b` 是每侧腿根相对 base 原点的资产几何配置；只有验证 x/z 均为 0 后才允许省略。
- 左右输出固定为 `[left, right]`。
- helper 每个仿真步只计算一次，所有 `nominal_state` 和 Jump 奖励复用缓存结果。

## 3. 时间和缓冲区契约

物理 `dt=0.005` s，`decimation=2`，策略周期 `step_dt=0.01` s（100 Hz），episode 20 s。

在一次策略步完成物理推进后，按以下逻辑顺序更新：

1. 刷新物理状态、接触和扫描结果。
2. 计算 clean proprioception、派生量、终止状态及所有 raw reward term。
3. 聚合奖励。
4. 生成 clean critic observation。
5. 从同一份 clean proprioception 复制并加噪，生成 actor observation。
6. 将 noisy actor observation 推入 5 帧 policy history。
7. 完成 reset 环境的缓冲初始化。
8. 在所有使用旧动作历史的计算结束后，执行 `a_{t-2} <- a_{t-1}`、`a_{t-1} <- a_t`。

其中 `a_t` 是本步刚施加的原始网络动作。下一次决策看到的 policy 末段因此是该动作。奖励中的
`action_rate` 比较 `a_t` 与 `a_{t-1}`，`action_smooth` 比较 `a_t - 2a_{t-1} + a_{t-2}`。

Fudan 没有启用 observation delay，且 `randomize_action_delay=False`。WYW 必须显式关闭 V3 的观测延迟、
动作延迟及相关随机事件；不能以“范围为零但机制仍开启”替代。reset 后动作历史清零，policy history 用
reset 状态生成的 noisy observation 填满 5 帧，不能留跨 episode 数据。

## 4. 观测契约

### 4.1 Policy observation：25 维、带噪

| 区间        | 维度 | 内容                                         |                     缩放 |
| ----------- | ---: | -------------------------------------------- | -----------------------: |
| `[0:3]`   |    3 | body-frame base angular velocity             |                     0.25 |
| `[3:6]`   |    3 | projected gravity                            |                      1.0 |
| `[6:9]`   |    3 | `[vx_cmd, yaw_rate_cmd, height_cmd]`       | `[2.0/3.0, 0.25, 1.0]` |
| `[9:13]`  |    4 | leg position delta，顺序`[LF, LR, RF, RR]` |                      1.0 |
| `[13:19]` |    6 | joint velocity，使用 §2.1 完整动作顺序      |                     0.05 |
| `[19:25]` |    6 | 当前已施加的原始动作`a_t`                  |                      1.0 |

Flat/Rough 的线速度命令缩放为 2.0，Jump 为 3.0。腿位置必须是相对当前每环境随机默认关节位置的偏差，
即 `joint_pos - randomized_default_joint_pos`。

噪声是逐元素均匀分布 `U[-amplitude, amplitude]`，在上述缩放之后加入：

| 段                       | 最终 amplitude |
| ------------------------ | -------------: |
| angular velocity         |           0.05 |
| projected gravity        |           0.05 |
| leg position             |           0.02 |
| all six joint velocities |          0.075 |
| commands、actions        |              0 |

Policy history 为 5 个按时间从旧到新排列的 noisy 25-D observation，共 125 维。它不是 critic history。

### 4.2 Critic observation：141 维、clean proprioception

Critic 必须在 actor 加噪前构造。它内嵌的 25-D proprioception 是 clean 数据，不得复用已经加噪的
policy tensor。

| 区间          | 维度 | 内容                                                              |
| ------------- | ---: | ----------------------------------------------------------------- |
| `[0:3]`     |    3 | clean body-frame base linear velocity，Flat/Rough x2.0，Jump x3.0 |
| `[3:28]`    |   25 | clean policy-layout proprioception                                |
| `[28:34]`   |    6 | `a_{t-1}`                                                       |
| `[34:40]`   |    6 | `a_{t-2}`                                                       |
| `[40:46]`   |    6 | joint acceleration x0.0025，§2.1 顺序                            |
| `[46:123]`  |   77 | terrain height samples                                            |
| `[123:129]` |    6 | applied joint torque x0.05，§2.1 顺序                            |
| `[129:130]` |    1 | `base_mass - mean(base_mass across envs)`                       |
| `[130:133]` |    3 | 本次采样并施加的 base COM offset                                  |
| `[133:139]` |    6 | randomized default joint position - raw nominal position          |
| `[139:140]` |    1 | 本环境采样的 friction scalar                                      |
| `[140:141]` |    1 | 本环境采样的 restitution scalar                                   |

高度网格为 x=`[-0.5, ..., 0.5]`（11 点）、y=`[-0.3, ..., 0.3]`（7 点），展平顺序固定并测试。
每个点使用地面命中高度 `ray_hit_z`，特征公式严格为：

```python
height_feature = clamp(root_z - 0.5 - ray_hit_z, -1.0, 1.0) * 5.0
```

不能使用 scanner origin 到地面的距离，也不能忽略常数 `0.5`。Flat/Jump 的 `ray_hit_z=0` 仍按同一
公式生成，不将整段强制清零。

DR 特权值必须来自事件随机化时保存的 per-env sample/source-of-truth：COM 用 offset 而非绝对 COM；
摩擦/恢复系数用单个采样标量而非所有 body/shape 的均值；默认关节偏差使用随机后的默认值；质量偏差
使用当前 batch 的均值，而非 asset 默认质量。

`ActorCriticSequence`、`PPOSequence` 和 `OnPolicySequenceRunner` 保持不变。encoder 输入为 125，
`latent_dim=3`，监督目标为 critic `[0:3]`。actor/critic/encoder 隐层分别为 `[128,64,32]`、
`[256,128,64]`、`[128,64]`；其余已用的 PPO 超参可保留，但必须分别为 Flat/Rough/Jump 使用独立
`experiment_name`。

## 5. 命令

迁移基线的命令固定为 `[vx, yaw_rate, height]`，无 `vy`、heading、standing env 和 special mode：

```text
heading_command = False
rel_heading_envs = 0
rel_standing_envs = 0
lin_vel_y = [0, 0]
```

| 任务  | vx                            | yaw rate    | height           |         重采样 | command curriculum |
| ----- | ----------------------------- | ----------- | ---------------- | -------------: | ------------------ |
| Flat  | `[-2.0, 2.0]`               | `[-2, 2]` | `[0.20, 0.42]` |   `(5, 5)` s | 平地上关闭实际更新 |
| Rough | 由 Fudan 课程在上限内逐步扩展 | `[-2, 2]` | `[0.20, 0.42]` |   `(5, 5)` s | 开                 |
| Jump  | `[-2.1, 2.1]`               | `[-2, 2]` | `[0.20, 0.42]` | `(20, 20)` s | 关                 |

偏航角速度直接均匀采样，不经过 heading error 控制器。Isaac Lab command term 若不能同时表达上述契约，
写 WYW 自有 command term，不通过继承 `SpecialModeUniformVelocityCommand` 后再尝试关闭多数分支。

Rough 命令课程复现 Fudan 的性能阈值和 basic/advanced terrain 分类；不得使用训练 iteration 作为速度
命令或地形升级条件。Flat 虽在 Fudan config 中写有 `commands.curriculum=True`，但 plane 会关闭 terrain
curriculum，相关 terrain-dependent 路径不应在 WYW Flat 产生动态命令范围。

### 5.1 当前实现、Fudan 与常用 V3 命令的区别

| 项目 | Fudan 原训练 | 当前 WYW 迁移基线 | V3 常用配置 | 结论 |
| ---- | ------------ | ----------------- | ----------- | ---- |
| standing env | 无 | `0%` | `10%` | 有助于补足静止控制，但不是 Fudan 基线 |
| heading control | 代码支持但配置关闭；若打开则 `yaw_rate=clip(1.5*heading_error,-5,5)` | 关闭，直接采样 `yaw_rate` | `50%` heading env，stiffness `1.0` | 可作为增强项，但目标分布和限幅必须重新明确 |
| command 周期 | Plane/Rough `5 s`，Jump `20 s` | 与 Fudan 相同 | `1--8 s` | 第一版增强配置仍保留 `5/20 s`，不要同时改变多个变量 |
| 前进/偏航命令 | 直接均匀采样 | 与 Fudan 相同 | heading env 的偏航由闭环生成 | heading 打开后，策略仍只观察最终 yaw-rate，不增加 actor 维度 |
| 高度命令 | 始终采样 | 与 Fudan 相同 | 始终采样 | standing 只清零平面速度命令，高度命令继续有效 |

### 5.2 未来命令增强变体（规划，当前不启用）

我预计后续会加入独立的 `CommandResetAug` 训练变体，而不直接改写 Flat/Rough/Jump 的 Fudan 兼容基线。
首个增强版本建议只加入：

```text
rel_standing_envs = 0.10
heading_command = True
rel_heading_envs = 0.50
heading_range = [-pi, pi]
heading_control_stiffness = 1.0
effective_yaw_rate_limit = [-2, 2]
Flat/Rough resampling_time = 5 s
Jump resampling_time = 20 s
```

这里的 heading 闭环应先计算 wrap 后的 heading error，再把最终 yaw-rate 裁剪到现有 `[-2,2] rad/s`，以免
在同一次实验中把任务定义扩展到 Fudan dormant code 的 `[-5,5] rad/s`。actor 仍接收最终
`[vx,yaw_rate,height]`，不接收绝对 heading，因此 observation 的 25 维布局和已训练策略接口保持不变。

standing env 的定义为 `vx=0, vy=0, yaw_rate=0`，但保留采样到的 height。standing mask 最后施加，优先于
heading mask；两种 mask 即使重叠也不能产生非零 yaw-rate。Rough 的 command curriculum 统计应排除
standing env，或分别记录 moving/standing tracking 指标，否则较容易的零速样本会虚高成功率并提前扩展
速度范围。

引入顺序固定为：先完成 Fudan 兼容基线验收，再仅打开 standing 做消融，最后加入 heading。增强配置使用
独立 task ID、runner `experiment_name` 和 checkpoint 目录；至少分别记录 tracking、静止漂移、偏航误差、
terrain level 和 reset 后恢复率。不能用增强实验覆盖迁移基线。

## 6. Reset 和 termination

### 6.1 Reset

每个环境 reset 时：

- 关节位置 = 本环境的 randomized default joint position。
- 所有关节速度 = 0。
- root 姿态和 z 使用任务配置的初始状态。
- root 6 维线/角速度分别独立均匀采样于 `[-0.5, 0.5]`。
- Flat/Jump 的 root XY 固定在环境原点，不额外随机偏移。
- Rough 的 root XY 在 tile origin 周围分别均匀偏移 `[-1, 1]` m。
- 清空 failure persistence、动作历史和 episode 统计。Jump 的 `base_air_time` 例外，按 §7.4 的 Fudan
  兼容行为处理。

必须关闭 V3 继承的随机腿长/腿角、随机轮角和随机关节速度 reset。default joint randomization 属于域随机化，
它先生成每环境默认位姿，reset 再精确写入该位姿；两者不能混为第二套 reset 姿态随机化。

### 6.1.1 Reset 配置对比

| 项目 | Fudan 原训练 | 当前 WYW 迁移基线 | V3 当前/常用做法 |
| ---- | ------------ | ----------------- | ---------------- |
| root XY | Flat/Jump 固定；Rough `[-1,1] m` | 相同 | 当前 V3 reset 未随机 XY；注释方案为 `[-0.5,0.5] m` |
| root z 偏移 | 不随机 | 不随机 | 当前 V3 为相对默认位姿 `[0,0.1] m` |
| roll/pitch/yaw | 不随机 | 不随机 | 当前 V3 为 roll/pitch `[-0.1,0.1] rad`、yaw `[-pi,pi]` |
| root 线速度 | xyz 均为 `[-0.5,0.5] m/s` | 相同 | 当前 V3 仅 x/y 为 `[-0.5,0.5] m/s`，z 未随机 |
| root 角速度 | xyz 均为 `[-0.5,0.5] rad/s` | 相同 | 当前 V3 未随机 |
| 驱动关节位置 | randomized default position | 相同 | 可额外采样腿姿态、轮角或预定义恢复姿态 |
| 所有关节速度 | `0` | `0` | 可随机，但当前迁移基线关闭 |

因此“当前 WYW 有 reset 随机化”和“当前 WYW 关闭 V3 reset 扩展”并不矛盾：当前已启用的是 Fudan 的
root 六维初速度随机化以及 Rough XY 随机化；关闭的是 root pose、额外关节姿态和关节速度随机化。

### 6.1.2 未来 reset 增强变体（规划，当前不启用）

建议按恢复难度逐级加入，且每一级都保持可单独关闭：

1. `PoseResetMild`：在 Fudan 初速度随机化之上，加入 yaw `[-pi,pi]`、roll/pitch `[-0.1,0.1] rad` 和
   相对默认 root z `[0,0.1] m`；Flat/Jump XY 仍固定，Rough 继续使用 `[-1,1] m`。
2. `JointResetMild`：只对四个实体驱动杆添加小范围 reset offset，轮关节角可任意但轮速仍从 0 开始；范围需
   在 FDU 闭链的无自碰、无装配冲击可行域标定后填写，不能直接复用 Fudan 两虚拟腿关节或 V3 串联腿范围。
3. `RecoveryReset`：最后再加入趴倒、侧倒、预定义离地/落地等混合 reset。此阶段需同时定义各模式概率、
   起始无力矩时间、临时 command 限制和成功判据，并与普通 locomotion reset 分开统计。

增强 reset 必须继续保留 Fudan 的六维 root velocity 随机化，除非某个 recovery mode 明确覆写；不能在加入
pose 随机化时意外退化成 V3 当前仅 x/y 速度随机。`default_dof_pos` 的 startup 域随机化与每次 episode 的
joint reset offset 必须使用两个独立 sample，并分别保存供 critic/日志审计，避免重复随机化后无法还原来源。

heading 与随机 yaw 应协同上线：reset 后 heading target 应以机器人当前 yaw 为参考采样或显式采样世界目标，
不能沿用 reset 前的 target。Jump 在完成基线几何和 air-time bug 消融前，不启用 `RecoveryReset`，防止把
reset 分布变化误判为跳跃奖励修复的效果。

### 6.2 Termination

Fudan Plane/Jump 的 `terminate_after_contacts_on=[]`，因此 WYW 不使用接触终止。失败条件只有：

```python
is_bad_orientation = projected_gravity_z > -0.1
```

该条件必须连续保持 1.0 s 才 terminal；中途恢复则 persistence counter 清零。episode 超过 20 s 为 timeout。
Rough 越过全局 terrain boundary 也分类为 timeout，不是 terminal。timeout 必须通过 `extras["time_outs"]`
传给算法，以得到正确 bootstrapping。

V3 的 NaN/Inf、仿真状态非法等 numerical-safety termination 作为引擎安全设施保留，但应单独统计，不能
伪装成 Fudan 行为失败。不要继续使用 V3 的 roll/pitch 40 度即时终止、base 接触终止或 termination
reward；Fudan 的有效 reward scales 中没有 termination 项。

## 7. 奖励公式和聚合

### 7.1 强制聚合顺序

V3 当前奖励求和没有 Fudan 的逐项裁剪，因此 WYW 必须覆写 reward aggregation 或提供等价 hook。
对每个非 termination 项 `i`：

```python
weighted_i = raw_i * weight_i * step_dt
clipped_i = clamp(
    weighted_i,
    -clip_single_reward * step_dt,
    +clip_single_reward * step_dt,
)
total = sum(clipped_i)
if only_positive_rewards:
    total = clamp_min(total, 0.0)
total += termination_reward  # 本任务未启用
```

Flat/Rough 的 `clip_single_reward=1.0`，Jump 为 2.5，三者均
`only_positive_rewards=False`。裁剪发生在乘权重和 `step_dt` 之后、求和之前；不能裁 raw term、不能对总和
统一裁剪。episode sums 累计的是逐项裁剪后的值。

### 7.2 通用定义

以下符号均为 clean 仿真量：`v_b`/`omega_b` 是 body-frame base 速度，`g_b` 是 projected gravity，
`q`、`qdot`、`qddot`、`tau` 使用 §2.1 顺序，`c=[vx_cmd,yaw_cmd,height_cmd]`，`sigma=0.25`。

Plane 和 Jump 都必须使用 WYW 自有的精确公式 term；仅当 pure-tensor golden test 证明公式完全相同时，
才可复用 V3 现成 term。尤其不能加入 pitch 投影、standing gate、方向 gate、特殊模式或其他 V3 条件。

| term                         | raw formula                                                   |
| ---------------------------- | ------------------------------------------------------------- |
| `tracking_lin_vel`         | Plane:`exp(-(c0-v_b.x)^2/sigma)`；Jump: 前式 `* 2`        |
| `tracking_lin_vel_enhance` | Plane:`exp(-(c0-v_b.x)^2/(10*sigma))-1`；Jump: 前式 `* 2` |
| `tracking_ang_vel`         | `exp(-(c1-omega_b.z)^2/sigma)`                              |
| `tracking_ang_vel_enhance` | `exp(-(c1-omega_b.z)^2/(10*sigma))-1`                       |
| `orientation`              | `sum(square(g_b.xy))`                                       |
| `ang_vel_xy`               | `sum(square(omega_b.xy))`                                   |
| `torques`                  | `sum(square(tau))`，六个动作关节                            |
| `action_rate`              | `sum(square(a_t-a_{t-1}))`，六维                            |
| `collision`                | 指定 body 中接触力 norm > 0.1 N 的数量                        |

`tracking_lin_vel` 直接使用 body-frame `v_b.x`，不做 pitch projection 或额外门控。

### 7.3 Plane（Flat/Rough）

| term                         |  weight | raw formula / 范围                                    |
| ---------------------------- | ------: | ----------------------------------------------------- |
| `tracking_lin_vel`         |     1.0 | §7.2                                                 |
| `tracking_lin_vel_enhance` |     1.0 | §7.2                                                 |
| `tracking_ang_vel`         |     1.0 | §7.2                                                 |
| `tracking_ang_vel_enhance` |     1.0 | §7.2                                                 |
| `base_height`              |     1.0 | `exp(-(base_height-c2)^2/0.001)`                    |
| `nominal_state`            |    -1.0 | `(theta0_left-theta0_right)^2`                      |
| `lin_vel_z`                |    -1.0 | `v_b.z^2`                                           |
| `ang_vel_xy`               |   -0.20 | §7.2                                                 |
| `orientation`              |  -100.0 | §7.2                                                 |
| `dof_vel`                  |   -5e-5 | `sum(square(qdot[LEG_ACTION_IDS]))`                 |
| `dof_acc`                  | -2.5e-7 | `sum(square(qddot))`，六维                          |
| `torques`                  |   -1e-4 | §7.2                                                 |
| `action_rate`              |   -0.01 | §7.2                                                 |
| `action_smooth`            |   -0.01 | `sum(square(a_t-2a_{t-1}+a_{t-2})[LEG_ACTION_IDS])` |
| `collision`                |    -1.0 | Fudan Plane body 集为空，因此 raw 恒为 0              |
| `dof_pos_limits`           |    -1.0 | 四个腿关节超出 97% soft limit 的线性距离之和          |

`base_height` 使用 Fudan 环境中的 base-to-local-ground height，与命令高度追踪；不能替换成 V3 的其他
height reward。soft limit 以每个关节 hard limit 的中心和 97% 半范围构造。

### 7.4 Jump

Jump 只包含下表项目，不包含 `tracking_ang_vel_enhance`、`base_height`、`lin_vel_z`、`dof_vel`、
`dof_acc`、`action_smooth` 或 `dof_pos_limits`。

| term                         | weight | raw formula                                                            |
| ---------------------------- | -----: | ---------------------------------------------------------------------- |
| `tracking_lin_vel`         |    1.0 | §7.2 Jump 公式                                                        |
| `tracking_lin_vel_enhance` |    1.0 | §7.2 Jump 公式                                                        |
| `tracking_ang_vel`         |    1.0 | §7.2                                                                  |
| `flight`                   |   0.15 | `in_flight.float()`                                                  |
| `encourage_jump`           |    1.0 | 下述有状态公式                                                         |
| `base_height_flight`       |    6.0 | `exp(-abs(root_z-H_flight)*6) * in_flight`                           |
| `leg_tuck`                 |    1.7 | `exp(-4*sum(abs(L0-L_tuck))) * in_flight`                            |
| `takeoff_extend`           |    0.5 | `exp(-4*sum(abs(L0-L_extend))) * any(contact_filt) * (root_vz>0.15)` |
| `line_z`                   |    6.0 | `max(root_vz,0) * in_flight`                                         |
| `pen_theta_no0`            |   -2.0 | `sum(square(theta0))`                                                |
| `action_rate`              |  -0.04 | §7.2                                                                  |
| `torques`                  |  -5e-5 | §7.2                                                                  |
| `orientation`              |  -25.0 | §7.2                                                                  |
| `ang_vel_xy`               |  -0.10 | §7.2                                                                  |
| `nominal_state`            |   -1.0 | `(thetaL-thetaR)^2 + 10*(L0L-L0R)^2`                                 |
| `collision`                |   -1.0 | 下述 FDU body 映射                                                     |

Jump 接触状态严格按两个 wheel 的世界系竖直接触力计算：

```python
contact_now = wheel_contact_force_z > 1.0
contact_filt = contact_now | contact_previous_frame
in_flight = all(~contact_filt, dim=wheel)
contact_previous_frame = contact_now
```

`encourage_jump` 是有状态项，不能简化为瞬时高度或常规 airtime。第一版迁移必须精确保留已训练
Fudan 快照中的以下实现：

```python
first_contact = (base_air_time > 0) & (~in_flight)
base_air_time += step_dt * clamp(root_z, 0.0, 0.5)
reward = (base_air_time - 5e-5) * first_contact * 0.15
reward += max(root_vz, 0.0) * 0.15
base_air_time *= ~in_flight
```

> **已知兼容 bug，第一版有意保留：**`in_flight=True` 表示左右车轮连续两帧无接触，但源码使用
> `base_air_time *= ~in_flight`，因此 accumulator 在滞空时清零、接地时累计；`first_contact` 在接地的
> 第二步之后会持续为真，并不只在首次落地为真。已训练快照也包含同一行，说明现有跳跃行为主要由
> `line_z`、`flight`、`base_height_flight`、`leg_tuck`、`takeoff_extend` 及本项的正 `root_vz` 分量驱动。
> 为复现基线，本次不“顺手修正”为 `*= in_flight`。

这里同时保留 Fudan 的实际执行顺序：先根据旧 accumulator 判断 `first_contact`，再无条件累加本步
`clamp(root_z, 0, 0.5)`，最后用错误方向的 mask 处理 accumulator。Fudan reset 路径没有显式清零
`base_air_time`，第一版兼容实现也不额外清零；它只会在检测为滞空时被上述 mask 清零。该跨 episode
行为必须进入 golden test，避免以后重构时无意改变基线。

Fudan Jump 的 collision body 为两侧非轮腿体和 base。FDU 映射为：

```text
base_link_del
[lr]f[01]_Link, [lr]2[0-3]_Link
```

不包含 `[lr]_wheel_Link`，也不把这些 body 设为 termination contact。初始化时记录正则解析后的精确名称，
测试集合中不得意外包含云台或车轮。

## 8. 域随机化和材料

| 随机化                        | Flat/Rough            | Jump                               |
| ----------------------------- | --------------------- | ---------------------------------- |
| robot friction                | `[0.6, 1.4]`        | `[0.1, 2.0]`                     |
| robot restitution             | `[0.6, 1.0]`        | `[0.5, 1.0]`                     |
| added base mass               | `[-1, 2]` kg        | `[-2, 3]` kg                     |
| base COM offset each axis     | `[-0.02, 0.02]` m   | `[-0.05, 0.05]` m                |
| inertia multiplier            | `[0.9, 1.1]`        | `[0.8, 1.2]`                     |
| Kp/Kd/motor torque multiplier | `[0.95, 1.05]`      | `[0.9, 1.1]`                     |
| default joint position offset | `[-0.03, 0.03]` rad | `[-0.05, 0.05]` rad              |
| push                          | off                   | every 5 s，max XY velocity 1.5 m/s |

随机化样本必须 per-env 保存，既用于实际施加，也用于 critic privilege；不能事后从物理张量均值反推。
default position、Kp/Kd 和 torque multiplier 只作用于六个驱动关节，并遵循 §2.1 顺序。

Fudan terrain material 固定 static/dynamic friction=0.5、restitution=0.5；每环境 robot shapes 直接使用其
采样标量。Isaac Lab 中显式使用与 PhysX 默认接触材料组合等价的 `average` combine mode，不沿用 V3 当前
`multiply` 设置。加入最小物理验证：以已知 terrain/robot 参数建立接触，确认仿真实际组合结果符合
预期；若 Isaac Sim 当前版本的默认规则与 `average` 不同，以该测试结果修正文档和配置，不静默猜测。

Play 变体可关闭 push、质量/COM/惯量和控制器随机化以便观察，但不能改变动作/观测顺序、奖励公式或
终止语义。Play 的具体弱化项必须作为独立 config 显式列出。

## 9. Rough 地形和课程

Rough 激活 Fudan 配置中原本休眠的 trimesh 规格。六类地形及比例为：

| 类型               | 比例 |
| ------------------ | ---: |
| flat               |  0.2 |
| smooth slope       |  0.2 |
| rough slope        |  0.2 |
| down stairs        |  0.1 |
| up stairs          |  0.2 |
| discrete obstacles |  0.1 |

注意：Fudan 的数组顺序和源码内地形类型注释存在容易误读之处；WYW 使用具名 sub-terrain 字典表达上述
类别和比例，并通过采样统计测试验证，不再传匿名比例数组。

公共参数：tile 8x8 m、`num_rows=10`、`num_cols=20`、`max_init_terrain_level=5`、
`horizontal_scale=0.1`、`vertical_scale=0.005`、`border_width=25`、`slope_threshold=0.75`。迁移字段名时
使用 Isaac Lab 的 `border_width` 和 `slope_threshold`，不要照抄 Fudan 拼写 `border_size`、
`slope_treshold`。

随 difficulty `d` 的幅度为：

```text
slope        = 0.5 * d
random_height = 0.05 + 0.05 * d
step_height   = 0.05 + 0.18 * d
discrete_h    = 0.05 + 0.10 * d
```

课程在环境 reset 时按 episode 表现更新：

```python
distance = norm(root_xy - tile_origin_xy)
move_up = distance > terrain_length / 4
move_down = (
    episode_tracking_lin_vel / episode_length_s
    < (tracking_lin_vel_weight / step_dt) * 0.4
) & (~move_up)
level += move_up - move_down
```

达到或超过最高 level 的成功环境重新随机分配有效 level；低于 0 的环境夹到 0。随后用新的 level/type
更新 `env_origins`。边界 reset 必须计入 timeout。课程依据每环境表现，不依据 runner iteration。

## 10. 实现落点和代码规范

建议保持 `source/agent_tasks/agent_tasks/direct/wheelbipe/wyw/` 内职责清晰：

- `env.py`：WYW 生命周期覆写、观测构造、精确 reward terms、reward aggregation、reset/termination。
- `env_cfg.py`：WYW base/Flat/Rough/Jump/Play configclass 和所有可序列化字段。
- `joint_map.py`：唯一具名关节/body 解析及断言；若代码量很小可并入 `env.py`。
- `rough_cfg.py`：具名 terrain generator 配置和性能课程 helper。
- `rewards.py`：无环境副作用的 pure-tensor 公式；Jump 状态更新仍由 env 控制。
- `wyw_constants.py`：只保留维度和不可配置的字段布局，不保存机器人选择或任务参数。
- `__init__.py`：六个 Gym 注册项和各任务 runner config。

实现要求：

1. 一个配置来源：同一数值不同时出现在 constants、config 和 env 分支中。
2. 语义索引：leg/wheel/left/right 全由具名映射产生；奖励中不写魔法切片。
3. Shape 断言：policy、history、critic、action 在构造点断言 25/125/141/6。
4. 设备与 dtype：新 tensor 从现有 tensor 派生或显式指定 device/dtype，避免循环内创建 CPU scalar tensor。
5. 无静默补齐：critic 77 点、body 集合、关节集合解析不符时直接报错，不允许 pad/truncate 掩盖配置错误。
6. 副作用清晰：raw reward helper 除 `encourage_jump` 状态机外均为纯函数；episode logging 不改变训练状态。
7. Play/Train 共用核心契约：只通过配置改变随机化和 env 数，不复制一套 observation/reward 代码。
8. 配置落盘：动作顺序、body 映射、噪声、阈值、几何目标、combine mode 和课程参数均可在 YAML 审计。

## 11. 验证与验收标准

### 11.1 不依赖仿真的 golden tests

至少覆盖：

- 六动作具名顺序、`LEG_ACTION_IDS=[0,1,3,4]` 和所有 25/141 维 segment 顺序。
- 给定相同 clean input 和固定噪声，critic 内 25-D 段保持 clean，actor 及 history 为 noisy。
- 三连续动作下，observation、`action_rate`、`action_smooth`、critic 两帧动作历史的时序。
- Plane/Jump 每个 raw formula，与 Fudan 代码在固定 tensor fixture 上逐项相等。
- 奖励流程“乘 weight/dt -> 单项裁剪 -> 求和”的边界样例，验证不能被总和裁剪替代。
- 1 秒 failure persistence：连续 100 个 policy step 后触发，中途恢复清零。
- reset 分布、关节速度为零、Flat/Jump XY 固定、Rough XY 在 `[-1,1]`。
- command 直接 yaw-rate 模式、范围及严格 5/20 s 重采样。
- DR privilege 的质量中心化、COM offset、default delta、per-env friction/restitution。
- Jump 两帧 contact filter，以及兼容 bug 下 `base_air_time` 接地累计、滞空清零、跨 reset 保留的行为。
- 六类 terrain 比例、难度幅度、move-up/down 和最高 level 随机回绕。
- Rough boundary reset 被标为 timeout，非 terminal。

### 11.2 仿真集成测试

1. 注册：六个 task ID 可创建，三主任务写入独立实验目录。
2. 单环境：解析出的 joint/body 名称与 §2/§7.4 完全一致；云台不进入 action。
3. 观测：Flat/Rough/Jump 均为 policy 25、history 125、critic 141；扫描非空且公式用 root z。
4. 材料：验证 terrain/robot combine mode 和 critic 中保存的 sample 一致。
5. Reset/termination：通过人工设置状态验证 1 秒倾倒宽限、20 秒 timeout 和 Rough boundary timeout。
6. 冒烟目标：Flat、Rough、Jump 各 64 env、至少 3 iterations，无 NaN/shape error；这一步不能替代 golden tests。
7. YAML 审计：确认无 heading/standing/special mode、无 obs/action delay、奖励集合和 clip 值准确，
   `height_range=[0.20,0.42]`、`height_cmd_scale=1.0`。

验收不是“可以训练”或“曲线有上升”，而是上述张量、公式、时序和配置测试全部通过。之后才进入性能
调参；性能调参产生的有意偏离必须以新配置版本记录，不回改迁移基线。

### 11.3 当前验收记录（2026-08-28）

| 项目 | 实际运行 | 结果 |
| --- | --- | --- |
| FDU asset 标定 | `scripts/test_fdu_cfg.py --headless` | 通过：14 DOF、15 刚体、状态有限 |
| Flat GUI Play | `scripts/view_robot.py`，宿主 `DISPLAY=:1` | 通过：闭链显示、无明显振动 |
| Flat PPO | 16 env、1 iteration、seed 42、非无头 | 通过：GPU/Vulkan、网络维度、loss/KL 有限 |
| Rough 语义冒烟 | 4 env、headless、seed 42 | 通过：reset/obs/reward/termination fixture |
| Rough PPO | 16 env、2 iterations、seed 42、非无头 | 通过：课程地形生成、网络维度、loss/KL 有限 |
| Jump 语义冒烟 | 2 env、headless、seed 42、宿主 GPU | 通过：Jump reward、50 N m 轮力矩、20 s 命令周期 |
| Jump PPO | 16 env、2 iterations、seed 42、非无头 | 通过：GPU/Vulkan、encoder/actor/critic、loss/KL 有限 |

上述 PPO 运行均未出现 NaN/Inf 或崩溃；2 iterations 约覆盖 0.96 s，因而日志中的
`terminate=0/time_out=0` 不能证明 1 s 姿态持续终止或 20 s timeout 已在自然 rollout 中触发。
语义冒烟通过人工 fixture 单独验证了 100 个 policy step 的姿态持续终止和 timeout 边界。
Rough 课程的跨 episode 晋级/降级仍需一次覆盖 reset 的长时运行；目标门槛“64 env、3 iterations”也尚未执行。

运行环境说明：本代理沙箱没有 CUDA 设备，Jump headless 在沙箱内会报 `No CUDA GPUs are available`；
同一命令在宿主 RTX 3070、`DISPLAY=:1`/CUDA 访问下已通过。PCIe 链宽、IOMMU、TGS velocity iterations、
缺失 STL visual 子路径和 Direct RL manager visualizer warning 均未导致本次验收失败。

## 12. 已知风险和标定项

以下问题不阻塞实现正确的迁移骨架，但在正式长训前必须完成：

1. **后续提醒：修正 Jump air-time bug 做消融**。第一版基线保留 `base_air_time *= ~in_flight` 和
   跨 reset 状态；完成 Fudan 行为对齐后，新增独立实验改为“滞空累计、落地奖励后清零”，比较跳跃频率、
   滞空时间、落地稳定性和总回报。修正版必须使用新实验名，不能覆盖兼容基线。
2. **腿根偏移**：从 USD/仿真读取左右 hip offset，确认 §2.2 的几何原点。不能继续默认 hip x/z=0。
3. **L0 目标**：Fudan 的 `L_tuck=0.16` m、`L_extend=0.31` m 只可作为临时占位；用 FDU 可达
   腿长范围标定后设置 WYW config。
4. **滞空高度**：Fudan `H_flight=0.65` m；FDU 当前使用 0.65 m，仍需用 base root z 分布而非视觉估计复核。
5. **Jump PD**：FDU Jump 使用 Kp=6/Kd=0.5；在固定
   动作下测量可达腿长、峰值力矩、起跳速度和接触稳定性，再决定 Jump 专用 PD。
6. **动作方向**：左右 front/rear 关节的正方向可能与 Fudan 镜像定义不同。动作顺序已经确定，但仍需
   单关节正动作测试确认 `theta0` 和 `L0` 变化方向；必要时用显式 sign map，不改变网络数组顺序。
7. **材料组合**：`average` 是基于 PhysX 默认语义的迁移选择，最终以当前 Isaac Sim 版本的物理验证为准。

上述标定结果必须写入 configclass、测试 fixture 和一次标定记录，不能只改模块常量或留在训练命令行中。
