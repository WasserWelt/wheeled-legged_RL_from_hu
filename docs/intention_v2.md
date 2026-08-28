# wyw 任务族设计（intention_v2）

> 本文是 wyw 任务族（Flat / Rough / Jump + Play）的权威设计说明。
> 配套核对表见 [`wyw_config_table.md`](./wyw_config_table.md)（记录 as-built 逐项配置）。

---

## 0. 一句话设计原则

**除机器人本身之外，wyw 的全部训练要点（观测 / 奖励 / 事件·域随机化 / 命令）都与 fudan
（`fudan_rl_wheel_leg`）一致。**

- 这里的"一致"指**效果一致**，不要求代码逐行相同：只要训练出的行为目标、观测语义、奖励导向、
  随机化强度与 fudan 等价即可。实现上以 IsaacLab（DirectRLEnv + configclass + manager MDP terms）
  风格落地，复用本仓库既有引擎设施，**不**照搬 fudan 的 isaacgym/legged_gym 写法。
- **唯一的差异来源是机器人本体**：wyw 用本仓库 world 层的 `Wheelbipe_V14_2_CFG`（from_hu 机器人，
  含云台），它与 fudan 的机器人结构不同。凡是**与机器人结构绑定**的量（驱动关节、名义位姿、腿部
  PD、动作缩放、高度命令量纲、腿长/腿角几何）都要从 fudan 的语义**重新映射到本机器人**——这些是
  不显眼但关键的点，见 §3。

---

## 1. 架构：建在共享引擎上的独立任务

wyw 是一个**完全独立的任务族**，直接建在底层仿真引擎 `Wheelbipe25V3Env`（`DirectRLEnv` 子类）之上，
与整个 `WheelbipeV13Env` / `WheelbipeV14Env` 任务血缘**平级、互不相关**。

```
DirectRLEnv
 └─ Wheelbipe25V3Env        ← 共享底层引擎（关节发现/PD控制/接触/复位随机化/obs延迟/高度扫描/奖励求和）
     ├─ WheelbipeV13Env → WheelbipeV14Env   （既有任务血缘，wyw 不再继承）
     └─ WheelbipeWywEnv                      （wyw：独立任务，配置=fudan，本体=V14_2 机器人）
```

- **为什么建在 `Wheelbipe25V3Env` 上**：它不是"某个 wheelbipe 任务"，而是本机器人类的**底层仿真引擎**
  （约 5700 行：按正则发现关节、IdealPD/速度控制、接触传感、复位随机化、观测延迟/噪声、地形高度扫描、
  `_get_rewards` 奖励求和循环）。wyw 复用引擎是复用基础设施，不是复用任务逻辑。
- **云台按被动处理**：`Wheelbipe_V14_2_CFG` 带 `gimbal_yaw/gimbal_pitch` 两个关节，但它们不在训练驱动
  关节集合内。引擎每次复位会把所有关节写回默认状态（gimbal→0），IdealPD 使 pitch(Kp=20) 保持水平、
  yaw(Kp=0,Kd=0.5,力矩≤2N·m) 阻尼自由——稳定无害，不需要 V14 的云台驱动代码。
- **Rough 需要、但只在 V14 里实现的两个行为**，复制进 wyw 自有代码（见 §4）：地形高度偏移课程、
  地形边界超时复位。V14 的小陀螺(gimbal-spin)、按地形偏置命令(V13 `TerrainCommandManager`)、
  velocity-trace 调试等**不复制**（fudan 无对应，也不属 wyw 需求）。

---

## 2. 训练要点 = fudan（效果一致）

以下均以 fudan 为目标；表中数值是**要达到的效果目标**，实现时映射到本仓库设施。

### 2.1 观测（与 fudan 逐段等价）

|                      | 维度          | 内容（fudan 语义）                                                                                                                                                    |
| -------------------- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Policy（actor）      | **25**  | 机身角速度(3) + 投影重力(3) + 命令[vx, yaw, height](3) + 腿关节位置(4) + 关节速度(6) + 上一步动作(6)                                                                   |
| 历史（encoder 输入） | **125** | 5 帧 × 25（隐式基座线速度估计的输入）                                                                                                                                |
| Critic（特权）       | **141** | base_lin_vel(3) + 本体观测(25) + 上一步/上上步动作(6+6) + 关节加速度(6) + 地形高度扫描(77) + 关节力矩(6) + 域随机化特权[质量偏差1/质心3/默认关节偏差6/摩擦1/恢复1](12) |

- `base_lin_vel` 必须是 critic 前 3 维：供 `PPOSequence` 的 encoder 做隐式线速度监督
  （`MSE(latent[:, :3], critic_obs[:, :3])`）。
- 网络/PPO 三件套：`OnPolicySequenceRunner` / `ActorCriticSequence` / `PPOSequence`，latent_dim=3，
  actor [128,64,32]、critic [256,128,64]、encoder [128,64]，其余 PPO 超参见 `wyw_config_table.md` §1。
- 观测缩放作为 configclass 字段（`wyw_*_scale`），随 `params/env.yaml` 落盘、可按任务覆写。

### 2.2 奖励（= fudan）

奖励机制沿用引擎 `_get_rewards`：每项 `× step_dt × 权重`，权重来自 `cfg.rewards`（`OrderedDict`）。
下表是 fudan 的奖励集合与权重（效果目标）；实现时能对应到引擎现成奖励项的直接用，fudan 特有、引擎无
等价的项（`*_enhance`、`nominal_state`、`pen_theta_no0` 及跳跃项）由 wyw 在 `_postprocess_reward_terms`
里自算后注入——对训练者而言，效果就是"和 fudan 一样"。

**Plane（Flat / Rough 共用）** — fudan `plane/.../legged_robot_config.py:183-215`：

| 项                       | 权重  |  | 项             | 权重    |
| ------------------------ | ----- | - | -------------- | ------- |
| tracking_lin_vel         | 1.0   |  | orientation    | -100.0  |
| tracking_lin_vel_enhance | 1.0   |  | dof_vel        | -5e-5   |
| tracking_ang_vel         | 1.0   |  | dof_acc        | -2.5e-7 |
| tracking_ang_vel_enhance | 1.0   |  | torques        | -1e-4   |
| base_height              | 1.0   |  | action_rate    | -0.01   |
| nominal_state            | -1.0  |  | action_smooth  | -0.01   |
| lin_vel_z                | -1.0  |  | collision      | -1.0    |
| ang_vel_xy               | -0.20 |  | dof_pos_limits | -1.0    |

超参：`tracking_sigma=0.25`、`only_positive_rewards=False`、`clip_single_reward=1`、
`max_contact_force=100`。
`*_enhance` = `exp(-err/σ/10) - 1`；`nominal_state`(plane) = `square(θ0ₗ - θ0ᵣ)`（左右腿摆角对称）。
`base_height` 奖励**追踪当前高度命令**（`track_height_exp` 目标 = 命令第 3 维 height，在 [0.20,0.42]
间采样），**不用** fudan 的固定站高目标 0.18——fudan 高度本就是命令，wyw 一致地按命令追踪。

**虚拟腿 L0 / θ0（腿根→轮 joint 的虚拟杆长度与摆角）** —— 只进奖励、**不进观测**（与 fudan 一致；
fudan 策略只看原始腿关节角）：

- fudan 由两腿关节经串联双连杆正运动学算 `L0=√(x²+y²)`、`θ0=atan2(y,x)−π/2`（**机身系**，正下方=0）。
- wyw 新增 helper `_get_wyw_leg_L0_theta0()`：直接读仿真 `left/right_wheel_link` 实际位姿
  （`_get_root_quat_inv_and_wheel_pos_b`），**避开五连杆闭链的解析求解**。
  - `L0 = ‖wheel_pos_heading_b‖`（矢状面，y 清零）。
  - `θ0 = atan2(-wheel_pos_b_x, -wheel_pos_b_z)`，用**机身系** `wheel_pos_b`（非 heading/重力系），
    即腿相对机身自身竖直的摆角、正下方=0，**严格对齐 fudan 机身系**。
- `nominal_state` / `pen_theta_no0` / 跳跃 `leg_tuck` / `takeoff_extend` 均复用此 helper。

**Jump（在 Plane 上改）** — fudan `jump/.../legged_robot_config.py:183-221`：

- **加入**跳跃项：`flight=0.15`、`encourage_jump=1.0`、`base_height_flight=6.0`、`leg_tuck=1.7`、
  `takeoff_extend=0.5`、`line_z=6.0`、`pen_theta_no0=-2.0`（`sum(square(θ0))`）。
- `nominal_state`(jump) = `square(θ0ₗ-θ0ᵣ) + 10·square(L0ₗ-L0ᵣ)`（多加左右腿长非对称）。
- **移除**（jump 不含）：`tracking_ang_vel_enhance`、`base_height`、`lin_vel_z`、`dof_vel`、`dof_acc`、
  `action_smooth`、`dof_pos_limits`。
- **改权重**：orientation −100→−25、ang_vel_xy −0.20→−0.10、torques −1e-4→−5e-5、
  action_rate −0.01→−0.04、`clip_single_reward` 1→2.5。

跳跃奖励公式与门控（涌现式、无显式状态机）见 `wyw_config_table.md` §9；门控用车轮接触力峰值判定
滞空/触地。

### 2.3 事件 / 域随机化（= fudan `domain_rand`）

以 IsaacLab `@configclass EventCfg` + `EventTerm` 实现，复用既有 MDP 事件项。效果目标：

| fudan 随机化       | plane 范围   | jump 范围                          |
| ------------------ | ------------ | ---------------------------------- |
| 摩擦 friction      | [0.6, 1.4]   | [0.1, 2.0]                         |
| 恢复 restitution   | [0.6, 1.0]   | [0.5, 1.0]                         |
| 附加质量 base_mass | [-1, 2]      | [-2, 3]                            |
| 质心 com（每轴）   | ±0.02       | ±0.05                             |
| 惯量 inertia       | [0.9, 1.1]   | [0.8, 1.2]                         |
| Kp / Kd / 电机力矩 | [0.95, 1.05] | [0.9, 1.1]                         |
| 默认关节位抖动     | ±0.03       | ±0.05                             |
| 推力 push_robots   | **关** | **开**（间隔 5s，≤1.5 m/s） |

（plane push 间隔 7s / ≤2.0 m/s，但 plane `push_robots=False` 故不生效。）Play 变体弱化随机化。

### 2.4 命令（= fudan `commands`）

3 维命令 = (前进速度 vx, 偏航角速度 yaw, 机身高度 height)，**无侧向 vy**、无特殊模式（不引入
V14 的 `SpecialMode` 命令）。用 `mdp.UniformVelocityCommandCfg`，高度走引擎既有 height-cmd 机制。

|       | vx          | yaw     | 重采样 | 课程 |
| ----- | ----------- | ------- | ------ | ---- |
| plane | [-2.0, 2.0] | [-2, 2] | 5 s    | 开   |
| jump  | [-2.1, 2.1] | [-2, 2] | 20 s   | 关   |

（高度命令量纲是机器人结构绑定量，见 §3。）

### 2.5 控制频率

decimation=2、物理 dt=0.005 → 策略 100 Hz。与 fudan 一致。

---

## 3. 与机器人结构绑定的映射点（不显眼但关键）

fudan 机器人（2 串联腿关节/侧 + 轮）与本机器人 `Wheelbipe_V14_2`（五连杆腿 + 轮 + 云台）结构不同。
以下量**按语义等价**从 fudan 映射到本机器人，而非照抄 fudan 数值：

| 项                                                                        | fudan                                                | 本机器人（V14_2）映射                                                                                                                                                                                                                                                                                    |
| ------------------------------------------------------------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **训练驱动关节**（动作 6 维）                                       | 腿位置 lf0/lf1/rf0/rf1(4) + 轮速 l/r_wheel(2)        | 腿位置`.*_rear1_joint`/`.*_front1_joint`(4) + 轮速 `.*_wheel_joint`(2)；`_actuate_idx=legs_act+wheel`=6。**维度与"4 腿位置 + 2 轮速"结构等价**                                                                                                                                             |
| **腿关节观测**                                                      | 4 个腿关节位置                                       | `obs_joint_pos[:, :4]`（前 4 = 驱动腿关节）                                                                                                                                                                                                                                                            |
| **名义位姿**                                                        | lf0=0.2,lf1=0.4,rf0=−0.2,rf1=−0.4（镜像对称）      | 本体 USD 默认全 0；`nominal_state`/`pen_theta_no0` 用**实际连杆几何**算 θ0，与关节零点定义无关                                                                                                                                                                                                |
| **腿长 L0 / 腿角 θ0**（跳跃、nominal）                             | 由 asset.l1/l2(0.175/0.208) 串联 FK 算，θ0 在机身系 | L0=`‖wheel_pos_heading_b‖`、θ0=`atan2(-x,-z)` 用**机身系** `wheel_pos_b`（对齐 fudan 机身系）——**从仿真实际位姿读取，换 USD 自动正确**，避开五连杆闭链 FK。⚠️ L0 目标常量 `WYW_L0_TUCK/EXTEND`（fudan 0.16/0.31）是 fudan 腿几何，**待 view_robot 实测 from_hu 后重定** |
| **腿部 PD**                                                         | plane Kp=20/Kd=1，jump Kp=**6**/Kd=0.5（软）   | 本体 legs_act Kp=**60**/Kd=2（硬得多）。⚠️ **起跳软腿风险**：Jump 可能蹬不起来。默认先沿用本体 PD，训练观测后再决定是否给 Jump 单独调软（作为 robot cfg / cfg 字段覆写）                                                                                                                     |
| **动作缩放**                                                        | 腿位置 ×0.5、轮速 ×10.0                            | 效果对齐：腿位置增量缩放 `leg_action_scale=0.5`（`0.5·a + default_leg_pos`）、轮速控制 `wheel_vel_action_scale=10.0`（`use_wheel_vel_control=True`）。已确认为引擎默认，wyw 不覆写                                                                                                          |
| **高度命令量纲**                                                    | plane [0.10,0.20]/jump [0.12,0.15]（fudan 站高）     | **from_hu 骑行高度 [0.20, 0.42]**（用户定；高度是几何量，由本体决定），`default_height_cmd=0.22`                                                                                                                                                                                                 |
| **滞空期望机身高度**                                                | 0.65                                                 | from_hu 待实测初值 0.60（换 fudan USD 后 0.65）                                                                                                                                                                                                                                                          |

> 结论：动作维度（4 腿 + 2 轮）、观测语义、奖励导向都与 fudan 等价；差异全部收敛在"用本机器人的
> 实际关节/连杆/几何"上，且腿长/腿角均从仿真位姿读取，不写死杆长。

---

## 4. 三变体差异（= fudan plane/jump 差异）

|                       | Flat                     | Rough                                                                    | Jump                                       |
| --------------------- | ------------------------ | ------------------------------------------------------------------------ | ------------------------------------------ |
| 地形                  | 平地                     | trimesh`WYW_ROUGH_TERRAINS_CFG`（fudan 规格激活）+ 高度课程 + 边界复位 | 平地                                       |
| 奖励                  | plane（§2.2）           | plane                                                                    | plane + 跳跃项（§2.2 jump）               |
| 事件/命令             | fudan plane（§2.3/2.4） | fudan plane                                                              | fudan jump（push 开、vx±2.1、重采样 20s） |
| `wyw_lin_vel_scale` | 2.0                      | 2.0                                                                      | **3.0**（对齐 fudan jump obs_scale） |
| obs/critic/网络/PPO   | 三任务共享               |                                                                          |                                            |

**Rough 地形来自 fudan（激活其休眠规格）**：fudan 两变体实际都只在平地训练
（`plane`/`jump` 均 `mesh_type="plane"`），但配置里**配了一份地形生成规格却休眠**。wyw Rough =
把该规格**激活**为 Isaac Lab `TerrainGeneratorCfg`（trimesh），落为 wyw 自有常量
`WYW_ROUGH_TERRAINS_CFG`——**不用**本地 `RM_ROTATION_TERRAINS_CFG_99`。fudan 规格要点：

- 子地形比例 `[平滑坡, 粗糙坡, 上楼梯, 下楼梯, 离散障碍]=[0.2,0.2,0.2,0.1,0.2,0.1]`；tile 8×8m；
  `num_rows=10`（难度等级）、`num_cols=20`（类型）、`max_init_terrain_level=5`；
  `horizontal_scale=0.1`、`vertical_scale=0.005`、`border_size=25`、`slope_treshold=0.75`；
  地形摩擦 static/dynamic=0.5、restitution=0.5。
- 幅度随难度 d：`slope=0.5·d`、`random_height=0.05+0.05·d`、`step_height=0.05+0.18·d`、
  `discrete_h=0.05+0.1·d`。

**Rough 复制进 wyw 的两个 V14 专有行为**（引擎 `Wheelbipe25V3Env` 无）：

1. **地形高度偏移课程**：`_reset_idx` 内按训练进度改 `terrain_levels/types/env_origins`，逐步升难度。
2. **地形边界超时复位**：`_get_dones` 里把越界 OR 进 `time_out`（而非 terminate，不触发终止惩罚）。
   复制时**删去** V14 的 `terrain_command_overrides` / `TerrainCommandManager` 分支（fudan 无此机制）。

三变体均直接继承 wyw 基类（互不继承）：Flat/Rough/Jump 都建在 `WheelbipeWywBaseEnvCfg` 上。

---

## 5. 实现落点（`direct/wheelbipe/wyw/`）

- **`env.py`** `WheelbipeWywEnv(Wheelbipe25V3Env)`：reparent 到引擎；复制两个 rough 行为；保留现有
  obs 组装（policy/critic/policy_hist 覆写、`_wyw_obs_hist` 历史缓冲）；新增 `_get_wyw_leg_L0_theta0()`
  （θ0 机身系）；`_postprocess_reward_terms` 注入 fudan 特有奖励项（`*_enhance` /
  `nominal_state`(按 jump 切公式) / `pen_theta_no0` / 6 跳跃项）。
- **`env_cfg.py`**：
  - `WheelbipeWywBaseEnvCfg`（继承引擎 flat cfg）——声明本机器人接线（`robot_cfg=Wheelbipe_V14_2_CFG`、
    腿/轮/弹簧关节正则、腿部 PD/动作缩放、高度模式 `height_range=[0.20,0.42]`）+ fudan 奖励/事件/命令 +
    obs 形状（25/141）+ obs 缩放字段；`_apply_wyw_common` 收编为**基类方法**，`__post_init__` 末尾调用。
  - `WheelbipeWywFlatEnvCfg` / `WheelbipeWywRoughEnvCfg` / `WheelbipeWywJumpEnvCfg` 均直继 Base；
    Rough 调 wyw 自有 rough helper；Jump 置 `wyw_jump_enabled=True`、`wyw_lin_vel_scale=3.0`、jump 奖励/事件/命令。
  - Play 变体用弱化随机化的 `EventCfgWyw_Play`。
- **`wyw_constants.py`**：维度/几何/阈值 + 跳跃奖励权重（含 `pen_theta_no0`）+ wyw 自有 ordered 关节/连杆名单
  + `WYW_ROUGH_TERRAINS_CFG`（fudan 规格激活的 `TerrainGeneratorCfg`）+ L0 目标常量 `WYW_L0_TUCK/EXTEND`（待实测）。
- **`rough_cfg_utils.py`（新）**：从 V14 `_apply_v14_rough_runtime_cfg` 复制，删 terrain-command 分支；
  `WheelbipeWywRoughEnvCfg` 的地形 generator 指向 `WYW_ROUGH_TERRAINS_CFG`。
- **`__init__.py`**：6 个 `gym.register` 指向新 cfg，runner 三件套不变。

---

## 6. 验证

- `python scripts/list_envs.py`：6 个 wyw 任务注册正确。
- `python scripts/view_robot.py --task=<wyw-Flat> --num_envs=1 --device=cpu`：机器人（含被动云台）
  加载正常，关节映射无报错，云台稳定不发散。
- `train.py --task=<wyw-{Flat,Rough,Jump}> --num_envs=64 --max_iterations=3 --headless`：三任务冒烟；
  核对 `params/env.yaml`：obs 25/critic 141、decimation=2、命令范围=fudan、`height_range=[0.20,0.42]`、
  奖励集合=fudan（jump 含 `pen_theta_no0`）、事件数值=fudan（jump push 开）。
- Rough：`dot_scanner` 命中 generator 地形（高度扫描非恒 0）、边界超时复位生效、高度课程推进。

## 7. 风险 / 待核对

1. **起跳软腿**：本体 legs_act Kp=60 远硬于 fudan jump 的 6，Jump 或蹬不起来 → 训练后按需给 Jump 调软腿 PD。（暂时不处理)
2. **滞空期望高度** `WYW_BASE_HEIGHT_FLIGHT=0.60`（from_hu 待实测），换 fudan USD 后 0.65。（暂时不处理)
3. **L0 目标常量** `WYW_L0_TUCK=0.16` / `WYW_L0_EXTEND=0.31` 为 fudan 腿几何占位，**待 view_robot 实测 from_hu 腿长后重定**。
4. **髋偏移核实**：L0 用机身中心→轮距离近似"腿根→轮"，前提是 V14_2 髋 x/z≈0（仅横向 y=±0.217）；实现前须核对 USD，若非 0 则先减髋偏移。
5. **高度命令 obs 缩放**：wyw 用 1.0（fudan 实为 5.0，用户拍板 1.0，是**有意偏离**）——确认与部署端一致。注意地形高度扫描缩放 `WYW_HEIGHT_SCALE=5.0` 是**另一个量**（77 维、进 critic）：引擎实测**传感器到脚下地面点的垂直距离**（`pos_w.z − ray_hit.z`，随机身升降的相对量），先 `clamp(±1)` 再 ×5.0 归一化，保持不变。
6. **动作缩放（已确认）**：wyw 不在任务层设 action-scale，直接继承 V3 引擎默认——`leg_action_scale=0.5`（增量式位置：`0.5·a + default_leg_pos`）、`use_wheel_vel_control=True` 且 `wheel_vel_action_scale=10.0`（目标轮速 rad/s）。语义与 fudan"腿位置 ×0.5 / 轮速 ×10.0"一致。⚠️ 两处实现差异：腿是叠加**本体 USD 默认姿态**的增量（与 fudan default 数值无关）；轮速有引擎夹取 `max_wheel_vel=90 rad/s`（fudan 无，量级足够大通常不触发）。部署端沿用同缩放即可。
