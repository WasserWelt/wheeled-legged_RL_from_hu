# WYW FDU 当前配置核对表

> 更新时间：2026-08-28。本表记录当前实装；目标语义和已知风险以
> [`intention_v3.md`](./intention_v3.md) 为准。

## 任务注册

| Task ID | 配置 | 地形 | 奖励 |
| --- | --- | --- | --- |
| `Robotics-Wheelbipe-FDU-wyw-Flat-v1` | `WheelbipeWywFlatEnvCfg` | flat USD | Fudan Plane |
| `Robotics-Wheelbipe-FDU-wyw-Flat-Play-v1` | `..._Play` | flat USD | Fudan Plane |
| `Robotics-Wheelbipe-FDU-wyw-Rough-v1` | `WheelbipeWywRoughEnvCfg` | trimesh | Fudan Plane |
| `Robotics-Wheelbipe-FDU-wyw-Rough-Play-v1` | `..._Play` | trimesh | Fudan Plane |
| `Robotics-Wheelbipe-FDU-wyw-Jump-v1` | `WheelbipeWywJumpEnvCfg` | flat USD | Fudan Jump |
| `Robotics-Wheelbipe-FDU-wyw-Jump-Play-v1` | `..._Play` | flat USD | Fudan Jump |

## 机器人和控制

- 权威输入：`robot_models/fdu_infantry_V4_mujoco/meshes/infantry_V2.urdf`。
- 资产：`Wheelbipe_FDU_CFG`，14 articulation DOF、15 rigid bodies，根体 `base_link_del`。
- 闭链：四个 spherical loop constraint 排除在 reduced-coordinate articulation 外。
- 策略顺序：`lf0_Joint, l20_Joint, l_wheel_Joint, rf0_Joint, r20_Joint, r_wheel_Joint`。
- 四个实体驱动杆使用位置目标：`default + 0.5 * action`。
- 两个车轮使用速度目标：`10.0 * action`，WYW adapter 上限为 `60 rad/s`；FDU asset 的 `velocity_limit`、`velocity_limit_sim`、WYW 的 `max_wheel_vel` 和运行时 clamp 均保持一致。Fudan 原训练与闭链 MuJoCo sim2sim 都没有 wheel target speed clamp，而是以 `vel_ref=10*action` 形成速度误差，再通过阻尼增益产生并裁剪轮力矩。这里的 `60 rad/s` 是当前 velocity-target adapter 的有意边界：轮半径约 0.06 m，训练最大前进命令 2.1 m/s 对应 35 rad/s，最大偏航命令 2 rad/s、半轮距约 0.1877 m 对单轮再贡献约 6.3 rad/s，极端组合约 41.3 rad/s，60 rad/s 留有约 45% 裕量。它不是原始 Isaac Gym URDF 的 `1500`，也不是由 `clip_actions=100` 推算。
- Flat/Rough 腿 PD：Kp=20、Kd=1；Jump：Kp=6、Kd=0.5。
- 腿/轮仿真 effort limit 均为 30 N m。被动闭链关节 Kp=0、Kd=0.01。

这些是仿真控制参数，不是实际电机额定值。真实力矩常数、减速比、电流限幅、编码器零点和方向仍需硬件辨识。

## 时间与张量契约

| 项 | 值 |
| --- | ---: |
| physics dt | 0.005 s |
| decimation | 2 |
| policy rate | 100 Hz |
| action | 6 |
| actor observation | 25 |
| actor history | 5 x 25 = 125 |
| critic observation | 141 |
| encoder latent | 3 |

Jump 几何/接触配置：`l0_tuck=0.16 m`、`l0_extend=0.31 m`、`base_height_flight=0.65 m`、
`takeoff_vz=0.15 m/s`、`flight_contact_force=1.0 N`；这些均为 configclass 字段并写入 YAML。

Actor 25 维为 `ang_vel(3), projected_gravity(3), command(vx,yaw,height)(3), leg_pos(4), dof_vel(6), action(6)`。
Critic 141 维为 `base_lin_vel(3), actor_obs(25), previous_actions(6), before_previous_actions(6), dof_acc(6), heights(77), torque(6), DR(12)`。
Train actor/history 对 ang_vel、gravity、joint position/velocity 加 Fudan 幅度的均匀噪声；critic 内的 25-D
proprioception 使用同一步 clean 副本。Play 关闭观测噪声。obs/action delay 均关闭。

## 奖励

Flat/Rough 精确使用以下 Fudan Plane 项：

```text
tracking_lin_vel, tracking_lin_vel_enhance,
tracking_ang_vel, tracking_ang_vel_enhance,
base_height, nominal_state, lin_vel_z, ang_vel_xy, orientation,
dof_vel, dof_acc, torques, action_rate, action_smooth,
collision, dof_pos_limits
```

Jump 精确使用以下 Fudan Jump 项，不叠加 V3 奖励：

```text
tracking_lin_vel, tracking_lin_vel_enhance, tracking_ang_vel,
flight, encourage_jump, base_height_flight, leg_tuck,
takeoff_extend, line_z, pen_theta_no0, action_rate, torques,
orientation, ang_vel_xy, nominal_state, collision
```

Flat/Rough `clip_single_reward=1.0`，Jump 为 2.5；裁剪发生在 `weight * step_dt` 后、求和前。
Jump 有意保留 Fudan `base_air_time *= ~in_flight` bug 及跨 reset 状态，后续必须用独立实验做修正版消融。

## 域随机化

| 项 | Flat/Rough | Jump |
| --- | --- | --- |
| robot friction | [0.6, 1.4] | [0.1, 2.0] |
| robot restitution | [0.6, 1.0] | [0.5, 1.0] |
| added base mass | [-1, 2] kg | [-2, 3] kg |
| per-body mass/inertia scale | [0.9, 1.1] | [0.8, 1.2] |
| base COM offset | xyz +/-0.02 m | xyz +/-0.05 m |
| Kp/Kd/torque scale | [0.95, 1.05] | [0.9, 1.1] |
| default joint offset | +/-0.03 rad | +/-0.05 rad |
| push | off | 5 s, XY velocity +/-1.5 m/s |

Robot 每个环境使用一组 friction/restitution；terrain material 固定为 0.5/0.5，combine mode 为 `average`。
采样的 base mass addition、default offset、friction 和 restitution 直接写入 critic privilege。Play 关闭上述
startup 随机化和 push，只保留 reset pose event。

## 当前验证

- FDU asset CPU 标定：14 DOF、15 bodies、根质量 11.718 kg、短步状态有限。
- Flat/Rough/Jump CPU 单环境：reset + 两次 step 均通过，25/125/141 shape 和 reward finite 均通过。
- Rough 的左右轮扫描器已重绑定到 `r_wheel_Link` / `l_wheel_Link`。
- pure-tensor mapping/air-time tests：4 passed。

详细质量、惯量和 joint limit 快照见 [`fdu_calibration_report.json`](./fdu_calibration_report.json)。
