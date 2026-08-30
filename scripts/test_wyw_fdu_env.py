"""CPU smoke test for the WYW task wired to the FDU closed-chain asset."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=("flat", "rough", "jump"), default="flat")
parser.add_argument("--play", action="store_true")
parser.add_argument("--num_envs", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

from agent_tasks.direct.wheelbipe.wyw.env import WheelbipeWywEnv  # noqa: E402
from agent_tasks.direct.wheelbipe.wyw.env_cfg import (  # noqa: E402
    FDU_JUMP_REWARDS,
    FDU_PLANE_REWARDS,
    WheelbipeWywFlatEnvCfg,
    WheelbipeWywRoughEnvCfg,
    WheelbipeWywJumpEnvCfg,
    WheelbipeWywFlatEnvCfg_Play,
    WheelbipeWywRoughEnvCfg_Play,
    WheelbipeWywJumpEnvCfg_Play,
)


def main():
    cfg_type = ({
        "flat": WheelbipeWywFlatEnvCfg_Play,
        "rough": WheelbipeWywRoughEnvCfg_Play,
        "jump": WheelbipeWywJumpEnvCfg_Play,
    } if args_cli.play else {
        "flat": WheelbipeWywFlatEnvCfg,
        "rough": WheelbipeWywRoughEnvCfg,
        "jump": WheelbipeWywJumpEnvCfg,
    })[args_cli.variant]
    cfg = cfg_type()
    cfg.seed = 42
    cfg.scene.num_envs = args_cli.num_envs
    cfg.scene.env_spacing = 2.0
    cfg.sim.device = args_cli.device
    cfg.play = False
    cfg.commands.debug_vis = False
    env = WheelbipeWywEnv(cfg)
    try:
        n = env.num_envs
        assert env.cfg.sim.dt == 0.002
        assert env.cfg.decimation == 5
        assert env.cfg.sim.render_interval == 5
        assert env.step_dt == 0.01
        assert env.cfg.episode_length_s == 20.0
        assert env.cfg.enable_state_machines is False
        assert env.cfg.airborne_state_machine_cfg["enabled"] is False
        assert env.cfg.wheel_forward_scan_cfg["enabled"] is False
        assert env.cfg.use_obs_delay is False
        assert env.cfg.use_act_delay is False
        assert not hasattr(env.cfg, "wyw_safe_l0_range")
        assert not hasattr(env.cfg, "wyw_safe_theta0_abs")
        assert env.cfg.termination_duration_enabled is True
        assert env.cfg.termination_duration_steps == 100
        assert env.cfg.max_wheel_vel == 60.0
        assert env.max_wheel_vel == 60.0
        assert env.robot.cfg.actuators["wheel"].velocity_limit == 60.0
        assert env.robot.cfg.actuators["wheel"].velocity_limit_sim == 60.0
        expected_wheel_effort = 50.0 if args_cli.variant == "jump" else 5.0
        assert env.robot.cfg.actuators["wheel"].effort_limit == expected_wheel_effort
        assert env.robot.cfg.actuators["legs_act"].effort_limit == 40.0
        expected_leg_pd = (6.0, 0.5) if args_cli.variant == "jump" else (20.0, 1.0)
        assert env.robot.cfg.actuators["legs_act"].stiffness == expected_leg_pd[0]
        assert env.robot.cfg.actuators["legs_act"].damping == expected_leg_pd[1]
        assert env.robot.cfg.spawn.articulation_props.solver_position_iteration_count == 16
        assert env.robot.cfg.spawn.articulation_props.solver_velocity_iteration_count == 6
        assert env.cfg.wyw_l0_stability_monitor_enabled is True
        assert env.cfg.wyw_l0_stability_boundary_m == 0.14
        assert env.cfg.commands.heading_command is False
        assert env.cfg.commands.rel_heading_envs == 0.0
        assert env.cfg.commands.rel_standing_envs == 0.0
        assert tuple(env.cfg.commands.ranges.lin_vel_y) == (0.0, 0.0)
        assert tuple(env.cfg.commands.ranges.ang_vel_z) == (-2.0, 2.0)
        assert tuple(env.cfg.height_range) == (0.15, 0.30)
        assert tuple(cfg.height_range) == (0.15, 0.30)
        if args_cli.variant == "rough":
            terrain_cfg = env.cfg.terrain.terrain_generator
            assert terrain_cfg.curriculum is True
            assert terrain_cfg.num_rows == 10 and terrain_cfg.num_cols == 20
            assert terrain_cfg.size == (8.0, 8.0)
            assert terrain_cfg.border_width == 25.0
            assert terrain_cfg.horizontal_scale == 0.1
            assert terrain_cfg.vertical_scale == 0.005
            assert terrain_cfg.slope_threshold == 0.75
            proportions = [float(item.proportion) for item in terrain_cfg.sub_terrains.values()]
            assert abs(sum(proportions) - 1.0) < 1.0e-6
            assert proportions == [0.20, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20, 0.10]
        expected_command_period = (20.0, 20.0) if args_cli.variant == "jump" else (5.0, 5.0)
        assert tuple(env.cfg.commands.resampling_time_range) == expected_command_period
        obs, _ = env.reset()
        assert torch.all(env.height_cmd >= 0.15) and torch.all(env.height_cmd <= 0.30)
        assert obs["policy"].shape == (n, 25)
        assert obs["policy_hist"].shape == (n, 125)
        assert obs["critic"].shape == (n, 141)
        assert torch.isfinite(obs["policy_hist"]).all()
        initial_history = obs["policy_hist"].reshape(n, 5, 25)
        assert torch.equal(initial_history, initial_history[:, :1].expand_as(initial_history))

        # Reset contract: all joint velocities are zero; root velocity is the
        # Fudan six-axis uniform sample. Rough alone also offsets tile-local XY.
        assert torch.allclose(env.robot.data.joint_vel, torch.zeros_like(env.robot.data.joint_vel), atol=1.0e-6)
        root_vel = torch.cat((env.robot.data.root_lin_vel_w, env.robot.data.root_ang_vel_w), dim=-1)
        assert torch.all(root_vel >= -0.5001) and torch.all(root_vel <= 0.5001)
        root_xy_delta = env.robot.data.root_pos_w[:, :2] - env.terrain.env_origins[:, :2]
        if args_cli.variant == "rough":
            assert torch.all(root_xy_delta >= -1.0001) and torch.all(root_xy_delta <= 1.0001)
        else:
            assert torch.allclose(root_xy_delta, torch.zeros_like(root_xy_delta), atol=1.0e-5)

        # Action-history contract in critic: after reset [a_{t-1},a_{t-2}]=0;
        # on the second observation it must contain [a_1,0], not [a_2,a_1].
        assert torch.count_nonzero(obs["critic"][:, 28:40]) == 0
        action_1 = torch.linspace(-0.3, 0.3, 6, device=env.device).repeat(n, 1)
        obs, reward, terminated, truncated, _ = env.step(action_1)
        expected_leg_targets = (
            env.robot.data.default_joint_pos[:, env._wyw_leg_joint_idx]
            + env.leg_action_scale * action_1[:, [0, 1, 3, 4]]
        )
        assert torch.allclose(env.leg_actions, expected_leg_targets)
        assert torch.count_nonzero(obs["critic"][:, 28:40]) == 0
        action_2 = -action_1
        obs, reward, terminated, truncated, _ = env.step(action_2)
        assert torch.allclose(obs["critic"][:, 28:34], action_1)
        assert torch.count_nonzero(obs["critic"][:, 34:40]) == 0
        for _ in range(1):
            obs, reward, terminated, truncated, _ = env.step(torch.zeros(n, 6, device=env.device))
            assert torch.isfinite(obs["policy"]).all()
            assert torch.isfinite(obs["policy_hist"]).all()
            assert torch.isfinite(obs["critic"]).all()
            assert torch.isfinite(reward).all()
        expected_rewards = FDU_JUMP_REWARDS if args_cli.variant == "jump" else FDU_PLANE_REWARDS
        assert list(env.cfg.rewards) == list(expected_rewards)
        assert set(env._last_reward_terms) == set(expected_rewards)
        reward_bound = env.cfg.clip_single_reward * env.step_dt + 1.0e-6
        assert all(torch.all(torch.abs(value) <= reward_bound) for value in env._last_reward_terms.values())
        if args_cli.variant != "jump":
            assert torch.count_nonzero(env._last_reward_terms["collision"]) == 0

        # Test the exact 1 s persistence helper without relying on a random
        # policy to happen to overturn during this short smoke run.
        all_ids = torch.arange(n, device=env.device)
        env._clear_termination_duration_buffers(
            all_ids,
            counter_attr="_wyw_orientation_termination_counter",
            raw_attr="_wyw_orientation_termination_raw_buf",
        )
        bad_orientation = torch.ones(n, dtype=torch.bool, device=env.device)
        for _ in range(99):
            assert not torch.any(env._apply_termination_duration(
                bad_orientation,
                counter_attr="_wyw_orientation_termination_counter",
                raw_attr="_wyw_orientation_termination_raw_buf",
            ))
        assert torch.all(env._apply_termination_duration(
            bad_orientation,
            counter_attr="_wyw_orientation_termination_counter",
            raw_attr="_wyw_orientation_termination_raw_buf",
        ))
        assert not torch.any(env._apply_termination_duration(
            torch.zeros_like(bad_orientation),
            counter_attr="_wyw_orientation_termination_counter",
            raw_attr="_wyw_orientation_termination_raw_buf",
        ))

        # The calibrated-boundary monitor must retain a substep event, expose
        # it in runner/TensorBoard logs, and avoid turning it into a reset.
        synthetic_l0 = torch.full((n, 2), 0.20, device=env.device)
        synthetic_l0[0, 0] = 0.13
        previous_entry_count = int(env._wyw_l0_boundary_total_entries.item())
        env._update_wyw_l0_stability_monitor(synthetic_l0)
        env._flush_wyw_l0_stability_monitor()
        assert int(env._wyw_l0_boundary_total_entries.item()) == previous_entry_count + 1
        assert int(env._wyw_l0_boundary_episode_samples[0].item()) >= 1
        assert env._wyw_l0_global_min_m <= 0.13
        assert "Diagnostics/FDU_L0Boundary/total_entry_events" in env.extras["log"]
        assert "Diagnostics/FDU_L0Boundary/total_physics_samples" in env.extras["log"]

        episode_length_saved = env.episode_length_buf.clone()
        env.episode_length_buf.fill_(env.max_episode_length - 1)
        _, time_out = env._get_dones()
        assert torch.all(time_out)
        env.episode_length_buf.copy_(episode_length_saved)
        print("WYW FDU ENV SMOKE TEST PASSED", flush=True)
        print(f"variant={args_cli.variant} play={args_cli.play} reward_terms={list(env.cfg.rewards)}", flush=True)
        print(f"policy_order={env._wyw_policy_joint_idx}", flush=True)
        print(f"leg_entity_indices={env._wyw_leg_joint_idx}", flush=True)
        print(f"wheel_entity_indices={env._wyw_wheel_joint_idx}", flush=True)
        print(f"obs_dims=policy:25 policy_hist:125 critic:141", flush=True)
        print(f"reward_terms={list(env.cfg.rewards)} clip_per_term={reward_bound}", flush=True)
        print("termination=projected_gravity_z>-0.1 for 100 consecutive policy steps; timeout=20 s", flush=True)
        print(
            f"wheel_velocity_limit={env.robot.cfg.actuators['wheel'].velocity_limit} rad/s "
            f"runtime_clamp={env.max_wheel_vel} rad/s "
            f"wheel_training_effort_limit={expected_wheel_effort} N*m "
            f"leg_effort_limit=40.0 N*m leg_PD={expected_leg_pd}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
