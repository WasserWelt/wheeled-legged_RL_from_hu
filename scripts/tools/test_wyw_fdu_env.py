"""CPU/GPU smoke test for the WYW task wired to the FDU closed-chain asset."""

import argparse
import tempfile
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=("flat", "rough", "jump"), default="flat")
parser.add_argument("--play", action="store_true")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument(
    "--rollout-steps",
    type=int,
    default=0,
    help="Additional zero-action policy steps used for finite-value/stability diagnostics.",
)
parser.add_argument(
    "--runner-lifecycle",
    action="store_true",
    help="Instantiate the real sequence runner and execute learn(0) to verify reset/randomization order.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402

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
from agent_tasks.direct.wheelbipe.wyw.fdu_semantics import (  # noqa: E402
    compute_fdu_collision_count,
    compute_fdu_failure_contact_condition,
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
    # Headless CI may not have access to Isaac's remote arrow-marker USD.
    # Disable that visualization only; retain the real Play runtime semantics.
    cfg.play_ang_vel_z_debug_vis = False
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
        assert env.cfg.wyw_training_semantics_version == "fdu_flat_p0_direct_bars_fd_vel_v2"
        assert env.cfg.wyw_joint_velocity_source == "wrapped_position_difference"
        assert env.cfg.wyw_joint_velocity_diff_dt == env.cfg.sim.dt == 0.002
        assert env.cfg.wyw_collision_contact_force == 0.1
        assert env.cfg.wyw_failure_contact_force == 10.0
        expected_failure_contact_bodies = {
            "base_link_del",
            "lf0_Link",
            "lf1_Link",
            "l20_Link",
            "l21_Link",
            "l22_Link",
            "l23_Link",
            "rf0_Link",
            "rf1_Link",
            "r20_Link",
            "r21_Link",
            "r22_Link",
            "r23_Link",
        }
        mapped_failure_contact_bodies = {
            env.contact_sensor.body_names[index] for index in env._reset_contact_link_idx
        }
        assert mapped_failure_contact_bodies == expected_failure_contact_bodies
        assert env._undesired_contact_link_idx == env._reset_contact_link_idx
        expected_flat_curriculum = args_cli.variant == "flat" and not args_cli.play
        assert env.cfg.wyw_flat_command_curriculum_enabled is expected_flat_curriculum
        assert env.cfg.max_wheel_vel == 60.0
        assert env.max_wheel_vel == 60.0
        assert env.robot.cfg.actuators["wheel"].velocity_limit == 60.0
        assert env.robot.cfg.actuators["wheel"].velocity_limit_sim == 60.0
        if args_cli.variant == "rough":
            assert env.cfg.rough_terrain_boundary_reset_cfg["margin"] == 1.0
        expected_wheel_effort = 50.0 if args_cli.variant == "jump" else 5.0
        assert env.robot.cfg.actuators["wheel"].effort_limit == expected_wheel_effort
        assert env.robot.cfg.actuators["legs_act"].effort_limit == 40.0
        expected_leg_pd = (6.0, 0.5) if args_cli.variant == "jump" else (20.0, 1.0)
        assert env.robot.cfg.actuators["legs_act"].stiffness == expected_leg_pd[0]
        assert env.robot.cfg.actuators["legs_act"].damping == expected_leg_pd[1]
        assert all(
            actuator.cfg.class_type.__name__ == "DiffVelPDActuator"
            for actuator in env.robot.actuators.values()
        )
        assert env.robot.cfg.spawn.articulation_props.solver_position_iteration_count == 8
        assert env.robot.cfg.spawn.articulation_props.solver_velocity_iteration_count == 4
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
        # Disable actor-only noise so this smoke can compare observation slices
        # against the exact runtime finite-difference state.
        env.use_self_obs_noise = False
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

        env_velocity_samples = []
        original_velocity_update = env._wyw_joint_velocity_estimator.update

        def record_env_velocity(position, *, sample_id=None):
            previous_position = env._wyw_joint_velocity_estimator.previous_position.clone()
            previous_sample_id = env._wyw_joint_velocity_estimator.last_sample_id
            velocity = original_velocity_update(position, sample_id=sample_id)
            env_velocity_samples.append(
                {
                    "sample_id": sample_id,
                    "fresh": sample_id != previous_sample_id,
                    "position": position.clone(),
                    "previous_position": previous_position,
                    "velocity": velocity.clone(),
                }
            )
            return velocity

        env._wyw_joint_velocity_estimator.update = record_env_velocity

        actuator_samples = {name: [] for name in env.robot.actuators}
        for actuator_name, actuator in env.robot.actuators.items():
            original_compute = actuator.compute

            def record_actuator_compute(
                control_action,
                joint_pos,
                joint_vel,
                *,
                _name=actuator_name,
                _actuator=actuator,
                _compute=original_compute,
            ):
                position_target = control_action.joint_positions.clone()
                velocity_target = control_action.joint_velocities.clone()
                feedforward = control_action.joint_efforts.clone()
                output = _compute(control_action, joint_pos, joint_vel)
                actuator_samples[_name].append(
                    {
                        "position": joint_pos.clone(),
                        "raw_velocity": joint_vel.clone(),
                        "position_target": position_target,
                        "velocity_target": velocity_target,
                        "feedforward": feedforward,
                        "fd_velocity": _actuator._velocity_estimator.velocity.clone(),
                        "computed_effort": _actuator.computed_effort.clone(),
                    }
                )
                return output

            actuator.compute = record_actuator_compute

        velocity_before_step = env._wyw_joint_velocity.clone()
        action_1 = torch.linspace(-0.3, 0.3, 6, device=env.device).repeat(n, 1)
        obs, reward, terminated, truncated, _ = env.step(action_1)
        assert not torch.any(terminated | truncated)

        # DirectRLEnv increments the simulation counter before _apply_action.
        # The five fresh states must therefore be sample ids 1..5, with id 5
        # captured after the final PhysX substep for observations/rewards.
        fresh_samples = [sample for sample in env_velocity_samples if sample["fresh"]]
        assert [sample["sample_id"] for sample in fresh_samples] == [1, 2, 3, 4, 5]
        for sample in fresh_samples:
            delta = torch.atan2(
                torch.sin(sample["position"] - sample["previous_position"]),
                torch.cos(sample["position"] - sample["previous_position"]),
            )
            torch.testing.assert_close(
                sample["velocity"], delta / env.cfg.wyw_joint_velocity_diff_dt
            )
        torch.testing.assert_close(
            env._wyw_joint_velocity_estimator.previous_position,
            env.robot.data.joint_pos[:, env._wyw_policy_joint_idx],
        )

        expected_scaled_velocity = env._wyw_joint_velocity * env.cfg.wyw_dof_vel_scale
        torch.testing.assert_close(obs["policy"][:, 13:19], expected_scaled_velocity)
        torch.testing.assert_close(obs["policy_hist"].reshape(n, 5, 25)[:, -1, 13:19], expected_scaled_velocity)
        torch.testing.assert_close(obs["critic"][:, 16:22], expected_scaled_velocity)
        expected_acceleration = (velocity_before_step - env._wyw_joint_velocity) / env.step_dt
        torch.testing.assert_close(env._wyw_policy_joint_acceleration, expected_acceleration)
        torch.testing.assert_close(
            obs["critic"][:, 40:46], expected_acceleration * env.cfg.wyw_joint_acc_scale
        )
        raw_reward_terms = env._compute_fdu_reward_terms()
        expected_dof_vel_reward = torch.square(
            env._wyw_joint_velocity[:, [0, 1, 3, 4]]
        ).sum(dim=-1)
        torch.testing.assert_close(raw_reward_terms["dof_vel"], expected_dof_vel_reward)

        # The real articulation called every actuator exactly once per 2 ms
        # substep. Its computed torque must use the estimator velocity even
        # when the PhysX generalized velocity differs.
        saw_raw_velocity_difference = False
        for actuator_name, actuator in env.robot.actuators.items():
            records = actuator_samples[actuator_name]
            assert len(records) == env.cfg.decimation
            assert torch.count_nonzero(records[0]["fd_velocity"]) == 0
            for previous, current in zip(records, records[1:]):
                delta = torch.atan2(
                    torch.sin(current["position"] - previous["position"]),
                    torch.cos(current["position"] - previous["position"]),
                )
                torch.testing.assert_close(
                    current["fd_velocity"], delta / env.cfg.wyw_joint_velocity_diff_dt
                )
            final = records[-1]
            nominal_effort = (
                actuator.stiffness * (final["position_target"] - final["position"])
                + actuator.damping * (final["velocity_target"] - final["fd_velocity"])
                + final["feedforward"]
            )
            output_scale = getattr(
                actuator, "_wb_effort_output_scale", torch.ones_like(nominal_effort)
            )
            output_bias = getattr(
                actuator, "_wb_effort_output_bias", torch.zeros_like(nominal_effort)
            )
            output_noise_std = getattr(
                actuator, "_wb_effort_output_noise_std", torch.zeros_like(nominal_effort)
            )
            assert torch.count_nonzero(output_noise_std) == 0
            expected_effort = nominal_effort * output_scale + output_bias
            if not torch.allclose(final["computed_effort"], expected_effort):
                print(
                    f"[FDVelDebug:{actuator_name}] "
                    f"max_effort_error={float(torch.max(torch.abs(final['computed_effort'] - expected_effort))):.6f}\n"
                    f"stiffness={actuator.stiffness[0].detach().cpu().tolist()}\n"
                    f"damping={actuator.damping[0].detach().cpu().tolist()}\n"
                    f"q={final['position'][0].detach().cpu().tolist()}\n"
                    f"q_target={final['position_target'][0].detach().cpu().tolist()}\n"
                    f"fd_vel={final['fd_velocity'][0].detach().cpu().tolist()}\n"
                    f"raw_vel={final['raw_velocity'][0].detach().cpu().tolist()}\n"
                    f"vel_target={final['velocity_target'][0].detach().cpu().tolist()}\n"
                    f"feedforward={final['feedforward'][0].detach().cpu().tolist()}\n"
                    f"computed={final['computed_effort'][0].detach().cpu().tolist()}\n"
                    f"expected={expected_effort[0].detach().cpu().tolist()}",
                    flush=True,
                )
            torch.testing.assert_close(final["computed_effort"], expected_effort)
            saw_raw_velocity_difference |= bool(
                torch.any(torch.abs(final["raw_velocity"] - final["fd_velocity"]) > 1.0e-3)
            )
        assert saw_raw_velocity_difference

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

        # Articulation.reset() must invalidate only the selected actuator rows;
        # WYW primes its observation estimator from the newly written qpos.
        reset_id = torch.tensor([0], device=env.device)
        env._reset_idx(reset_id)
        assert torch.count_nonzero(env._wyw_joint_velocity[reset_id]) == 0
        assert bool(env._wyw_joint_velocity_estimator.initialized[reset_id].all())
        for actuator in env.robot.actuators.values():
            assert not bool(actuator._velocity_estimator.initialized[reset_id].any())
            assert bool(actuator._velocity_estimator.initialized[1:].all())
        previous_record_counts = {
            name: len(records) for name, records in actuator_samples.items()
        }
        obs, reward, terminated, truncated, _ = env.step(torch.zeros(n, 6, device=env.device))
        for actuator_name, records in actuator_samples.items():
            first_after_reset = records[previous_record_counts[actuator_name]]
            assert torch.count_nonzero(first_after_reset["fd_velocity"][reset_id]) == 0

        expected_rewards = FDU_JUMP_REWARDS if args_cli.variant == "jump" else FDU_PLANE_REWARDS
        assert list(env.cfg.rewards) == list(expected_rewards)
        assert set(env._last_reward_terms) == set(expected_rewards)
        reward_bound = env.cfg.clip_single_reward * env.step_dt + 1.0e-6
        assert all(torch.all(torch.abs(value) <= reward_bound) for value in env._last_reward_terms.values())
        if args_cli.variant != "jump":
            assert "tracking_ang_vel_enhance" not in env.cfg.rewards
            assert env.cfg.rewards["orientation"] == -15.0
            assert env.cfg.rewards["ang_vel_xy"] == -0.05
            assert env.cfg.rewards["torques"] == -1.0e-3
            assert env.cfg.rewards["action_rate"] == -0.3
            assert env.cfg.rewards["action_smooth"] == -0.3
            assert env.cfg.rewards["dof_acc"] == -3.0e-7

        # Runtime YAML is the reproducibility contract used by cloud runs.
        with tempfile.TemporaryDirectory(dir="/tmp") as dump_dir:
            env_yaml = Path(dump_dir) / "env.yaml"
            dump_yaml(str(env_yaml), env.cfg)
            dumped = env_yaml.read_text(encoding="utf-8")
            for required in (
                "wyw_training_semantics_version: fdu_flat_p0_direct_bars_fd_vel_v2",
                "wyw_joint_velocity_source: wrapped_position_difference",
                "wyw_joint_velocity_diff_dt: 0.002",
                "wyw_failure_contact_force: 10.0",
                "wyw_collision_contact_force: 0.1",
                "wyw_flat_command_curriculum_enabled:",
                "wyw_command_curriculum_interval_steps: 2000",
            ):
                assert required in dumped

        if expected_flat_curriculum:
            saved_step = env.common_step_counter
            saved_ranges = env._wyw_command_ranges_x.clone()
            saved_lin = env._episode_sums["tracking_lin_vel"].clone()
            saved_yaw = env._episode_sums["tracking_ang_vel"].clone()
            saved_episode_length = env.episode_length_buf.clone()
            saved_curriculum_last_step = env._wyw_flat_curriculum_last_step
            interval = env.cfg.wyw_command_curriculum_interval_steps
            env.common_step_counter = interval + 17
            env._wyw_flat_curriculum_last_step = 0
            env._episode_sums["tracking_lin_vel"].fill_(0.71 * env.max_episode_length_s)
            env._episode_sums["tracking_ang_vel"].fill_(0.57 * env.max_episode_length_s)
            env._reset_idx(torch.arange(n, device=env.device))
            expected_expanded_ranges = saved_ranges.clone()
            expected_expanded_ranges[:, 0] = torch.clamp(
                expected_expanded_ranges[:, 0] - env.cfg.wyw_command_curriculum_step,
                min=-env.cfg.wyw_command_curriculum_max_abs,
                max=0.0,
            )
            expected_expanded_ranges[:, 1] = torch.clamp(
                expected_expanded_ranges[:, 1] + env.cfg.wyw_command_curriculum_step,
                min=0.0,
                max=env.cfg.wyw_command_curriculum_max_abs,
            )
            assert torch.allclose(
                env._wyw_command_ranges_x,
                expected_expanded_ranges,
            )
            assert env.extras["log"]["Curriculum/FDUFlat/expanded"] == 1
            assert env.extras["log"]["Curriculum/FDUFlat/cadence_step"] == interval
            assert env.extras["log"]["Curriculum/FDUFlat/consumed_at_step"] == interval + 17
            assert torch.count_nonzero(env._episode_sums["tracking_lin_vel"]) == 0
            assert torch.count_nonzero(env._episode_sums["tracking_ang_vel"]) == 0

            # The same cadence cannot be consumed twice, even if another reset
            # arrives with qualifying episode rewards before the next boundary.
            env._episode_sums["tracking_lin_vel"].fill_(0.71 * env.max_episode_length_s)
            env._episode_sums["tracking_ang_vel"].fill_(0.57 * env.max_episode_length_s)
            env._update_fdu_flat_command_curriculum(torch.arange(n, device=env.device))
            assert torch.allclose(
                env._wyw_command_ranges_x,
                expected_expanded_ranges,
            )
            env.common_step_counter = saved_step
            env._wyw_command_ranges_x.copy_(saved_ranges)
            env._episode_sums["tracking_lin_vel"].copy_(saved_lin)
            env._episode_sums["tracking_ang_vel"].copy_(saved_yaw)
            env.episode_length_buf.copy_(saved_episode_length)
            env._wyw_flat_curriculum_last_step = saved_curriculum_last_step
            env._wyw_flat_curriculum_pending_log = None

        # Test Fudan's exact shared 1 s failure counter on the selected device.
        # Switching from contact failure to tilt failure must not reset it.
        all_ids = torch.arange(n, device=env.device)
        env._clear_termination_duration_buffers(
            all_ids,
            counter_attr="_wyw_failure_termination_counter",
            raw_attr="_wyw_failure_termination_raw_buf",
        )
        synthetic_contact_forces = torch.zeros(n, 13, 3, device=env.device)
        synthetic_contact_forces[:, 1, 0] = env.cfg.wyw_collision_contact_force + 0.01
        synthetic_contact_forces[:, 2, 1] = -(env.cfg.wyw_collision_contact_force + 0.02)
        collision_count = compute_fdu_collision_count(
            synthetic_contact_forces, env.cfg.wyw_collision_contact_force
        )
        assert torch.all(collision_count == 2)
        synthetic_contact_forces.zero_()
        synthetic_contact_forces[:, 0, 0] = env.cfg.wyw_failure_contact_force + 0.01
        bad_contact = compute_fdu_failure_contact_condition(
            synthetic_contact_forces, env.cfg.wyw_failure_contact_force
        )
        assert torch.all(bad_contact)
        good_contact = compute_fdu_failure_contact_condition(
            torch.zeros_like(synthetic_contact_forces), env.cfg.wyw_failure_contact_force
        )
        bad_orientation = torch.zeros(n, dtype=torch.bool, device=env.device)
        for _ in range(50):
            assert not torch.any(env._apply_termination_duration(
                bad_contact | bad_orientation,
                counter_attr="_wyw_failure_termination_counter",
                raw_attr="_wyw_failure_termination_raw_buf",
            ))
        bad_orientation.fill_(True)
        for _ in range(49):
            assert not torch.any(env._apply_termination_duration(
                good_contact | bad_orientation,
                counter_attr="_wyw_failure_termination_counter",
                raw_attr="_wyw_failure_termination_raw_buf",
            ))
        assert torch.all(env._apply_termination_duration(
            good_contact | bad_orientation,
            counter_attr="_wyw_failure_termination_counter",
            raw_attr="_wyw_failure_termination_raw_buf",
        ))
        assert not torch.any(env._apply_termination_duration(
            torch.zeros_like(bad_orientation),
            counter_attr="_wyw_failure_termination_counter",
            raw_attr="_wyw_failure_termination_raw_buf",
        ))

        # The compact boundary monitor retains the per-episode affected flag.
        synthetic_l0 = torch.full((n, 2), 0.20, device=env.device)
        synthetic_l0[0, 0] = 0.13
        env._update_wyw_l0_stability_monitor(synthetic_l0)
        assert int(env._wyw_l0_boundary_episode_samples[0].item()) >= 1

        episode_length_saved = env.episode_length_buf.clone()
        env.episode_length_buf.fill_(env.max_episode_length - 1)
        _, time_out = env._get_dones()
        assert torch.all(time_out)
        env.episode_length_buf.copy_(episode_length_saved)

        if args_cli.variant == "rough":
            root_pos_saved = env.robot.data.root_pos_w[0].clone()
            episode_length_0_saved = env.episode_length_buf[0].clone()
            env.episode_length_buf[0] = 0
            env.robot.data.root_pos_w[0, 0] = 1.0e6
            boundary = env._get_rough_terrain_boundary_termination()
            terminated, time_out = env._get_dones()
            assert boundary[0]
            if args_cli.play and not env.cfg.play_keep_done_reset:
                assert not terminated[0]
            else:
                assert terminated[0]
            assert not time_out[0]
            env.robot.data.root_pos_w[0].copy_(root_pos_saved)
            env.episode_length_buf[0].copy_(episode_length_0_saved)

        if args_cli.rollout_steps < 0:
            raise ValueError("--rollout-steps must be non-negative")
        if args_cli.rollout_steps:
            rollout_terminated = 0
            rollout_truncated = 0
            max_joint_vel = 0.0
            max_root_lin_vel = 0.0
            max_root_ang_vel = 0.0
            max_applied_torque = 0.0
            for _ in range(args_cli.rollout_steps):
                obs, reward, terminated, truncated, _ = env.step(
                    torch.zeros(n, 6, device=env.device)
                )
                assert all(torch.isfinite(value).all() for value in obs.values())
                assert torch.isfinite(reward).all()
                assert torch.isfinite(env.robot.data.joint_pos).all()
                assert torch.isfinite(env.robot.data.joint_vel).all()
                assert torch.isfinite(env.robot.data.root_state_w).all()
                assert torch.isfinite(env.robot.data.applied_torque).all()
                rollout_terminated += int(terminated.sum().item())
                rollout_truncated += int(truncated.sum().item())
                max_joint_vel = max(max_joint_vel, float(env.robot.data.joint_vel.abs().max().item()))
                max_root_lin_vel = max(
                    max_root_lin_vel, float(env.robot.data.root_lin_vel_b.abs().max().item())
                )
                max_root_ang_vel = max(
                    max_root_ang_vel, float(env.robot.data.root_ang_vel_b.abs().max().item())
                )
                max_applied_torque = max(
                    max_applied_torque, float(env.robot.data.applied_torque.abs().max().item())
                )
            print(
                "WYW FDU GPU ROLLOUT PASSED "
                f"steps={args_cli.rollout_steps} terminated={rollout_terminated} "
                f"truncated={rollout_truncated} max_joint_vel={max_joint_vel:.6f} "
                f"max_root_lin_vel={max_root_lin_vel:.6f} "
                f"max_root_ang_vel={max_root_ang_vel:.6f} "
                f"max_applied_torque={max_applied_torque:.6f}",
                flush=True,
            )

        if args_cli.runner_lifecycle:
            if args_cli.variant != "flat" or args_cli.play:
                raise ValueError("--runner-lifecycle is only supported for the Flat training config")
            from agent_rl.rsl_rl.env import RslRlVecEnvWrapper
            from agent_rl.rsl_rl.runners import OnPolicySequenceRunner
            from agent_tasks.direct.wheelbipe.agents.rsl_rl_ppo_cfg import WheelbipeWywPPORunnerCfg

            wrapped_env = RslRlVecEnvWrapper(env, clip_actions=100.0)
            runner_cfg = WheelbipeWywPPORunnerCfg().to_dict()
            runner = OnPolicySequenceRunner(
                wrapped_env, runner_cfg, log_dir=None, device=str(env.device)
            )
            env.episode_length_buf.fill_(env.max_episode_length - 1)
            torch.manual_seed(20260830)
            torch.cuda.manual_seed_all(20260830)
            runner.learn(num_learning_iterations=0, init_at_random_ep_len=True)
            assert torch.all(env.episode_length_buf >= 0)
            assert torch.all(env.episode_length_buf < env.max_episode_length)
            assert torch.any(env.episode_length_buf != 0), (
                "episode lengths remained reset to zero; randomization likely ran before reset"
            )
            print(
                "WYW FDU RUNNER LIFECYCLE PASSED "
                f"episode_length_buf={env.episode_length_buf.detach().cpu().tolist()}",
                flush=True,
            )
        print("WYW FDU ENV SMOKE TEST PASSED", flush=True)
        print(f"variant={args_cli.variant} play={args_cli.play} reward_terms={list(env.cfg.rewards)}", flush=True)
        print(f"policy_order={env._wyw_policy_joint_idx}", flush=True)
        print(f"leg_entity_indices={env._wyw_leg_joint_idx}", flush=True)
        print(f"wheel_entity_indices={env._wyw_wheel_joint_idx}", flush=True)
        print(f"obs_dims=policy:25 policy_hist:125 critic:141", flush=True)
        print(f"reward_terms={list(env.cfg.rewards)} clip_per_term={reward_bound}", flush=True)
        print(
            "termination=(projected_gravity_z>-0.1 OR mapped contact force>10 N) "
            "for 100 consecutive policy steps; timeout=20 s",
            flush=True,
        )
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
