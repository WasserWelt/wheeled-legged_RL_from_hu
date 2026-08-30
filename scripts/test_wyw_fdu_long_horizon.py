"""Long-horizon integration acceptance for WYW/FDU reset and command timing.

Runs real 500 Hz physics until a natural 20 s timeout.  It verifies the
5 s (Flat/Rough) or 20 s (Jump) command resampling boundary, automatic reset,
observation-history refill, episode L0 logs, and Rough curriculum execution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=("flat", "rough", "jump"), default="flat")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

from agent_tasks.direct.wheelbipe.wyw.env import WheelbipeWywEnv  # noqa: E402
from agent_tasks.direct.wheelbipe.wyw.env_cfg import (  # noqa: E402
    WheelbipeWywFlatEnvCfg,
    WheelbipeWywJumpEnvCfg,
    WheelbipeWywRoughEnvCfg,
)


def main() -> None:
    cfg_type = {
        "flat": WheelbipeWywFlatEnvCfg,
        "rough": WheelbipeWywRoughEnvCfg,
        "jump": WheelbipeWywJumpEnvCfg,
    }[args_cli.variant]
    cfg = cfg_type()
    cfg.seed = 42
    cfg.scene.num_envs = args_cli.num_envs
    cfg.scene.env_spacing = 2.0
    cfg.sim.device = args_cli.device
    cfg.commands.debug_vis = False
    # Isolate natural timeout from random-policy tilt. The persistence behavior
    # remains covered by the semantic fixture and smoke test.
    production_termination_duration_steps = cfg.termination_duration_steps
    cfg.termination_duration_steps = 100000

    env = WheelbipeWywEnv(cfg)
    report: dict[str, object] = {
        "variant": args_cli.variant,
        "num_envs": args_cli.num_envs,
        "physics_dt_s": env.physics_dt,
        "policy_dt_s": env.step_dt,
        "decimation": env.cfg.decimation,
        "solver_iterations": [
            env.robot.cfg.spawn.articulation_props.solver_position_iteration_count,
            env.robot.cfg.spawn.articulation_props.solver_velocity_iteration_count,
        ],
        "expected_command_period_s": 20.0 if args_cli.variant == "jump" else 5.0,
        "height_command_range_m": list(env.cfg.height_range),
        "production_termination_duration_steps": production_termination_duration_steps,
        "termination_duration_steps_under_test": cfg.termination_duration_steps,
    }
    try:
        obs, _ = env.reset(seed=42)
        assert tuple(env.cfg.height_range) == (0.15, 0.30)
        assert torch.all(env.height_cmd >= 0.15) and torch.all(env.height_cmd <= 0.30)
        n = env.num_envs
        actions = torch.zeros(n, 6, device=env.device)
        initial_counter = env.command_generator.command_counter.clone()
        command_resample_steps: list[int] = []
        timeout_steps: list[int] = []
        command_resampled_by_episode_reset = False
        unexpected_terminations = 0
        curriculum_changed = False
        initial_levels = None
        initial_ranges = env._wyw_command_ranges_x.clone()
        if args_cli.variant == "rough":
            initial_levels = env.terrain.terrain_levels.clone()

        for step in range(1, env.max_episode_length + 2):
            previous_counter = env.command_generator.command_counter.clone()
            obs, reward, terminated, truncated, extras = env.step(actions)
            assert torch.isfinite(reward).all()
            assert torch.isfinite(obs["policy"]).all()
            assert torch.isfinite(obs["policy_hist"]).all()
            assert torch.isfinite(obs["critic"]).all()

            counter = env.command_generator.command_counter
            if torch.any(counter > previous_counter):
                command_resample_steps.append(step)
                assert torch.all(env.height_cmd >= 0.15) and torch.all(env.height_cmd <= 0.30)
            unexpected_terminations += int(terminated.sum().item())
            if torch.any(truncated):
                timeout_steps.append(step)
                reset_ids = truncated.nonzero(as_tuple=False).flatten()
                command_resampled_by_episode_reset = bool(
                    torch.all(env.command_generator.command_counter[reset_ids] == 1).item()
                )
                # DirectRLEnv has already reset and produced the first new obs.
                history = obs["policy_hist"].reshape(n, 5, 25)
                assert torch.equal(
                    history[reset_ids], history[reset_ids, :1].expand_as(history[reset_ids])
                )
                assert torch.count_nonzero(env._previous_actions[reset_ids]) == 0
                assert torch.count_nonzero(env._before_previous_actions[reset_ids]) == 0
                assert torch.count_nonzero(env.episode_length_buf[reset_ids]) == 0
                assert torch.all(env.height_cmd[reset_ids] >= 0.15)
                assert torch.all(env.height_cmd[reset_ids] <= 0.30)
                log = extras.get("log", {})
                for key in (
                    "Episode/FDU_L0Boundary/affected_env_fraction",
                    "Episode/FDU_L0Boundary/mean_physics_samples",
                    "Episode/FDU_L0Boundary/entry_events",
                    "Episode/FDU_L0Boundary/min_measured_l0_m",
                ):
                    assert key in log, key
                if args_cli.variant == "rough":
                    curriculum_changed = bool(
                        torch.any(env.terrain.terrain_levels != initial_levels).item()
                        or torch.any(env._wyw_command_ranges_x != initial_ranges).item()
                    )
                break

        assert timeout_steps == [env.max_episode_length - 1], (
            timeout_steps,
            env.max_episode_length,
        )
        expected_first_resample = int(round(report["expected_command_period_s"] / env.step_dt))
        if args_cli.variant == "jump":
            # 20 s command period equals the episode length.  The environment
            # times out at step 1999 and reset resamples the command before a
            # standalone periodic resample can occur at step 2000.
            assert command_resample_steps == []
            assert command_resampled_by_episode_reset
        else:
            assert command_resample_steps and command_resample_steps[0] == expected_first_resample
        assert unexpected_terminations == 0
        if args_cli.variant == "rough":
            assert curriculum_changed, "Rough curriculum did not update across the natural timeout reset"

        report.update(
            {
                "max_episode_length_steps": env.max_episode_length,
                "first_periodic_command_resample_step": (
                    command_resample_steps[0] if command_resample_steps else None
                ),
                "command_resampled_by_episode_reset": command_resampled_by_episode_reset,
                "command_counter_initial": initial_counter.detach().cpu().tolist(),
                "timeout_step": timeout_steps[0],
                "unexpected_terminations": unexpected_terminations,
                "reset_history_refilled": True,
                "reset_action_history_zero": True,
                "episode_l0_log_keys_present": True,
                "rough_curriculum_changed": curriculum_changed if args_cli.variant == "rough" else None,
                "final_command_ranges_x": env._wyw_command_ranges_x.detach().cpu().tolist(),
            }
        )
        if initial_levels is not None:
            report["initial_terrain_levels"] = initial_levels.detach().cpu().tolist()
            report["final_terrain_levels"] = env.terrain.terrain_levels.detach().cpu().tolist()

        output = args_cli.output or Path(
            f"docs/fdu_validation/training/fdu_{args_cli.variant}_long_horizon_500hz.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("WYW FDU LONG-HORIZON ACCEPTANCE PASSED", flush=True)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        print(f"report={output}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
