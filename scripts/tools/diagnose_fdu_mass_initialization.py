"""Diagnose WYW/FDU reset and zero-action stability at large environment counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import MethodType

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--env-spacing", type=float, default=4.0)
parser.add_argument("--policy-steps", type=int, default=200)
parser.add_argument("--sample-interval", type=int, default=10)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--restitution-range",
    type=float,
    nargs=2,
    metavar=("MIN", "MAX"),
    default=None,
    help="Override the wheel restitution randomization range.",
)
parser.add_argument(
    "--ground-restitution",
    type=float,
    default=None,
    help="Override both simulation and terrain restitution.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import agent_tasks  # noqa: E402
import agent_world  # noqa: E402
from agent_tasks.direct.wheelbipe.wyw.env import WheelbipeWywEnv  # noqa: E402
from agent_tasks.direct.wheelbipe.wyw.env_cfg import WheelbipeWywFlatEnvCfg  # noqa: E402


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().reshape(-1)
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return {key: float("nan") for key in ("min", "p01", "p05", "p50", "p95", "p99", "max")}
    probs = torch.tensor((0.01, 0.05, 0.50, 0.95, 0.99), device=finite.device)
    result = torch.quantile(finite, probs).cpu().tolist()
    return {
        "min": float(finite.min().item()),
        "p01": float(result[0]),
        "p05": float(result[1]),
        "p50": float(result[2]),
        "p95": float(result[3]),
        "p99": float(result[4]),
        "max": float(finite.max().item()),
    }


def _state_snapshot(env: WheelbipeWywEnv) -> dict[str, object]:
    root_lin = env.robot.data.root_lin_vel_b
    root_ang = env.robot.data.root_ang_vel_b
    joint_vel = env.robot.data.joint_vel[:, env._actuate_idx]
    result: dict[str, object] = {
        "nonfinite_envs": int(
            (~torch.isfinite(root_lin).all(dim=-1)
             | ~torch.isfinite(root_ang).all(dim=-1)
             | ~torch.isfinite(joint_vel).all(dim=-1)).sum().item()
        ),
        "root_linear_speed": _quantiles(torch.linalg.vector_norm(root_lin, dim=-1)),
        "root_angular_speed": _quantiles(torch.linalg.vector_norm(root_ang, dim=-1)),
        "active_joint_abs_max": _quantiles(joint_vel.abs().amax(dim=-1)),
        "root_height": _quantiles(env.robot.data.root_pos_w[:, 2]),
        "projected_gravity_z": _quantiles(env.robot.data.projected_gravity_b[:, 2]),
        "root_below_ground_envs": int((env.robot.data.root_pos_w[:, 2] < 0.0).sum().item()),
        "root_speed_over_5_envs": int((torch.linalg.vector_norm(root_lin, dim=-1) > 5.0).sum().item()),
        "root_speed_over_10_envs": int((torch.linalg.vector_norm(root_lin, dim=-1) > 10.0).sum().item()),
    }
    for index, axis in enumerate("xyz"):
        result[f"root_linear_{axis}_abs"] = _quantiles(root_lin[:, index].abs())
        result[f"root_angular_{axis}_abs"] = _quantiles(root_ang[:, index].abs())
    return result


def main() -> None:
    if args_cli.num_envs < 1 or args_cli.policy_steps < 0 or args_cli.sample_interval < 1:
        raise ValueError("num_envs and sample_interval must be positive; policy_steps must be non-negative")

    cfg = WheelbipeWywFlatEnvCfg()
    cfg.seed = 42
    cfg.scene.num_envs = args_cli.num_envs
    cfg.scene.env_spacing = args_cli.env_spacing
    cfg.terrain.env_spacing = args_cli.env_spacing
    cfg.sim.device = args_cli.device
    cfg.play_ang_vel_z_debug_vis = False
    cfg.commands.debug_vis = False
    cfg.height_scanner.debug_vis = False
    cfg.dot_scanner.debug_vis = False
    if args_cli.restitution_range is not None:
        cfg.events.physics_material.params["restitution_range"] = tuple(args_cli.restitution_range)
    if args_cli.ground_restitution is not None:
        cfg.sim.physics_material.restitution = args_cli.ground_restitution
        cfg.terrain.physics_material.restitution = args_cli.ground_restitution

    env = WheelbipeWywEnv(cfg)
    report: dict[str, object] = {
        "num_envs": args_cli.num_envs,
        "env_spacing": args_cli.env_spacing,
        "policy_steps": args_cli.policy_steps,
        "step_dt": env.step_dt,
        "wheel_restitution_range_config": list(cfg.events.physics_material.params["restitution_range"]),
        "link_restitution_range_config": list(
            cfg.events.physics_material.params["link_restitution_range"]
        ),
        "wheel_static_friction_range_config": list(
            cfg.events.physics_material.params["friction_range"]
        ),
        "wheel_dynamic_friction_range_config": list(
            cfg.events.physics_material.params["dynamic_friction_range"]
        ),
        "ground_static_friction_config": float(cfg.terrain.physics_material.static_friction),
        "ground_dynamic_friction_config": float(cfg.terrain.physics_material.dynamic_friction),
        "ground_restitution_config": float(cfg.terrain.physics_material.restitution),
        "restitution_combine_mode": str(cfg.terrain.physics_material.restitution_combine_mode),
        "terrain_type": str(cfg.terrain.terrain_type),
        "agent_tasks_module": str(Path(agent_tasks.__file__).resolve()),
        "agent_world_module": str(Path(agent_world.__file__).resolve()),
        "wheel_velocity_action_scale": float(cfg.wheel_vel_action_scale),
    }
    try:
        obs, _ = env.reset()
        report["initial_observation_nonfinite_envs"] = int(
            torch.stack([~torch.isfinite(value).all(dim=-1) for value in obs.values()]).any(dim=0).sum().item()
        )
        report["initial_state"] = _state_snapshot(env)
        origins = env.terrain.env_origins
        inside_ground = (origins[:, 0].abs() <= 100.0) & (origins[:, 1].abs() <= 100.0)
        origin_report = {
            "x": _quantiles(origins[:, 0]),
            "y": _quantiles(origins[:, 1]),
        }
        if cfg.terrain.terrain_type == "usd":
            origin_report.update({
                "inside_200m_ground": int(inside_ground.sum().item()),
                "outside_200m_ground": int((~inside_ground).sum().item()),
            })
        else:
            origin_report["ground_coverage"] = "unbounded_plane"
        report["environment_origins"] = origin_report
        if hasattr(env, "_wyw_restitution_sample"):
            ground_static = float(cfg.terrain.physics_material.static_friction)
            ground_dynamic = float(cfg.terrain.physics_material.dynamic_friction)
            ground_restitution = float(cfg.terrain.physics_material.restitution)
            wheel_static = env._wyw_friction_sample
            wheel_dynamic = env._wyw_dynamic_friction_sample
            wheel_restitution = env._wyw_restitution_sample
            report["sampled_wheel_static_friction"] = _quantiles(wheel_static)
            report["sampled_wheel_dynamic_friction"] = _quantiles(wheel_dynamic)
            report["sampled_wheel_restitution"] = _quantiles(wheel_restitution)
            report["sampled_link_restitution"] = _quantiles(
                env._wyw_link_restitution_sample
            )
            report["effective_wheel_ground_static_friction_average"] = _quantiles(
                0.5 * (wheel_static + ground_static)
            )
            report["effective_wheel_ground_dynamic_friction_average"] = _quantiles(
                0.5 * (wheel_dynamic + ground_dynamic)
            )
            report["effective_wheel_ground_restitution_average"] = _quantiles(
                0.5 * (wheel_restitution + ground_restitution)
            )
            properties = env.robot.root_physx_view.get_material_properties()
            wheel_indices = env._get_material_indices(["[lr]_wheel_Link"])
            link_indices = [
                index for index in range(properties.shape[1]) if index not in wheel_indices
            ]
            if not wheel_indices or not link_indices:
                raise RuntimeError(
                    f"invalid wheel/link material partition: wheels={wheel_indices}, links={link_indices}"
                )
            report["physx_material_shape_partition"] = {
                "wheel_shape_indices": wheel_indices,
                "link_shape_count": len(link_indices),
                "wheel_static_friction": _quantiles(properties[:, wheel_indices, 0]),
                "wheel_dynamic_friction": _quantiles(properties[:, wheel_indices, 1]),
                "wheel_restitution": _quantiles(properties[:, wheel_indices, 2]),
                "link_restitution": _quantiles(properties[:, link_indices, 2]),
                "dynamic_not_above_static": bool(
                    torch.all(properties[:, :, 1] <= properties[:, :, 0]).item()
                ),
            }

        reset_stats = {
            "reset_envs": 0,
            "numerical_safety": 0,
            "persistent_failure": 0,
            "orientation": 0,
            "contact": 0,
            "nan_or_inf": 0,
            "joint_velocity_outlier": 0,
            "root_linear_velocity_outlier": 0,
            "root_angular_velocity_outlier": 0,
            "terminal_root_linear_speed_max": 0.0,
            "terminal_root_angular_speed_max": 0.0,
            "terminal_active_joint_abs_max": 0.0,
        }
        original_reset_idx = env._reset_idx

        def capture_reset(this, env_ids):
            ids = this._as_env_ids_tensor(env_ids)
            if ids.numel() > 0:
                root_lin = this.robot.data.root_lin_vel_b[ids]
                root_ang = this.robot.data.root_ang_vel_b[ids]
                joint_vel = this.robot.data.joint_vel[ids][:, this._actuate_idx]
                reset_stats["reset_envs"] += int(ids.numel())
                reset_stats["nan_or_inf"] += int(
                    (~torch.isfinite(root_lin).all(dim=-1)
                     | ~torch.isfinite(root_ang).all(dim=-1)
                     | ~torch.isfinite(joint_vel).all(dim=-1)).sum().item()
                )
                joint_abs = joint_vel.abs().amax(dim=-1)
                root_lin_abs = root_lin.abs().amax(dim=-1)
                root_ang_abs = root_ang.abs().amax(dim=-1)
                reset_stats["joint_velocity_outlier"] += int((joint_abs > 500.0).sum().item())
                reset_stats["root_linear_velocity_outlier"] += int((root_lin_abs > 100.0).sum().item())
                reset_stats["root_angular_velocity_outlier"] += int((root_ang_abs > 200.0).sum().item())
                reset_stats["terminal_root_linear_speed_max"] = max(
                    reset_stats["terminal_root_linear_speed_max"],
                    float(torch.linalg.vector_norm(root_lin, dim=-1).max().item()),
                )
                reset_stats["terminal_root_angular_speed_max"] = max(
                    reset_stats["terminal_root_angular_speed_max"],
                    float(torch.linalg.vector_norm(root_ang, dim=-1).max().item()),
                )
                reset_stats["terminal_active_joint_abs_max"] = max(
                    reset_stats["terminal_active_joint_abs_max"], float(joint_abs.max().item())
                )
                for name in ("numerical_safety", "persistent_failure", "orientation", "contact"):
                    mask = getattr(this, f"_wyw_done_reason_{name}", None)
                    if mask is not None:
                        reset_stats[name] += int(mask[ids].sum().item())
            return original_reset_idx(env_ids)

        env._reset_idx = MethodType(capture_reset, env)
        samples = [{"step": 0, **_state_snapshot(env)}]
        zero_actions = torch.zeros(env.num_envs, 6, device=env.device)
        reward_nonfinite_env_steps = 0
        observation_nonfinite_env_steps = 0
        terminated_env_steps = 0
        truncated_env_steps = 0
        for step in range(1, args_cli.policy_steps + 1):
            obs, reward, terminated, truncated, _ = env.step(zero_actions)
            reward_nonfinite_env_steps += int((~torch.isfinite(reward)).sum().item())
            observation_nonfinite_env_steps += int(
                torch.stack([~torch.isfinite(value).all(dim=-1) for value in obs.values()]).any(dim=0).sum().item()
            )
            terminated_env_steps += int(terminated.sum().item())
            truncated_env_steps += int(truncated.sum().item())
            if step % args_cli.sample_interval == 0 or step == args_cli.policy_steps:
                samples.append({"step": step, **_state_snapshot(env)})

        report["rollout"] = {
            "reward_nonfinite_env_steps": reward_nonfinite_env_steps,
            "observation_nonfinite_env_steps": observation_nonfinite_env_steps,
            "terminated_env_steps": terminated_env_steps,
            "truncated_env_steps": truncated_env_steps,
            "reset_capture": reset_stats,
            "samples": samples,
        }
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        print(f"WYW mass initialization diagnostic written to {args_cli.output}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
