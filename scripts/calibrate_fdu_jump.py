"""Measure open-loop takeoff height for the calibrated FDU closed chain.

The robot settles at a vertical tucked length, then switches both legs to a
vertical extended length using the same Jump PD and action path as training.
The resulting report is a calibration diagnostic, not a claim that an
open-loop pulse is an optimal jump policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--tuck-length", type=float, default=0.23)
parser.add_argument("--extend-lengths", type=float, nargs="+", default=(0.29, 0.30, 0.31))
parser.add_argument("--settle-steps", type=int, default=150)
parser.add_argument("--extend-steps", type=int, default=40)
parser.add_argument("--flight-steps", type=int, default=180)
parser.add_argument("--output", default="docs/fdu_validation/jump/fdu_jump_calibration.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

from agent_tasks.direct.wheelbipe.wyw.env import WheelbipeWywEnv  # noqa: E402
from agent_tasks.direct.wheelbipe.wyw.env_cfg import WheelbipeWywJumpEnvCfg_Play  # noqa: E402
from agent_tasks.direct.wheelbipe.wyw.fdu_mapping import inverse_equivalent_kite  # noqa: E402


def _vertical_action(env: WheelbipeWywEnv, lengths: torch.Tensor) -> torch.Tensor:
    theta = torch.zeros_like(lengths)
    front, rear, valid = inverse_equivalent_kite(lengths, theta)
    if not bool(torch.all(valid)):
        raise ValueError("requested leg length is outside the analytic workspace")
    target = torch.stack((front, rear, -front, -rear), dim=-1)
    defaults = env.robot.data.default_joint_pos[:, env._wyw_leg_joint_idx]
    leg_action = (target - defaults) / float(env.leg_action_scale)
    action = torch.zeros(env.num_envs, 6, dtype=torch.float, device=env.device)
    action[:, [0, 1, 3, 4]] = leg_action
    return action


def main() -> None:
    extend_lengths = torch.tensor(args_cli.extend_lengths, dtype=torch.float, device=args_cli.device)
    cfg = WheelbipeWywJumpEnvCfg_Play()
    cfg.seed = 42
    cfg.scene.num_envs = len(extend_lengths)
    cfg.scene.env_spacing = 2.0
    cfg.sim.device = args_cli.device
    cfg.play = False
    cfg.commands.debug_vis = False
    env = WheelbipeWywEnv(cfg)
    try:
        env.reset()
        env.command.zero_()
        tuck = torch.full_like(extend_lengths, args_cli.tuck_length)
        tuck_action = _vertical_action(env, tuck)
        extend_action = _vertical_action(env, extend_lengths)

        for _ in range(args_cli.settle_steps):
            env.step(tuck_action)
        start_z = env.robot.data.root_pos_w[:, 2].clone()
        min_z = start_z.clone()
        peak_z = start_z.clone()
        peak_vz = env.robot.data.root_lin_vel_w[:, 2].clone()
        airborne_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        first_takeoff_step = torch.full_like(airborne_steps, -1)
        first_landing_step = torch.full_like(airborne_steps, -1)
        was_airborne = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        total_steps = args_cli.extend_steps + args_cli.flight_steps
        for step in range(total_steps):
            action = extend_action if step < args_cli.extend_steps else tuck_action
            env.step(action)
            root_z = env.robot.data.root_pos_w[:, 2]
            root_vz = env.robot.data.root_lin_vel_w[:, 2]
            min_z = torch.minimum(min_z, root_z)
            peak_z = torch.maximum(peak_z, root_z)
            peak_vz = torch.maximum(peak_vz, root_vz)
            forces = env.contact_sensor.data.net_forces_w[:, env._desired_contact_link_idx, 2]
            airborne = torch.all(forces <= env.cfg.wyw_flight_contact_force, dim=1)
            new_takeoff = airborne & ~was_airborne & (first_takeoff_step < 0) & (step > 1)
            first_takeoff_step[new_takeoff] = step
            new_landing = ~airborne & was_airborne & (first_takeoff_step >= 0) & (first_landing_step < 0)
            first_landing_step[new_landing] = step
            airborne_steps += airborne.long()
            was_airborne = airborne

        rows = []
        for i, length in enumerate(args_cli.extend_lengths):
            rows.append(
                {
                    "extend_length_m": length,
                    "start_root_z_m": float(start_z[i]),
                    "minimum_root_z_m": float(min_z[i]),
                    "peak_root_z_m": float(peak_z[i]),
                    "height_gain_m": float(peak_z[i] - start_z[i]),
                    "peak_vertical_velocity_m_s": float(peak_vz[i]),
                    "airborne_steps": int(airborne_steps[i]),
                    "first_takeoff_step": int(first_takeoff_step[i]),
                    "first_landing_step": int(first_landing_step[i]),
                }
            )
        report = {
            "tuck_length_m": args_cli.tuck_length,
            "policy_dt_s": env.step_dt,
            "settle_steps": args_cli.settle_steps,
            "extend_steps": args_cli.extend_steps,
            "flight_steps": args_cli.flight_steps,
            "jump_leg_pd": {"stiffness": 6.0, "damping": 0.5, "effort_limit": 40.0},
            "samples": rows,
        }
        out = Path(args_cli.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"FDU JUMP CALIBRATION WRITTEN: {out}")
        print(json.dumps(rows, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
