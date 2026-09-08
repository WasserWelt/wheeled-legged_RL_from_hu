"""Record an Isaac Lab WYW policy rollout and a replayable action tape.

The trace and action tape are sampled at the 100 Hz policy rate.  The actions
can be replayed without policy feedback in another simulator to isolate
plant-model divergence.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--command-vx", type=float, default=1.0)
parser.add_argument("--command-yaw", type=float, default=0.0)
parser.add_argument("--command-height", type=float, default=0.22)
parser.add_argument("--settle-s", type=float, default=0.5)
parser.add_argument("--score-s", type=float, default=3.0)
parser.add_argument("--num-envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import agent_tasks  # noqa: E402,F401
import agent_world  # noqa: E402,F401
from agent_rl.rsl_rl.env import RslRlVecEnvWrapper  # noqa: E402
from agent_rl.rsl_rl.runners import OnPolicySequenceRunner  # noqa: E402
from agent_tasks.direct.wheelbipe.wyw.fdu_mapping import POLICY_JOINT_NAMES  # noqa: E402


TASK_ID = "Robotics-Wheelbipe-FDU-wyw-Flat-Play-v1"


def _heading_vx(robot) -> torch.Tensor:
    w, x, y, z = robot.data.root_quat_w[0]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    velocity = robot.data.root_lin_vel_w[0]
    return torch.cos(yaw) * velocity[0] + torch.sin(yaw) * velocity[1]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    checkpoint = args_cli.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args_cli.settle_s < 0.0 or args_cli.score_s <= 0.0:
        raise ValueError("settle-s must be non-negative and score-s must be positive")
    if args_cli.num_envs <= 0:
        raise ValueError("num-envs must be positive")
    output_dir = args_cli.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(TASK_ID, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = 42
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.play = True
    env_cfg.play_keep_done_reset = True
    env_cfg.commands.debug_vis = False
    env_cfg.height_scanner.debug_vis = False
    env_cfg.play_ang_vel_z_debug_vis = False
    env_cfg.wyw_flat_command_curriculum_enabled = False
    env_cfg.curriculum = None
    agent_cfg = load_cfg_from_registry(TASK_ID, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args_cli.device
    gym_env = gym.make(TASK_ID, cfg=env_cfg)
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    raw = env.unwrapped
    runner = OnPolicySequenceRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=raw.device)
    raw.use_self_obs_noise = False

    ids = torch.arange(raw.num_envs, device=raw.device)
    raw.set_wyw_command_override(ids, vx=0.0, yaw=0.0, height=args_cli.command_height)
    observations, _ = env.reset()
    policy_joint_ids = raw._wyw_policy_joint_idx
    trace_values: list[torch.Tensor] = []
    trace_policy_steps: list[int] = []
    trace_phases: list[str] = []
    trace_commands: list[tuple[float, float, float]] = []
    action_tensors: list[torch.Tensor] = []
    done_tensors: list[torch.Tensor] = []
    commands: list[list[float]] = []
    phases: list[str] = []
    current_policy_step = -1
    current_phase = "settle"
    current_command = (0.0, 0.0, args_cli.command_height)

    def record_state() -> None:
        q = raw.robot.data.joint_pos[0, policy_joint_ids]
        qd = raw._wyw_joint_velocity[0]
        raw_qd = raw.robot.data.joint_vel[0, policy_joint_ids]
        torque = raw.robot.data.applied_torque[0, policy_joint_ids]
        action = raw._actions[0]
        values = torch.cat(
            (
                raw.robot.data.root_pos_w[0, 0].view(1),
                _heading_vx(raw.robot).view(1),
                q,
                qd,
                raw_qd,
                torque,
                action,
            )
        ).detach()
        trace_values.append(values.clone())
        trace_policy_steps.append(current_policy_step)
        trace_phases.append(current_phase)
        trace_commands.append(current_command)

    def build_rows(values_array) -> list[dict]:
        rows: list[dict] = []
        for sample, values in enumerate(values_array):
            root_x, root_vx = values[:2]
            offset = 2
            q = values[offset : offset + 6]
            offset += 6
            qd = values[offset : offset + 6]
            offset += 6
            raw_qd = values[offset : offset + 6]
            offset += 6
            torque = values[offset : offset + 6]
            offset += 6
            action = values[offset : offset + 6]
            command = trace_commands[sample]
            row: dict[str, float | int | str] = {
                "sample": sample,
                "time_s": (sample + 1) * float(raw.step_dt),
                "policy_step": trace_policy_steps[sample],
                "substep": int(raw.cfg.decimation) - 1,
                "phase": trace_phases[sample],
                "command_vx": command[0],
                "command_yaw": command[1],
                "command_height": command[2],
                "root_x_m": float(root_x),
                "root_vx_m_s": float(root_vx),
            }
            for index, name in enumerate(POLICY_JOINT_NAMES):
                row[f"action_{name}"] = float(action[index])
                row[f"q_{name}"] = float(q[index])
                row[f"qd_{name}"] = float(qd[index])
                row[f"raw_qd_{name}"] = float(raw_qd[index])
                row[f"tau_{name}"] = float(torque[index])
            rows.append(row)
        return rows

    settle_steps = round(args_cli.settle_s / float(raw.step_dt))
    score_steps = round(args_cli.score_s / float(raw.step_dt))
    for step in range(settle_steps + score_steps):
        if step % 25 == 0:
            print(f"Recording policy step {step}/{settle_steps + score_steps}", flush=True)
        if step == settle_steps:
            current_phase = "score"
            current_command = (
                args_cli.command_vx,
                args_cli.command_yaw,
                args_cli.command_height,
            )
            raw.set_wyw_command_override(
                ids,
                vx=args_cli.command_vx,
                yaw=args_cli.command_yaw,
                height=args_cli.command_height,
            )
            block = raw._get_wyw_command_block()
            observations["policy"][:, 6:9] = block
            history = observations["policy_hist"].view(raw.num_envs, raw.cfg.num_obs_hist, -1)
            history[:, -1, 6:9] = block
        current_policy_step = step
        with torch.inference_mode():
            action = policy(observations)
        if env.clip_actions is not None:
            action = torch.clamp(action, -float(env.clip_actions), float(env.clip_actions))
        action_tensors.append(action[0].detach().clone())
        commands.append(list(current_command))
        phases.append(current_phase)
        observations, _, terminated, truncated, _ = env.step(action)
        done_tensors.append((terminated | truncated).detach().clone())
        record_state()

    done_array = torch.stack(done_tensors).cpu().numpy()[:, 0]
    if done_array.any():
        raise RuntimeError(f"Isaac rollout terminated at policy step {int(done_array.argmax())}")
    actions = torch.stack(action_tensors).cpu().tolist()
    trace_array = torch.stack(trace_values).cpu().numpy()
    rows = build_rows(trace_array)
    trace_path = output_dir / "isaac_trace.csv"
    tape_path = output_dir / "action_tape.json"
    _write_csv(trace_path, rows)
    tape = {
        "schema_version": 1,
        "source": "Isaac Lab WYW Sequence policy",
        "checkpoint": str(checkpoint),
        "task_id": TASK_ID,
        "seed": 42,
        "num_envs": args_cli.num_envs,
        "physics_dt": float(env_cfg.sim.dt),
        "policy_dt": float(env_cfg.sim.dt * env_cfg.decimation),
        "trace_dt": float(env_cfg.sim.dt * env_cfg.decimation),
        "decimation": int(env_cfg.decimation),
        "settle_steps": settle_steps,
        "score_steps": score_steps,
        "scenario": {
            "vx": args_cli.command_vx,
            "yaw": args_cli.command_yaw,
            "height": args_cli.command_height,
        },
        "policy_joint_names": list(POLICY_JOINT_NAMES),
        "actions": actions,
        "commands": commands,
        "phases": phases,
        "isaac_trace": str(trace_path),
    }
    tape_path.write_text(json.dumps(tape, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {tape_path}")
    print(f"Wrote {trace_path} ({len(rows)} policy-rate samples)")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
