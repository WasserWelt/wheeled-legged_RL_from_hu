"""Measure static jitter of the closed-chain FDU leg without camera/rendering.

This diagnostic separates actual physics jitter from 30 FPS video aliasing.
For each requested L0 it ramps both mirrored legs to the same vertical target,
settles, and records joint velocity, target tracking, physical L0 variation,
left/right symmetry, and all four loop-anchor residuals.

Run the current and recommended PhysX settings as an A/B pair::

    python scripts/diagnose_fdu_jitter.py --headless \
        --output docs/fdu_validation/jitter/fdu_jitter_baseline.json
    python scripts/diagnose_fdu_jitter.py --headless \
        --external-forces-every-iteration --velocity-iterations 2 \
        --output docs/fdu_validation/jitter/fdu_jitter_recommended.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--lengths", type=float, nargs="+", default=(0.10, 0.20, 0.285, 0.335))
parser.add_argument("--ramp-steps", type=int, default=300)
parser.add_argument("--settle-steps", type=int, default=300)
parser.add_argument("--sample-steps", type=int, default=400)
parser.add_argument("--dt", type=float, default=0.005)
parser.add_argument("--velocity-iterations", type=int, default=6)
parser.add_argument("--position-iterations", type=int, default=16)
parser.add_argument("--external-forces-every-iteration", action="store_true")
parser.add_argument("--drive-stiffness", type=float, default=20.0)
parser.add_argument("--drive-damping", type=float, default=1.0)
parser.add_argument("--drive-effort-limit", type=float, default=40.0)
parser.add_argument("--passive-damping", type=float, default=0.01)
parser.add_argument("--usd", default=None, help="optional USD override for constraint A/B tests")
parser.add_argument("--output", default="docs/fdu_validation/jitter/fdu_jitter_diagnostic.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import IdealPDActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402
from pxr import UsdPhysics  # noqa: E402

from agent_tasks.direct.wheelbipe.wyw.fdu_mapping import inverse_equivalent_kite  # noqa: E402
from agent_world.assets.wheelbipe_fdu import Wheelbipe_FDU_CFG  # noqa: E402


def _rms(value: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean(torch.square(value))))


def _body_id(robot: Articulation, name: str) -> int:
    ids, _ = robot.find_bodies(name)
    if len(ids) != 1:
        raise RuntimeError(f"expected one body {name!r}, got {ids}")
    return int(ids[0])


def _loop_anchors(sim: SimulationContext, robot: Articulation) -> list[tuple[str, int, torch.Tensor, int, torch.Tensor]]:
    anchors = []
    for prim in sim.stage.Traverse():
        if not prim.GetName().endswith("_loop1_joint") and not prim.GetName().endswith("_loop2_joint"):
            continue
        joint = UsdPhysics.Joint(prim)
        body0 = joint.GetBody0Rel().GetTargets()[0].name
        body1 = joint.GetBody1Rel().GetTargets()[0].name
        local0 = torch.tensor(joint.GetLocalPos0Attr().Get(), dtype=torch.float, device=robot.device)
        local1 = torch.tensor(joint.GetLocalPos1Attr().Get(), dtype=torch.float, device=robot.device)
        anchors.append((prim.GetName(), _body_id(robot, body0), local0, _body_id(robot, body1), local1))
    if len(anchors) != 4:
        raise RuntimeError(f"expected four external loop joints, found {[item[0] for item in anchors]}")
    return anchors


def _loop_gaps(robot: Articulation, anchors) -> torch.Tensor:
    gaps = []
    for _, body0, local0, body1, local1 in anchors:
        world0 = robot.data.body_pos_w[0, body0] + quat_apply(robot.data.body_quat_w[0, body0], local0)
        world1 = robot.data.body_pos_w[0, body1] + quat_apply(robot.data.body_quat_w[0, body1], local1)
        gaps.append(torch.linalg.vector_norm(world0 - world1))
    return torch.stack(gaps)


def main() -> None:
    if min(args_cli.ramp_steps, args_cli.settle_steps, args_cli.sample_steps) < 1:
        raise ValueError("ramp, settle and sample steps must all be positive")
    sim_dt = float(args_cli.dt)
    if sim_dt <= 0.0:
        raise ValueError("--dt must be positive")
    sim_cfg = sim_utils.SimulationCfg(
        dt=sim_dt,
        device=args_cli.device,
        gravity=(0.0, 0.0, 0.0),
        physx=sim_utils.PhysxCfg(
            enable_external_forces_every_iteration=args_cli.external_forces_every_iteration,
        ),
    )
    sim = SimulationContext(sim_cfg)
    cfg = Wheelbipe_FDU_CFG.replace(prim_path="/World/Robot")
    if args_cli.usd is not None:
        cfg.spawn = cfg.spawn.replace(usd_path=str(Path(args_cli.usd).resolve()))
    cfg.spawn = cfg.spawn.replace(
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True,
            enabled_self_collisions=False,
            solver_position_iteration_count=args_cli.position_iterations,
            solver_velocity_iteration_count=args_cli.velocity_iterations,
        )
    )
    cfg.actuators = {
        "drive": IdealPDActuatorCfg(
            joint_names_expr=["lf0_Joint", "l20_Joint", "rf0_Joint", "r20_Joint"],
            stiffness=args_cli.drive_stiffness,
            damping=args_cli.drive_damping,
            effort_limit=args_cli.drive_effort_limit,
            velocity_limit=30.0,
        ),
        "passive": IdealPDActuatorCfg(
            joint_names_expr=[
                "rf1_Joint", "lf1_Joint", "r21_Joint", "r22_Joint", "r23_Joint",
                "l21_Joint", "l22_Joint", "l23_Joint", "l_wheel_Joint", "r_wheel_Joint",
            ],
            stiffness=0.0,
            damping=args_cli.passive_damping,
            effort_limit=100.0,
            velocity_limit=300.0,
            armature=0.0001,
        ),
    }
    robot = Articulation(cfg)
    sim.reset()

    drive_names = ("lf0_Joint", "l20_Joint", "rf0_Joint", "r20_Joint")
    drive_ids = [robot.joint_names.index(name) for name in drive_names]
    passive_ids = [index for index in range(robot.num_joints) if index not in drive_ids]
    hip_ids = [_body_id(robot, "lf0_Link"), _body_id(robot, "rf0_Link")]
    wheel_ids = [_body_id(robot, "l_wheel_Link"), _body_id(robot, "r_wheel_Link")]
    anchors = _loop_anchors(sim, robot)
    previous_target = robot.data.joint_pos[0, drive_ids].clone()
    rows = []

    for length in args_cli.lengths:
        lengths = torch.full((2,), float(length), device=robot.device)
        zero_theta = torch.zeros_like(lengths)
        common_front, common_rear, valid = inverse_equivalent_kite(lengths, zero_theta)
        if not bool(torch.all(valid)):
            raise ValueError(f"L0={length} m violates the specified passive-joint limits")
        target = torch.stack(
            (common_front[0], common_rear[0], -common_front[1], -common_rear[1])
        )
        for step in range(args_cli.ramp_steps):
            alpha = (step + 1) / args_cli.ramp_steps
            command = previous_target + alpha * (target - previous_target)
            robot.set_joint_position_target(command[None], joint_ids=drive_ids)
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(sim_dt)
        previous_target = target
        robot.set_joint_position_target(target[None], joint_ids=drive_ids)
        for _ in range(args_cli.settle_steps):
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(sim_dt)

        drive_pos, drive_vel, passive_pos, passive_vel, physical_l0, loop_gap = [], [], [], [], [], []
        for _ in range(args_cli.sample_steps):
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(sim_dt)
            drive_pos.append(robot.data.joint_pos[0, drive_ids].clone())
            drive_vel.append(robot.data.joint_vel[0, drive_ids].clone())
            passive_pos.append(robot.data.joint_pos[0, passive_ids].clone())
            passive_vel.append(robot.data.joint_vel[0, passive_ids].clone())
            delta = robot.data.body_pos_w[0, wheel_ids] - robot.data.body_pos_w[0, hip_ids]
            physical_l0.append(torch.linalg.vector_norm(delta[:, (0, 2)], dim=-1))
            loop_gap.append(_loop_gaps(robot, anchors))

        drive_pos_t = torch.stack(drive_pos)
        drive_vel_t = torch.stack(drive_vel)
        passive_pos_t = torch.stack(passive_pos)
        passive_vel_t = torch.stack(passive_vel)
        physical_l0_t = torch.stack(physical_l0)
        loop_gap_t = torch.stack(loop_gap)
        tracking = drive_pos_t - target
        rows.append({
            "target_l0_m": float(length),
            "drive_target_rad": target.detach().cpu().tolist(),
            "drive_tracking_rms_rad": _rms(tracking),
            "drive_tracking_max_rad": float(torch.max(torch.abs(tracking))),
            "drive_velocity_rms_rad_s": _rms(drive_vel_t),
            "drive_velocity_max_rad_s": float(torch.max(torch.abs(drive_vel_t))),
            "passive_velocity_rms_rad_s": _rms(passive_vel_t),
            "passive_velocity_max_rad_s": float(torch.max(torch.abs(passive_vel_t))),
            "passive_position_min_rad": torch.min(passive_pos_t, dim=0).values.cpu().tolist(),
            "passive_position_max_rad": torch.max(passive_pos_t, dim=0).values.cpu().tolist(),
            "physical_l0_mean_m": torch.mean(physical_l0_t, dim=0).cpu().tolist(),
            "physical_l0_std_m": torch.std(physical_l0_t, dim=0).cpu().tolist(),
            "physical_l0_peak_to_peak_m": (
                torch.max(physical_l0_t, dim=0).values - torch.min(physical_l0_t, dim=0).values
            ).cpu().tolist(),
            "left_right_l0_rms_error_m": _rms(physical_l0_t[:, 0] - physical_l0_t[:, 1]),
            "loop_gap_rms_mm": 1000.0 * _rms(loop_gap_t),
            "loop_gap_max_mm": 1000.0 * float(torch.max(loop_gap_t)),
        })

    report = {
        "physics": {
            "usd": str(Path(args_cli.usd).resolve()) if args_cli.usd else "project default",
            "dt_s": sim_dt,
            "position_iterations": args_cli.position_iterations,
            "velocity_iterations": args_cli.velocity_iterations,
            "external_forces_every_iteration": args_cli.external_forces_every_iteration,
            "self_collisions_enabled": False,
        },
        "actuators": {
            "drive_stiffness": args_cli.drive_stiffness,
            "drive_damping": args_cli.drive_damping,
            "drive_effort_limit": args_cli.drive_effort_limit,
            "passive_damping": args_cli.passive_damping,
        },
        "samples": rows,
    }
    output = Path(args_cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"FDU JITTER DIAGNOSTIC WRITTEN: {output}", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
