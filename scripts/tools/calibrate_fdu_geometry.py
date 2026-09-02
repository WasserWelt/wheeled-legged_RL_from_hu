"""Scan the FDU closed-chain workspace and compare it with the kite solver.

The scan is deliberately separate from training.  It drives only the four
entity bars, waits for the PhysX loop constraints to settle, and writes a JSON
report containing reachable ranges, action signs, Jacobian conditioning and
the analytic-vs-physical wheel-point residual.  Run on the host with Isaac
Lab/GPU access (CPU is also supported):

    python scripts/calibrate_fdu_geometry.py --headless --grid 9

No training config is changed by this script.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--grid", type=int, default=9, help="samples per driven-bar angle (odd >= 1)")
parser.add_argument("--angle-limit", type=float, default=0.8, help="symmetric scan range in rad")
parser.add_argument("--center", type=float, default=None, help="left joint center; default is the vertical-pose solution")
parser.add_argument(
    "--target-lengths",
    type=float,
    nargs="+",
    default=None,
    help="scan theta0=0 inverse-kinematic targets instead of a rectangular joint grid",
)
parser.add_argument("--ramp-steps", type=int, default=80, help="steps used to approach each target continuously")
parser.add_argument("--settle-steps", type=int, default=120)
parser.add_argument("--tracking-threshold", type=float, default=0.05)
parser.add_argument("--drive-stiffness", type=float, default=20.0)
parser.add_argument("--drive-damping", type=float, default=1.0)
parser.add_argument("--drive-effort-limit", type=float, default=40.0)
parser.add_argument("--output", default="docs/fdu_validation/geometry/fdu_geometry_scan.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import IdealPDActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402

from agent_tasks.direct.wheelbipe.wyw.fdu_mapping import (  # noqa: E402
    FDU_DEFAULT_OFFSET,
    FDU_VERTICAL_JOINT_CENTER,
    compute_fdu_equivalent_leg_state,
    equivalent_kite_jacobian,
    inverse_equivalent_kite,
)
from agent_world.assets.wheelbipe_fdu import Wheelbipe_FDU_CFG  # noqa: E402


def _body_index(robot: Articulation, name: str) -> int:
    ids, _ = robot.find_bodies(name)
    if len(ids) != 1:
        raise RuntimeError(f"expected one body {name!r}, got {ids}")
    return int(ids[0])


def main() -> None:
    if args_cli.grid < 1 or args_cli.grid % 2 == 0:
        raise ValueError("--grid must be an odd integer >= 1")
    device = args_cli.device
    sim = SimulationContext(
        sim_utils.SimulationCfg(dt=0.005, device=device, gravity=(0.0, 0.0, 0.0))
    )
    cfg = Wheelbipe_FDU_CFG.replace(prim_path="/World/Robot")
    cfg.spawn = cfg.spawn.replace(
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True,
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=6,
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
            stiffness=0.0, damping=0.05, effort_limit=100.0, velocity_limit=300.0
        ),
    }
    robot = Articulation(cfg)
    sim.reset()
    drive_names = ("lf0_Joint", "l20_Joint", "rf0_Joint", "r20_Joint")
    drive_ids = [robot.joint_names.index(name) for name in drive_names]
    root_id = _body_index(robot, "base_link_del")
    wheel_ids = [_body_index(robot, "l_wheel_Link"), _body_index(robot, "r_wheel_Link")]
    hip_ids = [_body_index(robot, "lf0_Link"), _body_index(robot, "rf0_Link")]
    center = FDU_VERTICAL_JOINT_CENTER if args_cli.center is None else float(args_cli.center)
    if args_cli.target_lengths:
        lengths = torch.tensor(args_cli.target_lengths, dtype=torch.float, device=device)
        zeros = torch.zeros_like(lengths)
        target_front, target_rear, valid = inverse_equivalent_kite(lengths, zeros)
        if not bool(torch.all(valid)):
            invalid = lengths[~valid].detach().cpu().tolist()
            raise ValueError(f"unreachable --target-lengths values: {invalid}")
        target_pairs = list(zip(target_front, target_rear))
    else:
        q_offsets = (
            torch.zeros(1, device=device)
            if args_cli.grid == 1
            else torch.linspace(-args_cli.angle_limit, args_cli.angle_limit, args_cli.grid, device=device)
        )
        q_values = center + q_offsets
        target_pairs = [(qf, qr) for qf in q_values for qr in q_values]
    rows: list[dict] = []
    center_joint_positions: dict[str, float] | None = None
    zero_pos = robot.data.default_joint_pos.clone()
    zero_vel = torch.zeros_like(zero_pos)

    for lf0, l20 in target_pairs:
            # Remove path/branch history before approaching every point.
            robot.write_joint_state_to_sim(zero_pos, zero_vel)
            robot.set_joint_position_target(zero_pos)
            for _ in range(max(10, args_cli.settle_steps // 4)):
                robot.write_data_to_sim()
                sim.step()
                robot.update(sim.get_physics_dt())
            # URDF joint axes are mirrored: a symmetric physical pose uses
            # [lf0,l20,rf0,r20] = [qf,qr,-qf,-qr].
            target = torch.stack((lf0, l20, -lf0, -l20)).reshape(1, 4)
            # Ramp into the target so a valid branch is not rejected merely
            # because a maximal-coordinate loop received a discontinuous step.
            for step in range(args_cli.ramp_steps):
                alpha = (step + 1) / args_cli.ramp_steps
                robot.set_joint_position_target(alpha * target, joint_ids=drive_ids)
                robot.write_data_to_sim()
                sim.step()
                robot.update(sim.get_physics_dt())
            # The ramp only reaches alpha=1 on its final physics step.  Hold
            # the final target separately; otherwise the recorded error is a
            # transient lag rather than a dynamic-reachability measurement.
            robot.set_joint_position_target(target, joint_ids=drive_ids)
            for _ in range(args_cli.settle_steps):
                robot.write_data_to_sim()
                sim.step()
                robot.update(sim.get_physics_dt())

            target_analytic = compute_fdu_equivalent_leg_state(
                lf0[None], l20[None], -lf0[None], -l20[None]
            )
            root_pos = robot.data.body_pos_w[:, root_id]
            root_quat = robot.data.body_quat_w[:, root_id]
            wheel_pos = robot.data.body_pos_w[:, wheel_ids] - root_pos[:, None]
            hip_pos = robot.data.body_pos_w[:, hip_ids] - root_pos[:, None]
            wheel_b = quat_apply_inverse(root_quat[:, None].expand(-1, 2, -1), wheel_pos)
            hip_b = quat_apply_inverse(root_quat[:, None].expand(-1, 2, -1), hip_pos)
            delta = wheel_b[:, :, (0, 2)] - hip_b[:, :, (0, 2)]
            actual_l0 = torch.linalg.vector_norm(delta, dim=-1)[0]
            actual_theta = torch.atan2(-delta[0, :, 0], -delta[0, :, 1])
            actual_q = robot.data.joint_pos[0, drive_ids]
            if (
                abs(float(lf0) - center) < 1.0e-6
                and abs(float(l20) - center) < 1.0e-6
            ):
                center_joint_positions = {
                    name: float(robot.data.joint_pos[0, joint_id])
                    for joint_id, name in enumerate(robot.joint_names)
                }
            actual_analytic = compute_fdu_equivalent_leg_state(
                actual_q[0:1], actual_q[1:2], actual_q[2:3], actual_q[3:4]
            )
            for side_idx, side in enumerate(("left", "right")):
                target_l0 = target_analytic[0 if side_idx == 0 else 2][0]
                target_theta = target_analytic[1 if side_idx == 0 else 3][0]
                expected_l0 = actual_analytic[0 if side_idx == 0 else 2][0]
                expected_theta = actual_analytic[1 if side_idx == 0 else 3][0]
                theta_error = torch.atan2(
                    torch.sin(actual_theta[side_idx] - expected_theta),
                    torch.cos(actual_theta[side_idx] - expected_theta),
                )
                if side_idx == 0:
                    side_drive_ids = (0, 1)
                    absolute_front = FDU_DEFAULT_OFFSET + actual_q[1:2]
                    absolute_rear = actual_q[0:1]
                else:
                    side_drive_ids = (2, 3)
                    absolute_front = FDU_DEFAULT_OFFSET - actual_q[3:4]
                    absolute_rear = -actual_q[2:3]
                jac = equivalent_kite_jacobian(absolute_front, absolute_rear)[0]
                singular_values = torch.linalg.svdvals(jac)
                condition = singular_values.max() / singular_values.min().clamp_min(1.0e-8)
                rows.append(
                    {
                        "side": side,
                        "front_target": float(target[0, 0 if side_idx == 0 else 2]),
                        "rear_target": float(target[0, 1 if side_idx == 0 else 3]),
                        "front_offset_from_center": float(target[0, 0 if side_idx == 0 else 2] - (center if side_idx == 0 else -center)),
                        "rear_offset_from_center": float(target[0, 1 if side_idx == 0 else 3] - (center if side_idx == 0 else -center)),
                        "front_actual": float(actual_q[0 if side_idx == 0 else 2]),
                        "rear_actual": float(actual_q[1 if side_idx == 0 else 3]),
                        "target_analytic_l0": float(target_l0),
                        "target_analytic_theta0": float(target_theta),
                        "actual_analytic_l0": float(expected_l0),
                        "actual_analytic_theta0": float(expected_theta),
                        "physical_l0": float(actual_l0[side_idx]),
                        "physical_theta0": float(actual_theta[side_idx]),
                        "wheel_x_from_root": float(wheel_b[0, side_idx, 0]),
                        "wheel_z_from_root": float(wheel_b[0, side_idx, 2]),
                        "root_height_for_ground_contact": float(0.06 - wheel_b[0, side_idx, 2]),
                        "geometry_l0_error": float(actual_l0[side_idx] - expected_l0),
                        "geometry_theta0_error": float(theta_error),
                        "joint_tracking_error": float(
                            torch.max(torch.abs(actual_q[list(side_drive_ids)] - target[0, list(side_drive_ids)]))
                        ),
                        "jacobian_condition": float(condition),
                        "d_l0_d_front": float(jac[0, 1]),
                        "d_l0_d_rear": float(jac[0, 0]),
                        "d_theta0_d_front": float(jac[1, 1]),
                        "d_theta0_d_rear": float(jac[1, 0]),
                    }
                )

    finite = [r for r in rows if math.isfinite(r["physical_l0"])]
    reached = [r for r in finite if r["joint_tracking_error"] <= args_cli.tracking_threshold]
    report = {
        "source_urdf": "robot_models/fdu_infantry_V4_mujoco/meshes/infantry_V2.urdf",
        "grid": args_cli.grid,
        "angle_limit_rad": args_cli.angle_limit,
        "target_lengths_m": args_cli.target_lengths,
        "left_joint_center_rad": center,
        "tracking_threshold_rad": args_cli.tracking_threshold,
        "ramp_steps": args_cli.ramp_steps,
        "settle_steps": args_cli.settle_steps,
        "drive": {
            "stiffness": args_cli.drive_stiffness,
            "damping": args_cli.drive_damping,
            "effort_limit": args_cli.drive_effort_limit,
        },
        "analytic_geometry": {"crank_length": 0.17472, "coupler_length": 0.208, "default_offset": FDU_DEFAULT_OFFSET},
        "center_joint_positions": center_joint_positions,
        "samples": rows,
        "summary": {
            "sample_count": len(rows),
            "finite_count": len(finite),
            "reached_count": len(reached),
            "physical_l0_min": min(r["physical_l0"] for r in finite) if finite else None,
            "physical_l0_max": max(r["physical_l0"] for r in finite) if finite else None,
            "reached_physical_l0_min": min(r["physical_l0"] for r in reached) if reached else None,
            "reached_physical_l0_max": max(r["physical_l0"] for r in reached) if reached else None,
            "max_abs_geometry_l0_error": max(abs(r["geometry_l0_error"]) for r in finite) if finite else None,
            "max_abs_geometry_theta0_error": max(abs(r["geometry_theta0_error"]) for r in finite) if finite else None,
            "reached_max_abs_geometry_l0_error": max(abs(r["geometry_l0_error"]) for r in reached) if reached else None,
            "reached_max_abs_geometry_theta0_error": max(abs(r["geometry_theta0_error"]) for r in reached) if reached else None,
            "max_abs_joint_tracking_error": max(
                r["joint_tracking_error"]
                for r in finite
            ) if finite else None,
            "reached_max_jacobian_condition": max(r["jacobian_condition"] for r in reached) if reached else None,
            "center_root_height_for_ground_contact": next(
                (r["root_height_for_ground_contact"] for r in reached if abs(r["front_offset_from_center"]) < 1.0e-6 and abs(r["rear_offset_from_center"]) < 1.0e-6),
                None,
            ),
        },
    }
    out = Path(args_cli.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"FDU GEOMETRY SCAN WRITTEN: {out}")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    try:
        main()
    finally:
        # This script does not use Replicator.  Isaac Sim 5.1 may otherwise
        # wait indefinitely in global Kit/cache cleanup after the JSON has
        # already been finalized, leaving a calibration process behind.
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
