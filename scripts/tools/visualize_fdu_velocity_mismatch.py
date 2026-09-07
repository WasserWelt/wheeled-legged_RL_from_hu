"""Record WYW finite-difference versus PhysX joint velocity at 500 Hz.

This uses WYW's current FDU USD and actuator configuration with the training
8/4 PhysX solver iterations and 2 ms physics step.  The root is fixed and
gravity is disabled to isolate the closed-chain joint-coordinate behavior from
balance and ground contact.  The MP4 combines an Isaac camera with scrolling
diagnostics, while the CSV retains every physics sample (video frames are
intentionally only 30 Hz).

Example::

    python scripts/tools/visualize_fdu_velocity_mismatch.py --headless
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=4.0)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--width", type=int, default=800)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--position-iterations", type=int, default=8)
parser.add_argument("--velocity-iterations", type=int, default=4)
parser.add_argument(
    "--output",
    default="docs/fdu_validation/video/fdu_velocity_mismatch.mp4",
)
parser.add_argument("--trace-only", action="store_true", help="skip RTX camera and write CSV/JSON only")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = not args_cli.trace_only
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402
from pxr import UsdPhysics  # noqa: E402

from agent_tasks.direct.wheelbipe.wyw.fdu_mapping import POLICY_JOINT_NAMES  # noqa: E402
from agent_world.assets.wheelbipe_fdu import Wheelbipe_FDU_CFG  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402


PANEL_WIDTH = 760
FD_COLOR = (70, 215, 255)       # BGR
RAW_COLOR = (255, 155, 70)
DIFF_COLOR = (90, 90, 255)
GAP_COLOR = (100, 235, 145)
GRID_COLOR = (64, 67, 74)
TEXT_COLOR = (235, 237, 242)
ALERT_THRESHOLD = 5.0


def _put_text(
    image: np.ndarray,
    value: str,
    xy: tuple[int, int],
    scale: float = 0.48,
    color: tuple[int, int, int] = TEXT_COLOR,
    thickness: int = 1,
) -> None:
    cv2.putText(image, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _body_id(robot, name: str) -> int:
    ids, _ = robot.find_bodies(name)
    if len(ids) != 1:
        raise RuntimeError(f"expected one body {name!r}, got {ids}")
    return int(ids[0])


def _loop_anchors(sim, robot):
    anchors = []
    for prim in sim.stage.Traverse():
        if not (prim.GetName().endswith("_loop1_joint") or prim.GetName().endswith("_loop2_joint")):
            continue
        joint = UsdPhysics.Joint(prim)
        body0 = joint.GetBody0Rel().GetTargets()[0].name
        body1 = joint.GetBody1Rel().GetTargets()[0].name
        local0 = torch.tensor(joint.GetLocalPos0Attr().Get(), dtype=torch.float, device=robot.device)
        local1 = torch.tensor(joint.GetLocalPos1Attr().Get(), dtype=torch.float, device=robot.device)
        anchors.append((prim.GetName(), _body_id(robot, body0), local0, _body_id(robot, body1), local1))
    anchors.sort(key=lambda item: item[0])
    if len(anchors) != 4:
        raise RuntimeError(f"expected four external loop joints, found {[item[0] for item in anchors]}")
    return anchors


def _loop_gaps_mm(robot, anchors) -> list[float]:
    gaps = []
    for _, body0, local0, body1, local1 in anchors:
        world0 = robot.data.body_pos_w[0, body0] + quat_apply(robot.data.body_quat_w[0, body0], local0)
        world1 = robot.data.body_pos_w[0, body1] + quat_apply(robot.data.body_quat_w[0, body1], local1)
        gaps.append(1000.0 * float(torch.linalg.vector_norm(world0 - world1)))
    return gaps


def _phase_action(time_s: float, device: str) -> tuple[torch.Tensor, str]:
    """Zero hold, smooth step, hold, then a small sine sweep."""
    target = torch.linspace(-0.3, 0.3, 6, device=device)
    if time_s < 0.50:
        return torch.zeros(1, 6, device=device), "ZERO ACTION"
    if time_s < 0.90:
        x = (time_s - 0.50) / 0.40
        alpha = x * x * (3.0 - 2.0 * x)
        return (alpha * target).reshape(1, 6), "SMOOTH ACTION RAMP"
    if time_s < 1.70:
        return target.reshape(1, 6), "CONSTANT ACTION"
    omega = 2.0 * math.pi * 0.7
    scale = math.sin(omega * (time_s - 1.70))
    return (scale * target).reshape(1, 6), "SMALL SINE SWEEP"


def _draw_plot(
    panel: np.ndarray,
    rect: tuple[int, int, int, int],
    samples: list[dict],
    key_a: str,
    key_b: str | None,
    limit: float,
    title: str,
    color_a: tuple[int, int, int],
    color_b: tuple[int, int, int] | None = None,
) -> None:
    x0, y0, x1, y1 = rect
    cv2.rectangle(panel, (x0, y0), (x1, y1), (29, 31, 37), -1)
    mid = (y0 + y1) // 2
    cv2.line(panel, (x0, mid), (x1, mid), GRID_COLOR, 1, cv2.LINE_AA)
    cv2.line(panel, (x0, y0), (x0, y1), GRID_COLOR, 1, cv2.LINE_AA)
    _put_text(panel, title, (x0 + 8, y0 + 18), 0.43)
    _put_text(panel, f"+{limit:g}", (x0 + 3, y0 + 37), 0.32, (150, 153, 160))
    _put_text(panel, f"-{limit:g}", (x0 + 3, y1 - 5), 0.32, (150, 153, 160))
    if len(samples) < 2:
        return

    def points(key: str) -> np.ndarray:
        values = np.asarray([float(row[key]) for row in samples], dtype=np.float64)
        values = np.clip(values, -limit, limit)
        xs = np.linspace(x0 + 48, x1 - 4, len(values))
        ys = mid - values / limit * (y1 - y0 - 18) * 0.5
        return np.rint(np.stack((xs, ys), axis=-1)).astype(np.int32)

    cv2.polylines(panel, [points(key_a)], False, color_a, 2, cv2.LINE_AA)
    if key_b is not None and color_b is not None:
        cv2.polylines(panel, [points(key_b)], False, color_b, 2, cv2.LINE_AA)


def _compose_frame(rgb: np.ndarray, samples: list[dict], phase: str, frame_index: int) -> np.ndarray:
    camera = cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(camera, cv2.COLOR_BGR2LAB)
    luminance, channel_a, channel_b = cv2.split(lab)
    luminance = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(luminance)
    camera = cv2.cvtColor(cv2.merge((luminance, channel_a, channel_b)), cv2.COLOR_LAB2BGR)
    panel = np.full((camera.shape[0], PANEL_WIDTH, 3), (20, 22, 27), dtype=np.uint8)
    latest = samples[-1]
    window = samples[-500:]
    max_diff = max(abs(value) for key, value in latest.items() if key.startswith("diff_"))
    alert = max_diff > ALERT_THRESHOLD

    _put_text(panel, "WYW 500 Hz JOINT VELOCITY DIAGNOSTIC", (18, 29), 0.63, TEXT_COLOR, 2)
    _put_text(panel, f"phase: {phase}", (18, 54), 0.47, (180, 230, 255), 1)
    _put_text(panel, "FD = wrapped delta(q) / 0.002 s", (18, 76), 0.43, FD_COLOR, 1)
    _put_text(panel, "PhysX = articulation joint_vel buffer", (330, 76), 0.43, RAW_COLOR, 1)
    status_color = DIFF_COLOR if alert else GAP_COLOR
    _put_text(
        panel,
        f"latest max |FD-PhysX| = {max_diff:7.3f} rad/s  {'MISMATCH' if alert else 'OK'}",
        (18, 100), 0.49, status_color, 2,
    )

    _draw_plot(panel, (18, 116, 742, 250), window, "fd_lf0_Joint", "raw_lf0_Joint", 35.0,
               "lf0 velocity [rad/s]    FD(yellow) / PhysX(blue)", FD_COLOR, RAW_COLOR)
    _draw_plot(panel, (18, 263, 742, 397), window, "fd_l20_Joint", "raw_l20_Joint", 35.0,
               "l20 velocity [rad/s]    FD(yellow) / PhysX(blue)", FD_COLOR, RAW_COLOR)
    _draw_plot(panel, (18, 410, 742, 522), window, "diff_lf0_Joint", "diff_l20_Joint", 30.0,
               "FD - PhysX [rad/s]    lf0(red) / l20(green)", DIFF_COLOR, GAP_COLOR)
    _draw_plot(panel, (18, 535, 742, 637), window, "delta_q_l20_Joint_mrad", "loop_gap_max_mm", 50.0,
               "l20 delta q [mrad] (red) / max loop gap [mm] (green)", DIFF_COLOR, GAP_COLOR)

    _put_text(panel, f"lf0: FD {latest['fd_lf0_Joint']:+8.3f}  PhysX {latest['raw_lf0_Joint']:+8.3f}",
              (18, 665), 0.43)
    _put_text(panel, f"l20: FD {latest['fd_l20_Joint']:+8.3f}  PhysX {latest['raw_l20_Joint']:+8.3f}",
              (18, 688), 0.43)
    _put_text(panel, f"torque lf0/l20: {latest['torque_lf0_Joint']:+6.2f} / {latest['torque_l20_Joint']:+6.2f} Nm",
              (390, 665), 0.40)
    _put_text(panel, f"loop gap max: {latest['loop_gap_max_mm']:.4f} mm", (390, 688), 0.40)

    cv2.rectangle(camera, (12, 12), (560, 92), (14, 16, 20), -1)
    _put_text(camera, "WYW FDU closed-loop isolation (solver 8/4)", (24, 42), 0.60, TEXT_COLOR, 2)
    _put_text(camera, f"t={latest['time_s']:.3f}s  sample={int(latest['sample'])}  {phase}",
              (24, 72), 0.52, (185, 225, 255), 1)
    if alert:
        cv2.rectangle(camera, (12, camera.shape[0] - 54), (575, camera.shape[0] - 12), DIFF_COLOR, 2)
        _put_text(camera, f"FD/PhysX mismatch: {max_diff:.2f} rad/s", (25, camera.shape[0] - 26),
                  0.62, DIFF_COLOR, 2)
    return np.concatenate((camera, panel), axis=1)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if args_cli.seconds <= 0 or args_cli.fps <= 0:
        raise ValueError("--seconds and --fps must be positive")
    if args_cli.width < 640 or args_cli.height < 640:
        raise ValueError("use --width and --height >= 640 for readable diagnostics")

    sim_dt = 0.002
    sim = SimulationContext(
        sim_utils.SimulationCfg(
            dt=sim_dt, render_interval=1, device=args_cli.device,
            gravity=(0.0, 0.0, 0.0),
            physx=sim_utils.PhysxCfg(enable_external_forces_every_iteration=False),
        )
    )
    light_cfg = sim_utils.DomeLightCfg(intensity=700.0, color=(0.85, 0.88, 0.95))
    light_cfg.func("/World/Light", light_cfg)
    cfg = Wheelbipe_FDU_CFG.replace(prim_path="/World/Robot")
    cfg.init_state.pos = (0.0, 0.0, 0.25)
    cfg.spawn = cfg.spawn.replace(
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True,
            enabled_self_collisions=False,
            solver_position_iteration_count=args_cli.position_iterations,
            solver_velocity_iteration_count=args_cli.velocity_iterations,
        )
    )
    robot = Articulation(cfg)
    camera_cfg = CameraCfg(
        prim_path="/World/VelocityDiagnosticCamera",
        update_period=0.0,
        data_types=["rgb"],
        width=args_cli.width,
        height=args_cli.height,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=34.0,
            focus_distance=2.0,
            horizontal_aperture=32.0,
            clipping_range=(0.05, 30.0),
        ),
    )
    camera = None if args_cli.trace_only else Camera(camera_cfg)
    sim.reset()
    if camera is not None:
        camera.set_world_poses_from_view(
            torch.tensor([[1.35, 1.55, 0.75]], dtype=torch.float, device=camera.device),
            torch.tensor([[0.0, 0.0, 0.24]], dtype=torch.float, device=camera.device),
        )
    policy_ids = torch.tensor(
        [robot.joint_names.index(name) for name in POLICY_JOINT_NAMES],
        dtype=torch.long, device=robot.device,
    )
    drive_ids = torch.tensor(
        [robot.joint_names.index(name) for name in ("lf0_Joint", "l20_Joint", "rf0_Joint", "r20_Joint")],
        dtype=torch.long, device=robot.device,
    )
    anchors = _loop_anchors(sim, robot)
    samples: list[dict] = []
    previous_q: torch.Tensor | None = None
    previous_q = robot.data.joint_pos[0, policy_ids].detach().clone()

    def record_sample() -> None:
        nonlocal previous_q
        q = robot.data.joint_pos[0, policy_ids].detach().clone()
        delta_q = torch.atan2(torch.sin(q - previous_q), torch.cos(q - previous_q))
        previous_q = q
        fd = delta_q / sim_dt
        raw = robot.data.joint_vel[0, policy_ids].detach().clone()
        torque = robot.data.applied_torque[0, policy_ids].detach().clone()
        gaps = _loop_gaps_mm(robot, anchors)
        row: dict[str, float | int] = {
            "sample": len(samples),
            "time_s": len(samples) * sim_dt,
            "sim_step_counter": len(samples),
            "loop_gap_max_mm": max(gaps),
            "root_x_m": float(robot.data.root_pos_w[0, 0]),
            "root_z_m": float(robot.data.root_pos_w[0, 2]),
            "root_vx_m_s": float(robot.data.root_lin_vel_w[0, 0]),
            "root_vz_m_s": float(robot.data.root_lin_vel_w[0, 2]),
        }
        for index, name in enumerate(POLICY_JOINT_NAMES):
            row[f"q_{name}"] = float(q[index])
            row[f"delta_q_{name}_mrad"] = 1000.0 * float(delta_q[index])
            row[f"fd_{name}"] = float(fd[index])
            row[f"raw_{name}"] = float(raw[index])
            row[f"diff_{name}"] = float(fd[index] - raw[index])
            row[f"torque_{name}"] = float(torque[index])
        for index, gap in enumerate(gaps):
            row[f"loop_gap_{index}_mm"] = gap
        samples.append(row)

    output = Path(args_cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    if not args_cli.trace_only:
        writer = cv2.VideoWriter(
            str(output), cv2.VideoWriter_fourcc(*"mp4v"), args_cli.fps,
            (args_cli.width + PANEL_WIDTH, args_cli.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"could not open MP4 writer for {output}")

    try:
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(sim_dt)
        record_sample()
        if camera is not None:
            for _ in range(12):
                sim.render()
                camera.update(sim_dt, force_recompute=True)
        frame_count = round(args_cli.seconds * args_cli.fps)
        physics_steps = 0
        for frame_index in range(frame_count):
            target_physics_steps = math.ceil((frame_index + 1) * (1.0 / args_cli.fps) / sim_dt)
            phase = ""
            while physics_steps < target_physics_steps:
                action, phase = _phase_action(physics_steps * sim_dt, robot.device)
                target = action[0, [0, 1, 3, 4]]
                nominal = robot.data.default_joint_pos[0, drive_ids] + 0.5 * target
                robot.set_joint_position_target(nominal[None], joint_ids=drive_ids)
                robot.write_data_to_sim()
                sim.step(render=False)
                robot.update(sim_dt)
                record_sample()
                physics_steps += 1
            if writer is not None and camera is not None:
                sim.render()
                camera.update(sim_dt, force_recompute=True)
                rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
                writer.write(_compose_frame(rgb, samples, phase, frame_index))
    finally:
        if writer is not None:
            writer.release()

    csv_path = output.with_suffix(".csv")
    json_path = output.with_suffix(".json")
    _write_csv(csv_path, samples)
    diff_keys = [f"diff_{name}" for name in POLICY_JOINT_NAMES]
    worst_key, worst_row, worst_value = "", samples[0], -1.0
    for row in samples:
        for key in diff_keys:
            value = abs(float(row[key]))
            if value > worst_value:
                worst_key, worst_row, worst_value = key, row, value
    per_joint = {}
    for name in POLICY_JOINT_NAMES:
        values = np.asarray([float(row[f"diff_{name}"]) for row in samples])
        per_joint[name] = {
            "max_abs_fd_minus_physx_rad_s": float(np.max(np.abs(values))),
            "rms_fd_minus_physx_rad_s": float(np.sqrt(np.mean(values * values))),
            "fraction_abs_difference_gt_5": float(np.mean(np.abs(values) > ALERT_THRESHOLD)),
        }
    report = {
        "video": str(output.resolve()),
        "csv": str(csv_path.resolve()),
        "environment": "Wheelbipe_FDU_CFG, fixed root, one env, no checkpoint",
        "physics_dt_s": sim_dt,
        "decimation": 1,
        "solver_iterations": [
            int(cfg.spawn.articulation_props.solver_position_iteration_count),
            int(cfg.spawn.articulation_props.solver_velocity_iteration_count),
        ],
        "samples": len(samples),
        "worst": {
            "joint": worst_key.removeprefix("diff_"),
            "sample": int(worst_row["sample"]),
            "time_s": float(worst_row["time_s"]),
            "fd_rad_s": float(worst_row[worst_key.replace("diff_", "fd_")]),
            "physx_rad_s": float(worst_row[worst_key.replace("diff_", "raw_")]),
            "difference_rad_s": float(worst_row[worst_key]),
            "delta_q_mrad": float(worst_row[worst_key.replace("diff_", "delta_q_") + "_mrad"]),
            "loop_gap_max_mm": float(worst_row["loop_gap_max_mm"]),
        },
        "per_joint": per_joint,
    }
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not args_cli.trace_only:
        print("FDU VELOCITY MISMATCH VIDEO WRITTEN:", output.resolve(), flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
