"""Render the FDU closed-chain drive bars and equivalent ``L0/phi0`` to MP4.

The left side of the video is an Isaac Sim render of the authoritative
``infantry_V2.urdf``-derived USD.  Colored cylinders are overlaid on the four
physical drive bars and on each hip-to-wheel virtual leg.  The right side is
an analytical kite diagram computed from the *measured* joint positions at
the same frame, including the vertical-zero ``phi0`` arc and live values.

Example (from the repository root)::

    python scripts/visualize_fdu_l0_phi0.py --headless \
        --output docs/fdu_validation/video/fdu_l0_phi0.mp4

The script enables RTX cameras itself.  It fixes the root and disables gravity
so the recording isolates linkage geometry rather than balance control.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="docs/fdu_validation/video/fdu_l0_phi0.mp4")
parser.add_argument("--seconds", type=float, default=10.0)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument(
    "--physics-hz",
    type=float,
    default=400.0,
    help="Physics and continuous command update frequency (default: 400 Hz)",
)
parser.add_argument("--width", type=int, default=800, help="Isaac render width")
parser.add_argument("--height", type=int, default=720)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils.math import quat_from_matrix  # noqa: E402

from agent_tasks.direct.wheelbipe.wyw.fdu_mapping import (  # noqa: E402
    FDU_DEFAULT_OFFSET,
    FDU_L1,
    FDU_L2,
    FDU_LEFT_PASSIVE_LIMITS,
    FDU_MECHANICAL_L0_MAX,
    compute_fdu_equivalent_leg_state,
    inverse_equivalent_kite,
    solve_equivalent_kite,
)
from agent_world.assets.wheelbipe_fdu import Wheelbipe_FDU_CFG  # noqa: E402


COL_FRONT = (255, 60, 210)   # RGB: lf0/rf0 physical front drive bar
COL_REAR = (30, 220, 255)    # RGB: l20/r20 physical rear drive bar
COL_L0 = (255, 220, 30)      # RGB: equivalent hip-to-wheel leg
COL_COUPLER = (115, 230, 130)
COL_PHI = (255, 145, 45)


def _material(rgb: tuple[int, int, int]) -> sim_utils.PreviewSurfaceCfg:
    return sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(channel / 255.0 for channel in rgb), roughness=0.35)


def _body_id(robot: Articulation, name: str) -> int:
    ids, _ = robot.find_bodies(name)
    if len(ids) != 1:
        raise RuntimeError(f"expected one body {name!r}, got {ids}")
    return int(ids[0])


def _segment_poses(starts: torch.Tensor, ends: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return midpoint, Z-axis quaternion and scale for unit-height cylinders."""
    delta = ends - starts
    lengths = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1.0e-8)
    z_axis = delta / lengths[:, None]
    reference = torch.tensor([0.0, 1.0, 0.0], device=starts.device).expand_as(z_axis).clone()
    parallel = torch.abs(torch.sum(reference * z_axis, dim=-1)) > 0.95
    reference[parallel] = torch.tensor([1.0, 0.0, 0.0], device=starts.device)
    x_axis = torch.linalg.cross(reference, z_axis, dim=-1)
    x_axis = x_axis / torch.linalg.vector_norm(x_axis, dim=-1, keepdim=True).clamp_min(1.0e-8)
    y_axis = torch.linalg.cross(z_axis, x_axis, dim=-1)
    rotation = torch.stack((x_axis, y_axis, z_axis), dim=-1)
    orientation = quat_from_matrix(rotation)
    scales = torch.ones(len(starts), 3, dtype=starts.dtype, device=starts.device)
    scales[:, 2] = lengths
    return 0.5 * (starts + ends), orientation, scales


def _put_text(image: np.ndarray, text: str, xy: tuple[int, int], scale: float = 0.65,
              color: tuple[int, int, int] = (235, 235, 235), thickness: int = 1) -> None:
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_kite(
    panel: np.ndarray,
    center: tuple[int, int],
    q_front: float,
    q_rear: float,
    *,
    side: str,
    l0: float,
    phi0: float,
    physical_l0: float,
    physical_phi0: float,
    passive_joint: float,
) -> None:
    """Draw one measured joint state in the solver's common planar frame."""
    if side == "LEFT":
        angle_l20 = FDU_DEFAULT_OFFSET + q_rear
        angle_lf0 = q_front
        labels = ("l20", "lf0")
    else:
        angle_l20 = FDU_DEFAULT_OFFSET - q_rear
        angle_lf0 = -q_front
        labels = ("r20", "rf0")
    dtype = torch.float64
    a0 = torch.tensor([angle_l20], dtype=dtype)
    b0 = torch.tensor([angle_lf0], dtype=dtype)
    point, _, _, _, _ = solve_equivalent_kite(a0, b0)
    p = point[0].numpy()
    a = np.array([FDU_L1 * math.cos(angle_l20), FDU_L1 * math.sin(angle_l20)])
    b = np.array([FDU_L1 * math.cos(angle_lf0), FDU_L1 * math.sin(angle_lf0)])
    origin = np.asarray(center, dtype=np.float64)
    pixels_per_m = 400.0

    def px(xy: np.ndarray) -> tuple[int, int]:
        # solver +y is the physical downward direction; screen +y is down too
        out = origin + pixels_per_m * xy
        return int(round(out[0])), int(round(out[1]))

    o, pa, pb, pp = px(np.zeros(2)), px(a), px(b), px(p)
    # reference vertical and linkage
    cv2.line(panel, o, (o[0], o[1] + 130), (85, 85, 85), 1, cv2.LINE_AA)
    cv2.line(panel, o, pa, COL_REAR[::-1], 8, cv2.LINE_AA)
    cv2.line(panel, o, pb, COL_FRONT[::-1], 8, cv2.LINE_AA)
    cv2.line(panel, pa, pp, COL_COUPLER[::-1], 5, cv2.LINE_AA)
    cv2.line(panel, pb, pp, COL_COUPLER[::-1], 5, cv2.LINE_AA)
    cv2.line(panel, o, pp, COL_L0[::-1], 4, cv2.LINE_AA)
    for point_px in (o, pa, pb, pp):
        cv2.circle(panel, point_px, 6, (245, 245, 245), -1, cv2.LINE_AA)

    # phi0 is zero on the downward vertical; positive follows solver theta0.
    phi_deg = math.degrees(phi0)
    absolute_deg = 90.0 + phi_deg
    start_deg, end_deg = sorted((90.0, absolute_deg))
    cv2.ellipse(panel, o, (44, 44), 0.0, start_deg, end_deg, COL_PHI[::-1], 3, cv2.LINE_AA)
    _put_text(panel, side, (18, center[1] - 85), 0.72, (255, 255, 255), 2)
    _put_text(panel, f"L0 ana/phys = {l0:.4f}/{physical_l0:.4f} m",
              (center[0] - 122, center[1] + 137), 0.50, COL_L0[::-1], 2)
    _put_text(panel, f"phi ana/phys = {phi0:+.3f}/{physical_phi0:+.3f} rad",
              (center[0] - 122, center[1] + 158), 0.45, COL_PHI[::-1], 1)
    _put_text(
        panel,
        f"drive=[{q_front:+.3f},{q_rear:+.3f}] passive={passive_joint:+.3f} rad",
        (center[0] - 125, center[1] + 179), 0.38, (185, 190, 200), 1,
    )
    _put_text(panel, labels[0], (pa[0] + 5, pa[1]), 0.45, COL_REAR[::-1], 1)
    _put_text(panel, labels[1], (pb[0] + 5, pb[1]), 0.45, COL_FRONT[::-1], 1)


def _compose_frame(rgb: np.ndarray, values: dict[str, float], frame_index: int, fps: int) -> np.ndarray:
    camera_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # The source meshes and the RTX clear background are both nearly white.
    # Local luminance equalization makes STL edges readable without changing
    # the analytical panel or the semantic overlay colors.
    lab = cv2.cvtColor(camera_bgr, cv2.COLOR_BGR2LAB)
    luminance, channel_a, channel_b = cv2.split(lab)
    luminance = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(luminance)
    camera_bgr = cv2.cvtColor(cv2.merge((luminance, channel_a, channel_b)), cv2.COLOR_LAB2BGR)
    panel_width = 480
    panel = np.full((camera_bgr.shape[0], panel_width, 3), (22, 24, 30), dtype=np.uint8)
    _put_text(panel, "MEASURED JOINTS -> EQUIVALENT KITE", (18, 34), 0.62, (245, 245, 245), 2)
    _put_text(panel, "phi0: vertical-down = 0", (18, 58), 0.50, COL_PHI[::-1], 1)
    _put_text(panel, f"TEST: {values['stage']}", (18, 82), 0.52, (180, 235, 255), 2)
    _put_text(panel, f"TARGET both legs: L0={values['target_l0']:.3f} m  phi0={values['target_phi0']:+.3f} rad",
              (18, 105), 0.43, (205, 210, 220), 1)
    _draw_kite(panel, (panel_width // 2, 215), values["lf0"], values["l20"], side="LEFT",
               l0=values["left_l0"], phi0=values["left_phi0"],
               physical_l0=values["left_physical_l0"], physical_phi0=values["left_physical_phi0"],
               passive_joint=values["lf1"])
    _draw_kite(panel, (panel_width // 2, 535), values["rf0"], values["r20"], side="RIGHT",
               l0=values["right_l0"], phi0=values["right_phi0"],
               physical_l0=values["right_physical_l0"], physical_phi0=values["right_physical_phi0"],
               passive_joint=values["rf1"])

    # A compact legend over the real render.
    cv2.rectangle(camera_bgr, (12, 12), (490, 116), (12, 14, 18), -1)
    _put_text(camera_bgr, "FDU closed-chain physical motion", (24, 40), 0.72, (250, 250, 250), 2)
    _put_text(camera_bgr, "MAGENTA: lf0 / rf0 drive bars", (24, 66), 0.56, COL_FRONT[::-1], 2)
    _put_text(camera_bgr, "CYAN: l20 / r20 drive bars", (24, 89), 0.56, COL_REAR[::-1], 2)
    _put_text(camera_bgr, "YELLOW: measured hip-to-wheel L0", (24, 112), 0.56, COL_L0[::-1], 2)
    _put_text(camera_bgr, f"t = {frame_index / fps:5.2f} s", (camera_bgr.shape[1] - 145, 32), 0.58)
    return np.concatenate((camera_bgr, panel), axis=1)


def _test_trajectory(time_s: float, duration_s: float) -> tuple[float, float, str]:
    """Symmetric endpoint holds followed by angle and combined checks."""
    fraction = min(max(time_s / duration_s, 0.0), 1.0 - 1.0e-9)
    def smoothstep(value: float) -> float:
        value = min(max(value, 0.0), 1.0)
        return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)

    if fraction < 0.12:
        alpha = smoothstep(fraction / 0.12)
        return 0.285 + (0.20 - 0.285) * alpha, 0.0, "LENGTH: MOVE TO 0.20 m"
    if fraction < 0.22:
        return 0.20, 0.0, "LENGTH: HOLD 0.20 m"
    if fraction < 0.46:
        alpha = smoothstep((fraction - 0.22) / 0.24)
        return 0.20 + (0.335 - 0.20) * alpha, 0.0, "LENGTH: MOVE TO 0.335 m"
    if fraction < 0.58:
        return 0.335, 0.0, "LENGTH: HOLD 0.335 m (LIMIT-SAFE)"
    if fraction < 0.68:
        alpha = smoothstep((fraction - 0.58) / 0.10)
        return 0.335 + (0.285 - 0.335) * alpha, 0.0, "LENGTH: RETURN NEUTRAL"
    if fraction < 0.84:
        local = (fraction - 0.68) / 0.16
        phi0 = 0.32 * math.sin(2.0 * math.pi * local)
        return 0.285, phi0, "SWING SWEEP (same L/R)"
    local = (fraction - 0.84) / 0.16
    l0 = 0.270 + 0.060 * math.sin(2.0 * math.pi * local)
    phi0 = 0.24 * math.sin(2.0 * math.pi * local)
    return l0, phi0, "COMBINED (same L/R)"


def main() -> None:
    if args_cli.seconds <= 0.0 or args_cli.fps <= 0 or args_cli.physics_hz <= 0.0:
        raise ValueError("--seconds, --fps and --physics-hz must be positive")
    if args_cli.width < 640 or args_cli.height < 720:
        raise ValueError("use --width >= 640 and --height >= 720 so both geometry panels remain readable")
    # The loop closures are maximal-coordinate spherical constraints outside
    # the reduced-coordinate articulation.  At 200 Hz they enter a persistent
    # millimetre-scale limit cycle even under a static target.  The measured
    # 0.20--0.335 m trajectory is stable at 400 Hz; video FPS stays unchanged.
    sim_dt = 1.0 / args_cli.physics_hz
    sim = SimulationContext(
        sim_utils.SimulationCfg(dt=sim_dt, render_interval=1, device=args_cli.device, gravity=(0.0, 0.0, 0.0))
    )
    print("[FDU-VIS] simulation context created", flush=True)
    light_cfg = sim_utils.DomeLightCfg(intensity=700.0, color=(0.85, 0.88, 0.95))
    light_cfg.func("/World/Light", light_cfg)
    print("[FDU-VIS] light created", flush=True)

    cfg = Wheelbipe_FDU_CFG.replace(prim_path="/World/Robot")
    cfg.init_state.pos = (0.0, 0.0, 0.0)
    cfg.spawn = cfg.spawn.replace(
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True,
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=6,
        )
    )
    robot = Articulation(cfg)
    print("[FDU-VIS] robot configured", flush=True)

    marker_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/FduGeometry",
        markers={
            "front_drive": sim_utils.CylinderCfg(radius=0.011, height=1.0, visual_material=_material(COL_FRONT)),
            "rear_drive": sim_utils.CylinderCfg(radius=0.011, height=1.0, visual_material=_material(COL_REAR)),
            "virtual_l0": sim_utils.CylinderCfg(radius=0.008, height=1.0, visual_material=_material(COL_L0)),
        },
    )
    markers = VisualizationMarkers(marker_cfg)
    print("[FDU-VIS] markers configured", flush=True)

    camera_cfg = CameraCfg(
        prim_path="/World/Camera",
        update_period=0.0,
        data_types=["rgb"],
        width=args_cli.width,
        height=args_cli.height,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=34.0,
            focus_distance=2.0,
            horizontal_aperture=32.0,
            clipping_range=(0.05, 20.0),
        ),
    )
    camera = Camera(camera_cfg)
    print("[FDU-VIS] camera configured", flush=True)
    sim.reset()
    print("[FDU-VIS] simulation reset", flush=True)
    eye = torch.tensor([[0.72, 0.88, 0.30]], dtype=torch.float, device=camera.device)
    target = torch.tensor([[0.0, 0.0, -0.08]], dtype=torch.float, device=camera.device)
    camera.set_world_poses_from_view(eye, target)
    # RTX/MDL textures need several rendered frames before the first capture.
    for _ in range(12):
        robot.write_data_to_sim()
        sim.step(render=True)
        robot.update(sim_dt)
    camera.update(sim_dt, force_recompute=True)

    drive_names = ("lf0_Joint", "l20_Joint", "rf0_Joint", "r20_Joint")
    drive_ids = [robot.joint_names.index(name) for name in drive_names]
    passive_ids = [robot.joint_names.index(name) for name in ("lf1_Joint", "rf1_Joint")]
    body_names = (
        "lf0_Link", "lf1_Link", "l20_Link", "l21_Link",
        "rf0_Link", "rf1_Link", "r20_Link", "r21_Link",
        "l_wheel_Link", "r_wheel_Link",
    )
    body_ids = {name: _body_id(robot, name) for name in body_names}

    output = Path(args_cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    video_size = (args_cli.width + 480, args_cli.height)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), args_cli.fps, video_size)
    if not writer.isOpened():
        raise RuntimeError(f"could not open MP4 writer for {output}")

    frame_count = round(args_cli.seconds * args_cli.fps)
    physics_step = 0
    measured_l0_min = [float("inf"), float("inf")]
    measured_l0_max = [float("-inf"), float("-inf")]
    max_l0_symmetry_error = 0.0
    max_phi0_symmetry_error = 0.0
    try:
        for frame_index in range(frame_count):
            # Continuous command updates at the selected physics rate. Capture
            # at the video FPS with a fractional cadence instead of holding one target for a
            # whole video frame (which caused visible closed-loop twitching).
            target_physics_step = math.ceil((frame_index + 1) / (args_cli.fps * sim_dt))
            while physics_step < target_physics_step:
                time_s = physics_step * sim_dt
                target_l0, target_phi0, stage = _test_trajectory(time_s, args_cli.seconds)
                lengths = torch.full((2,), target_l0, device=args_cli.device)
                phis = torch.full((2,), target_phi0, device=args_cli.device)
                common_front, common_rear, valid = inverse_equivalent_kite(lengths, phis)
                if not bool(torch.all(valid)):
                    raise RuntimeError("visualization trajectory left the analytic workspace")
                # Both sides receive identical physical L0/phi0.  Entity
                # signs differ only because the right URDF joint axes mirror
                # the left axes.
                target_q = torch.stack(
                    (common_front[0], common_rear[0], -common_front[1], -common_rear[1])
                ).reshape(1, 4)
                robot.set_joint_position_target(target_q, joint_ids=drive_ids)
                robot.write_data_to_sim()
                sim.step(render=True)
                robot.update(sim_dt)
                physics_step += 1

            time_s = physics_step * sim_dt
            target_l0, target_phi0, stage = _test_trajectory(time_s, args_cli.seconds)

            positions = robot.data.body_pos_w[0]
            starts = torch.stack(
                (
                    positions[body_ids["lf0_Link"]], positions[body_ids["l20_Link"]],
                    positions[body_ids["rf0_Link"]], positions[body_ids["r20_Link"]],
                    positions[body_ids["lf0_Link"]], positions[body_ids["rf0_Link"]],
                )
            )
            ends = torch.stack(
                (
                    positions[body_ids["lf1_Link"]], positions[body_ids["l21_Link"]],
                    positions[body_ids["rf1_Link"]], positions[body_ids["r21_Link"]],
                    positions[body_ids["l_wheel_Link"]], positions[body_ids["r_wheel_Link"]],
                )
            )
            translations, orientations, scales = _segment_poses(starts, ends)
            markers.visualize(
                translations=translations,
                orientations=orientations,
                scales=scales,
                marker_indices=[0, 1, 0, 1, 2, 2],
            )
            # Render once after updating visualization markers.
            sim.render()
            camera.update(sim_dt, force_recompute=True)
            rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
            actual_q = robot.data.joint_pos[0, drive_ids]
            passive_q = robot.data.joint_pos[0, passive_ids]
            ll, lp, rl, rp = compute_fdu_equivalent_leg_state(
                actual_q[0:1], actual_q[1:2], actual_q[2:3], actual_q[3:4]
            )
            physical_delta = ends[-2:] - starts[-2:]
            physical_l0 = torch.linalg.vector_norm(physical_delta[:, (0, 2)], dim=-1)
            physical_phi0 = torch.atan2(-physical_delta[:, 0], -physical_delta[:, 2])
            for side_idx in range(2):
                measured_l0_min[side_idx] = min(measured_l0_min[side_idx], float(physical_l0[side_idx]))
                measured_l0_max[side_idx] = max(measured_l0_max[side_idx], float(physical_l0[side_idx]))
            max_l0_symmetry_error = max(
                max_l0_symmetry_error, float(torch.abs(physical_l0[0] - physical_l0[1]))
            )
            phi_symmetry_error = torch.atan2(
                torch.sin(physical_phi0[0] - physical_phi0[1]),
                torch.cos(physical_phi0[0] - physical_phi0[1]),
            )
            max_phi0_symmetry_error = max(max_phi0_symmetry_error, float(torch.abs(phi_symmetry_error)))
            values = {
                "stage": stage, "target_l0": target_l0, "target_phi0": target_phi0,
                "lf0": float(actual_q[0]), "l20": float(actual_q[1]),
                "rf0": float(actual_q[2]), "r20": float(actual_q[3]),
                "lf1": float(passive_q[0]), "rf1": float(passive_q[1]),
                "left_l0": float(ll[0]), "left_phi0": float(lp[0]),
                "right_l0": float(rl[0]), "right_phi0": float(rp[0]),
                "left_physical_l0": float(physical_l0[0]), "left_physical_phi0": float(physical_phi0[0]),
                "right_physical_l0": float(physical_l0[1]), "right_physical_phi0": float(physical_phi0[1]),
            }
            writer.write(_compose_frame(rgb, values, frame_index, args_cli.fps))
    finally:
        writer.release()

    print(f"FDU L0/PHI0 VIDEO WRITTEN: {output}")
    print(
        f"frames={frame_count} fps={args_cli.fps} physics_hz={args_cli.physics_hz:g} "
        f"size={video_size[0]}x{video_size[1]}"
    )
    print(
        "measured_physical_l0_range="
        f"left[{measured_l0_min[0]:.4f}, {measured_l0_max[0]:.4f}] m "
        f"right[{measured_l0_min[1]:.4f}, {measured_l0_max[1]:.4f}] m"
    )
    print(
        f"mechanical_l0_max={FDU_MECHANICAL_L0_MAX:.6f} m "
        f"from lf1 limit {FDU_LEFT_PASSIVE_LIMITS[0]:.3f} rad"
    )
    print(
        f"max_left_right_physical_error: L0={max_l0_symmetry_error:.6f} m "
        f"phi0={max_phi0_symmetry_error:.6f} rad"
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        # The MP4 is already finalized by main().  Isaac Sim 5.1 can block
        # indefinitely while waiting for Replicator/RTX cleanup, especially
        # when another Kit process owns the shared cache.  Do not turn a
        # completed recording into a hung command.
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
