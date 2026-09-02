"""Drop the closed-chain FDU robot onto a plane and measure impact stability."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--dt", type=float, default=0.002)
parser.add_argument("--seconds", type=float, default=3.0)
parser.add_argument("--root-height", type=float, default=0.55)
parser.add_argument("--l0", type=float, default=0.16)
parser.add_argument("--position-iterations", type=int, default=16)
parser.add_argument("--velocity-iterations", type=int, default=6)
parser.add_argument("--drive-stiffness", type=float, default=None, help="optional legs_act Kp override")
parser.add_argument("--drive-damping", type=float, default=None, help="optional legs_act Kd override")
parser.add_argument("--drive-effort-limit", type=float, default=None, help="optional legs_act effort override")
parser.add_argument(
    "--guided-vertical",
    action="store_true",
    help="emulate a vertical guide rail by suppressing root x/y and rotation after each physics step",
)
parser.add_argument("--output", default="docs/fdu_validation/drop/fdu_drop.json")
parser.add_argument(
    "--video-output",
    default="",
    help="optional MP4 path; leave empty to run the numeric test without RTX rendering",
)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=720)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = bool(args_cli.video_output)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import carb  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402
from pxr import UsdPhysics  # noqa: E402

from agent_tasks.direct.wheelbipe.wyw.fdu_mapping import inverse_equivalent_kite  # noqa: E402
from agent_world.assets.wheelbipe_fdu import Wheelbipe_FDU_CFG  # noqa: E402


def _body_id(robot: Articulation, name: str) -> int:
    ids, _ = robot.find_bodies(name)
    if len(ids) != 1:
        raise RuntimeError(f"expected one body {name!r}, got {ids}")
    return int(ids[0])


def _loop_anchors(sim: SimulationContext, robot: Articulation):
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
    if len(anchors) != 4:
        raise RuntimeError(f"expected four loop constraints, found {len(anchors)}")
    return anchors


def _loop_gaps(robot: Articulation, anchors) -> torch.Tensor:
    gaps = []
    for _, body0, local0, body1, local1 in anchors:
        p0 = robot.data.body_pos_w[0, body0] + quat_apply(robot.data.body_quat_w[0, body0], local0)
        p1 = robot.data.body_pos_w[0, body1] + quat_apply(robot.data.body_quat_w[0, body1], local1)
        gaps.append(torch.linalg.vector_norm(p0 - p1))
    return torch.stack(gaps)


def _put_text(
    image: np.ndarray,
    text: str,
    xy: tuple[int, int],
    *,
    scale: float = 0.58,
    color: tuple[int, int, int] = (238, 238, 238),
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _video_frame(rgb: np.ndarray, values: dict[str, float | list[float] | bool]) -> np.ndarray:
    """Add enough live physics data to make the drop recording auditable."""
    frame = cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2BGR)
    panel_width = min(545, frame.shape[1] - 24)
    cv2.rectangle(frame, (12, 12), (12 + panel_width, 230), (15, 17, 22), -1)
    mode = "GUIDED VERTICAL" if values["guided"] else "FREE BODY"
    lines = (
        ("FDU CLOSED-CHAIN DROP TEST", (245, 245, 245), 0.68, 2),
        (f"mode={mode}  physics={values['physics_hz']:.0f} Hz  t={values['time_s']:.3f} s", (205, 215, 225), 0.50, 1),
        (f"target L0={values['target_l0']:.3f} m", (110, 220, 255), 0.56, 2),
        (f"measured L/R={values['left_l0']:.3f} / {values['right_l0']:.3f} m", (110, 220, 255), 0.54, 1),
        (f"root z={values['root_z']:.3f} m   vz={values['root_vz']:+.3f} m/s", (225, 225, 225), 0.51, 1),
        (f"contact={values['contact_n']:.0f} N   loop gap={values['loop_gap_mm']:.3f} mm", (160, 240, 180), 0.51, 1),
        (f"drive torque={values['torque_nm']:.1f} / {values['effort_limit_nm']:.1f} N m", (120, 180, 255), 0.53, 2),
        (f"PD: Kp={values['stiffness']:.1f}, Kd={values['damping']:.2f}", (205, 205, 215), 0.50, 1),
    )
    for index, (line, color, scale, thickness) in enumerate(lines):
        _put_text(frame, line, (24, 40 + 25 * index), scale=scale, color=color, thickness=thickness)
    return frame


def main() -> None:
    if min(args_cli.dt, args_cli.seconds, args_cli.root_height, args_cli.l0) <= 0.0:
        raise ValueError("dt, seconds, root-height and l0 must be positive")
    if args_cli.video_output and (args_cli.fps <= 0 or args_cli.width < 640 or args_cli.height < 480):
        raise ValueError("video requires --fps > 0, --width >= 640 and --height >= 480")
    sim = SimulationContext(
        sim_utils.SimulationCfg(
            dt=args_cli.dt,
            render_interval=1,
            device=args_cli.device,
            gravity=(0.0, 0.0, 0.0),
        )
    )
    ground_cfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8, dynamic_friction=0.8, restitution=0.0
        )
    )
    ground_cfg.func("/World/Ground", ground_cfg)
    camera = None
    if args_cli.video_output:
        light_cfg = sim_utils.DomeLightCfg(intensity=850.0, color=(0.86, 0.89, 0.96))
        light_cfg.func("/World/Light", light_cfg)
        camera = Camera(
            CameraCfg(
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
        )

    cfg = Wheelbipe_FDU_CFG.replace(prim_path="/World/Robot")
    if args_cli.drive_stiffness is not None:
        cfg.actuators["legs_act"].stiffness = args_cli.drive_stiffness
    if args_cli.drive_damping is not None:
        cfg.actuators["legs_act"].damping = args_cli.drive_damping
    if args_cli.drive_effort_limit is not None:
        cfg.actuators["legs_act"].effort_limit = args_cli.drive_effort_limit
    cfg.init_state.pos = (0.0, 0.0, args_cli.root_height)
    cfg.spawn = cfg.spawn.replace(
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=False,
            enabled_self_collisions=False,
            solver_position_iteration_count=args_cli.position_iterations,
            solver_velocity_iteration_count=args_cli.velocity_iterations,
        )
    )
    robot = Articulation(cfg)
    contact = ContactSensor(ContactSensorCfg(prim_path="/World/Robot/.*", update_period=0.0))
    sim.reset()
    if camera is not None:
        eye = torch.tensor([[1.05, 0.95, 0.52]], dtype=torch.float, device=camera.device)
        look_at = torch.tensor([[0.0, 0.0, 0.20]], dtype=torch.float, device=camera.device)
        camera.set_world_poses_from_view(eye, look_at)
        # Warm RTX/MDL before the recorded interval. Gravity is still off.
        for _ in range(12):
            robot.write_data_to_sim()
            sim.step(render=True)
            robot.update(args_cli.dt)
        camera.update(args_cli.dt, force_recompute=True)

    drive_names = ("lf0_Joint", "l20_Joint", "rf0_Joint", "r20_Joint")
    drive_ids = [robot.joint_names.index(name) for name in drive_names]
    passive_ids = [i for i in range(robot.num_joints) if i not in drive_ids]
    hip_ids = [_body_id(robot, "lf0_Link"), _body_id(robot, "rf0_Link")]
    wheel_ids = [_body_id(robot, "l_wheel_Link"), _body_id(robot, "r_wheel_Link")]
    anchors = _loop_anchors(sim, robot)
    length = torch.full((2,), args_cli.l0, device=robot.device)
    front, rear, valid = inverse_equivalent_kite(length, torch.zeros_like(length))
    if not bool(torch.all(valid)):
        raise ValueError(f"L0={args_cli.l0} violates the calibrated workspace")
    target = torch.stack((front[0], rear[0], -front[1], -rear[1]))
    robot.set_joint_position_target(target[None], joint_ids=drive_ids)

    leg_actuator_cfg = cfg.actuators["legs_act"]
    effort_limit_nm = float(leg_actuator_cfg.effort_limit)
    stiffness = float(leg_actuator_cfg.stiffness)
    damping = float(leg_actuator_cfg.damping)

    # Pre-shape the closed linkage without falling.  Then reset the floating
    # base pose/velocity exactly and release gravity.  This separates impact
    # response from a simultaneous large leg-position transient.
    preshape_steps = max(1, round(1.0 / args_cli.dt))
    for _ in range(preshape_steps):
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(args_cli.dt)
    root_pose = robot.data.root_pose_w.clone()
    root_pose[:, 0:3] = torch.tensor((0.0, 0.0, args_cli.root_height), device=robot.device)
    root_velocity = torch.zeros((1, 6), device=robot.device)
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(root_velocity)
    robot.write_data_to_sim()
    sim.physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, -9.81))

    steps = round(args_cli.seconds / args_cli.dt)
    times, root_z, root_vz, force, gaps, l0s, drive_vel, passive_vel = [], [], [], [], [], [], [], []
    drive_tracking, applied_torque = [], []
    per_body_peak = torch.zeros(len(contact.body_names), device=robot.device)
    writer = None
    video_frame_index = 0
    video_frame_count = round(args_cli.seconds * args_cli.fps) if camera is not None else 0
    if camera is not None:
        video_output = Path(args_cli.video_output)
        video_output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(video_output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args_cli.fps,
            (args_cli.width, args_cli.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"could not open MP4 writer for {video_output}")
    for step in range(steps):
        robot.write_data_to_sim()
        next_capture_step = (
            math.ceil((video_frame_index + 1) / (args_cli.fps * args_cli.dt))
            if video_frame_index < video_frame_count
            else steps + 1
        )
        render_now = camera is not None and step + 1 >= next_capture_step
        sim.step(render=render_now)
        robot.update(args_cli.dt)
        contact.update(args_cli.dt, force_recompute=True)
        body_force = torch.linalg.vector_norm(contact.data.net_forces_w[0], dim=-1)
        per_body_peak = torch.maximum(per_body_peak, body_force)
        delta = robot.data.body_pos_w[0, wheel_ids] - robot.data.body_pos_w[0, hip_ids]
        times.append(step * args_cli.dt)
        root_z.append(float(robot.data.root_pos_w[0, 2]))
        root_vz.append(float(robot.data.root_lin_vel_w[0, 2]))
        force.append(float(torch.sum(body_force)))
        gaps.append(float(torch.max(_loop_gaps(robot, anchors))) * 1000.0)
        l0s.append(torch.linalg.vector_norm(delta[:, (0, 2)], dim=-1).cpu().tolist())
        drive_vel.append(float(torch.max(torch.abs(robot.data.joint_vel[0, drive_ids]))))
        passive_vel.append(float(torch.max(torch.abs(robot.data.joint_vel[0, passive_ids]))))
        drive_tracking.append(float(torch.max(torch.abs(robot.data.joint_pos[0, drive_ids] - target))))
        applied_torque.append(float(torch.max(torch.abs(robot.data.applied_torque[0, drive_ids]))))
        if render_now and camera is not None and writer is not None:
            camera.update(args_cli.dt, force_recompute=True)
            rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
            writer.write(
                _video_frame(
                    rgb,
                    {
                        "guided": args_cli.guided_vertical,
                        "physics_hz": 1.0 / args_cli.dt,
                        "time_s": (step + 1) * args_cli.dt,
                        "target_l0": args_cli.l0,
                        "left_l0": l0s[-1][0],
                        "right_l0": l0s[-1][1],
                        "root_z": root_z[-1],
                        "root_vz": root_vz[-1],
                        "contact_n": force[-1],
                        "loop_gap_mm": gaps[-1],
                        "torque_nm": applied_torque[-1],
                        "effort_limit_nm": effort_limit_nm,
                        "stiffness": stiffness,
                        "damping": damping,
                    },
                )
            )
            video_frame_index += 1
        if args_cli.guided_vertical:
            # Guide only the floating base's lateral/rotational motion.  Keep
            # z/vz and every internal joint dynamic, so wheel impact, actuator
            # loading, and loop-constraint residuals are still measured.
            guided_pose = robot.data.root_pose_w.clone()
            guided_pose[:, 0:2] = 0.0
            guided_pose[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=robot.device)
            guided_velocity = robot.data.root_vel_w.clone()
            guided_velocity[:, 0:2] = 0.0
            guided_velocity[:, 3:6] = 0.0
            robot.write_root_pose_to_sim(guided_pose)
            robot.write_root_velocity_to_sim(guided_velocity)

    if writer is not None:
        writer.release()
        print(
            f"FDU DROP VIDEO WRITTEN: {args_cli.video_output} "
            f"frames={video_frame_index} fps={args_cli.fps} size={args_cli.width}x{args_cli.height}"
        )

    contact_ids = [i for i, value in enumerate(force) if value > 5.0]
    first_contact = contact_ids[0] if contact_ids else None
    tail_start = max(0, steps - round(0.5 / args_cli.dt))
    l0_t = torch.tensor(l0s)
    report = {
        "settings": {
            "dt_s": args_cli.dt,
            "duration_s": args_cli.seconds,
            "initial_root_height_m": args_cli.root_height,
            "target_l0_m": args_cli.l0,
            "position_iterations": args_cli.position_iterations,
            "velocity_iterations": args_cli.velocity_iterations,
            "preshape_duration_s": preshape_steps * args_cli.dt,
            "self_collisions_enabled": False,
            "ground_friction": 0.8,
            "ground_restitution": 0.0,
            "guided_vertical": args_cli.guided_vertical,
            "drive_effort_limit_nm": effort_limit_nm,
            "drive_stiffness_nm_per_rad": stiffness,
            "drive_damping_nm_s_per_rad": damping,
            "video_output": args_cli.video_output or None,
        },
        "result": {
            "first_contact_time_s": times[first_contact] if first_contact is not None else None,
            "peak_total_contact_force_n": max(force),
            "peak_contact_force_by_body_n": dict(zip(contact.body_names, per_body_peak.cpu().tolist())),
            "min_root_height_m": min(root_z),
            "max_abs_root_vertical_velocity_m_s": max(abs(v) for v in root_vz),
            "max_loop_gap_mm": max(gaps),
            "post_contact_max_loop_gap_mm": max(gaps[first_contact:]) if first_contact is not None else None,
            "max_drive_joint_speed_rad_s": max(drive_vel),
            "max_passive_joint_speed_rad_s": max(passive_vel),
            "max_drive_tracking_error_rad": max(drive_tracking),
            "final_drive_tracking_error_rad": drive_tracking[-1],
            "peak_applied_drive_torque_nm": max(applied_torque),
            "tail_root_height_peak_to_peak_mm": 1000.0 * (max(root_z[tail_start:]) - min(root_z[tail_start:])),
            "tail_l0_peak_to_peak_mm": (1000.0 * (l0_t[tail_start:].max(0).values - l0_t[tail_start:].min(0).values)).tolist(),
            "final_l0_m": l0s[-1],
        },
    }
    output = Path(args_cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"FDU DROP TEST WRITTEN: {output}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
