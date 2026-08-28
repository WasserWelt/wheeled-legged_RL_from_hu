"""Calibrate and smoke-test the FDU articulation on CPU.

The report covers topology, semantic actuator mapping, limits, defaults, body
mass/COM/inertia and a short finite-state settling check.
Run from repo root:  python scripts/test_fdu_cfg.py --headless
"""

import argparse
import json
import math
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.device = "cpu"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from agent_world.assets.wheelbipe_fdu import Wheelbipe_FDU_CFG  # noqa: E402


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device="cpu"))
    cfg = Wheelbipe_FDU_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(cfg)
    sim.reset()

    expected_joints = {
        "lf0_Joint", "l20_Joint", "l_wheel_Joint", "rf0_Joint", "r20_Joint", "r_wheel_Joint",
        "lf1_Joint", "rf1_Joint", "l21_Joint", "l22_Joint", "l23_Joint",
        "r21_Joint", "r22_Joint", "r23_Joint",
    }
    assert robot.num_joints == 14, robot.joint_names
    assert robot.num_bodies == 15, robot.body_names
    assert set(robot.joint_names) == expected_joints
    assert "base_link_del" in robot.body_names

    lines = ["=" * 60, f"num_joints={robot.num_joints}  num_bodies={robot.num_bodies}"]
    actuator_groups = {}
    for name, act in robot.actuators.items():
        js = [robot.joint_names[i] for i in act.joint_indices]
        actuator_groups[name] = js
        lines.append(f"actuator '{name}': {js}")
    wheel_actuator = robot.actuators["wheel"]
    assert wheel_actuator.cfg.velocity_limit == 60.0
    assert wheel_actuator.cfg.velocity_limit_sim == 60.0
    assert wheel_actuator.cfg.effort_limit == 5.0
    for joint_name in ("l_wheel_Joint", "r_wheel_Joint"):
        joint_index = robot.joint_names.index(joint_name)
        joint_velocity_limit = float(robot.data.joint_vel_limits[0, joint_index])
        assert math.isclose(
            joint_velocity_limit,
            60.0,
            rel_tol=1.0e-6,
        ), f"{joint_name} velocity limit is {joint_velocity_limit}, expected 60.0"
    lines.append("wheel limits: velocity=60.0 rad/s, Plane training effort=5.0 N*m")
    lines.append("ALL ACTUATOR GROUPS RESOLVED OK")
    lines.append("=" * 60)
    for ln in lines:
        print(ln, flush=True)
    with open("/tmp/fdu_cfg_test.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")

    for _ in range(10):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())

    assert robot.data.joint_pos.isfinite().all()
    assert robot.data.joint_vel.isfinite().all()
    masses = robot.root_physx_view.get_masses()[0].cpu()
    if masses.ndim == 2 and masses.shape[-1] == 1:
        masses = masses[:, 0]
    inertias = robot.root_physx_view.get_inertias()[0].cpu()
    assert masses.shape == (robot.num_bodies,), masses.shape
    assert inertias.shape == (robot.num_bodies, 9), inertias.shape
    report = {
        "source_urdf": "robot_models/fdu_infantry_V4_mujoco/meshes/infantry_V2.urdf",
        "root_body": "base_link_del",
        "joint_names": list(robot.joint_names),
        "body_names": list(robot.body_names),
        "actuator_groups": actuator_groups,
        "policy_order": [
            "lf0_Joint", "l20_Joint", "l_wheel_Joint",
            "rf0_Joint", "r20_Joint", "r_wheel_Joint",
        ],
        "joint_limits": {
            name: {
                "lower": float(robot.data.joint_pos_limits[0, i, 0]),
                "upper": float(robot.data.joint_pos_limits[0, i, 1]),
                "velocity": float(robot.data.joint_vel_limits[0, i]),
                "effort": float(robot.data.joint_effort_limits[0, i]),
                "default_pos": float(robot.data.default_joint_pos[0, i]),
            }
            for i, name in enumerate(robot.joint_names)
        },
        "bodies": {
            name: {
                "mass": float(masses[i]),
                "inertia": inertias[i].tolist(),
            }
            for i, name in enumerate(robot.body_names)
        },
        "finite_after_10_steps": True,
    }
    for report_path in ("/tmp/fdu_calibration_report.json", "docs/fdu_calibration_report.json"):
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
    print("FDU CALIBRATION TEST PASSED", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
