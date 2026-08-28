"""Smoke-test Wheelbipe_FDU_CFG: spawn it and confirm every actuator group
resolves (Isaac Lab raises if a regex matches no joint or a DOF is unclaimed).
Run from repo root:  python scripts/test_fdu_cfg.py --headless
"""

import argparse
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

    lines = ["=" * 60, f"num_joints={robot.num_joints}  num_bodies={robot.num_bodies}"]
    for name, act in robot.actuators.items():
        js = [robot.joint_names[i] for i in act.joint_indices]
        lines.append(f"actuator '{name}': {js}")
    lines.append("ALL ACTUATOR GROUPS RESOLVED OK")
    lines.append("=" * 60)
    for ln in lines:
        print(ln, flush=True)
    with open("/tmp/fdu_cfg_test.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")

    for _ in range(60):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())


if __name__ == "__main__":
    main()
    simulation_app.close()
