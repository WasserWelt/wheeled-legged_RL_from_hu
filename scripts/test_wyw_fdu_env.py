"""CPU smoke test for the WYW task wired to the FDU closed-chain asset."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.device = "cpu"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

from agent_tasks.direct.wheelbipe.wyw.env import WheelbipeWywEnv  # noqa: E402
from agent_tasks.direct.wheelbipe.wyw.env_cfg import WheelbipeWywFlatEnvCfg  # noqa: E402


def main():
    cfg = WheelbipeWywFlatEnvCfg()
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 2.0
    cfg.sim.device = "cpu"
    cfg.play = False
    cfg.commands.debug_vis = False
    env = WheelbipeWywEnv(cfg)
    try:
        obs, _ = env.reset()
        assert obs["policy"].shape == (1, 25)
        assert obs["policy_hist"].shape == (1, 125)
        assert obs["critic"].shape == (1, 141)
        for _ in range(2):
            obs, reward, terminated, truncated, _ = env.step(torch.zeros(1, 6, device=env.device))
            assert torch.isfinite(obs["policy"]).all()
            assert torch.isfinite(obs["critic"]).all()
            assert torch.isfinite(reward).all()
        print("WYW FDU ENV SMOKE TEST PASSED")
        print(f"policy_order={env._wyw_policy_joint_idx}")
        print(f"leg_entity_indices={env._wyw_leg_joint_idx}")
        print(f"wheel_entity_indices={env._wyw_wheel_joint_idx}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
