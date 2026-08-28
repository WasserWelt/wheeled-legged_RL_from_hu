"""CPU smoke test for the WYW task wired to the FDU closed-chain asset."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--variant", choices=("flat", "rough", "jump"), default="flat")
parser.add_argument("--play", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.device = "cpu"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

from agent_tasks.direct.wheelbipe.wyw.env import WheelbipeWywEnv  # noqa: E402
from agent_tasks.direct.wheelbipe.wyw.env_cfg import (  # noqa: E402
    FDU_JUMP_REWARDS,
    FDU_PLANE_REWARDS,
    WheelbipeWywFlatEnvCfg,
    WheelbipeWywRoughEnvCfg,
    WheelbipeWywJumpEnvCfg,
    WheelbipeWywFlatEnvCfg_Play,
    WheelbipeWywRoughEnvCfg_Play,
    WheelbipeWywJumpEnvCfg_Play,
)


def main():
    cfg_type = ({
        "flat": WheelbipeWywFlatEnvCfg_Play,
        "rough": WheelbipeWywRoughEnvCfg_Play,
        "jump": WheelbipeWywJumpEnvCfg_Play,
    } if args_cli.play else {
        "flat": WheelbipeWywFlatEnvCfg,
        "rough": WheelbipeWywRoughEnvCfg,
        "jump": WheelbipeWywJumpEnvCfg,
    })[args_cli.variant]
    cfg = cfg_type()
    cfg.scene.num_envs = 1
    cfg.scene.env_spacing = 2.0
    cfg.sim.device = "cpu"
    cfg.play = False
    cfg.commands.debug_vis = False
    env = WheelbipeWywEnv(cfg)
    try:
        assert env.cfg.max_wheel_vel == 60.0
        assert env.max_wheel_vel == 60.0
        assert env.robot.cfg.actuators["wheel"].velocity_limit == 60.0
        assert env.robot.cfg.actuators["wheel"].velocity_limit_sim == 60.0
        obs, _ = env.reset()
        assert obs["policy"].shape == (1, 25)
        assert obs["policy_hist"].shape == (1, 125)
        assert obs["critic"].shape == (1, 141)
        for _ in range(2):
            obs, reward, terminated, truncated, _ = env.step(torch.zeros(1, 6, device=env.device))
            assert torch.isfinite(obs["policy"]).all()
            assert torch.isfinite(obs["critic"]).all()
            assert torch.isfinite(reward).all()
        expected_rewards = FDU_JUMP_REWARDS if args_cli.variant == "jump" else FDU_PLANE_REWARDS
        assert list(env.cfg.rewards) == list(expected_rewards)
        assert set(env._last_reward_terms) == set(expected_rewards)
        print("WYW FDU ENV SMOKE TEST PASSED")
        print(f"variant={args_cli.variant} play={args_cli.play} reward_terms={list(env.cfg.rewards)}")
        print(f"policy_order={env._wyw_policy_joint_idx}")
        print(f"leg_entity_indices={env._wyw_leg_joint_idx}")
        print(f"wheel_entity_indices={env._wyw_wheel_joint_idx}")
        print(
            f"wheel_velocity_limit={env.robot.cfg.actuators['wheel'].velocity_limit} rad/s "
            f"runtime_clamp={env.max_wheel_vel} rad/s"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
