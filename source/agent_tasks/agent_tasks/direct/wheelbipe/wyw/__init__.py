# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Authors:
#     Wasser_welt
# =============================================================================

"""wyw 任务族注册（Fudan 语义 + FDU infantry_V2 闭链本体）。

三个任务共享 Actor/Critic/Obs/网络/PPO 超参（``WheelbipeWywPPORunnerCfg`` →
``OnPolicySequenceRunner`` / ``ActorCriticSequence`` / ``PPOSequence``）：
- Robotics-Wheelbipe-FDU-wyw-Flat-v1   平地
- Robotics-Wheelbipe-FDU-wyw-Rough-v1  trimesh + 课程
- Robotics-Wheelbipe-FDU-wyw-Jump-v1   平地 + 涌现式跳跃
（各带 -Play-v1 变体）
"""

import gymnasium as gym

from agent_tasks.direct.wheelbipe import agents

# 三任务共享超参，仅日志目录（experiment_name）分开，便于区分 checkpoint/曲线
_RUNNER_FLAT = f"{agents.__name__}.rsl_rl_ppo_cfg:WheelbipeWywPPORunnerCfg"
_RUNNER_ROUGH = f"{agents.__name__}.rsl_rl_ppo_cfg:WheelbipeWywRoughPPORunnerCfg"
_RUNNER_JUMP = f"{agents.__name__}.rsl_rl_ppo_cfg:WheelbipeWywJumpPPORunnerCfg"


gym.register(
    id="Robotics-Wheelbipe-FDU-wyw-Flat-v1",
    entry_point=f"{__name__}.env:WheelbipeWywEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelbipeWywFlatEnvCfg",
        "rsl_rl_cfg_entry_point": _RUNNER_FLAT,
    },
)

gym.register(
    id="Robotics-Wheelbipe-FDU-wyw-Flat-Play-v1",
    entry_point=f"{__name__}.env:WheelbipeWywEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelbipeWywFlatEnvCfg_Play",
        "rsl_rl_cfg_entry_point": _RUNNER_FLAT,
    },
)

gym.register(
    id="Robotics-Wheelbipe-FDU-wyw-Rough-v1",
    entry_point=f"{__name__}.env:WheelbipeWywEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelbipeWywRoughEnvCfg",
        "rsl_rl_cfg_entry_point": _RUNNER_ROUGH,
    },
)

gym.register(
    id="Robotics-Wheelbipe-FDU-wyw-Rough-Play-v1",
    entry_point=f"{__name__}.env:WheelbipeWywEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelbipeWywRoughEnvCfg_Play",
        "rsl_rl_cfg_entry_point": _RUNNER_ROUGH,
    },
)

gym.register(
    id="Robotics-Wheelbipe-FDU-wyw-Jump-v1",
    entry_point=f"{__name__}.env:WheelbipeWywEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelbipeWywJumpEnvCfg",
        "rsl_rl_cfg_entry_point": _RUNNER_JUMP,
    },
)

gym.register(
    id="Robotics-Wheelbipe-FDU-wyw-Jump-Play-v1",
    entry_point=f"{__name__}.env:WheelbipeWywEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelbipeWywJumpEnvCfg_Play",
        "rsl_rl_cfg_entry_point": _RUNNER_JUMP,
    },
)
