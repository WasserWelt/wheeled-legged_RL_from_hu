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

"""wyw 任务族注册（fudan 设计 + from_hu 本体）。

三个任务共享 Actor/Critic/Obs/网络/PPO 超参（``WheelbipeWywPPORunnerCfg`` →
``OnPolicySequenceRunner`` / ``ActorCriticSequence`` / ``PPOSequence``）：
- Robotics-Wheelbipe-V14-wyw-Flat-v1   平地
- Robotics-Wheelbipe-V14-wyw-Rough-v1  trimesh + 课程
- Robotics-Wheelbipe-V14-wyw-Jump-v1   平地 + 涌现式跳跃
（各带 -Play-v1 变体）
"""

import gymnasium as gym

from agent_tasks.direct.wheelbipe import agents

_RUNNER = f"{agents.__name__}.rsl_rl_ppo_cfg:WheelbipeWywPPORunnerCfg"


gym.register(
    id="Robotics-Wheelbipe-V14-wyw-Flat-v1",
    entry_point=f"{__name__}.env:WheelbipeWywEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelbipeWywFlatEnvCfg",
        "rsl_rl_cfg_entry_point": _RUNNER,
    },
)

gym.register(
    id="Robotics-Wheelbipe-V14-wyw-Flat-Play-v1",
    entry_point=f"{__name__}.env:WheelbipeWywEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelbipeWywFlatEnvCfg_Play",
        "rsl_rl_cfg_entry_point": _RUNNER,
    },
)

gym.register(
    id="Robotics-Wheelbipe-V14-wyw-Rough-v1",
    entry_point=f"{__name__}.env:WheelbipeWywEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelbipeWywRoughEnvCfg",
        "rsl_rl_cfg_entry_point": _RUNNER,
    },
)

gym.register(
    id="Robotics-Wheelbipe-V14-wyw-Rough-Play-v1",
    entry_point=f"{__name__}.env:WheelbipeWywEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelbipeWywRoughEnvCfg_Play",
        "rsl_rl_cfg_entry_point": _RUNNER,
    },
)

gym.register(
    id="Robotics-Wheelbipe-V14-wyw-Jump-v1",
    entry_point=f"{__name__}.env:WheelbipeWywEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelbipeWywJumpEnvCfg",
        "rsl_rl_cfg_entry_point": _RUNNER,
    },
)

gym.register(
    id="Robotics-Wheelbipe-V14-wyw-Jump-Play-v1",
    entry_point=f"{__name__}.env:WheelbipeWywEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:WheelbipeWywJumpEnvCfg_Play",
        "rsl_rl_cfg_entry_point": _RUNNER,
    },
)
