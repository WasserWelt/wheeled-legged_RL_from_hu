# =============================================================================
# Copyright (c) 2026 SCUTRobotLab
# SPDX-License-Identifier: MIT
#
# Part of the wheeled-legged_RL project.
# See LICENSE for full license terms.
#
# Authors:
#     Zhang Zhirui <2231625449@qq.com>
#     Cui Yu       <ctty694@gmail.com>
# =============================================================================

"""PD actuator that estimates joint velocity from finite differences."""

from __future__ import annotations

import torch
from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions

from .finite_difference_velocity import FiniteDifferenceJointVelocity


class DiffVelPDActuator(IdealPDActuator):
    """Ideal PD actuator using finite-difference joint velocity feedback.

    IsaacLab passes simulator joint velocity into :meth:`compute`. This actuator
    keeps the standard ``IdealPDActuator`` position/velocity PD law, but replaces
    the velocity feedback with ``(q_t - q_{t-1}) / dt``. This is useful when the
    policy or sim2real setup should see/control the same velocity estimator used
    on hardware.
    """

    cfg: "DiffVelPDActuatorCfg"

    def __init__(self, cfg: "DiffVelPDActuatorCfg", *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._velocity_estimator = FiniteDifferenceJointVelocity(
            cfg.diff_dt,
            wrap_to_pi=cfg.wrap_to_pi,
            reset_position_jump_threshold=cfg.reset_position_jump_threshold,
            velocity_filter_alpha=cfg.velocity_filter_alpha,
        )

    def reset(self, env_ids):
        """Invalidate only the reset environments' finite-difference history."""
        self._velocity_estimator.reset(env_ids)

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        """Compute actuator efforts using finite-difference joint velocity."""
        del joint_vel
        diff_joint_vel = self._velocity_estimator.update(joint_pos)
        return super().compute(control_action, joint_pos, diff_joint_vel)


@configclass
class DiffVelPDActuatorCfg(IdealPDActuatorCfg):
    """Configuration for :class:`DiffVelPDActuator`."""

    class_type: type = DiffVelPDActuator

    diff_dt: float = 0.002
    """Time interval used by the finite-difference velocity estimator."""

    wrap_to_pi: bool = True
    """Wrap position differences to [-pi, pi] before dividing by ``diff_dt``."""

    velocity_filter_alpha: float = 1.0
    """EMA coefficient for finite-difference velocity; 1.0 disables filtering."""

    reset_position_jump_threshold: float = float("inf")
    """Position jump threshold treated as reset/teleport; <=0 disables detection."""
