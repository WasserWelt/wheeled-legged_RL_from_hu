"""Stateful wrapped finite-difference joint-velocity estimator."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


class FiniteDifferenceJointVelocity:
    """Estimate batched joint velocity while handling resets per environment."""

    def __init__(
        self,
        dt: float,
        *,
        wrap_to_pi: bool = True,
        reset_position_jump_threshold: float = math.inf,
        velocity_filter_alpha: float = 1.0,
    ):
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        if not 0.0 <= velocity_filter_alpha <= 1.0:
            raise ValueError("velocity_filter_alpha must be in [0, 1]")
        self.dt = float(dt)
        self.wrap_to_pi = bool(wrap_to_pi)
        self.reset_position_jump_threshold = float(reset_position_jump_threshold)
        self.velocity_filter_alpha = float(velocity_filter_alpha)
        self.previous_position: torch.Tensor | None = None
        self.velocity: torch.Tensor | None = None
        self.initialized: torch.Tensor | None = None
        self.last_sample_id: int | None = None

    def _allocate_like(self, position: torch.Tensor) -> None:
        if position.ndim != 2:
            raise ValueError(f"position must have shape (num_envs, num_joints), got {position.shape}")
        self.previous_position = position.detach().clone()
        self.velocity = torch.zeros_like(position)
        self.initialized = torch.zeros(position.shape[0], dtype=torch.bool, device=position.device)
        self.last_sample_id = None

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | slice,
        current_position: torch.Tensor | None = None,
    ) -> None:
        """Clear selected histories, or prime them from their reset positions."""
        if self.previous_position is None:
            if current_position is None:
                return
            self._allocate_like(current_position)
            self.previous_position.copy_(current_position.detach())
            self.initialized.fill_(True)
            return

        assert self.velocity is not None and self.initialized is not None
        self.velocity[env_ids] = 0.0
        if current_position is None:
            self.initialized[env_ids] = False
        else:
            selected = current_position
            if current_position.shape == self.previous_position.shape:
                selected = current_position[env_ids]
            self.previous_position[env_ids] = selected.detach()
            self.initialized[env_ids] = True
        if current_position is None:
            self.last_sample_id = None

    def update(self, position: torch.Tensor, *, sample_id: int | None = None) -> torch.Tensor:
        """Consume one fresh position sample and return the current estimate."""
        if self.previous_position is None or self.previous_position.shape != position.shape:
            self._allocate_like(position)
        assert self.previous_position is not None
        assert self.velocity is not None
        assert self.initialized is not None

        if sample_id is not None and sample_id == self.last_sample_id:
            return self.velocity

        delta = position - self.previous_position
        if self.wrap_to_pi:
            delta = torch.atan2(torch.sin(delta), torch.cos(delta))
        raw_velocity = delta / self.dt

        valid = self.initialized.unsqueeze(-1).expand_as(raw_velocity)
        threshold = self.reset_position_jump_threshold
        if math.isfinite(threshold) and threshold > 0.0:
            jumped = torch.any(torch.abs(delta) > threshold, dim=-1, keepdim=True)
            valid = valid & ~jumped

        raw_velocity = torch.where(valid, raw_velocity, torch.zeros_like(raw_velocity))
        alpha = self.velocity_filter_alpha
        if alpha < 1.0:
            filtered = alpha * raw_velocity + (1.0 - alpha) * self.velocity
            raw_velocity = torch.where(valid, filtered, raw_velocity)

        self.velocity.copy_(raw_velocity.detach())
        self.previous_position.copy_(position.detach())
        self.initialized.fill_(True)
        self.last_sample_id = sample_id
        return self.velocity
