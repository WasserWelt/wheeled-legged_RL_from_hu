"""Explicit Fudan closed-chain action mapping helpers."""

from __future__ import annotations

import torch

FDU_L1 = 0.175
FDU_L2 = 0.208
FDU_DEFAULT_OFFSET = 1.6614
FDU_MAP_EPS = 1.0e-5

POLICY_JOINT_NAMES = (
    "lf0_Joint", "l20_Joint", "l_wheel_Joint",
    "rf0_Joint", "r20_Joint", "r_wheel_Joint",
)


def update_buggy_fudan_airtime(
    base_air_time: torch.Tensor,
    in_flight: torch.Tensor,
    root_z: torch.Tensor,
    root_vz: torch.Tensor,
    step_dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce the trained Fudan Jump airtime bug, including update order.

    The source masks the accumulator with ``~in_flight``. Consequently it
    accumulates while grounded and is cleared in flight. This is intentionally
    retained for baseline compatibility; do not correct the mask here.
    """
    first_contact = (base_air_time > 0.0) & (~in_flight)
    updated_air_time = base_air_time + step_dt * torch.clamp(root_z, min=0.0, max=0.5)
    reward = (updated_air_time - 5.0e-5) * first_contact.float() * 0.15
    reward = reward + torch.clamp(root_vz, min=0.0) * 0.15
    updated_air_time = updated_air_time * (~in_flight).to(updated_air_time.dtype)
    return reward, updated_air_time


def _solve(phi1: torch.Tensor, phi4: torch.Tensor) -> tuple[torch.Tensor, ...]:
    xb, yb = FDU_L1 * torch.cos(phi1), FDU_L1 * torch.sin(phi1)
    xd, yd = FDU_L1 * torch.cos(phi4), FDU_L1 * torch.sin(phi4)
    dx, dy = xd - xb, yd - yb
    a0, b0 = 2.0 * FDU_L2 * dx, 2.0 * FDU_L2 * dy
    c0 = dx.square() + dy.square()
    disc = torch.clamp(a0.square() + b0.square() - c0.square(), min=0.0)
    phi2 = 2.0 * torch.atan2(b0 + torch.sqrt(disc), a0 + c0)
    xc = xb + FDU_L2 * torch.cos(phi2)
    yc = yb + FDU_L2 * torch.sin(phi2)
    phi3 = torch.atan2(yc - yd, xc - xd)
    l0 = torch.sqrt(torch.clamp(xc.square() + yc.square(), min=1.0e-12))
    return phi2, phi3, l0


def _left_virtual(front: torch.Tensor, rear: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    phi4 = FDU_DEFAULT_OFFSET + rear
    _, phi3, l0 = _solve(front, phi4)
    return phi3 - phi4 - torch.pi / 2.0, l0


def _right_virtual(front: torch.Tensor, rear: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    phi1, phi4 = -front, FDU_DEFAULT_OFFSET - rear
    _, phi3, l0 = _solve(phi1, phi4)
    return -phi3 + phi4 + torch.pi / 2.0, l0


def compute_fudan_virtual_leg_state(
    lf0: torch.Tensor,
    l20: torch.Tensor,
    rf0: torch.Tensor,
    r20: torch.Tensor,
    lf0d: torch.Tensor | None = None,
    l20d: torch.Tensor | None = None,
    rf0d: torch.Tensor | None = None,
    r20d: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return virtual knee positions/velocities for diagnostics only."""
    lf1, _ = _left_virtual(lf0, l20)
    rf1, _ = _right_virtual(rf0, r20)
    zeros = torch.zeros_like(lf0)
    lf0d = zeros if lf0d is None else lf0d
    l20d = zeros if l20d is None else l20d
    rf0d = zeros if rf0d is None else rf0d
    r20d = zeros if r20d is None else r20d
    eps = FDU_MAP_EPS
    l_front = (_left_virtual(lf0 + eps, l20)[0] - _left_virtual(lf0 - eps, l20)[0]) / (2 * eps)
    l_rear = (_left_virtual(lf0, l20 + eps)[0] - _left_virtual(lf0, l20 - eps)[0]) / (2 * eps)
    r_front = (_right_virtual(rf0 + eps, r20)[0] - _right_virtual(rf0 - eps, r20)[0]) / (2 * eps)
    r_rear = (_right_virtual(rf0, r20 + eps)[0] - _right_virtual(rf0, r20 - eps)[0]) / (2 * eps)
    return lf1, rf1, l_front * lf0d + l_rear * l20d, r_front * rf0d + r_rear * r20d


def map_virtual_leg_torque(*args, **kwargs):
    """Deprecated compatibility hook; WYW does not use virtual torque mapping."""
    raise RuntimeError("WYW FDU control uses direct entity-bar position targets; virtual torque mapping is disabled")
