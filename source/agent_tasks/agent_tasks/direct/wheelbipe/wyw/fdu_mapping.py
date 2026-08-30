"""Explicit Fudan closed-chain action mapping helpers."""

from __future__ import annotations

import math

import torch

# Geometry extracted from the authoritative infantry_V2.urdf.  The URDF has
# 0.17472 m front/rear crank offsets and a 0.208 m crank-to-wheel link.  Keep
# the measured value here (rather than rounding it to Fudan's 0.175 m) so the
# analytical kite solver and the asset agree at sub-millimetre scale.
FDU_L1 = 0.17472
FDU_L2 = 0.208
# At the URDF assembly pose the front chain ends at (L1, L2).  Requiring the
# rear L2 link to meet that point gives the rear-crank installation angle
# 2*atan2(L2, L1).  The previous 1.6614 value belonged to the old mapping and
# produces a false ~1--2 cm geometry bias on this specified closed-chain body.
FDU_DEFAULT_OFFSET = 2.0 * math.atan2(FDU_L2, FDU_L1)
FDU_VERTICAL_JOINT_CENTER = math.pi / 2.0 - math.atan2(FDU_L2, FDU_L1)
# The specified model keeps the real mirrored passive-joint limits:
# lf1 in [-0.63, 1.10] rad and rf1 in [-1.10, 0.63] rad.  On the imported
# assembly branch rf1 == -lf1.  These limits are stricter than the bare
# triangle inequality and bound vertical L0 to approximately
# [0.09495, 0.34149] m.  Keep the geometric and mechanical bounds distinct:
# the linkage-length-only upper bound is FDU_L1 + FDU_L2 = 0.38272 m.
FDU_LEFT_PASSIVE_LIMITS = (-0.63, 1.10)
FDU_MECHANICAL_L0_MIN = math.sqrt(
    FDU_L1**2
    + FDU_L2**2
    + 2.0 * FDU_L1 * FDU_L2 * math.cos(math.pi / 2.0 + FDU_LEFT_PASSIVE_LIMITS[1])
)
FDU_MECHANICAL_L0_MAX = math.sqrt(
    FDU_L1**2
    + FDU_L2**2
    + 2.0 * FDU_L1 * FDU_L2 * math.cos(math.pi / 2.0 + FDU_LEFT_PASSIVE_LIMITS[0])
)
FDU_MAP_EPS = 1.0e-5

POLICY_JOINT_NAMES = (
    "lf0_Joint", "l20_Joint", "l_wheel_Joint",
    "rf0_Joint", "r20_Joint", "r_wheel_Joint",
)


def solve_equivalent_kite(
    phi_front: torch.Tensor,
    phi_rear: torch.Tensor,
    *,
    crank_length: float = FDU_L1,
    coupler_length: float = FDU_L2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve the planar equivalent kite for batched actuator angles.

    ``phi_front`` and ``phi_rear`` are absolute crank angles in the Fudan
    right-side planar frame.  The returned point ``P`` is the virtual wheel
    point, ``phi0`` is its polar angle in that frame, and ``valid`` marks
    inputs whose circle intersection is real.  The positive square-root branch
    is the assembly branch used by Fudan's trained solver.
    """
    l1 = torch.as_tensor(crank_length, dtype=phi_front.dtype, device=phi_front.device)
    l2 = torch.as_tensor(coupler_length, dtype=phi_front.dtype, device=phi_front.device)
    xa, ya = l1 * torch.cos(phi_front), l1 * torch.sin(phi_front)
    xb, yb = l1 * torch.cos(phi_rear), l1 * torch.sin(phi_rear)
    dx, dy = xb - xa, yb - ya
    c = torch.square(dx) + torch.square(dy)
    a, b = 2.0 * l2 * dx, 2.0 * l2 * dy
    discriminant = a.square() + b.square() - c.square()
    valid = discriminant >= -1.0e-7
    root = torch.sqrt(torch.clamp(discriminant, min=0.0))
    phi_link = 2.0 * torch.atan2(b + root, a + c)
    px = xa + l2 * torch.cos(phi_link)
    py = ya + l2 * torch.sin(phi_link)
    phi0 = torch.atan2(py, px)
    l0 = torch.sqrt(torch.clamp(px.square() + py.square(), min=1.0e-12))
    return torch.stack((px, py), dim=-1), l0, phi0, discriminant, valid


def equivalent_kite_jacobian(
    phi_front: torch.Tensor,
    phi_rear: torch.Tensor,
    *,
    crank_length: float = FDU_L1,
    coupler_length: float = FDU_L2,
    eps: float = FDU_MAP_EPS,
) -> torch.Tensor:
    """Return ``d[L0, phi0]/d[phi_front, phi_rear]`` by central differences."""
    front_p = solve_equivalent_kite(phi_front + eps, phi_rear, crank_length=crank_length, coupler_length=coupler_length)
    front_m = solve_equivalent_kite(phi_front - eps, phi_rear, crank_length=crank_length, coupler_length=coupler_length)
    rear_p = solve_equivalent_kite(phi_front, phi_rear + eps, crank_length=crank_length, coupler_length=coupler_length)
    rear_m = solve_equivalent_kite(phi_front, phi_rear - eps, crank_length=crank_length, coupler_length=coupler_length)
    d_front = torch.stack(((front_p[1] - front_m[1]) / (2.0 * eps), (front_p[2] - front_m[2]) / (2.0 * eps)), dim=-1)
    d_rear = torch.stack(((rear_p[1] - rear_m[1]) / (2.0 * eps), (rear_p[2] - rear_m[2]) / (2.0 * eps)), dim=-1)
    return torch.stack((d_front, d_rear), dim=-1)


def inverse_equivalent_kite(
    l0: torch.Tensor,
    theta0: torch.Tensor,
    *,
    elbow_sign: float = 1.0,
    crank_length: float = FDU_L1,
    coupler_length: float = FDU_L2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map desired ``L0/theta0`` to left front/rear entity-joint angles.

    The returned order is ``(front_joint, rear_joint, valid)``.  The default
    positive branch is continuous with the imported assembly and gives the
    vertical zero-height pose at ``front=rear=FDU_VERTICAL_JOINT_CENTER``.
    Right-side entity angles are the negatives of these left-side values.
    """
    l1 = torch.as_tensor(crank_length, dtype=l0.dtype, device=l0.device)
    l2 = torch.as_tensor(coupler_length, dtype=l0.dtype, device=l0.device)
    geometric_valid = (l0 >= torch.abs(l1 - l2)) & (l0 <= l1 + l2) & (l0 > 1.0e-8)
    cosine = (l1.square() + l0.square() - l2.square()) / (2.0 * l1 * l0.clamp_min(1.0e-8))
    beta = torch.acos(torch.clamp(cosine, -1.0, 1.0)) * float(elbow_sign)
    polar = theta0 + torch.pi / 2.0
    rear_absolute = polar + beta
    front_joint = polar - beta
    rear_joint = rear_absolute - FDU_DEFAULT_OFFSET
    # The passive joint zero has its lower link along local +Y, hence the
    # extra pi/2 installation rotation.  This angle is invariant to theta0.
    px = l0 * torch.cos(polar)
    py = l0 * torch.sin(polar)
    knee_x = l1 * torch.cos(front_joint)
    knee_y = l1 * torch.sin(front_joint)
    coupler_angle = torch.atan2(py - knee_y, px - knee_x)
    passive_joint = torch.atan2(
        torch.sin(coupler_angle - front_joint - torch.pi / 2.0),
        torch.cos(coupler_angle - front_joint - torch.pi / 2.0),
    )
    mechanical_valid = (
        (passive_joint >= FDU_LEFT_PASSIVE_LIMITS[0] - 1.0e-6)
        & (passive_joint <= FDU_LEFT_PASSIVE_LIMITS[1] + 1.0e-6)
    )
    valid = geometric_valid & mechanical_valid
    return front_joint, rear_joint, valid


def compute_fdu_equivalent_leg_state(
    lf0: torch.Tensor,
    l20: torch.Tensor,
    rf0: torch.Tensor,
    r20: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(L0_left, theta_left, L0_right, theta_right)``.

    The joint signs follow the specified policy order and the Fudan mirror:
    left uses ``(offset+l20, lf0)``; right uses ``(offset-r20, -rf0)``.
    ``theta0`` uses Fudan's zero along +vertical, matching the reward code.
    """
    _, left_l0, left_phi0, _, left_valid = solve_equivalent_kite(
        FDU_DEFAULT_OFFSET + l20, lf0
    )
    _, right_l0, right_phi0, _, right_valid = solve_equivalent_kite(
        FDU_DEFAULT_OFFSET - r20, -rf0
    )
    # Keep tensor outputs usable for batched scans; callers that need the
    # reachability mask can inspect solve_equivalent_kite directly.
    del left_valid, right_valid
    return left_l0, left_phi0 - torch.pi / 2.0, right_l0, right_phi0 - torch.pi / 2.0


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
    phi1 = FDU_DEFAULT_OFFSET + rear
    _, phi3, l0 = _solve(phi1, front)
    return phi3 - front - torch.pi / 2.0, l0


def _right_virtual(front: torch.Tensor, rear: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    phi1, phi4 = FDU_DEFAULT_OFFSET - rear, -front
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
