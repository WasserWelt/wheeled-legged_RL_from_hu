"""Pure tensor semantics shared by the WYW/FDU environment and golden tests.

This module deliberately has no Isaac Lab imports.  Keeping observation layout,
reward algebra, clipping order, and history timing here makes those contracts
testable without launching a simulator while ensuring the tests exercise the
same implementation used by training.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch


def build_fdu_policy_observation(
    *,
    base_ang_vel: torch.Tensor,
    projected_gravity: torch.Tensor,
    command_block: torch.Tensor,
    leg_pos_deviation: torch.Tensor,
    policy_joint_vel: torch.Tensor,
    actions: torch.Tensor,
    ang_vel_scale: float,
    projected_gravity_scale: float,
    joint_pos_scale: float,
    dof_vel_scale: float,
) -> torch.Tensor:
    """Build Fudan's 25-D actor observation in its authoritative segment order."""
    obs = torch.cat(
        (
            base_ang_vel * ang_vel_scale,
            projected_gravity * projected_gravity_scale,
            command_block,
            leg_pos_deviation * joint_pos_scale,
            policy_joint_vel * dof_vel_scale,
            actions,
        ),
        dim=-1,
    )
    if obs.shape[-1] != 25:
        raise ValueError(f"FDU policy observation must be 25-D, got {obs.shape}")
    return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)


def build_fdu_dr_privilege(
    *,
    centered_base_mass: torch.Tensor,
    base_com_offset: torch.Tensor,
    default_dof_delta: torch.Tensor,
    friction: torch.Tensor,
    restitution: torch.Tensor,
) -> torch.Tensor:
    """Build the 12-D domain-randomization privilege tail."""
    dr = torch.cat(
        (
            centered_base_mass.reshape(-1, 1),
            base_com_offset.reshape(-1, 3),
            default_dof_delta.reshape(-1, 6),
            friction.reshape(-1, 1),
            restitution.reshape(-1, 1),
        ),
        dim=-1,
    )
    if dr.shape[-1] != 12:
        raise ValueError(f"FDU DR privilege must be 12-D, got {dr.shape}")
    return torch.nan_to_num(dr, nan=0.0, posinf=0.0, neginf=0.0)


def build_fdu_critic_observation(
    *,
    scaled_base_lin_vel: torch.Tensor,
    clean_policy_observation: torch.Tensor,
    previous_actions: torch.Tensor,
    before_previous_actions: torch.Tensor,
    scaled_joint_acc: torch.Tensor,
    height_scan: torch.Tensor,
    scaled_torque: torch.Tensor,
    dr_privilege: torch.Tensor,
) -> torch.Tensor:
    """Build Fudan's 141-D privileged observation in exact segment order."""
    critic = torch.cat(
        (
            scaled_base_lin_vel,
            clean_policy_observation,
            previous_actions,
            before_previous_actions,
            scaled_joint_acc,
            height_scan,
            scaled_torque,
            dr_privilege,
        ),
        dim=-1,
    )
    if critic.shape[-1] != 141:
        raise ValueError(f"FDU critic observation must be 141-D, got {critic.shape}")
    return torch.nan_to_num(critic, nan=0.0, posinf=0.0, neginf=0.0)


def update_fdu_observation_history(
    history: torch.Tensor,
    policy_observation: torch.Tensor,
    needs_fill: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append one actor frame; reset-marked rows are filled with that frame."""
    updated = torch.roll(history, shifts=-1, dims=1)
    updated[:, -1] = policy_observation
    fill_ids = needs_fill.nonzero(as_tuple=False).flatten()
    if fill_ids.numel() > 0:
        updated[fill_ids] = policy_observation[fill_ids].unsqueeze(1)
    return updated, torch.zeros_like(needs_fill)


def compute_fdu_plane_reward_terms(
    *,
    command_vx: torch.Tensor,
    command_yaw: torch.Tensor,
    base_lin_vel: torch.Tensor,
    base_ang_vel: torch.Tensor,
    tracking_lin_vel_x: torch.Tensor,
    tracking_yaw_rate: torch.Tensor,
    projected_gravity: torch.Tensor,
    observed_height: torch.Tensor,
    height_command: torch.Tensor,
    left_l0: torch.Tensor,
    right_l0: torch.Tensor,
    left_theta: torch.Tensor,
    right_theta: torch.Tensor,
    joint_vel: torch.Tensor,
    joint_acc: torch.Tensor,
    applied_torque: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    before_previous_actions: torch.Tensor,
    leg_positions: torch.Tensor,
    leg_soft_lower: torch.Tensor,
    leg_soft_upper: torch.Tensor,
    collision_count: torch.Tensor,
    tracking_sigma: float,
    upright_orientation_sigma: float,
    jump: bool,
) -> dict[str, torch.Tensor]:
    """Return raw Fudan Plane terms plus the shared subset used by Jump."""
    sigma = max(float(tracking_sigma), 1.0e-6)
    upright_sigma = max(float(upright_orientation_sigma), 1.0e-6)
    lin_err = torch.square(command_vx - tracking_lin_vel_x)
    ang_err = torch.square(command_yaw - tracking_yaw_rate)
    lin_factor = 2.0 if jump else 1.0
    tracking_gate = (
        torch.ones_like(command_vx)
        if jump
        else torch.clamp(-projected_gravity[:, 2], min=0.0, max=0.7) / 0.7
    )
    action_second = actions - 2.0 * previous_actions + before_previous_actions
    pos_limit_penalty = torch.sum(
        torch.clamp(leg_soft_lower - leg_positions, min=0.0)
        + torch.clamp(leg_positions - leg_soft_upper, min=0.0),
        dim=-1,
    )
    return {
        "tracking_lin_vel": torch.exp(-lin_err / sigma) * lin_factor * tracking_gate,
        "tracking_lin_vel_enhance": (torch.exp(-lin_err / (10.0 * sigma)) - 1.0) * lin_factor,
        "tracking_ang_vel": torch.exp(-ang_err / sigma) * tracking_gate,
        "tracking_ang_vel_enhance": torch.exp(-ang_err / (10.0 * sigma)) - 1.0,
        "base_height": torch.exp(-torch.square(observed_height - height_command) / 0.001),
        # Narrow positive upright reward. The xy norm is sin(tilt)^2 for a
        # normalized gravity vector; the hemisphere gate rejects inverted
        # poses that otherwise share the same xy projection as upright ones.
        "upright_orientation": torch.exp(
            -torch.square(projected_gravity[:, :2]).sum(dim=-1) / upright_sigma
        ) * (projected_gravity[:, 2] < 0.0).to(projected_gravity.dtype),
        "nominal_state": torch.square(left_theta - right_theta),
        "lin_vel_z": torch.square(base_lin_vel[:, 2]),
        "ang_vel_xy": torch.square(base_ang_vel[:, :2]).sum(dim=-1),
        "orientation": torch.square(projected_gravity[:, :2]).sum(dim=-1),
        "dof_vel": torch.square(joint_vel[:, (0, 1, 3, 4)]).sum(dim=-1),
        "dof_acc": torch.square(joint_acc).sum(dim=-1),
        "torques": torch.square(applied_torque).sum(dim=-1),
        "action_rate": torch.square(actions - previous_actions).sum(dim=-1),
        "action_smooth": torch.square(action_second[:, (0, 1, 3, 4)]).sum(dim=-1),
        "collision": collision_count,
        "dof_pos_limits": pos_limit_penalty,
    }


def compute_fdu_jump_reward_terms(
    *,
    leg_lengths: torch.Tensor,
    in_flight: torch.Tensor,
    any_contact: torch.Tensor,
    root_z: torch.Tensor,
    root_vz: torch.Tensor,
    base_air_time: torch.Tensor,
    step_dt: float,
    l0_tuck: float,
    l0_extend: float,
    base_height_flight: float,
    takeoff_vz: float,
    airtime_update,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Return raw Fudan Jump-only terms and the updated bug-compatible timer."""
    in_flight_f = in_flight.to(dtype=leg_lengths.dtype)
    left_l0, right_l0 = leg_lengths[:, 0], leg_lengths[:, 1]
    encourage_jump, updated_air_time = airtime_update(
        base_air_time, in_flight, root_z, root_vz, step_dt
    )
    terms = {
        "base_height_flight": torch.exp(-torch.abs(root_z - base_height_flight) * 6.0) * in_flight_f,
        "leg_tuck": torch.exp(
            -(torch.abs(left_l0 - l0_tuck) + torch.abs(right_l0 - l0_tuck)) * 4.0
        ) * in_flight_f,
        "takeoff_extend": torch.exp(
            -(torch.abs(left_l0 - l0_extend) + torch.abs(right_l0 - l0_extend)) * 4.0
        ) * (any_contact & (root_vz > takeoff_vz)).to(dtype=leg_lengths.dtype),
        "line_z": torch.clamp(root_vz, min=0.0) * in_flight_f,
        "flight": in_flight_f,
        "encourage_jump": encourage_jump,
    }
    return terms, updated_air_time


def filter_fdu_wheel_contacts(
    contact_now: torch.Tensor,
    previous_contact: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply Fudan's two-frame wheel contact filter.

    Returns ``in_flight, any_contact, next_previous_contact``.  A wheel is
    treated as touching when either the current or immediately preceding
    policy frame reports contact.
    """
    if contact_now.shape != previous_contact.shape or contact_now.shape[-1] != 2:
        raise ValueError(
            "FDU wheel contact tensors must have matching shape (N,2), got "
            f"{contact_now.shape} and {previous_contact.shape}"
        )
    contact_now = contact_now.to(dtype=torch.bool)
    contact_filt = contact_now | previous_contact.to(dtype=torch.bool)
    return torch.all(~contact_filt, dim=-1), torch.any(contact_filt, dim=-1), contact_now


def aggregate_fdu_rewards(
    raw_terms: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
    *,
    step_dt: float,
    clip_single_reward: float | None,
    only_positive_rewards: bool,
    invalid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply Fudan's weight×dt, per-term clipping, then summation contract."""
    if not weights:
        raise ValueError("FDU reward weights must not be empty")
    exemplar = next(iter(raw_terms.values()))
    weighted = {
        name: float(weight) * raw_terms.get(name, torch.zeros_like(exemplar)) * step_dt
        for name, weight in weights.items()
    }
    if clip_single_reward is not None:
        bound = abs(float(clip_single_reward)) * step_dt
        weighted = {name: torch.clamp(value, -bound, bound) for name, value in weighted.items()}
    for name, value in weighted.items():
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        if invalid_mask is not None:
            value = torch.where(invalid_mask, torch.zeros_like(value), value)
        weighted[name] = value
    total = torch.stack([weighted[name] for name in weights], dim=-1).sum(dim=-1)
    if only_positive_rewards:
        total = torch.clamp_min(total, 0.0)
    return torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0), weighted


def compute_fdu_failure_termination_reward(
    persistent_failure: torch.Tensor,
    termination_scale: float,
    step_dt: float,
) -> torch.Tensor:
    """Return the one-shot failure reward applied after per-term clipping."""
    return persistent_failure.to(dtype=torch.float) * float(termination_scale) * float(step_dt)


def update_persistent_condition(
    condition: torch.Tensor,
    counter: torch.Tensor,
    required_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Count consecutive true samples and return the persistent condition."""
    if required_steps < 1:
        raise ValueError(f"required_steps must be >=1, got {required_steps}")
    updated = torch.where(condition, counter + 1, torch.zeros_like(counter))
    return updated >= required_steps, updated


def compute_fdu_failure_contact_condition(
    contact_forces: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Return Fudan's per-env rf/lf/base contact failure condition."""
    if contact_forces.ndim != 3 or contact_forces.shape[-1] != 3:
        raise ValueError(
            "failure contact forces must have shape (N,B,3), got "
            f"{contact_forces.shape}"
        )
    return torch.any(torch.linalg.vector_norm(contact_forces, dim=-1) > float(threshold), dim=-1)


def compute_fdu_collision_count(
    contact_forces: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Count mapped Fudan collision bodies above the force threshold per env."""
    if contact_forces.ndim != 3 or contact_forces.shape[-1] != 3:
        raise ValueError(
            "collision contact forces must have shape (N,B,3), got "
            f"{contact_forces.shape}"
        )
    return (
        torch.linalg.vector_norm(contact_forces, dim=-1) > float(threshold)
    ).to(dtype=contact_forces.dtype).sum(dim=-1)


def sample_uniform_command_from_ranges(
    ranges: torch.Tensor,
    unit_samples: torch.Tensor,
) -> torch.Tensor:
    """Map per-environment U[0,1] samples into per-environment command ranges."""
    if ranges.ndim != 2 or ranges.shape[-1] != 2:
        raise ValueError(f"command ranges must have shape (N,2), got {ranges.shape}")
    if unit_samples.shape != ranges.shape[:-1]:
        raise ValueError(
            f"unit samples must have shape {ranges.shape[:-1]}, got {unit_samples.shape}"
        )
    return ranges[:, 0] + unit_samples * (ranges[:, 1] - ranges[:, 0])


def compute_fdu_flat_command_curriculum_transition(
    *,
    command_ranges_x: torch.Tensor,
    mean_tracking_lin_rate: torch.Tensor | float,
    mean_tracking_yaw_rate: torch.Tensor | float,
    lin_threshold: float,
    yaw_threshold: float,
    expansion_step: float,
    max_abs: float,
) -> tuple[bool, torch.Tensor]:
    """Apply Fudan Plane's global vx-range expansion for one curriculum tick."""
    if command_ranges_x.ndim != 2 or command_ranges_x.shape[-1] != 2:
        raise ValueError(f"command ranges must have shape (N,2), got {command_ranges_x.shape}")
    lin_rate = float(torch.as_tensor(mean_tracking_lin_rate).item())
    yaw_rate = float(torch.as_tensor(mean_tracking_yaw_rate).item())
    should_expand = lin_rate > float(lin_threshold) and yaw_rate > float(yaw_threshold)
    updated = command_ranges_x.clone()
    if should_expand:
        updated[:, 0] = torch.clamp(
            updated[:, 0] - float(expansion_step), min=-float(max_abs), max=0.0
        )
        updated[:, 1] = torch.clamp(
            updated[:, 1] + float(expansion_step), min=0.0, max=float(max_abs)
        )
    return should_expand, updated


def get_due_fdu_flat_command_curriculum_step(
    *, current_step: int, interval: int, last_consumed_step: int
) -> int | None:
    """Return the earliest reached Flat cadence not yet consumed."""
    if current_step < interval:
        return None
    due_step = ((max(last_consumed_step, 0) // interval) + 1) * interval
    if due_step <= last_consumed_step:
        return None
    if due_step > current_step:
        return None
    return due_step


def compute_fdu_rough_curriculum_transition(
    *,
    old_levels: torch.Tensor,
    terrain_types: torch.Tensor,
    distance: torch.Tensor,
    tracking_rate: torch.Tensor,
    command_ranges_x: torch.Tensor,
    terrain_length: float,
    max_terrain_level: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Fudan Rough move masks and command-range updates for one reset.

    Returns ``move_up, move_down, success, updated_command_ranges``.  Terrain
    level clamping/random wrap remains owned by Isaac Lab's terrain importer.
    """
    move_up = distance > float(terrain_length) / 4.0
    move_down = (tracking_rate < 0.4) & (~move_up)
    candidates = old_levels + move_up.long() - move_down.long()
    success = candidates >= int(max_terrain_level)
    failure = candidates < 0
    updated = command_ranges_x.clone()

    if torch.any(failure):
        updated[failure, 0] = torch.clamp(updated[failure, 0] + 0.25, min=-2.5, max=-1.0)
        updated[failure, 1] = torch.clamp(updated[failure, 1] - 0.25, min=1.0, max=2.5)

    expand = success & (tracking_rate > 0.7)
    if torch.any(expand):
        basic = (terrain_types < 12) | ((terrain_types >= 14) & (terrain_types < 18))
        delta = torch.where(basic, 0.5, 0.05)
        max_abs = torch.where(basic, 2.5, 1.5)
        mask = expand
        updated[mask, 0] = torch.maximum(updated[mask, 0] - delta[mask], -max_abs[mask])
        updated[mask, 1] = torch.minimum(updated[mask, 1] + delta[mask], max_abs[mask])
    return move_up, move_down, success, updated
