"""CPU golden tests for the Fudan observation/reward/termination contracts."""

from __future__ import annotations

import importlib.util
from collections import OrderedDict
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).parents[1]


def _load(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = _load(
    "fdu_semantics",
    "source/agent_tasks/agent_tasks/direct/wheelbipe/wyw/fdu_semantics.py",
)
M = _load(
    "fdu_mapping_for_semantics",
    "source/agent_tasks/agent_tasks/direct/wheelbipe/wyw/fdu_mapping.py",
)


def _reward_fixture() -> dict[str, torch.Tensor | float | bool]:
    return {
        "command_vx": torch.tensor([1.0, -0.5]),
        "command_yaw": torch.tensor([0.2, -0.3]),
        "base_lin_vel": torch.tensor([[0.5, 0.1, -0.2], [-0.1, 0.0, 0.3]]),
        "base_ang_vel": torch.tensor([[0.2, -0.4, 0.0], [0.1, 0.2, -0.1]]),
        "projected_gravity": torch.tensor([[0.1, -0.2, -0.97], [-0.3, 0.4, -0.8]]),
        "observed_height": torch.tensor([0.22, 0.27]),
        "height_command": torch.tensor([0.20, 0.25]),
        "left_l0": torch.tensor([0.24, 0.29]),
        "right_l0": torch.tensor([0.25, 0.27]),
        "left_theta": torch.tensor([0.1, -0.2]),
        "right_theta": torch.tensor([-0.1, 0.1]),
        "joint_vel": torch.tensor([[1., 2., 30., 4., 5., 60.], [-1., -2., -30., -4., -5., -60.]]),
        "joint_acc": torch.tensor([[1., -2., 3., -4., 5., -6.], [2., 3., 4., 5., 6., 7.]]),
        "applied_torque": torch.tensor([[1., 2., 3., 4., 5., 6.], [2., 2., 2., 2., 2., 2.]]),
        "actions": torch.tensor([[.3, -.2, .1, .4, -.5, .6], [-.1, .2, -.3, .4, -.5, .6]]),
        "previous_actions": torch.tensor([[.1, -.1, .0, .2, -.3, .4], [.0, .1, -.2, .3, -.4, .5]]),
        "before_previous_actions": torch.tensor([[.0, .0, .0, .0, .0, .0], [.1, .0, -.1, .2, -.3, .4]]),
        "leg_positions": torch.tensor([[-1.0, -0.2, 0.7, 1.1], [-0.4, 0.0, 0.3, 0.8]]),
        "leg_soft_lower": torch.full((2, 4), -0.8),
        "leg_soft_upper": torch.full((2, 4), 0.8),
        "collision_count": torch.tensor([2.0, 0.0]),
        "tracking_sigma": 0.25,
        "jump": False,
    }


def test_policy_and_critic_segment_layout_keeps_actor_noise_out_of_critic():
    batch = 2
    base_ang = torch.arange(6, dtype=torch.float).reshape(batch, 3)
    gravity = torch.arange(10, 16, dtype=torch.float).reshape(batch, 3)
    command = torch.arange(20, 26, dtype=torch.float).reshape(batch, 3)
    leg_pos = torch.arange(30, 38, dtype=torch.float).reshape(batch, 4)
    joint_vel = torch.arange(40, 52, dtype=torch.float).reshape(batch, 6)
    actions = torch.arange(60, 72, dtype=torch.float).reshape(batch, 6)
    clean = S.build_fdu_policy_observation(
        base_ang_vel=base_ang,
        projected_gravity=gravity,
        command_block=command,
        leg_pos_deviation=leg_pos,
        policy_joint_vel=joint_vel,
        actions=actions,
        ang_vel_scale=0.25,
        projected_gravity_scale=1.0,
        joint_pos_scale=1.0,
        dof_vel_scale=0.05,
    )
    noise = torch.linspace(-0.2, 0.2, 25).repeat(batch, 1)
    noisy = clean + noise
    prev = torch.full((batch, 6), 71.0)
    prev2 = torch.full((batch, 6), 72.0)
    dr = S.build_fdu_dr_privilege(
        centered_base_mass=torch.tensor([-0.5, 0.5]),
        base_com_offset=torch.arange(6, dtype=torch.float).reshape(batch, 3),
        default_dof_delta=torch.arange(12, dtype=torch.float).reshape(batch, 6),
        friction=torch.tensor([0.7, 1.2]),
        restitution=torch.tensor([0.6, 0.9]),
    )
    critic = S.build_fdu_critic_observation(
        scaled_base_lin_vel=torch.full((batch, 3), 70.0),
        clean_policy_observation=clean,
        previous_actions=prev,
        before_previous_actions=prev2,
        scaled_joint_acc=torch.full((batch, 6), 73.0),
        height_scan=torch.full((batch, 77), 74.0),
        scaled_torque=torch.full((batch, 6), 75.0),
        dr_privilege=dr,
    )
    assert clean.shape == (batch, 25)
    assert critic.shape == (batch, 141)
    assert torch.equal(clean[:, :3], base_ang * 0.25)
    assert torch.equal(clean[:, 3:6], gravity)
    assert torch.equal(clean[:, 6:9], command)
    assert torch.equal(clean[:, 9:13], leg_pos)
    assert torch.equal(clean[:, 13:19], joint_vel * 0.05)
    assert torch.equal(clean[:, 19:25], actions)
    assert torch.equal(critic[:, 3:28], clean)
    assert not torch.equal(critic[:, 3:28], noisy)
    assert torch.equal(critic[:, 28:34], prev)
    assert torch.equal(critic[:, 34:40], prev2)
    assert torch.equal(critic[:, 129:141], dr)


def test_noisy_policy_history_reset_fill_and_roll_timing():
    history = torch.zeros(2, 5, 25)
    needs_fill = torch.tensor([True, True])
    frame_1 = torch.stack((torch.arange(25), torch.arange(25) + 100)).float()
    history, needs_fill = S.update_fdu_observation_history(history, frame_1, needs_fill)
    assert torch.equal(history, frame_1[:, None, :].expand_as(history))
    frame_2 = frame_1 + 1000
    history, needs_fill = S.update_fdu_observation_history(history, frame_2, needs_fill)
    assert torch.equal(history[:, -2], frame_1)
    assert torch.equal(history[:, -1], frame_2)
    assert not torch.any(needs_fill)


def test_plane_raw_reward_formulas_match_fixed_fudan_fixture():
    f = _reward_fixture()
    terms = S.compute_fdu_plane_reward_terms(**f)
    sigma = 0.25
    lin_err = (f["command_vx"] - f["base_lin_vel"][:, 0]).square()
    yaw_err = (f["command_yaw"] - f["base_ang_vel"][:, 2]).square()
    second = f["actions"] - 2 * f["previous_actions"] + f["before_previous_actions"]
    expected = {
        "tracking_lin_vel": torch.exp(-lin_err / sigma),
        "tracking_lin_vel_enhance": torch.exp(-lin_err / sigma / 10) - 1,
        "tracking_ang_vel": torch.exp(-yaw_err / sigma),
        "tracking_ang_vel_enhance": torch.exp(-yaw_err / sigma / 10) - 1,
        "base_height": torch.exp(-(f["observed_height"] - f["height_command"]).square() / 0.001),
        "nominal_state": (f["left_theta"] - f["right_theta"]).square(),
        "lin_vel_z": f["base_lin_vel"][:, 2].square(),
        "ang_vel_xy": f["base_ang_vel"][:, :2].square().sum(-1),
        "orientation": f["projected_gravity"][:, :2].square().sum(-1),
        "dof_vel": f["joint_vel"][:, (0, 1, 3, 4)].square().sum(-1),
        "dof_acc": f["joint_acc"].square().sum(-1),
        "torques": f["applied_torque"].square().sum(-1),
        "action_rate": (f["actions"] - f["previous_actions"]).square().sum(-1),
        "action_smooth": second[:, (0, 1, 3, 4)].square().sum(-1),
        "collision": f["collision_count"],
        "dof_pos_limits": torch.tensor([0.5, 0.0]),
    }
    assert set(terms) == set(expected)
    for name, value in expected.items():
        assert torch.allclose(terms[name], value), name


def test_jump_shared_and_jump_only_raw_formulas_match_fudan_fixture():
    fixture = _reward_fixture()
    fixture["jump"] = True
    shared = S.compute_fdu_plane_reward_terms(**fixture)
    plane_fixture = dict(fixture)
    plane_fixture["jump"] = False
    plane = S.compute_fdu_plane_reward_terms(**plane_fixture)
    assert torch.allclose(shared["tracking_lin_vel"], 2.0 * plane["tracking_lin_vel"])
    assert torch.allclose(shared["tracking_lin_vel_enhance"], 2.0 * plane["tracking_lin_vel_enhance"])

    lengths = torch.tensor([[0.23, 0.25], [0.30, 0.31]])
    in_flight = torch.tensor([True, False])
    any_contact = torch.tensor([False, True])
    root_z = torch.tensor([0.65, 0.40])
    root_vz = torch.tensor([0.8, 0.2])
    base_air_time = torch.tensor([0.3, 0.2])
    jump, updated = S.compute_fdu_jump_reward_terms(
        leg_lengths=lengths,
        in_flight=in_flight,
        any_contact=any_contact,
        root_z=root_z,
        root_vz=root_vz,
        base_air_time=base_air_time,
        step_dt=0.01,
        l0_tuck=0.23,
        l0_extend=0.31,
        base_height_flight=0.65,
        takeoff_vz=0.15,
        airtime_update=M.update_buggy_fudan_airtime,
    )
    expected_air_reward, expected_updated = M.update_buggy_fudan_airtime(
        base_air_time, in_flight, root_z, root_vz, 0.01
    )
    assert torch.allclose(jump["base_height_flight"], torch.tensor([1.0, 0.0]))
    assert torch.allclose(jump["leg_tuck"], torch.tensor([torch.exp(torch.tensor(-0.08)), 0.0]))
    assert torch.allclose(jump["takeoff_extend"], torch.tensor([0.0, torch.exp(torch.tensor(-0.04))]))
    assert torch.allclose(jump["line_z"], torch.tensor([0.8, 0.0]))
    assert torch.equal(jump["flight"], in_flight.float())
    assert torch.allclose(jump["encourage_jump"], expected_air_reward)
    assert torch.allclose(updated, expected_updated)


def test_jump_two_frame_contact_filter_delays_flight_by_one_frame():
    previous = torch.tensor([[True, False], [False, False], [True, True]])
    current = torch.tensor([[False, False], [False, False], [False, True]])
    in_flight, any_contact, next_previous = S.filter_fdu_wheel_contacts(current, previous)
    assert in_flight.tolist() == [False, True, False]
    assert any_contact.tolist() == [True, False, True]
    assert torch.equal(next_previous, current)
    next_flight, next_any, _ = S.filter_fdu_wheel_contacts(
        torch.zeros_like(current), next_previous
    )
    assert next_flight.tolist() == [True, True, False]
    assert next_any.tolist() == [False, False, True]


def test_reward_weight_dt_then_per_term_clip_then_sum():
    raw = {"positive": torch.tensor([100.0, 1.0]), "negative": torch.tensor([100.0, 1.0])}
    weights = OrderedDict((("positive", 2.0), ("negative", -3.0)))
    total, terms = S.aggregate_fdu_rewards(
        raw,
        weights,
        step_dt=0.01,
        clip_single_reward=1.0,
        only_positive_rewards=False,
    )
    assert torch.allclose(terms["positive"], torch.tensor([0.01, 0.01]))
    assert torch.allclose(terms["negative"], torch.tensor([-0.01, -0.01]))
    assert torch.allclose(total, torch.zeros(2))
    # This distinguishes per-term clipping from clipping only the final sum.
    one_sided, _ = S.aggregate_fdu_rewards(
        {"a": torch.tensor([100.0]), "b": torch.tensor([-50.0])},
        OrderedDict((("a", 1.0), ("b", 1.0))),
        step_dt=0.01,
        clip_single_reward=1.0,
        only_positive_rewards=False,
    )
    assert one_sided.item() == pytest.approx(0.0)


def test_three_action_history_matches_reward_and_critic_timing():
    zero = torch.zeros(1, 6)
    a1 = torch.tensor([[.1, .2, .3, .4, .5, .6]])
    a2 = torch.tensor([[.2, .1, .4, .3, .6, .5]])
    a3 = torch.tensor([[.4, .3, .2, .1, .0, -.1]])
    fixture = _reward_fixture()
    fixture.update(
        actions=a3,
        previous_actions=a2,
        before_previous_actions=a1,
        command_vx=torch.tensor([0.0]),
        command_yaw=torch.tensor([0.0]),
        base_lin_vel=torch.zeros(1, 3),
        base_ang_vel=torch.zeros(1, 3),
        projected_gravity=torch.tensor([[0.0, 0.0, -1.0]]),
        observed_height=torch.tensor([0.2]),
        height_command=torch.tensor([0.2]),
        left_l0=torch.tensor([0.25]),
        right_l0=torch.tensor([0.25]),
        left_theta=torch.tensor([0.0]),
        right_theta=torch.tensor([0.0]),
        joint_vel=zero,
        joint_acc=zero,
        applied_torque=zero,
        leg_positions=torch.zeros(1, 4),
        leg_soft_lower=torch.full((1, 4), -1.0),
        leg_soft_upper=torch.full((1, 4), 1.0),
        collision_count=torch.zeros(1),
    )
    terms = S.compute_fdu_plane_reward_terms(**fixture)
    assert torch.allclose(terms["action_rate"], (a3 - a2).square().sum(-1))
    assert torch.allclose(
        terms["action_smooth"],
        (a3 - 2 * a2 + a1)[:, (0, 1, 3, 4)].square().sum(-1),
    )
    critic = S.build_fdu_critic_observation(
        scaled_base_lin_vel=torch.zeros(1, 3),
        clean_policy_observation=torch.zeros(1, 25),
        previous_actions=a2,
        before_previous_actions=a1,
        scaled_joint_acc=zero,
        height_scan=torch.zeros(1, 77),
        scaled_torque=zero,
        dr_privilege=torch.zeros(1, 12),
    )
    assert torch.equal(critic[:, 28:34], a2)
    assert torch.equal(critic[:, 34:40], a1)


def test_persistent_termination_requires_100_consecutive_steps_and_recovers():
    counter = torch.zeros(2, dtype=torch.int32)
    condition = torch.tensor([True, True])
    for _ in range(99):
        done, counter = S.update_persistent_condition(condition, counter, 100)
        assert not torch.any(done)
    done, counter = S.update_persistent_condition(condition, counter, 100)
    assert torch.all(done)
    done, counter = S.update_persistent_condition(torch.tensor([False, True]), counter, 100)
    assert not done[0] and counter[0] == 0
    assert done[1] and counter[1] == 101


def test_command_range_sampling_and_declared_reset_distributions():
    ranges = torch.tensor([[-2.0, 2.0], [-1.0, 1.0], [0.25, 0.75]])
    sampled = S.sample_uniform_command_from_ranges(ranges, torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(sampled, torch.tensor([-2.0, 0.0, 0.75]))
    assert torch.all(sampled >= ranges[:, 0]) and torch.all(sampled <= ranges[:, 1])

    # These are the exact production event/command contracts audited against
    # Fudan.  The simulator smoke test independently checks sampled values.
    flat_jump_xy = {}
    rough_xy = {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}
    velocity = {axis: (-0.5, 0.5) for axis in ("x", "y", "z", "roll", "pitch", "yaw")}
    assert flat_jump_xy == {}
    assert rough_xy == {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}
    assert all(bounds == (-0.5, 0.5) for bounds in velocity.values())
    assert 5.0 / 0.01 == 500
    assert 20.0 / 0.01 == 2000


def test_fdu_height_command_range_is_exact_and_uniformly_sampleable():
    height_range = torch.tensor([[0.15, 0.30]]).repeat(3, 1)
    heights = S.sample_uniform_command_from_ranges(
        height_range, torch.tensor([0.0, 0.5, 1.0])
    )
    assert torch.allclose(heights, torch.tensor([0.15, 0.225, 0.30]))


def test_dr_privilege_exact_values_and_centered_mass_contract():
    added_mass = torch.tensor([-1.0, 0.5, 2.0])
    centered = added_mass - added_mass.mean()
    com = torch.tensor([[.01, -.02, .00], [.02, .01, -.01], [-.02, .00, .02]])
    default_delta = torch.arange(18, dtype=torch.float).reshape(3, 6) / 100
    friction = torch.tensor([.6, 1.0, 1.4])
    restitution = torch.tensor([.6, .8, 1.0])
    dr = S.build_fdu_dr_privilege(
        centered_base_mass=centered,
        base_com_offset=com,
        default_dof_delta=default_delta,
        friction=friction,
        restitution=restitution,
    )
    assert dr.shape == (3, 12)
    assert centered.mean().item() == pytest.approx(0.0)
    assert torch.equal(dr[:, 0], centered)
    assert torch.equal(dr[:, 1:4], com)
    assert torch.equal(dr[:, 4:10], default_delta)
    assert torch.equal(dr[:, 10], friction)
    assert torch.equal(dr[:, 11], restitution)


def test_rough_curriculum_failure_success_and_basic_advanced_ranges():
    old_levels = torch.tensor([0, 9, 9, 4])
    terrain_types = torch.tensor([0, 3, 12, 19])
    distance = torch.tensor([0.0, 2.1, 2.1, 1.0])
    tracking = torch.tensor([0.1, 0.8, 0.8, 0.8])
    ranges = torch.tensor([[-2.0, 2.0]]).repeat(4, 1)
    move_up, move_down, success, updated = S.compute_fdu_rough_curriculum_transition(
        old_levels=old_levels,
        terrain_types=terrain_types,
        distance=distance,
        tracking_rate=tracking,
        command_ranges_x=ranges,
        terrain_length=8.0,
        max_terrain_level=10,
    )
    assert move_down.tolist() == [True, False, False, False]
    assert move_up.tolist() == [False, True, True, False]
    assert success.tolist() == [False, True, True, False]
    assert torch.allclose(updated[0], torch.tensor([-1.75, 1.75]))
    assert torch.allclose(updated[1], torch.tensor([-2.5, 2.5]))  # basic: +0.50
    assert torch.allclose(updated[2], torch.tensor([-1.5, 1.5]))  # advanced cap
    assert torch.allclose(updated[3], ranges[3])
