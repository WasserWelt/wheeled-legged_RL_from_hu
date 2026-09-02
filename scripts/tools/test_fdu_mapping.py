"""Pure torch checks for the FDU semantic and diagnostic mapping."""

import pytest
import torch
import importlib.util
from pathlib import Path

_PATH = Path(__file__).parents[1] / "source/agent_tasks/agent_tasks/direct/wheelbipe/wyw/fdu_mapping.py"
_SPEC = importlib.util.spec_from_file_location("fdu_mapping", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
POLICY_JOINT_NAMES = _MODULE.POLICY_JOINT_NAMES
compute_fudan_virtual_leg_state = _MODULE.compute_fudan_virtual_leg_state
compute_fdu_equivalent_leg_state = _MODULE.compute_fdu_equivalent_leg_state
equivalent_kite_jacobian = _MODULE.equivalent_kite_jacobian
inverse_equivalent_kite = _MODULE.inverse_equivalent_kite
solve_equivalent_kite = _MODULE.solve_equivalent_kite
map_virtual_leg_torque = _MODULE.map_virtual_leg_torque
update_buggy_fudan_airtime = _MODULE.update_buggy_fudan_airtime


def test_policy_order_is_left_then_right_entity_control():
    assert POLICY_JOINT_NAMES == (
        "lf0_Joint", "l20_Joint", "l_wheel_Joint",
        "rf0_Joint", "r20_Joint", "r_wheel_Joint",
    )


def test_virtual_diagnostics_are_finite_and_symmetric():
    front = torch.tensor([0.0, 0.2, -0.2])
    rear = torch.tensor([0.0, 0.1, -0.1])
    lf1, rf1, lf1d, rf1d = compute_fudan_virtual_leg_state(
        front, rear, -front, -rear, torch.ones_like(front), torch.zeros_like(rear),
        -torch.ones_like(front), torch.zeros_like(rear),
    )
    for value in (lf1, rf1, lf1d, rf1d):
        assert torch.isfinite(value).all()
    assert torch.allclose(lf1, -rf1, atol=2.0e-5)
    assert torch.allclose(lf1d, -rf1d, atol=2.0e-3)


def test_virtual_torque_mapping_is_explicitly_disabled():
    with pytest.raises(RuntimeError, match="direct entity-bar"):
        map_virtual_leg_torque()


def test_equivalent_kite_is_finite_and_left_right_mirror_symmetric():
    left = torch.tensor([0.20, -0.23])
    rear = torch.tensor([0.40, -0.65])
    ll, lt, rl, rt = compute_fdu_equivalent_leg_state(left, rear, -left, -rear)
    assert torch.isfinite(torch.stack((ll, lt, rl, rt))).all()
    assert torch.allclose(ll, rl, atol=2.0e-6)
    # Both sides use the same body x/z convention; the y-mirror is removed
    # before theta0 is formed, so theta0 itself has the same sign.
    assert torch.allclose(lt, rt, atol=2.0e-6)
    zero = torch.zeros(1)
    zero_l0, zero_theta, _, _ = compute_fdu_equivalent_leg_state(zero, zero, zero, zero)
    expected_l0 = torch.sqrt(torch.tensor([_MODULE.FDU_L1**2 + _MODULE.FDU_L2**2]))
    expected_theta = torch.atan2(torch.tensor([_MODULE.FDU_L2]), torch.tensor([_MODULE.FDU_L1])) - torch.pi / 2.0
    assert torch.allclose(zero_l0, expected_l0, atol=2.0e-6)
    assert torch.allclose(zero_theta, expected_theta, atol=2.0e-6)


def test_equivalent_kite_jacobian_matches_finite_difference_shape():
    front = torch.tensor([0.2, -0.1])
    rear = torch.tensor([1.9, 1.5])
    jac = equivalent_kite_jacobian(front, rear)
    assert jac.shape == (2, 2, 2)
    assert torch.isfinite(jac).all()
    point, l0, phi0, disc, valid = solve_equivalent_kite(front, rear)
    assert point.shape == (2, 2)
    assert torch.all(valid)
    assert torch.all(disc > 0.0)
    assert torch.allclose(torch.linalg.vector_norm(point, dim=-1), l0, atol=2.0e-6)
    assert torch.allclose(torch.atan2(point[:, 1], point[:, 0]), phi0, atol=2.0e-6)


def test_equivalent_kite_inverse_round_trip_and_vertical_center():
    lengths = torch.tensor([0.20, 0.27, 0.34])
    theta = torch.tensor([-0.2, 0.0, 0.25])
    front, rear, valid = inverse_equivalent_kite(lengths, theta)
    assert torch.all(valid)
    l0, out_theta, _, _ = compute_fdu_equivalent_leg_state(front, rear, -front, -rear)
    assert torch.allclose(l0, lengths, atol=3.0e-6)
    angle_error = torch.atan2(torch.sin(out_theta - theta), torch.cos(out_theta - theta))
    assert torch.allclose(angle_error, torch.zeros_like(angle_error), atol=3.0e-6)

    vertical = torch.tensor([_MODULE.FDU_VERTICAL_JOINT_CENTER])
    _, vertical_theta, _, _ = compute_fdu_equivalent_leg_state(vertical, vertical, -vertical, -vertical)
    assert torch.allclose(vertical_theta, torch.zeros_like(vertical_theta), atol=3.0e-6)


def test_inverse_respects_passive_joint_limits_not_only_triangle_inequality():
    lengths = torch.tensor([
        _MODULE.FDU_MECHANICAL_L0_MIN - 1.0e-3,
        _MODULE.FDU_MECHANICAL_L0_MIN + 1.0e-3,
        _MODULE.FDU_MECHANICAL_L0_MAX - 1.0e-3,
        _MODULE.FDU_MECHANICAL_L0_MAX + 1.0e-3,
        _MODULE.FDU_L1 + _MODULE.FDU_L2 - 1.0e-3,
    ])
    _, _, valid = inverse_equivalent_kite(lengths, torch.zeros_like(lengths))
    assert valid.tolist() == [False, True, True, False, False]
    assert _MODULE.FDU_MECHANICAL_L0_MAX == pytest.approx(0.3414853, abs=1.0e-6)


def test_buggy_airtime_accumulates_on_ground_and_clears_in_flight():
    base_air_time = torch.tensor([0.0, 0.2])
    in_flight = torch.tensor([False, True])
    root_z = torch.tensor([0.4, 0.4])
    root_vz = torch.tensor([1.0, 1.0])

    reward, updated = update_buggy_fudan_airtime(
        base_air_time, in_flight, root_z, root_vz, step_dt=0.01
    )

    assert torch.allclose(updated, torch.tensor([0.004, 0.0]))
    assert torch.allclose(reward[0], torch.tensor(0.15))
    assert torch.allclose(reward[1], torch.tensor(0.15))
