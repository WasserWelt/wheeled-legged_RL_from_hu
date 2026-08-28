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
