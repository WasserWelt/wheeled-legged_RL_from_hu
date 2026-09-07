"""Pure-Torch contracts for Fudan's encoder-derived joint velocity."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch


ROOT = Path(__file__).parents[2]
MODULE_PATH = (
    ROOT
    / "source/agent_world/agent_world/actuators/finite_difference_velocity.py"
)
SPEC = importlib.util.spec_from_file_location("fdu_finite_difference_velocity", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
FiniteDifferenceJointVelocity = MODULE.FiniteDifferenceJointVelocity


def test_first_sample_is_zero_then_uses_physics_rate_difference():
    estimator = FiniteDifferenceJointVelocity(0.002)
    initial = torch.tensor([[0.1, -0.2, 0.3, -0.4, 0.5, -0.6]])
    assert torch.equal(estimator.update(initial), torch.zeros_like(initial))

    moved = initial + torch.tensor([[0.002, -0.004, 0.006, -0.008, 0.010, -0.012]])
    torch.testing.assert_close(
        estimator.update(moved),
        torch.tensor([[1.0, -2.0, 3.0, -4.0, 5.0, -6.0]]),
    )


def test_wheel_position_wrap_does_not_create_a_2pi_velocity_spike():
    estimator = FiniteDifferenceJointVelocity(0.002)
    initial = torch.tensor([[math.pi - 0.001]])
    estimator.reset(slice(None), initial)
    crossed = torch.tensor([[-math.pi + 0.003]])
    torch.testing.assert_close(estimator.update(crossed), torch.tensor([[2.0]]), atol=1e-4, rtol=0.0)


def test_partial_environment_reset_is_zero_without_disturbing_other_histories():
    estimator = FiniteDifferenceJointVelocity(0.002)
    initial = torch.zeros(2, 2)
    estimator.update(initial, sample_id=0)
    estimator.update(torch.tensor([[0.002, 0.004], [0.006, 0.008]]), sample_id=1)

    reset_position = torch.tensor([[0.002, 0.004], [1.0, -1.0]])
    estimator.reset(torch.tensor([1]), reset_position)
    torch.testing.assert_close(estimator.velocity[0], torch.tensor([1.0, 2.0]))
    assert torch.equal(estimator.velocity[1], torch.zeros(2))
    assert estimator.update(reset_position, sample_id=1) is estimator.velocity
    torch.testing.assert_close(estimator.velocity[0], torch.tensor([1.0, 2.0]))

    next_position = reset_position + torch.tensor([[0.002, 0.002], [0.004, -0.006]])
    torch.testing.assert_close(
        estimator.update(next_position, sample_id=2),
        torch.tensor([[1.0, 1.0], [2.0, -3.0]]),
        atol=1e-4,
        rtol=0.0,
    )


def test_repeated_simulation_sample_id_is_not_differenced_twice():
    estimator = FiniteDifferenceJointVelocity(0.002)
    estimator.update(torch.zeros(1, 1), sample_id=10)
    first = estimator.update(torch.tensor([[0.004]]), sample_id=11).clone()
    repeated = estimator.update(torch.tensor([[0.008]]), sample_id=11).clone()
    torch.testing.assert_close(first, torch.tensor([[2.0]]))
    torch.testing.assert_close(repeated, first)
    torch.testing.assert_close(estimator.previous_position, torch.tensor([[0.004]]))
