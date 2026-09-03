"""CPU contracts for the standardized WYW verification play."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[2]
VERIFY_PATH = ROOT / "source/agent_tasks/agent_tasks/direct/wheelbipe/wyw/verify.py"


def _load_verify():
    spec = importlib.util.spec_from_file_location("wyw_verify_contract", VERIFY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


V = _load_verify()


def test_standard_scenario_counts_and_runtime_budget():
    expected = {
        ("flat", "nominal"): 17,
        ("flat", "robust"): 3,
        ("rough", "nominal"): 35,
        ("rough", "robust"): 4,
        ("jump", "nominal"): 3,
        ("jump", "robust"): 2,
    }
    total_seconds = 0.0
    for key, count in expected.items():
        scenarios = V.build_scenarios(*key)
        assert len(scenarios) == count
        assert len({scenario.id for scenario in scenarios}) == count
        total_seconds += sum(scenario.settle_s + scenario.score_s for scenario in scenarios)
    assert total_seconds == pytest.approx(298.5)


def test_rough_has_no_flat_and_keeps_dense_stair_coverage():
    nominal = V.build_scenarios("rough", "nominal")
    assert all(scenario.terrain != "flat" for scenario in nominal)
    single_steps = [scenario for scenario in nominal if scenario.terrain.startswith("single_step_")]
    assert len(single_steps) == 16
    assert {scenario.step_height_m for scenario in single_steps} == {0.05, 0.10, 0.15, 0.20}
    assert {scenario.direction for scenario in single_steps} == {"up", "down"}
    assert {scenario.approach_deg for scenario in single_steps if scenario.approach_deg} == {
        -30.0,
        -15.0,
        15.0,
        30.0,
    }
    pyramid = [scenario for scenario in nominal if scenario.terrain in {"stairs_up", "stairs_down"}]
    assert len(pyramid) == 14
    assert all(scenario.target_distance_m == 2.5 for scenario in nominal)
    assert all(scenario.score_s == 6.0 for scenario in nominal if scenario.vx == 0.5)


def test_manifest_is_order_sensitive_and_stable():
    scenarios = V.build_scenarios("flat", "nominal")
    first = V.manifest_hash(scenarios)
    assert first == V.manifest_hash(V.build_scenarios("flat", "nominal"))
    assert first != V.manifest_hash(list(reversed(scenarios)))


def _summary(scenario_id: str, value: float, *, failure: str | None = None):
    return {
        "scenario_id": scenario_id,
        "finite": True,
        "failure_reasons": {} if failure is None else {failure: 1},
        "metrics": {
            "vx_rmse_m_s": {"min": value, "median": value, "max": value},
            "yaw_rmse_rad_s": {"min": value, "median": value, "max": value},
            "tilt_rms_deg": {"min": value, "median": value, "max": value},
            "tilt_peak_deg": {"min": value, "median": value, "max": value},
            "height_rmse_m": {"min": value, "median": value, "max": value},
            "survival_rate": {"min": 1.0, "median": 1.0, "max": 1.0},
        },
    }


def test_candidate_uses_median_without_margin_and_requires_freeze():
    summaries = {"nominal": [_summary("s0", 0.2)]}
    candidate = V.make_candidate_thresholds(
        variant="flat",
        profile_summaries=summaries,
        manifest_sha256="manifest",
        checkpoint_sha256="checkpoint",
    )
    limit = candidate["scenario_limits"]["s0"]["metrics"]["vx_rmse_m_s"]
    assert limit == {"direction": "max", "value": 0.2}
    with pytest.raises(ValueError, match="frozen"):
        V.evaluate_summaries(profile_summaries=summaries, thresholds=candidate)
    candidate["status"] = "frozen"
    assert V.evaluate_summaries(profile_summaries=summaries, thresholds=candidate)["pass"]
    summaries["nominal"][0]["metrics"]["vx_rmse_m_s"]["median"] = 0.20001
    assert not V.evaluate_summaries(profile_summaries=summaries, thresholds=candidate)["pass"]


def test_ninety_percent_gate_and_safety_are_both_required():
    summaries = {"robust": [_summary(f"s{i}", 0.1) for i in range(10)]}
    candidate = V.make_candidate_thresholds(
        variant="flat",
        profile_summaries=summaries,
        manifest_sha256="manifest",
        checkpoint_sha256="checkpoint",
    )
    candidate["status"] = "frozen"
    summaries["robust"][0]["metrics"]["vx_rmse_m_s"]["median"] = 0.2
    result = V.evaluate_summaries(profile_summaries=summaries, thresholds=candidate)
    assert result["pass"]
    assert result["profiles"]["robust"]["required"] == 9
    summaries["robust"][1]["metrics"]["vx_rmse_m_s"]["median"] = 0.2
    assert not V.evaluate_summaries(profile_summaries=summaries, thresholds=candidate)["pass"]
    summaries["robust"][1]["metrics"]["vx_rmse_m_s"]["median"] = 0.1
    summaries["robust"][2]["failure_reasons"] = {"contact": 1}
    result = V.evaluate_summaries(profile_summaries=summaries, thresholds=candidate)
    assert not result["safety_pass"]
    assert not result["pass"]


def test_frozen_thresholds_require_complete_metrics_and_matching_profile():
    summaries = {"nominal": [_summary("s0", 0.1)]}
    frozen = V.make_candidate_thresholds(
        variant="flat",
        profile_summaries=summaries,
        manifest_sha256="manifest",
        checkpoint_sha256="checkpoint",
    )
    frozen["status"] = "frozen"
    frozen["scenario_limits"]["s0"]["profile"] = "robust"
    with pytest.raises(ValueError, match="profile mismatch"):
        V.evaluate_summaries(profile_summaries=summaries, thresholds=frozen)

    frozen["scenario_limits"]["s0"]["profile"] = "nominal"
    del frozen["scenario_limits"]["s0"]["metrics"]["height_rmse_m"]
    with pytest.raises(ValueError, match="missing metrics"):
        V.evaluate_summaries(profile_summaries=summaries, thresholds=frozen)


def test_profile_and_final_report_format_validation():
    scenario = V.build_scenarios("flat", "robust")[0]
    summaries = [
        {"scenario_id": item.id, "finite": True, "failure_reasons": {}, "metrics": {}}
        for item in V.build_scenarios("flat", "robust")
    ]
    samples = [
        {"scenario_id": item.id, "env_id": env_id}
        for item in V.build_scenarios("flat", "robust")
        for env_id in range(V.ROBUST_NUM_ENVS)
    ]
    profile_report = {
        "schema_version": V.VERIFY_SCHEMA_VERSION,
        "variant": "flat",
        "profile": "robust",
        "task_id": "task",
        "seed": V.STANDARD_SEED,
        "num_envs": V.ROBUST_NUM_ENVS,
        "simulated_duration_s": 10.5,
        "checkpoint": "model.pt",
        "checkpoint_sha256": "hash",
        "metadata": {},
        "manifest_sha256": "manifest",
        "scenario_summaries": summaries,
        "samples": samples,
        "video": "flat_robust.mp4",
    }
    V.validate_profile_report(profile_report)
    final_report = {
        "schema_version": V.VERIFY_SCHEMA_VERSION,
        "status": "CALIBRATION",
        "mode": "calibrate",
        "variant": "flat",
        "profiles": ["robust"],
        "standard": True,
        "seed": V.STANDARD_SEED,
        "simulated_duration_s": 10.5,
        "manifest_sha256": "manifest",
        "checkpoint": "model.pt",
        "checkpoint_sha256": "hash",
        "git": {},
        "video": "flat_verify.mp4",
        "evaluation": None,
        "runs": [profile_report],
    }
    V.validate_final_report(final_report)
    profile_report["scenario_summaries"][0]["scenario_id"] = scenario.id + "_wrong"
    with pytest.raises(ValueError, match="coverage/order"):
        V.validate_profile_report(profile_report)


def test_single_step_is_verification_only_and_has_four_heights():
    rough_path = ROOT / "source/agent_tasks/agent_tasks/direct/wheelbipe/wyw/rough_cfg.py"
    source = rough_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    training_assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "FDU_ROUGH_TERRAIN_CFG" for target in node.targets)
    )
    training_source = ast.get_source_segment(source, training_assignment)
    assert training_source is not None and "single_step" not in training_source
    assert "FDU_ROUGH_VERIFY_TERRAIN_CFG" in source
    assert "for _height_cm in (5, 10, 15, 20)" in source
    assert "heights[width_pixels // 2 :, :] = height_units" in source


def test_environment_exposes_reset_safe_command_override():
    env_path = ROOT / "source/agent_tasks/agent_tasks/direct/wheelbipe/wyw/env.py"
    tree = ast.parse(env_path.read_text(encoding="utf-8"))
    methods = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "set_wyw_command_override" in methods
    assert "clear_wyw_command_override" in methods
    reset_calls = [
        node
        for node in ast.walk(methods["_reset_idx"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_apply_wyw_command_override"
    ]
    assert reset_calls
