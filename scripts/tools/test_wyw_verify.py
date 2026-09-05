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
            "leg_action_delta_rms": {"min": value, "median": value, "max": value},
            "wheel_action_delta_rms": {"min": value, "median": value, "max": value},
            "leg_action_second_diff_rms": {"min": value, "median": value, "max": value},
            "wheel_action_second_diff_rms": {"min": value, "median": value, "max": value},
            "survival_rate": {"min": 1.0, "median": 1.0, "max": 1.0},
        },
    }


def _all_flat_summaries(value: float = 0.1):
    return {
        profile: [_summary(scenario.id, value) for scenario in V.build_scenarios("flat", profile)]
        for profile in ("nominal", "robust")
    }


def test_baseline_package_is_frozen_and_ordered():
    baseline = V.make_baseline_package(
        variant="flat",
        profile_summaries=_all_flat_summaries(),
        checkpoint="logs/baseline.pt",
        video="baseline.mp4",
    )
    V.validate_baseline_package(baseline)
    assert baseline["status"] == "frozen"
    assert "manifest_sha256" not in baseline
    assert "baseline_checkpoint_sha256" not in baseline
    baseline["profiles"]["nominal"]["scenario_summaries"].reverse()
    with pytest.raises(ValueError, match="coverage/order"):
        V.validate_baseline_package(baseline)


def test_baseline_comparison_uses_deltas_and_90_percent_gate():
    baseline = V.make_baseline_package(
        variant="flat",
        profile_summaries=_all_flat_summaries(0.1),
        checkpoint="logs/baseline.pt",
        video="baseline.mp4",
    )
    current = _all_flat_summaries(0.1)
    current["robust"][0]["metrics"]["vx_rmse_m_s"]["median"] = 0.2
    result = V.compare_summaries(profile_summaries=current, baseline=baseline)
    assert not result["pass"]
    assert result["profiles"]["robust"]["required"] == 3
    failed_metric = result["profiles"]["robust"]["scenarios"][0]["metrics"]["vx_rmse_m_s"]
    assert failed_metric["delta"] == pytest.approx(0.1)
    current["robust"][0]["metrics"]["vx_rmse_m_s"]["median"] = 0.1
    assert V.compare_summaries(profile_summaries=current, baseline=baseline)["pass"]
    current["robust"][1]["metrics"]["vx_rmse_m_s"]["median"] = 0.2
    assert not V.compare_summaries(profile_summaries=current, baseline=baseline)["pass"]
    current["robust"][1]["metrics"]["vx_rmse_m_s"]["median"] = 0.1
    current["robust"][2]["failure_reasons"] = {"contact": 1}
    result = V.compare_summaries(profile_summaries=current, baseline=baseline)
    assert not result["safety_pass"]
    assert not result["pass"]


def test_baseline_comparison_rejects_profile_mismatch_and_missing_metrics():
    baseline = V.make_baseline_package(
        variant="flat",
        profile_summaries=_all_flat_summaries(),
        checkpoint="logs/baseline.pt",
        video="baseline.mp4",
    )
    current = _all_flat_summaries()
    current["nominal"][0]["scenario_id"] = "wrong"
    with pytest.raises(ValueError, match="baseline missing scenario"):
        V.compare_summaries(profile_summaries=current, baseline=baseline)
    baseline = V.make_baseline_package(
        variant="flat",
        profile_summaries=_all_flat_summaries(),
        checkpoint="logs/baseline.pt",
        video="baseline.mp4",
    )
    del baseline["profiles"]["nominal"]["scenario_summaries"][0]["metrics"]["height_rmse_m"]
    with pytest.raises(ValueError, match="baseline missing metrics"):
        V.validate_baseline_package(baseline)


def test_action_smoothness_metrics_are_required_and_lower_is_better():
    smoothness = {
        "leg_action_delta_rms",
        "wheel_action_delta_rms",
        "leg_action_second_diff_rms",
        "wheel_action_second_diff_rms",
    }
    assert smoothness <= V.required_metrics("flat")
    assert smoothness <= V.required_metrics("rough")
    assert smoothness <= V.required_metrics("jump")
    assert smoothness <= V.LOWER_IS_BETTER

    baseline = V.make_baseline_package(
        variant="flat",
        profile_summaries=_all_flat_summaries(0.1),
        checkpoint="logs/baseline.pt",
        video="baseline.mp4",
    )
    current = _all_flat_summaries(0.1)
    current["nominal"][0]["metrics"]["wheel_action_second_diff_rms"]["median"] = 0.05
    result = V.compare_summaries(profile_summaries=current, baseline=baseline)
    metric = result["profiles"]["nominal"]["scenarios"][0]["metrics"][
        "wheel_action_second_diff_rms"
    ]
    assert metric["pass"]
    assert metric["delta"] == pytest.approx(-0.05)


def test_aggregate_uses_scenario_medians_without_expanding_robust_envs():
    baseline_summaries = _all_flat_summaries(0.1)
    current = _all_flat_summaries(0.1)
    all_summaries = current["nominal"] + current["robust"]
    for index, summary in enumerate(all_summaries):
        summary["metrics"]["vx_rmse_m_s"]["median"] = float(index + 1)
        summary["metrics"]["vx_rmse_m_s"]["min"] = -10000.0
        summary["metrics"]["vx_rmse_m_s"]["max"] = 10000.0
        summary["num_samples"] = 10 if summary in current["robust"] else 1
    baseline = V.make_baseline_package(
        variant="flat",
        profile_summaries=baseline_summaries,
        checkpoint="logs/baseline.pt",
        video="baseline.mp4",
    )
    aggregate = V.aggregate_comparison(profile_summaries=current, baseline=baseline)
    vx = next(metric for metric in aggregate["metrics"] if metric["name"] == "vx_rmse_m_s")
    assert aggregate["method"] == "median_of_scenario_medians"
    assert aggregate["scenario_count"] == 20
    assert vx["current"] == pytest.approx(10.5)
    assert vx["direction"] == "lower"
    assert not vx["pass"]
    survival = next(metric for metric in aggregate["metrics"] if metric["name"] == "survival_rate")
    assert survival["direction"] == "higher"
    assert survival["pass"]


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
        "metadata": {},
        "scenario_summaries": summaries,
        "samples": samples,
        "video": "flat_robust.mp4",
    }
    V.validate_profile_report(profile_report)
    final_report = {
        "schema_version": V.VERIFY_SCHEMA_VERSION,
        "status": "BASELINE",
        "mode": "baseline",
        "variant": "flat",
        "profiles": ["robust"],
        "standard": True,
        "seed": V.STANDARD_SEED,
        "simulated_duration_s": 10.5,
        "checkpoint": "model.pt",
        "git": {},
        "video": "flat_verify.mp4",
        "chart": None,
        "evaluation": None,
        "aggregate_comparison": None,
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
