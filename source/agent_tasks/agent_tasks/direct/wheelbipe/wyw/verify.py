"""Pure-Python scenario and result contracts for WYW policy verification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from statistics import median
from typing import Any


VERIFY_SCHEMA_VERSION = 2
STANDARD_SEED = 42
NOMINAL_NUM_ENVS = 1
ROBUST_NUM_ENVS = 10


@dataclass(frozen=True)
class VerifyScenario:
    id: str
    variant: str
    profile: str
    vx: float = 0.0
    yaw: float = 0.0
    height: float = 0.22
    settle_s: float = 0.5
    score_s: float = 3.0
    terrain: str | None = None
    difficulty: float | None = None
    step_height_m: float | None = None
    direction: str | None = None
    approach_deg: float = 0.0
    target_distance_m: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _number(value: float) -> str:
    sign = "m" if value < 0 else "p"
    return f"{sign}{abs(value):g}".replace(".", "p")


def _flat_scenarios(profile: str) -> list[VerifyScenario]:
    if profile == "robust":
        return [
            VerifyScenario("flat_robust_stand_h022", "flat", profile),
            VerifyScenario("flat_robust_vx_p2", "flat", profile, vx=2.0),
            VerifyScenario("flat_robust_yaw_p2", "flat", profile, yaw=2.0),
        ]

    scenarios = [
        VerifyScenario(f"flat_stand_h{int(height * 100):03d}", "flat", profile, height=height)
        for height in (0.15, 0.22, 0.30)
    ]
    scenarios += [
        VerifyScenario(f"flat_vx_{_number(vx)}", "flat", profile, vx=vx)
        for vx in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)
    ]
    scenarios += [
        VerifyScenario(f"flat_yaw_{_number(yaw)}", "flat", profile, yaw=yaw)
        for yaw in (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0)
    ]
    scenarios += [
        VerifyScenario(f"flat_curve_vx_p1_yaw_{_number(yaw)}", "flat", profile, vx=1.0, yaw=yaw)
        for yaw in (-1.0, 1.0)
    ]
    return scenarios


def _rough_scenario(
    scenario_id: str,
    profile: str,
    terrain: str,
    *,
    vx: float = 1.5,
    difficulty: float | None = 0.5,
    step_height_m: float | None = None,
    direction: str,
    approach_deg: float = 0.0,
) -> VerifyScenario:
    return VerifyScenario(
        id=scenario_id,
        variant="rough",
        profile=profile,
        vx=vx,
        settle_s=0.5,
        score_s=6.0 if math.isclose(vx, 0.5) else 4.0,
        terrain=terrain,
        difficulty=difficulty,
        step_height_m=step_height_m,
        direction=direction,
        approach_deg=approach_deg,
        target_distance_m=2.5,
    )


def _rough_scenarios(profile: str) -> list[VerifyScenario]:
    if profile == "robust":
        return [
            _rough_scenario(
                "rough_robust_rough_slope_up_d08_v15",
                profile,
                "rough_slope_up",
                difficulty=0.8,
                direction="up",
            ),
            _rough_scenario(
                "rough_robust_stairs_up_d08_v15_a30",
                profile,
                "stairs_up",
                difficulty=0.8,
                direction="up",
                approach_deg=30.0,
            ),
            _rough_scenario(
                "rough_robust_single_step_up_h20_v15",
                profile,
                "single_step_20cm",
                difficulty=None,
                step_height_m=0.20,
                direction="up",
            ),
            _rough_scenario(
                "rough_robust_single_step_down_h20_v15",
                profile,
                "single_step_20cm",
                difficulty=None,
                step_height_m=0.20,
                direction="down",
            ),
        ]

    scenarios: list[VerifyScenario] = []
    representative = (
        ("smooth_slope_up", "up"),
        ("smooth_slope_down", "down"),
        ("rough_slope_up", "up"),
        ("rough_slope_down", "down"),
        ("discrete_obstacles", "forward"),
    )
    for terrain, direction in representative:
        scenarios.append(
            _rough_scenario(
                f"rough_{terrain}_d05_v15", profile, terrain, direction=direction
            )
        )

    for terrain, direction in (("stairs_up", "up"), ("stairs_down", "down")):
        for difficulty in (0.2, 0.5, 0.8):
            scenarios.append(
                _rough_scenario(
                    f"rough_{terrain}_d{int(difficulty * 10):02d}_v15",
                    profile,
                    terrain,
                    difficulty=difficulty,
                    direction=direction,
                )
            )
        for vx in (0.5, 2.5):
            scenarios.append(
                _rough_scenario(
                    f"rough_{terrain}_d05_v{int(vx * 10):02d}",
                    profile,
                    terrain,
                    vx=vx,
                    direction=direction,
                )
            )
    for angle in (-30.0, -15.0, 15.0, 30.0):
        scenarios.append(
            _rough_scenario(
                f"rough_stairs_up_d05_v15_a{_number(angle)}",
                profile,
                "stairs_up",
                direction="up",
                approach_deg=angle,
            )
        )

    for height in (0.05, 0.10, 0.15, 0.20):
        terrain = f"single_step_{int(height * 100):02d}cm"
        for direction in ("up", "down"):
            scenarios.append(
                _rough_scenario(
                    f"rough_{terrain}_{direction}_v15",
                    profile,
                    terrain,
                    difficulty=None,
                    step_height_m=height,
                    direction=direction,
                )
            )
    for direction in ("up", "down"):
        for vx in (0.5, 2.5):
            scenarios.append(
                _rough_scenario(
                    f"rough_single_step_15cm_{direction}_v{int(vx * 10):02d}",
                    profile,
                    "single_step_15cm",
                    vx=vx,
                    difficulty=None,
                    step_height_m=0.15,
                    direction=direction,
                )
            )
    for angle in (-30.0, -15.0, 15.0, 30.0):
        scenarios.append(
            _rough_scenario(
                f"rough_single_step_15cm_up_v15_a{_number(angle)}",
                profile,
                "single_step_15cm",
                difficulty=None,
                step_height_m=0.15,
                direction="up",
                approach_deg=angle,
            )
        )
    return scenarios


def _jump_scenarios(profile: str) -> list[VerifyScenario]:
    speeds = (0.0, 2.5) if profile == "robust" else (0.0, 1.5, 2.5)
    return [
        VerifyScenario(
            id=f"jump_{profile}_vx_{_number(vx)}",
            variant="jump",
            profile=profile,
            vx=vx,
            settle_s=1.0,
            score_s=8.0,
        )
        for vx in speeds
    ]


def build_scenarios(variant: str, profile: str) -> list[VerifyScenario]:
    """Build the versioned standard scenario list."""
    if profile not in {"nominal", "robust"}:
        raise ValueError(f"unsupported profile: {profile}")
    builders = {"flat": _flat_scenarios, "rough": _rough_scenarios, "jump": _jump_scenarios}
    try:
        scenarios = builders[variant](profile)
    except KeyError as exc:
        raise ValueError(f"unsupported variant: {variant}") from exc
    ids = [scenario.id for scenario in scenarios]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"duplicate verification scenario ids for {variant}/{profile}")
    return scenarios


def standard_num_envs(profile: str) -> int:
    if profile == "nominal":
        return NOMINAL_NUM_ENVS
    if profile == "robust":
        return ROBUST_NUM_ENVS
    raise ValueError(f"unsupported profile: {profile}")


def validate_profile_report(report: dict[str, Any]) -> None:
    """Validate the serialized per-profile report and scenario coverage."""
    required = {
        "schema_version",
        "variant",
        "profile",
        "task_id",
        "seed",
        "num_envs",
        "simulated_duration_s",
        "checkpoint",
        "metadata",
        "scenario_summaries",
        "samples",
        "video",
    }
    missing = required - set(report)
    if missing:
        raise ValueError(f"profile report missing fields: {sorted(missing)}")
    if report["schema_version"] != VERIFY_SCHEMA_VERSION:
        raise ValueError("profile report schema version mismatch")
    variant, profile = report["variant"], report["profile"]
    scenarios = build_scenarios(variant, profile)
    scenario_ids = [scenario.id for scenario in scenarios]
    summary_ids = [summary.get("scenario_id") for summary in report["scenario_summaries"]]
    if summary_ids != scenario_ids:
        raise ValueError("profile report scenario summary coverage/order mismatch")
    expected_samples = len(scenarios) * standard_num_envs(profile)
    if report["num_envs"] != standard_num_envs(profile):
        raise ValueError("profile report environment count mismatch")
    if len(report["samples"]) != expected_samples:
        raise ValueError("profile report sample count mismatch")
    sample_counts = {
        scenario_id: sum(
            sample.get("scenario_id") == scenario_id for sample in report["samples"]
        )
        for scenario_id in scenario_ids
    }
    if any(count != standard_num_envs(profile) for count in sample_counts.values()):
        raise ValueError("profile report sample scenario coverage mismatch")
    expected_duration = sum(scenario.settle_s + scenario.score_s for scenario in scenarios)
    if not math.isclose(float(report["simulated_duration_s"]), expected_duration):
        raise ValueError("profile report simulated duration mismatch")


def validate_final_report(report: dict[str, Any]) -> None:
    """Validate the top-level report before it is committed to disk."""
    required = {
        "schema_version",
        "status",
        "mode",
        "variant",
        "profiles",
        "standard",
        "seed",
        "simulated_duration_s",
        "checkpoint",
        "git",
        "video",
        "chart",
        "evaluation",
        "aggregate_comparison",
        "runs",
    }
    missing = required - set(report)
    if missing:
        raise ValueError(f"final report missing fields: {sorted(missing)}")
    if report["schema_version"] != VERIFY_SCHEMA_VERSION:
        raise ValueError("final report schema version mismatch")
    if report["mode"] not in {"baseline", "calibrate", "evaluate"}:
        raise ValueError("final report mode is invalid")
    if report["status"] not in {"BASELINE", "PASS", "FAIL"}:
        raise ValueError("final report status is invalid")
    if report["mode"] == "evaluate":
        if not isinstance(report["chart"], str) or not report["chart"]:
            raise ValueError("evaluate report must reference a comparison chart")
        aggregate = report["aggregate_comparison"]
        if not isinstance(aggregate, dict) or aggregate.get("method") != "median_of_scenario_medians":
            raise ValueError("evaluate report aggregate comparison is invalid")
    run_profiles = [run.get("profile") for run in report["runs"]]
    if run_profiles != report["profiles"]:
        raise ValueError("final report profile coverage/order mismatch")
    for run in report["runs"]:
        if run.get("variant") != report["variant"]:
            raise ValueError("final report run variant mismatch")
        validate_profile_report(run)
    run_duration = sum(float(run["simulated_duration_s"]) for run in report["runs"])
    if not math.isclose(float(report["simulated_duration_s"]), run_duration):
        raise ValueError("final report simulated duration mismatch")


LOWER_IS_BETTER = {
    "vx_rmse_m_s",
    "yaw_rmse_rad_s",
    "tilt_rms_deg",
    "tilt_peak_deg",
    "height_rmse_m",
}
HIGHER_IS_BETTER = {
    "survival_rate",
    "completion_rate",
    "jump_count",
    "jump_height_gain_m",
    "airtime_s",
    "stable_landing_rate",
}

METRIC_PRESENTATION = {
    "survival_rate": ("Survival", "%"),
    "vx_rmse_m_s": ("VX RMSE", "m/s"),
    "yaw_rmse_rad_s": ("Yaw RMSE", "rad/s"),
    "tilt_rms_deg": ("Tilt RMS", "deg"),
    "tilt_peak_deg": ("Tilt Peak", "deg"),
    "height_rmse_m": ("Height RMSE", "m"),
    "completion_rate": ("Completion", "%"),
    "jump_count": ("Jump Count", "count"),
    "jump_height_gain_m": ("Jump Height", "m"),
    "airtime_s": ("Airtime", "s"),
    "stable_landing_rate": ("Stable Landing", "%"),
}
METRIC_PRESENTATION_ORDER = (
    "survival_rate",
    "vx_rmse_m_s",
    "yaw_rmse_rad_s",
    "tilt_rms_deg",
    "tilt_peak_deg",
    "height_rmse_m",
    "completion_rate",
    "jump_count",
    "jump_height_gain_m",
    "airtime_s",
    "stable_landing_rate",
)


def required_metrics(variant: str) -> set[str]:
    common = {
        "survival_rate",
        "vx_rmse_m_s",
        "yaw_rmse_rad_s",
        "tilt_rms_deg",
        "tilt_peak_deg",
        "height_rmse_m",
    }
    if variant == "rough":
        return common | {"completion_rate"}
    if variant == "jump":
        return common | {
            "jump_count",
            "jump_height_gain_m",
            "airtime_s",
            "stable_landing_rate",
        }
    if variant == "flat":
        return common
    raise ValueError(f"unsupported variant: {variant}")


def aggregate_comparison(
    *,
    profile_summaries: dict[str, list[dict[str, Any]]],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate scenario medians for a compact baseline/current comparison.

    Per-environment samples and scenario min/max values are deliberately ignored.
    Each scenario contributes exactly one median, including robust scenarios.
    """
    validate_baseline_package(baseline)
    expected_profiles = ["nominal", "robust"]
    if list(profile_summaries) != expected_profiles:
        raise ValueError("aggregate comparison requires nominal then robust summaries")

    baseline_by_id = {
        summary["scenario_id"]: (profile, summary)
        for profile, entries in baseline["profiles"].items()
        for summary in entries["scenario_summaries"]
    }
    names = required_metrics(baseline["variant"])
    baseline_values = {name: [] for name in names}
    current_values = {name: [] for name in names}
    scenario_count = 0
    for profile in expected_profiles:
        expected_ids = [scenario.id for scenario in build_scenarios(baseline["variant"], profile)]
        summaries = profile_summaries.get(profile, [])
        actual_ids = [summary.get("scenario_id") for summary in summaries]
        if actual_ids != expected_ids:
            raise ValueError(f"aggregate scenario coverage/order mismatch for {profile}")
        for summary in summaries:
            scenario_id = summary["scenario_id"]
            baseline_profile, baseline_summary = baseline_by_id[scenario_id]
            if baseline_profile != profile:
                raise ValueError(f"baseline profile mismatch for scenario: {scenario_id}")
            for name in names:
                current_value = summary.get("metrics", {}).get(name, {}).get("median")
                baseline_value = baseline_summary.get("metrics", {}).get(name, {}).get("median")
                if current_value is None or baseline_value is None:
                    raise ValueError(f"missing aggregate metric {name} for {scenario_id}")
                current_value = float(current_value)
                baseline_value = float(baseline_value)
                if not math.isfinite(current_value) or not math.isfinite(baseline_value):
                    raise ValueError(f"non-finite aggregate metric {name} for {scenario_id}")
                current_values[name].append(current_value)
                baseline_values[name].append(baseline_value)
            scenario_count += 1

    metrics = []
    for name in (item for item in METRIC_PRESENTATION_ORDER if item in names):
        baseline_median = float(median(baseline_values[name]))
        current_median = float(median(current_values[name]))
        delta = current_median - baseline_median
        direction = "lower" if name in LOWER_IS_BETTER else "higher"
        improvement = -delta if direction == "lower" else delta
        improvement_percent = None
        if not math.isclose(baseline_median, 0.0):
            improvement_percent = 100.0 * improvement / abs(baseline_median)
        metrics.append(
            {
                "name": name,
                "label": METRIC_PRESENTATION[name][0],
                "unit": METRIC_PRESENTATION[name][1],
                "direction": direction,
                "baseline": baseline_median,
                "current": current_median,
                "delta": delta,
                "improvement": improvement,
                "improvement_percent": improvement_percent,
                "pass": improvement >= 0.0,
                "scenario_count": scenario_count,
            }
        )
    return {
        "method": "median_of_scenario_medians",
        "scenario_count": scenario_count,
        "profiles": expected_profiles,
        "metrics": metrics,
    }


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate finite numeric per-env metrics while retaining safety counts."""
    if not samples:
        raise ValueError("cannot summarize an empty sample list")
    metric_names = sorted(
        {
            key
            for sample in samples
            for key, value in sample.items()
            if key not in {"scenario_id", "env_id", "failure_reason"}
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        }
    )
    metrics: dict[str, dict[str, float]] = {}
    for name in metric_names:
        values = [float(sample[name]) for sample in samples if name in sample and math.isfinite(float(sample[name]))]
        if values:
            metrics[name] = {
                "min": min(values),
                "median": float(median(values)),
                "max": max(values),
            }
    reasons: dict[str, int] = {}
    for sample in samples:
        reason = sample.get("failure_reason")
        if reason:
            reasons[str(reason)] = reasons.get(str(reason), 0) + 1
    return {
        "num_samples": len(samples),
        "finite": all(bool(sample.get("finite", False)) for sample in samples),
        "failure_reasons": reasons,
        "metrics": metrics,
    }


def make_baseline_package(
    *,
    variant: str,
    profile_summaries: dict[str, list[dict[str, Any]]],
    checkpoint: str,
    video: str,
) -> dict[str, Any]:
    """Persist baseline medians and scenario order as the comparison contract."""
    profiles: dict[str, Any] = {}
    for profile, summaries in profile_summaries.items():
        expected_ids = [scenario.id for scenario in build_scenarios(variant, profile)]
        actual_ids = [summary.get("scenario_id") for summary in summaries]
        if actual_ids != expected_ids:
            raise ValueError(f"baseline scenario coverage/order mismatch for {profile}")
        for summary in summaries:
            missing = required_metrics(variant) - set(summary.get("metrics", {}))
            if missing:
                raise ValueError(
                    f"scenario {summary['scenario_id']} is missing calibration metrics: {sorted(missing)}"
                )
        profiles[profile] = {"scenario_summaries": summaries}
    return {
        "schema_version": VERIFY_SCHEMA_VERSION,
        "type": "wyw_baseline",
        "status": "frozen",
        "variant": variant,
        "checkpoint": checkpoint,
        "video": video,
        "profiles": profiles,
    }


def validate_baseline_package(baseline: dict[str, Any]) -> None:
    required = {"schema_version", "type", "status", "variant", "checkpoint", "video", "profiles"}
    missing = required - set(baseline)
    if missing:
        raise ValueError(f"baseline package missing fields: {sorted(missing)}")
    if baseline["schema_version"] != VERIFY_SCHEMA_VERSION:
        raise ValueError("baseline schema version mismatch")
    if baseline["type"] != "wyw_baseline" or baseline["status"] != "frozen":
        raise ValueError("baseline package must be frozen")
    for profile in ("nominal", "robust"):
        entries = baseline["profiles"].get(profile)
        if not isinstance(entries, dict):
            raise ValueError(f"baseline missing profile: {profile}")
        summaries = entries.get("scenario_summaries")
        if not isinstance(summaries, list):
            raise ValueError(f"baseline missing summaries: {profile}")
        expected_ids = [scenario.id for scenario in build_scenarios(baseline["variant"], profile)]
        actual_ids = [summary.get("scenario_id") for summary in summaries]
        if actual_ids != expected_ids:
            raise ValueError(f"baseline scenario coverage/order mismatch for {profile}")
        for summary in summaries:
            missing = required_metrics(baseline["variant"]) - set(summary.get("metrics", {}))
            if missing:
                raise ValueError(f"baseline missing metrics for {summary.get('scenario_id')}: {sorted(missing)}")


def compare_summaries(
    *,
    profile_summaries: dict[str, list[dict[str, Any]]],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Apply strict safety gates and compare performance against the baseline."""
    validate_baseline_package(baseline)
    if baseline["variant"] not in {"flat", "rough", "jump"}:
        raise ValueError("baseline variant is invalid")
    baseline_by_id = {
        summary["scenario_id"]: (profile, summary)
        for profile, entries in baseline["profiles"].items()
        for summary in entries["scenario_summaries"]
    }
    profile_results: dict[str, Any] = {}
    safety_pass = True
    for profile, summaries in profile_summaries.items():
        passed = 0
        scenario_results = []
        for summary in summaries:
            scenario_id = summary["scenario_id"]
            safety_ok = bool(summary["finite"]) and not summary["failure_reasons"]
            safety_pass &= safety_ok
            baseline_entry = baseline_by_id.get(scenario_id)
            if baseline_entry is None:
                raise ValueError(f"baseline missing scenario: {scenario_id}")
            baseline_profile, baseline_summary = baseline_entry
            if baseline_profile != profile:
                raise ValueError(f"baseline profile mismatch for scenario: {scenario_id}")
            required = required_metrics(baseline["variant"])
            configured = set(baseline_summary.get("metrics", {}))
            missing = required - configured
            if missing:
                raise ValueError(f"baseline missing metrics for {scenario_id}: {sorted(missing)}")
            metric_results = {}
            performance_ok = True
            for name in sorted(required):
                actual = summary.get("metrics", {}).get(name, {}).get("median")
                baseline_value = baseline_summary["metrics"][name]["median"]
                if name in LOWER_IS_BETTER:
                    ok = actual is not None and actual <= float(baseline_value)
                elif name in HIGHER_IS_BETTER:
                    ok = actual is not None and actual >= float(baseline_value)
                else:
                    raise ValueError(f"unsupported baseline metric: {name}")
                performance_ok &= ok
                metric_results[name] = {
                    "baseline": baseline_value,
                    "current": actual,
                    "delta": None if actual is None else float(actual) - float(baseline_value),
                    "pass": ok,
                }
            scenario_ok = safety_ok and performance_ok
            passed += int(scenario_ok)
            scenario_results.append(
                {
                    "scenario_id": scenario_id,
                    "safety_pass": safety_ok,
                    "performance_pass": performance_ok,
                    "pass": scenario_ok,
                    "metrics": metric_results,
                }
            )
        required_count = math.ceil(0.9 * len(summaries))
        profile_results[profile] = {
            "passed": passed,
            "total": len(summaries),
            "required": required_count,
            "pass": passed >= required_count,
            "scenarios": scenario_results,
        }
    performance_pass = all(result["pass"] for result in profile_results.values())
    return {
        "safety_pass": safety_pass,
        "performance_pass": performance_pass,
        "pass": safety_pass and performance_pass,
        "profiles": profile_results,
    }
