"""CPU-only plots and synchronized video composition for WYW verification."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import textwrap

BASELINE = "#737373"
CURRENT = "#0284c7"
FAIL = "#b91c1c"
PASS = "#15803d"
CONTRACT_PATH = (Path(__file__).resolve().parents[2]
                 / "source/agent_tasks/agent_tasks/direct/wheelbipe/wyw/verify.py")
_spec = importlib.util.spec_from_file_location("_wyw_presentation_contract", CONTRACT_PATH)
V = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = V
_spec.loader.exec_module(V)


def pyplot():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/wyw_verify_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # The installed TTC exposes the family as JP while covering CJK glyphs.
    plt.rcParams["font.family"] = ["Noto Serif CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    return plt


def metric_label(name):
    return V.METRIC_PRESENTATION.get(name, (name.replace("_", " "), ""))


def save_figure(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    pyplot().close(fig)


def _valid(value):
    return value is not None and math.isfinite(float(value))


def _short_scenario(scenario_id, variant, profile):
    for prefix in (f"{variant}_{profile}_", f"{variant}_"):
        if scenario_id.startswith(prefix):
            return scenario_id[len(prefix):]
    return scenario_id


def _metric_groups(names):
    configured = (
        ("tracking", ("vx_rmse_m_s", "yaw_rmse_rad_s", "height_rmse_m")),
        ("stability", ("tilt_rms_deg", "tilt_peak_deg", "height_peak_to_peak_m",
                       "vertical_velocity_rms_m_s", "wheel_contact_loss_mean")),
        ("smoothness", ("leg_action_delta_rms", "wheel_action_delta_rms",
                        "leg_action_second_diff_rms", "wheel_action_second_diff_rms")),
        ("actuator", ("wheel_torque_saturation_rate",
                      "wheel_velocity_target_saturation_rate")),
        ("outcomes", ("survival_rate", "completion_rate", "jump_count",
                      "jump_height_gain_m", "airtime_s", "stable_landing_rate")),
    )
    available = set(names)
    return [(group, [name for name in members if name in available])
            for group, members in configured if any(name in available for name in members)]


def _regression_score(name, metric):
    baseline, current = metric.get("baseline"), metric.get("current")
    if not _valid(baseline) or not _valid(current):
        return math.inf
    direction = 1.0 if name in V.LOWER_IS_BETTER else -1.0
    return direction * (float(current) - float(baseline)) / max(abs(float(baseline)), 1e-6)


def _format_value(value, unit):
    if not _valid(value):
        return "N/A"
    number = float(value)
    if unit == "%":
        return f"{100.0 * number:.1f}%"
    if unit == "count":
        return f"{number:.2f}"
    return f"{number:.3g} {unit}".rstrip()


def _write_overview(output_dir, *, variant, status, comparison, reports):
    plt = pyplot()
    profiles = comparison["profiles"]
    fig, ax = plt.subplots(figsize=(16, 10), layout="constrained")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    status_color = PASS if status == "PASS" else FAIL
    ax.text(0, .98, f"WYW {variant.upper()} VERIFICATION", fontsize=24, weight="bold",
            va="top", color="#171717")
    ax.text(1, .98, status, fontsize=24, weight="bold", va="top", ha="right",
            color=status_color)
    ax.plot([0, 1], [.91, .91], color="#d4d4d4", lw=1)

    y = .83
    for profile, result in profiles.items():
        fraction = result["passed"] / max(result["total"], 1)
        ax.text(0, y + .035, profile.upper(), fontsize=12, weight="bold", va="center")
        ax.add_patch(plt.Rectangle((.12, y), .42, .065, color="#e5e5e5"))
        ax.add_patch(plt.Rectangle((.12, y), .42 * fraction, .065,
                                   color=PASS if result.get("pass", False) else FAIL))
        ax.text(.555, y + .032,
                f"{result['passed']} / {result['total']} passed  |  required {result['required']}",
                fontsize=11, va="center")
        y -= .105

    gates = (("Safety", comparison.get("safety_pass", True)),
             ("Performance", comparison.get("performance_pass", True)),
             ("High-speed gates", comparison.get("high_speed_pass", True)))
    ax.text(0, .57, "ACCEPTANCE GATES", fontsize=12, weight="bold")
    for index, (label, passed) in enumerate(gates):
        x = index * .19
        ax.text(x, .515, f"{'PASS' if passed else 'FAIL'}  {label}", fontsize=11,
                weight="bold", color=PASS if passed else FAIL)

    failures = [(summary["scenario_id"], summary.get("failure_reasons"))
                for report in reports for summary in report["scenario_summaries"]
                if summary.get("failure_reasons")]
    checks = []
    for result in profiles.values():
        for item in result["scenarios"]:
            for name, metric in item["metrics"].items():
                if not metric.get("pass", False):
                    checks.append((_regression_score(name, metric), item["scenario_id"], name, metric))
    checks.sort(key=lambda entry: entry[0], reverse=True)

    ax.text(0, .43, f"FAILURE EVENTS  {len(failures)}", fontsize=14, weight="bold",
            color=FAIL if failures else "#404040")
    failure_lines = [f"{sid}: {reason}" for sid, reason in failures[:6]] or ["None"]
    ax.text(0, .39, "\n".join(failure_lines), fontsize=11, va="top", linespacing=1.45)

    ax.text(.52, .43, f"FAILED METRIC CHECKS  {len(checks)}", fontsize=14, weight="bold",
            color=FAIL if checks else "#404040")
    regression_lines = []
    for _, scenario_id, name, metric in checks[:7]:
        label, unit = metric_label(name)
        regression_lines.append(
            f"{scenario_id} | {label}: "
            f"{_format_value(metric.get('baseline'), unit)} -> "
            f"{_format_value(metric.get('current'), unit)}"
        )
    ax.text(.52, .39, "\n".join(regression_lines or ["None"]), fontsize=10.5,
            va="top", linespacing=1.45)
    ax.text(0, .17, "READING GUIDE", fontsize=14, weight="bold")
    ax.text(0, .13,
            "Gray = baseline median   Blue = current value   Red = failed gate or failed metric\n"
            "Profile pass counts use the configured scenario threshold; absolute safety gates remain strict.",
            fontsize=11, va="top", linespacing=1.5, color="#404040")
    ax.text(0, .035,
            "Blue = current pass | red = current fail | gray = baseline. "
            "Open the grouped evidence sheets for scenario-level values and limits.",
            fontsize=10, color="#525252")
    path = output_dir / "comparison.png"
    save_figure(fig, path)
    return path


def _write_metric_overview(output_dir, *, variant, status, comparison, reports):
    """Single compact chart: raw-unit metric panels, speed colors, robust dots."""
    plt = pyplot()
    import numpy as np

    metrics = ("vx_rmse_m_s", "yaw_rmse_rad_s", "tilt_peak_deg", "height_rmse_m",
               "wheel_contact_loss_mean", "wheel_torque_saturation_rate",
               "leg_action_delta_rms", "wheel_action_delta_rms",
               "leg_action_second_diff_rms", "wheel_action_second_diff_rms")
    colors = { -2.0: "#2563eb", -1.0: "#0891b2", -0.5: "#16a34a",
               0.5: "#ca8a04", 1.0: "#ea580c", 2.0: "#dc2626" }
    speeds = tuple(colors)
    contracts = {s.id: s for profile in ("nominal", "robust")
                 for s in V.build_scenarios(variant, profile)}
    nominal = comparison["profiles"].get("nominal", {}).get("scenarios", [])
    robust = comparison["profiles"].get("robust", {}).get("scenarios", [])
    robust_report = next((r for r in reports if r.get("profile") == "robust"), {})
    samples = robust_report.get("samples", [])
    # Keep all ten panels in a dense 5 x 2 matrix.  The previous 4 x 4 grid
    # left the final row mostly empty, which made the evidence look smaller
    # than it is and pushed the labels apart.
    fig = plt.figure(figsize=(16, 8.8))
    # The header is positioned in figure coordinates, leaving a clean band
    # above a two-row, five-column panel matrix.
    grid = fig.add_gridspec(2, 5, left=.035, right=.99, bottom=.08, top=.73,
                            hspace=.42, wspace=.16)
    fig.text(.035, .915, "总体结果", fontsize=16, weight="bold", va="top")
    profile_text = "       ".join(
        f"{profile.title()}: 通过 {result['passed']}/{result['total']}"
        for profile, result in comparison["profiles"].items()
    )
    safety = "PASS" if comparison.get("safety_pass", True) else "FAIL"
    fig.text(.035, .875, f"{profile_text}       Safety: {safety}",
             fontsize=13, weight="bold", color=PASS if safety == "PASS" else FAIL)
    fig.text(.035, .835, "关键结论：Current 相对 Baseline 的改善/退化",
             fontsize=11.5, weight="bold")
    key_names = (("VX RMSE", "vx_rmse_m_s"), ("Yaw RMSE", "yaw_rmse_rad_s"),
                 ("Tilt Peak", "tilt_peak_deg"), ("Height RMSE", "height_rmse_m"),
                 ("Wheel Contact", "wheel_contact_loss_mean"),
                 ("Wheel Torque", "wheel_torque_saturation_rate"),
                 ("Action Smoothness", "wheel_action_second_diff_rms"))
    conclusions = []
    for label, name in key_names:
        deltas = [float(item["metrics"][name]["delta"]) for result in comparison["profiles"].values()
                  for item in result["scenarios"] if _valid(item.get("metrics", {}).get(name, {}).get("delta"))]
        if deltas:
            delta = float(np.median(deltas))
            direction = "改善" if (delta <= 0 if name in V.LOWER_IS_BETTER else delta >= 0) else "退化"
            conclusions.append(f"{label} {'↓' if name in V.LOWER_IS_BETTER else '↑'} {direction} {delta:+.3g}")
    fig.text(.035, .797, "   |   ".join(conclusions), fontsize=10.5, color="#404040")
    axes = [fig.add_subplot(grid[row, column]) for row in range(2) for column in range(5)]
    for index, name in enumerate(metrics):
        ax = axes[index]
        title, unit = metric_label(name)
        factor = 100.0 if unit == "%" else 1.0
        for profile, entries in (("nominal", nominal), ("robust", robust)):
            by_speed = {speed: next((item for item in entries
                                     if item.get("scenario_id") in contracts
                                     and math.isclose(float(contracts[item["scenario_id"]].vx), speed)
                                     and math.isclose(float(contracts[item["scenario_id"]].yaw), 0.0)), None)
                        for speed in speeds}
            baseline_line, current_line = [], []
            for speed in speeds:
                item = by_speed[speed]
                metric = item.get("metrics", {}).get(name, {}) if item else {}
                baseline_line.append(float(metric["baseline"]) * factor if _valid(metric.get("baseline")) else np.nan)
                current_line.append(float(metric["current"]) * factor if _valid(metric.get("current")) else np.nan)
            if profile == "nominal":
                # Draw current first, then the darker dashed baseline on top so it
                # remains visible when the two traces overlap closely.
                ax.plot(speeds, current_line, "-", color="#111827", lw=1.8, zorder=3)
                ax.plot(speeds, baseline_line, linestyle=(0, (5, 3)), color="#4b5563",
                        lw=2.2, zorder=5)
                for speed, base, current in zip(speeds, baseline_line, current_line):
                    if np.isfinite(base):
                        ax.scatter(speed, base, facecolors="white", edgecolors="#374151",
                                   linewidths=1.4, s=48, zorder=7)
                    if np.isfinite(current):
                        ax.scatter(speed, current, color=colors[speed], s=38, zorder=5)
            else:
                # Robust environments are shown as same-color sample clouds.
                for speed in speeds:
                    scenario_ids = {item["scenario_id"] for item in robust
                                    if item.get("scenario_id") in contracts
                                    and math.isclose(float(contracts[item["scenario_id"]].vx), speed)
                                    and math.isclose(float(contracts[item["scenario_id"]].yaw), 0.0)}
                    values = [float(sample[name]) * factor for sample in samples
                              if sample.get("scenario_id") in scenario_ids and _valid(sample.get(name))]
                    if values:
                        jitter = np.linspace(-.055, .055, len(values))
                        ax.scatter(np.full(len(values), speed) + jitter, values, color=colors[speed],
                                   alpha=.32, s=18, zorder=1)
                        ax.scatter(speed, float(np.median(values)), color=colors[speed],
                                   marker="D", s=34, zorder=6)
        # Overall point uses the aggregate of displayed nominal scenario points.
        overall_x = 2.7
        all_current = [float(item.get("metrics", {}).get(name, {}).get("current")) * factor for item in nominal
                       if _valid(item.get("metrics", {}).get(name, {}).get("current"))]
        all_baseline = [float(item.get("metrics", {}).get(name, {}).get("baseline")) * factor for item in nominal
                        if _valid(item.get("metrics", {}).get(name, {}).get("baseline"))]
        if all_baseline:
            ax.scatter(overall_x, float(np.median(all_baseline)), facecolors="white",
                       edgecolors="#737373", s=54, zorder=4)
        if all_current:
            ax.scatter(overall_x, float(np.median(all_current)), color="#111827", s=48, zorder=5)
        # Use a compact title from the metric contract.  Keeping the unit on
        # the second line guarantees that every panel title is at most two
        # lines, including the long contact/action metrics.
        compact_titles = {
            "wheel_contact_loss_mean": "Missing Wheel Contacts",
            "wheel_torque_saturation_rate": "Torque Saturation",
            "leg_action_delta_rms": "Leg Action Delta",
            "wheel_action_delta_rms": "Wheel Action Delta",
            "leg_action_second_diff_rms": "Leg 2nd Diff RMS",
            "wheel_action_second_diff_rms": "Wheel 2nd Diff RMS",
        }
        panel_title = compact_titles.get(name, title)
        ax.set_title(f"{panel_title}\n[{unit or 'value'}]", fontsize=12,
                     weight="bold", loc="left", pad=4)
        ax.set_xticks((*speeds, overall_x), (*[f"{s:+g}" for s in speeds], "Overall"), rotation=45, fontsize=9)
        if index >= 5:
            ax.set_xlabel("Commanded VX (m/s)", fontsize=9, labelpad=3)
        ax.grid(axis="y", alpha=.2)
        ax.tick_params(axis="y", labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[len(metrics):]:
        ax.axis("off")
    from matplotlib.lines import Line2D
    legend = [Line2D([0], [0], color=colors[speed], marker="o", ls="None",
                     label=f"VX {speed:+g}") for speed in speeds]
    legend += [Line2D([0], [0], color="#4b5563", linestyle=(0, (5, 3)), lw=2.2,
                      label="Baseline trend"),
              Line2D([0], [0], color="#111827", ls="-", label="Current trend"),
              Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                     markeredgecolor="#111827", label="Nominal baseline"),
              Line2D([0], [0], marker="o", color="#111827", label="Nominal current"),
              Line2D([0], [0], marker=".", color="#60a5fa", alpha=.5, ls="None", label="Robust samples"),
              Line2D([0], [0], marker="D", color="#111827", ls="None", label="Robust median")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .945),
               ncol=12, frameon=False, fontsize=8.0, columnspacing=.9,
               handletextpad=.45)
    fig.suptitle(f"WYW {variant.upper()} | {status} | Raw-unit metric comparison",
                 fontsize=20, weight="bold", x=.01, ha="left", y=.985)
    path = output_dir / "comparison.png"
    save_figure(fig, path)
    return path


def _write_metric_sheet(output_dir, *, variant, profile, group, names, result):
    plt = pyplot()
    from matplotlib.lines import Line2D

    items = result["scenarios"]
    figure_height = max(5.0, 2.2 + .34 * len(items))
    fig, axes = plt.subplots(1, len(names), figsize=(3.7 * len(names) + 3.2, figure_height),
                             squeeze=False, layout="constrained")
    axes = axes[0]
    labels = [_short_scenario(item["scenario_id"], variant, profile) for item in items]
    for column, (ax, name) in enumerate(zip(axes, names)):
        title, unit = metric_label(name)
        factor = 100.0 if unit == "%" else 1.0
        values = []
        for item in items:
            metric = item["metrics"].get(name, {})
            values.extend(float(metric[key]) * factor for key in ("baseline", "current", "absolute_limit")
                          if _valid(metric.get(key)))
        xmax = max(values, default=1.0)
        xmax = (xmax * 1.22) if xmax > 0 else 1.0
        for y, item in enumerate(items):
            metric = item["metrics"].get(name, {})
            baseline = float(metric["baseline"]) * factor if _valid(metric.get("baseline")) else None
            current = float(metric["current"]) * factor if _valid(metric.get("current")) else None
            if baseline is not None and current is not None:
                ax.plot([baseline, current], [y, y], color="#d4d4d4", lw=1.8, zorder=1)
            if baseline is not None:
                ax.scatter(baseline, y, color=BASELINE, s=28, zorder=2)
            if current is not None:
                ax.scatter(current, y, color=CURRENT if metric.get("pass", False) else FAIL,
                           marker="o" if metric.get("pass", False) else "X", s=38, zorder=3)
                current_text = f"{current:.1f}" if unit == "%" else f"{current:.3g}"
                ax.annotate(current_text, (current, y), xytext=(5, 0),
                            textcoords="offset points", va="center", fontsize=6.5,
                            color=FAIL if not metric.get("pass", False) else "#404040")
            if _valid(metric.get("absolute_limit")):
                limit = float(metric["absolute_limit"]) * factor
                ax.plot([limit, limit], [y - .32, y + .32], color=FAIL, lw=2.2, zorder=2)
        ax.set_xlim(0, xmax)
        ax.set_ylim(len(items) - .5, -.5)
        direction = "higher is better" if name in V.HIGHER_IS_BETTER else "lower is better"
        ax.set_title(f"{textwrap.fill(title, 22)}\n[{unit or 'value'}] | {direction}",
                     loc="left", fontsize=10.5, weight="bold")
        ax.grid(axis="x", alpha=.22)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels(labels if column == 0 else [], fontsize=8.5)
    direction_note = "Each panel keeps its physical unit; metric direction follows the verification contract."
    fig.suptitle(f"{variant.upper()} | {profile.upper()} | {group.upper()}\n{direction_note}",
                 x=.01, ha="left", fontsize=14, weight="bold")
    legend = [Line2D([0], [0], marker="o", color="none", markerfacecolor=BASELINE,
                     markeredgecolor=BASELINE, label="Baseline"),
              Line2D([0], [0], marker="o", color="none", markerfacecolor=CURRENT,
                     markeredgecolor=CURRENT, label="Current pass"),
              Line2D([0], [0], marker="X", color="none", markerfacecolor=FAIL,
                     markeredgecolor=FAIL, label="Current fail"),
              Line2D([0], [0], color=FAIL, lw=2, label="Absolute limit")]
    fig.legend(handles=legend, loc="upper right", ncol=4, frameon=False, fontsize=9)
    path = output_dir / f"{group}_{profile}.png"
    save_figure(fig, path)
    return path


def _write_robust_distributions(output_dir, *, variant, reports, baseline, names):
    plt = pyplot()
    import numpy as np

    robust = next((report for report in reports if report["profile"] == "robust"), None)
    if robust is None:
        return None
    preferred = {
        "flat": ("vx_rmse_m_s", "tilt_peak_deg", "height_peak_to_peak_m",
                 "wheel_contact_loss_mean"),
        "rough": ("completion_rate", "vx_rmse_m_s", "tilt_peak_deg", "height_rmse_m"),
        "jump": ("jump_count", "jump_height_gain_m", "stable_landing_rate", "tilt_peak_deg"),
    }[variant]
    selected = [name for name in preferred if name in names and
                any(name in sample for sample in robust.get("samples", []))]
    if not selected:
        return None
    ids = [summary["scenario_id"] for summary in robust["scenario_summaries"]]
    baseline_summaries = {summary["scenario_id"]: summary for summary in baseline.get(
        "profiles", {}).get("robust", {}).get("scenario_summaries", [])}
    fig, axes = plt.subplots(1, len(selected), figsize=(4.2 * len(selected) + 3,
                                                       max(5, 2.2 + .62 * len(ids))),
                             squeeze=False, layout="constrained")
    for column, (ax, name) in enumerate(zip(axes[0], selected)):
        title, unit = metric_label(name)
        factor = 100.0 if unit == "%" else 1.0
        for y, scenario_id in enumerate(ids):
            values = [factor * float(sample[name]) for sample in robust["samples"]
                      if sample["scenario_id"] == scenario_id and _valid(sample.get(name))]
            if values:
                offsets = np.linspace(-.16, .16, len(values))
                ax.scatter(values, y + offsets, color=CURRENT, alpha=.38, s=18)
                ax.scatter(float(np.median(values)), y, color=CURRENT, marker="D", s=52)
            summary = baseline_summaries.get(scenario_id, {}).get("metrics", {}).get(name, {})
            if all(_valid(summary.get(key)) for key in ("min", "median", "max")):
                ax.plot([factor * float(summary["min"]), factor * float(summary["max"])],
                        [y + .24, y + .24], color=BASELINE, lw=2)
                ax.scatter(factor * float(summary["median"]), y + .24,
                           color=BASELINE, marker="D", s=34)
        direction = "higher is better" if name in V.HIGHER_IS_BETTER else "lower is better"
        ax.set_title(f"{textwrap.fill(title, 22)}\n[{unit or 'value'}] | {direction}",
                     loc="left", fontsize=10.5, weight="bold")
        ax.set_ylim(len(ids) - .5, -.5)
        ax.set_yticks(range(len(ids)))
        ax.set_yticklabels([_short_scenario(sid, variant, "robust") for sid in ids]
                           if column == 0 else [], fontsize=8.5)
        ax.grid(axis="x", alpha=.22)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.suptitle(f"{variant.upper()} | ROBUST SAMPLE DISTRIBUTIONS\n"
                 "Blue dots/diamond: current environments/median | gray line/diamond: baseline range/median",
                 x=.01, ha="left", fontsize=14, weight="bold")
    path = output_dir / "robust_distributions.png"
    save_figure(fig, path)
    return path


def write_charts(output_dir, *, variant, status, comparison, reports, baseline):
    """Write one readable overview image; video telemetry remains separate."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.png"):
        if stale.name != "comparison.png":
            stale.unlink()
    stale_index = output_dir / "charts.md"
    if stale_index.is_file():
        stale_index.unlink()
    profiles = comparison["profiles"]
    return [_write_metric_overview(output_dir, variant=variant, status=status,
                                   comparison=comparison, reports=reports)]


def load_trace(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _validate_plot_comparison(report):
    comparison = report["evaluation"]
    if not isinstance(comparison, dict) or not isinstance(comparison.get("profiles"), dict):
        raise ValueError("evaluation report is missing comparison profiles")
    required_metrics = V.required_metrics(report["variant"])
    for profile in ("nominal", "robust"):
        result = comparison["profiles"].get(profile)
        if not isinstance(result, dict):
            raise ValueError(f"evaluation comparison is missing profile: {profile}")
        if not isinstance(result.get("passed"), int) or not isinstance(result.get("total"), int):
            raise ValueError(f"evaluation comparison has invalid pass counts: {profile}")
        scenarios = result.get("scenarios")
        if not isinstance(scenarios, list):
            raise ValueError(f"evaluation comparison is missing scenarios: {profile}")
        expected_ids = [scenario.id for scenario in V.build_scenarios(report["variant"], profile)]
        if [scenario.get("scenario_id") for scenario in scenarios] != expected_ids:
            raise ValueError(f"evaluation comparison scenario coverage/order mismatch: {profile}")
        for scenario in scenarios:
            metrics = scenario.get("metrics")
            if not isinstance(metrics, dict):
                raise ValueError(
                    f"evaluation comparison is missing metrics: {scenario.get('scenario_id')}"
                )
            missing = required_metrics - set(metrics)
            if missing:
                raise ValueError(
                    f"evaluation comparison is missing metrics for {scenario['scenario_id']}: "
                    f"{sorted(missing)}"
                )
            for name in required_metrics:
                if not {"baseline", "current", "delta", "pass"} <= set(metrics[name]):
                    raise ValueError(
                        f"evaluation comparison metric is incomplete: "
                        f"{scenario['scenario_id']}/{name}"
                    )


def load_plot_inputs(
    report_path, baseline_path=None, *, expected_checkpoint=None, expected_variant=None
):
    """Load and validate a complete sampled evaluation before drawing anything."""
    report_path = Path(report_path).resolve()
    if not report_path.is_file():
        raise ValueError(f"evaluation report does not exist: {report_path}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read evaluation report: {report_path}") from exc
    V.validate_final_report(report)
    if report["mode"] != "evaluate":
        raise ValueError("plot-only input must be an evaluate report")
    _validate_plot_comparison(report)
    if expected_variant is not None and report["variant"] != expected_variant:
        raise ValueError(
            f"report variant is {report['variant']!r}, expected {expected_variant!r}"
        )
    if expected_checkpoint is not None:
        recorded = Path(report["checkpoint"]).resolve()
        if recorded != Path(expected_checkpoint).resolve():
            raise ValueError(
                f"report checkpoint is {recorded}, expected {Path(expected_checkpoint).resolve()}"
            )

    required_metrics = V.required_metrics(report["variant"])
    for run in report["runs"]:
        for index, sample in enumerate(run["samples"]):
            missing = required_metrics - set(sample)
            if missing:
                raise ValueError(
                    f"{run['profile']} sample {index} is missing plotting metrics: {sorted(missing)}"
                )

    if baseline_path is None:
        baseline_ref = report.get("baseline")
        if not isinstance(baseline_ref, dict) or not baseline_ref.get("package"):
            raise ValueError("evaluation report does not reference a baseline package")
        baseline_path = Path(baseline_ref["package"])
        if not baseline_path.is_absolute():
            baseline_path = report_path.parent / baseline_path
    baseline_path = Path(baseline_path).resolve()
    if not baseline_path.is_file():
        raise ValueError(f"baseline package does not exist: {baseline_path}")
    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read baseline package: {baseline_path}") from exc
    V.validate_baseline_package(baseline)
    if baseline["variant"] != report["variant"]:
        raise ValueError("baseline package variant does not match evaluation report")
    recorded_baseline = report.get("baseline", {}).get("checkpoint")
    if recorded_baseline is not None and baseline["checkpoint"] != recorded_baseline:
        raise ValueError("baseline package checkpoint does not match evaluation report")
    return report, baseline, baseline_path


def match_frames(baseline, current):
    """Match within a scenario and phase, allowing differing capture rates."""
    import numpy as np

    groups = {}
    for i, row in enumerate(baseline):
        groups.setdefault((row["scenario_id"], row["phase"]), []).append(i)
    current_keys = {(r["scenario_id"], r["phase"]) for r in current}
    if set(groups) != current_keys:
        raise ValueError("Baseline/current scenario or phase coverage differs; regenerate baseline")
    result = []
    for key, indexes in groups.items():
        times = np.array([baseline[i]["elapsed"] for i in indexes])
        selected = [(i, r) for i, r in enumerate(current) if (r["scenario_id"], r["phase"]) == key]
        if np.any(np.diff(times) <= 0) or any(
            selected[j][1]["elapsed"] <= selected[j-1][1]["elapsed"] for j in range(1, len(selected))
        ):
            raise ValueError(f"Non-monotonic frame times: {key}")
        tolerance = max(float(np.max(np.diff(times))) if len(times) > 1 else 0,
                        max((selected[j][1]["elapsed"] - selected[j-1][1]["elapsed"]
                             for j in range(1, len(selected))), default=0), .04) + 1e-6
        if (abs(times[-1] - selected[-1][1]["elapsed"]) > tolerance
                or abs(times[0] - selected[0][1]["elapsed"]) > tolerance):
            raise ValueError(f"Baseline/current duration mismatch: {key}")
        for i, row in selected:
            nearest = int(np.argmin(abs(times - row["elapsed"])))
            if abs(times[nearest] - row["elapsed"]) > tolerance:
                raise ValueError(f"Missing synchronized frame: {key}")
            reference = baseline[indexes[nearest]]
            if reference.get("command") != row.get("command") or reference.get("settle_s") != row.get("settle_s"):
                raise ValueError(f"Baseline/current command or phase contract differs: {key}")
            result.append((i, indexes[nearest]))
    return [b for _, b in sorted(result)]


def compose_video(output, baseline_video, current_video, current_trace, baseline_trace=None):
    """Compose a 16:9 evidence video with shared-time plots and explicit failures."""
    import cv2
    import imageio.v2 as imageio
    import numpy as np

    plt = pyplot()
    current = load_trace(current_trace)
    if not current.get("frames"):
        raise ValueError("Current telemetry contains no frames")
    base = load_trace(baseline_trace) if baseline_trace and Path(baseline_trace).is_file() else None
    matches = match_frames(base["frames"], current["frames"]) if base else None
    left, right = cv2.VideoCapture(str(baseline_video)), cv2.VideoCapture(str(current_video))
    fps = float(current["fps"])
    temporary = Path(output).with_suffix(".partial.mp4")
    writer = None
    fig = None
    completed = False
    try:
        if not left.isOpened() or not right.isOpened():
            raise ValueError("Cannot open comparison source videos")
        if int(right.get(cv2.CAP_PROP_FRAME_COUNT)) != len(current["frames"]):
            raise ValueError("Current video/trace frame count mismatch")
        if base and int(left.get(cv2.CAP_PROP_FRAME_COUNT)) != len(base["frames"]):
            raise ValueError("Baseline video/trace frame count mismatch")
        if not base:
            duration = left.get(cv2.CAP_PROP_FRAME_COUNT) / left.get(cv2.CAP_PROP_FPS)
            if abs(duration - len(current["frames"]) / fps) > .15:
                raise ValueError("Legacy baseline duration differs; regenerate baseline with traces")
        writer = imageio.get_writer(str(temporary), fps=fps, codec="libx264", pixelformat="yuv420p",
                                    macro_block_size=2, output_params=["-crf", "20", "-preset", "medium"])
        last_sid, last_index, cached_left = None, -1, None
        for index, row in enumerate(current["frames"]):
            bindex = matches[index] if matches is not None else min(
                int(left.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, round(index / fps * left.get(cv2.CAP_PROP_FPS)))
            if bindex != last_index:
                if bindex != last_index + 1:
                    left.set(cv2.CAP_PROP_POS_FRAMES, bindex)
                ok, cached_left = left.read()
                if not ok:
                    raise ValueError("Baseline video ended before the matching frame")
                last_index = bindex
            ok, right_frame = right.read()
            if not ok:
                raise ValueError("Current video ended before its trace")
            canvas = np.full((1080, 1920, 3), 255, dtype=np.uint8)
            canvas[90:630, :960] = cv2.resize(cached_left, (960, 540))
            canvas[90:630, 960:] = cv2.resize(right_frame, (960, 540))
            def text(value, x, y, color=(40, 40, 40), scale=.75):
                cv2.putText(canvas, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
            text(f"{row['scenario_id']} | {row.get('profile', '').upper()} | "
                 f"{row['phase']} | t={row['time']:.2f}s | env 0", 24, 30)
            text("BASELINE", 24, 68, (115, 115, 115))
            text("CURRENT", 984, 68, (199, 132, 2))
            if not base:
                text("Legacy baseline: time-only alignment; telemetry unavailable", 280, 68, (30, 80, 170), .6)
            for x, sample in ((24, base["frames"][bindex] if base else None), (984, row)):
                if sample and sample.get("failure"):
                    cv2.rectangle(canvas, (x - 8, 560), (x + 920, 618), (235, 235, 255), -1)
                    text(f"FAILED: {sample['failure']} | post-reset state excluded", x, 595, (30, 30, 180), .65)
            if last_sid != row["scenario_id"]:
                if fig is not None:
                    plt.close(fig)
                fig, axes = plt.subplots(1, 4, figsize=(19.2, 4.5), dpi=100)
                fig.subplots_adjust(left=.045, right=.985, bottom=.16, top=.83, wspace=.28)
                for ax, key, label in zip(
                    axes,
                    ("vx", "yaw", "height", "tilt"),
                    ("Forward speed (m/s)", "Yaw rate (rad/s)", "Height (m)", "Tilt (deg)"),
                ):
                    for data, color, label_policy in ((base, BASELINE, "Baseline"), (current, CURRENT, "Current")):
                        if data is None:
                            continue
                        rows = [r for r in data["frames"] if r["scenario_id"] == row["scenario_id"]]
                        ax.plot([r["time"] for r in rows], [r["state"].get(key, np.nan)
                                if not r.get("failure") else np.nan for r in rows], color=color, label=label_policy)
                    rows = [r for r in current["frames"] if r["scenario_id"] == row["scenario_id"]]
                    if key in {"vx", "yaw", "height"}:
                        ax.plot([r["time"] for r in rows], [r["command"].get(key, np.nan) for r in rows],
                                "--", color="#262626", lw=1, label="Command")
                    elif row.get("tilt_limit") is not None:
                        ax.axhline(row["tilt_limit"], color=FAIL, ls="--", label="Limit")
                    ax.axvspan(0, row["settle_s"], color="#eeeeee")
                    ax.set_title(label, loc="left")
                    ax.set_xlabel("Scenario time (s)")
                    ax.grid(alpha=.2)
                    ax.legend(fontsize=8, loc="upper right")
                fig.canvas.draw()
                chart = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy(), cv2.COLOR_RGB2BGR)
                last_sid = row["scenario_id"]
            canvas[630:] = chart
            for ax in axes:
                px = int(ax.transData.transform((row["time"], 0))[0])
                box = ax.get_window_extent()
                cv2.line(canvas, (px, 630 + 450 - int(box.y1)), (px, 630 + 450 - int(box.y0)), (50, 50, 50), 1)
            writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        completed = True
    finally:
        left.release()
        right.release()
        if writer is not None:
            writer.close()
        if fig is not None:
            plt.close(fig)
        if not completed and temporary.is_file():
            temporary.unlink()
    temporary.replace(output)
    return Path(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    report, baseline, baseline_path = load_plot_inputs(args.report, args.baseline)
    output_dir = args.report.resolve().parent
    paths = write_charts(output_dir, variant=report["variant"], status=report["status"],
                         comparison=report["evaluation"], reports=report["runs"], baseline=baseline)
    if args.video:
        video = Path(baseline["video"])
        if not video.is_absolute():
            video = baseline_path.parent / video
        baseline_trace = Path(baseline.get("trace", "trace.json"))
        if not baseline_trace.is_absolute():
            baseline_trace = baseline_path.parent / baseline_trace
        compose_video(output_dir / "comparison.mp4", video, output_dir / "current.mp4",
                      output_dir / "trace.json", baseline_trace)
    print(f"Wrote {len(paths)} charts to {output_dir}")


if __name__ == "__main__":
    main()
