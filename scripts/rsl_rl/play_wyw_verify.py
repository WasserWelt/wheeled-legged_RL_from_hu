"""Standardized policy verification for the WYW Flat, Rough and Jump tasks."""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
VERIFY_PATH = (
    REPO_ROOT
    / "source/agent_tasks/agent_tasks/direct/wheelbipe/wyw/verify.py"
)


def _load_verify_contract():
    name = "_wyw_verify_contract"
    spec = importlib.util.spec_from_file_location(name, VERIFY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load verification contract: {VERIFY_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V = _load_verify_contract()

TASK_IDS = {
    "flat": {
        "nominal": "Robotics-Wheelbipe-FDU-wyw-Flat-Play-v1",
        "robust": "Robotics-Wheelbipe-FDU-wyw-Flat-v1",
    },
    "rough": {
        "nominal": "Robotics-Wheelbipe-FDU-wyw-Rough-Play-v1",
        "robust": "Robotics-Wheelbipe-FDU-wyw-Rough-v1",
    },
    "jump": {
        "nominal": "Robotics-Wheelbipe-FDU-wyw-Jump-Play-v1",
        "robust": "Robotics-Wheelbipe-FDU-wyw-Jump-v1",
    },
}

EXPECTED_EXPERIMENTS = {
    "flat": "wheelbipe_fdu_wyw_flat_direct",
    "rough": "wheelbipe_fdu_wyw_rough_direct",
    "jump": "wheelbipe_fdu_wyw_jump_direct",
}
VIDEO_FPS = 15.0
VIDEO_RESOLUTION = (960, 540)
# H.264 keeps the recordings free of the blocky artifacts the legacy mp4v
# (MPEG-4 Part 2) encoder produced; libx264 is provided by imageio-ffmpeg.
VIDEO_CRF = "20"


def _open_video_writer(path: "Path", fps: float):
    """Create an H.264 (libx264) mp4 writer expecting RGB frames."""
    import imageio

    return imageio.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=2,
        output_params=["-crf", VIDEO_CRF, "-preset", "medium"],
    )


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--variant", choices=("flat", "rough", "jump"), default=None)
parser.add_argument("--checkpoint", type=Path, default=None)
parser.add_argument("--mode", choices=("baseline", "calibrate", "evaluate"), default="evaluate")
parser.add_argument("--profile", choices=("nominal", "robust", "all"), default="all")
parser.add_argument(
    "--baseline-config",
    type=Path,
    default=REPO_ROOT / "configs" / "wyw_verify_baseline.json",
)
parser.add_argument("--output-dir", type=Path, default=None)
parser.add_argument("--list-scenarios", action="store_true")
parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.variant is None:
    parser.error("--variant is required")
if args_cli.checkpoint is None and not args_cli.list_scenarios:
    parser.error("--checkpoint is required")
if not args_cli.list_scenarios and not args_cli._worker and args_cli.profile != "all":
    parser.error("baseline/evaluate output requires --profile all")


def _git_metadata() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--short"], cwd=REPO_ROOT, text=True, capture_output=True, check=False
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _default_output_dir() -> Path:
    if args_cli.mode == "baseline":
        return _load_baseline_config()["package_dir"]
    if args_cli.checkpoint is None:
        raise ValueError("--checkpoint is required to create verification outputs")
    return args_cli.checkpoint.resolve().parent / "acceptance" / args_cli.variant / "evaluate"


def _load_baseline_config() -> dict[str, Any]:
    path = args_cli.baseline_config.resolve()
    if not path.is_file():
        raise ValueError(f"baseline config does not exist: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("variant") != args_cli.variant:
        raise ValueError("baseline config variant mismatch")
    for key in ("checkpoint", "package_dir"):
        if key not in config:
            raise ValueError(f"baseline config missing {key}")
    config["checkpoint"] = (REPO_ROOT / config["checkpoint"]).resolve()
    config["package_dir"] = (REPO_ROOT / config["package_dir"]).resolve()
    return config


def _scenario_profiles(profile: str) -> list[str]:
    return ["nominal", "robust"] if profile == "all" else [profile]


def _selected_scenarios() -> list[Any]:
    return [
        scenario
        for profile in _scenario_profiles(args_cli.profile)
        for scenario in V.build_scenarios(args_cli.variant, profile)
    ]


def _strip_child_args(arguments: list[str]) -> list[str]:
    result: list[str] = []
    skip_next = False
    for token in arguments:
        if skip_next:
            skip_next = False
            continue
        if token in {"--profile", "--output-dir"}:
            skip_next = True
            continue
        if token.startswith("--profile=") or token.startswith("--output-dir="):
            continue
        if token == "--_worker":
            continue
        result.append(token)
    return result


def _write_result_table(
    output_dir: Path,
    reports: list[dict[str, Any]],
    comparison: dict[str, Any] | None = None,
    baseline: dict[str, Any] | None = None,
) -> Path:
    """Write one concise row per scenario with baseline/current deltas."""
    rows: list[dict[str, Any]] = []
    comparison_by_id = {}
    baseline_by_id = {}
    if comparison is not None:
        comparison_by_id = {
            item["scenario_id"]: item
            for profile in comparison["profiles"].values()
            for item in profile["scenarios"]
        }
    if baseline is not None:
        for profile in baseline.get("profiles", {}).values():
            for item in profile.get("scenario_summaries", []):
                baseline_by_id[item["scenario_id"]] = item
    for report in reports:
        profile = report["profile"]
        for summary in report["scenario_summaries"]:
            item = comparison_by_id.get(summary["scenario_id"], {})
            metrics = summary["metrics"]
            row: dict[str, Any] = {
                "profile": profile,
                "scenario_id": summary["scenario_id"],
                "baseline_status": "PASS"
                if baseline_by_id.get(summary["scenario_id"], {}).get("finite", True)
                and not baseline_by_id.get(summary["scenario_id"], {}).get("failure_reasons", {})
                else "FAIL",
                "current_status": "PASS" if summary["finite"] and not summary["failure_reasons"] else "FAIL",
                "safety_status": "PASS" if item.get("safety_pass", False) else "FAIL",
                "performance_status": "PASS" if item.get("performance_pass", False) else "FAIL",
                "failure_reasons": json.dumps(summary["failure_reasons"], sort_keys=True),
            }
            for name in sorted(metrics):
                current = metrics[name]["median"]
                metric = item.get("metrics", {}).get(name, {})
                row[f"baseline_{name}"] = metric.get("baseline")
                row[f"current_{name}"] = current
                row[f"delta_{name}"] = metric.get("delta")
            rows.append(row)
    path = output_dir / "results.csv"
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _concat_profile_videos(
    output_dir: Path, reports: list[dict[str, Any]], name: str
) -> Path:
    by_profile = {report["profile"]: Path(report["video"]) for report in reports}
    if set(by_profile) != {"nominal", "robust"}:
        raise ValueError("all profiles are required to concatenate verification video")
    output = output_dir / name
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(by_profile["nominal"]), "-i", str(by_profile["robust"]),
        "-filter_complex",
        "[0:v][1:v]concat=n=2:v=1:a=0[v]",
        "-map", "[v]", "-r", str(VIDEO_FPS), "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(output),
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    if completed.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
        detail = completed.stderr.strip() or "unknown ffmpeg error"
        raise RuntimeError(f"cannot create comparison video: {detail}")
    return output


def _write_baseline_comparison_video(
    output_dir: Path, baseline_video: Path, current_video: Path
) -> Path:
    output = output_dir / "comparison.mp4"
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(baseline_video), "-i", str(current_video),
        "-filter_complex",
        "[0:v]setpts=PTS-STARTPTS,drawtext=text='BASELINE':x=24:y=h-th-24:fontsize=28:fontcolor=yellow:box=1:boxcolor=black@0.55[left];"
        "[1:v]setpts=PTS-STARTPTS,drawtext=text='CURRENT':x=24:y=h-th-24:fontsize=28:fontcolor=cyan:box=1:boxcolor=black@0.55[right];"
        "[left][right]hstack=inputs=2:shortest=1[v]",
        "-map", "[v]", "-r", str(VIDEO_FPS), "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(output),
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    if completed.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"cannot create baseline comparison video: {completed.stderr.strip()}")
    return output


def _format_metric_value(value: float, unit: str) -> str:
    if unit == "%":
        return f"{100.0 * value:.1f}%"
    if unit == "count":
        return f"{value:.2f}"
    return f"{value:.3f} {unit}"


def _write_comparison_chart(
    output_dir: Path,
    *,
    status: str,
    comparison: dict[str, Any],
    aggregate: dict[str, Any],
) -> Path:
    """Render one aggregate row per metric; robust envs are never expanded."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/wyw_verify_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = aggregate["metrics"]
    baseline_color = "#4B5563"
    current_color = "#0891B2"
    green = "#15803D"
    red = "#B91C1C"
    pale = "#F3F4F6"

    relative_current = []
    for metric in metrics:
        baseline_value = metric["baseline"]
        current_value = metric["current"]
        if math.isclose(baseline_value, 0.0):
            relative_current.append(100.0 if math.isclose(current_value, 0.0) else 200.0)
        else:
            relative_current.append(100.0 * current_value / abs(baseline_value))

    figure_height = max(7.6, 2.4 + 0.46 * len(metrics))
    fig = plt.figure(figsize=(13.5, figure_height), facecolor="white")
    grid = fig.add_gridspec(2, 1, height_ratios=(1.15, 5.0), hspace=0.18)
    header = fig.add_subplot(grid[0])
    header.axis("off")
    header.text(0.0, 0.92, f"WYW {args_cli.variant.title()} Baseline Comparison", fontsize=22, weight="bold", va="top")
    header.text(
        0.0,
        0.54,
        f"Median across {aggregate['scenario_count']} scenario medians; robust environments are aggregated per scenario",
        fontsize=10.5,
        color="#4B5563",
        va="top",
    )
    profile_parts = []
    for profile in ("nominal", "robust"):
        result = comparison["profiles"][profile]
        profile_parts.append(f"{profile.title()} {result['passed']}/{result['total']}")
    safety_passed = sum(
        int(item["safety_pass"])
        for result in comparison["profiles"].values()
        for item in result["scenarios"]
    )
    total = sum(result["total"] for result in comparison["profiles"].values())
    summary = f"{status}     Safety {safety_passed}/{total}     " + "     ".join(profile_parts)
    header.text(
        0.0,
        0.12,
        summary,
        fontsize=12,
        weight="bold",
        color=green if status == "PASS" else red,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#F0FDF4" if status == "PASS" else "#FEF2F2", "edgecolor": "none"},
    )

    axis = fig.add_subplot(grid[1])
    y = np.arange(len(metrics))
    height = 0.31
    axis.barh(y - height / 2, [100.0] * len(metrics), height, color=baseline_color, label="Baseline")
    axis.barh(y + height / 2, relative_current, height, color=current_color, label="Current")
    axis.axvline(100.0, color="#9CA3AF", linewidth=1, linestyle="--", zorder=0)
    max_relative = max([100.0, *relative_current])
    axis.set_xlim(0.0, max(125.0, max_relative * 1.22))
    axis.set_yticks(y, [metric["label"] for metric in metrics], fontsize=11)
    axis.invert_yaxis()
    axis.set_xlabel("Relative magnitude (baseline = 100)", color="#4B5563")
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#D1D5DB")
    axis.tick_params(axis="y", length=0)
    axis.legend(
        loc="lower right", bbox_to_anchor=(1.0, 1.01), frameon=False, ncol=2
    )

    x_text = axis.get_xlim()[1] * 0.995
    for index, metric in enumerate(metrics):
        baseline_text = _format_metric_value(metric["baseline"], metric["unit"])
        current_text = _format_metric_value(metric["current"], metric["unit"])
        change = metric["improvement_percent"]
        if math.isclose(metric["improvement"], 0.0, abs_tol=1e-12):
            change_text = "same"
        elif change is None:
            change_text = "better" if metric["improvement"] > 0 else "same" if metric["pass"] else "worse"
        elif change > 0:
            change_text = f"{change:+.1f}% better"
        else:
            change_text = f"{abs(change):.1f}% worse"
        color = green if metric["pass"] else red
        axis.text(x_text, index - height / 2, baseline_text, ha="right", va="center", fontsize=9, color=baseline_color)
        axis.text(x_text, index + height / 2, f"{current_text}  ({change_text})", ha="right", va="center", fontsize=9, color=color, weight="bold")

    fig.text(
        0.012,
        0.012,
        "Lower is better for tracking error, tilt and action variation; higher is better for survival and task outcomes.",
        fontsize=9,
        color="#6B7280",
    )
    path = output_dir / "comparison.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=pale)
    plt.close(fig)
    return path


def _cleanup_intermediates(
    output_dir: Path, reports: list[dict[str, Any]], *, remove_videos: bool
) -> None:
    """Keep only final evidence after all-profile aggregation."""
    for report in reports:
        names = [f"profile_report.{report['profile']}.json"]
        if remove_videos:
            names.append(Path(report["video"]).name)
        for name in names:
            path = output_dir / name
            if path.is_file():
                path.unlink()


def _load_baseline_package() -> dict[str, Any]:
    config = _load_baseline_config()
    path = config["package_dir"] / "baseline.json"
    if not path.is_file():
        raise ValueError(f"baseline package does not exist: {path}; run --mode baseline first")
    baseline = json.loads(path.read_text(encoding="utf-8"))
    V.validate_baseline_package(baseline)
    if baseline["variant"] != args_cli.variant:
        raise ValueError("baseline package variant mismatch")
    video = Path(baseline["video"])
    if not video.is_absolute():
        video = (path.parent / video).resolve()
    if not video.is_file():
        raise ValueError(f"baseline video does not exist: {video}")
    return baseline


def _finalize_reports(output_dir: Path, reports: list[dict[str, Any]]) -> int:
    if args_cli.checkpoint is None:
        raise ValueError("--checkpoint is required to finalize verification reports")
    scenarios = _selected_scenarios()
    profile_summaries = {
        report["profile"]: report["scenario_summaries"] for report in reports
    }
    profile_video_names = [Path(report["video"]).name for report in reports]
    combined_video = _concat_profile_videos(output_dir, reports, "combined.mp4")
    comparison: dict[str, Any] | None = None
    aggregate: dict[str, Any] | None = None
    chart: Path | None = None
    if args_cli.mode == "baseline" or args_cli.mode == "calibrate":
        config_checkpoint = _load_baseline_config()["checkpoint"]
        if args_cli.checkpoint.resolve() != config_checkpoint:
            raise ValueError(
                f"baseline checkpoint must match configured canonical checkpoint: {config_checkpoint}"
            )
        baseline = V.make_baseline_package(
            variant=args_cli.variant,
            profile_summaries=profile_summaries,
            checkpoint=str(args_cli.checkpoint.resolve()),
            video="baseline.mp4",
        )
        baseline["video"] = "baseline.mp4"
        (output_dir / "baseline.json").write_text(
            json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        final_video = output_dir / "baseline.mp4"
        combined_video.replace(final_video)
        status = "BASELINE"
        exit_code = 0
    else:
        baseline = _load_baseline_package()
        comparison = V.compare_summaries(
            profile_summaries=profile_summaries, baseline=baseline
        )
        aggregate = V.aggregate_comparison(
            profile_summaries=profile_summaries, baseline=baseline
        )
        baseline_video = Path(baseline["video"])
        if not baseline_video.is_absolute():
            baseline_video = (_load_baseline_config()["package_dir"] / baseline_video).resolve()
        final_video = _write_baseline_comparison_video(output_dir, baseline_video, combined_video)
        status = "PASS" if comparison["pass"] else "FAIL"
        exit_code = 0 if comparison["pass"] else 1
        chart = _write_comparison_chart(
            output_dir, status=status, comparison=comparison, aggregate=aggregate
        )
    if args_cli.mode not in {"baseline", "calibrate"}:
        _write_result_table(output_dir, reports, comparison, baseline)
    for report in reports:
        report["video"] = str(final_video)
    final_report = {
        "schema_version": V.VERIFY_SCHEMA_VERSION,
        "status": status,
        "mode": args_cli.mode,
        "variant": args_cli.variant,
        "profiles": _scenario_profiles(args_cli.profile),
        "standard": True,
        "seed": V.STANDARD_SEED,
        "simulated_duration_s": sum(
            scenario.settle_s + scenario.score_s for scenario in scenarios
        ),
        "checkpoint": str(args_cli.checkpoint.resolve()),
        "git": _git_metadata(),
        "video": str(final_video),
        "chart": None if chart is None else str(chart),
        "baseline": None
        if args_cli.mode in {"baseline", "calibrate"}
        else {"package": str(_load_baseline_config()["package_dir"] / "baseline.json"), "checkpoint": baseline["checkpoint"]},
        "evaluation": comparison,
        "aggregate_comparison": aggregate,
        "runs": reports,
    }
    if args_cli.mode not in {"baseline", "calibrate"}:
        V.validate_final_report(final_report)
        (output_dir / "report.json").write_text(
            json.dumps(final_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    for name in profile_video_names:
        path = output_dir / name
        if path.is_file():
            path.unlink()
    _cleanup_intermediates(output_dir, reports, remove_videos=False)
    if combined_video.is_file():
        combined_video.unlink()
    if args_cli.mode == "baseline" or args_cli.mode == "calibrate":
        # baseline.json and baseline.mp4 are the persistent reference package.
        (output_dir / "baseline.json").write_text(
            json.dumps(baseline, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"WYW VERIFY {status}: {output_dir}", flush=True)
    return exit_code


def _run_all_profiles() -> int:
    output_dir = (args_cli.output_dir or _default_output_dir()).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    child_base = _strip_child_args(sys.argv[1:])
    reports = []
    for profile in ("nominal", "robust"):
        # Workers share the flat output dir; per-profile files carry a suffix.
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            *child_base,
            "--profile",
            profile,
            "--output-dir",
            str(output_dir),
            "--_worker",
        ]
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"{profile} verification worker exited with {completed.returncode}")
        report_path = output_dir / f"profile_report.{profile}.json"
        if not report_path.is_file():
            raise RuntimeError(
                f"{profile} verification worker did not produce {report_path}"
            )
        reports.append(
            json.loads(report_path.read_text(encoding="utf-8"))
        )
    return _finalize_reports(output_dir, reports)


if args_cli.list_scenarios:
    listed = [scenario.to_dict() for scenario in _selected_scenarios()]
    print(json.dumps(listed, indent=2, sort_keys=True))
    raise SystemExit(0)

if args_cli.profile == "all" and not args_cli._worker:
    raise SystemExit(_run_all_profiles())

args_cli.enable_cameras = True
if getattr(args_cli, "rendering_mode", None) is None:
    # "performance" starves the RTX path tracer of samples, leaving grainy
    # noise in the recordings; "balanced" denoises cleanly at moderate cost.
    args_cli.rendering_mode = "balanced"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from isaaclab.sensors.ray_caster import RayCaster  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz, quat_inv, quat_apply  # noqa: E402
from isaaclab.utils.warp import raycast_mesh  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

import agent_world  # noqa: E402,F401
import agent_tasks  # noqa: E402,F401
from agent_rl.rsl_rl.env import RslRlVecEnvWrapper  # noqa: E402
from agent_rl.rsl_rl.runners import OnPolicySequenceRunner  # noqa: E402
from agent_tasks.direct.wheelbipe.wyw.rough_cfg import (  # noqa: E402
    FDU_ROUGH_VERIFY_TERRAIN_CFG,
    fdu_verify_terrain_columns,
)
from agent_tasks.direct.wheelbipe.wyw import wyw_constants as C  # noqa: E402
from agent_tasks.direct.wheelbipe.wyw.fdu_semantics import (  # noqa: E402
    compute_fdu_action_differences,
)


def _nested(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _validate_checkpoint_metadata(checkpoint: Path, variant: str) -> dict[str, Any]:
    params_dir = checkpoint.resolve().parent / "params"
    env_path = params_dir / "env.yaml"
    agent_path = params_dir / "agent.yaml"
    if not env_path.is_file() or not agent_path.is_file():
        raise ValueError(f"checkpoint metadata missing under {params_dir}")
    # Isaac Lab's dump_yaml emits Python-specific tuple and slice tags. Only
    # scalar contract fields are needed here, so BaseLoader safely ignores
    # object construction and represents all scalar values as strings.
    env_meta = yaml.load(env_path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    agent_meta = yaml.load(agent_path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    if not isinstance(env_meta, dict) or not isinstance(agent_meta, dict):
        raise ValueError("checkpoint metadata must contain YAML mappings")
    checks = {
        "action_space": env_meta.get("action_space") == "6",
        "observation_space": env_meta.get("observation_space") == "25",
        "state_space": env_meta.get("state_space") == "141",
        "history_frames": env_meta.get("num_obs_hist") == "5",
        "semantics_version": env_meta.get("wyw_training_semantics_version")
        == "fdu_flat_p0_direct_bars_fd_vel_v3_material_split",
        "joint_velocity_source": env_meta.get("wyw_joint_velocity_source")
        == "wrapped_position_difference",
        "joint_velocity_diff_dt": math.isclose(
            float(env_meta.get("wyw_joint_velocity_diff_dt") or -1.0), 0.002
        ),
        "physics_dt": math.isclose(float(_nested(env_meta, "sim", "dt") or -1.0), 0.002),
        "decimation": env_meta.get("decimation") == "5",
        "runner": agent_meta.get("runner_class") == "OnPolicySequenceRunner",
        "latent_dim": _nested(agent_meta, "policy", "latent_dim") == "3",
        "experiment": agent_meta.get("experiment_name") == EXPECTED_EXPERIMENTS[variant],
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"checkpoint metadata contract mismatch: {', '.join(failed)}")
    return {"env_yaml": str(env_path), "agent_yaml": str(agent_path), "checks": checks}


def _configure_env(variant: str, profile: str):
    task_id = TASK_IDS[variant][profile]
    cfg = parse_env_cfg(task_id, device=args_cli.device, num_envs=V.standard_num_envs(profile))
    cfg.seed = V.STANDARD_SEED
    cfg.scene.num_envs = V.standard_num_envs(profile)
    cfg.play = True
    cfg.play_keep_done_reset = True
    cfg.commands.debug_vis = False
    cfg.wyw_flat_command_curriculum_enabled = False
    cfg.wyw_rough_curriculum_enabled = False
    cfg.curriculum = None
    cfg.height_scanner.debug_vis = False
    cfg.viewer.resolution = VIDEO_RESOLUTION
    if profile == "nominal":
        cfg.events.reset_base = copy.deepcopy(cfg.events.reset_base)
        cfg.events.reset_base.params["pose_range"] = {}
        cfg.events.reset_base.params["velocity_range"] = {}
    if variant == "rough":
        cfg.terrain = copy.deepcopy(cfg.terrain)
        cfg.terrain.terrain_generator = copy.deepcopy(FDU_ROUGH_VERIFY_TERRAIN_CFG)
        cfg.terrain.max_init_terrain_level = 9
    return task_id, cfg


def _set_reset_pose(raw, *, yaw: float, keep_velocity_randomization: bool) -> None:
    term = copy.deepcopy(raw.event_manager.get_term_cfg("reset_base"))
    # Verification courses require an exact starting line. Robustness comes
    # from startup domain randomization and, where configured, reset velocity.
    term.params["pose_range"] = {"yaw": (yaw, yaw)}
    if not keep_velocity_randomization:
        term.params["velocity_range"] = {}
    raw.event_manager.set_term_cfg("reset_base", term)


def _terrain_ground_height(raw, points_xy: torch.Tensor) -> torch.Tensor:
    mesh_path = raw.dot_scanner.cfg.mesh_prim_paths[0]
    mesh = RayCaster.meshes[mesh_path]
    starts = torch.zeros(points_xy.shape[0], 3, device=raw.device)
    starts[:, :2] = points_xy
    starts[:, 2] = 10.0
    directions = torch.zeros_like(starts)
    directions[:, 2] = -1.0
    hits = raycast_mesh(starts, directions, mesh=mesh)[0]
    if not torch.isfinite(hits[:, 2]).all():
        raise RuntimeError("terrain ground-height raycast missed")
    return hits[:, 2]


def _prepare_rough_scene(raw, scenario, num_envs: int, robust: bool) -> tuple[torch.Tensor, torch.Tensor]:
    columns = fdu_verify_terrain_columns()[scenario.terrain]
    row = 5 if scenario.difficulty is None else int(round(scenario.difficulty * 10))
    row = min(max(row, 0), raw.terrain.terrain_origins.shape[0] - 1)
    selected_columns = torch.tensor(
        [columns[index % len(columns)] for index in range(num_envs)],
        dtype=torch.long,
        device=raw.device,
    )
    ids = torch.arange(num_envs, device=raw.device)
    raw.terrain.terrain_levels[ids] = row
    raw.terrain.terrain_types[ids] = selected_columns
    centers = raw.terrain.terrain_origins[row, selected_columns].clone()
    if scenario.terrain.startswith("single_step_"):
        start_x = 1.25 if scenario.direction == "down" else -1.25
        base_yaw = math.pi if scenario.direction == "down" else 0.0
    else:
        start_x = -3.0
        base_yaw = 0.0
    lanes = torch.zeros(num_envs, device=raw.device)
    if robust:
        lane_values = torch.tensor((-1.6, -0.8, 0.0, 0.8, 1.6), device=raw.device)
        lanes = lane_values[torch.div(ids, max(len(columns), 1), rounding_mode="floor") % 5]
    starts_xy = centers[:, :2].clone()
    starts_xy[:, 0] += start_x
    starts_xy[:, 1] += lanes
    start_ground = _terrain_ground_height(raw, starts_xy)
    raw.terrain.env_origins[ids, :2] = starts_xy
    raw.terrain.env_origins[ids, 2] = start_ground
    yaw = base_yaw + math.radians(scenario.approach_deg)
    _set_reset_pose(raw, yaw=yaw, keep_velocity_randomization=robust)
    heading = torch.tensor((math.cos(yaw), math.sin(yaw)), device=raw.device).repeat(num_envs, 1)
    return starts_xy, heading


def _patch_current_command_observation(obs: dict[str, torch.Tensor], raw) -> None:
    block = raw._get_wyw_command_block()
    obs["policy"][:, 6:9] = block
    history = obs["policy_hist"].view(raw.num_envs, raw.cfg.num_obs_hist, -1)
    history[:, -1, 6:9] = block


def _failure_reasons(raw, dones: torch.Tensor, timeouts: torch.Tensor) -> list[str | None]:
    names = (
        "numerical_safety",
        "terrain_boundary",
        "persistent_failure",
        "contact",
        "orientation",
    )
    reasons: list[str | None] = []
    for index in range(raw.num_envs):
        if not bool(dones[index]):
            reasons.append(None)
            continue
        if bool(timeouts[index]):
            reasons.append("timeout")
            continue
        reason = "terminated"
        for name in names:
            mask = getattr(raw, f"_wyw_done_reason_{name}", None)
            if mask is not None and bool(mask[index]):
                reason = name
                break
        reasons.append(reason)
    return reasons


class _VideoRecorder:
    def __init__(self, gym_env, raw, path: Path):
        self.gym_env = gym_env
        self.raw = raw
        self.path = path
        self.writer = None
        self.accumulator = 0.0

    def capture(
        self,
        scenario,
        phase: str,
        elapsed: float,
        active: int,
        actions: torch.Tensor,
    ) -> None:
        import cv2

        self.accumulator += VIDEO_FPS * float(self.raw.step_dt)
        if self.accumulator < 1.0:
            return
        self.accumulator -= 1.0
        root = self.raw.robot.data.root_pos_w[0].detach().cpu().numpy()
        eye = root + np.array((-3.0, -3.0, 1.5))
        self.raw.viewport_camera_controller.set_view_env_index(0)
        self.raw.viewport_camera_controller.update_view_location(eye=eye, lookat=root)
        frame = self.gym_env.render()
        if frame is None:
            raise RuntimeError("environment render returned no frame")
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        frame = np.ascontiguousarray(frame)
        command = (
            float(self.raw.command[0, 0].item()),
            float(self.raw.command[0, 2].item()),
            float(self.raw._get_observation_height_cmd()[0].item()),
        )
        action = actions[0].detach().cpu().tolist()
        line_columns = (
            (
                f"WYW {scenario.variant.upper()} / {scenario.profile.upper()}",
                scenario.id,
                f"{phase}  t={elapsed:5.2f}s  active={active}/{self.raw.num_envs}",
            ),
            (
                f"cmd   vx={command[0]:+5.2f} yaw={command[1]:+5.2f} h={command[2]:.3f}",
                f"act L lf0={action[0]:+6.3f} l20={action[1]:+6.3f} whl={action[2]:+6.3f}",
                f"act R rf0={action[3]:+6.3f} r20={action[4]:+6.3f} whl={action[5]:+6.3f}",
            ),
        )
        overlay = frame.copy()
        cv2.rectangle(overlay, (12, 10), (frame.shape[1] - 12, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.58, frame, 0.42, 0.0, frame)
        for column_index, lines in enumerate(line_columns):
            x = 24 if column_index == 0 else frame.shape[1] // 2
            for line_index, line in enumerate(lines):
                cv2.putText(
                    frame,
                    line,
                    (x, 34 + 31 * line_index),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.54,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
        if self.writer is None:
            self.writer = _open_video_writer(self.path, VIDEO_FPS)
        # cv2.putText drew on the RGB array above; the H.264 writer expects RGB.
        self.writer.append_data(frame)

    def close(self, *, validate: bool = True) -> None:
        if self.writer is not None:
            self.writer.close()
        if validate and (not self.path.is_file() or self.path.stat().st_size == 0):
            raise RuntimeError(f"verification produced no video: {self.path}")


def _state_metrics(raw, variant: str, scenario) -> dict[str, torch.Tensor]:
    if variant == "jump":
        vx = raw.robot.data.root_lin_vel_b[:, 0]
        yaw = raw.robot.data.root_ang_vel_b[:, 2]
    else:
        _, _, heading_yaw = euler_xyz_from_quat(raw.robot.data.root_quat_w)
        zeros = torch.zeros_like(heading_yaw)
        heading_inv = quat_inv(quat_from_euler_xyz(zeros, zeros, heading_yaw))
        vx = quat_apply(heading_inv, raw.robot.data.root_lin_vel_w)[:, 0]
        yaw = raw.robot.data.root_ang_vel_w[:, 2]
    gravity_z = raw.robot.data.projected_gravity_b[:, 2].clamp(-1.0, 1.0)
    tilt = torch.rad2deg(torch.acos((-gravity_z).clamp(-1.0, 1.0)))
    height = raw._get_fdu_base_height()
    return {
        "vx": vx,
        "yaw": yaw,
        "tilt": tilt,
        "height": height,
        "root_z": raw.robot.data.root_pos_w[:, 2],
    }


def _run_scenario(env, gym_env, raw, policy, recorder, scenario) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    num_envs = raw.num_envs
    ids = torch.arange(num_envs, device=raw.device)
    robust = scenario.profile == "robust"
    with torch.inference_mode():
        raw.set_wyw_command_override(ids, vx=0.0, yaw=0.0, height=scenario.height)
        starts_xy = raw.robot.data.root_pos_w[:, :2].clone()
        heading = torch.tensor((1.0, 0.0), device=raw.device).repeat(num_envs, 1)
        if scenario.variant == "rough":
            starts_xy, heading = _prepare_rough_scene(raw, scenario, num_envs, robust)
        else:
            _set_reset_pose(raw, yaw=0.0, keep_velocity_randomization=robust)
        obs, _ = env.reset()
        starts_xy = raw.robot.data.root_pos_w[:, :2].clone()
    active = torch.ones(num_envs, dtype=torch.bool, device=raw.device)
    survived = torch.ones_like(active)
    finite = torch.ones_like(active)
    reasons: list[str | None] = [None] * num_envs

    def advance(phase: str, elapsed: float):
        nonlocal obs, active, survived, finite, reasons
        with torch.inference_mode():
            was_active = active.clone()
            actions = policy(obs)
            action_finite = torch.isfinite(actions).all(dim=-1)
            finite[was_active] &= action_finite[was_active]
            invalid_actions = was_active & ~action_finite
            for index in torch.nonzero(invalid_actions).flatten().tolist():
                survived[index] = False
                active[index] = False
                reasons[index] = "non_finite_action"
            actions = torch.where(torch.isfinite(actions), actions, torch.zeros_like(actions))
            actions[~active] = 0.0
            applied_actions = actions
            if env.clip_actions is not None:
                applied_actions = torch.clamp(
                    actions, -float(env.clip_actions), float(env.clip_actions)
                )
            before = _state_metrics(raw, scenario.variant, scenario)
            state_finite = torch.stack([torch.isfinite(value) for value in before.values()]).all(dim=0)
            finite[was_active] &= state_finite[was_active]
            obs, rewards, dones, extras = env.step(applied_actions)
            step_finite = torch.isfinite(rewards)
            if hasattr(obs, "values"):
                for value in obs.values():
                    step_finite &= torch.isfinite(value).reshape(num_envs, -1).all(dim=-1)
            finite[was_active] &= step_finite[was_active]
            timeouts = extras.get("time_outs", torch.zeros_like(dones, dtype=torch.bool))
            failure_names = _failure_reasons(raw, dones.bool(), timeouts.bool())
            for index, reason in enumerate(failure_names):
                if reason is not None and bool(active[index]):
                    survived[index] = False
                    active[index] = False
                    reasons[index] = reason
            recorder.capture(
                scenario,
                phase,
                elapsed,
                int(active.sum().item()),
                applied_actions,
            )
            after = _state_metrics(raw, scenario.variant, scenario)
            return before, after, rewards, applied_actions

    settle_steps = int(round(scenario.settle_s / raw.step_dt))
    for step in range(settle_steps):
        advance("settle", (step + 1) * raw.step_dt)

    with torch.inference_mode():
        raw.set_wyw_command_override(
            ids, vx=scenario.vx, yaw=scenario.yaw, height=scenario.height
        )
        _patch_current_command_observation(obs, raw)
    sums = {
        "vx": torch.zeros(num_envs, device=raw.device),
        "yaw": torch.zeros(num_envs, device=raw.device),
        "tilt": torch.zeros(num_envs, device=raw.device),
        "height": torch.zeros(num_envs, device=raw.device),
        "leg_action_delta": torch.zeros(num_envs, device=raw.device),
        "wheel_action_delta": torch.zeros(num_envs, device=raw.device),
        "leg_action_second_diff": torch.zeros(num_envs, device=raw.device),
        "wheel_action_second_diff": torch.zeros(num_envs, device=raw.device),
    }
    action_counts = {
        "delta": torch.zeros(num_envs, device=raw.device),
        "second_diff": torch.zeros(num_envs, device=raw.device),
    }
    # Keep the command step out of the jitter score by starting action history
    # at the score-window boundary.
    action_steps = torch.zeros(num_envs, dtype=torch.long, device=raw.device)
    previous_score_actions = torch.zeros(num_envs, C.WYW_ACTION_DIM, device=raw.device)
    before_previous_score_actions = torch.zeros_like(previous_score_actions)
    tilt_peak = torch.zeros(num_envs, device=raw.device)
    counts = torch.zeros(num_envs, device=raw.device)
    completed = torch.zeros(num_envs, dtype=torch.bool, device=raw.device)

    no_contact_run = torch.zeros(num_envs, dtype=torch.long, device=raw.device)
    contact_run = torch.zeros_like(no_contact_run)
    jump_armed = torch.zeros(num_envs, dtype=torch.bool, device=raw.device)
    if scenario.variant == "jump":
        initial_forces = raw.contact_sensor.data.net_forces_w[:, raw._desired_contact_link_idx, 2]
        jump_armed = torch.all(
            initial_forces > float(raw.cfg.wyw_flight_contact_force), dim=1
        )
    in_air = torch.zeros(num_envs, dtype=torch.bool, device=raw.device)
    takeoff_z = torch.zeros(num_envs, device=raw.device)
    peak_z = torch.zeros(num_envs, device=raw.device)
    air_steps = torch.zeros(num_envs, dtype=torch.long, device=raw.device)
    landing_recovery = torch.zeros(num_envs, dtype=torch.long, device=raw.device)
    jump_heights: list[list[float]] = [[] for _ in range(num_envs)]
    airtimes: list[list[float]] = [[] for _ in range(num_envs)]
    landed = torch.zeros(num_envs, dtype=torch.long, device=raw.device)
    stable_landed = torch.zeros_like(landed)

    score_steps = int(round(scenario.score_s / raw.step_dt))
    for step in range(score_steps):
        scoring = active.clone()
        state, after_state, _, actions = advance("score", (step + 1) * raw.step_dt)
        sums["vx"][scoring] += torch.square(state["vx"][scoring] - scenario.vx)
        sums["yaw"][scoring] += torch.square(state["yaw"][scoring] - scenario.yaw)
        sums["tilt"][scoring] += torch.square(state["tilt"][scoring])
        sums["height"][scoring] += torch.square(state["height"][scoring] - scenario.height)
        tilt_peak[scoring] = torch.maximum(tilt_peak[scoring], state["tilt"][scoring])
        counts[scoring] += 1

        has_previous = scoring & (action_steps >= 1)
        leg_delta, wheel_delta, leg_second_diff, wheel_second_diff = (
            compute_fdu_action_differences(
                actions, previous_score_actions, before_previous_score_actions
            )
        )
        sums["leg_action_delta"][has_previous] += torch.square(
            leg_delta[has_previous]
        ).sum(dim=-1)
        sums["wheel_action_delta"][has_previous] += torch.square(
            wheel_delta[has_previous]
        ).sum(dim=-1)
        action_counts["delta"][has_previous] += 1

        has_two_previous = scoring & (action_steps >= 2)
        sums["leg_action_second_diff"][has_two_previous] += torch.square(
            leg_second_diff[has_two_previous]
        ).sum(dim=-1)
        sums["wheel_action_second_diff"][has_two_previous] += torch.square(
            wheel_second_diff[has_two_previous]
        ).sum(dim=-1)
        action_counts["second_diff"][has_two_previous] += 1

        before_previous_score_actions[scoring] = previous_score_actions[scoring]
        previous_score_actions[scoring] = actions[scoring]
        action_steps[scoring] += 1

        if scenario.variant == "rough":
            displacement = raw.robot.data.root_pos_w[:, :2] - starts_xy
            progress = torch.sum(displacement * heading, dim=-1)
            newly_complete = active & (progress >= float(scenario.target_distance_m))
            completed |= newly_complete
            active &= ~newly_complete

        if scenario.variant == "jump":
            forces = raw.contact_sensor.data.net_forces_w[:, raw._desired_contact_link_idx, 2]
            both_contact = torch.all(forces > float(raw.cfg.wyw_flight_contact_force), dim=1)
            no_contact_run = torch.where(~both_contact, no_contact_run + 1, torch.zeros_like(no_contact_run))
            contact_run = torch.where(both_contact, contact_run + 1, torch.zeros_like(contact_run))
            new_takeoff = active & jump_armed & ~in_air & (no_contact_run >= 2)
            takeoff_z[new_takeoff] = state["root_z"][new_takeoff]
            peak_z[new_takeoff] = after_state["root_z"][new_takeoff]
            air_steps[new_takeoff] = no_contact_run[new_takeoff]
            jump_armed[new_takeoff] = False
            continuing_air = in_air & ~both_contact
            air_steps[continuing_air] += 1
            in_air |= new_takeoff
            peak_z[in_air] = torch.maximum(peak_z[in_air], after_state["root_z"][in_air])
            new_landing = active & in_air & (contact_run >= 2)
            for index in torch.nonzero(new_landing).flatten().tolist():
                jump_heights[index].append(float((peak_z[index] - takeoff_z[index]).item()))
                airtimes[index].append(float(air_steps[index].item() * raw.step_dt))
            landed += new_landing.long()
            landing_recovery[new_landing] = int(round(1.0 / raw.step_dt))
            in_air &= ~new_landing
            jump_armed |= new_landing
            interrupted_recovery = (landing_recovery > 0) & ~both_contact
            landing_recovery[interrupted_recovery] = 0
            recovering = landing_recovery > 0
            landing_recovery[recovering] -= 1
            recovered = recovering & (landing_recovery == 0) & active & ~in_air
            stable_landed += recovered.long()

    samples: list[dict[str, Any]] = []
    for index in range(num_envs):
        count = float(counts[index].item())
        sample: dict[str, Any] = {
            "scenario_id": scenario.id,
            "env_id": index,
            "finite": bool(finite[index]),
            "failure_reason": reasons[index],
            "survival_rate": float(survived[index]),
        }
        if count > 0:
            sample.update(
                {
                    "vx_rmse_m_s": math.sqrt(float(sums["vx"][index].item()) / count),
                    "yaw_rmse_rad_s": math.sqrt(float(sums["yaw"][index].item()) / count),
                    "tilt_rms_deg": math.sqrt(float(sums["tilt"][index].item()) / count),
                    "tilt_peak_deg": float(tilt_peak[index].item()),
                    "height_rmse_m": math.sqrt(float(sums["height"][index].item()) / count),
                }
            )
        delta_count = float(action_counts["delta"][index].item())
        if delta_count > 0:
            sample.update(
                {
                    "leg_action_delta_rms": math.sqrt(
                        float(sums["leg_action_delta"][index].item())
                        / (delta_count * len(C.WYW_LEG_ACTION_IDS))
                    ),
                    "wheel_action_delta_rms": math.sqrt(
                        float(sums["wheel_action_delta"][index].item())
                        / (delta_count * len(C.WYW_WHEEL_ACTION_IDS))
                    ),
                }
            )
        second_diff_count = float(action_counts["second_diff"][index].item())
        if second_diff_count > 0:
            sample.update(
                {
                    "leg_action_second_diff_rms": math.sqrt(
                        float(sums["leg_action_second_diff"][index].item())
                        / (second_diff_count * len(C.WYW_LEG_ACTION_IDS))
                    ),
                    "wheel_action_second_diff_rms": math.sqrt(
                        float(sums["wheel_action_second_diff"][index].item())
                        / (second_diff_count * len(C.WYW_WHEEL_ACTION_IDS))
                    ),
                }
            )
        if scenario.variant == "rough":
            sample["completion_rate"] = float(completed[index])
        if scenario.variant == "jump":
            sample["jump_count"] = int(landed[index].item())
            sample["jump_height_gain_m"] = float(np.median(jump_heights[index])) if jump_heights[index] else 0.0
            sample["airtime_s"] = float(np.median(airtimes[index])) if airtimes[index] else 0.0
            sample["stable_landing_rate"] = (
                float(stable_landed[index].item() / landed[index].item())
                if landed[index].item() > 0
                else 0.0
            )
        samples.append(sample)
    summary = V.summarize_samples(samples)
    summary["scenario_id"] = scenario.id
    summary["scenario"] = scenario.to_dict()
    return samples, summary


def _run_worker() -> dict[str, Any]:
    if args_cli.checkpoint is None:
        raise ValueError("--checkpoint is required")
    checkpoint = args_cli.checkpoint.resolve()
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint does not exist: {checkpoint}")
    metadata = _validate_checkpoint_metadata(checkpoint, args_cli.variant)
    task_id, env_cfg = _configure_env(args_cli.variant, args_cli.profile)
    agent_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args_cli.device
    gym_env = gym.make(task_id, cfg=env_cfg, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicySequenceRunner(
        env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
    )
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    raw = env.unwrapped
    output_dir = (args_cli.output_dir or _default_output_dir()).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{args_cli.variant}_{args_cli.profile}.mp4"
    recorder = _VideoRecorder(gym_env, raw, video_path)
    num_envs = raw.num_envs
    all_samples = []
    summaries = []
    failed = False
    try:
        for index, scenario in enumerate(V.build_scenarios(args_cli.variant, args_cli.profile), start=1):
            print(
                f"[WYW Verify] {args_cli.profile} {index}: {scenario.id}", flush=True
            )
            samples, summary = _run_scenario(env, gym_env, raw, policy, recorder, scenario)
            all_samples.extend(samples)
            summaries.append(summary)
    except BaseException:
        failed = True
        raise
    finally:
        try:
            recorder.close(validate=not failed)
        finally:
            env.close()
    report = {
        "schema_version": V.VERIFY_SCHEMA_VERSION,
        "variant": args_cli.variant,
        "profile": args_cli.profile,
        "task_id": task_id,
        "seed": V.STANDARD_SEED,
        "num_envs": num_envs,
        "video_fps": VIDEO_FPS,
        "video_resolution": list(VIDEO_RESOLUTION),
        "simulated_duration_s": sum(
            scenario.settle_s + scenario.score_s
            for scenario in V.build_scenarios(args_cli.variant, args_cli.profile)
        ),
        "checkpoint": str(checkpoint),
        "metadata": metadata,
        "scenario_summaries": summaries,
        "samples": all_samples,
        "video": str(video_path),
    }
    V.validate_profile_report(report)
    (output_dir / f"profile_report.{args_cli.profile}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    print(
        f"[WYW Verify] starting {args_cli.variant}/{args_cli.profile} {args_cli.mode}",
        flush=True,
    )
    output_dir = (args_cli.output_dir or _default_output_dir()).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = _run_worker()
    if args_cli._worker:
        return 0
    return _finalize_reports(output_dir, [report])


def _close_simulation_app() -> None:
    try:
        simulation_app.close()
    except SystemExit:
        pass


if args_cli._worker or __name__ == "__main__":
    try:
        exit_code = main()
    except BaseException:
        error = traceback.format_exc()
        print(error, file=sys.stderr, flush=True)
        if args_cli.output_dir is not None:
            args_cli.output_dir.mkdir(parents=True, exist_ok=True)
            (args_cli.output_dir / "error.log").write_text(error, encoding="utf-8")
        _close_simulation_app()
        raise
    _close_simulation_app()
    raise SystemExit(exit_code)
