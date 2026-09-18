"""Run WYW verification for the same checkpoint selection across log directories."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
VERIFY_SCRIPT = REPO_ROOT / "scripts/rsl_rl/play_wyw_verify.py"
PRESENTATION_SCRIPT = REPO_ROOT / "scripts/rsl_rl/wyw_verify_presentation.py"
EXPECTED_EXPERIMENTS = {
    "flat": "wheelbipe_fdu_wyw_flat_direct",
    "rough": "wheelbipe_fdu_wyw_rough_direct",
    "jump": "wheelbipe_fdu_wyw_jump_direct",
}
RESERVED_VERIFY_ARGS = {
    "--variant", "--checkpoint", "--mode", "--profile", "--baseline-config",
    "--output-dir", "--device", "--headless",
}


def _load_presentation():
    spec = importlib.util.spec_from_file_location("_wyw_verify_presentation", PRESENTATION_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load presentation module: {PRESENTATION_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


P = _load_presentation()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(EXPECTED_EXPERIMENTS), default="flat")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--iteration", type=int, help="Select model_<iteration>.pt in every run")
    selection.add_argument("--checkpoint-name", help="Select this exact checkpoint basename")
    selection.add_argument("--latest", action="store_true", help="Select the highest model_N.pt in every run")
    parser.add_argument("--logs-root", type=Path, default=REPO_ROOT / "logs/rsl_rl")
    parser.add_argument("--run-glob", default="*", help="Only inspect run directories matching this glob")
    parser.add_argument("--baseline-config", type=Path,
                        default=REPO_ROOT / "configs/wyw_verify_baseline.json")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-name", default="evaluate",
                        help="Folder below acceptance/<variant> for each result")
    parser.add_argument("--show-window", action="store_true", help="Do not pass --headless")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Reuse a complete report for the exact checkpoint")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Redraw charts from complete existing reports without sampling or launching Isaac Sim",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first execution error")
    parser.add_argument("--dry-run", action="store_true", help="List commands without running Isaac Sim")
    parser.add_argument(
        "verify_args",
        nargs=argparse.REMAINDER,
        help="Extra play_wyw_verify.py arguments after --",
    )
    return parser


def _selection_label(args: argparse.Namespace) -> str:
    if args.iteration is not None:
        if args.iteration < 0:
            raise ValueError("--iteration must be non-negative")
        return f"model_{args.iteration}.pt"
    if args.checkpoint_name is not None:
        name = Path(args.checkpoint_name)
        if name.name != args.checkpoint_name or not args.checkpoint_name.endswith(".pt"):
            raise ValueError("--checkpoint-name must be a .pt basename, not a path")
        return args.checkpoint_name
    return "latest"


def _experiment_name(run_dir: Path) -> str | None:
    path = run_dir / "params/agent.yaml"
    if not path.is_file():
        return None
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    except (OSError, yaml.YAMLError):
        return None
    return data.get("experiment_name") if isinstance(data, dict) else None


def _latest_checkpoint(run_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in run_dir.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if match and path.is_file():
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def discover_checkpoints(
    logs_root: Path,
    *,
    run_glob: str,
    variant: str,
    checkpoint_name: str,
) -> tuple[list[Path], list[dict[str, str]]]:
    """Return matching WYW checkpoints and explicit reasons for skipped runs."""
    expected_experiment = EXPECTED_EXPERIMENTS[variant]
    checkpoints: list[Path] = []
    skipped: list[dict[str, str]] = []
    for run_dir in sorted(path for path in logs_root.glob(run_glob) if path.is_dir()):
        experiment = _experiment_name(run_dir)
        if experiment != expected_experiment:
            reason = "missing params/agent.yaml" if experiment is None else f"experiment={experiment}"
            skipped.append({"run": str(run_dir), "reason": reason})
            continue
        checkpoint = (_latest_checkpoint(run_dir) if checkpoint_name == "latest"
                      else run_dir / checkpoint_name)
        if checkpoint is None or not checkpoint.is_file():
            skipped.append({"run": str(run_dir), "reason": f"checkpoint {checkpoint_name} not found"})
            continue
        checkpoints.append(checkpoint.resolve())
    return checkpoints, skipped


def _output_dir(checkpoint: Path, variant: str, output_name: str) -> Path:
    name = Path(output_name)
    if name.name != output_name or output_name in {"", ".", ".."}:
        raise ValueError("--output-name must be one directory name")
    return checkpoint.parent / "acceptance" / variant / output_name


def _validate_baseline_artifacts(settings: dict[str, Any], variant: str) -> bool:
    """Validate shared baseline inputs once; return whether synchronized telemetry exists."""
    package_dir = Path(settings.get("package_dir", ""))
    if not package_dir.is_absolute():
        package_dir = (REPO_ROOT / package_dir).resolve()
    package_path = package_dir / "baseline.json"
    if not package_path.is_file():
        raise ValueError(f"baseline package does not exist: {package_path}")
    try:
        package = json.loads(package_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read baseline package: {package_path}") from exc
    if package.get("variant") != variant:
        raise ValueError(
            f"baseline package variant is {package.get('variant')!r}, expected {variant!r}"
        )
    video = Path(package.get("video", ""))
    if not video.is_absolute():
        video = package_dir / video
    if not video.is_file():
        raise ValueError(f"baseline video does not exist: {video}")
    trace = Path(package.get("trace", "trace.json"))
    if not trace.is_absolute():
        trace = package_dir / trace
    return trace.is_file()


def _existing_status(output_dir: Path, checkpoint: Path) -> str | None:
    report_path = output_dir / "report.json"
    if not report_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        recorded = Path(report["checkpoint"]).resolve()
        video = Path(report["video"])
        chart = Path(report["chart"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if recorded != checkpoint.resolve() or not video.is_file() or not chart.is_file():
        return None
    return report.get("status") if report.get("status") in {"PASS", "FAIL"} else None


def _command(args: argparse.Namespace, checkpoint: Path, output_dir: Path) -> list[str]:
    command = [
        str(args.python),
        str(VERIFY_SCRIPT),
        "--variant", args.variant,
        "--checkpoint", str(checkpoint),
        "--mode", "evaluate",
        "--profile", "all",
        "--baseline-config", str(args.baseline_config.resolve()),
        "--output-dir", str(output_dir),
        "--device", args.device,
    ]
    if not args.show_window:
        command.append("--headless")
    extras = list(args.verify_args)
    if extras[:1] == ["--"]:
        extras = extras[1:]
    conflicts = sorted({token.split("=", 1)[0] for token in extras} & RESERVED_VERIFY_ARGS)
    if conflicts:
        raise ValueError(
            "batch-controlled arguments cannot follow --: " + ", ".join(conflicts)
        )
    return [*command, *extras]


def _plot_command(
    args: argparse.Namespace, report_path: Path, baseline_path: Path
) -> list[str]:
    return [
        str(args.python),
        str(PRESENTATION_SCRIPT),
        "--report", str(report_path),
        "--baseline", str(baseline_path),
    ]


def _run_command(command: list[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write(f"command: {shlex.join(command)}\n\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
        return process.wait()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        checkpoint_name = _selection_label(args)
        logs_root = args.logs_root.resolve()
        if not logs_root.is_dir():
            raise ValueError(f"logs root does not exist: {logs_root}")
        if not args.python.is_file():
            raise ValueError(f"Python executable does not exist: {args.python}")
        _output_dir(logs_root / "placeholder.pt", args.variant, args.output_name)
        if args.plot_only:
            if args.verify_args:
                raise ValueError("extra play_wyw_verify.py arguments cannot be used with --plot-only")
            baseline_has_trace = True
        else:
            baseline_config = args.baseline_config.resolve()
            if not baseline_config.is_file():
                raise ValueError(f"baseline config does not exist: {baseline_config}")
            try:
                baseline_settings = json.loads(baseline_config.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"cannot read baseline config: {baseline_config}") from exc
            if baseline_settings.get("variant") != args.variant:
                raise ValueError(
                    f"baseline config variant is {baseline_settings.get('variant')!r}, "
                    f"expected {args.variant!r}"
                )
            baseline_has_trace = _validate_baseline_artifacts(baseline_settings, args.variant)
            _command(args, logs_root / "placeholder.pt", logs_root / "placeholder-output")
        checkpoints, skipped = discover_checkpoints(
            logs_root,
            run_glob=args.run_glob,
            variant=args.variant,
            checkpoint_name=checkpoint_name,
        )
    except ValueError as exc:
        print(f"[WYW Batch] ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"[WYW Batch] variant={args.variant} selection={checkpoint_name}")
    print(f"[WYW Batch] found={len(checkpoints)} skipped={len(skipped)} under {logs_root}")
    if not baseline_has_trace:
        print("[WYW Batch] WARNING: baseline trace.json is missing; comparison videos use "
              "time-only alignment. Regenerate the baseline for scenario-level synchronization.")
    for item in skipped:
        print(f"[WYW Batch] SKIP {item['run']}: {item['reason']}")
    if not checkpoints:
        print("[WYW Batch] ERROR: no matching checkpoints", file=sys.stderr)
        return 2

    plot_jobs: dict[Path, tuple[str, Path, Path]] = {}
    if args.plot_only:
        errors = []
        for checkpoint in checkpoints:
            output_dir = _output_dir(checkpoint, args.variant, args.output_name)
            report_path = output_dir / "report.json"
            try:
                report, _, baseline_path = P.load_plot_inputs(
                    report_path,
                    expected_checkpoint=checkpoint,
                    expected_variant=args.variant,
                )
                plot_jobs[checkpoint] = (report["status"], report_path, baseline_path)
            except (OSError, KeyError, TypeError, ValueError) as exc:
                errors.append((checkpoint, str(exc)))
        if errors:
            print(
                "[WYW Batch] ERROR: --plot-only requires complete sampled data for every "
                "selected checkpoint; nothing was redrawn",
                file=sys.stderr,
            )
            for checkpoint, detail in errors:
                print(f"[WYW Batch] ERROR {checkpoint}: {detail}", file=sys.stderr)
            return 2

    if args.dry_run:
        for checkpoint in checkpoints:
            output_dir = _output_dir(checkpoint, args.variant, args.output_name)
            if args.plot_only:
                _, report_path, baseline_path = plot_jobs[checkpoint]
                print(f"[WYW Batch] REDRAW {checkpoint}")
                print(shlex.join(_plot_command(args, report_path, baseline_path)))
            else:
                print(f"[WYW Batch] RUN {checkpoint}")
                print(shlex.join(_command(args, checkpoint, output_dir)))
        return 0

    statuses: list[str] = []

    for index, checkpoint in enumerate(checkpoints, start=1):
        output_dir = _output_dir(checkpoint, args.variant, args.output_name)
        if args.plot_only:
            report_status, report_path, baseline_path = plot_jobs[checkpoint]
            log_path = output_dir / "batch_redraw.log"
            command = _plot_command(args, report_path, baseline_path)
            print(f"[WYW Batch] [{index}/{len(checkpoints)}] REDRAW {checkpoint}")
            started = time.monotonic()
            try:
                returncode = _run_command(command, log_path)
            except OSError as exc:
                returncode = None
                with log_path.open("a", encoding="utf-8") as stream:
                    stream.write(f"\nlauncher error: {exc}\n")
            duration = time.monotonic() - started
            status = report_status if returncode == 0 else "ERROR"
            print(f"[WYW Batch] [{index}/{len(checkpoints)}] {status} "
                  f"rc={returncode} duration={duration:.1f}s")
            statuses.append(status)
            if status == "ERROR" and args.fail_fast:
                break
            continue

        existing = _existing_status(output_dir, checkpoint) if args.skip_existing else None
        if existing is not None:
            print(f"[WYW Batch] [{index}/{len(checkpoints)}] {existing} existing {checkpoint}")
            statuses.append(existing)
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "batch_driver.log"
        command = _command(args, checkpoint, output_dir)
        print(f"[WYW Batch] [{index}/{len(checkpoints)}] RUN {checkpoint}")
        report_path = output_dir / "report.json"
        report_signature = (
            (report_path.stat().st_mtime_ns, report_path.stat().st_size)
            if report_path.is_file() else None
        )
        started = time.monotonic()
        execution_error = None
        try:
            returncode = _run_command(command, log_path)
        except OSError as exc:
            execution_error = str(exc)
            returncode = None
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(f"\nlauncher error: {execution_error}\n")
        duration = time.monotonic() - started
        new_signature = (
            (report_path.stat().st_mtime_ns, report_path.stat().st_size)
            if report_path.is_file() else None
        )
        status = None
        if execution_error is None and new_signature is not None and new_signature != report_signature:
            status = _existing_status(output_dir, checkpoint)
        expected_returncode = {"PASS": 0, "FAIL": 1}.get(status)
        if status is not None and returncode != expected_returncode:
            execution_error = (
                f"report status {status} is inconsistent with child return code {returncode}"
            )
            status = None
        if status is None:
            status = "ERROR"
        print(f"[WYW Batch] [{index}/{len(checkpoints)}] {status} "
              f"rc={returncode} duration={duration:.1f}s")
        statuses.append(status)
        if status == "ERROR" and args.fail_fast:
            break

    if "ERROR" in statuses:
        return 2
    if "FAIL" in statuses:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
