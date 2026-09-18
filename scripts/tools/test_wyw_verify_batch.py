"""CPU-only tests for WYW batch checkpoint discovery and command construction."""

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "wyw_verify_batch", ROOT / "scripts/rsl_rl/run_wyw_verify_batch.py"
)
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)


def _run(tmp_path, name, experiment="wheelbipe_fdu_wyw_flat_direct"):
    run = tmp_path / name
    (run / "params").mkdir(parents=True)
    (run / "params/agent.yaml").write_text(
        f"experiment_name: {experiment}\n", encoding="utf-8"
    )
    return run


def test_discovery_supports_exact_and_latest_checkpoint(tmp_path):
    first = _run(tmp_path, "first")
    second = _run(tmp_path, "second")
    other = _run(tmp_path, "rough", "wheelbipe_fdu_wyw_rough_direct")
    for run in (first, second, other):
        (run / "model_200.pt").touch()
    (first / "model_1000.pt").touch()

    exact, skipped = B.discover_checkpoints(
        tmp_path, run_glob="*", variant="flat", checkpoint_name="model_200.pt"
    )
    assert exact == [(first / "model_200.pt").resolve(), (second / "model_200.pt").resolve()]
    assert any("experiment=" in item["reason"] for item in skipped)

    latest, _ = B.discover_checkpoints(
        tmp_path, run_glob="first", variant="flat", checkpoint_name="latest"
    )
    assert latest == [(first / "model_1000.pt").resolve()]


def test_existing_result_must_match_checkpoint_and_artifacts(tmp_path):
    checkpoint = tmp_path / "model_10.pt"
    checkpoint.touch()
    output = tmp_path / "acceptance"
    output.mkdir()
    video, chart = output / "comparison.mp4", output / "comparison.png"
    video.touch()
    chart.touch()
    report = {"checkpoint": str(checkpoint), "video": str(video), "chart": str(chart),
              "status": "FAIL"}
    (output / "report.json").write_text(json.dumps(report), encoding="utf-8")
    assert B._existing_status(output, checkpoint) == "FAIL"
    checkpoint.rename(tmp_path / "model_11.pt")
    assert B._existing_status(output, tmp_path / "model_11.pt") is None


def test_dry_run_builds_one_command_per_matching_run(tmp_path, capsys):
    run = _run(tmp_path, "flat_run")
    (run / "model_4999.pt").touch()
    package = tmp_path / "baseline"
    package.mkdir()
    (package / "baseline.mp4").touch()
    (package / "baseline.json").write_text(json.dumps({
        "variant": "flat", "video": "baseline.mp4", "trace": "trace.json"
    }), encoding="utf-8")
    (package / "trace.json").touch()
    config = tmp_path / "baseline_config.json"
    config.write_text(json.dumps({"variant": "flat", "package_dir": str(package)}),
                      encoding="utf-8")
    result = B.main([
        "--logs-root", str(tmp_path),
        "--iteration", "4999",
        "--baseline-config", str(config),
        "--python", str(Path(B.sys.executable)),
        "--dry-run",
        "--",
        "--rendering_mode", "balanced",
    ])
    output = capsys.readouterr().out
    assert result == 0
    assert "model_4999.pt" in output
    assert "--headless" in output
    assert "--rendering_mode balanced" in output


def test_batch_controlled_arguments_cannot_be_overridden_after_separator(tmp_path):
    args = B._parser().parse_args(["--iteration", "10", "--", "--profile", "nominal"])
    try:
        B._command(args, tmp_path / "model_10.pt", tmp_path / "output")
    except ValueError as exc:
        assert "--profile" in str(exc)
    else:
        raise AssertionError("expected conflicting child argument to be rejected")


def test_plot_only_dry_run_uses_presentation_without_baseline_config(
    tmp_path, capsys, monkeypatch
):
    run = _run(tmp_path, "flat_run")
    checkpoint = run / "model_4999.pt"
    checkpoint.touch()
    baseline = tmp_path / "baseline.json"

    monkeypatch.setattr(
        B.P,
        "load_plot_inputs",
        lambda *args, **kwargs: ({"status": "PASS"}, {}, baseline),
    )
    result = B.main([
        "--logs-root", str(tmp_path),
        "--iteration", "4999",
        "--python", str(Path(B.sys.executable)),
        "--plot-only",
        "--dry-run",
    ])

    output = capsys.readouterr().out
    assert result == 0
    assert "REDRAW" in output
    assert str(B.PRESENTATION_SCRIPT) in output
    assert str(B.VERIFY_SCRIPT) not in output
    assert "--baseline" in output


def test_plot_only_validates_every_report_before_redrawing(
    tmp_path, capsys, monkeypatch
):
    first = _run(tmp_path, "first")
    second = _run(tmp_path, "second")
    for run in (first, second):
        (run / "model_10.pt").touch()

    def load_plot_inputs(report_path, **kwargs):
        if "second" in str(report_path):
            raise ValueError("robust samples are missing")
        return {"status": "PASS"}, {}, tmp_path / "baseline.json"

    executed = []
    monkeypatch.setattr(B.P, "load_plot_inputs", load_plot_inputs)
    monkeypatch.setattr(B, "_run_command", lambda *args: executed.append(args) or 0)
    result = B.main([
        "--logs-root", str(tmp_path),
        "--iteration", "10",
        "--python", str(Path(B.sys.executable)),
        "--plot-only",
    ])

    captured = capsys.readouterr()
    assert result == 2
    assert not executed
    assert "nothing was redrawn" in captured.err
    assert "robust samples are missing" in captured.err
