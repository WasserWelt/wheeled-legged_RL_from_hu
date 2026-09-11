"""CPU checks for telemetry alignment, charts, and encoded evidence videos."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "presentation", ROOT / "scripts/rsl_rl/wyw_verify_presentation.py")
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


def frames(fps, duration=1):
    return [{"scenario_id": "flat_vx_p2", "profile": "nominal", "phase": "score",
             "elapsed": (i + 1) / fps,
             "time": (i + 1) / fps, "settle_s": 0, "failure": None,
             "command": {"vx": 2, "yaw": 0, "height": .22},
             "state": {"vx": i / fps, "yaw": 0, "height": .22, "tilt": 1},
             "tilt_limit": 10}
            for i in range(round(fps * duration))]


def test_metric_groups_cover_each_variant_metric_once():
    for variant in ("flat", "rough", "jump"):
        names = P.V.required_metrics(variant)
        grouped = [name for _, members in P._metric_groups(names) for name in members]
        assert set(grouped) == names
        assert len(grouped) == len(set(grouped))


def test_alignment_uses_scenario_phase_and_time():
    baseline, current = frames(15), frames(30)
    matched = P.match_frames(baseline, current)
    assert len(matched) == 30
    assert max(abs(current[i]["elapsed"] - baseline[b]["elapsed"])
               for i, b in enumerate(matched)) <= 1 / 30 + 1e-8
    with pytest.raises(ValueError, match="duration mismatch"):
        P.match_frames(baseline, frames(30, 2))
    current[0]["scenario_id"] = "another"
    with pytest.raises(ValueError, match="coverage differs"):
        P.match_frames(baseline, current)


def test_alignment_rejects_changed_commands_and_duplicate_times():
    baseline, current = frames(15), frames(15)
    current[0]["command"]["vx"] = 3
    with pytest.raises(ValueError, match="contract differs"):
        P.match_frames(baseline, current)
    current = frames(15)
    current[1]["elapsed"] = current[0]["elapsed"]
    with pytest.raises(ValueError, match="Non-monotonic"):
        P.match_frames(baseline, current)


def test_charts_and_robust_samples(tmp_path):
    metric = {"baseline": .1, "current": .2, "pass": False, "absolute_limit": .15}
    summary = {"scenario_id": "test", "failure_reasons": {"contact": 1}}
    comparison = {"safety_pass": False, "performance_pass": False, "profiles": {
        profile: {"passed": 0, "total": 1, "required": 1,
                  "scenarios": [{"scenario_id": "test", "metrics": {"vx_rmse_m_s": metric}}]}
        for profile in ("nominal", "robust")}}
    reports = [{"profile": "robust", "scenario_summaries": [summary],
                "samples": [{"scenario_id": "test", "vx_rmse_m_s": .1},
                            {"scenario_id": "test", "vx_rmse_m_s": .3}]}]
    paths = P.write_charts(tmp_path, variant="flat", status="FAIL", comparison=comparison,
                           reports=reports, baseline={})
    assert len(paths) == 4
    assert all(path.stat().st_size > 1000 for path in paths)
    assert (tmp_path / "charts.md").is_file()


@pytest.mark.parametrize("legacy", [False, True])
def test_video_is_nonblank_full_hd_and_preserves_duration(tmp_path, legacy):
    import cv2
    import imageio.v2 as imageio
    import numpy as np

    for name, fps in (("baseline", 15), ("current", 30)):
        rows = frames(fps, .4)
        rows[-1]["failure"] = "contact"
        (tmp_path / f"{name}.json").write_text(json.dumps({"fps": fps, "frames": rows}))
        with imageio.get_writer(str(tmp_path / f"{name}.mp4"), fps=fps, codec="libx264") as writer:
            for i in range(len(rows)):
                frame = np.full((128, 224, 3), (30, 130, 60), dtype=np.uint8)
                frame[30:100, 20 + i:70 + i] = (220, 30, 40)
                writer.append_data(frame)
    output = P.compose_video(tmp_path / "comparison.mp4", tmp_path / "baseline.mp4",
                             tmp_path / "current.mp4", tmp_path / "current.json",
                             None if legacy else tmp_path / "baseline.json")
    cap = cv2.VideoCapture(str(output))
    try:
        assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 12
        assert cap.get(cv2.CAP_PROP_FPS) == 30
        ok, frame = cap.read()
        assert ok and frame.shape == (1080, 1920, 3)
        assert frame[90:630].std() > 20
        assert frame[630:].std() > 10
        cv2.imwrite(str(tmp_path / "preview.png"), frame)
    finally:
        cap.release()
