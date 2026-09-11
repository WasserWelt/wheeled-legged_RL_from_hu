"""Static contracts for the WYW server training workflows."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).parents[2]
PIPELINE = ROOT / "scripts/cloud/fdu_flat_train_pipeline.sh"
SETUP = ROOT / "scripts/cloud/lab_3090_server_setup.sh"
WORKFLOW_DOCS = (
    ROOT / "docs/cloud_gpu_isaac_workflow.md",
    ROOT / "docs/lab_3090_server_workflow.md",
)


def test_server_scripts_have_valid_bash_syntax():
    for path in (PIPELINE, SETUP):
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_lab_profile_separates_source_and_training_data():
    source = PIPELINE.read_text(encoding="utf-8")
    assert 'LAB_3090_REPO="/home/wyw/wheeled-legged_RL_from_hu"' in source
    assert 'LAB_3090_DATA_ROOT="/data1/wyw/wheeled-legged_RL_from_hu"' in source
    assert 'LAB_3090_PYTHON="/home/wyw/conda_envs/isaaclab_2/bin/python"' in source
    assert '--log_root="$data_root/logs/rsl_rl"' in source


def test_pipeline_checks_and_scopes_the_selected_gpu():
    source = PIPELINE.read_text(encoding="utf-8")
    assert "--query-compute-apps=pid,process_name,used_gpu_memory" in source
    assert 'CUDA_VISIBLE_DEVICES="$gpu"' in source
    assert 'gpu="$(resolve_gpu "$gpu")"' in source
    assert 'require_free_gpu "$gpu"' in source


def run_auto_gpu_start(tmp_path, *, gpu1_process):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_nvidia_smi = bin_dir / "nvidia-smi"
    fake_nvidia_smi.write_text(
        """#!/usr/bin/env bash
case "$*" in
  *--query-gpu=index,name,memory.used,memory.total,utilization.gpu*)
    printf '%s\\n' '0, NVIDIA RTX 3090, 8, 24576, 0' '1, NVIDIA RTX 3090, 12000, 24576, 91'
    ;;
  *--query-gpu=index*)
    printf '%s\\n' 0 1
    ;;
  *--id=0*--query-gpu=memory.used*)
    printf '%s\\n' 8
    ;;
  *--id=1*--query-gpu=memory.used*)
    printf '%s\\n' 12000
    ;;
  *--id=0*)
    ;;
  *--id=1*)
    if [[ -n "${FAKE_GPU1_PROCESS:-}" ]]; then
      printf '%s\\n' '424242, python, 11900'
    fi
    ;;
  *)
    exit 2
    ;;
esac
""",
        encoding="utf-8",
    )
    fake_nvidia_smi.chmod(0o755)

    repo = tmp_path / "repo"
    repo.mkdir()
    data_root = tmp_path / "data"
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    if gpu1_process:
        env["FAKE_GPU1_PROCESS"] = "1"
    result = subprocess.run(
        [
            "bash",
            str(PIPELINE),
            "start",
            "--repo",
            str(repo),
            "--data-root",
            str(data_root),
            "--python",
            "/bin/true",
            "--gpu",
            "auto",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    return result, data_root


def test_auto_gpu_reports_status_and_rejects_if_any_gpu_is_busy(tmp_path):
    result, data_root = run_auto_gpu_start(tmp_path, gpu1_process=True)

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "GPU 0: NVIDIA RTX 3090, memory 8/24576 MiB, utilization 0%" in output
    assert "GPU 1: NVIDIA RTX 3090, memory 12000/24576 MiB, utilization 91%" in output
    assert "GPU 1: PID 424242, user unknown, memory 11900 MiB, command python" in output
    assert "requires all 2 GPUs to be idle" in output
    assert not data_root.exists()


def test_auto_gpu_rejects_hidden_memory_use_without_a_compute_process(tmp_path):
    result, data_root = run_auto_gpu_start(tmp_path, gpu1_process=False)

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "active GPU compute processes" in output
    assert "  none" in output
    assert "process or memory use found on GPU(s): 1" in output
    assert not data_root.exists()


def test_native_training_video_interval_tracks_checkpoint_interval():
    source = PIPELINE.read_text(encoding="utf-8")
    assert "DEFAULT_STEPS_PER_ITERATION=48" in source
    assert "checkpoint_interval * DEFAULT_STEPS_PER_ITERATION" in source
    assert '--save_interval="$checkpoint_interval"' in source
    assert '--video_length="$checkpoint_video_length"' in source
    assert '--video_interval="$checkpoint_video_interval"' in source


def test_post_training_play_task_tracks_training_variant():
    source = PIPELINE.read_text(encoding="utf-8")
    for variant in ("Flat", "Rough", "Jump"):
        assert (
            f"Robotics-Wheelbipe-FDU-wyw-{variant}-v1) "
            f"printf '%s\\n' \"Robotics-Wheelbipe-FDU-wyw-{variant}-Play-v1\""
        ) in source
    assert 'play_task="$(play_task_for_training_task "$task")"' in source
    assert '--play-task "$play_task"' in source


def test_pipeline_forwards_optional_checkpoint_training_modes():
    source = PIPELINE.read_text(encoding="utf-8")
    assert '--checkpoint) checkpoint="$2"; shift 2' in source
    assert '--resume-training|--resume_training) resume_training=1; shift' in source
    assert 'checkpoint_args+=(--checkpoint "$checkpoint")' in source
    assert 'checkpoint_args+=(--resume_training)' in source
    assert '"${checkpoint_args[@]}"' in source
    assert '--resume-training requires --checkpoint PATH' in source


def test_resume_training_without_checkpoint_is_rejected_before_gpu_access(tmp_path):
    result = subprocess.run(
        [
            "/bin/bash",
            str(PIPELINE),
            "start",
            "--repo",
            str(tmp_path),
            "--python",
            "/bin/true",
            "--resume-training",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--resume-training requires --checkpoint PATH" in result.stderr
    assert "GPU status" not in result.stdout


def test_post_training_run_root_tracks_training_variant():
    source = PIPELINE.read_text(encoding="utf-8")
    for variant in ("flat", "rough", "jump"):
        assert f'"wheelbipe_fdu_wyw_{variant}_direct"' in source
    assert 'experiment_name="$(experiment_name_for_task "$play_task")"' in source
    assert 'local run_root="$data_root/logs/rsl_rl/$experiment_name"' in source


def test_pipeline_has_no_post_training_machine_shutdown_contract():
    sources = [PIPELINE.read_text(encoding="utf-8")]
    sources.extend(path.read_text(encoding="utf-8") for path in WORKFLOW_DOCS)
    combined = "\n".join(sources).lower()
    for forbidden in ("shutdown", "poweroff", "halt", "no-shutdown", "关机"):
        assert forbidden not in combined


def test_lab_environment_and_source_stay_outside_data_disk():
    pipeline = PIPELINE.read_text(encoding="utf-8")
    setup = SETUP.read_text(encoding="utf-8")
    assert 'REMOTE_ENV="/home/wyw/conda_envs/isaaclab_2"' in setup
    assert 'REMOTE_ARCHIVE_DIR="/home/wyw/env_archives"' in setup
    assert '/data1/wyw/conda_envs' not in pipeline
    assert '/data1/wyw/conda_envs' not in setup
    assert '/data1/wyw/env_archives' not in setup


def test_lab_setup_has_read_only_content_sync_check():
    source = SETUP.read_text(encoding="utf-8")
    assert "check-sync) check_repository_sync" in source
    assert "rsync -rlnc --delete --itemize-changes" in source
    assert '"${RSYNC_SOURCE_FILTERS[@]}"' in source
    assert "--exclude '/docs/'" in source
    assert "--exclude '/pretrained/'" in source
    assert "source content is synchronized" in source


def test_train_cli_supports_external_log_root():
    path = ROOT / "scripts/rsl_rl/train.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        arg.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        for arg in node.args
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
    }
    assert "--log_root" in names
    assert "--log-root" in names
