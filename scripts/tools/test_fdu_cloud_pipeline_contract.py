"""Static contracts for the WYW server training workflows."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess


ROOT = Path(__file__).parents[1]
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


def test_native_training_video_interval_tracks_checkpoint_interval():
    source = PIPELINE.read_text(encoding="utf-8")
    assert "DEFAULT_STEPS_PER_ITERATION=48" in source
    assert "checkpoint_interval * DEFAULT_STEPS_PER_ITERATION" in source
    assert '--save_interval="$checkpoint_interval"' in source
    assert '--video_length="$checkpoint_video_length"' in source
    assert '--video_interval="$checkpoint_video_interval"' in source


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
