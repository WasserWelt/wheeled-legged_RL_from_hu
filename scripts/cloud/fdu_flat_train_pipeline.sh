#!/usr/bin/env bash

# Reusable gpu_isaac workflow for WYW/FDU Flat training.
# The watch phase is intentionally separate so it survives SSH disconnects.

set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
DEFAULT_REPO="/root/gpufree-data/wheeled-legged_RL_from_hu"
DEFAULT_PYTHON="/opt/conda/envs/isaaclab/bin/python"
DEFAULT_TASK="Robotics-Wheelbipe-FDU-wyw-Flat-v1"
DEFAULT_PLAY_TASK="Robotics-Wheelbipe-FDU-wyw-Flat-Play-v1"
DEFAULT_NUM_ENVS=4096
DEFAULT_MAX_ITERATIONS=5000
DEFAULT_SEED=42
DEFAULT_VIDEO_LENGTH=1000
DEFAULT_SHUTDOWN_COMMAND="shutdown -h now"

usage() {
    cat <<'EOF'
Usage:
  fdu_flat_train_pipeline.sh start [options]
  fdu_flat_train_pipeline.sh watch [options]

start options:
  --repo PATH              Repository root on gpu_isaac
  --python PATH            Isaac Lab Python executable
  --task TASK              Training task id
  --num-envs N             Number of training environments (default: 4096)
  --max-iterations N       PPO iterations (default: 5000)
  --seed N                 Random seed (default: 42)
  --run-name NAME          Log directory suffix
  --shutdown-command CMD   Command run after a successful play (default: shutdown -h now)
  --no-shutdown            Keep the server on after post-training play

watch options:
  --repo PATH              Repository root on gpu_isaac
  --python PATH            Isaac Lab Python executable
  --train-pid PID          Training process to wait for
  --task TASK              Training task id
  --play-task TASK         Play task id
  --run-name NAME          Log directory suffix
  --video-length N         Number of play steps to record (default: 1000)
  --shutdown-command CMD   Command run after a successful play (default: shutdown -h now)
  --no-shutdown            Do not power off after a successful video
EOF
}

die() {
    echo "[FDU-PIPELINE] ERROR: $*" >&2
    exit 1
}

log() {
    echo "[FDU-PIPELINE] $(date -Is) $*"
}

latest_run_dir() {
    local run_root="$1"
    local run_name="$2"
    find "$run_root" -maxdepth 1 -type d -name "*_${run_name}" -printf "%T@ %p\n" \
        | sort -n | tail -1 | cut -d' ' -f2-
}

latest_checkpoint() {
    local run_dir="$1"
    find "$run_dir" -maxdepth 1 -type f -name "model_*.pt" -printf "%f\n" \
        | sort -V | tail -1
}

watch_pipeline() {
    local repo="$1"
    local python="$2"
    local train_pid="$3"
    local task="$4"
    local play_task="$5"
    local run_name="$6"
    local video_length="$7"
    local shutdown_after="$8"
    local shutdown_command="$9"
    local run_root="$repo/logs/rsl_rl/wheelbipe_fdu_wyw_flat_direct"
    local artifact_prefix="$repo/logs/cloud/${run_name}"
    local play_runtime_log="${artifact_prefix}.play_runtime.log"

    log "waiting for training pid=${train_pid}"
    while kill -0 "$train_pid" 2>/dev/null; do
        sleep 60
    done
    log "training pid=${train_pid} exited"
    sleep 10

    local run_dir
    run_dir="$(latest_run_dir "$run_root" "$run_name")"
    [[ -n "$run_dir" && -d "$run_dir" ]] || die "run directory not found for ${run_name}"

    local checkpoint
    checkpoint="$(latest_checkpoint "$run_dir")"
    [[ -n "$checkpoint" && -f "$run_dir/$checkpoint" ]] || die "checkpoint not found in $run_dir"
    log "run_dir=${run_dir}"
    log "checkpoint=${run_dir}/${checkpoint}"

    export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
    export OMNI_KIT_ACCEPT_EULA=YES
    set +e
    "$python" -u "$repo/scripts/rsl_rl/play.py" \
        --task="$play_task" \
        --num_envs=1 \
        --checkpoint="$run_dir/$checkpoint" \
        --device=cuda:0 \
        --headless \
        --video \
        --video_length="$video_length" \
        --max_steps="$video_length" > "$play_runtime_log" 2>&1
    local play_status=$?
    set -e
    log "play_exit=${play_status}"
    [[ "$play_status" -eq 0 ]] || die "post-training play failed; server will remain on"

    local video_path
    video_path="$(find "$run_dir/videos/play" -type f -name "*.mp4" -size +0c -printf "%T@ %p\n" \
        | sort -n | tail -1 | cut -d' ' -f2-)"
    [[ -n "$video_path" && -s "$video_path" ]] || die "play returned success but no non-empty MP4 was found"
    printf '%s\n' "$video_path" > "${artifact_prefix}.play_video.txt"
    touch "${artifact_prefix}.play.complete"
    log "video=${video_path}"

    if [[ "$shutdown_after" == "1" ]]; then
        touch "${artifact_prefix}.shutdown_requested"
        log "post-training play succeeded; requesting server shutdown"
        local shutdown_log="${artifact_prefix}.shutdown_command.log"
        local shutdown_output
        set +e
        shutdown_output="$(bash -lc "$shutdown_command" 2>&1)"
        local shutdown_status=$?
        set -e
        printf '%s\n' "$shutdown_output" > "$shutdown_log"
        [[ -z "$shutdown_output" ]] || log "shutdown_output=${shutdown_output//$'\n'/; }"
        if [[ "$shutdown_status" -ne 0 ]]; then
            touch "${artifact_prefix}.shutdown_failed"
            die "shutdown command failed (status=${shutdown_status}); server will remain on"
        fi
        if grep -Eiq '未设置关机密钥|shutdown key.*(not set|missing)|key.*(not set|missing)' "$shutdown_log"; then
            touch "${artifact_prefix}.shutdown_failed"
            die "shutdown command reported a missing key; server will remain on"
        fi
        touch "${artifact_prefix}.shutdown_accepted"
    else
        log "shutdown disabled by --no-shutdown"
    fi
}

start_pipeline() {
    local repo="$DEFAULT_REPO"
    local python="$DEFAULT_PYTHON"
    local task="$DEFAULT_TASK"
    local num_envs="$DEFAULT_NUM_ENVS"
    local max_iterations="$DEFAULT_MAX_ITERATIONS"
    local seed="$DEFAULT_SEED"
    local run_name="flat_500hz_height015_030_4096_iter5000"
    local shutdown_command="$DEFAULT_SHUTDOWN_COMMAND"
    local shutdown_after=1

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --repo) repo="$2"; shift 2 ;;
            --python) python="$2"; shift 2 ;;
            --task) task="$2"; shift 2 ;;
            --num-envs) num_envs="$2"; shift 2 ;;
            --max-iterations) max_iterations="$2"; shift 2 ;;
            --seed) seed="$2"; shift 2 ;;
            --run-name) run_name="$2"; shift 2 ;;
            --shutdown-command) shutdown_command="$2"; shift 2 ;;
            --no-shutdown) shutdown_after=0; shift ;;
            -h|--help) usage; return 0 ;;
            *) die "unknown start option: $1" ;;
        esac
    done

    [[ -d "$repo" ]] || die "repository does not exist: $repo"
    [[ -x "$python" ]] || die "Isaac Lab Python does not exist: $python"
    [[ "$run_name" != *[[:space:]]* ]] || die "run name must not contain whitespace"
    mkdir -p "$repo/logs/cloud"

    local train_log="$repo/logs/cloud/${run_name}.train.log"
    local watch_log="$repo/logs/cloud/${run_name}.post_play.log"
    log "starting task=${task} envs=${num_envs} iterations=${max_iterations} seed=${seed}"
    nohup env PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}" OMNI_KIT_ACCEPT_EULA=YES \
        "$python" -u "$repo/scripts/rsl_rl/train.py" \
        --task="$task" \
        --num_envs="$num_envs" \
        --max_iterations="$max_iterations" \
        --seed="$seed" \
        --device=cuda:0 \
        --headless \
        --run_name="$run_name" > "$train_log" 2>&1 < /dev/null &
    local train_pid=$!
    printf '%s\n' "$train_pid" > "$repo/logs/cloud/${run_name}.train.pid"
    log "train_pid=${train_pid}"
    log "train_log=${train_log}"

    local watch_args=(
        watch
        --repo "$repo" \
        --python "$python" \
        --train-pid "$train_pid" \
        --task "$task" \
        --play-task "$DEFAULT_PLAY_TASK" \
        --run-name "$run_name" \
        --video-length "$DEFAULT_VIDEO_LENGTH" \
        --shutdown-command "$shutdown_command"
    )
    [[ "$shutdown_after" == "1" ]] || watch_args+=(--no-shutdown)
    nohup "$SCRIPT_PATH" "${watch_args[@]}" > "$watch_log" 2>&1 < /dev/null &
    local watcher_pid=$!
    printf '%s\n' "$watcher_pid" > "$repo/logs/cloud/${run_name}.watch.pid"
    log "watcher_pid=${watcher_pid}"
    log "watch_log=${watch_log}"
}

watch_command() {
    local repo="$DEFAULT_REPO"
    local python="$DEFAULT_PYTHON"
    local train_pid=""
    local task="$DEFAULT_TASK"
    local play_task="$DEFAULT_PLAY_TASK"
    local run_name="flat_500hz_height015_030_4096_iter5000"
    local video_length="$DEFAULT_VIDEO_LENGTH"
    local shutdown_command="$DEFAULT_SHUTDOWN_COMMAND"
    local shutdown_after=1

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --repo) repo="$2"; shift 2 ;;
            --python) python="$2"; shift 2 ;;
            --train-pid) train_pid="$2"; shift 2 ;;
            --task) task="$2"; shift 2 ;;
            --play-task) play_task="$2"; shift 2 ;;
            --run-name) run_name="$2"; shift 2 ;;
            --video-length) video_length="$2"; shift 2 ;;
            --shutdown-command) shutdown_command="$2"; shift 2 ;;
            --no-shutdown) shutdown_after=0; shift ;;
            -h|--help) usage; return 0 ;;
            *) die "unknown watch option: $1" ;;
        esac
    done

    [[ -n "$train_pid" ]] || die "--train-pid is required for watch"
    watch_pipeline "$repo" "$python" "$train_pid" "$task" "$play_task" "$run_name" "$video_length" "$shutdown_after" "$shutdown_command"
}

command="${1:-}"
shift || true
case "$command" in
    start) start_pipeline "$@" ;;
    watch) watch_command "$@" ;;
    -h|--help|"") usage ;;
    *) die "expected start or watch (got: ${command})" ;;
esac
