#!/usr/bin/env bash

# Reusable server workflow for WYW/FDU Flat training.
# The watch phase is intentionally separate so it survives SSH disconnects.

set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
DEFAULT_REPO="/root/gpufree-data/wheeled-legged_RL_from_hu"
DEFAULT_PYTHON="/opt/conda/envs/isaaclab/bin/python"
LAB_3090_REPO="/home/wyw/wheeled-legged_RL_from_hu"
LAB_3090_DATA_ROOT="/data1/wyw/wheeled-legged_RL_from_hu"
LAB_3090_PYTHON="/home/wyw/conda_envs/isaaclab_2/bin/python"
DEFAULT_TASK="Robotics-Wheelbipe-FDU-wyw-Flat-v1"
DEFAULT_PLAY_TASK="Robotics-Wheelbipe-FDU-wyw-Flat-Play-v1"
DEFAULT_NUM_ENVS=4096
DEFAULT_MAX_ITERATIONS=5000
DEFAULT_SEED=42
DEFAULT_CHECKPOINT_INTERVAL=200
DEFAULT_STEPS_PER_ITERATION=48
DEFAULT_CHECKPOINT_VIDEO_LENGTH=200
DEFAULT_FINAL_VIDEO_LENGTH=1000

usage() {
    cat <<'EOF'
Usage:
  fdu_flat_train_pipeline.sh start [options]
  fdu_flat_train_pipeline.sh watch [options]

start options:
  --profile NAME           Server defaults: gpu-isaac or lab-3090 (default: gpu-isaac)
  --repo PATH              Repository root (overrides profile)
  --data-root PATH         Logs, checkpoints, and videos root (default: repository root)
  --python PATH            Isaac Lab Python executable
  --gpu ID|auto            Physical GPU exposed through CUDA_VISIBLE_DEVICES
  --skip-gpu-check         Start without rejecting a GPU used by another process
  --task TASK              Training task id
  --num-envs N             Number of training environments (default: 4096)
  --max-iterations N       PPO iterations (default: 5000)
  --seed N                 Random seed (default: 42)
  --run-name NAME          Log directory suffix
  --checkpoint-interval N  Iterations between checkpoints (default: 200)
  --checkpoint-video-length N
                           Steps per training video (default: 200)
  --checkpoint-video-interval N
                           Environment steps between videos (default: checkpoint interval x 48)

watch options:
  --profile NAME           Server defaults: gpu-isaac or lab-3090 (default: gpu-isaac)
  --repo PATH              Repository root (overrides profile)
  --data-root PATH         Logs, checkpoints, and videos root (default: repository root)
  --python PATH            Isaac Lab Python executable
  --gpu ID                 Physical GPU exposed through CUDA_VISIBLE_DEVICES
  --skip-gpu-check         Run final Play without waiting for the GPU to become free
  --train-pid PID          Training process to wait for
  --play-task TASK         Play task id
  --run-name NAME          Log directory suffix
  --video-length N         Final Play steps to record (default: 1000)
EOF
}

die() {
    echo "[FDU-PIPELINE] ERROR: $*" >&2
    exit 1
}

log() {
    echo "[FDU-PIPELINE] $(date -Is) $*"
}

load_profile_defaults() {
    case "$1" in
        gpu-isaac)
            PROFILE_REPO="$DEFAULT_REPO"
            PROFILE_DATA_ROOT="$DEFAULT_REPO"
            PROFILE_PYTHON="$DEFAULT_PYTHON"
            PROFILE_GPU="0"
            ;;
        lab-3090)
            PROFILE_REPO="$LAB_3090_REPO"
            PROFILE_DATA_ROOT="$LAB_3090_DATA_ROOT"
            PROFILE_PYTHON="$LAB_3090_PYTHON"
            PROFILE_GPU="auto"
            ;;
        *) die "unknown profile: $1" ;;
    esac
}

gpu_processes() {
    nvidia-smi --id="$1" \
        --query-compute-apps=pid,process_name,used_gpu_memory \
        --format=csv,noheader,nounits 2>/dev/null
}

gpu_is_free() {
    local processes
    processes="$(gpu_processes "$1")" || return 1
    [[ -z "${processes//[[:space:]]/}" ]]
}

resolve_gpu() {
    local requested="$1"
    command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi was not found"

    local gpu_count
    gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | wc -l)"
    [[ "$gpu_count" =~ ^[1-9][0-9]*$ ]] || die "no NVIDIA GPU was detected"

    if [[ "$requested" == "auto" ]]; then
        local candidate
        for ((candidate = 0; candidate < gpu_count; candidate++)); do
            if gpu_is_free "$candidate"; then
                printf '%s\n' "$candidate"
                return 0
            fi
        done
        die "all ${gpu_count} GPUs have active compute processes"
    fi

    [[ "$requested" =~ ^[0-9]+$ ]] || die "GPU must be a physical index or auto: $requested"
    (( requested < gpu_count )) || die "GPU index ${requested} is out of range (count=${gpu_count})"
    printf '%s\n' "$requested"
}

require_free_gpu() {
    local gpu="$1"
    local processes
    processes="$(gpu_processes "$gpu")" || die "failed to query GPU ${gpu} processes"
    if [[ -n "${processes//[[:space:]]/}" ]]; then
        die "GPU ${gpu} has active compute processes: ${processes//$'\n'/; }"
    fi
}

wait_for_free_gpu() {
    local gpu="$1"
    local processes
    while true; do
        processes="$(gpu_processes "$gpu")" || die "failed to query GPU ${gpu} processes"
        [[ -n "${processes//[[:space:]]/}" ]] || return 0
        log "GPU ${gpu} is in use; waiting before final Play: ${processes//$'\n'/; }"
        sleep 60
    done
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
    local data_root="$2"
    local python="$3"
    local gpu="$4"
    local check_gpu="$5"
    local train_pid="$6"
    local play_task="$7"
    local run_name="$8"
    local video_length="$9"
    local run_root="$data_root/logs/rsl_rl/wheelbipe_fdu_wyw_flat_direct"
    local artifact_prefix="$data_root/logs/cloud/${run_name}"
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

    if [[ "$check_gpu" == "1" ]]; then
        wait_for_free_gpu "$gpu"
    fi

    export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
    export OMNI_KIT_ACCEPT_EULA=YES
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" "$python" -u "$repo/scripts/rsl_rl/play.py" \
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
    [[ "$play_status" -eq 0 ]] || die "post-training play failed"

    local video_path
    video_path="$(find "$run_dir/videos/play" -type f -name "*.mp4" -size +0c -printf "%T@ %p\n" \
        | sort -n | tail -1 | cut -d' ' -f2-)"
    [[ -n "$video_path" && -s "$video_path" ]] || die "play returned success but no non-empty MP4 was found"
    printf '%s\n' "$video_path" > "${artifact_prefix}.play_video.txt"
    touch "${artifact_prefix}.play.complete"
    log "video=${video_path}"
}

start_pipeline() {
    local profile="gpu-isaac"
    local repo=""
    local data_root=""
    local python=""
    local gpu=""
    local check_gpu=1
    local task="$DEFAULT_TASK"
    local num_envs="$DEFAULT_NUM_ENVS"
    local max_iterations="$DEFAULT_MAX_ITERATIONS"
    local seed="$DEFAULT_SEED"
    local run_name="flat_500hz_height015_030_4096_iter5000"
    local checkpoint_interval="$DEFAULT_CHECKPOINT_INTERVAL"
    local checkpoint_video_length="$DEFAULT_CHECKPOINT_VIDEO_LENGTH"
    local checkpoint_video_interval=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --profile) profile="$2"; shift 2 ;;
            --repo) repo="$2"; shift 2 ;;
            --data-root) data_root="$2"; shift 2 ;;
            --python) python="$2"; shift 2 ;;
            --gpu) gpu="$2"; shift 2 ;;
            --skip-gpu-check) check_gpu=0; shift ;;
            --task) task="$2"; shift 2 ;;
            --num-envs) num_envs="$2"; shift 2 ;;
            --max-iterations) max_iterations="$2"; shift 2 ;;
            --seed) seed="$2"; shift 2 ;;
            --run-name) run_name="$2"; shift 2 ;;
            --checkpoint-interval) checkpoint_interval="$2"; shift 2 ;;
            --checkpoint-video-length) checkpoint_video_length="$2"; shift 2 ;;
            --checkpoint-video-interval) checkpoint_video_interval="$2"; shift 2 ;;
            -h|--help) usage; return 0 ;;
            *) die "unknown start option: $1" ;;
        esac
    done

    load_profile_defaults "$profile"
    repo="${repo:-$PROFILE_REPO}"
    data_root="${data_root:-$PROFILE_DATA_ROOT}"
    python="${python:-$PROFILE_PYTHON}"
    gpu="${gpu:-$PROFILE_GPU}"
    [[ "$checkpoint_interval" =~ ^[1-9][0-9]*$ ]] || die "checkpoint interval must be positive"
    [[ "$checkpoint_video_length" =~ ^[1-9][0-9]*$ ]] || die "checkpoint video length must be positive"
    if [[ -z "$checkpoint_video_interval" ]]; then
        checkpoint_video_interval=$((checkpoint_interval * DEFAULT_STEPS_PER_ITERATION))
    fi
    [[ "$checkpoint_video_interval" =~ ^[1-9][0-9]*$ ]] || die "checkpoint video interval must be positive"
    [[ -d "$repo" ]] || die "repository does not exist: $repo"
    [[ -x "$python" ]] || die "Isaac Lab Python does not exist: $python"
    [[ "$run_name" =~ ^[A-Za-z0-9._-]+$ ]] || die "run name may contain only letters, digits, dot, underscore, and hyphen"
    gpu="$(resolve_gpu "$gpu")"
    [[ "$check_gpu" == "0" ]] || require_free_gpu "$gpu"
    mkdir -p "$data_root/logs/cloud" "$data_root/logs/rsl_rl"

    local train_log="$data_root/logs/cloud/${run_name}.train.log"
    local watch_log="$data_root/logs/cloud/${run_name}.post_play.log"
    log "starting profile=${profile} task=${task} gpu=${gpu} envs=${num_envs} iterations=${max_iterations} seed=${seed}"
    log "data_root=${data_root} checkpoint_interval=${checkpoint_interval} video_interval=${checkpoint_video_interval}"
    nohup env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}" OMNI_KIT_ACCEPT_EULA=YES \
        "$python" -u "$repo/scripts/rsl_rl/train.py" \
        --task="$task" \
        --num_envs="$num_envs" \
        --max_iterations="$max_iterations" \
        --seed="$seed" \
        --save_interval="$checkpoint_interval" \
        --log_root="$data_root/logs/rsl_rl" \
        --device=cuda:0 \
        --headless \
        --video \
        --video_length="$checkpoint_video_length" \
        --video_interval="$checkpoint_video_interval" \
        --run_name="$run_name" > "$train_log" 2>&1 < /dev/null &
    local train_pid=$!
    printf '%s\n' "$train_pid" > "$data_root/logs/cloud/${run_name}.train.pid"
    log "train_pid=${train_pid}"
    log "train_log=${train_log}"

    local watch_args=(
        watch
        --profile "$profile"
        --repo "$repo"
        --data-root "$data_root"
        --python "$python"
        --gpu "$gpu"
        --train-pid "$train_pid"
        --play-task "$DEFAULT_PLAY_TASK"
        --run-name "$run_name"
        --video-length "$DEFAULT_FINAL_VIDEO_LENGTH"
    )
    [[ "$check_gpu" == "1" ]] || watch_args+=(--skip-gpu-check)
    nohup "$SCRIPT_PATH" "${watch_args[@]}" > "$watch_log" 2>&1 < /dev/null &
    local watcher_pid=$!
    printf '%s\n' "$watcher_pid" > "$data_root/logs/cloud/${run_name}.watch.pid"
    log "watcher_pid=${watcher_pid}"
    log "watch_log=${watch_log}"
}

watch_command() {
    local profile="gpu-isaac"
    local repo=""
    local data_root=""
    local python=""
    local gpu=""
    local check_gpu=1
    local train_pid=""
    local play_task="$DEFAULT_PLAY_TASK"
    local run_name="flat_500hz_height015_030_4096_iter5000"
    local video_length="$DEFAULT_FINAL_VIDEO_LENGTH"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --profile) profile="$2"; shift 2 ;;
            --repo) repo="$2"; shift 2 ;;
            --data-root) data_root="$2"; shift 2 ;;
            --python) python="$2"; shift 2 ;;
            --gpu) gpu="$2"; shift 2 ;;
            --skip-gpu-check) check_gpu=0; shift ;;
            --train-pid) train_pid="$2"; shift 2 ;;
            --play-task) play_task="$2"; shift 2 ;;
            --run-name) run_name="$2"; shift 2 ;;
            --video-length) video_length="$2"; shift 2 ;;
            -h|--help) usage; return 0 ;;
            *) die "unknown watch option: $1" ;;
        esac
    done

    load_profile_defaults "$profile"
    repo="${repo:-$PROFILE_REPO}"
    data_root="${data_root:-$PROFILE_DATA_ROOT}"
    python="${python:-$PROFILE_PYTHON}"
    gpu="${gpu:-$PROFILE_GPU}"
    gpu="$(resolve_gpu "$gpu")"
    [[ -n "$train_pid" ]] || die "--train-pid is required for watch"
    watch_pipeline "$repo" "$data_root" "$python" "$gpu" "$check_gpu" "$train_pid" "$play_task" "$run_name" "$video_length"
}

command="${1:-}"
shift || true
case "$command" in
    start) start_pipeline "$@" ;;
    watch) watch_command "$@" ;;
    -h|--help|"") usage ;;
    *) die "expected start or watch (got: ${command})" ;;
esac
