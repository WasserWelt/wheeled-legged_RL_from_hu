#!/usr/bin/env bash

# Prepare the lab 2xRTX3090 server without requiring it to download Isaac Sim.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_NAME="$(basename "$LOCAL_REPO")"

SSH_HOST="3090_wyw_local"
LOCAL_ENV="/home/wyw/anaconda3/envs/isaaclab_2"
REMOTE_REPO="/home/wyw/$REPO_NAME"
REMOTE_DATA_ROOT="/data1/wyw/$REPO_NAME"
REMOTE_ENV="/home/wyw/conda_envs/isaaclab_2"
REMOTE_ARCHIVE_DIR="/home/wyw/env_archives"
ARCHIVE=""
IGNORE_MISSING_FILES=0
VERIFY_GPU=0

usage() {
    cat <<'EOF'
Usage:
  lab_3090_server_setup.sh doctor [options]
  lab_3090_server_setup.sh check-sync [options]
  lab_3090_server_setup.sh pack --archive PATH [options]
  lab_3090_server_setup.sh sync [options]
  lab_3090_server_setup.sh install --archive PATH [options]
  lab_3090_server_setup.sh verify [options]
  lab_3090_server_setup.sh bootstrap --archive PATH [options]

Commands:
  doctor      Show remote OS, disks, NVIDIA GPUs, GPU processes, and Conda.
  check-sync  Compare local and remote source content without changing either side.
  pack        Pack the local Isaac Lab Conda environment. Reuses an existing archive.
  sync        Synchronize source code to /home without logs or Git metadata.
  install     Upload and unpack the environment, then install editable project packages.
  verify      Verify remote Python imports and show the installed versions.
  bootstrap   Run doctor, pack, sync, install, and verify in order.

Options:
  --host HOST              SSH host or alias (default: 3090_wyw_local)
  --archive PATH           Local conda-pack .tar.gz path; required by pack/install/bootstrap
  --local-env PATH         Local Conda environment (default: /home/wyw/anaconda3/envs/isaaclab_2)
  --remote-repo PATH       Remote source directory
  --remote-data-root PATH  Remote logs/checkpoints/videos root
  --remote-env PATH        Remote unpacked Conda environment
  --remote-archive-dir PATH
                           Temporary remote directory for the uploaded archive
  --gpu ID                 Physical GPU for the Isaac smoke test (default: 0)
  --ignore-missing-files   Allow known Conda/pip metadata drift while packing
EOF
}

RSYNC_SOURCE_FILTERS=(
    --exclude '/.git/'
    --exclude '/logs/'
    --exclude '/outputs/'
    --exclude '/.pytest_cache/'
    --exclude '/.mypy_cache/'
    --exclude '/.ruff_cache/'
    --exclude '**/__pycache__/'
    --exclude '*.pyc'
    --exclude '*.egg-info/'
)

die() {
    echo "[LAB-3090-SETUP] ERROR: $*" >&2
    exit 1
}

log() {
    echo "[LAB-3090-SETUP] $(date -Is) $*"
}

require_archive() {
    [[ -n "$ARCHIVE" ]] || die "--archive PATH is required"
    [[ "$ARCHIVE" == *.tar.gz ]] || die "archive must end in .tar.gz: $ARCHIVE"
}

validate_remote_settings() {
    [[ "$SSH_HOST" =~ ^[A-Za-z0-9._-]+$ ]] || die "unsafe SSH host: $SSH_HOST"
    [[ "$VERIFY_GPU" =~ ^[0-9]+$ ]] || die "GPU must be a physical index: $VERIFY_GPU"
    local path
    for path in "$REMOTE_REPO" "$REMOTE_DATA_ROOT" "$REMOTE_ENV" "$REMOTE_ARCHIVE_DIR"; do
        [[ "$path" =~ ^/[A-Za-z0-9._/-]+$ ]] || die "remote paths must be absolute and shell-safe: $path"
    done
}

doctor() {
    log "checking ${SSH_HOST}; connect to the internal Wi-Fi first"
    ssh "$SSH_HOST" bash -s -- "$REMOTE_REPO" "$REMOTE_DATA_ROOT" "$REMOTE_ENV" <<'REMOTE'
set -euo pipefail
remote_repo="$1"
remote_data_root="$2"
remote_env="$3"

echo "== identity =="
hostname
id
echo "== operating system =="
cat /etc/os-release
uname -m
ldd --version 2>&1 | sed -n '1p'
echo "== GPUs =="
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader
echo "== active compute processes =="
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
    --format=csv,noheader || true
echo "== storage =="
df -h /home /data1
echo "== paths =="
ls -ld "$remote_repo" "$remote_data_root" "$remote_env" 2>/dev/null || true
echo "== Conda =="
command -v conda || true
REMOTE
}

pack_environment() {
    require_archive
    [[ -x "$LOCAL_ENV/bin/python" ]] || die "local environment does not exist: $LOCAL_ENV"
    if [[ -s "$ARCHIVE" ]]; then
        log "reusing existing archive: $ARCHIVE"
        return 0
    fi

    local unreadable
    unreadable="$(find "$LOCAL_ENV" -not -readable -print -quit 2>/dev/null || true)"
    if [[ -n "$unreadable" ]]; then
        die "local environment contains an unreadable path: $unreadable (fix its owner permissions before packing)"
    fi

    local archive_parent
    archive_parent="$(dirname "$ARCHIVE")"
    mkdir -p "$archive_parent"

    local pack_args=(
        -p "$LOCAL_ENV"
        -o "$ARCHIVE"
        --ignore-editable-packages
    )
    if [[ "$IGNORE_MISSING_FILES" == "1" ]]; then
        "$LOCAL_ENV/bin/python" -m pip --version >/dev/null
        "$LOCAL_ENV/bin/python" -c 'import setuptools, wheel' >/dev/null
        pack_args+=(--ignore-missing-files)
    fi

    if command -v conda-pack >/dev/null 2>&1; then
        log "packing ${LOCAL_ENV} to ${ARCHIVE}"
        conda-pack "${pack_args[@]}"
    elif [[ -x /home/wyw/anaconda3/bin/conda-pack ]]; then
        log "packing ${LOCAL_ENV} to ${ARCHIVE}"
        /home/wyw/anaconda3/bin/conda-pack "${pack_args[@]}"
    else
        die "conda-pack is missing; install it with: conda install -n base -c conda-forge conda-pack"
    fi
}

sync_repository() {
    log "creating remote source and data directories"
    ssh "$SSH_HOST" mkdir -p "$REMOTE_REPO" "$REMOTE_DATA_ROOT"
    log "syncing source to ${SSH_HOST}:${REMOTE_REPO}"
    rsync -az --info=progress2 \
        "${RSYNC_SOURCE_FILTERS[@]}" \
        "$LOCAL_REPO/" "$SSH_HOST:$REMOTE_REPO/"
}

check_repository_sync() {
    [[ -d "$LOCAL_REPO" ]] || die "local repository does not exist: $LOCAL_REPO"
    log "checking source content against ${SSH_HOST}:${REMOTE_REPO}"

    local changes status
    set +e
    changes="$(rsync -rlnc --delete --itemize-changes \
        "${RSYNC_SOURCE_FILTERS[@]}" \
        "$LOCAL_REPO/" "$SSH_HOST:$REMOTE_REPO/" 2>&1)"
    status=$?
    set -e
    [[ "$status" -eq 0 ]] || die "source comparison failed: $changes"

    if [[ -n "${changes//[[:space:]]/}" ]]; then
        echo "$changes"
        die "local and remote source differ; review the list, use '$0 sync' for local changes, and resolve remote-only files explicitly"
    fi
    log "source content is synchronized"
}

install_environment() {
    require_archive
    [[ -s "$ARCHIVE" ]] || die "archive does not exist or is empty: $ARCHIVE"

    local archive_name remote_archive
    archive_name="$(basename "$ARCHIVE")"
    remote_archive="$REMOTE_ARCHIVE_DIR/$archive_name"
    log "uploading environment archive to ${SSH_HOST}:${remote_archive}"
    ssh "$SSH_HOST" mkdir -p "$REMOTE_ARCHIVE_DIR"
    rsync -a --partial --info=progress2 "$ARCHIVE" "$SSH_HOST:$remote_archive"

    local local_sha remote_sha
    log "verifying uploaded archive checksum"
    local_sha="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
    remote_sha="$(ssh "$SSH_HOST" sha256sum "$remote_archive" | awk '{print $1}')"
    [[ "$local_sha" == "$remote_sha" ]] || die "uploaded archive checksum mismatch"

    log "installing environment at ${REMOTE_ENV}"
    ssh "$SSH_HOST" bash -s -- "$remote_archive" "$REMOTE_ENV" "$REMOTE_REPO" <<'REMOTE'
set -euo pipefail
archive="$1"
remote_env="$2"
remote_repo="$3"

if [[ ! -x "$remote_env/bin/python" ]]; then
    if [[ -d "$remote_env" && -n "$(find "$remote_env" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "ERROR: partial environment exists at $remote_env; inspect it before retrying" >&2
        exit 1
    fi
    mkdir -p "$remote_env"
    tar -xzf "$archive" -C "$remote_env"
fi

if [[ ! -f "$remote_env/.wyw-conda-unpacked" ]]; then
    "$remote_env/bin/python" "$remote_env/bin/conda-unpack"
    touch "$remote_env/.wyw-conda-unpacked"
fi

[[ -d "$remote_repo/source/agent_world" ]] || {
    echo "ERROR: repository is missing at $remote_repo; run sync first" >&2
    exit 1
}

"$remote_env/bin/python" -m pip install \
    "click==8.1.7" \
    "typing_extensions==4.12.2" \
    "prettytable==3.3.0" \
    "hydra-core==1.3.2" \
    "moviepy==2.2.1" \
    "protobuf==6.33.0" \
    "platformdirs==4.5.0" \
    "tensordict==0.9.0"
"$remote_env/bin/python" -m pip install -e "$remote_repo/source/agent_world" --no-deps
"$remote_env/bin/python" -m pip install -e "$remote_repo/source/agent_tasks" --no-deps
"$remote_env/bin/python" -m pip install -e "$remote_repo/source/agent_rl" --no-deps
REMOTE
}

verify_environment() {
    log "verifying remote environment"
    ssh "$SSH_HOST" bash -s -- "$REMOTE_ENV" "$REMOTE_REPO" "$REMOTE_DATA_ROOT" "$VERIFY_GPU" <<'REMOTE'
set -euo pipefail
remote_env="$1"
remote_repo="$2"
remote_data_root="$3"
gpu="$4"
python="$remote_env/bin/python"

[[ -x "$python" ]] || { echo "ERROR: Python is missing: $python" >&2; exit 1; }
[[ -d "$remote_repo" ]] || { echo "ERROR: repository is missing: $remote_repo" >&2; exit 1; }
mkdir -p "$remote_data_root/logs/cloud" "$remote_data_root/logs/rsl_rl"
PYTHONPATH="$remote_repo${PYTHONPATH:+:$PYTHONPATH}" "$python" - <<'PY'
import importlib.metadata as metadata
import sys
import torch

print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
for package in ("isaaclab", "isaacsim", "agent_world", "agent_tasks", "agent_rl"):
    print(package, metadata.version(package))
PY

processes="$(nvidia-smi --id="$gpu" \
    --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits 2>/dev/null)"
if [[ -n "${processes//[[:space:]]/}" ]]; then
    echo "ERROR: GPU $gpu is in use; refusing Isaac smoke test: $processes" >&2
    exit 1
fi

smoke_log="$(mktemp)"
trap 'rm -f "$smoke_log"' EXIT
if ! DISPLAY= CUDA_VISIBLE_DEVICES="$gpu" OMNI_KIT_ACCEPT_EULA=YES \
    PYTHONPATH="$remote_repo${PYTHONPATH:+:$PYTHONPATH}" \
    timeout 300 "$python" -u "$remote_repo/scripts/list_envs.py" >"$smoke_log" 2>&1; then
    tail -n 120 "$smoke_log" >&2
    exit 1
fi
grep 'Robotics-Wheelbipe-FDU-wyw' "$smoke_log"
echo "isaac_smoke import-and-task-registration-ok"
REMOTE
}

command="${1:-}"
[[ -n "$command" ]] || { usage; exit 0; }
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host) SSH_HOST="$2"; shift 2 ;;
        --archive) ARCHIVE="$2"; shift 2 ;;
        --local-env) LOCAL_ENV="$2"; shift 2 ;;
        --remote-repo) REMOTE_REPO="$2"; shift 2 ;;
        --remote-data-root) REMOTE_DATA_ROOT="$2"; shift 2 ;;
        --remote-env) REMOTE_ENV="$2"; shift 2 ;;
        --remote-archive-dir) REMOTE_ARCHIVE_DIR="$2"; shift 2 ;;
        --gpu) VERIFY_GPU="$2"; shift 2 ;;
        --ignore-missing-files) IGNORE_MISSING_FILES=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

validate_remote_settings

case "$command" in
    doctor) doctor ;;
    check-sync) check_repository_sync ;;
    pack) pack_environment ;;
    sync) sync_repository ;;
    install) install_environment ;;
    verify) verify_environment ;;
    bootstrap)
        doctor
        pack_environment
        sync_repository
        install_environment
        verify_environment
        ;;
    -h|--help) usage ;;
    *) die "expected doctor, check-sync, pack, sync, install, verify, or bootstrap (got: ${command})" ;;
esac
