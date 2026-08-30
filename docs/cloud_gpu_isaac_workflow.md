# gpu_isaac training workflow

This is the reusable server procedure for the WYW/FDU Flat task. The server
alias is `gpu_isaac`; the persistent repository path is
`/root/gpufree-data/wheeled-legged_RL_from_hu`.

The image uses the Isaac Lab environment at
`/opt/conda/envs/isaaclab/bin/python` (Python 3.11, Torch 2.7.0+cu128). Do not
use the image's default `/opt/conda/bin/python` (Python 3.13), because the
repository intentionally declares Python `<3.12`.

## One-time setup

From the repository root on the workstation, synchronize the complete source
tree, including `source/agent_world/agent_world/assets/usd_files/wheelbipe_fdu/`.
On `gpu_isaac`, install the three local packages without replacing the image's
Isaac/Torch stack:

```bash
cd /root/gpufree-data/wheeled-legged_RL_from_hu
PY=/opt/conda/envs/isaaclab/bin/python
$PY -m pip install -e source/agent_world --no-deps
$PY -m pip install -e source/agent_tasks --no-deps
$PY -m pip install -e source/agent_rl --no-deps
$PY -m pip install psutil tensorboard pytorch-kinematics pybullet
```

The source package and the reusable pipeline are
`scripts/cloud/fdu_flat_train_pipeline.sh`.

## Start training and automatic acceptance

The default pipeline runs 4096 environments for 5000 iterations, waits for
training to finish, plays the final checkpoint for 1000 steps, validates that a
non-empty MP4 exists, and then powers off the server:

```bash
cd /root/gpufree-data/wheeled-legged_RL_from_hu
bash scripts/cloud/fdu_flat_train_pipeline.sh start \
  --num-envs 4096 \
  --max-iterations 5000 \
  --seed 42 \
  --run-name flat_500hz_height015_030_4096_iter5000
```

Use `--no-shutdown` when debugging the play stage. The shutdown is requested
only after `play.py` exits successfully and a non-zero-size MP4 is found. The
default command is `shutdown -h now`; an image-specific command can be supplied
with `--shutdown-command '...'`. The command output is recorded, and a reported
missing shutdown key is treated as a failed shutdown rather than silently
claiming success.

The pipeline records these artifacts under `logs/cloud/`:

- `<run>.train.log`, `<run>.train.pid`: training output and PID.
- `<run>.post_play.log`: watcher output and checkpoint selection.
- `<run>.play_runtime.log`: play output.
- `<run>.play_video.txt`: absolute path of the retained MP4.
- `<run>.play.complete`: acceptance completed successfully.
- `<run>.shutdown_requested`: shutdown command was issued.
- `<run>.shutdown_command.log`: stdout/stderr from the shutdown command.
- `<run>.shutdown_accepted` or `<run>.shutdown_failed`: command result.

The model directory is under
`logs/rsl_rl/wheelbipe_fdu_wyw_flat_direct/<timestamp>_<run-name>/`; the final
checkpoint is selected by numeric `model_*.pt` order. The video is under that
directory's `videos/play/` folder.

## Reattach an existing run

If SSH disconnects, the training and watcher are independent `nohup` processes.
To inspect them:

```bash
ssh gpu_isaac 'ps -p $(cat /root/gpufree-data/wheeled-legged_RL_from_hu/logs/cloud/flat_500hz_height015_030_4096_iter5000.train.pid) -o pid,stat,etime,cmd'
ssh gpu_isaac 'tail -f /root/gpufree-data/wheeled-legged_RL_from_hu/logs/cloud/flat_500hz_height015_030_4096_iter5000.train.log'
```

Do not start a second watcher for the same training PID. If a post-play run
must be debugged without shutdown, start `watch` manually with
`--no-shutdown` after stopping the old watcher.
