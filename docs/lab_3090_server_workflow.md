# 实验室双 3090 服务器工作流

服务器 SSH 别名为 `3090_wyw_local`，仅在连接实验室内网 Wi-Fi 时可达。该服务器直接运行任务，不使用 Slurm。约定路径如下：

- 源码：`/home/wyw/wheeled-legged_RL_from_hu`
- Conda 环境：`/home/wyw/conda_envs/isaaclab_2`
- 日志、checkpoint 和视频：`/data1/wyw/wheeled-legged_RL_from_hu/logs`

`/home/wyw` 保存源码和运行环境；`/data1/wyw` 只保存数据集、日志、checkpoint 和运行视频，不放源码或 Conda 环境。

## 首次检查

在本机仓库根目录执行：

```bash
bash scripts/cloud/lab_3090_server_setup.sh doctor
```

它只读取服务器的系统版本、架构、glibc、磁盘、GPU、GPU 进程和 Conda 状态。环境打包方案要求两端都是 x86-64，并且服务器 NVIDIA 驱动能够支持当前环境中的 Isaac Sim 5.1、Torch 2.7.0 和 CUDA 12.6 运行时。

## 打包并部署本机环境

本机 `isaaclab_2` 环境约 21 GB，系统盘当前空间有限。先在有足够空间的本机磁盘上选择压缩包位置；工具不会擅自在仓库或系统临时目录中生成大文件。

如果本机还没有 `conda-pack`：

```bash
conda install -n base -c conda-forge conda-pack
```

连接内网 Wi-Fi 后，一条命令完成服务器检查、环境打包、源码同步、上传解包、editable 包安装和导入验证：

```bash
bash scripts/cloud/lab_3090_server_setup.sh bootstrap \
  --archive /media/wyw/Anything/isaaclab_2-conda-pack.tar.gz \
  --ignore-missing-files
```

当前本机环境中的 `pip` / `setuptools` / `wheel` 曾由 pip 覆盖 Conda 版本，实际包可正常导入，但 Conda 元数据仍记录旧文件。`--ignore-missing-files` 是针对这个已核对的本机环境所需；不应在未检查的其他环境上盲目使用。

已有压缩包时会直接复用。也可以分步执行以便定位问题：

```bash
bash scripts/cloud/lab_3090_server_setup.sh pack \
  --archive /media/wyw/Anything/isaaclab_2-conda-pack.tar.gz \
  --ignore-missing-files
bash scripts/cloud/lab_3090_server_setup.sh sync
bash scripts/cloud/lab_3090_server_setup.sh install \
  --archive /media/wyw/Anything/isaaclab_2-conda-pack.tar.gz
bash scripts/cloud/lab_3090_server_setup.sh verify
```

`conda-pack` 携带 Python/Isaac Sim 主环境，也不包含 NVIDIA 驱动。解包后直接使用环境内 Python，不要求服务器已有同名 Conda 环境。当前本机环境有一部分 Isaac Lab 依赖来自用户级 `~/.local` ，它们不在 Conda 包内；因此部署时服务器需要联网补齐 Isaac Lab 声明依赖。源码不会放进环境包；部署工具会在同步后重新安装三个 editable 包。

## 检查 GPU 并启动训练

先查看两张卡及使用进程：

```bash
ssh 3090_wyw_local nvidia-smi
```

再登录服务器并启动：

```bash
ssh 3090_wyw_local
cd /home/wyw/wheeled-legged_RL_from_hu
bash scripts/cloud/fdu_flat_train_pipeline.sh start \
  --profile lab-3090 \
  --gpu auto \
  --num-envs 4096 \
  --max-iterations 5000 \
  --seed 42 \
  --run-name flat_3090_seed42
```

`--gpu auto` 会检查两张物理 GPU 的 compute process，选择第一张空闲卡；两张卡都有人使用时拒绝启动。也可以用 `--gpu 0` 或 `--gpu 1` 指定物理卡。训练和最终 Play 都通过 `CUDA_VISIBLE_DEVICES=<物理卡>` 隔离，并在程序内部使用逻辑设备 `cuda:0`。

不要在共享服务器上使用 `--skip-gpu-check`，除非已经通过其他方式独占该 GPU。

## Checkpoint 与录像

默认每 500 次迭代保存 checkpoint。WYW 每次迭代执行 48 个环境 step，因此流水线传入 Isaac 原生录像参数：

```text
--video --video_length=200 --video_interval=24000
```

训练录像位于：

```text
/data1/wyw/wheeled-legged_RL_from_hu/logs/rsl_rl/
  wheelbipe_fdu_wyw_flat_direct/<时间>_<run-name>/videos/train/
```

`model_N.pt` 对应的训练录像从 step `N x 48` 附近开始。周期 checkpoint 使用训练过程原生录像；训练结束时额外保存的最终 checkpoint 由 watcher 单独 Play，并把录像放在同一 run 的 `videos/play/` 下。

修改 checkpoint 间隔时，流水线会自动重新计算录像间隔：

```bash
bash scripts/cloud/fdu_flat_train_pipeline.sh start \
  --profile lab-3090 \
  --checkpoint-interval 250 \
  --checkpoint-video-length 200 \
  --run-name flat_3090_ckpt250
```

## 查看运行状态

```bash
DATA=/data1/wyw/wheeled-legged_RL_from_hu
RUN=flat_3090_seed42
ps -p "$(cat "$DATA/logs/cloud/$RUN.train.pid")" -o pid,stat,etime,cmd
tail -f "$DATA/logs/cloud/$RUN.train.log"
```

训练和 watcher 都通过 `nohup` 运行，SSH 断开不会终止任务。训练开始前会检查 GPU；训练完成后如果该卡已被别人占用，watcher 会等待 GPU 再执行最终 Play。
