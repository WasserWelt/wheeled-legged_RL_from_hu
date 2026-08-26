# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

RL training/deployment code for wheeled-legged (wheelbipe) robots, built on **Isaac Sim + Isaac Lab + RSL-RL**. Target: 26-season infantry robot. `README.md` (Chinese) is the authoritative user-facing doc; this file captures the non-obvious wiring.

## Setup & commands

use `conda activate isaaclab_2`

The three `source/` packages must be installed editable, and scripts must run **from the repo root** (env code does `import scripts.utils.velocity_trace_html`; otherwise `ModuleNotFoundError: No module named 'scripts'`). Activate the conda env with Isaac Sim/Lab first.

```bash
python -m pip install -e source/agent_world
python -m pip install -e source/agent_tasks
python -m pip install -e source/agent_rl

python scripts/list_envs.py                       # list registered tasks
python scripts/view_robot.py --task=<task> --num_envs=1 --device=cpu
python scripts/rsl_rl/train.py --task=<task> --num_envs=4096 --max_iterations=20000 --device=cuda:0 --headless
python scripts/rsl_rl/play.py  --task=<task> --num_envs=1 --checkpoint=<model.pt> --keyboard [--plot]
```

There is no test suite, linter, or build step. "Running" means launching a task in sim via the scripts above.

- `train.py --checkpoint=X` **without** `--resume_training` = finetune: loads only shape-matching policy tensors (2D input layers are zero-padded when the model is wider), resets iteration to 0. See `_load_policy_for_finetune` in `scripts/rsl_rl/train.py`.
- `train.py --checkpoint=X --resume_training` = true resume: restores optimizer + iteration counter.
- Logs/checkpoints/exports go to `logs/rsl_rl/<experiment_name>/<timestamp>/`. Velocity-trace CSV/HTML go to `logs/debug/`.

## Architecture

Three editable packages layered world → tasks → algorithms, plus entry scripts:

- **`source/agent_world`** — the Isaac Lab "world": robot USD assets (`assets/wheelbipe_V14*.py` etc.), custom actuators (`actuators/`: M3508, differential-velocity, learned-velocity), and terrain generators (`terrains/height_field.py`).
- **`source/agent_tasks`** — Gym environments and task logic.
  - `direct/wheelbipe/wheelbipe_V14/{env.py,env_cfg.py,cfg_utils.py}` — the active task family. Each `gym.register` id maps to one `env_cfg` class + one `rsl_rl_cfg_entry_point`; task *variants* (flat/rough, v0/v1/v2, DreamWaQ/HIM/NP3O, `_Play`) are almost all different `env_cfg` subclasses over the **same** `WheelbipeV14Env`.
  - `direct/wheelbipe/state_machines/` — per-env state machines (airborne / jump_takeoff / step_up / stair), driven by `manager/mdp/state_machine/`. Also handles "小陀螺" (spin-in-place) translation mode.
  - `manager/mdp/isaaclab/` — MDP terms: `observations.py`, `rewards.py`, `events.py`, `curriculums.py`, `terrains.py`, `commands.py`.
  - `manager/mdp/terrain/` — terrain command/task managers that bias command sampling per terrain.
- **`source/agent_rl`** — RL algorithms, extends RSL-RL. Each algorithm = a triple (algorithm + actor-critic module + runner):
  - PPO (stock RSL-RL `OnPolicyRunner`)
  - DreamWaQ: `algorithms/ppo_dreamwaq.py` + `modules/actor_critic_dreamwaq.py` + `runners/on_policy_runner_dreamwaq.py`
  - HIMLoco: `algorithms/ppo_him.py` + `modules/actor_critic_him.py` + `modules/him_estimator.py` + `runners/on_policy_runner_him.py`
  - NP3O (BarlowTwins + safety constraint): `algorithms/np3o.py` + `modules/actor_critic_balowtwins.py` + `runners/on_constraint_policy_runner.py`
  - `modules/policy/` and `utils/exporter*.py` handle TorchScript/ONNX export for deployment.

### How a task selects its algorithm (important)

Config is resolved through **Hydra + gymnasium**, not imports. `train.py`/`play.py` call `@hydra_task_config(task, "rsl_rl_cfg_entry_point")`, which pulls the agent cfg class registered for that task in `wheelbipe_V14/__init__.py`. The agent cfg (in `agents/rsl_rl_ppo_cfg.py`) sets `runner_class = "OnPolicy...Runner"` as a **string**; `train.py` does `runner_class = eval(getattr(agent_cfg, "runner_class", "OnPolicyRunner"))`, resolved against the wildcard-imported `rsl_rl.runners` + `agent_rl.rsl_rl.runners`. Non-`OnPolicyRunner` runners use `agent_rl.rsl_rl.env.RslRlVecEnvWrapper` instead of the stock Isaac Lab wrapper. So to add an algorithm variant you register a new task id → new `env_cfg` + new agent cfg class whose `runner_class`/`class_name` strings point at your runner/module/algorithm.

### Task registration is opt-in per module

Registration happens as an import side effect. `agent_tasks/__init__.py` imports **only** `agent_tasks.direct.wheelbipe.wheelbipe_V14`, so only the V14 task ids are actually registered/runnable — despite `direct/wheelbipe/README.md` listing V13 and Wheelbipe25-V3 ids and `wheelbipe_V13/`, `wheelbipe25_v3/` dirs existing. To enable those, add the import to `agent_tasks/__init__.py`. Use `scripts/list_envs.py` as the source of truth for what's currently registered.

## Conventions

- Source files carry an SCUTRobotLab MIT header; keep it on new files.
- Comments/docs are largely in Chinese — match the surrounding language.
- `train.py` strips a leading `=` from `--task` and suppresses `quat_rotate` deprecation warnings; don't "fix" these as bugs.
- `pretrained/26_infantry/` holds example checkpoints (`.pt`) + exported `.onnx`/`policy.pt` + dumped `params/{env,agent}.yaml` per run — a good reference for expected cfg shapes.
