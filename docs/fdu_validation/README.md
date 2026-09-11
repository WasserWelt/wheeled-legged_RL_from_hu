# FDU model validation outputs

This directory keeps generated validation artifacts out of the `docs/` root.

- `geometry/`: analytical/physics geometry calibration reports.
- `jitter/timestep/`: valid physics-frequency and short-leg boundary scans.
- `jitter/solver/`: valid iteration/damping solver A/B scans.
- `drop/`: controlled free-drop and guided-vertical impact reports.
- `video/`: current accepted geometry and drop-test recordings.
- `jump/`: jump-specific calibration reports.
- `training/`: smoke/PPO-adjacent long-horizon reset and command-timing reports.
- `archive/video_old/`: superseded recordings, retained for comparison.
- `archive/invalid/`: experiments that must not be used as evidence.

## Invalid archive

- `fdu_jitter_planar_d6.json`: the experimental planar D6 anchors did not
  close the mechanism correctly.
- `fdu_jitter_no_colliders_200hz.json`: collision disabling could not be
  authored through the instanceable USD; the run emitted an explicit warning,
  so its numbers are not accepted as a collider A/B test.
- `fdu_drop_500hz_l016.json`: gravity was enabled while the leg was still
  moving from its default pose to the requested length. It was replaced by a
  pre-shaped controlled drop.

The production articulation already sets `enabled_self_collisions=False`, so
all accepted static jitter and drop tests disable link-link self-collision.

The pre-40 N·m drop JSON files remain as historical 30 N·m baselines. New
reports and videos with an `_40nm` suffix use the production hard ceiling and
record the applied Kp/Kd/effort limit in their metadata.

Current 500 Hz guided-vertical drop videos (90 frames, 30 FPS, 960x720):

- `video/fdu_drop_guided_500hz_l016_h040_40nm.mp4`: short-leg, L0=0.16 m.
- `video/fdu_drop_guided_500hz_l0285_h045_40nm.mp4`: nominal-leg, L0=0.285 m.
- `video/fdu_drop_guided_800hz_l016_h040_40nm.mp4`: short-leg timestep A/B;
  steady-state L0 peak-to-peak falls from 4.37/4.29 mm at 500 Hz to
  0.282/0.108 mm at 800 Hz.

Their matching JSON reports are under `drop/`. The
`fdu_drop_guided_500hz_l0285_h045_kp30_kd2_40nm.json` run is diagnostic only;
production PD remains Kp=20/Kd=1 (Flat/Rough) and Kp=6/Kd=0.5 (Jump).
The `fdu_drop_guided_500hz_pos32_l016_h040_40nm.json` run is the solver-cost
A/B: 500 Hz with 32/6 iterations reaches 0.076/0.098 mm, but is not the
recommended production setting because its rough iteration rate exceeds
800 Hz with 16/6.

## Training semantic acceptance

`training/fdu_{flat,rough,jump}_long_horizon_500hz.json` records real 500 Hz
CPU physics through a natural 20 s timeout under the current 16/6 solver and
`height_range=[0.15,0.30]`. Flat/Rough periodically resample at policy step
500. Jump's 20 s period coincides with the episode boundary, so timeout/reset
at step 1999 resamples it before a standalone step-2000 resample. All three
reports verify reset history/action clearing and episode-level L0 log keys;
the Rough report additionally verifies that its curriculum ran and records
whether that episode happened to change a terrain level or command range.
These runs deliberately raise the tilt-persistence window from the production
100 steps to 100000 steps so random zero-action tilt cannot pre-empt the natural
timeout; the production 100-step behavior is covered separately by golden and
environment smoke fixtures.

The matching GPU PPO acceptance runs (64 environments, 3 iterations, seed 42)
use the same `height_range=[0.15,0.30]`, 500 Hz / 16/6 / decimation-5
configuration:

- `../../logs/rsl_rl/wheelbipe_fdu_wyw_flat_direct/2026-08-30_18-30-56_acceptance_height015_030_500hz_64env_3iter/`
- `../../logs/rsl_rl/wheelbipe_fdu_wyw_rough_direct/2026-08-30_18-31-45_acceptance_height015_030_500hz_64env_3iter/`
- `../../logs/rsl_rl/wheelbipe_fdu_wyw_jump_direct/2026-08-30_18-32-43_acceptance_height015_030_500hz_64env_3iter/`

Flat and Rough pass the numerical and L0-boundary gates. Jump has finite PPO
loss/KL and no NaN, but remains a physical failure because 58/64 environments
are still in the L0 boundary during the final iteration without the Fudan gas
spring.

## Standard policy verification play

`scripts/rsl_rl/play_wyw_verify.py` runs the versioned WYW policy-performance
suite. Flat has 17 nominal and 4 robust scenarios, Rough has 35 nominal and 4
robust scenarios, and Jump has 3 nominal and 2 robust scenarios. Rough excludes
flat terrain and includes both pyramid stairs and 5/10/15/20 cm single steps.
The maximum simulated/video duration across all three tasks is 330.0 seconds.

Create the fixed Flat baseline once:

```bash
python scripts/rsl_rl/play_wyw_verify.py \
  --variant flat \
  --checkpoint logs/rsl_rl/2026-09-07_14-56-06_flat_8192_5000_diff_vel/model_4999.pt \
  --mode baseline --profile all --headless
```

The baseline path is configured in `configs/wyw_verify_baseline.json`.
Baseline creation writes `baseline.json`, `baseline.mp4`, and `trace.json`;
nominal and robust are concatenated in that order. The trace stores env 0
telemetry at video-frame times so later comparisons can align by scenario,
phase, and simulated time rather than by raw frame index.

The verifier accepts the known v2 finite-difference-velocity training contract
used by the diffvel/no-upreward runs, the v3 material-split contract, and the
v4 Flat wheel-contact-loss contract. Their
action, observation, velocity-estimator, timing, and Sequence network contracts
are identical; the actual training semantics version is recorded in each
profile report. Evaluation always uses the current WYW environment settings.

Evaluate a new checkpoint with:

```bash
python scripts/rsl_rl/play_wyw_verify.py \
  --variant flat \
  --checkpoint <current_checkpoint.pt> \
  --mode evaluate --profile all --headless
```

Evaluation writes `report.json`, `results.csv`, `comparison.mp4`,
`comparison.png`, `current.mp4`, `trace.json`, grouped metric sheets, and a
`charts.md` index. Start with `comparison.png` for the acceptance result,
profile pass counts, failure events, and the largest failed metric checks. The
grouped tracking, stability, smoothness, actuator, and outcome sheets show one
physical unit per axis, with gray baseline points and blue/red current points.
`robust_distributions.png` shows the individual current environments and the
baseline range for four task-relevant metrics. Robust environments are still
summarized within their scenario for acceptance and are never treated as ten
independent scenarios.

The comparison video is 1920x1080: baseline is on the left, current is on the
right, and synchronized forward-speed, yaw-rate, height, and tilt plots occupy
the lower band. A moving time cursor connects the plots to the two views. The
composer validates scenario, phase, command, duration, and frame coverage when
both traces are available; failed env 0 frames are frozen and labelled so an
automatic reset cannot look like recovery. Each side contains nominal followed
by robust. Hashes are not used. Structural metadata, scenario IDs/order, metric
completeness, safety, and the nominal/robust 90% gates remain validated.
In addition to tracking, attitude, survival, and task outcomes, schema v4
records normalized policy-action smoothness separately for the four leg actions
and two wheel actions. It reports RMS first differences (action change per
control step) and RMS second differences (high-frequency action jitter); lower
is better for all four. Only scored samples are used, and the first one/two
samples are omitted where the corresponding difference has no in-window
history. Flat also records height peak-to-peak, vertical-velocity RMS, two-frame
filtered missing wheel contacts per step, wheel-torque saturation rate, and
wheel-velocity-target saturation rate. The nominal and
robust ``vx=+/-2 m/s`` cases run for 10 seconds and have absolute peak-tilt,
height-excursion, and wheel-contact-loss gates. The source-video overlay shows
env 0's actual/commanded forward speed, yaw rate and height plus actual tilt.
Because these fields are required comparison metrics, an older schema baseline
must be regenerated once before evaluating new checkpoints.

Charts and the final comparison video can be regenerated from an existing
evaluation without launching Isaac Sim:

```bash
python scripts/rsl_rl/wyw_verify_presentation.py \
  --report <evaluate_output>/report.json --video
```

To evaluate the same checkpoint iteration across all compatible run directories,
preview the selection first and then run the serial GPU batch:

```bash
python scripts/rsl_rl/run_wyw_verify_batch.py \
  --variant flat --iteration 4999 --dry-run

python scripts/rsl_rl/run_wyw_verify_batch.py \
  --variant flat --iteration 4999 --device cuda:0
```

`--checkpoint-name NAME.pt` selects an exact basename, while `--latest` selects
the numerically highest `model_N.pt` in each run. `--run-glob` limits the run
directories and `--skip-existing` reuses complete reports for the exact
checkpoint. Extra `play_wyw_verify.py` arguments go after `--`. Runs execute
sequentially and continue after policy failures or execution errors by default.
Each evaluation keeps `batch_driver.log`; timestamped JSON and CSV summaries are
written below `logs/rsl_rl/batch_verification/`. Exit status is 0 when all
policies pass, 1 when at least one policy fails acceptance, and 2 for execution
errors or an empty selection.
