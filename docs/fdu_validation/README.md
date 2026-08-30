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
the Rough report additionally records a terrain level change across reset.
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
