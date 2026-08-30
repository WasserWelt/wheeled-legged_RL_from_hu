# FDU model validation outputs

This directory keeps generated validation artifacts out of the `docs/` root.

- `geometry/`: analytical/physics geometry calibration reports.
- `jitter/timestep/`: valid physics-frequency and short-leg boundary scans.
- `jitter/solver/`: valid iteration/damping solver A/B scans.
- `drop/`: controlled free-drop and guided-vertical impact reports.
- `video/`: current accepted geometry and drop-test recordings.
- `jump/`: jump-specific calibration reports.
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
