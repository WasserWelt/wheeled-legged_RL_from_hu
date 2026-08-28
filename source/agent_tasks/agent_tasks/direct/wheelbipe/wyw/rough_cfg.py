"""Fudan rough-terrain generator translated to Isaac Lab named sub-terrains."""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg


# In curriculum mode columns are assigned by cumulative proportions.  Splitting
# positive and negative slopes preserves Fudan's 50/50 direction choice inside
# the smooth- and rough-slope branches.
FDU_ROUGH_TERRAIN_CFG = TerrainGeneratorCfg(
    seed=None,
    curriculum=True,
    size=(8.0, 8.0),
    border_width=25.0,
    num_rows=10,
    num_cols=20,
    color_scheme="none",
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    cache_dir="/tmp/isaaclab/fdu_rough_terrains",
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.20),
        "smooth_slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.10, slope_range=(0.0, 0.5), platform_width=3.0
        ),
        "smooth_slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.10, slope_range=(0.0, 0.5), platform_width=3.0
        ),
        # Fudan applies a 0.5 multiplier to the rough-slope branch.
        "rough_slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.10, slope_range=(0.0, 0.25), platform_width=3.0
        ),
        "rough_slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.10, slope_range=(0.0, 0.25), platform_width=3.0
        ),
        "stairs_down": terrain_gen.HfInvertedPyramidStairsTerrainCfg(
            proportion=0.10, step_height_range=(0.05, 0.23), step_width=0.7, platform_width=4.0
        ),
        "stairs_up": terrain_gen.HfPyramidStairsTerrainCfg(
            proportion=0.20, step_height_range=(0.05, 0.23), step_width=0.7, platform_width=4.0
        ),
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.10,
            obstacle_height_mode="fixed",
            obstacle_width_range=(1.0, 2.0),
            obstacle_height_range=(0.05, 0.15),
            num_obstacles=20,
            platform_width=3.0,
        ),
    },
)

