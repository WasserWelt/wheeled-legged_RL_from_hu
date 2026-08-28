"""Exact Fudan rough-terrain generator translated to Isaac Lab."""

from __future__ import annotations

import numpy as np

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.height_field import hf_terrains
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.terrains.terrain_generator import TerrainGenerator
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass


class FduTerrainGenerator(TerrainGenerator):
    """Isaac Lab generator with Fudan's exact ``row / num_rows`` difficulty."""

    def _generate_curriculum_terrains(self):
        proportions = np.asarray([cfg.proportion for cfg in self.cfg.sub_terrains.values()], dtype=np.float64)
        proportions /= proportions.sum()
        cumulative = np.cumsum(proportions)
        sub_indices = np.asarray(
            [np.min(np.where(col / self.cfg.num_cols + 0.001 < cumulative)[0]) for col in range(self.cfg.num_cols)],
            dtype=np.int32,
        )
        sub_cfgs = list(self.cfg.sub_terrains.values())
        lower, upper = self.cfg.difficulty_range
        for col in range(self.cfg.num_cols):
            for row in range(self.cfg.num_rows):
                difficulty = lower + (upper - lower) * (row / self.cfg.num_rows)
                mesh, origin = self._get_terrain_mesh(difficulty, sub_cfgs[sub_indices[col]])
                self._add_sub_terrain(mesh, origin, row, col, sub_cfgs[sub_indices[col]])


@height_field_to_mesh
def fdu_rough_slope_terrain(difficulty: float, cfg: "FduRoughSlopeTerrainCfg") -> np.ndarray:
    """Fudan rough slope: half-strength pyramid slope plus random-uniform height."""

    slope = hf_terrains.pyramid_sloped_terrain.__wrapped__(difficulty, cfg)
    noise_cfg = cfg.copy()
    random_height = 0.05 + 0.05 * difficulty
    noise_cfg.noise_range = (-random_height, random_height)
    noise = hf_terrains.random_uniform_terrain.__wrapped__(difficulty, noise_cfg)
    return (slope.astype(np.int32) + noise.astype(np.int32)).astype(np.int16)


@configclass
class FduRoughSlopeTerrainCfg(terrain_gen.HfPyramidSlopedTerrainCfg):
    """Configuration for Fudan's combined sloped and noisy height field."""

    function = fdu_rough_slope_terrain
    noise_range: tuple[float, float] = (-0.05, 0.05)
    noise_step: float = 0.005
    downsampled_scale: float = 0.2


# In curriculum mode columns are assigned by cumulative proportions.  Splitting
# positive and negative slopes preserves Fudan's 50/50 direction choice inside
# the smooth- and rough-slope branches.
FDU_ROUGH_TERRAIN_CFG = TerrainGeneratorCfg(
    class_type=FduTerrainGenerator,
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
        # Fudan applies a 0.5 multiplier to the branch slope and then adds
        # random_uniform_terrain with +/- (0.05 + 0.05 * difficulty) m.
        "rough_slope_up": FduRoughSlopeTerrainCfg(
            proportion=0.10, slope_range=(0.0, 0.25), platform_width=3.0
        ),
        "rough_slope_down": FduRoughSlopeTerrainCfg(
            proportion=0.10, slope_range=(0.0, 0.25), platform_width=3.0, inverted=True
        ),
        "stairs_down": terrain_gen.HfInvertedPyramidStairsTerrainCfg(
            proportion=0.10, step_height_range=(0.05, 0.23), step_width=0.7, platform_width=4.0
        ),
        "stairs_up": terrain_gen.HfPyramidStairsTerrainCfg(
            proportion=0.20, step_height_range=(0.05, 0.23), step_width=0.7, platform_width=4.0
        ),
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.10,
            # Isaac Gym samples {-h, -h/2, h/2, h}; Isaac Lab's choice mode
            # implements the same distribution.
            obstacle_height_mode="choice",
            obstacle_width_range=(1.0, 2.0),
            obstacle_height_range=(0.05, 0.15),
            num_obstacles=20,
            platform_width=3.0,
        ),
    },
)
