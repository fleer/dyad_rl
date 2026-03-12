import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import numpy as np
from omegaconf import DictConfig

import rlp  # noqa: F401 — registers rlp/Puzzle-v0
from utils.obs_processing import (
    NormalizeDualPuzzleStateWrapper,
    NormalizePuzzleStateWrapper,
)


class ActionMaskWrapper(gym.Wrapper):
    """Ensures action_masks() is accessible through any wrapper chain."""

    def action_masks(self) -> np.ndarray:
        return self.env.unwrapped.action_masks()


def make_env(cfg: DictConfig) -> gym.Env:
    """Create a PuzzleEnv from Hydra config.

    For puzzle_state obs_type, wraps with FlattenObservation to produce a 1-D vector.
    """
    env = gym.make(
        "rlp/Puzzle-v0",
        puzzle=cfg.env.puzzle,
        render_mode=cfg.env.render_mode,
        obs_type=cfg.env.obs_type,
        window_width=cfg.env.window_width,
        window_height=cfg.env.window_height,
        allow_undo=cfg.env.allow_undo,
        max_state_repeats=cfg.env.max_state_repeats,
        include_cursor_in_state_info=cfg.env.include_cursor_in_state_info,
        params=cfg.env.params,
    )
    if cfg.env.obs_type == "puzzle_state":
        env = FlattenObservation(env)
        env = NormalizePuzzleStateWrapper(env)
    env = ActionMaskWrapper(env)
    return env


def make_dual_obs_env(cfg: DictConfig) -> gym.Env:
    """Create a PuzzleEnv with obs_type='dual'.

    Returns both puzzle_state (flattened) and RGB observations in every step.
    """
    env = gym.make(
        "rlp/Puzzle-v0",
        puzzle=cfg.env.puzzle,
        render_mode=cfg.env.render_mode,
        obs_type="dual",
        window_width=cfg.env.window_width,
        window_height=cfg.env.window_height,
        allow_undo=cfg.env.allow_undo,
        max_state_repeats=cfg.env.max_state_repeats,
        include_cursor_in_state_info=cfg.env.include_cursor_in_state_info,
        params=cfg.env.params,
    )
    env = NormalizeDualPuzzleStateWrapper(env)
    env = ActionMaskWrapper(env)
    return env
