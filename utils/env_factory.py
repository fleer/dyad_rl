import gymnasium as gym
import numpy as np
import rlp  # noqa: F401 — registers rlp/Puzzle-v0
from omegaconf import DictConfig

from utils.obs_processing import (
    FlattenObservationDual,
)


class ActionMaskWrapper(gym.Wrapper):
    """Ensures action_masks() is accessible through any wrapper chain."""

    def action_masks(self) -> np.ndarray:
        """Get Action Mask.

        Forwards action-mask retrieval to the unwrapped base environment.

        Args:
            None: This method reads wrapped environment state.

        Returns:
            np.ndarray: Boolean mask indicating valid actions.
        """
        return self.env.unwrapped.action_masks()


def make_dual_obs_env(cfg: DictConfig) -> gym.Env:
    """Create Dual-Observation Environment.

    Builds a puzzle environment that returns both normalized puzzle-state and
    RGB observations at each step.

    Args:
        cfg (DictConfig): Environment and wrapper configuration.

    Returns:
        gym.Env: Wrapped dual-observation environment.
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
    env = FlattenObservationDual(env)
    env = ActionMaskWrapper(env)
    env.reset(seed=cfg.seed)
    return env
