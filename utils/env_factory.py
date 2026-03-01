import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import numpy as np
from omegaconf import DictConfig

import rlp  # noqa: F401 — registers rlp/Puzzle-v0


class ActionMaskWrapper(gym.Wrapper):
    """Ensures action_masks() is accessible through any wrapper chain."""

    def action_masks(self) -> np.ndarray:
        return self.env.unwrapped.action_masks()


class DualObsWrapper(gym.Wrapper):
    """Wraps a PuzzleEnv to capture both RGB and discrete observations on every step.

    Regardless of the underlying env's obs_type, this wrapper always provides
    both observation formats in the info dict under 'obs_discrete' and 'obs_rgb'.
    The primary observation returned follows the base env's obs_type.

    'obs_discrete' is the gymnasium-flattened puzzle_state observation (matching
    what FlattenObservation would produce), and 'obs_rgb' is the raw RGB pixels
    normalized to float32 [0, 1] matching CNN agent input format.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Cache the puzzle_state observation space for proper flattening
        from rlp.envs import observation_spaces as obs_spaces
        base_env = self.env.unwrapped
        self._ps_obs_space = gym.spaces.Dict(
            obs_spaces.get_observation_space(
                base_env.puzzle_name,
                base_env.puzzle.fe.contents.me.contents.states[0].state.contents,
                None if base_env.puzzle_name in __import__('rlp').api.specific.ui_reset_never
                else base_env.puzzle.fe.contents.me.contents.ui,
            )
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["obs_discrete"] = self._get_discrete_obs()
        info["obs_rgb"] = self._get_rgb_obs()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["obs_discrete"] = self._get_discrete_obs()
        info["obs_rgb"] = self._get_rgb_obs()
        return obs, reward, terminated, truncated, info

    def _get_discrete_obs(self) -> np.ndarray:
        """Get the puzzle_state observation, flattened consistently with FlattenObservation."""
        from rlp.envs import observation_spaces as obs_spaces
        base_env = self.env.unwrapped
        if not base_env._state_dict_up_to_date:
            base_env._update_state_dict()
        obs_dict = obs_spaces.get_observation(
            base_env.puzzle_name, base_env.state_dict,
            None if base_env.include_cursor_in_state_info
            else __import__('rlp').api.specific.get_cursor_coords(
                base_env.puzzle_name,
                base_env.puzzle.fe.contents.me.contents,
            )
        )
        return gym.spaces.utils.flatten(self._ps_obs_space, obs_dict).astype(np.float32)

    def _get_rgb_obs(self) -> np.ndarray:
        """Capture current RGB pixels from the pygame surface."""
        import pygame
        surf = self.env.unwrapped.puzzle.surf
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(surf)), axes=(2, 1, 0)
        ).astype(np.uint8)


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
    env = ActionMaskWrapper(env)
    return env


def make_dual_obs_env(cfg: DictConfig) -> gym.Env:
    """Create a PuzzleEnv wrapped with DualObsWrapper.

    Always provides both RGB and discrete observations in info,
    and flattens the primary observation if obs_type is puzzle_state.
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
    env = DualObsWrapper(env)
    if cfg.env.obs_type == "puzzle_state":
        env = FlattenObservation(env)
    env = ActionMaskWrapper(env)
    return env
