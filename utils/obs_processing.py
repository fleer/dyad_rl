import gymnasium as gym
import numpy as np
from gymnasium.spaces.utils import flatten_space


_MAX_LINEAR_SPAN = 1024.0
_FALLBACK_LOG_SCALE = 256.0


def normalize_rgb(obs: np.ndarray) -> np.ndarray:
    return obs.astype(np.float32) / 255.0


def _normalize_puzzle_state_vector(
    obs: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)

    out = obs.copy()
    spans = high - low
    linear_mask = (
        np.isfinite(low)
        & np.isfinite(high)
        & (spans > 0)
        & (spans <= _MAX_LINEAR_SPAN)
    )

    if np.any(linear_mask):
        out[linear_mask] = (
            2.0 * (obs[linear_mask] - low[linear_mask]) / spans[linear_mask]
        ) - 1.0

    fallback_mask = ~linear_mask
    if np.any(fallback_mask):
        non_negative_mask = fallback_mask & np.isfinite(low) & (low >= 0)
        if np.any(non_negative_mask):
            relative = np.maximum(obs[non_negative_mask] - low[non_negative_mask], 0.0)
            out[non_negative_mask] = (
                2.0 * np.log1p(relative) / np.log1p(_FALLBACK_LOG_SCALE)
            ) - 1.0

        signed_mask = fallback_mask & ~non_negative_mask
        if np.any(signed_mask):
            signed = np.sign(obs[signed_mask]) * np.log1p(np.abs(obs[signed_mask]))
            out[signed_mask] = signed / np.log1p(_FALLBACK_LOG_SCALE)

    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _get_puzzle_state_bounds(env: gym.Env) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(env.unwrapped, "_ps_obs_space"):
        flat_space = flatten_space(env.unwrapped._ps_obs_space)
        return (
            np.asarray(flat_space.low, dtype=np.float32),
            np.asarray(flat_space.high, dtype=np.float32),
        )

    obs_space = env.observation_space
    return (
        np.asarray(obs_space.low, dtype=np.float32),
        np.asarray(obs_space.high, dtype=np.float32),
    )


class NormalizePuzzleStateWrapper(gym.ObservationWrapper):
    """Normalize flattened puzzle-state observations into a stable float range."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._low, self._high = _get_puzzle_state_bounds(env)
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return _normalize_puzzle_state_vector(obs, self._low, self._high)


class NormalizeDualPuzzleStateWrapper(gym.ObservationWrapper):
    """Normalize the `puzzle_state` field while preserving dual observations."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._low, self._high = _get_puzzle_state_bounds(env)
        self.observation_space = gym.spaces.Dict(
            {
                "puzzle_state": gym.spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=env.observation_space["puzzle_state"].shape,
                    dtype=np.float32,
                ),
                "pixels": env.observation_space["pixels"],
            }
        )

    def observation(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        normalized = dict(obs)
        normalized["puzzle_state"] = _normalize_puzzle_state_vector(
            obs["puzzle_state"], self._low, self._high
        )
        return normalized


def process_obs(
    obs: dict | np.ndarray,
    agent_obs_type: str,
    dual_obs: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Extract agent observations with consistent preprocessing."""
    if dual_obs:
        state_discrete = np.asarray(obs["puzzle_state"], dtype=np.float32)
        state_rgb = obs["pixels"]
        if agent_obs_type == "rgb":
            agent_obs = normalize_rgb(state_rgb)
        else:
            agent_obs = state_discrete
        return agent_obs, state_discrete, state_rgb

    if agent_obs_type == "rgb":
        return normalize_rgb(obs["pixels"]), None, None

    return np.asarray(obs, dtype=np.float32), None, None
