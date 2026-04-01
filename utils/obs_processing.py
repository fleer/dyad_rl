import gymnasium as gym
import numpy as np
from gymnasium.spaces.utils import flatten_space


_MAX_LINEAR_SPAN = 1024.0
_FALLBACK_LOG_SCALE = 256.0


def normalize_rgb(obs: np.ndarray) -> np.ndarray:
    """Normalize RGB Observation.

    Converts uint8 RGB observations in ``[0, 255]`` to float32 in ``[0, 1]``.

    Args:
        obs (np.ndarray): RGB observation array.

    Returns:
        np.ndarray: Normalized float32 RGB array.
    """
    return obs.astype(np.float32) / 255.0


def _normalize_puzzle_state_vector(
    obs: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    """Normalize Puzzle-State Vector.

    Normalizes puzzle-state features to a stable range using linear scaling when
    bounds are well-behaved and logarithmic fallback otherwise.

    Args:
        obs (np.ndarray): Raw puzzle-state vector.
        low (np.ndarray): Lower bounds per feature.
        high (np.ndarray): Upper bounds per feature.

    Returns:
        np.ndarray: Normalized puzzle-state vector clipped to ``[-1, 1]``.
    """
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
    """Get Puzzle-State Bounds.

    Retrieves flattened lower and upper bounds for puzzle-state features.

    Args:
        env (gym.Env): Environment providing observation spaces.

    Returns:
        tuple[np.ndarray, np.ndarray]: Lower and upper bounds arrays.
    """
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
        """Initialize Puzzle-State Normalization Wrapper.

        Configures normalized observation bounds and output space.

        Args:
            env (gym.Env): Environment with flattened puzzle-state observations.

        Returns:
            None: Wrapper state is initialized in place.
        """
        super().__init__(env)
        self._low, self._high = _get_puzzle_state_bounds(env)
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Normalize Wrapped Observation.

        Applies puzzle-state normalization to an observation from the wrapped
        environment.

        Args:
            obs (np.ndarray): Raw puzzle-state observation.

        Returns:
            np.ndarray: Normalized puzzle-state observation.
        """
        return _normalize_puzzle_state_vector(obs, self._low, self._high)


class NormalizeDualPuzzleStateWrapper(gym.ObservationWrapper):
    """Normalize the `puzzle_state` field while preserving dual observations."""

    def __init__(self, env: gym.Env):
        """Initialize Dual-Observation Normalization Wrapper.

        Configures normalization for the puzzle-state branch while preserving
        RGB observations.

        Args:
            env (gym.Env): Environment producing dual observations.

        Returns:
            None: Wrapper state is initialized in place.
        """
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
        """Normalize Dual Observation.

        Normalizes the ``puzzle_state`` field and keeps pixel data unchanged.

        Args:
            obs (dict[str, np.ndarray]): Dual observation dictionary.

        Returns:
            dict[str, np.ndarray]: Dual observation with normalized puzzle-state.
        """
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
    """Process Observation.

    Extracts the agent-specific observation and optionally returns both raw
    modality branches for dyad sharing.

    Args:
        obs (dict | np.ndarray): Raw environment observation.
        agent_obs_type (str): Agent observation type (``"rgb"`` or
            ``"puzzle_state"``).
        dual_obs (bool): Whether the environment returns dual observations.

    Returns:
        tuple[np.ndarray, np.ndarray | None, np.ndarray | None]: Agent
        observation, discrete-state branch, and RGB branch.
    """
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
