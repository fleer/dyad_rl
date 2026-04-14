import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TransformObservation
from gymnasium import spaces, Space
from gymnasium.core import ActType, ObsType, WrapperObsType


class FlattenObservationDual(
    TransformObservation[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Flattens the environment's observation space and each observation from ``reset`` and ``step`` functions.

    A vector version of the wrapper exists :class:`gymnasium.wrappers.vector.FlattenObservation`.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import FlattenObservation
        >>> env = gym.make("CarRacing-v3")
        >>> env.observation_space.shape
        (96, 96, 3)
        >>> env = FlattenObservation(env)
        >>> env.observation_space.shape
        (27648,)
        >>> obs, _ = env.reset()
        >>> obs.shape
        (27648,)

    Change logs:
     * v0.15.0 - Initially added
    """

    def __init__(self, env: gym.Env[ObsType, ActType]):
        """Constructor for any environment's observation space that implements ``spaces.utils.flatten_space`` and ``spaces.utils.flatten``.

        Args:
            env:  The environment to wrap
        """

        gym.utils.RecordConstructorArgs.__init__(self)
        TransformObservation.__init__(
            self,
            env=env,
            # func=lambda obs: spaces.utils.flatten(
            #     env.observation_space["puzzle_state"], obs
            # ),
            func=lambda obs: {
                "puzzle_state": spaces.utils.flatten(
                    env.observation_space["puzzle_state"], obs["puzzle_state"]
                ),
                "pixels": obs["pixels"],
            },
            observation_space=spaces.utils.flatten_space(
                env.observation_space["puzzle_state"]
            ),
        )

    # def observation(obs_space: Space[ObsType], obs: ObsType) -> ObsType:
    #     """Observation method for the wrapper.
    #
    #     Args:
    #         obs: The observation to transform
    #     """
    #
    #     return {
    #         "puzzle_state": spaces.utils.flatten(obs_space["puzzle_state"], obs),
    #         "pixels": obs["pixels"],
    #     }


def normalize_rgb(obs: np.ndarray) -> np.ndarray:
    """Normalize RGB Observation.

    Converts uint8 RGB observations in ``[0, 255]`` to float32 in ``[0, 1]``.

    Args:
        obs (np.ndarray): RGB observation array.

    Returns:
        np.ndarray: Normalized float32 RGB array.
    """
    return obs.astype(np.float32) / 255.0


def process_obs(
    obs: dict | np.ndarray,
    agent_obs_type: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Process Observation.

    Extracts the agent-specific observation and optionally returns both raw
    modality branches for dyad sharing.

    Args:
        obs (dict | np.ndarray): Raw environment observation.
        agent_obs_type (str): Agent observation type (``"rgb"`` or
            ``"puzzle_state"``).

    Returns:
        tuple[np.ndarray, np.ndarray | None, np.ndarray | None]: Agent
        observation, discrete-state branch, and RGB branch.
    """
    state_discrete = np.asarray(obs["puzzle_state"], dtype=np.float32)
    state_rgb = normalize_rgb(obs["pixels"])
    if agent_obs_type == "rgb":
        agent_obs = state_rgb
    else:
        agent_obs = state_discrete
    return agent_obs, state_discrete, state_rgb
