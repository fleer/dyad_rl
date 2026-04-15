import gymnasium as gym
import numpy as np

from agents.dqn_agent import DQNAgent
from utils.obs_processing import process_obs


def _sem(values: np.ndarray) -> float:
    """Compute Standard Error.

    Computes the standard error of the mean for a one-dimensional array.

    Args:
        values (np.ndarray): Input numeric array.

    Returns:
        float: Standard error of the mean, or ``0.0`` for arrays of size one
        or less.
    """
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def evaluate(
    agent: DQNAgent,
    env: gym.Env,
    n_episodes: int = 1000,
    max_steps: int = 10000,
    reward_step_penalty: float = 0.0,
) -> dict:
    """Evaluate Agent Policy.

    Runs evaluation episodes with a greedy policy (no exploration).

    Args:
        agent (DQNAgent): Trained DQN agent.
        env (gym.Env): Gymnasium environment.
        n_episodes (int): Number of evaluation episodes.
        max_steps (int): Maximum steps per episode.
        reward_step_penalty (float): Per-step shaping penalty subtracted from
            environment reward for return accounting.

    Returns:
        dict: Aggregated return, win-rate, length, and uncertainty metrics.
    """
    returns: list[float] = []
    lengths: list[int] = []
    successes: list[bool] = []

    for _ in range(n_episodes):
        obs_raw, info = env.reset()
        obs, _, _ = process_obs(obs_raw, agent.obs_type)

        total_return = 0.0
        reward = 0.0

        for step in range(max_steps):
            action_mask = env.action_masks()
            action = agent.select_action(obs, action_mask, explore=False)

            obs_raw, reward, terminated, truncated, info = env.step(action)
            obs, _, _ = process_obs(obs_raw, agent.obs_type)

            shaped_reward = reward - reward_step_penalty
            total_return += shaped_reward

            if terminated or truncated:
                break

        returns.append(total_return)
        lengths.append(step + 1)
        successes.append(reward > 0)

    returns_arr = np.array(returns)
    lengths_arr = np.array(lengths)
    successes_arr = np.array(successes)

    success_lengths = lengths_arr[successes_arr]
    success_lengths_arr = np.asarray(success_lengths, dtype=float)

    return {
        "avg_return": float(returns_arr.mean()),
        "sem_return": _sem(returns_arr),
        "win_rate": float(successes_arr.mean()),
        "sem_win_rate": _sem(successes_arr.astype(float)),
        "avg_length": float(lengths_arr.mean()),
        "sem_length": _sem(lengths_arr),
        "avg_success_length": float(success_lengths.mean())
        if len(success_lengths) > 0
        else 0.0,
        "sem_success_length": _sem(success_lengths_arr),
        "std_length": float(lengths_arr.std()),
    }


def collect_eval_trajectory(
    agent: DQNAgent,
    env: gym.Env,
    max_steps: int = 10000,
) -> list[dict]:
    """Collect Evaluation Trajectory.

    Runs one greedy evaluation episode and collects transition dictionaries for
    dyad experience sharing.

    Args:
        agent (DQNAgent): Agent used to select actions.
        env (gym.Env): Environment configured with dual observations.
        max_steps (int): Maximum steps for the trajectory.

    Returns:
        list[dict]: Collected transition records for one episode.
    """
    trajectory: list[dict] = []
    obs_raw, info = env.reset(seed=42)
    obs, state_discrete, state_rgb = process_obs(obs_raw, agent.obs_type)

    for step in range(max_steps):
        action_mask = env.action_masks()
        action = agent.select_action(obs, action_mask, explore=False)

        next_obs_raw, reward, terminated, truncated, next_info = env.step(action)
        next_obs, next_state_discrete, next_state_rgb = process_obs(
            next_obs_raw, agent.obs_type
        )

        episode_done = terminated or truncated
        next_action_mask = (
            np.zeros(env.action_space.n, dtype=bool)
            if episode_done
            else np.asarray(env.action_masks(), dtype=bool)
        )

        trajectory.append(
            {
                "action": action,
                "reward": reward,
                "done": episode_done,
                "next_action_mask": next_action_mask,
                "state_discrete": state_discrete,
                "next_state_discrete": next_state_discrete,
                "state_rgb": state_rgb,
                "next_state_rgb": next_state_rgb,
            }
        )

        obs = next_obs
        state_discrete = next_state_discrete
        state_rgb = next_state_rgb

        if episode_done:
            break

    return trajectory
