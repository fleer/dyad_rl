import gymnasium as gym
import numpy as np

from agents.dqn_agent import DQNAgent


def evaluate(
    agent: DQNAgent,
    env: gym.Env,
    n_episodes: int = 1000,
    max_steps: int = 10000
) -> dict:
    """Run evaluation episodes with greedy policy (no exploration).

    Args:
        agent: Trained DQN agent.
        env: The Gymnasium environment.
        n_episodes: Number of evaluation episodes.
        max_steps: Max steps per episode to prevent infinite loops.

    Returns:
        Dict with avg_return, win_rate, avg_length, avg_success_length, std_length.
    """
    returns: list[float] = []
    lengths: list[int] = []
    successes: list[bool] = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        if agent.obs_type == "rgb":
            obs = obs["pixels"].astype(np.float32) / 255.0

        total_return = 0.0

        for step in range(max_steps):
            action_mask = env.action_masks()
            action = agent.select_action(obs, action_mask, explore=False)

            obs, reward, terminated, truncated, info = env.step(action)
            if agent.obs_type == "rgb":
                obs = obs["pixels"].astype(np.float32) / 255.0

            total_return += reward

            if terminated or truncated:
                break

        returns.append(total_return)
        lengths.append(step + 1)
        successes.append(reward > 0)

    returns_arr = np.array(returns)
    lengths_arr = np.array(lengths)
    successes_arr = np.array(successes)

    success_lengths = lengths_arr[successes_arr]

    return {
        "avg_return": float(returns_arr.mean()),
        "win_rate": float(successes_arr.mean()),
        "avg_length": float(lengths_arr.mean()),
        "avg_success_length": float(success_lengths.mean()) if len(success_lengths) > 0 else 0.0,
        "std_length": float(lengths_arr.std()),
    }


def collect_eval_trajectory(
    agent: DQNAgent,
    env: gym.Env,
    max_steps: int = 10000,
) -> list[dict]:
    """Run one evaluation episode and collect full trajectory with both obs types.

    Used by the dyad training loop for experience sharing.
    Each element contains: state, action, reward, next_state, done,
    plus obs_discrete and obs_rgb from info if available.
    """
    trajectory: list[dict] = []
    obs, info = env.reset()
    if agent.obs_type == "rgb":
        obs_proc = obs["pixels"].astype(np.float32) / 255.0
    else:
        obs_proc = obs

    for _ in range(max_steps):
        action_mask = env.action_masks()
        action = agent.select_action(obs_proc, action_mask, explore=False)

        next_obs, reward, terminated, truncated, next_info = env.step(action)
        if agent.obs_type == "rgb":
            next_obs_proc = next_obs["pixels"].astype(np.float32) / 255.0
        else:
            next_obs_proc = next_obs

        done = terminated or truncated

        trajectory.append({
            "state": obs_proc,
            "action": action,
            "reward": reward,
            "next_state": next_obs_proc,
            "done": done,
            "state_discrete": info.get("obs_discrete"),
            "next_state_discrete": next_info.get("obs_discrete"),
            "state_rgb": info.get("obs_rgb"),
            "next_state_rgb": next_info.get("obs_rgb"),
        })

        obs_proc = next_obs_proc
        info = next_info

        if done:
            break

    return trajectory
