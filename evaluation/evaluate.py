import gymnasium as gym
import numpy as np

from agents.dqn_agent import DQNAgent
from utils.obs_processing import process_obs


def _transition_done(terminated: bool) -> bool:
    """Only true terminals should be marked done for replay-style transitions."""
    return terminated

def _is_episode_done(
    terminated: bool,
    truncated: bool,
    step_index: int,
    max_steps: int,
) -> bool:
    return terminated or truncated or (step_index + 1) >= max_steps


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
    dual_obs = getattr(env.unwrapped, "obs_type", None) == "dual"
    returns: list[float] = []
    lengths: list[int] = []
    successes: list[bool] = []

    for _ in range(n_episodes):
        obs_raw, info = env.reset()
        obs, _, _ = process_obs(obs_raw, agent.obs_type, dual_obs)

        total_return = 0.0
        reward = 0.0

        for step in range(max_steps):
            action_mask = env.action_masks()
            action = agent.select_action(obs, action_mask, explore=False)

            obs_raw, reward, terminated, truncated, info = env.step(action)
            obs, _, _ = process_obs(obs_raw, agent.obs_type, dual_obs)

            total_return += reward

            if _is_episode_done(terminated, truncated, step, max_steps):
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


def evaluate_masked_random(
    env: gym.Env,
    n_episodes: int = 100,
    max_steps: int = 10000,
) -> dict:
    """Evaluate a masked-random policy as a regression baseline."""
    returns: list[float] = []
    lengths: list[int] = []
    successes: list[bool] = []

    for episode in range(n_episodes):
        obs_raw, info = env.reset(seed=episode)
        total_return = 0.0
        reward = 0.0

        for step in range(max_steps):
            valid_actions = np.where(env.action_masks())[0]
            action = int(np.random.choice(valid_actions))
            obs_raw, reward, terminated, truncated, info = env.step(action)
            total_return += reward

            if _is_episode_done(terminated, truncated, step, max_steps):
                break

        returns.append(total_return)
        lengths.append(step + 1)
        successes.append(reward > 0)

    return {
        "avg_return": float(np.mean(returns)),
        "win_rate": float(np.mean(successes)),
        "avg_length": float(np.mean(lengths)),
    }


def collect_eval_trajectory(
    agent: DQNAgent,
    env: gym.Env,
    max_steps: int = 10000,
) -> list[dict]:
    """Run one evaluation episode and collect full trajectory with both obs types.

    Used by the dyad training loop for experience sharing.
    The env must use obs_type='dual' so both puzzle_state and pixels are available.
    """
    trajectory: list[dict] = []
    obs_raw, info = env.reset()
    obs, state_discrete, state_rgb = process_obs(obs_raw, agent.obs_type, dual_obs=True)

    for step in range(max_steps):
        action_mask = env.action_masks()
        action = agent.select_action(obs, action_mask, explore=False)

        next_obs_raw, reward, terminated, truncated, next_info = env.step(action)
        next_obs, next_state_discrete, next_state_rgb = process_obs(
            next_obs_raw, agent.obs_type, dual_obs=True
        )

        episode_done = _is_episode_done(terminated, truncated, step, max_steps)
        transition_done = _transition_done(terminated)
        next_action_mask = (
            np.zeros(env.action_space.n, dtype=bool)
            if transition_done
            else np.asarray(env.action_masks(), dtype=bool)
        )

        trajectory.append({
            "state": obs,
            "action": action,
            "reward": reward,
            "next_state": next_obs,
            "done": transition_done,
            "next_action_mask": next_action_mask,
            "state_discrete": state_discrete,
            "next_state_discrete": next_state_discrete,
            "state_rgb": state_rgb,
            "next_state_rgb": next_state_rgb,
        })

        obs = next_obs
        state_discrete = next_state_discrete
        state_rgb = next_state_rgb

        if episode_done:
            break

    return trajectory
