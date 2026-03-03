import gymnasium as gym
import numpy as np

from agents.dqn_agent import DQNAgent


def _process_obs(obs: dict | np.ndarray, agent_obs_type: str, dual_obs: bool) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Extract agent obs and both modality obs from an environment observation.

    Returns (agent_obs, state_discrete, state_rgb).
    """
    if dual_obs:
        state_discrete = obs["puzzle_state"]
        state_rgb = obs["pixels"]
        if agent_obs_type == "rgb":
            agent_obs = state_rgb.astype(np.float32) / 255.0
        else:
            agent_obs = state_discrete
        return agent_obs, state_discrete, state_rgb
    elif agent_obs_type == "rgb":
        return obs["pixels"].astype(np.float32) / 255.0, None, None
    else:
        return obs, None, None


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
        obs, _, _ = _process_obs(obs_raw, agent.obs_type, dual_obs)

        total_return = 0.0

        for step in range(max_steps):
            action_mask = env.action_masks()
            action = agent.select_action(obs, action_mask, explore=False)

            obs_raw, reward, terminated, truncated, info = env.step(action)
            obs, _, _ = _process_obs(obs_raw, agent.obs_type, dual_obs)

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
    The env must use obs_type='dual' so both puzzle_state and pixels are available.
    """
    trajectory: list[dict] = []
    obs_raw, info = env.reset()
    obs, state_discrete, state_rgb = _process_obs(obs_raw, agent.obs_type, dual_obs=True)

    for _ in range(max_steps):
        action_mask = env.action_masks()
        action = agent.select_action(obs, action_mask, explore=False)

        next_obs_raw, reward, terminated, truncated, next_info = env.step(action)
        next_obs, next_state_discrete, next_state_rgb = _process_obs(
            next_obs_raw, agent.obs_type, dual_obs=True
        )

        done = terminated or truncated

        trajectory.append({
            "state": obs,
            "action": action,
            "reward": reward,
            "next_state": next_obs,
            "done": done,
            "state_discrete": state_discrete,
            "next_state_discrete": next_state_discrete,
            "state_rgb": state_rgb,
            "next_state_rgb": next_state_rgb,
        })

        obs = next_obs
        state_discrete = next_state_discrete
        state_rgb = next_state_rgb

        if done:
            break

    return trajectory
