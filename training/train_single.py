import logging
import os

import gymnasium as gym
import numpy as np
from omegaconf import DictConfig
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from agents.dqn_agent import DQNAgent
from utils.metrics import MetricsLogger, EvalRecord
from utils.obs_processing import process_obs
from evaluation.evaluate import evaluate

log = logging.getLogger(__name__)


def prefill_buffer(
    agent: DQNAgent,
    env: gym.Env,
    num_transitions: int,
    max_steps: int,
    reward_step_penalty: float = 0.0,
) -> None:
    """Prefill Replay Buffer.

    Fills replay memory with random valid transitions before learning starts.

    Args:
        agent (DQNAgent): Agent whose replay buffer is filled.
        env (gym.Env): Training environment.
        num_transitions (int): Number of transitions to collect.
        max_steps (int): Maximum steps per episode during prefill.
        reward_step_penalty (float): Per-step shaping penalty subtracted from
            reward.

    Returns:
        None: Transitions are written to the replay buffer.
    """
    log.info(f"Prefilling replay buffer with {num_transitions} random transitions...")
    collected = 0
    with tqdm(total=num_transitions) as pbar:
        while collected < num_transitions:
            obs_raw, _ = env.reset()
            obs, state_discrete, state_rgb = process_obs(obs_raw, agent.obs_type)

            for _ in range(max_steps):
                action_mask = env.action_masks()
                valid_actions = np.where(action_mask)[0]
                action = int(np.random.choice(valid_actions))

                next_obs_raw, reward, terminated, truncated, next_info = env.step(
                    action
                )
                shaped_reward = reward - reward_step_penalty
                next_obs, next_state_discrete, next_state_rgb = process_obs(
                    next_obs_raw, agent.obs_type
                )

                episode_done = terminated or truncated
                next_action_mask = (
                    np.zeros(env.action_space.n, dtype=bool)
                    if episode_done
                    else np.asarray(env.action_masks(), dtype=bool)
                )

                agent.replay_buffer.push(
                    action,
                    shaped_reward,
                    episode_done,
                    next_action_mask,
                    state_discrete,
                    next_state_discrete,
                    state_rgb,
                    next_state_rgb,
                )
                state_discrete = next_state_discrete
                state_rgb = next_state_rgb
                collected += 1
                pbar.update(1)

                if episode_done or collected >= num_transitions:
                    break
    log.info(f"Prefilled buffer with {len(agent.replay_buffer)} transitions.")


def _run_episode(
    agent: DQNAgent,
    env: gym.Env,
    max_steps: int,
    reward_step_penalty: float = 0.0,
    train_freq: int = 4,
    gradient_steps: int = 1,
) -> tuple[float, int, bool, float]:
    """Run Training Episode.

    Executes one environment episode and performs in-loop DQN optimization.

    Args:
        agent (DQNAgent): Agent to train.
        env (gym.Env): Training environment.
        max_steps (int): Maximum environment steps.
        reward_step_penalty (float): Per-step shaping penalty.
        train_freq (int): Number of environment steps between optimization
            calls.
        gradient_steps (int): Number of optimizer steps per train call.

    Returns:
        tuple[float, int, bool, float, dict[str, float]]: Episode return,
        episode length, success flag, average loss, and average optimizer
        diagnostics.
    """
    obs_raw, info = env.reset()
    obs, state_discrete, state_rgb = process_obs(obs_raw, agent.obs_type)

    total_return = 0.0
    total_loss = 0.0
    loss_count = 0

    for step in range(max_steps):
        action_mask = env.action_masks()
        action = agent.select_action(obs, action_mask, explore=True)

        next_obs_raw, reward, terminated, truncated, next_info = env.step(action)
        shaped_reward = reward - reward_step_penalty
        next_obs, next_state_discrete, next_state_rgb = process_obs(
            next_obs_raw, agent.obs_type
        )

        episode_done = terminated or truncated
        next_action_mask = (
            np.zeros(env.action_space.n, dtype=bool)
            if episode_done
            else np.asarray(env.action_masks(), dtype=bool)
        )

        agent.replay_buffer.push(
            action,
            shaped_reward,
            episode_done,
            next_action_mask,
            state_discrete,
            next_state_discrete,
            state_rgb,
            next_state_rgb,
        )

        # Optimize every train_freq steps (mirrors SB3 train_freq logic)
        if (
            agent.steps_done % train_freq == 0
            and len(agent.replay_buffer) >= agent.batch_size
        ):
            for _ in range(gradient_steps):
                loss = agent.optimize()
                if loss > 0:
                    total_loss += loss
                    loss_count += 1
        # Keep target-update cadence tied to environment steps (SB3-style).
        agent.update_target_net()

        total_return += shaped_reward
        obs = next_obs
        state_discrete = next_state_discrete
        state_rgb = next_state_rgb

        if episode_done:
            break

    success = reward > 0  # +100 for solved
    avg_loss = total_loss / max(loss_count, 1)
    return total_return, step + 1, success, avg_loss


def train_single(
    agent: DQNAgent,
    env: gym.Env,
    cfg: DictConfig,
    logger: MetricsLogger,
    checkpoint_dir: str = "checkpoints",
) -> float:
    """Train Single DQN Agent.

    Trains one DQN agent, logs metrics, evaluates periodically, and saves
    checkpoints.

    Args:
        agent (DQNAgent): DQN agent to train.
        env (gym.Env): Wrapped training environment.
        cfg (DictConfig): Training configuration.
        logger (MetricsLogger): Metrics logger instance.
        checkpoint_dir (str): Directory for model checkpoints.

    Returns:
        float: Best evaluation win rate achieved.
    """
    total_episodes = cfg.training.total_episodes
    max_steps = cfg.training.max_steps
    eval_interval = cfg.training.eval_interval
    eval_episodes = cfg.training.eval_episodes
    checkpoint_interval = cfg.training.checkpoint_interval
    log_interval = cfg.training.log_interval
    reward_step_penalty = float(cfg.training.get("reward_step_penalty", 0.0))
    train_freq = int(cfg.agent.get("train_freq", 4))
    gradient_steps = int(cfg.agent.get("gradient_steps", 1))

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_win_rate = -1.0

    learning_starts = int(cfg.agent.get("learning_starts", 100))
    if learning_starts > 0:
        prefill_buffer(agent, env, learning_starts, max_steps, reward_step_penalty)

    eval_result = evaluate(
        agent,
        env,
        n_episodes=eval_episodes,
        max_steps=max_steps,
        reward_step_penalty=reward_step_penalty,
    )
    eval_record = EvalRecord(
        episode=0,
        avg_return=eval_result["avg_return"],
        sem_return=eval_result["sem_return"],
        win_rate=eval_result["win_rate"],
        sem_win_rate=eval_result["sem_win_rate"],
        avg_length=eval_result["avg_length"],
        sem_length=eval_result["sem_length"],
        avg_success_length=eval_result["avg_success_length"],
        sem_success_length=eval_result["sem_success_length"],
        std_length=eval_result["std_length"],
    )
    logger.log_eval(eval_record)
    log.info(
        f"  EVAL @ {0}: Win Rate={eval_result['win_rate']:.3f} | "
        f"Avg Return={eval_result['avg_return']:.1f} | "
        f"Avg Length={eval_result['avg_length']:.0f} | "
    )

    with logging_redirect_tqdm(loggers=[logging.getLogger()]):
        with tqdm(
            range(1, total_episodes + 1),
            desc="Single-agent training",
            unit="ep",
            dynamic_ncols=True,
            leave=True,
            position=0,
            mininterval=0.2,
            smoothing=0.1,
            colour="green",
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            ),
        ) as progress_bar:
            for episode in progress_bar:
                total_return, length, success, avg_loss = _run_episode(
                    agent,
                    env,
                    max_steps,
                    reward_step_penalty,
                    train_freq=train_freq,
                    gradient_steps=gradient_steps,
                )

                logger.log_episode(
                    episode=episode,
                    total_return=total_return,
                    length=length,
                    success=success,
                    epsilon=agent.current_epsilon,
                    loss=avg_loss,
                )

                progress_bar.set_postfix(
                    eps=f"{agent.current_epsilon:.3f}",
                    ret=f"{total_return:.1f}",
                    len=f"{length:d}",
                    loss=f"{avg_loss:.5f}",
                )

                # Periodic logging
                if episode % log_interval == 0:
                    stats = logger.get_recent_stats()
                    log.debug(
                        f"Episode {episode}/{total_episodes} | "
                        f"Avg Return: {stats.get('avg_return', 0):.3f} | "
                        f"Avg Loss: {stats.get('avg_loss', 0):.5f} | "
                        f"Avg Length: {stats.get('avg_length', 0):.0f} | "
                        f"Epsilon: {agent.current_epsilon:.3f} | "
                        f"Buffer: {len(agent.replay_buffer)}"
                    )

                # Periodic evaluation
                if episode % eval_interval == 0:
                    eval_result = evaluate(
                        agent,
                        env,
                        n_episodes=eval_episodes,
                        max_steps=max_steps,
                        reward_step_penalty=reward_step_penalty,
                    )
                    eval_record = EvalRecord(
                        episode=episode,
                        avg_return=eval_result["avg_return"],
                        sem_return=eval_result["sem_return"],
                        win_rate=eval_result["win_rate"],
                        sem_win_rate=eval_result["sem_win_rate"],
                        avg_length=eval_result["avg_length"],
                        sem_length=eval_result["sem_length"],
                        avg_success_length=eval_result["avg_success_length"],
                        sem_success_length=eval_result["sem_success_length"],
                        std_length=eval_result["std_length"],
                    )
                    logger.log_eval(eval_record)
                    log.info(
                        f"  EVAL @ {episode}: Win Rate={eval_result['win_rate']:.3f} | "
                        f"Avg Return={eval_result['avg_return']:.1f} | "
                        f"Avg Length={eval_result['avg_length']:.0f} | "
                    )

                    # Save best model
                    if eval_result["win_rate"] > best_win_rate:
                        best_win_rate = eval_result["win_rate"]
                        agent.save(os.path.join(checkpoint_dir, "best_model.pt"))
                        log.info(
                            f"  New best model saved (win_rate={best_win_rate:.3f})"
                        )

                # Periodic checkpoint
                if episode % checkpoint_interval == 0:
                    agent.save(os.path.join(checkpoint_dir, f"checkpoint_{episode}.pt"))

    agent.save(os.path.join(checkpoint_dir, "final_model.pt"))
    logger.save_eval_json()

    return best_win_rate
