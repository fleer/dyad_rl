import logging
import os

import gymnasium as gym
import numpy as np
from omegaconf import DictConfig

from agents.dqn_agent import DQNAgent
from utils.metrics import MetricsLogger, EvalRecord
from utils.obs_processing import process_obs
from evaluation.evaluate import evaluate, evaluate_masked_random

log = logging.getLogger(__name__)

def _is_episode_done(
    terminated: bool,
    truncated: bool,
    step_index: int,
    max_steps: int,
) -> bool:
    return terminated or truncated or (step_index + 1) >= max_steps


def prefill_buffer(
    agent: DQNAgent,
    env: gym.Env,
    num_transitions: int,
    max_steps: int,
    dual_obs: bool = False,
    reward_step_penalty: float = 0.0,
) -> None:
    """Fill the replay buffer with random transitions before training."""
    log.info(f"Prefilling replay buffer with {num_transitions} random transitions...")
    collected = 0
    while collected < num_transitions:
        obs_raw, info = env.reset()
        obs, state_discrete, state_rgb = process_obs(obs_raw, agent.obs_type, dual_obs)

        for step in range(max_steps):
            action_mask = env.action_masks()
            valid_actions = np.where(action_mask)[0]
            action = int(np.random.choice(valid_actions))

            next_obs_raw, reward, terminated, truncated, next_info = env.step(action)
            shaped_reward = reward - reward_step_penalty
            next_obs, next_state_discrete, next_state_rgb = process_obs(
                next_obs_raw, agent.obs_type, dual_obs
            )

            done = _is_episode_done(terminated, truncated, step, max_steps)
            next_action_mask = (
                np.zeros(env.action_space.n, dtype=bool)
                if done
                else np.asarray(env.action_masks(), dtype=bool)
            )

            agent.replay_buffer.push(
                obs,
                action,
                shaped_reward,
                next_obs,
                done,
                next_action_mask,
                state_discrete,
                next_state_discrete,
                state_rgb,
                next_state_rgb,
            )
            collected += 1
            obs = next_obs
            state_discrete = next_state_discrete
            state_rgb = next_state_rgb

            if done or collected >= num_transitions:
                break
    log.info(f"Prefilled buffer with {len(agent.replay_buffer)} transitions.")


def _run_episode(
    agent: DQNAgent,
    env: gym.Env,
    max_steps: int,
    dual_obs: bool = False,
    reward_step_penalty: float = 0.0,
    train_freq: int = 4,
    gradient_steps: int = 1,
) -> tuple[float, int, bool, float, dict[str, float]]:
    """Run one training episode.

    Returns (total_return, length, success, avg_loss, avg_opt_stats).
    """
    obs_raw, info = env.reset()
    obs, state_discrete, state_rgb = process_obs(obs_raw, agent.obs_type, dual_obs)

    total_return = 0.0
    total_loss = 0.0
    loss_count = 0
    total_nz_reward_frac = 0.0
    total_terminal_frac = 0.0
    total_td_abs_zero = 0.0
    total_td_abs_pos = 0.0
    total_td_abs_neg = 0.0

    for step in range(max_steps):
        action_mask = env.action_masks()
        action = agent.select_action(obs, action_mask, explore=True)

        next_obs_raw, reward, terminated, truncated, next_info = env.step(action)
        shaped_reward = reward - reward_step_penalty
        next_obs, next_state_discrete, next_state_rgb = process_obs(
            next_obs_raw, agent.obs_type, dual_obs
        )

        done = _is_episode_done(terminated, truncated, step, max_steps)
        next_action_mask = (
            np.zeros(env.action_space.n, dtype=bool)
            if done
            else np.asarray(env.action_masks(), dtype=bool)
        )

        agent.replay_buffer.push(
            obs,
            action,
            shaped_reward,
            next_obs,
            done,
            next_action_mask,
            state_discrete,
            next_state_discrete,
            state_rgb,
            next_state_rgb,
        )

        # Optimize every train_freq steps (mirrors SB3 train_freq logic)
        if agent.steps_done % train_freq == 0 and len(agent.replay_buffer) >= agent.batch_size:
            for _ in range(gradient_steps):
                loss = agent.optimize()
                if loss > 0:
                    total_loss += loss
                    loss_count += 1
                    total_nz_reward_frac += agent.last_optimize_stats.get("non_zero_reward_frac", 0.0)
                    total_terminal_frac += agent.last_optimize_stats.get("terminal_frac", 0.0)
                    total_td_abs_zero += agent.last_optimize_stats.get("td_abs_zero", 0.0)
                    total_td_abs_pos += agent.last_optimize_stats.get("td_abs_pos", 0.0)
                    total_td_abs_neg += agent.last_optimize_stats.get("td_abs_neg", 0.0)
            agent.update_target_net()

        total_return += reward
        obs = next_obs
        state_discrete = next_state_discrete
        state_rgb = next_state_rgb

        if done:
            break

    success = reward > 0  # +100 for solved
    avg_loss = total_loss / max(loss_count, 1)
    avg_opt_stats = {
        "non_zero_reward_frac": total_nz_reward_frac / max(loss_count, 1),
        "terminal_frac": total_terminal_frac / max(loss_count, 1),
        "td_abs_zero": total_td_abs_zero / max(loss_count, 1),
        "td_abs_pos": total_td_abs_pos / max(loss_count, 1),
        "td_abs_neg": total_td_abs_neg / max(loss_count, 1),
    }
    return total_return, step + 1, success, avg_loss, avg_opt_stats


def train_single(
    agent: DQNAgent,
    env: gym.Env,
    cfg: DictConfig,
    logger: MetricsLogger,
    checkpoint_dir: str = "checkpoints",
) -> float:
    """Train a single DQN agent.

    Args:
        agent: The DQN agent to train.
        env: The Gymnasium environment (already wrapped).
        cfg: Hydra config with training parameters.
        logger: MetricsLogger for recording metrics.
        checkpoint_dir: Directory for saving model checkpoints.

    Returns:
        Best evaluation win rate (for Optuna optimization).
    """
    total_episodes = cfg.training.total_episodes
    max_steps = cfg.training.max_steps
    eval_interval = cfg.training.eval_interval
    eval_episodes = cfg.training.eval_episodes
    checkpoint_interval = cfg.training.checkpoint_interval
    log_interval = cfg.training.log_interval
    reward_step_penalty = float(cfg.training.get("reward_step_penalty", 0.0))
    dual_obs = getattr(env.unwrapped, "obs_type", None) == "dual"
    train_freq = int(cfg.agent.get("train_freq", 4))
    gradient_steps = int(cfg.agent.get("gradient_steps", 1))

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_win_rate = -1.0
    baseline_result = evaluate_masked_random(
        env,
        n_episodes=eval_episodes,
        max_steps=max_steps,
    )
    logger.log_baseline("masked_random", baseline_result, eval_episodes, max_steps)
    log.info(
        "Masked-random baseline | "
        f"Win Rate: {baseline_result['win_rate']:.3f} | "
        f"Avg Return: {baseline_result['avg_return']:.1f} | "
        f"Avg Length: {baseline_result['avg_length']:.0f}"
    )

    learning_starts = int(cfg.agent.get("learning_starts", 100))
    if learning_starts > 0:
        prefill_buffer(agent, env, learning_starts, max_steps, dual_obs, reward_step_penalty)

    for episode in range(1, total_episodes + 1):
        total_return, length, success, avg_loss, opt_stats = _run_episode(
            agent,
            env,
            max_steps,
            dual_obs,
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
            non_zero_reward_frac=opt_stats.get("non_zero_reward_frac", 0.0),
            terminal_frac=opt_stats.get("terminal_frac", 0.0),
            td_abs_zero=opt_stats.get("td_abs_zero", 0.0),
            td_abs_pos=opt_stats.get("td_abs_pos", 0.0),
            td_abs_neg=opt_stats.get("td_abs_neg", 0.0),
        )

        # Periodic logging
        if episode % log_interval == 0:
            stats = logger.get_recent_stats(window=log_interval)
            log.info(
                f"Episode {episode}/{total_episodes} | "
                f"Avg Return: {stats.get('avg_return', 0):.3f} | "
                # f"Win Rate: {stats.get('win_rate', 0):.3f} | "
                f"Avg Loss: {stats.get('avg_loss', 0):.3f} | "
                f"NZRewardFrac: {stats.get('avg_non_zero_reward_frac', 0):.3f} | "
                f"TermFrac: {stats.get('avg_terminal_frac', 0):.3f} | "
                f"Avg Length: {stats.get('avg_length', 0):.0f} | "
                f"Epsilon: {agent.current_epsilon:.3f} | "
                f"Buffer: {len(agent.replay_buffer)}"
            )

        # Periodic evaluation
        if episode % eval_interval == 0:
            eval_result = evaluate(agent, env, n_episodes=eval_episodes, max_steps=max_steps)
            eval_record = EvalRecord(
                episode=episode,
                avg_return=eval_result["avg_return"],
                win_rate=eval_result["win_rate"],
                avg_length=eval_result["avg_length"],
                avg_success_length=eval_result["avg_success_length"],
                std_length=eval_result["std_length"],
            )
            logger.log_eval(eval_record)
            log.info(
                f"  EVAL @ {episode}: Win Rate={eval_result['win_rate']:.3f} | "
                f"Avg Return={eval_result['avg_return']:.1f} | "
                f"Avg Length={eval_result['avg_length']:.0f} | "
                f"Delta WR={eval_result['win_rate'] - baseline_result['win_rate']:+.3f} | "
                f"Delta Ret={eval_result['avg_return'] - baseline_result['avg_return']:+.1f}"
            )

            # Save best model
            if eval_result["win_rate"] > best_win_rate:
                best_win_rate = eval_result["win_rate"]
                agent.save(os.path.join(checkpoint_dir, "best_model.pt"))
                log.info(f"  New best model saved (win_rate={best_win_rate:.3f})")

        # Periodic checkpoint
        if episode % checkpoint_interval == 0:
            agent.save(os.path.join(checkpoint_dir, f"checkpoint_{episode}.pt"))

    # Save final metrics
    logger.save_csv()
    logger.save_eval_json()
    logger.save_baseline_json()
    agent.save(os.path.join(checkpoint_dir, "final_model.pt"))

    return best_win_rate
