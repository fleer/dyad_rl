import json
import logging
import os

import gymnasium as gym
import numpy as np
from omegaconf import DictConfig
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from agents.dqn_agent import DQNAgent
from evaluation.evaluate import (
    collect_eval_trajectory,
    evaluate,
)
from training.train_single import _run_episode, prefill_buffer
from utils.metrics import EvalRecord, MetricsLogger
from utils.replay_buffer import Transition

log = logging.getLogger(__name__)


def _share_experience(
    rater: DQNAgent,
    provider_trajectory: list[Transition],
    share_all: bool,
    rating_threshold: float,
) -> list[Transition]:
    """Share Rated Experience.

    Rates another agent's trajectory with the rater's value function and
    returns transitions that pass the acceptance threshold.

    Args:
        rater (DQNAgent): Agent evaluating the provider trajectory.
        provider_trajectory (list[Transition]): Provider trajectory transitions.
        share_all (bool): Whether to share all transitions regardless of rating.
        rating_threshold (float): Minimum accepted rating value.

    Returns:
        list[Transition]: Accepted transitions converted to the rater modality.
    """
    if not provider_trajectory:
        return []

    expected_returns = rater.compute_expected_return(provider_trajectory)

    # TODO: Consider
    # Compute actual cumulative discounted return from provider's trajectory
    actual_returns = np.zeros(len(provider_trajectory), dtype=np.float32)
    running_return = 0.0
    for i in reversed(range(len(provider_trajectory))):
        running_return = provider_trajectory[
            i
        ].reward + rater.gamma * running_return * (
            1.0 - float(provider_trajectory[i].done)
        )
        actual_returns[i] = running_return

    # ATTENTION: This is the most critical part of the dyad learning algorithm.
    # The choice of rating_threshold determines which transitions are considered valuable.
    # Accept transitions where rater's Q-value suggests some value
    # Rating = actual return - expected return; accept if rating >= threshold
    # This means that the accepted transitions are those where the provider's outcome was better
    # than what the rater expected, according to the rater's own value function.
    _ratings = actual_returns - expected_returns
    accepted: list[Transition] = []

    for i, step in enumerate(provider_trajectory):
        # Rating is computed before buffer insertion so both observation
        # branches remain available during cross-agent scoring.
        if _ratings[i] > rating_threshold or share_all:
            accepted.append(
                Transition(
                    action=step.action,
                    reward=step.reward,
                    done=step.done,
                    next_action_mask=step.next_action_mask,
                    state_discrete=step.state_discrete,
                    next_state_discrete=step.next_state_discrete,
                    state_rgb=step.state_rgb,
                    next_state_rgb=step.next_state_rgb,
                )
            )

    return accepted


def train_dyad(
    agent_a: DQNAgent,
    agent_b: DQNAgent,
    env_a: gym.Env,
    env_b: gym.Env,
    cfg: DictConfig,
    logger_a: MetricsLogger,
    logger_b: MetricsLogger,
    checkpoint_dir: str = "checkpoints",
) -> float:
    """Train Dyad Agents.

    Trains two DQN agents with periodic cross-rating and experience sharing.

    Args:
        agent_a (DQNAgent): First agent.
        agent_b (DQNAgent): Second agent.
        env_a (gym.Env): Environment for first agent.
        env_b (gym.Env): Environment for second agent.
        cfg (DictConfig): Dyad training configuration.
        logger_a (MetricsLogger): Metrics logger for first agent.
        logger_b (MetricsLogger): Metrics logger for second agent.
        checkpoint_dir (str): Base directory for checkpoints.

    Returns:
        float: Best evaluation win rate across both agents.
    """
    total_episodes = cfg.training.total_episodes
    max_steps = cfg.training.max_steps
    eval_interval = cfg.training.eval_interval
    eval_episodes = cfg.training.eval_episodes
    checkpoint_interval = cfg.training.checkpoint_interval
    log_interval = cfg.training.log_interval
    share_interval = cfg.training.share_interval
    rating_threshold = cfg.training.rating_threshold
    share_all = cfg.training.share_all
    reward_step_penalty = float(cfg.training.get("reward_step_penalty", 0.0))
    _cfg_a = cfg.get("agent_a", cfg.agent)
    _cfg_b = cfg.get("agent_b", cfg.agent)
    train_freq_a = int(_cfg_a.get("train_freq", 4))
    gradient_steps_a = int(_cfg_a.get("gradient_steps", 1))
    train_freq_b = int(_cfg_b.get("train_freq", 4))
    gradient_steps_b = int(_cfg_b.get("gradient_steps", 1))

    dir_a = os.path.join(checkpoint_dir, "agent_a")
    dir_b = os.path.join(checkpoint_dir, "agent_b")
    os.makedirs(dir_a, exist_ok=True)
    os.makedirs(dir_b, exist_ok=True)

    best_win_rate_a = -1.0
    best_win_rate_b = -1.0
    total_shared_to_a = 0
    total_shared_to_b = 0
    sharing_stats: list[dict] = []

    learning_starts_a = int(_cfg_a.get("learning_starts", 100))
    learning_starts_b = int(_cfg_b.get("learning_starts", 100))
    if learning_starts_a > 0:
        prefill_buffer(
            agent_a,
            env_a,
            learning_starts_a,
            max_steps,
            reward_step_penalty=reward_step_penalty,
        )
    if learning_starts_b > 0:
        prefill_buffer(
            agent_b,
            env_b,
            learning_starts_b,
            max_steps,
            reward_step_penalty=reward_step_penalty,
        )

    with logging_redirect_tqdm(loggers=[logging.getLogger()]):
        with tqdm(
            range(1, total_episodes + 1),
            desc="Dyad training",
            unit="ep",
            dynamic_ncols=True,
            leave=True,
            position=0,
            mininterval=0.2,
            smoothing=0.1,
            colour="cyan",
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            ),
        ) as progress_bar:
            for episode in progress_bar:
                # Train agent A for 1 episode
                ret_a, len_a, suc_a, loss_a = _run_episode(
                    agent_a,
                    env_a,
                    max_steps,
                    reward_step_penalty=reward_step_penalty,
                    train_freq=train_freq_a,
                    gradient_steps=gradient_steps_a,
                )
                logger_a.log_episode(
                    episode=episode,
                    total_return=ret_a,
                    length=len_a,
                    success=suc_a,
                    epsilon=agent_a.current_epsilon,
                    loss=loss_a,
                )

                # Train agent B for 1 episode
                ret_b, len_b, suc_b, loss_b = _run_episode(
                    agent_b,
                    env_b,
                    max_steps,
                    reward_step_penalty=reward_step_penalty,
                    train_freq=train_freq_b,
                    gradient_steps=gradient_steps_b,
                )
                logger_b.log_episode(
                    episode=episode,
                    total_return=ret_b,
                    length=len_b,
                    success=suc_b,
                    epsilon=agent_b.current_epsilon,
                    loss=loss_b,
                )

                progress_bar.set_postfix(
                    buffer_a=f"{len(agent_a.replay_buffer):d}",
                    buffer_b=f"{len(agent_b.replay_buffer):d}",
                    eps_a=f"{agent_a.current_epsilon:.3f}",
                    eps_b=f"{agent_b.current_epsilon:.3f}",
                )

                # Experience sharing
                if episode % share_interval == 0:
                    # Each agent runs 1 eval episode to generate a trajectory
                    # TODO: Range should be configurable and a parameter!!
                    ret_a = ret_b = 0.0
                    accepted_for_a: list[Transition] = []
                    accepted_for_b: list[Transition] = []
                    traj_a: list[Transition] = []
                    traj_b: list[Transition] = []
                    traj_a = collect_eval_trajectory(agent_a, env_a, max_steps)
                    # Agent B rates Agent A's trajectory using B's own value function
                    accepted_for_b = _share_experience(
                        agent_b, traj_a, share_all, rating_threshold
                    )
                    agent_b.add_to_buffer(accepted_for_b)
                    total_shared_to_b += len(accepted_for_b)
                    traj_b = collect_eval_trajectory(agent_b, env_b, max_steps)
                    # Agent A rates Agent B's trajectory using A's own value function
                    accepted_for_a = _share_experience(
                        agent_a, traj_b, share_all, rating_threshold
                    )
                    # Add accepted transitions to replay buffers
                    agent_a.add_to_buffer(accepted_for_a)
                    total_shared_to_a += len(accepted_for_a)
                    sharing_stats.append(
                        {
                            "episode": episode,
                            "accepted_for_a": len(accepted_for_a),
                            "traj_b_len": len(traj_b),
                            "accepted_for_b": len(accepted_for_b),
                            "traj_a_len": len(traj_a),
                        }
                    )
                    # log.info(
                    #     f"  SHARE @ {episode}: "
                    #     f"A accepted {len(accepted_for_a)}/{len(traj_b)} from B | "
                    #     f"B accepted {len(accepted_for_b)}/{len(traj_a)} from A | "
                    #     f"Total shared: A={total_shared_to_a}, B={total_shared_to_b}"
                    # )

                # Periodic logging
                if episode % log_interval == 0:
                    stats_a = logger_a.get_recent_stats()
                    stats_b = logger_b.get_recent_stats()
                    log.info(
                        f"Episode {episode}/{total_episodes}\n"
                        f"Total shared: A={total_shared_to_a}, B={total_shared_to_b}"
                        f"  Agent A: Ret={stats_a.get('avg_return', 0):.3f} "
                        f"WR={stats_a.get('win_rate', 0):.3f} "
                        f"Len={stats_a.get('avg_length', 0):.0f} "
                        f"Eps={agent_a.current_epsilon:.3f} "
                        f"Buf={len(agent_a.replay_buffer)}\n"
                        f"  Agent B: Ret={stats_b.get('avg_return', 0):.3f} "
                        f"WR={stats_b.get('win_rate', 0):.3f} "
                        f"Len={stats_b.get('avg_length', 0):.0f} "
                        f"Eps={agent_b.current_epsilon:.3f} "
                        f"Buf={len(agent_b.replay_buffer)}"
                    )

                # Periodic evaluation
                if episode % eval_interval == 0:
                    for name, agent, env, lgr, best_wr, ckdir in [
                        ("A", agent_a, env_a, logger_a, best_win_rate_a, dir_a),
                        ("B", agent_b, env_b, logger_b, best_win_rate_b, dir_b),
                    ]:
                        result = evaluate(
                            agent,
                            env,
                            n_episodes=eval_episodes,
                            max_steps=max_steps,
                            reward_step_penalty=reward_step_penalty,
                        )
                        record = EvalRecord(
                            episode=episode,
                            avg_return=result["avg_return"],
                            sem_return=result["sem_return"],
                            win_rate=result["win_rate"],
                            sem_win_rate=result["sem_win_rate"],
                            avg_length=result["avg_length"],
                            sem_length=result["sem_length"],
                            avg_success_length=result["avg_success_length"],
                            sem_success_length=result["sem_success_length"],
                            std_length=result["std_length"],
                        )
                        lgr.log_eval(record)
                        log.info(
                            f"  EVAL Agent {name} @ {episode}: "
                            f"WR={result['win_rate']:.3f} "
                            f"Ret={result['avg_return']:.1f} "
                            f"Len={result['avg_length']:.0f} "
                        )
                        if result["win_rate"] > best_wr:
                            if name == "A":
                                best_win_rate_a = result["win_rate"]
                            else:
                                best_win_rate_b = result["win_rate"]
                            agent.save(os.path.join(ckdir, "best_model.pt"))

                # Periodic checkpoint
                if episode % checkpoint_interval == 0:
                    agent_a.save(os.path.join(dir_a, f"checkpoint_{episode}.pt"))
                    agent_b.save(os.path.join(dir_b, f"checkpoint_{episode}.pt"))

    # Save final metrics and models
    logger_a.save_eval_json()
    logger_b.save_eval_json()

    sharing_stats_path = os.path.join(logger_a.log_dir, "sharing_stats.json")
    with open(sharing_stats_path, "w") as f:
        json.dump(sharing_stats, f, indent=2)
    log.info(f"Sharing stats saved to {sharing_stats_path}")
    agent_a.save(os.path.join(dir_a, "final_model.pt"))
    agent_b.save(os.path.join(dir_b, "final_model.pt"))

    return max(best_win_rate_a, best_win_rate_b)
