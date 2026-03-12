import logging
import os

import gymnasium as gym
import numpy as np
from omegaconf import DictConfig

from agents.dqn_agent import DQNAgent
from utils.replay_buffer import Transition
from utils.metrics import MetricsLogger, EvalRecord
from evaluation.evaluate import evaluate, evaluate_masked_random, collect_eval_trajectory
from utils.obs_processing import normalize_rgb
from training.train_single import _run_episode, prefill_buffer

log = logging.getLogger(__name__)


def _share_experience(
    rater: DQNAgent,
    provider_trajectory: list[dict],
    rater_obs_key: str,
    rater_next_obs_key: str,
    rating_threshold: float,
) -> list[Transition]:
    """Rate another agent's trajectory and return accepted transitions.

    The rater evaluates the provider's transitions using its own Q-network.
    Transitions where the rater's expected Q-value exceeds the threshold
    (relative to the actual reward) are accepted.

    Args:
        rater: The agent doing the rating.
        provider_trajectory: List of transition dicts from the provider agent.
        rater_obs_key: Key for rater-compatible observations ('state_discrete' or 'state_rgb').
        rater_next_obs_key: Key for rater-compatible next observations.
        rating_threshold: Minimum Q-value advantage for acceptance.

    Returns:
        List of accepted Transition namedtuples (using rater-compatible obs format).
    """
    if not provider_trajectory:
        return []

    # Extract rater-compatible states from the trajectory
    rater_states = []
    for step in provider_trajectory:
        s = step.get(rater_obs_key)
        if s is None:
            return []  # Dual obs not available
        rater_states.append(s)

    rater_states_arr = np.array(rater_states, dtype=np.float32)
    if rater.obs_type == "rgb":
        rater_states_arr = normalize_rgb(rater_states_arr)

    # Compute rater's Q-values for the provider's (state, action) pairs
    actions = [step["action"] for step in provider_trajectory]
    transitions_for_rating = [
        Transition(
            state=rater_states_arr[i],
            action=actions[i],
            reward=0,
            next_state=np.zeros_like(rater_states_arr[i]),
            done=False,
            next_action_mask=None,
            state_discrete=None,
            next_state_discrete=None,
            state_rgb=None,
            next_state_rgb=None,
        )
        for i in range(len(provider_trajectory))
    ]
    expected_returns = rater.compute_expected_return(transitions_for_rating)

    # TODO: Consider
    # Compute actual cumulative discounted return from provider's trajectory
    actual_returns = np.zeros(len(provider_trajectory), dtype=np.float32)
    running_return = 0.0
    for i in reversed(range(len(provider_trajectory))):
        running_return = provider_trajectory[i]["reward"] + rater.gamma * running_return
        actual_returns[i] = running_return

    # ATTENTION: This is the most critical part of the dyad learning algorithm. 
    # The choice of rating_threshold determines which transitions are considered valuable.
    # Accept transitions where rater's Q-value suggests some value
    # Rating = actual return - expected return; accept if rating >= threshold
    # This means that the accepted transitions are those where the provider's outcome was better 
    # than what the rater expected, according to the rater's own value function.
    ratings = actual_returns - expected_returns
    accepted: list[Transition] = []

    for i, step in enumerate(provider_trajectory):
        if ratings[i] > rating_threshold:
            # Build transition using rater-compatible observations
            rater_next_s = step.get(rater_next_obs_key)
            if rater_next_s is not None and rater.obs_type == "rgb":
                rater_next_s = normalize_rgb(rater_next_s)
            elif rater_next_s is None:
                continue

            accepted.append(Transition(
                state=rater_states_arr[i],
                action=step["action"],
                reward=step["reward"],
                next_state=rater_next_s,
                done=step["done"],
                next_action_mask=step.get("next_action_mask"),
                state_discrete=step.get("state_discrete"),
                next_state_discrete=step.get("next_state_discrete"),
                state_rgb=step.get("state_rgb"),
                next_state_rgb=step.get("next_state_rgb"),
            ))

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
) -> tuple[DQNAgent, DQNAgent]:
    """Train two DQN agents in a dyad learning setup with experience sharing.

    Agent A and Agent B each train independently, and periodically share
    and rate each other's evaluation trajectories.

    Args:
        agent_a: First DQN agent (e.g., MLP on discrete state).
        agent_b: Second DQN agent (e.g., CNN on RGB pixels).
        env_a: Environment for agent A (obs_type='dual').
        env_b: Environment for agent B (obs_type='dual').
        cfg: Hydra config with training and dyad parameters.
        logger_a: MetricsLogger for agent A.
        logger_b: MetricsLogger for agent B.
        checkpoint_dir: Directory for saving model checkpoints.

    Returns:
        Tuple of trained (agent_a, agent_b).
    """
    total_episodes = cfg.training.total_episodes
    max_steps = cfg.training.max_steps
    eval_interval = cfg.training.eval_interval
    eval_episodes = cfg.training.eval_episodes
    checkpoint_interval = cfg.training.checkpoint_interval
    log_interval = cfg.training.log_interval
    share_interval = cfg.training.share_interval
    rating_threshold = cfg.training.rating_threshold

    # Determine obs keys for cross-rating
    # Agent A rates Agent B's trajectory using A's observation format
    a_obs_key = "state_discrete" if agent_a.obs_type == "puzzle_state" else "state_rgb"
    a_next_key = "next_state_discrete" if agent_a.obs_type == "puzzle_state" else "next_state_rgb"
    b_obs_key = "state_discrete" if agent_b.obs_type == "puzzle_state" else "state_rgb"
    b_next_key = "next_state_discrete" if agent_b.obs_type == "puzzle_state" else "next_state_rgb"

    dir_a = os.path.join(checkpoint_dir, "agent_a")
    dir_b = os.path.join(checkpoint_dir, "agent_b")
    os.makedirs(dir_a, exist_ok=True)
    os.makedirs(dir_b, exist_ok=True)

    best_win_rate_a = -1.0
    best_win_rate_b = -1.0
    total_shared_to_a = 0
    total_shared_to_b = 0
    baseline_a = evaluate_masked_random(env_a, n_episodes=eval_episodes, max_steps=max_steps)
    baseline_b = evaluate_masked_random(env_b, n_episodes=eval_episodes, max_steps=max_steps)
    logger_a.log_baseline("masked_random", baseline_a, eval_episodes, max_steps)
    logger_b.log_baseline("masked_random", baseline_b, eval_episodes, max_steps)
    log.info(
        "Masked-random baselines | "
        f"Agent A env WR={baseline_a['win_rate']:.3f}, Ret={baseline_a['avg_return']:.1f} | "
        f"Agent B env WR={baseline_b['win_rate']:.3f}, Ret={baseline_b['avg_return']:.1f}"
    )

    if cfg.training.get("prefill_buffer", True):
        prefill_buffer(agent_a, env_a, agent_a.buffer_size, max_steps, dual_obs=True)
        prefill_buffer(agent_b, env_b, agent_b.buffer_size, max_steps, dual_obs=True)

    for episode in range(1, total_episodes + 1):
        # Train agent A for 1 episode
        ret_a, len_a, suc_a, loss_a = _run_episode(agent_a, env_a, max_steps, dual_obs=True)
        logger_a.log_episode(episode, ret_a, len_a, suc_a, agent_a.current_epsilon, loss_a)

        # Train agent B for 1 episode
        ret_b, len_b, suc_b, loss_b = _run_episode(agent_b, env_b, max_steps, dual_obs=True)
        logger_b.log_episode(episode, ret_b, len_b, suc_b, agent_b.current_epsilon, loss_b)

        # Experience sharing
        if episode % share_interval == 0:
            # Each agent runs 1 eval episode to generate a trajectory
            traj_a = collect_eval_trajectory(agent_a, env_a, max_steps)
            traj_b = collect_eval_trajectory(agent_b, env_b, max_steps)

            # Agent A rates Agent B's trajectory using A's own value function
            accepted_for_a = _share_experience(
                agent_a, traj_b, a_obs_key, a_next_key, rating_threshold
            )
            # Agent B rates Agent A's trajectory using B's own value function
            accepted_for_b = _share_experience(
                agent_b, traj_a, b_obs_key, b_next_key, rating_threshold
            )

            # Add accepted transitions to replay buffers
            agent_a.add_to_buffer(accepted_for_a)
            agent_b.add_to_buffer(accepted_for_b)

            total_shared_to_a += len(accepted_for_a)
            total_shared_to_b += len(accepted_for_b)

            log.info(
                f"  SHARE @ {episode}: "
                f"A accepted {len(accepted_for_a)}/{len(traj_b)} from B | "
                f"B accepted {len(accepted_for_b)}/{len(traj_a)} from A | "
                f"Total shared: A={total_shared_to_a}, B={total_shared_to_b}"
            )

        # Periodic logging
        if episode % log_interval == 0:
            stats_a = logger_a.get_recent_stats(window=log_interval)
            stats_b = logger_b.get_recent_stats(window=log_interval)
            log.info(
                f"Episode {episode}/{total_episodes}\n"
                f"  Agent A: Ret={stats_a.get('avg_return', 0):.1f} "
                f"WR={stats_a.get('win_rate', 0):.3f} "
                f"Len={stats_a.get('avg_length', 0):.0f} "
                f"Eps={agent_a.current_epsilon:.3f} "
                f"Buf={len(agent_a.replay_buffer)}\n"
                f"  Agent B: Ret={stats_b.get('avg_return', 0):.1f} "
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
                baseline = baseline_a if name == "A" else baseline_b
                result = evaluate(agent, env, n_episodes=eval_episodes, max_steps=max_steps)
                record = EvalRecord(
                    episode=episode,
                    avg_return=result["avg_return"],
                    win_rate=result["win_rate"],
                    avg_length=result["avg_length"],
                    avg_success_length=result["avg_success_length"],
                    std_length=result["std_length"],
                )
                lgr.log_eval(record)
                log.info(
                    f"  EVAL Agent {name} @ {episode}: "
                    f"WR={result['win_rate']:.3f} "
                    f"Ret={result['avg_return']:.1f} "
                    f"Len={result['avg_length']:.0f} "
                    f"DeltaWR={result['win_rate'] - baseline['win_rate']:+.3f} "
                    f"DeltaRet={result['avg_return'] - baseline['avg_return']:+.1f}"
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
    logger_a.save_csv()
    logger_a.save_eval_json()
    logger_a.save_baseline_json()
    logger_b.save_csv()
    logger_b.save_eval_json()
    logger_b.save_baseline_json()
    agent_a.save(os.path.join(dir_a, "final_model.pt"))
    agent_b.save(os.path.join(dir_b, "final_model.pt"))

    return max(best_win_rate_a, best_win_rate_b)
