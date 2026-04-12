"""Main entry point for DQN and dyad RL experiments.

Usage:
    # Baseline MLP on netslide 2x3
    python experiment.py +experiment=baseline_mlp

    # Baseline CNN on netslide 2x3
    python experiment.py +experiment=baseline_cnn

    # Dyad learning
    python experiment.py +experiment=dyad

    # Override environment
    python experiment.py +experiment=baseline_mlp env=netslide_3x3

    # Evaluation only
    python experiment.py +experiment=baseline_mlp mode=eval experiment_name=baseline_mlp

    # Masked-random baseline only
    python experiment.py +experiment=baseline_mlp mode=baseline experiment_name=baseline_mlp
"""

import logging
import os

import gymnasium as gym
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from agents.dqn_agent import DQNAgent
from utils.env_factory import make_env, make_dual_obs_env
from utils.metrics import MetricsLogger
from training.train_single import train_single
from training.train_dyad import train_dyad
from evaluation.evaluate import evaluate

log = logging.getLogger(__name__)


def _resolve_device(device_str: str) -> torch.device:
    """Resolve Torch Device.

    Resolves a configured device string into a concrete torch device.

    Args:
        device_str (str): Device selector such as ``"auto"``, ``"cpu"``, or
            ``"cuda"``.

    Returns:
        torch.device: Resolved device object.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_str)


def _get_obs_shape(env: gym.Env, obs_type: str, cfg: DictConfig) -> tuple[int, ...]:
    """Get Observation Shape.

    Determines the observation shape expected by the selected observation mode.

    Args:
        env (gym.Env): Environment instance.
        obs_type (str): Observation type (for example ``"rgb"`` or
            ``"puzzle_state"``).
        cfg (DictConfig): Experiment configuration.

    Returns:
        tuple[int, ...]: Observation tensor shape for agent initialization.
    """
    if obs_type == "rgb":
        return (3, cfg.env.window_width, cfg.env.window_height)
    else:
        # For puzzle_state (or dual), get the flat discrete shape
        obs_space = env.unwrapped.observation_space
        # TODO: This should be rewritten
        # if "puzzle_state" in obs_space.spaces:
        #    return obs_space["puzzle_state"].shape
        return env.observation_space.shape


def run_train_single(cfg: DictConfig) -> float:
    """Run Single-Agent Training.

    Trains one DQN agent and returns the best observed evaluation win rate.

    Args:
        cfg (DictConfig): Experiment configuration.

    Returns:
        float: Best evaluation win rate achieved during training.
    """
    device = _resolve_device(cfg.device)
    log.info(f"Device: {device}")
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    env = make_env(cfg)
    obs_shape = _get_obs_shape(env, cfg.env.obs_type, cfg)
    num_actions = env.action_space.n
    log.info(f"Obs shape: {obs_shape}, Actions: {num_actions}")

    agent = DQNAgent(
        obs_type=cfg.env.obs_type,
        obs_shape=obs_shape,
        num_actions=num_actions,
        cfg=cfg,
        device=device,
    )

    results_dir = os.path.join("results", cfg.experiment_name)
    checkpoint_dir = os.path.join("checkpoints", cfg.experiment_name)
    logger = MetricsLogger(results_dir, agent_name=cfg.experiment_name)

    best_win_rate = train_single(agent, env, cfg, logger, checkpoint_dir)
    log.info("Training complete.")
    return best_win_rate


def run_train_dyad(cfg: DictConfig) -> float:
    """Run Dyad Training.

    Trains two DQN agents with dyad sharing and returns the best evaluation win
    rate across agents.

    Args:
        cfg (DictConfig): Experiment configuration.

    Returns:
        float: Best evaluation win rate across both dyad agents.
    """
    device = _resolve_device(cfg.device)
    log.info(f"Device: {device}")
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Both agents share a dual-obs environment that provides both modalities
    env_a = make_dual_obs_env(cfg)
    env_b = make_dual_obs_env(cfg)

    obs_shape_a = _get_obs_shape(env_a, "puzzle_state", cfg)
    obs_shape_b = _get_obs_shape(env_b, "rgb", cfg)
    num_actions = env_a.action_space.n

    agent_a_cfg = cfg.get("agent_a", cfg.agent)
    agent_b_cfg = cfg.get("agent_b", cfg.agent)

    agent_a = DQNAgent(
        obs_type="puzzle_state",
        obs_shape=obs_shape_a,
        num_actions=num_actions,
        cfg=cfg,
        agent_cfg=agent_a_cfg,
        device=device,
    )
    agent_b = DQNAgent(
        obs_type="rgb",
        obs_shape=obs_shape_b,
        num_actions=num_actions,
        cfg=cfg,
        agent_cfg=agent_b_cfg,
        device=device,
    )

    results_dir = os.path.join("results", cfg.experiment_name)
    checkpoint_dir = os.path.join("checkpoints", cfg.experiment_name)
    logger_a = MetricsLogger(results_dir, agent_name="agent_a")
    logger_b = MetricsLogger(results_dir, agent_name="agent_b")

    log.info(f"Agent A obs shape: {obs_shape_a}, Agent B obs shape: {obs_shape_b}")

    best_win_rate = train_dyad(
        agent_a, agent_b, env_a, env_b, cfg, logger_a, logger_b, checkpoint_dir
    )
    log.info("Dyad training complete.")
    return best_win_rate


def run_eval(cfg: DictConfig) -> None:
    """Run Checkpoint Evaluation.

    Loads a trained checkpoint and evaluates the agent on configured episodes.

    Args:
        cfg (DictConfig): Experiment configuration.

    Returns:
        None: Evaluation results are logged.
    """
    device = _resolve_device(cfg.device)
    env = make_env(cfg)
    obs_shape = _get_obs_shape(env, cfg.env.obs_type, cfg)
    num_actions = env.action_space.n

    agent = DQNAgent(
        obs_type=cfg.env.obs_type,
        obs_shape=obs_shape,
        num_actions=num_actions,
        cfg=cfg,
        device=device,
    )

    checkpoint_path = os.path.join("checkpoints", cfg.experiment_name, "best_model.pt")
    agent.load(checkpoint_path)
    log.info(f"Loaded checkpoint from {checkpoint_path}")

    result = evaluate(agent, env, n_episodes=cfg.training.eval_episodes)
    log.info(f"Evaluation results: {result}")


@hydra.main(config_path="config", config_name="default", version_base=None)
def main(cfg: DictConfig) -> float | None:
    """Dispatch Experiment Mode.

    Routes execution to train, evaluation, or baseline mode based on config.

    Args:
        cfg (DictConfig): Hydra-composed runtime configuration.

    Returns:
        float | None: Best win rate for training modes, otherwise ``None``.
    """
    # Set random seeds
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if cfg.mode == "train":
        if cfg.training.dyad:
            return run_train_dyad(cfg)
        else:
            return run_train_single(cfg)
    elif cfg.mode == "eval":
        run_eval(cfg)
        return None
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


if __name__ == "__main__":
    main()
