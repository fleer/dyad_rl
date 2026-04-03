"""Resume DQN training from a checkpoint for additional episodes.

Finds the latest checkpoint in a directory (or uses a given .pt file), restores
the agent state, and trains for ``--n-episodes`` more episodes. Results and new
checkpoints are written to separate ``_resumed`` directories by default.

Usage:
    # Resume latest checkpoint in a directory for 5 000 more episodes
    python continue_training.py checkpoints/exp1_mlp_2x3 --n-episodes 5000

    # Resume a specific checkpoint file
    python continue_training.py checkpoints/exp1_mlp_2x3/best_model.pt --n-episodes 5000

    # Custom output name and device
    python continue_training.py checkpoints/exp1_mlp_2x3 --n-episodes 5000 \\
        --experiment-name exp1_mlp_2x3_ft --device cpu

    # Override any config key (key=value pairs after positional args)
    python continue_training.py checkpoints/exp1_mlp_2x3 --n-episodes 5000 \\
        agent.learning_rate=5e-5 training.eval_interval=500
"""

import argparse
import logging
import os
import re
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

import rlp  # noqa: F401 — registers rlp/Puzzle-v0
from agents.dqn_agent import DQNAgent
from training.train_single import train_single
from utils.env_factory import make_env
from utils.metrics import MetricsLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def find_latest_checkpoint(path: str) -> str:
    """Find Latest Checkpoint.

    Returns the path to the most recent checkpoint given a file or directory.
    When a directory is supplied, prefers the numerically highest
    ``checkpoint_N.pt``, then ``final_model.pt``, then ``best_model.pt``.

    Args:
        path (str): Path to a ``.pt`` file or a directory of checkpoints.

    Returns:
        str: Absolute path to the selected checkpoint file.

    Raises:
        FileNotFoundError: When no checkpoint can be located.
    """
    p = Path(path)
    if p.is_file():
        return str(p)

    numbered: list[tuple[int, str]] = []
    for f in p.glob("checkpoint_*.pt"):
        m = re.match(r"checkpoint_(\d+)\.pt$", f.name)
        if m:
            numbered.append((int(m.group(1)), str(f)))

    if numbered:
        numbered.sort()
        chosen = numbered[-1][1]
        log.info(f"Latest checkpoint: {chosen}  (episode {numbered[-1][0]})")
        return chosen

    for name in ("final_model.pt", "best_model.pt"):
        candidate = p / name
        if candidate.exists():
            log.info(f"Using fallback checkpoint: {candidate}")
            return str(candidate)

    raise FileNotFoundError(f"No checkpoint found in {path!r}")


def infer_start_episode(checkpoint_path: str) -> int:
    """Infer Episode Number from Checkpoint Filename.

    Extracts the episode number from a ``checkpoint_N.pt`` filename if present,
    otherwise returns 0.

    Args:
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        int: Episode number, or 0 when the filename carries no episode info.
    """
    m = re.search(r"checkpoint_(\d+)\.pt$", os.path.basename(checkpoint_path))
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _find_hydra_config(experiment_name: str) -> str | None:
    """Search Hydra Outputs for a Matching Config.

    Args:
        experiment_name (str): Experiment name to match against saved configs.

    Returns:
        str | None: Path to the first matching ``config.yaml``, or ``None``.
    """
    outputs = Path("outputs")
    if not outputs.exists():
        return None
    for cfg_file in outputs.rglob(".hydra/config.yaml"):
        cfg = OmegaConf.load(cfg_file)
        if cfg.get("experiment_name", "").lower() == experiment_name.lower():
            log.info(f"Found Hydra config: {cfg_file}")
            return str(cfg_file)
    return None


def load_config(checkpoint_path: str, overrides: list[str]) -> DictConfig:
    """Load Configuration for Resumed Training.

    Attempts to load the Hydra-saved config that matches the checkpoint's
    experiment name.  Falls back to composing a config from the repository
    YAML files when no saved config is found.  CLI overrides are applied last.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        overrides (list[str]): ``key=value`` override strings.

    Returns:
        DictConfig: Resolved and overridden configuration.
    """
    experiment_name = Path(checkpoint_path).parent.name

    # 1. Try Hydra saved config
    config_path = _find_hydra_config(experiment_name)
    if config_path:
        cfg = OmegaConf.load(config_path)
    else:
        log.warning("No saved Hydra config found — inferring from checkpoint path.")
        repo_root = Path(__file__).resolve().parent
        config_dir = repo_root / "config"

        inferred_agent = "mlp" if "mlp" in experiment_name.lower() else "cnn"
        inferred_obs = "puzzle_state" if inferred_agent == "mlp" else "rgb"

        cfg = OmegaConf.create(
            {"mode": "train", "experiment_name": experiment_name, "seed": 42, "device": "auto"}
        )
        for fname, key in [
            (config_dir / "env/netslide_2x3.yaml", "env"),
            (config_dir / f"agent/{inferred_agent}.yaml", "agent"),
            (config_dir / "training/dqn.yaml", "training"),
        ]:
            if fname.exists():
                cfg = OmegaConf.merge(cfg, OmegaConf.create({key: OmegaConf.load(fname)}))
        if "env" in cfg:
            cfg.env.obs_type = inferred_obs

    # 2. Apply CLI overrides
    for ov in overrides:
        if "=" not in ov:
            log.warning(f"Ignoring override without '=': {ov!r}")
            continue
        key, _, value = ov.partition("=")
        OmegaConf.update(cfg, key.lstrip("+"), value, merge=True)

    return cfg


# ---------------------------------------------------------------------------
# Device / shape helpers (mirrors experiment.py)
# ---------------------------------------------------------------------------

def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(device_str)


def _get_obs_shape(env, obs_type: str, cfg: DictConfig) -> tuple[int, ...]:
    if obs_type == "rgb":
        return (3, cfg.env.window_width, cfg.env.window_height)
    obs_space = env.unwrapped.observation_space
    if hasattr(obs_space, "spaces") and "puzzle_state" in obs_space.spaces:
        return obs_space["puzzle_state"].shape
    return env.observation_space.shape


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume DQN training from a checkpoint for additional episodes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "checkpoint",
        help="Path to a checkpoint .pt file or a directory containing checkpoints.",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        required=True,
        metavar="N",
        help="Number of additional training episodes.",
    )
    parser.add_argument(
        "--device",
        default=None,
        metavar="DEV",
        help="Device override: auto | cpu | cuda | rocm (default: from config).",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        metavar="NAME",
        help=(
            "Name used for results/ and checkpoints/ subdirectories. "
            "Defaults to <original_name>_resumed."
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="key=value",
        help="Config overrides, e.g. agent.learning_rate=5e-5",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Locate checkpoint
    # ------------------------------------------------------------------
    checkpoint_path = find_latest_checkpoint(args.checkpoint)
    start_episode = infer_start_episode(checkpoint_path)

    # Peek at steps_done so the epsilon schedule can be anchored correctly
    raw_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    steps_done_at_resume = int(raw_ckpt.get("steps_done", 0))
    log.info(
        f"Checkpoint: {checkpoint_path}  "
        f"(episode ~{start_episode}, steps_done={steps_done_at_resume})"
    )

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    cfg = load_config(checkpoint_path, args.overrides)

    if args.device is not None:
        cfg.device = args.device

    cfg.training.total_episodes = args.n_episodes

    base_name = cfg.get("experiment_name", Path(checkpoint_path).parent.name)
    resumed_name = args.experiment_name or f"{base_name}_resumed"
    cfg.experiment_name = resumed_name

    # ------------------------------------------------------------------
    # Environment + agent
    # ------------------------------------------------------------------
    device = _resolve_device(cfg.device)
    log.info(f"Device: {device}")

    env = make_env(cfg)
    obs_shape = _get_obs_shape(env, cfg.env.obs_type, cfg)
    num_actions = env.action_space.n
    log.info(f"Obs shape: {obs_shape}  |  Actions: {num_actions}")

    # Total steps for epsilon schedule: prior progress + new budget so that
    # epsilon continues exactly from where it was rather than resetting.
    additional_steps = args.n_episodes * cfg.training.max_steps
    total_steps_for_schedule = steps_done_at_resume + additional_steps

    agent = DQNAgent(
        obs_type=cfg.env.obs_type,
        obs_shape=obs_shape,
        num_actions=num_actions,
        cfg=cfg,
        device=device,
        total_steps=total_steps_for_schedule,
    )
    agent.load(checkpoint_path)
    log.info(
        f"Agent loaded.  steps_done={agent.steps_done}  "
        f"epsilon={agent.current_epsilon:.4f}"
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    results_dir = os.path.join("results", resumed_name)
    checkpoint_dir = os.path.join("checkpoints", resumed_name)
    logger = MetricsLogger(results_dir, agent_name=resumed_name)

    log.info(f"Results    → {results_dir}")
    log.info(f"Checkpoints→ {checkpoint_dir}")
    log.info(f"Training for {args.n_episodes} additional episodes…")

    best_win_rate = train_single(agent, env, cfg, logger, checkpoint_dir)
    log.info(f"Resumed training complete.  Best win rate: {best_win_rate:.3f}")


if __name__ == "__main__":
    main()
