"""Load a trained model and visualize one episode of puzzle solving.

This script:
1. Loads a checkpoint from the checkpoints/ directory
2. Infers the agent config and environment setup
3. Runs one episode with the trained agent using greedy policy
4. Optionally saves frames and creates an MP4 video

Usage:
    # Basic visualization
    python visualize_episode.py --checkpoint checkpoints/exp1_mlp_2x3/best_model.pt

    # Save frames to disk
    python visualize_episode.py --checkpoint checkpoints/exp1_mlp_2x3/best_model.pt --save-frames

    # Automatically create MP4 video (requires ffmpeg)
    python visualize_episode.py --checkpoint checkpoints/exp1_mlp_2x3/best_model.pt --mp4-output demo.mp4

    # Override puzzle params
    python visualize_episode.py --checkpoint checkpoints/exp1_mlp_2x3/best_model.pt --params "3x3b1" --mp4-output demo.mp4
"""

import argparse
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import FlattenObservation
from omegaconf import DictConfig, OmegaConf

import rlp  # noqa: F401 — registers rlp/Puzzle-v0
from agents.dqn_agent import DQNAgent
from utils.obs_processing import (
    NormalizePuzzleStateWrapper,
    process_obs,
)

log = logging.getLogger(__name__)


class ActionMaskWrapper(gym.Wrapper):
    """Ensures action_masks() is accessible through wrapper chain."""

    def action_masks(self) -> np.ndarray:
        """Get Action Mask.

        Forwards action-mask retrieval to the unwrapped base environment.

        Args:
            None: This method reads wrapped environment state.

        Returns:
            np.ndarray: Boolean mask of valid actions.
        """
        return self.env.unwrapped.action_masks()


def find_experiment_config(checkpoint_name: str) -> str | None:
    """Find Experiment Config.

    Searches Hydra output directories for a config matching the checkpoint
    experiment name.

    Args:
        checkpoint_name (str): Experiment name inferred from checkpoint path.

    Returns:
        str | None: Matching config path, or ``None`` if not found.
    """
    outputs_dir = "outputs"
    if not os.path.exists(outputs_dir):
        return None

    # Recursively search for .hydra/config.yaml files
    for root, _, _ in os.walk(outputs_dir):
        if ".hydra" in root and "config.yaml" in os.listdir(root):
            config_path = os.path.join(root, "config.yaml")
            cfg = OmegaConf.load(config_path)
            if cfg.get("experiment_name", "").lower() == checkpoint_name.lower():
                return config_path

    return None


def load_config_from_checkpoint(checkpoint_path: str) -> DictConfig:
    """Load Checkpoint Config.

    Loads the matching Hydra config for a checkpoint, or infers a fallback
    config when no saved config is available.

    Args:
        checkpoint_path (str): Path to checkpoint file.

    Returns:
        DictConfig: Loaded or inferred configuration.
    """
    # Extract experiment name from checkpoint path
    # e.g., "checkpoints/exp1_mlp_2x3/best_model.pt" -> "exp1_mlp_2x3"
    experiment_name = os.path.basename(os.path.dirname(checkpoint_path))

    log.info(f"Looking for config for experiment: {experiment_name}")

    # Try to find matching saved config
    config_path = find_experiment_config(experiment_name)

    if config_path is not None:
        log.info(f"Found config at: {config_path}")
        cfg = OmegaConf.load(config_path)
        return cfg

    log.warning(f"Could not find saved config for {experiment_name}")
    log.info("Inferring config from experiment name...")

    # Build a fallback config by composing repository YAMLs so mandatory
    # agent/training keys are present even when Hydra outputs are missing.
    repo_root = Path(__file__).resolve().parent
    config_dir = repo_root / "config"

    inferred_agent = "mlp" if "mlp" in experiment_name.lower() else "cnn"
    inferred_obs_type = "puzzle_state" if inferred_agent == "mlp" else "rgb"

    cfg = OmegaConf.create(
        {
            "mode": "train",
            "experiment_name": experiment_name,
            "seed": 42,
            "device": "auto",
        }
    )

    default_cfg_path = config_dir / "default.yaml"
    if default_cfg_path.exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(default_cfg_path))

    env_cfg_path = config_dir / "env" / "netslide_2x3.yaml"
    if env_cfg_path.exists():
        cfg = OmegaConf.merge(
            cfg, OmegaConf.create({"env": OmegaConf.load(env_cfg_path)})
        )

    agent_cfg_path = config_dir / "agent" / f"{inferred_agent}.yaml"
    if agent_cfg_path.exists():
        cfg = OmegaConf.merge(
            cfg, OmegaConf.create({"agent": OmegaConf.load(agent_cfg_path)})
        )

    training_cfg_path = config_dir / "training" / "dqn.yaml"
    if training_cfg_path.exists():
        cfg = OmegaConf.merge(
            cfg, OmegaConf.create({"training": OmegaConf.load(training_cfg_path)})
        )

    # Ensure visualization-safe settings.
    cfg.experiment_name = experiment_name
    cfg.env.obs_type = inferred_obs_type
    cfg.env.render_mode = "rgb_array"

    return cfg


def create_env_for_visualization(cfg: DictConfig) -> gym.Env:
    """Create Visualization Environment.

    Creates a visualization-safe environment using ``rgb_array`` rendering and
    applies required wrappers.

    Args:
        cfg (DictConfig): Environment configuration.

    Returns:
        gym.Env: Wrapped environment for episode visualization.
    """
    log.info("Creating environment in rgb_array mode (recommended for stability)")

    env = gym.make(
        "rlp/Puzzle-v0",
        puzzle=cfg.env.puzzle,
        render_mode="rgb_array",  # More stable than "human" with C backend
        obs_type=cfg.env.obs_type,
        window_width=cfg.env.window_width,
        window_height=cfg.env.window_height,
        allow_undo=cfg.env.allow_undo,
        max_state_repeats=cfg.env.max_state_repeats,
        include_cursor_in_state_info=cfg.env.include_cursor_in_state_info,
        params=cfg.env.params,
    )

    # Wrap with observation normalization and action masking
    if cfg.env.obs_type == "puzzle_state":
        env = FlattenObservation(env)
        env = NormalizePuzzleStateWrapper(env)
    elif cfg.env.obs_type == "rgb":
        env = ActionMaskWrapper(env)

    return ActionMaskWrapper(env)


# TODO: Infer from env.observation_space or agent type instead of hardcoding known sizes for specific puzzles
def get_obs_shape_for_agent(cfg: DictConfig) -> tuple[int, ...]:
    """Get Agent Observation Shape.

    Determines observation tensor shape expected by the configured agent.

    Args:
        cfg (DictConfig): Experiment configuration.

    Returns:
        tuple[int, ...]: Observation shape for agent construction.
    """
    if cfg.agent.type == "cnn":
        return (3, cfg.env.window_width, cfg.env.window_height)
    else:  # mlp or puzzle_state
        # For MLP, we know the discrete state is flattened
        # Netslide 2x3: flattened size is typically 34 floats
        # We can hardcode this or try to infer from the agent type
        if "netslide" in cfg.env.puzzle.lower():
            if "2x3" in cfg.env.params:
                return (34,)  # Known size for netslide 2x3
            elif "3x3" in cfg.env.params:
                return (46,)  # Known size for netslide 3x3
        if "samegame" in cfg.env.puzzle.lower():
            if "2x3c3s2" in cfg.env.params:
                return (43,)  # Known size for samegame 2x3c3s2

        # Fallback: Try to create a temp env (may crash with C backend)
        log.warning(
            "Could not infer obs shape, attempting to create temp environment..."
        )
        try:
            temp_env = gym.make(
                "rlp/Puzzle-v0",
                puzzle=cfg.env.puzzle,
                render_mode="rgb_array",
                obs_type="puzzle_state",
                window_width=cfg.env.window_width,
                window_height=cfg.env.window_height,
                params=cfg.env.params,
            )
            temp_env = FlattenObservation(temp_env)
            obs, _ = temp_env.reset()
            shape = obs.shape
            temp_env.close()
            return shape
        except Exception as e:
            log.error(f"Failed to determine obs shape: {e}")
            # Last resort fallback
            return (100,)


def load_agent(
    checkpoint_path: str,
    cfg: DictConfig,
    device: torch.device,
    num_actions: int | None = None,
) -> DQNAgent:
    """Load Agent From Checkpoint.

    Builds a DQN agent from config and restores model state from checkpoint.

    Args:
        checkpoint_path (str): Path to checkpoint file.
        cfg (DictConfig): Experiment configuration.
        device (torch.device): Torch device for model tensors.
        num_actions (int | None): Optional action count override.

    Returns:
        DQNAgent: Restored agent in evaluation mode.
    """
    log.info("Building agent with config...")
    log.info(f"  Agent type: {cfg.agent.type}")
    log.info(f"  Obs type: {cfg.env.obs_type}")

    obs_shape = get_obs_shape_for_agent(cfg)
    log.info(f"  Obs shape: {obs_shape}")

    # If num_actions not provided, we'll need to infer it
    # For now, use a fixed value for netslide
    if num_actions is None:
        # Netslide typically has 5 actions
        num_actions = 5
        log.info(f"  Using inferred num_actions: {num_actions}")

    log.info("Initializing agent...")
    agent = DQNAgent(
        obs_type=cfg.env.obs_type,
        obs_shape=obs_shape,
        num_actions=num_actions,
        cfg=cfg,
        device=device,
    )

    log.info("Agent initialized. Loading checkpoint...")
    log.info(f"  Checkpoint path: {checkpoint_path}")
    log.info(f"  File exists: {os.path.exists(checkpoint_path)}")
    log.info(
        f"  File size: {os.path.getsize(checkpoint_path) if os.path.exists(checkpoint_path) else 'N/A'} bytes"
    )

    # Load the checkpoint
    try:
        agent.load(checkpoint_path)
        log.info("Checkpoint loaded successfully")
    except RuntimeError as e:
        log.error(f"Failed to load checkpoint: {e}")
        # Try to infer num_actions from checkpoint
        log.info("Attempting to auto-detect num_actions from checkpoint...")
        try:
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=True
            )
            policy_state = checkpoint["policy_net"]

            # Detect output layer generically from the highest indexed linear
            # weight in the state dict (works for arbitrary hidden sizes).
            weight_entries = [
                (k, v)
                for k, v in policy_state.items()
                if "weight" in k
                and "bias" not in k
                and getattr(v, "dim", lambda: 0)() == 2
            ]
            if not weight_entries:
                raise RuntimeError(
                    "Could not find any 2D weight tensors in checkpoint policy_net"
                )

            def _layer_index(key: str) -> int:
                parts = key.split(".")
                for part in reversed(parts):
                    if part.isdigit():
                        return int(part)
                return -1

            last_layer_key, last_layer_weight = max(
                weight_entries, key=lambda kv: _layer_index(kv[0])
            )
            num_actions = int(last_layer_weight.shape[0])
            log.info(
                "Auto-detected num_actions from checkpoint layer %s: %d",
                last_layer_key,
                num_actions,
            )

            # Recreate agent with correct num_actions
            agent = DQNAgent(
                obs_type=cfg.env.obs_type,
                obs_shape=obs_shape,
                num_actions=num_actions,
                cfg=cfg,
                device=device,
            )
            agent.load(checkpoint_path)
            log.info("Checkpoint loaded successfully after auto-detection")
        except Exception as e2:
            log.error(f"Still failed: {e2}")
            raise

    agent.policy_net.eval()  # Set to eval mode
    log.info("Agent set to eval mode")

    return agent


def visualize_episode(
    agent: DQNAgent,
    env: gym.Env,
    cfg: DictConfig,
    max_steps: int = 1000,
    save_frames: bool = False,
    output_dir: str = "episode_frames",
) -> dict:
    """Visualize Single Episode.

    Runs one greedy episode with a trained agent and optionally saves rendered
    frames.

    Args:
        agent (DQNAgent): Trained agent used for action selection.
        env (gym.Env): Visualization environment.
        cfg (DictConfig): Experiment configuration.
        max_steps (int): Maximum steps to run.
        save_frames (bool): Whether to save rendered frames.
        output_dir (str): Directory where frames are saved.

    Returns:
        dict: Episode statistics including total return, steps, and success.
    """
    dual_obs = getattr(env.unwrapped, "obs_type", None) == "dual"

    if save_frames:
        os.makedirs(output_dir, exist_ok=True)
        log.info(f"Saving frames to {output_dir}/")

    obs_raw, info = env.reset()
    obs, _, _ = process_obs(obs_raw, agent.obs_type, dual_obs)

    log.info("Starting episode...")
    log.info(f"Agent type: {agent.policy_net.__class__.__name__}")
    log.info(f"Environment: {cfg.env.puzzle} {cfg.env.params}")

    total_return = 0.0
    step_count = 0
    frames = []

    for step in range(max_steps):
        # Get action mask and select action
        action_mask = env.action_masks()
        action = agent.select_action(obs, action_mask, explore=False)

        # Step environment
        obs_raw, reward, terminated, truncated, info = env.step(action)
        obs, _, _ = process_obs(obs_raw, agent.obs_type, dual_obs)

        total_return += reward
        step_count += 1

        # Render and optionally save frame
        frame = env.render()
        if frame is not None and save_frames:
            frames.append(frame)

        # Log progress periodically
        if (step + 1) % 100 == 0:
            log.info(f"Step {step + 1}: total_return = {total_return}")

        if terminated or truncated:
            break

    # Save frames if requested
    if save_frames and frames:
        for i, frame in enumerate(frames):
            frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
            try:
                from PIL import Image

                img = Image.fromarray(frame.astype(np.uint8))
                img.save(frame_path)
            except ImportError:
                log.warning("PIL not available; skipping frame saves")
                break

    agent.policy_net.train()  # Back to train mode

    success = total_return > 0

    log.info("Episode finished!")
    log.info(f"  Total return: {total_return}")
    log.info(f"  Steps: {step_count}")
    log.info(f"  Success: {success}")
    if save_frames:
        log.info(f"  Frames saved to: {output_dir}/")

    return {
        "total_return": total_return,
        "steps": step_count,
        "success": success,
    }


def create_mp4_from_frames(
    frames_dir: str,
    output_path: str = "episode.mp4",
    framerate: int = 10,
) -> bool:
    """Create MP4 From Frames.

    Encodes sequential PNG frames into an MP4 using ``ffmpeg``.

    Args:
        frames_dir (str): Directory containing frame PNG files.
        output_path (str): Output MP4 path.
        framerate (int): Output frame rate.

    Returns:
        bool: ``True`` on successful video creation, else ``False``.
    """
    import shutil

    # Check if ffmpeg is available
    if shutil.which("ffmpeg") is None:
        log.error(
            "ffmpeg not found. Install with: apt-get install ffmpeg (Ubuntu) or brew install ffmpeg (macOS)"
        )
        return False

    # Check if frames directory exists and has frames
    if not os.path.exists(frames_dir):
        log.error(f"Frames directory not found: {frames_dir}")
        return False

    frame_files = sorted(
        [
            f
            for f in os.listdir(frames_dir)
            if f.startswith("frame_") and f.endswith(".png")
        ]
    )
    if not frame_files:
        log.error(f"No frame files found in {frames_dir}")
        return False

    log.info(f"Creating MP4 video from {len(frame_files)} frames...")
    log.info(f"  Input frames: {frames_dir}/frame_*.png")
    log.info(f"  Output video: {output_path}")
    log.info(f"  Framerate: {framerate} fps")

    # Build ffmpeg command
    input_pattern = os.path.join(frames_dir, "frame_%04d.png")
    cmd = [
        "ffmpeg",
        "-framerate",
        str(framerate),
        "-i",
        input_pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-y",  # Overwrite output file without asking
        output_path,
    ]

    try:
        log.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            log.info(f"Video created successfully: {output_path}")
            return True
        else:
            log.error(f"ffmpeg failed with return code {result.returncode}")
            log.error(f"stderr: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        log.error("ffmpeg command timed out (>5 minutes)")
        return False
    except Exception as e:
        log.error(f"Failed to create video: {e}")
        return False


def main():
    """Run Visualization CLI.

    Parses CLI arguments, loads config and checkpoint, visualizes one episode,
    and optionally exports an MP4.

    Args:
        None: This function reads arguments from the command line.

    Returns:
        None: Results are logged and optional artifacts are written to disk.
    """
    parser = argparse.ArgumentParser(
        description="Visualize a trained RL agent solving a puzzle."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (e.g., checkpoints/exp1_mlp_2x3/best_model.pt)",
    )
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="Optional puzzle params override (e.g., 3x3b1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cpu, cuda, rocm",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save episode frames as PNG images to episode_frames/",
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        default="episode_frames",
        help="Directory to save frames (only used with --save-frames or --mp4-output)",
    )
    parser.add_argument(
        "--mp4-output",
        type=str,
        default=None,
        help="Output MP4 filename. If specified, automatically saves frames and creates video (requires ffmpeg)",
    )
    parser.add_argument(
        "--framerate",
        type=int,
        default=10,
        help="Video framerate in fps (default: 10, only used with --mp4-output)",
    )
    args = parser.parse_args()

    # If mp4_output is specified, automatically enable frame saving
    if args.mp4_output is not None:
        args.save_frames = True

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint):
        log.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    log.info(f"Visualizing agent from checkpoint: {args.checkpoint}")

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    log.info(f"Using device: {device}")

    # Load config
    cfg = load_config_from_checkpoint(args.checkpoint)
    log.info(f"Loaded config:\n{OmegaConf.to_yaml(cfg)}")

    # Override params if provided
    if args.params:
        cfg.env.params = args.params
        log.info(f"Overriding puzzle params to: {args.params}")

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create environment
    log.info("Creating environment...")
    env = create_env_for_visualization(cfg)

    # Load agent
    log.info("Loading agent...")
    agent = load_agent(args.checkpoint, cfg, device)
    log.info(f"Agent loaded: {agent.policy_net.__class__.__name__}")

    # Run visualization
    stats = visualize_episode(
        agent,
        env,
        cfg,
        max_steps=args.max_steps,
        save_frames=args.save_frames,
        output_dir=args.frames_dir,
    )
    log.info(f"Episode stats: {stats}")

    # The Simon Tatham C backend can segfault on explicit close at process
    # shutdown in some environments; leaving cleanup to interpreter teardown
    # avoids a successful run being reported as a crash.

    # Create video if mp4_output was specified
    if args.mp4_output is not None:
        log.info("Creating MP4 video from frames...")
        success = create_mp4_from_frames(
            frames_dir=args.frames_dir,
            output_path=args.mp4_output,
            framerate=args.framerate,
        )
        if success:
            log.info(f"Video saved to: {args.mp4_output}")
            log.info(
                f"You can play it with: mpv {args.mp4_output} (or any video player)"
            )
        else:
            log.error("Failed to create MP4 video")
        shutil.rmtree(args.frames_dir)  # Clean up frames directory after video creation

    log.info("Visualization complete!")


if __name__ == "__main__":
    main()
