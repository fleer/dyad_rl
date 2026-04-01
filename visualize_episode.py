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
    NormalizeDualPuzzleStateWrapper,
    NormalizePuzzleStateWrapper,
    process_obs,
)

log = logging.getLogger(__name__)


class ActionMaskWrapper(gym.Wrapper):
    """Ensures action_masks() is accessible through wrapper chain."""

    def action_masks(self) -> np.ndarray:
        return self.env.unwrapped.action_masks()


def find_experiment_config(checkpoint_name: str) -> str | None:
    """Find the most recent .hydra/config.yaml for a given experiment.

    checkpoint_name: e.g., "exp1_mlp_2x3" — we'll search outputs/ for matching config.
    """
    outputs_dir = "outputs"
    if not os.path.exists(outputs_dir):
        return None

    # Recursively search for .hydra/config.yaml files
    for root, dirs, files in os.walk(outputs_dir):
        if ".hydra" in root and "config.yaml" in os.listdir(root):
            config_path = os.path.join(root, "config.yaml")
            with open(config_path) as f:
                cfg = OmegaConf.load(config_path)
                if cfg.get("experiment_name", "").lower() == checkpoint_name.lower():
                    return config_path

    return None


def load_config_from_checkpoint(checkpoint_path: str) -> DictConfig:
    """Load or infer experiment config based on checkpoint path.

    Tries to find the matching .hydra/config.yaml from outputs/ directory.
    Falls back to default config if not found.
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

    # Create a minimal config
    cfg = OmegaConf.create({
        "env": {
            "puzzle": "netslide",
            "params": "2x3b1",
            "obs_type": "puzzle_state",
            "render_mode": "rgb_array",
            "window_width": 128,
            "window_height": 128,
            "allow_undo": False,
            "max_state_repeats": 200,
            "include_cursor_in_state_info": True,
        },
        "agent": {
            "type": "mlp" if "mlp" in experiment_name.lower() else "cnn",
            "learning_rate": 1e-4,
            "net_arch": [64, 32, 16],
        },
        "training": {
            "total_episodes": 40000,
            "max_steps": 1000,
        },
        "seed": 42,
        "device": "auto",
    })

    return cfg


def create_env_for_visualization(cfg: DictConfig) -> gym.Env:
    """Create environment with rgb_array render mode.
    
    Note: render_mode="human" can cause segfaults with the C pygame backend,
    so we use rgb_array which is more stable.
    """
    log.info(f"Creating environment in rgb_array mode (recommended for stability)")

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


def get_obs_shape_for_agent(cfg: DictConfig) -> tuple[int, ...]:
    """Determine observation shape based on agent type and config."""
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
        
        # Fallback: Try to create a temp env (may crash with C backend)
        log.warning("Could not infer obs shape, attempting to create temp environment...")
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
    """Load a DQN agent from checkpoint."""
    log.info(f"Building agent with config...")
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
    
    total_steps = cfg.training.total_episodes * cfg.training.max_steps
    log.info(f"  Total steps for schedules: {total_steps}")

    log.info(f"Initializing agent...")
    agent = DQNAgent(
        obs_type=cfg.env.obs_type,
        obs_shape=obs_shape,
        num_actions=num_actions,
        cfg=cfg,
        device=device,
        total_steps=total_steps,
    )
    
    log.info(f"Agent initialized. Loading checkpoint...")
    log.info(f"  Checkpoint path: {checkpoint_path}")
    log.info(f"  File exists: {os.path.exists(checkpoint_path)}")
    log.info(f"  File size: {os.path.getsize(checkpoint_path) if os.path.exists(checkpoint_path) else 'N/A'} bytes")

    # Load the checkpoint
    try:
        agent.load(checkpoint_path)
        log.info(f"Checkpoint loaded successfully")
    except RuntimeError as e:
        log.error(f"Failed to load checkpoint: {e}")
        # Try to infer num_actions from checkpoint
        log.info("Attempting to auto-detect num_actions from checkpoint...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            # Get the output layer weight shape
            policy_state = checkpoint["policy_net"]
            # The last layer should be the output layer
            for key in sorted(policy_state.keys(), reverse=True):
                if "weight" in key and "bias" not in key:
                    last_layer_weight = policy_state[key]
                    if last_layer_weight.dim() == 2 and last_layer_weight.shape[1] == 16:  # 16 is last hidden layer size
                        num_actions = last_layer_weight.shape[0]
                        log.info(f"Auto-detected num_actions: {num_actions}")
                        break
            
            # Recreate agent with correct num_actions
            agent = DQNAgent(
                obs_type=cfg.env.obs_type,
                obs_shape=obs_shape,
                num_actions=num_actions,
                cfg=cfg,
                device=device,
                total_steps=total_steps,
            )
            agent.load(checkpoint_path)
            log.info(f"Checkpoint loaded successfully after auto-detection")
        except Exception as e2:
            log.error(f"Still failed: {e2}")
            raise

    agent.policy_net.eval()  # Set to eval mode
    log.info(f"Agent set to eval mode")

    return agent


def visualize_episode(
    agent: DQNAgent,
    env: gym.Env,
    cfg: DictConfig,
    max_steps: int = 1000,
    save_frames: bool = False,
    output_dir: str = "episode_frames",
) -> dict:
    """Run one episode with the trained agent and visualize it.
    
    Optionally saves frames to disk for creating a video later.

    Returns:
        Dictionary with episode stats (reward, length, success).
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

    log.info(f"Episode finished!")
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
    """Create an MP4 video from PNG frames using ffmpeg.
    
    Args:
        frames_dir: Directory containing frame_0000.png, frame_0001.png, etc.
        output_path: Output video filename (default: episode.mp4)
        framerate: Output video framerate in fps (default: 10)
    
    Returns:
        True if successful, False otherwise.
    """
    import shutil
    
    # Check if ffmpeg is available
    if shutil.which("ffmpeg") is None:
        log.error("ffmpeg not found. Install with: apt-get install ffmpeg (Ubuntu) or brew install ffmpeg (macOS)")
        return False
    
    # Check if frames directory exists and has frames
    if not os.path.exists(frames_dir):
        log.error(f"Frames directory not found: {frames_dir}")
        return False
    
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_") and f.endswith(".png")])
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
        "-framerate", str(framerate),
        "-i", input_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
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
    logging.basicConfig(
        level=logging.INFO, format="%(name)s %(levelname)s %(message)s"
    )

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
    try:
        stats = visualize_episode(
            agent,
            env,
            cfg,
            max_steps=args.max_steps,
            save_frames=args.save_frames,
            output_dir=args.frames_dir,
        )
        log.info(f"Episode stats: {stats}")
    finally:
        env.close()

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
            log.info(f"You can play it with: mpv {args.mp4_output} (or any video player)")
        else:
            log.error("Failed to create MP4 video")
        shutil.rmtree(args.frames_dir)  # Clean up frames directory after video creation

    log.info("Visualization complete!")


if __name__ == "__main__":
    main()
