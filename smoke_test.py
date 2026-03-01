"""Smoke test: verify the DQN pipeline works end-to-end.

Note: PuzzleEnv uses a C backend + pygame that crashes if env.close() is called
and then a new env is created. We avoid closing envs between tests.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import FlattenObservation
from omegaconf import OmegaConf
import rlp

from agents.dqn_agent import DQNAgent
from agents.networks import MLPNetwork, CNNNetwork
from utils.replay_buffer import ReplayBuffer
from utils.env_factory import make_env, make_dual_obs_env


def test_networks():
    print("=== Testing Networks ===")
    mlp = MLPNetwork(input_dim=20, num_actions=5, hidden_sizes=[128, 128])
    x = torch.randn(4, 20)
    out = mlp(x)
    assert out.shape == (4, 5), f"MLP output shape wrong: {out.shape}"
    print(f"  MLP: input (4,20) -> output {out.shape} OK")

    cnn = CNNNetwork(3, 128, 128, 5, [32, 64, 64], [8, 4, 3], [4, 2, 1], 512)
    x = torch.randn(2, 3, 128, 128)
    out = cnn(x)
    assert out.shape == (2, 5), f"CNN output shape wrong: {out.shape}"
    print(f"  CNN: input (2,3,128,128) -> output {out.shape} OK")


def test_replay_buffer():
    print("=== Testing Replay Buffer ===")
    buf = ReplayBuffer(capacity=1000)
    for i in range(500):
        buf.push(
            state=np.random.randn(20).astype(np.float32),
            action=np.random.randint(5),
            reward=float(np.random.randn()),
            next_state=np.random.randn(20).astype(np.float32),
            done=bool(np.random.random() > 0.9),
        )
    assert len(buf) == 500
    batch = buf.sample(64, torch.device("cpu"))
    assert batch["states"].shape == (64, 20)
    assert batch["actions"].shape == (64, 1)
    print(f"  Buffer size={len(buf)}, sample shapes OK")


def _make_cfg(obs_type="puzzle_state"):
    return OmegaConf.create({
        "env": {
            "puzzle": "netslide",
            "params": "2x3b1",
            "obs_type": obs_type,
            "render_mode": "rgb_array",
            "window_width": 128,
            "window_height": 128,
            "allow_undo": False,
            "max_state_repeats": 200,
            "include_cursor_in_state_info": True,
        },
        "agent": {"type": "mlp", "hidden_sizes": [128, 128]},
        "training": {
            "batch_size": 32,
            "gamma": 0.99,
            "eps_start": 0.9,
            "eps_end": 0.05,
            "eps_decay": 1000,
            "tau": 0.005,
            "lr": 1e-4,
            "buffer_size": 10000,
            "total_episodes": 10,
            "max_steps": 100,
            "eval_interval": 5,
            "eval_episodes": 5,
            "checkpoint_interval": 10,
            "log_interval": 5,
        },
    })


def test_env_and_agent():
    """Test environment, dual obs, and agent in a single env lifecycle."""
    print("=== Testing Environment ===")
    cfg = _make_cfg()
    env = make_env(cfg)
    obs, info = env.reset(seed=42)
    print(f"  Obs shape (flattened): {obs.shape}")
    print(f"  Action space: {env.action_space}")
    mask = env.action_masks()
    print(f"  Action mask: {mask}")

    for i in range(10):
        valid = np.where(mask)[0]
        action = int(np.random.choice(valid)) if len(valid) > 0 else env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        mask = env.action_masks()
    print(f"  10 steps complete. Last obs shape: {obs.shape}")
    print("  Environment OK")

    print("=== Testing DQN Agent (1 episode) ===")
    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n
    print(f"  Obs shape: {obs_shape}, Actions: {num_actions}")

    agent = DQNAgent(
        obs_type="puzzle_state",
        obs_shape=obs_shape,
        num_actions=num_actions,
        cfg=cfg,
        device=torch.device("cpu"),
    )

    obs, info = env.reset(seed=42)
    total_return = 0.0
    for step in range(200):
        mask = env.action_masks()
        action = agent.select_action(obs, mask, explore=True)
        next_obs, reward, term, trunc, info = env.step(action)
        agent.replay_buffer.push(obs, action, reward, next_obs, term or trunc)
        loss = agent.optimize()
        agent.update_target_net()
        total_return += reward
        obs = next_obs
        if term or trunc:
            break

    print(f"  Episode: {step+1} steps, return={total_return}, buffer={len(agent.replay_buffer)}")

    # Test save/load
    agent.save("/tmp/test_agent.pt")
    agent.load("/tmp/test_agent.pt")
    print("  Save/load OK")
    print("  Agent episode OK")

    # Don't close env — the C backend doesn't support recreating after close


def test_dual_obs_standalone():
    """Test DualObsWrapper in its own env lifecycle."""
    print("=== Testing Dual Obs Environment ===")
    cfg = _make_cfg()
    env = make_dual_obs_env(cfg)
    obs, info = env.reset(seed=42)
    assert "obs_discrete" in info, "obs_discrete missing from info"
    assert "obs_rgb" in info, "obs_rgb missing from info"
    print(f"  obs_discrete shape: {info['obs_discrete'].shape}")
    print(f"  obs_rgb shape: {info['obs_rgb'].shape}")

    mask = env.action_masks()
    valid = np.where(mask)[0]
    action = int(np.random.choice(valid))
    obs, reward, term, trunc, info = env.step(action)
    assert "obs_discrete" in info
    assert "obs_rgb" in info
    print("  Step with dual obs OK")
    print("  Dual obs wrapper OK")


if __name__ == "__main__":
    test_networks()
    test_replay_buffer()
    test_dual_obs_standalone()
    test_env_and_agent()
    print("\n=== All smoke tests PASSED ===")
