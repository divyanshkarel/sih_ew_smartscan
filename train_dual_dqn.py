"""
train_dual_dqn.py - Upgrade 1: Train Deep Q-Network for Multi-Receiver Cooperative Scheduling

Instantiates DualSmartScanEnv with 400 discrete joint actions.
Configures Stable-Baselines3 DQN (MlpPolicy, net_arch=[256, 256]) to learn non-redundant,
collaborative spectrum scanning across agile radar emitters.
Trains for 100,000 steps, saves model as smart_scan_dual_dqn.zip, and automatically executes
evaluate_dual.py upon completion.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from evaluate_dual import evaluate_dual_system
from rf_env_dual import DualSmartScanEnv, make_default_dual_env


# Training Configuration
TOTAL_TIMESTEPS: int = 100_000
MODEL_SAVE_PATH: str = "smart_scan_dual_dqn.zip"
LOG_INTERVAL: int = 10_000

# Hyperparameters for 400-Action Cooperative Space
LEARNING_RATE: float = 7e-4
BUFFER_SIZE: int = 50_000
LEARNING_STARTS: int = 1_000
BATCH_SIZE: int = 128
GAMMA: float = 0.95
EXPLORATION_FRACTION: float = 0.30
EXPLORATION_FINAL_EPS: float = 0.03
TARGET_UPDATE_INTERVAL: int = 1_000
NET_ARCH: list[int] = [256, 256]


class DualTrainingProgressCallback(BaseCallback):
    """Logs cooperative dual-receiver training metrics, exploration rate, and speed."""

    def __init__(self, log_interval: int = LOG_INTERVAL, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.start_time = time.time()
        self.last_step = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.log_interval == 0:
            elapsed = time.time() - self.start_time
            fps = (self.n_calls - self.last_step) / max(elapsed, 1e-3)
            current_eps = getattr(self.model, "exploration_rate", 0.0)

            info = self.locals.get("infos", [{}])[0]
            pd = info.get("pd", 0.0) * 100.0
            pfa = info.get("pfa", 0.0) * 100.0
            cum_reward = info.get("cumulative_reward", 0.0)
            redundancies = info.get("redundancy_count", 0)

            print(
                f"[*] Step {self.n_calls:6d}/{TOTAL_TIMESTEPS} | "
                f"Eps: {current_eps:.3f} | "
                f"Pd: {pd:5.2f}% | "
                f"Pfa: {pfa:5.2f}% | "
                f"Redundant: {redundancies:4d} | "
                f"Reward: {cum_reward:9.1f} | "
                f"Speed: {fps:5.1f} steps/s",
                flush=True,
            )
            self.start_time = time.time()
            self.last_step = self.n_calls
        return True


def train_dual_agent() -> DQN:
    """Main training routine for DualSmartScanEnv with automated benchmark evaluation."""
    # Ensure UTF-8 output on Windows
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("=" * 84)
    print("   UPGRADE 1: TRAINING MULTI-RECEIVER COOPERATIVE SCHEDULING AGENT (DQN)   ")
    print("=" * 84)

    # 1. Initialize Dual Environment
    print("[*] Instantiating DualSmartScanEnv (20 bands, 400 discrete joint actions)...")
    env = make_default_dual_env()
    print(f"    - Action Space      : {env.action_space} (400 joint receiver pairs)")
    print(f"    - Observation Space : {env.observation_space.shape} (52-D continuous EW state)")
    print(f"    - Episode Length    : {env.T:,} time steps (2 scans/step = {env.T * 2:,} scans)")

    # 2. Configure DQN Agent
    policy_kwargs = dict(net_arch=NET_ARCH)
    print(f"\n[*] Configuring Stable-Baselines3 DQN:")
    print(f"    - Policy            : MlpPolicy (net_arch={NET_ARCH})")
    print(f"    - Learning Rate     : {LEARNING_RATE}")
    print(f"    - Buffer Size       : {BUFFER_SIZE:,}")
    print(f"    - Batch Size        : {BATCH_SIZE}")
    print(f"    - Gamma             : {GAMMA}")
    print(f"    - Exploration       : {EXPLORATION_FRACTION * 100:.0f}% decay to eps={EXPLORATION_FINAL_EPS}")
    print(f"    - Target Update Int : {TARGET_UPDATE_INTERVAL} steps")

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=TARGET_UPDATE_INTERVAL,
        exploration_fraction=EXPLORATION_FRACTION,
        exploration_initial_eps=1.0,
        exploration_final_eps=EXPLORATION_FINAL_EPS,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=42,
    )

    # 3. Train Model
    print(f"\n[*] Launching training for {TOTAL_TIMESTEPS:,} timesteps...")
    callback = DualTrainingProgressCallback(log_interval=LOG_INTERVAL)
    start_train_time = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    train_duration = time.time() - start_train_time

    print(f"\n[+] Training complete in {train_duration:.2f}s ({TOTAL_TIMESTEPS / train_duration:.1f} steps/s)!")

    # 4. Save Trained Weights
    model.save(MODEL_SAVE_PATH)
    print(f"[+] Model weights saved to: {Path(MODEL_SAVE_PATH).resolve()}")

    # 5. Automatically Run Comparative Evaluation
    print("\n[*] Automatically executing benchmark evaluation against Dual Linear Baseline...")
    evaluate_dual_system(model_path=MODEL_SAVE_PATH)

    return model


if __name__ == "__main__":
    train_dual_agent()
