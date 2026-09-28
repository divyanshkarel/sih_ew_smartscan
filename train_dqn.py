"""
train_dqn.py - Phase 2: Train Deep Q-Network Agent for Electronic Warfare Smart Scanning

Trains an agile receiver agent using Stable-Baselines3 DQN with the Time-Series Agile
Emitter Predictor wrapper. Configured with hyperparameters for rapid convergence,
checkpointing, and saving trained weights to smart_scan_dqn.zip.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from data_pipeline import (
    DEFAULT_MAX_ROWS,
    DEFAULT_NUM_BANDS,
    DEFAULT_TIME_STEP_US,
    discretize_pdw_stream,
    find_default_dataset_path,
    generate_synthetic_pdw_dataset,
    load_radar_data,
)
from rf_env import AgileEmitterPredictorWrapper, SmartScanEnv


# Training Configuration
TOTAL_TIMESTEPS: int = 50_000
CHECKPOINT_FREQ: int = 10_000
MODEL_SAVE_PATH: str = "smart_scan_dqn.zip"
CHECKPOINT_DIR: str = "checkpoints"

# Hyperparameters (as specified)
LEARNING_RATE: float = 1e-3
BUFFER_SIZE: int = 20_000
LEARNING_STARTS: int = 1_000
BATCH_SIZE: int = 64
GAMMA: float = 0.95
EXPLORATION_FRACTION: float = 0.20
EXPLORATION_FINAL_EPS: float = 0.05


class EWTrainingProgressCallback(BaseCallback):
    """
    Custom callback to log Electronic Warfare training progress, exploration rate,
    and rolling episode performance at regular step intervals.
    """

    def __init__(self, log_interval: int = 5_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.start_time = time.time()
        self.last_step = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.log_interval == 0:
            elapsed = time.time() - self.start_time
            fps = (self.n_calls - self.last_step) / max(time.time() - self.start_time, 1e-3)
            current_eps = getattr(self.model, "exploration_rate", 0.0)

            # Extract recent info metrics if available
            info = self.locals.get("infos", [{}])[0]
            pd = info.get("pd", 0.0) * 100
            pfa = info.get("pfa", 0.0) * 100
            cum_reward = info.get("cumulative_reward", 0.0)

            print(
                f"[*] Step {self.n_calls:6d}/{TOTAL_TIMESTEPS} | "
                f"Eps: {current_eps:.3f} | "
                f"Pd: {pd:5.2f}% | "
                f"Pfa: {pfa:5.2f}% | "
                f"Reward: {cum_reward:8.1f} | "
                f"Elapsed: {elapsed:5.1f}s ({fps:5.1f} steps/s)"
            )
            self.start_time = time.time()
            self.last_step = self.n_calls
        return True


def create_training_environment() -> AgileEmitterPredictorWrapper:
    """Ingests radar data and instantiates SmartScanEnv with the predictor wrapper."""
    dataset_path = find_default_dataset_path()
    if dataset_path and dataset_path.exists():
        print(f"[*] Ingesting radar dataset from: {dataset_path}")
        pdw_df = load_radar_data(dataset_path, max_rows=DEFAULT_MAX_ROWS)
    else:
        print("[!] Dataset path not found. Generating synthetic radar dataset...")
        pdw_df = generate_synthetic_pdw_dataset(num_pulses=DEFAULT_MAX_ROWS)

    truth_matrix, metadata = discretize_pdw_stream(
        pdw_df,
        num_bands=DEFAULT_NUM_BANDS,
        time_step_us=DEFAULT_TIME_STEP_US,
    )
    print(
        f"[*] Truth Matrix prepared: shape {truth_matrix.shape} "
        f"({metadata['total_time_steps']} steps, {metadata['num_bands']} bands, "
        f"density: {metadata['cell_density']*100:.2f}%)"
    )

    base_env = SmartScanEnv(
        truth_matrix=truth_matrix,
        metadata=metadata,
        hardware_penalty_coeff=0.2,
    )
    wrapped_env = AgileEmitterPredictorWrapper(base_env)
    return wrapped_env


def train():
    print("=" * 80)
    print("     PHASE 2: TRAINING DEEP Q-NETWORK FOR AGILE EW SMART SCANNING     ")
    print("=" * 80)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 1. Initialize Environment with Agile Predictor Wrapper
    env = create_training_environment()
    print(f"[*] Observation Space: {env.observation_space}")
    print(f"[*] Action Space:      {env.action_space}")

    # 2. Configure DQN Agent
    policy_kwargs = dict(
        net_arch=[128, 128],
    )

    print("\n[*] Initializing Stable-Baselines3 DQN Agent...")
    print(f"    - Policy:               MlpPolicy [128, 128]")
    print(f"    - Learning Rate:        {LEARNING_RATE}")
    print(f"    - Replay Buffer Size:   {BUFFER_SIZE:,}")
    print(f"    - Learning Starts:      {LEARNING_STARTS:,}")
    print(f"    - Batch Size:           {BATCH_SIZE}")
    print(f"    - Gamma:                {GAMMA}")
    print(f"    - Exploration Fraction: {EXPLORATION_FRACTION}")
    print(f"    - Final Epsilon:        {EXPLORATION_FINAL_EPS}")
    print(f"    - Total Timesteps:      {TOTAL_TIMESTEPS:,}")

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        learning_starts=LEARNING_STARTS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=EXPLORATION_FRACTION,
        exploration_final_eps=EXPLORATION_FINAL_EPS,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=42,
    )

    # 3. Callbacks for Checkpointing and Logging
    progress_callback = EWTrainingProgressCallback(log_interval=5_000)
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="dqn_smart_scan",
        verbose=0,
    )

    # 4. Execute Training Loop
    print("\n[*] Starting training loop...")
    t_start = time.time()
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[progress_callback, checkpoint_callback],
    )
    t_total = time.time() - t_start

    print("\n" + "=" * 80)
    print(f"[+] Training completed in {t_total:.2f} seconds ({TOTAL_TIMESTEPS / t_total:.1f} steps/s)")

    # 5. Save Model
    model.save(MODEL_SAVE_PATH)
    print(f"[+] Model weights successfully saved to: {MODEL_SAVE_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    train()
