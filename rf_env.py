"""
rf_env.py - Electronic Warfare Reinforcement Learning Environment (SmartScanEnv)

Phase 1 EW Reinforcement Learning Simulation.
Provides a Gymnasium-compliant RF spectrum scanning environment where an agile receiver
agent selects frequency bands to intercept radar transmissions while balancing hardware
tuning penalties. Implements running trackers for Pd, Pfa, and Intercept Time Error.
Includes an open-loop linear sweep baseline agent.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from data_pipeline import (
    DEFAULT_MAX_ROWS,
    DEFAULT_NUM_BANDS,
    DEFAULT_TIME_STEP_US,
    discretize_pdw_stream,
    find_default_dataset_path,
    generate_synthetic_pdw_dataset,
    load_radar_data,
)


class SmartScanEnv(gym.Env):
    """
    SmartScanEnv: Custom Gymnasium Environment for Agile RF Receiver Control.

    State:
        - Currently tuned frequency band (0 to N - 1)
        - Sliding window of the last 5 time steps (1.0 for Hit, 0.0 for Miss)

    Actions:
        - Discrete(N): Receiver tunes to one of N frequency bands [0, N - 1].

    Reward:
        - Base reward: +10 for Hit (transmission present), -1 for Miss (empty).
        - Hardware penalty: -0.2 * abs(current_band - chosen_band) to penalize
          wide frequency jumps due to local oscillator settling latency.

    Metrics:
        - Probability of Detection (Pd): Successful intercepts / total actual transmissions.
        - Probability of False Alarm (Pfa): Empty scans / total scans.
        - Intercept Time Error: Time difference between truth ToA and receiver scan time.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        truth_matrix: np.ndarray,
        metadata: dict[str, Any] | None = None,
        hardware_penalty_coeff: float = 0.2,
        window_size: int = 5,
        hit_reward: float = 10.0,
        miss_penalty: float = 1.0,
    ) -> None:
        super().__init__()

        if truth_matrix.ndim != 2:
            raise ValueError(
                f"truth_matrix must be 2D of shape (T, N), got shape {truth_matrix.shape}"
            )

        self.truth_matrix = np.asarray(truth_matrix, dtype=np.uint8)
        self.T, self.N = self.truth_matrix.shape

        self.env_metadata = metadata or {}
        self.time_step_us: float = float(
            self.env_metadata.get("time_step_us", DEFAULT_TIME_STEP_US)
        )
        self.start_toa_us: float = float(self.env_metadata.get("min_toa_us", 0.0))
        self.pulse_toa_matrix: np.ndarray | None = self.env_metadata.get(
            "pulse_toa_matrix", None
        )

        self.hardware_penalty_coeff = float(hardware_penalty_coeff)
        self.window_size = int(window_size)
        self.hit_reward = float(hit_reward)
        self.miss_penalty = float(miss_penalty)

        # Action Space: spaces.Discrete(N)
        self.action_space = spaces.Discrete(self.N)

        # Observation Space: Box of shape (1 + window_size,)
        # Element 0: currently tuned band [0, N - 1]
        # Elements 1..window_size: sliding window of hits/misses [0.0, 1.0]
        obs_low = np.array([0.0] + [0.0] * self.window_size, dtype=np.float32)
        obs_high = np.array([float(self.N - 1)] + [1.0] * self.window_size, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, shape=(1 + self.window_size,), dtype=np.float32
        )

        # Internal state
        self.current_step: int = 0
        self.current_band: int = 0
        self.history: deque[float] = deque([0.0] * self.window_size, maxlen=self.window_size)

        # Metric trackers
        self.cumulative_hits: int = 0
        self.cumulative_misses: int = 0
        self.total_scans: int = 0
        self.cumulative_transmissions: int = 0
        self.total_intercept_time_error_us: float = 0.0
        self.intercept_count: int = 0
        self.cumulative_reward: float = 0.0

    def _get_obs(self) -> np.ndarray:
        """Construct the observation vector [current_band, h_4, h_3, h_2, h_1, h_0]."""
        obs = np.array([float(self.current_band), *self.history], dtype=np.float32)
        return obs

    def _get_info(
        self,
        is_hit: bool = False,
        base_reward: float = 0.0,
        hardware_penalty: float = 0.0,
        jump_distance: int = 0,
        step_intercept_error: float = 0.0,
    ) -> dict[str, Any]:
        """Compute running metrics and compile info dictionary."""
        pd = (
            float(self.cumulative_hits / self.cumulative_transmissions)
            if self.cumulative_transmissions > 0
            else 0.0
        )
        pfa = (
            float(self.cumulative_misses / self.total_scans)
            if self.total_scans > 0
            else 0.0
        )
        mean_intercept_error = (
            float(self.total_intercept_time_error_us / self.intercept_count)
            if self.intercept_count > 0
            else 0.0
        )

        return {
            "step": self.current_step,
            "is_hit": is_hit,
            "base_reward": base_reward,
            "hardware_penalty": hardware_penalty,
            "jump_distance": jump_distance,
            "pd": pd,
            "pfa": pfa,
            "intercept_time_error_us": mean_intercept_error,
            "step_intercept_time_error_us": step_intercept_error,
            "cumulative_hits": self.cumulative_hits,
            "cumulative_misses": self.cumulative_misses,
            "total_scans": self.total_scans,
            "cumulative_transmissions": self.cumulative_transmissions,
            "cumulative_reward": self.cumulative_reward,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to initial conditions at time step 0."""
        super().reset(seed=seed)

        self.current_step = 0
        self.current_band = 0
        self.history = deque([0.0] * self.window_size, maxlen=self.window_size)

        self.cumulative_hits = 0
        self.cumulative_misses = 0
        self.total_scans = 0
        self.cumulative_transmissions = 0
        self.total_intercept_time_error_us = 0.0
        self.intercept_count = 0
        self.cumulative_reward = 0.0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: int | np.integer
    ) -> Tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Execute one receiver scan step.

        Parameters:
            action: Selected frequency band index in [0, N - 1].

        Returns:
            observation: Updated agent state vector.
            reward: Base reward minus hardware slew penalty.
            terminated: True when episode time steps exhausted.
            truncated: False.
            info: Real-time EW metrics dictionary.
        """
        chosen_band = int(action)
        if not self.action_space.contains(chosen_band):
            raise ValueError(
                f"Invalid action {chosen_band}. Must be an integer in [0, {self.N - 1}]."
            )

        if self.current_step >= self.T:
            raise RuntimeError(
                f"Step called after episode completion (step {self.current_step} >= {self.T}). Please reset."
            )

        # 1. Hardware Slew Penalty: proportional to frequency jump distance
        jump_distance = abs(chosen_band - self.current_band)
        hardware_penalty = self.hardware_penalty_coeff * jump_distance

        # 2. Check Ground Truth Transmission in Chosen Band
        is_hit = bool(self.truth_matrix[self.current_step, chosen_band] == 1)

        # 3. Base Reward
        base_reward = self.hit_reward if is_hit else -self.miss_penalty
        reward = float(base_reward - hardware_penalty)
        self.cumulative_reward += reward

        # 4. Receiver State Update
        self.current_band = chosen_band
        self.history.append(1.0 if is_hit else 0.0)

        # 5. Running Metrics Tracking
        self.total_scans += 1
        active_in_current_step = int(self.truth_matrix[self.current_step].sum())
        self.cumulative_transmissions += active_in_current_step

        step_intercept_error: float = 0.0
        if is_hit:
            self.cumulative_hits += 1
            self.intercept_count += 1

            # Receiver scan nominal time
            scan_time_us = self.start_toa_us + (self.current_step + 0.5) * self.time_step_us

            # Calculate Intercept Time Error against truth ToA
            if (
                self.pulse_toa_matrix is not None
                and not np.isnan(self.pulse_toa_matrix[self.current_step, chosen_band])
            ):
                actual_toa_us = float(self.pulse_toa_matrix[self.current_step, chosen_band])
                step_intercept_error = abs(scan_time_us - actual_toa_us)
            else:
                # Default within-step uncertainty
                step_intercept_error = 0.5 * self.time_step_us

            self.total_intercept_time_error_us += step_intercept_error
        else:
            self.cumulative_misses += 1

        # 6. Advance Time Step
        self.current_step += 1
        terminated = self.current_step >= self.T
        truncated = False

        obs = self._get_obs()
        info = self._get_info(
            is_hit=is_hit,
            base_reward=base_reward,
            hardware_penalty=hardware_penalty,
            jump_distance=jump_distance,
            step_intercept_error=step_intercept_error,
        )

        return obs, reward, terminated, truncated, info


class AgileEmitterPredictorWrapper(gym.Wrapper):
    """
    Time-Series Agile Emitter Predictor Wrapper for SmartScanEnv.

    Augments the agent's observation space with:
    1. Normalized currently tuned frequency band [1 dim].
    2. Sliding window of recent scan hit/miss outcomes [window_size dim].
    3. Moving-average pulse activity density across all N bands [N dim].
       Tracks which channels have active radar emissions using exponential decay.
    4. Empirical frequency-hopping transition probabilities [N dim].
       Tracks the transition likelihood P(next_band = j | last_hit_band = i)
       conditioned on the most recently intercepted emitter band.

    Total observation dimension: 1 + window_size + 2 * N (e.g., 46 for N=20).
    """

    def __init__(
        self,
        env: SmartScanEnv,
        density_alpha: float = 0.15,
        transition_decay: float = 0.999,
    ) -> None:
        super().__init__(env)
        self.num_bands = env.N
        self.window_size = env.window_size
        self.density_alpha = float(density_alpha)
        self.transition_decay = float(transition_decay)

        total_obs_dim = 1 + self.window_size + 2 * self.num_bands
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(total_obs_dim,),
            dtype=np.float32,
        )

        self.activity_density = np.zeros(self.num_bands, dtype=np.float32)
        self.transition_counts = np.ones((self.num_bands, self.num_bands), dtype=np.float32)
        self.last_hit_band: int | None = None

    def _get_transition_vector(self) -> np.ndarray:
        """Returns the estimated transition probabilities conditioned on last hit."""
        if self.last_hit_band is not None:
            row = self.transition_counts[self.last_hit_band, :]
            total = float(row.sum())
            return (row / total) if total > 0 else np.full(self.num_bands, 1.0 / self.num_bands, dtype=np.float32)
        else:
            return np.full(self.num_bands, 1.0 / self.num_bands, dtype=np.float32)

    def _augment_obs(self, base_obs: np.ndarray) -> np.ndarray:
        """Combine base observation with activity density and transition predictor."""
        norm_band = base_obs[0] / max(float(self.num_bands - 1), 1.0)
        hits_misses = base_obs[1 : 1 + self.window_size]
        trans_vector = self._get_transition_vector()

        return np.concatenate(
            [
                np.array([norm_band], dtype=np.float32),
                hits_misses.astype(np.float32),
                self.activity_density.copy(),
                trans_vector.astype(np.float32),
            ]
        ).astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        base_obs, info = self.env.reset(seed=seed, options=options)
        self.activity_density = np.zeros(self.num_bands, dtype=np.float32)
        self.transition_counts = np.ones((self.num_bands, self.num_bands), dtype=np.float32)
        self.last_hit_band = None

        aug_obs = self._augment_obs(base_obs)
        return aug_obs, info

    def step(
        self, action: int | np.integer
    ) -> Tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        chosen_band = int(action)
        base_obs, reward, terminated, truncated, info = self.env.step(chosen_band)

        is_hit = info["is_hit"]

        # Update Time-Series Agile Emitter Predictor State:
        # 1. Update activity density
        target = 1.0 if is_hit else 0.0
        self.activity_density[chosen_band] = (
            (1.0 - self.density_alpha) * self.activity_density[chosen_band]
            + self.density_alpha * target
        )
        unscanned_mask = np.ones(self.num_bands, dtype=bool)
        unscanned_mask[chosen_band] = False
        self.activity_density[unscanned_mask] *= (1.0 - 0.02)

        # 2. Update Frequency Hopping Transition Matrix
        if is_hit:
            if self.last_hit_band is not None:
                self.transition_counts *= self.transition_decay
                self.transition_counts[self.last_hit_band, chosen_band] += 1.0
            self.last_hit_band = chosen_band

        aug_obs = self._augment_obs(base_obs)
        return aug_obs, reward, terminated, truncated, info


def run_linear_sweep_baseline(env: SmartScanEnv) -> dict[str, Any]:
    """
    Executes an Open Loop Linear Sweep agent scanning frequency bands 1 to N
    sequentially (0 to N - 1 in 0-indexed representation) across the environment.
    """
    obs, info = env.reset()
    num_bands = env.N
    total_steps = env.T

    for step_idx in range(total_steps):
        # Open Loop: linear sequential raster scan from band 0 to N - 1 cyclically
        action = step_idx % num_bands
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    return info


if __name__ == "__main__":
    print("=" * 75)
    print("Electronic Warfare RL Simulation - RF Environment & Baseline (Phase 1)")
    print("=" * 75)

    # 1. Ingest Data via data_pipeline
    dataset_path = find_default_dataset_path()
    if dataset_path and dataset_path.exists():
        print(f"[*] Ingesting local dataset: {dataset_path}")
        pdw_df = load_radar_data(dataset_path, max_rows=DEFAULT_MAX_ROWS)
    else:
        print("[!] Generating synthetic dataset for demonstration...")
        pdw_df = generate_synthetic_pdw_dataset(num_pulses=DEFAULT_MAX_ROWS)

    truth_matrix, metadata = discretize_pdw_stream(
        pdw_df,
        num_bands=DEFAULT_NUM_BANDS,
        time_step_us=DEFAULT_TIME_STEP_US,
    )
    print(f"[*] Truth Matrix loaded: shape {truth_matrix.shape}, density {metadata['cell_density']*100:.2f}%")

    # 2. Instantiate Gymnasium Environment
    env = SmartScanEnv(
        truth_matrix=truth_matrix,
        metadata=metadata,
        hardware_penalty_coeff=0.2,
    )

    print(f"[*] Initialized SmartScanEnv:")
    print(f"    - Action Space: {env.action_space}")
    print(f"    - Observation Space: {env.observation_space}")
    print(f"    - Episode Length (T): {env.T} time steps")
    print(f"    - Frequency Bands (N): {env.N}")

    # 3. Run Open Loop Linear Sweep Baseline Agent
    print("\n[*] Executing Open Loop Linear Sweep Baseline Agent (Scanning bands 1 to N sequentially)...")
    final_metrics = run_linear_sweep_baseline(env)

    # 4. Display Final Metrics Table
    print("\n" + "=" * 75)
    print("                  FINAL BASELINE EVALUATION METRICS                  ")
    print("=" * 75)
    print(f" Total Episode Steps (T)           : {final_metrics['step']:,}")
    print(f" Total Receiver Scans              : {final_metrics['total_scans']:,}")
    print(f" Successful Intercepts (Hits)      : {final_metrics['cumulative_hits']:,}")
    print(f" Empty Scans (Misses)              : {final_metrics['cumulative_misses']:,}")
    print(f" Total Actual Transmissions        : {final_metrics['cumulative_transmissions']:,}")
    print(f" Cumulative Episode Reward         : {final_metrics['cumulative_reward']:,.2f}")
    print("-" * 75)
    print(f" Probability of Detection (Pd)     : {final_metrics['pd'] * 100:.2f}%")
    print(f" Probability of False Alarm (Pfa)  : {final_metrics['pfa'] * 100:.2f}%")
    print(f" Mean Intercept Time Error         : {final_metrics['intercept_time_error_us']:.2f} us")
    print("=" * 75)
