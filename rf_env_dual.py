"""
rf_env_dual.py - Multi-Receiver Cooperative Scheduling Environment (DualSmartScanEnv)

Upgrade 1: Multi-Receiver Cooperative Scheduling for Electronic Warfare.
Simultaneously coordinates two agile receivers across N frequency bands (N=20, Action Space=400).
Decodes joint actions into (rx1_band, rx2_band), penalizes redundancy (-20.0 penalty for channel overlap),
awards hits independently (+10 each, once if overlapped), penalizes misses (-1 each), and accounts
for hardware frequency slew penalties on each receiver independently.
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


class DualSmartScanEnv(gym.Env):
    """
    DualSmartScanEnv: Cooperative Dual-Receiver RF Spectrum Scanning Environment.

    State (52 dimensions):
        - Rx1 currently tuned band (normalized [0, 1])
        - Rx2 currently tuned band (normalized [0, 1])
        - Rx1 sliding window of the last 5 time steps (1.0 for Hit, 0.0 for Miss)
        - Rx2 sliding window of the last 5 time steps (1.0 for Hit, 0.0 for Miss)
        - 20-band Exponential Moving Average (EMA) activity density vector
        - 20-band transition probability vector conditioned on recent agile hops

    Action Space:
        - Discrete(N * N) = Discrete(400) where N=20:
          rx1_band = action // 20
          rx2_band = action % 20

    Cooperative Reward Function:
        - Hits: +10 for each receiver landing on an active radar transmission.
        - Redundancy Penalty: If rx1_band == rx2_band, applies strict penalty of -20.0.
          If overlapping on a valid transmission, only awards the +10 hit once.
        - Misses: -1 for each receiver scanning empty air.
        - Hardware Switching Penalty: 0.2 * dist1 + 0.2 * dist2 subtracted from total reward.

    Metrics:
        - Pd: Cumulative unique hits / cumulative ground truth transmissions.
        - Pfa: Cumulative misses / total receiver scans (2 scans per time step).
        - Redundancy Count: Number of steps where receivers scanned the same band.
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
        redundancy_penalty: float = 20.0,
        density_alpha: float = 0.15,
        transition_decay: float = 0.999,
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
        self.redundancy_penalty = float(redundancy_penalty)
        self.density_alpha = float(density_alpha)
        self.transition_decay = float(transition_decay)

        # Action Space: Discrete(N * N) = Discrete(400)
        self.action_space = spaces.Discrete(self.N * self.N)

        # Observation Space: Box of shape (2 + 2 * window_size + 2 * N,) = 52
        # Elements 0, 1: normalized rx1_band, rx2_band
        # Elements 2..6: rx1 hit/miss history
        # Elements 7..11: rx2 hit/miss history
        # Elements 12..31: 20-band activity density
        # Elements 32..51: 20-band transition probabilities
        total_obs_dim = 2 + 2 * self.window_size + 2 * self.N
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(total_obs_dim,),
            dtype=np.float32,
        )

        # Internal Receiver State
        self.current_step: int = 0
        self.current_rx1_band: int = 0
        self.current_rx2_band: int = 10
        self.rx1_history: deque[float] = deque([0.0] * self.window_size, maxlen=self.window_size)
        self.rx2_history: deque[float] = deque([0.0] * self.window_size, maxlen=self.window_size)

        # Agile Emitter Predictor State
        self.activity_density = np.zeros(self.N, dtype=np.float32)
        self.transition_counts = np.ones((self.N, self.N), dtype=np.float32)
        self.last_hit_band: int | None = None

        # Cumulative Metrics Trackers
        self.cumulative_hits: int = 0
        self.cumulative_misses: int = 0
        self.total_scans: int = 0
        self.cumulative_transmissions: int = 0
        self.redundancy_count: int = 0
        self.total_hardware_penalty: float = 0.0
        self.total_intercept_time_error_us: float = 0.0
        self.intercept_count: int = 0
        self.cumulative_reward: float = 0.0

    def _get_transition_vector(self) -> np.ndarray:
        """Returns the estimated transition probabilities conditioned on last hit."""
        if self.last_hit_band is not None:
            row = self.transition_counts[self.last_hit_band, :]
            total = float(row.sum())
            return (row / total) if total > 0 else np.full(self.N, 1.0 / self.N, dtype=np.float32)
        else:
            return np.full(self.N, 1.0 / self.N, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        """Construct the 52-dimensional observation vector."""
        norm_rx1 = float(self.current_rx1_band) / max(float(self.N - 1), 1.0)
        norm_rx2 = float(self.current_rx2_band) / max(float(self.N - 1), 1.0)
        rx1_hist = np.array(self.rx1_history, dtype=np.float32)
        rx2_hist = np.array(self.rx2_history, dtype=np.float32)
        trans_vector = self._get_transition_vector().astype(np.float32)

        return np.concatenate(
            [
                np.array([norm_rx1, norm_rx2], dtype=np.float32),
                rx1_hist,
                rx2_hist,
                self.activity_density.copy(),
                trans_vector,
            ]
        ).astype(np.float32)

    def _get_info(
        self,
        rx1_band: int = 0,
        rx2_band: int = 0,
        is_redundant: bool = False,
        hit1: bool = False,
        hit2: bool = False,
        step_hits: int = 0,
        base_reward: float = 0.0,
        hardware_penalty: float = 0.0,
        step_intercept_error: float = 0.0,
    ) -> dict[str, Any]:
        """Compile running metrics and step details into info dictionary."""
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
            "rx1_band": rx1_band,
            "rx2_band": rx2_band,
            "is_redundant": is_redundant,
            "hit1": hit1,
            "hit2": hit2,
            "step_hits": step_hits,
            "base_reward": base_reward,
            "hardware_penalty": hardware_penalty,
            "pd": pd,
            "pfa": pfa,
            "redundancy_count": self.redundancy_count,
            "intercept_time_error_us": mean_intercept_error,
            "step_intercept_time_error_us": step_intercept_error,
            "cumulative_hits": self.cumulative_hits,
            "cumulative_misses": self.cumulative_misses,
            "total_scans": self.total_scans,
            "cumulative_transmissions": self.cumulative_transmissions,
            "cumulative_reward": self.cumulative_reward,
            "total_hardware_penalty": self.total_hardware_penalty,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        """Reset the dual environment to step 0."""
        super().reset(seed=seed)

        self.current_step = 0
        self.current_rx1_band = 0
        self.current_rx2_band = 10
        self.rx1_history = deque([0.0] * self.window_size, maxlen=self.window_size)
        self.rx2_history = deque([0.0] * self.window_size, maxlen=self.window_size)

        self.activity_density = np.zeros(self.N, dtype=np.float32)
        self.transition_counts = np.ones((self.N, self.N), dtype=np.float32)
        self.last_hit_band = None

        self.cumulative_hits = 0
        self.cumulative_misses = 0
        self.total_scans = 0
        self.cumulative_transmissions = 0
        self.redundancy_count = 0
        self.total_hardware_penalty = 0.0
        self.total_intercept_time_error_us = 0.0
        self.intercept_count = 0
        self.cumulative_reward = 0.0

        obs = self._get_obs()
        info = self._get_info(
            rx1_band=self.current_rx1_band,
            rx2_band=self.current_rx2_band,
            is_redundant=False,
        )
        return obs, info

    def step(
        self, action: int | np.integer
    ) -> Tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Execute one cooperative dual-receiver scan step.

        Parameters:
            action: Encoded joint action in [0, 399].
                    rx1_band = action // 20
                    rx2_band = action % 20

        Returns:
            observation: 52-dimensional state vector.
            reward: Net reward including hits, misses, redundancy penalty, and slew costs.
            terminated: True when all T time steps are completed.
            truncated: False.
            info: Comprehensive EW metrics dictionary.
        """
        action_int = int(action)
        if not self.action_space.contains(action_int):
            raise ValueError(
                f"Invalid action {action_int}. Must be an integer in [0, {self.action_space.n - 1}]."
            )

        if self.current_step >= self.T:
            raise RuntimeError(
                f"Step called after episode completion (step {self.current_step} >= {self.T}). Please reset."
            )

        # 1. Action Decoding
        rx1_band = action_int // self.N
        rx2_band = action_int % self.N

        is_redundant = bool(rx1_band == rx2_band)
        hit1 = bool(self.truth_matrix[self.current_step, rx1_band] == 1)
        hit2 = bool(self.truth_matrix[self.current_step, rx2_band] == 1)

        # 2. Cooperative Reward Calculation
        base_reward = 0.0
        step_hits = 0
        step_misses = 0

        if is_redundant:
            # Redundancy penalty: agent failed to coordinate distinct bands
            base_reward -= self.redundancy_penalty  # -20.0
            self.redundancy_count += 1

            if hit1:  # (hit1 == hit2 since same band)
                # Overlap on valid transmission: award +10 hit only once
                base_reward += self.hit_reward
                step_hits = 1
                step_misses = 0
            else:
                # Both receivers scanned empty air
                base_reward -= (2.0 * self.miss_penalty)  # -1 for Rx1, -1 for Rx2
                step_hits = 0
                step_misses = 2
        else:
            # Independent distinct channels
            if hit1:
                base_reward += self.hit_reward
                step_hits += 1
            else:
                base_reward -= self.miss_penalty
                step_misses += 1

            if hit2:
                base_reward += self.hit_reward
                step_hits += 1
            else:
                base_reward -= self.miss_penalty
                step_misses += 1

        # 3. Hardware Switching Slew Penalty
        dist1 = abs(rx1_band - self.current_rx1_band)
        dist2 = abs(rx2_band - self.current_rx2_band)
        hardware_penalty = (
            self.hardware_penalty_coeff * dist1 + self.hardware_penalty_coeff * dist2
        )
        self.total_hardware_penalty += hardware_penalty

        reward = float(base_reward - hardware_penalty)
        self.cumulative_reward += reward

        # 4. Update Receiver State & History
        self.current_rx1_band = rx1_band
        self.current_rx2_band = rx2_band
        self.rx1_history.append(1.0 if hit1 else 0.0)
        self.rx2_history.append(1.0 if hit2 else 0.0)

        # 5. Update Time-Series Activity Density & Transition Predictor
        if is_redundant:
            target = 1.0 if hit1 else 0.0
            self.activity_density[rx1_band] = (
                (1.0 - self.density_alpha) * self.activity_density[rx1_band]
                + self.density_alpha * target
            )
            unscanned = np.ones(self.N, dtype=bool)
            unscanned[rx1_band] = False
            self.activity_density[unscanned] *= (1.0 - 0.02)
        else:
            t1 = 1.0 if hit1 else 0.0
            t2 = 1.0 if hit2 else 0.0
            self.activity_density[rx1_band] = (
                (1.0 - self.density_alpha) * self.activity_density[rx1_band]
                + self.density_alpha * t1
            )
            self.activity_density[rx2_band] = (
                (1.0 - self.density_alpha) * self.activity_density[rx2_band]
                + self.density_alpha * t2
            )
            unscanned = np.ones(self.N, dtype=bool)
            unscanned[rx1_band] = False
            unscanned[rx2_band] = False
            self.activity_density[unscanned] *= (1.0 - 0.02)

        if hit1 or hit2:
            latest_hit = rx1_band if hit1 else rx2_band
            if self.last_hit_band is not None:
                self.transition_counts *= self.transition_decay
                self.transition_counts[self.last_hit_band, latest_hit] += 1.0
            self.last_hit_band = latest_hit

        # 6. Metrics Tracking (account for 2 scans per time step)
        self.total_scans += 2
        active_transmissions = int(self.truth_matrix[self.current_step].sum())
        self.cumulative_transmissions += active_transmissions
        self.cumulative_hits += step_hits
        self.cumulative_misses += step_misses

        # Intercept Time Error
        step_intercept_error = 0.0
        if step_hits > 0:
            self.intercept_count += step_hits
            scan_time_us = self.start_toa_us + (self.current_step + 0.5) * self.time_step_us
            if hit1 and self.pulse_toa_matrix is not None and not np.isnan(self.pulse_toa_matrix[self.current_step, rx1_band]):
                step_intercept_error += abs(scan_time_us - float(self.pulse_toa_matrix[self.current_step, rx1_band]))
            if hit2 and not is_redundant and self.pulse_toa_matrix is not None and not np.isnan(self.pulse_toa_matrix[self.current_step, rx2_band]):
                step_intercept_error += abs(scan_time_us - float(self.pulse_toa_matrix[self.current_step, rx2_band]))
            self.total_intercept_time_error_us += step_intercept_error

        self.current_step += 1
        terminated = bool(self.current_step >= self.T)
        truncated = False

        obs = self._get_obs()
        info = self._get_info(
            rx1_band=rx1_band,
            rx2_band=rx2_band,
            is_redundant=is_redundant,
            hit1=hit1,
            hit2=hit2,
            step_hits=step_hits,
            base_reward=base_reward,
            hardware_penalty=hardware_penalty,
            step_intercept_error=step_intercept_error,
        )

        return obs, reward, terminated, truncated, info


def make_default_dual_env(
    max_rows: int = DEFAULT_MAX_ROWS,
    num_bands: int = DEFAULT_NUM_BANDS,
    time_step_us: float = DEFAULT_TIME_STEP_US,
    **env_kwargs: Any,
) -> DualSmartScanEnv:
    """Helper to instantiate DualSmartScanEnv using local radar dataset."""
    dataset_path = find_default_dataset_path()
    if dataset_path and dataset_path.exists():
        pdw_df = load_radar_data(dataset_path, max_rows=max_rows)
    else:
        pdw_df = generate_synthetic_pdw_dataset(num_pulses=max_rows)

    truth_matrix, metadata = discretize_pdw_stream(
        pdw_df,
        num_bands=num_bands,
        time_step_us=time_step_us,
    )
    return DualSmartScanEnv(truth_matrix=truth_matrix, metadata=metadata, **env_kwargs)
