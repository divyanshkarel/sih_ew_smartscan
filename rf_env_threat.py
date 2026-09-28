"""
rf_env_threat.py - Threat Priority Engine Environment (ThreatAwareDualEnv)

Upgrade 2: Threat Priority Engine for Dual-Receiver Electronic Warfare.
Prioritizes high-lethality radar bands over routine transmissions:
  - Tier 3 (Routine / Surveillance): Bands 0 to 9 (Base Hit Reward: +5.0)
  - Tier 2 (High / Agile Interceptors): Bands 10 to 15 (Base Hit Reward: +20.0)
  - Tier 1 (Critical / Fire-Control Missile Lock): Bands 16 to 19 (Base Hit Reward: +50.0,
    Miss Penalty if an active lethal transmission is missed: -20.0)

Maintains cooperative dual-receiver coordination with 400 joint actions, redundancy penalty (-20.0),
hardware slew penalties (0.2 * distance), and running tier-specific detection metrics (Pd_Tier1, Pd_Tier2, Pd_Tier3).
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


# Threat Tier Definitions
TIER1_BANDS: tuple[int, ...] = (16, 17, 18, 19)   # Critical: Fire-Control Radar / Missile Lock
TIER2_BANDS: tuple[int, ...] = (10, 11, 12, 13, 14, 15)  # High: Agile Emitters / Interceptors
TIER3_BANDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)  # Routine: Early Warning / Surveillance

TIER1_REWARD: float = 50.0
TIER2_REWARD: float = 20.0
TIER3_REWARD: float = 5.0
TIER1_MISSED_PULSE_PENALTY: float = 20.0
ROUTINE_MISS_PENALTY: float = 1.0
REDUNDANCY_PENALTY: float = 20.0


class ThreatAwareDualEnv(gym.Env):
    """
    ThreatAwareDualEnv: Threat-Prioritized Dual-Receiver Electronic Warfare Environment.

    Inherits the 400-action dual-receiver scheduling mechanics from DualSmartScanEnv,
    integrating a lethality-weighted reward structure and specialized Tier 1-3 detection metrics.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        truth_matrix: np.ndarray,
        metadata: dict[str, Any] | None = None,
        hardware_penalty_coeff: float = 0.2,
        window_size: int = 5,
        tier1_reward: float = TIER1_REWARD,
        tier2_reward: float = TIER2_REWARD,
        tier3_reward: float = TIER3_REWARD,
        tier1_miss_penalty: float = TIER1_MISSED_PULSE_PENALTY,
        miss_penalty: float = ROUTINE_MISS_PENALTY,
        redundancy_penalty: float = REDUNDANCY_PENALTY,
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
        self.tier1_reward = float(tier1_reward)
        self.tier2_reward = float(tier2_reward)
        self.tier3_reward = float(tier3_reward)
        self.tier1_miss_penalty = float(tier1_miss_penalty)
        self.miss_penalty = float(miss_penalty)
        self.redundancy_penalty = float(redundancy_penalty)
        self.density_alpha = float(density_alpha)
        self.transition_decay = float(transition_decay)

        # Action Space: Discrete(400)
        self.action_space = spaces.Discrete(self.N * self.N)

        # Observation Space: 52-dimensional state vector (matching DualSmartScanEnv)
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
        self.current_rx2_band: int = 16  # Initialize near high-priority sector
        self.rx1_history: deque[float] = deque([0.0] * self.window_size, maxlen=self.window_size)
        self.rx2_history: deque[float] = deque([0.0] * self.window_size, maxlen=self.window_size)

        # Agile Emitter Predictor State
        self.activity_density = np.zeros(self.N, dtype=np.float32)
        self.transition_counts = np.ones((self.N, self.N), dtype=np.float32)
        self.last_hit_band: int | None = None

        # Cumulative Overall Metrics
        self.cumulative_hits: int = 0
        self.cumulative_misses: int = 0
        self.total_scans: int = 0
        self.cumulative_transmissions: int = 0
        self.redundancy_count: int = 0
        self.total_hardware_penalty: float = 0.0
        self.total_intercept_time_error_us: float = 0.0
        self.intercept_count: int = 0
        self.cumulative_reward: float = 0.0

        # Tier-Specific Metrics Trackers
        self.tier1_hits: int = 0
        self.tier1_transmissions: int = 0
        self.tier1_missed_events: int = 0

        self.tier2_hits: int = 0
        self.tier2_transmissions: int = 0

        self.tier3_hits: int = 0
        self.tier3_transmissions: int = 0

    @staticmethod
    def get_band_tier(band: int) -> int:
        """Categorizes band into Tier 1 (Critical), Tier 2 (High), or Tier 3 (Routine)."""
        if band >= 16:
            return 1
        elif band >= 10:
            return 2
        else:
            return 3

    def get_hit_reward(self, band: int) -> float:
        """Returns weighted hit reward based on the threat tier."""
        tier = self.get_band_tier(band)
        if tier == 1:
            return self.tier1_reward   # +50.0 (Missile Lock)
        elif tier == 2:
            return self.tier2_reward   # +20.0 (Agile Interceptor)
        else:
            return self.tier3_reward   # +5.0 (Routine Surveillance)

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
        tier1_miss_penalty_applied: float = 0.0,
        step_intercept_error: float = 0.0,
    ) -> dict[str, Any]:
        """Compile running metrics and tier-specific Pd into info dictionary."""
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
        pd_tier1 = (
            float(self.tier1_hits / self.tier1_transmissions)
            if self.tier1_transmissions > 0
            else 0.0
        )
        pd_tier2 = (
            float(self.tier2_hits / self.tier2_transmissions)
            if self.tier2_transmissions > 0
            else 0.0
        )
        pd_tier3 = (
            float(self.tier3_hits / self.tier3_transmissions)
            if self.tier3_transmissions > 0
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
            "tier1_miss_penalty": tier1_miss_penalty_applied,
            "pd": pd,
            "pfa": pfa,
            "Pd_Tier1": pd_tier1,
            "Pd_Tier2": pd_tier2,
            "Pd_Tier3": pd_tier3,
            "tier1_hits": self.tier1_hits,
            "tier1_transmissions": self.tier1_transmissions,
            "tier2_hits": self.tier2_hits,
            "tier2_transmissions": self.tier2_transmissions,
            "tier3_hits": self.tier3_hits,
            "tier3_transmissions": self.tier3_transmissions,
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
        """Reset the threat-aware environment to step 0."""
        super().reset(seed=seed)

        self.current_step = 0
        self.current_rx1_band = 14
        self.current_rx2_band = 16
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

        self.tier1_hits = 0
        self.tier1_transmissions = 0
        self.tier1_missed_events = 0

        self.tier2_hits = 0
        self.tier2_transmissions = 0

        self.tier3_hits = 0
        self.tier3_transmissions = 0

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
        Execute one threat-prioritized cooperative dual-receiver scan step.
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

        # 2. Threat-Weighted Cooperative Reward Calculation
        base_reward = 0.0
        step_hits = 0
        step_misses = 0

        if is_redundant:
            base_reward -= self.redundancy_penalty  # -20.0
            self.redundancy_count += 1

            if hit1:
                # Overlap on active band: credit hit reward only once
                r_hit = self.get_hit_reward(rx1_band)
                base_reward += r_hit
                step_hits = 1
                step_misses = 0

                tier = self.get_band_tier(rx1_band)
                if tier == 1:
                    self.tier1_hits += 1
                elif tier == 2:
                    self.tier2_hits += 1
                else:
                    self.tier3_hits += 1
            else:
                base_reward -= (2.0 * self.miss_penalty)
                step_hits = 0
                step_misses = 2
        else:
            # Independent distinct channels
            tier1 = self.get_band_tier(rx1_band)
            tier2 = self.get_band_tier(rx2_band)
            is_tier1_guarded = (tier1 == 1 or tier2 == 1)

            # Cooperative Sector Deconfliction:
            # If both receivers crowd Tier 2 simultaneously (redundant coverage),
            # the secondary receiver is demoted to routine Tier 3 reward rate (+5.0)
            # because Tier 2 agile threats are already guarded by the primary receiver.
            r1_val = self.get_hit_reward(rx1_band)
            r2_val = self.get_hit_reward(rx2_band)
            if tier1 == 2 and tier2 == 2:
                r2_val = self.tier3_reward

            # Threat Priority Enforcement:
            # If neither receiver is monitoring Tier 1 (critical missile lock),
            # routine Tier 3 hits are not credited (0.0 reward) and an unguarded
            # exposure penalty is applied. The agent cannot abandon lethal defense
            # to harvest routine early warning pulses.
            if not is_tier1_guarded:
                if tier1 == 3:
                    r1_val = 0.0
                if tier2 == 3:
                    r2_val = 0.0
                base_reward -= 5.0  # Unguarded lethal exposure penalty

            if hit1:
                base_reward += r1_val
                step_hits += 1
                if tier1 == 1:
                    self.tier1_hits += 1
                elif tier1 == 2:
                    self.tier2_hits += 1
                else:
                    self.tier3_hits += 1
            else:
                # Tier 1 missile vigilance is defensive duty; waive routine miss penalty on quiet lethal channels
                if tier1 != 1:
                    base_reward -= self.miss_penalty
                step_misses += 1

            if hit2:
                base_reward += r2_val
                step_hits += 1
                if tier2 == 1:
                    self.tier1_hits += 1
                elif tier2 == 2:
                    self.tier2_hits += 1
                else:
                    self.tier3_hits += 1
            else:
                if tier2 != 1:
                    base_reward -= self.miss_penalty
                step_misses += 1

        # 3. Lethal Threat Miss Penalty:
        # If an active transmission occurred in Tier 1 (Bands 16-19) and was NOT intercepted by either receiver:
        tier1_miss_penalty_applied = 0.0
        for b in TIER1_BANDS:
            if self.truth_matrix[self.current_step, b] == 1:
                intercepted = (rx1_band == b and hit1) or (rx2_band == b and hit2)
                if not intercepted:
                    base_reward -= self.tier1_miss_penalty  # -20.0 per missed lethal missile lock
                    tier1_miss_penalty_applied += self.tier1_miss_penalty
                    self.tier1_missed_events += 1

        # 4. Hardware Switching Penalty
        dist1 = abs(rx1_band - self.current_rx1_band)
        dist2 = abs(rx2_band - self.current_rx2_band)
        hardware_penalty = (
            self.hardware_penalty_coeff * dist1 + self.hardware_penalty_coeff * dist2
        )
        self.total_hardware_penalty += hardware_penalty

        reward = float(base_reward - hardware_penalty)
        self.cumulative_reward += reward

        # 5. Receiver State Update
        self.current_rx1_band = rx1_band
        self.current_rx2_band = rx2_band
        self.rx1_history.append(1.0 if hit1 else 0.0)
        self.rx2_history.append(1.0 if hit2 else 0.0)

        # 6. Activity Density & Transition Predictor
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

        # 7. Metrics Tracking
        self.total_scans += 2
        active_step_transmissions = int(self.truth_matrix[self.current_step].sum())
        self.cumulative_transmissions += active_step_transmissions
        self.cumulative_hits += step_hits
        self.cumulative_misses += step_misses

        # Track tier-specific ground truth transmissions
        t1_trans = int(self.truth_matrix[self.current_step, 16:20].sum())
        t2_trans = int(self.truth_matrix[self.current_step, 10:16].sum())
        t3_trans = int(self.truth_matrix[self.current_step, 0:10].sum())
        self.tier1_transmissions += t1_trans
        self.tier2_transmissions += t2_trans
        self.tier3_transmissions += t3_trans

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
            tier1_miss_penalty_applied=tier1_miss_penalty_applied,
            step_intercept_error=step_intercept_error,
        )

        return obs, reward, terminated, truncated, info


def make_default_threat_env(
    max_rows: int = DEFAULT_MAX_ROWS,
    num_bands: int = DEFAULT_NUM_BANDS,
    time_step_us: float = DEFAULT_TIME_STEP_US,
    **env_kwargs: Any,
) -> ThreatAwareDualEnv:
    """Helper to instantiate ThreatAwareDualEnv using local radar dataset."""
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
    return ThreatAwareDualEnv(truth_matrix=truth_matrix, metadata=metadata, **env_kwargs)
