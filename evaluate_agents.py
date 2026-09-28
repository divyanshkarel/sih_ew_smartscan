"""
evaluate_agents.py - Phase 2: Comparative Evaluation of Open-Loop Baseline vs. Trained DQN

Loads the trained smart_scan_dqn.zip model, executes both the Open-Loop Linear Sweep
Baseline and the Agile DQN Agent over the identical 6,823 time steps in SmartScanEnv,
and outputs a comprehensive side-by-side comparative metrics table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import DQN

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


MODEL_PATH: str = "smart_scan_dqn.zip"


def run_evaluation(
    env: SmartScanEnv,
    is_dqn: bool = False,
    model: DQN | None = None,
) -> dict[str, Any]:
    """
    Executes an evaluation episode over the environment.

    If is_dqn is True, wraps the env in AgileEmitterPredictorWrapper and uses model.predict.
    If is_dqn is False, runs the open-loop linear sweep baseline (action = step % N).
    """
    if is_dqn:
        wrapped_env = AgileEmitterPredictorWrapper(env)
        obs, info = wrapped_env.reset()
    else:
        obs, info = env.reset()

    num_bands = env.N
    total_steps = env.T

    total_hardware_penalty = 0.0

    for step_idx in range(total_steps):
        if is_dqn:
            assert model is not None, "DQN model must be provided for DQN evaluation"
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = wrapped_env.step(int(action))
        else:
            action = step_idx % num_bands
            obs, reward, terminated, truncated, info = env.step(action)

        total_hardware_penalty += info.get("hardware_penalty", 0.0)

        if terminated or truncated:
            break

    # Consolidate results
    results = {
        "total_scans": info["total_scans"],
        "cumulative_hits": info["cumulative_hits"],
        "cumulative_misses": info["cumulative_misses"],
        "cumulative_transmissions": info["cumulative_transmissions"],
        "cumulative_reward": info["cumulative_reward"],
        "total_hardware_penalty": total_hardware_penalty,
        "pd": info["pd"],
        "pfa": info["pfa"],
        "intercept_time_error_us": info["intercept_time_error_us"],
    }
    return results


def print_comparison_table(baseline: dict[str, Any], dqn: dict[str, Any]) -> None:
    """Formats and prints a clean, side-by-side comparative terminal table."""
    pd_diff = (dqn["pd"] - baseline["pd"]) * 100
    pfa_diff = (dqn["pfa"] - baseline["pfa"]) * 100
    hits_diff = dqn["cumulative_hits"] - baseline["cumulative_hits"]
    reward_diff = dqn["cumulative_reward"] - baseline["cumulative_reward"]
    reward_ratio = (
        (dqn["cumulative_reward"] / baseline["cumulative_reward"])
        if baseline["cumulative_reward"] != 0
        else 1.0
    )

    col_w = 26
    sep = "=" * 82
    mid_sep = "-" * 82

    print("\n" + sep)
    print("      ELECTRONIC WARFARE SMART SCANNING - AGENT BENCHMARK COMPARISON      ")
    print(sep)
    header = f"{'Metric':<32} | {'Open-Loop Baseline':<{col_w}} | {'Agile DQN Agent':<{col_w}}"
    print(header)
    print(mid_sep)

    rows = [
        (
            "Total Receiver Scans",
            f"{baseline['total_scans']:,}",
            f"{dqn['total_scans']:,}",
        ),
        (
            "Total Ground Truth Transmissions",
            f"{baseline['cumulative_transmissions']:,}",
            f"{dqn['cumulative_transmissions']:,}",
        ),
        (
            "Successful Intercepts (Hits)",
            f"{baseline['cumulative_hits']:,}",
            f"{dqn['cumulative_hits']:,}  (+{hits_diff:,})",
        ),
        (
            "Empty Channel Scans (Misses)",
            f"{baseline['cumulative_misses']:,}",
            f"{dqn['cumulative_misses']:,}",
        ),
        (
            "Probability of Detection (Pd)",
            f"{baseline['pd'] * 100:.2f}%",
            f"{dqn['pd'] * 100:.2f}%  ({'+' if pd_diff >= 0 else ''}{pd_diff:.2f}%)",
        ),
        (
            "Probability of False Alarm (Pfa)",
            f"{baseline['pfa'] * 100:.2f}%",
            f"{dqn['pfa'] * 100:.2f}%  ({'+' if pfa_diff >= 0 else ''}{pfa_diff:.2f}%)",
        ),
        (
            "Mean Intercept Time Error",
            f"{baseline['intercept_time_error_us']:.2f} us",
            f"{dqn['intercept_time_error_us']:.2f} us",
        ),
        (
            "Total Hardware Penalty Incurred",
            f"-{baseline['total_hardware_penalty']:,.1f}",
            f"-{dqn['total_hardware_penalty']:,.1f}",
        ),
        (
            "Total Cumulative Episode Reward",
            f"{baseline['cumulative_reward']:,.2f}",
            f"{dqn['cumulative_reward']:,.2f}  ({reward_ratio:.2f}x)",
        ),
    ]

    for label, base_val, dqn_val in rows:
        print(f"{label:<32} | {base_val:<{col_w}} | {dqn_val:<{col_w}}")

    print(sep)

    # Verification checks
    print("\n" + "=" * 82)
    print("                         QUANTITATIVE VERIFICATION                         ")
    print("=" * 82)
    pd_passed = dqn["pd"] > baseline["pd"]
    pfa_passed = dqn["pfa"] < baseline["pfa"]
    reward_passed = dqn["cumulative_reward"] > baseline["cumulative_reward"]

    print(f"[*] Pd Superiority Check    : {'PASSED' if pd_passed else 'FAILED'} (DQN: {dqn['pd']*100:.2f}% vs Baseline: {baseline['pd']*100:.2f}%)")
    print(f"[*] Pfa Reduction Check     : {'PASSED' if pfa_passed else 'FAILED'} (DQN: {dqn['pfa']*100:.2f}% vs Baseline: {baseline['pfa']*100:.2f}%)")
    print(f"[*] Reward Efficiency Check : {'PASSED' if reward_passed else 'FAILED'} (DQN: {dqn['cumulative_reward']:,.1f} vs Baseline: {baseline['cumulative_reward']:,.1f})")
    print("=" * 82 + "\n")


def main():
    print("=" * 82)
    print("     PHASE 2: EVALUATING AGILE DQN AGENT VS. OPEN-LOOP BASELINE     ")
    print("=" * 82)

    # 1. Load Data
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
        f"[*] Truth Matrix loaded: shape {truth_matrix.shape} "
        f"({truth_matrix.shape[0]} time steps x {truth_matrix.shape[1]} bands)"
    )

    # 2. Check for Trained Model
    model_file = Path(MODEL_PATH)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{MODEL_PATH}'. Please run 'train_dqn.py' first."
        )

    print(f"[*] Loading trained DQN agent from: {MODEL_PATH}...")
    dqn_model = DQN.load(str(model_file))

    # 3. Evaluate Baseline Agent
    print("[*] Running Open-Loop Linear Sweep Baseline Agent...")
    env_baseline = SmartScanEnv(
        truth_matrix=truth_matrix,
        metadata=metadata,
        hardware_penalty_coeff=0.2,
    )
    baseline_metrics = run_evaluation(env_baseline, is_dqn=False)

    # 4. Evaluate Trained DQN Agent
    print("[*] Running Trained Agile DQN Agent...")
    env_dqn = SmartScanEnv(
        truth_matrix=truth_matrix,
        metadata=metadata,
        hardware_penalty_coeff=0.2,
    )
    dqn_metrics = run_evaluation(env_dqn, is_dqn=True, model=dqn_model)

    # 5. Display Comparison
    print_comparison_table(baseline_metrics, dqn_metrics)


if __name__ == "__main__":
    main()
