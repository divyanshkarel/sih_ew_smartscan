"""
export_onnx.py - Phase 3: Export PyTorch Q-Network to ONNX & Measure Edge Latency

Extracts the trained Q-Network from smart_scan_dqn.zip, exports it to an optimized
smart_scan_edge.onnx model, validates graph integrity, and benchmarks inference latency
in microseconds using ONNX Runtime.
"""

from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import onnx
import onnxruntime as ort
import torch
from stable_baselines3 import DQN


INPUT_MODEL_PATH: str = "smart_scan_dqn.zip"
OUTPUT_ONNX_PATH: str = "smart_scan_edge.onnx"
OBSERVATION_DIM: int = 46
ACTION_DIM: int = 20
NUM_BENCHMARK_RUNS: int = 1_000


class OnnxQNetworkWrapper(torch.nn.Module):
    """
    Lightweight wrapper module ensuring clean observation-to-Q-values mapping
    without any Stable-Baselines3 training-specific metadata.
    """

    def __init__(self, q_net: torch.nn.Module):
        super().__init__()
        self.q_net = q_net

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch_size, 46) -> (batch_size, 20)"""
        return self.q_net(observation)


def export_dqn_to_onnx(
    model_path: str = INPUT_MODEL_PATH,
    output_path: str = OUTPUT_ONNX_PATH,
) -> Path:
    """
    Loads SB3 DQN, extracts the Q-Network, and exports to ONNX format.
    """
    print("=" * 75)
    print("      PHASE 3: ONNX EDGE EXPORT & LATENCY BENCHMARKING      ")
    print("=" * 75)

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Trained model '{model_path}' not found. Run train_dqn.py first.")

    print(f"[*] Loading trained Stable-Baselines3 model from: {model_path}...")
    model = DQN.load(model_path)

    # 1. Extract PyTorch Q-Network
    q_net = model.q_net
    q_net.eval()
    print("[*] Successfully extracted PyTorch QNetwork:")
    print(f"    - Input Features:  {OBSERVATION_DIM} dimensions")
    print(f"    - Output Q-Values: {ACTION_DIM} frequency bands")

    wrapper = OnnxQNetworkWrapper(q_net)
    wrapper.eval()

    # 2. Create Dummy Observation Tensor matching observation space
    dummy_input = torch.randn(1, OBSERVATION_DIM, dtype=torch.float32)

    # 3. Export to ONNX
    print(f"[*] Exporting graph to: {output_path}...")
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["q_values"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "q_values": {0: "batch_size"},
        },
        dynamo=False,
    )
    print(f"[+] Export completed successfully: {output_path} ({os.path.getsize(output_path) / 1024:.2f} KB)")

    # 4. Verify Model Integrity
    print("[*] Validating ONNX graph definition with onnx.checker...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("[+] ONNX model integrity check passed.")

    return Path(output_path)


def benchmark_onnx_latency(
    onnx_path: str = OUTPUT_ONNX_PATH,
    num_runs: int = NUM_BENCHMARK_RUNS,
) -> dict[str, float]:
    """
    Benchmarks single-sample edge inference latency using ONNX Runtime.
    """
    print("\n" + "=" * 75)
    print(f"   BENCHMARKING EDGE INFERENCE LATENCY ({num_runs:,} EVALUATION PASSES)   ")
    print("=" * 75)

    # Initialize ONNX Runtime Inference Session
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 1  # Single-thread test for edge embedded realism

    session = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print(f"[*] Inference Provider: {session.get_providers()[0]}")
    print(f"[*] Input Node:  '{input_name}'  {session.get_inputs()[0].shape}")
    print(f"[*] Output Node: '{output_name}' {session.get_outputs()[0].shape}")

    # Generate synthetic input observation vector
    dummy_obs = np.random.uniform(0.0, 1.0, size=(1, OBSERVATION_DIM)).astype(np.float32)

    # Warmup passes
    for _ in range(50):
        _ = session.run(None, {input_name: dummy_obs})

    # Benchmark loop measuring latency via high-resolution counter
    latencies_us = np.zeros(num_runs, dtype=np.float64)
    for i in range(num_runs):
        t0 = time.perf_counter_ns()
        _ = session.run(None, {input_name: dummy_obs})
        t1 = time.perf_counter_ns()
        latencies_us[i] = (t1 - t0) / 1000.0  # nanoseconds to microseconds

    mean_lat = float(np.mean(latencies_us))
    median_lat = float(np.median(latencies_us))
    p95_lat = float(np.percentile(latencies_us, 95))
    p99_lat = float(np.percentile(latencies_us, 99))
    min_lat = float(np.min(latencies_us))
    max_lat = float(np.max(latencies_us))
    throughput_fps = 1_000_000.0 / mean_lat

    print("\n" + "-" * 75)
    print("                  EDGE INFERENCE LATENCY METRICS                  ")
    print("-" * 75)
    print(f" Mean Latency          : {mean_lat:6.2f} us   ({mean_lat / 1000:.4f} ms)")
    print(f" Median Latency (P50)  : {median_lat:6.2f} us")
    print(f" 95th Percentile (P95) : {p95_lat:6.2f} us")
    print(f" 99th Percentile (P99) : {p99_lat:6.2f} us")
    print(f" Minimum Latency       : {min_lat:6.2f} us")
    print(f" Maximum Latency       : {max_lat:6.2f} us")
    print(f" Inferences Per Second : {throughput_fps:,.0f} req/s")
    print("-" * 75)

    if mean_lat < 1000.0:
        print("[+] VERIFICATION: Microsecond latency confirmed for edge receiver deployment!")
    else:
        print("[!] Note: Latency exceeded 1 ms threshold.")

    print("=" * 75 + "\n")

    return {
        "mean_us": mean_lat,
        "median_us": median_lat,
        "p95_us": p95_lat,
        "p99_us": p99_lat,
        "min_us": min_lat,
        "max_us": max_lat,
        "throughput_fps": throughput_fps,
    }


if __name__ == "__main__":
    exported_file = export_dqn_to_onnx()
    benchmark_onnx_latency(str(exported_file))
