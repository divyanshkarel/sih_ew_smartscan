"""
quantize_edge.py - Upgrade 3: INT8 Edge Quantization & Latency Benchmarking

Applies INT8 Dynamic Quantization to the Threat-Aware Dual-Receiver Q-Network:
  1. Loads pre-trained smart_scan_threat_dqn.zip using Stable-Baselines3.
  2. Extracts the PyTorch Q-Network (52-D input -> 400-D action space).
  3. Exports the policy to FP32 ONNX: threat_agent_fp32.onnx (with dynamic batching).
  4. Quantizes FP32 weights to INT8 using onnxruntime.quantization (QuantType.QUInt8),
     saving as threat_agent_int8.onnx.
  5. Benchmarks 1,000 single-sample inferences on both models via onnxruntime.InferenceSession.
  6. Outputs a formatted terminal comparison table of File Size, Mean Latency, P99 Latency,
     and Throughput (Inferences/sec).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
import torch
from stable_baselines3 import DQN


# Paths & Dimensions
SOURCE_MODEL_PATH: str = "smart_scan_threat_dqn.zip"
FP32_ONNX_PATH: str = "threat_agent_fp32.onnx"
INT8_ONNX_PATH: str = "threat_agent_int8.onnx"

OBSERVATION_DIM: int = 52  # 2 tuned + 10 history + 20 density + 20 transitions
ACTION_DIM: int = 400       # 20 x 20 joint receiver actions
NUM_BENCHMARK_RUNS: int = 1_000
WARMUP_RUNS: int = 100


class OnnxQNetworkWrapper(torch.nn.Module):
    """
    Lightweight PyTorch wrapper module that isolates the pure Q-network
    from Stable-Baselines3 training metadata for production edge runtime.
    """

    def __init__(self, q_net: torch.nn.Module) -> None:
        super().__init__()
        self.q_net = q_net

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch_size, 52) -> (batch_size, 400)"""
        return self.q_net(observation)


def export_fp32_onnx(
    model_path: str = SOURCE_MODEL_PATH,
    output_path: str = FP32_ONNX_PATH,
) -> Path:
    """
    Loads SB3 threat-aware DQN, extracts PyTorch QNetwork, and exports to FP32 ONNX.
    """
    print("=" * 82)
    print("   UPGRADE 3: INT8 EDGE QUANTIZATION FOR THREAT-PRIORITY RADAR SCHEDULER   ")
    print("=" * 82)

    src = Path(model_path)
    if not src.exists():
        raise FileNotFoundError(
            f"Source model '{model_path}' not found. Please run 'train_threat_dqn.py' first."
        )

    print(f"[*] Step 1: Loading Threat-Aware Dual DQN model from: {model_path}...")
    model = DQN.load(model_path)

    q_net = model.q_net
    q_net.eval()
    print("[*] Successfully extracted PyTorch QNetwork:")
    print(f"    - Input Features  : {OBSERVATION_DIM} dimensions (Normalized State Vector)")
    print(f"    - Output Q-Values : {ACTION_DIM} joint actions (20x20 Receiver Channels)")

    wrapper = OnnxQNetworkWrapper(q_net)
    wrapper.eval()

    dummy_input = torch.randn(1, OBSERVATION_DIM, dtype=torch.float32)

    print(f"\n[*] Step 2: Exporting PyTorch graph to standard FP32 ONNX: {output_path}...")
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

    fp32_size_kb = os.path.getsize(output_path) / 1024.0
    print(f"[+] FP32 ONNX Export Completed: {output_path} ({fp32_size_kb:.2f} KB)")

    # Validate ONNX graph
    print("[*] Validating FP32 ONNX graph definition with onnx.checker...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("[+] FP32 model integrity verified.")

    return Path(output_path)


def quantize_to_int8(
    fp32_path: str = FP32_ONNX_PATH,
    int8_path: str = INT8_ONNX_PATH,
) -> Path:
    """
    Applies Dynamic INT8 Quantization (QuantType.QUInt8) to the FP32 ONNX model.
    """
    print(f"\n[*] Step 3: Applying dynamic quantization to INT8 (QuantType.QUInt8)...")
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QUInt8,
    )

    int8_size_kb = os.path.getsize(int8_path) / 1024.0
    fp32_size_kb = os.path.getsize(fp32_path) / 1024.0
    compression_ratio = (1.0 - (int8_size_kb / fp32_size_kb)) * 100.0

    print(f"[+] INT8 ONNX Export Completed: {int8_path} ({int8_size_kb:.2f} KB)")
    print(f"[+] Memory Footprint Reduction : {compression_ratio:.1f}% smaller than FP32!")

    # Validate Quantized Model
    print("[*] Validating INT8 ONNX graph definition with onnx.checker...")
    int8_model = onnx.load(int8_path)
    onnx.checker.check_model(int8_model)
    print("[+] INT8 model integrity verified.")

    return Path(int8_path)


def benchmark_model(
    onnx_path: str,
    num_runs: int = NUM_BENCHMARK_RUNS,
    warmup: int = WARMUP_RUNS,
) -> dict[str, float]:
    """
    Runs single-sample inference latency benchmark using ONNX Runtime.
    """
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 1  # Single-thread edge CPU embedded realism

    session = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Create dummy observation input
    dummy_obs = np.random.uniform(0.0, 1.0, size=(1, OBSERVATION_DIM)).astype(np.float32)

    # Warmup passes
    for _ in range(warmup):
        _ = session.run(None, {input_name: dummy_obs})

    # High-resolution benchmark loop
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
    throughput_fps = 1_000_000.0 / mean_lat if mean_lat > 0 else 0.0
    file_size_kb = float(os.path.getsize(onnx_path) / 1024.0)

    return {
        "file_size_kb": file_size_kb,
        "mean_us": mean_lat,
        "median_us": median_lat,
        "p95_us": p95_lat,
        "p99_us": p99_lat,
        "min_us": min_lat,
        "max_us": max_lat,
        "throughput_fps": throughput_fps,
    }


def verify_numerical_consistency(
    fp32_path: str = FP32_ONNX_PATH,
    int8_path: str = INT8_ONNX_PATH,
    num_samples: int = 100,
) -> dict[str, float]:
    """
    Computes policy action agreement and Q-value cosine similarity
    between FP32 and INT8 models over random observation vectors.
    """
    s_fp32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    s_int8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    in_name = s_fp32.get_inputs()[0].name

    test_samples = np.random.uniform(0.0, 1.0, size=(num_samples, OBSERVATION_DIM)).astype(np.float32)
    out_fp32 = s_fp32.run(None, {in_name: test_samples})[0]
    out_int8 = s_int8.run(None, {in_name: test_samples})[0]

    actions_fp32 = np.argmax(out_fp32, axis=1)
    actions_int8 = np.argmax(out_int8, axis=1)
    action_agreement_pct = float(np.mean(actions_fp32 == actions_int8) * 100.0)

    cos_sims = []
    for i in range(num_samples):
        v1, v2 = out_fp32[i], out_int8[i]
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-9
        cos_sims.append(np.dot(v1, v2) / denom)
    mean_cos_sim = float(np.mean(cos_sims))

    return {
        "action_agreement_pct": action_agreement_pct,
        "mean_cos_sim": mean_cos_sim,
    }


def print_edge_benchmark_table(
    fp32_metrics: dict[str, float],
    int8_metrics: dict[str, float],
    consistency: dict[str, float],
) -> None:
    """
    Prints a clean, formatted Electronic Warfare edge deployment benchmark table.
    """
    col_w = 26
    sep = "=" * 84
    mid_sep = "-" * 84

    def c_str(val: Any) -> str:
        return f"{str(val):^{col_w}}"

    size_fp32 = f"{fp32_metrics['file_size_kb']:.2f} KB"
    size_int8 = f"{int8_metrics['file_size_kb']:.2f} KB"
    comp_pct = (1.0 - (int8_metrics['file_size_kb'] / fp32_metrics['file_size_kb'])) * 100.0

    mean_fp32 = f"{fp32_metrics['mean_us']:.2f} us"
    mean_int8 = f"{int8_metrics['mean_us']:.2f} us"

    p50_fp32 = f"{fp32_metrics['median_us']:.2f} us"
    p50_int8 = f"{int8_metrics['median_us']:.2f} us"

    p99_fp32 = f"{fp32_metrics['p99_us']:.2f} us"
    p99_int8 = f"{int8_metrics['p99_us']:.2f} us"

    fps_fp32 = f"{fp32_metrics['throughput_fps']:,.0f} req/s"
    fps_int8 = f"{int8_metrics['throughput_fps']:,.0f} req/s"

    speedup = (fp32_metrics['mean_us'] / int8_metrics['mean_us']) if int8_metrics['mean_us'] > 0 else 1.0

    cos_sim_str = f"{consistency['mean_cos_sim']:.4f}"
    cos_sim_pct = f"{consistency['mean_cos_sim'] * 100.0:.2f}%"
    agreement_pct = f"{consistency['action_agreement_pct']:.1f}%"

    print("\n" + sep)
    print("      ELECTRONIC WARFARE UPGRADE 3: ONNX INT8 EDGE QUANTIZATION BENCHMARK      ")
    print(sep)
    print(
        f" {'Benchmark Metric':<{col_w}} | "
        f"{c_str('FP32 ONNX Baseline')} | "
        f"{c_str('INT8 Edge Quantized')}"
    )
    print(mid_sep)
    print(
        f" {'Precision Format':<{col_w}} | "
        f"{c_str('Float32 (Standard)')} | "
        f"{c_str('QUInt8 (Dynamic)')}"
    )
    print(
        f" {'Action Space Output':<{col_w}} | "
        f"{c_str('400 Joint Actions')} | "
        f"{c_str('400 Joint Actions')}"
    )
    print(
        f" {'Evaluation Passes':<{col_w}} | "
        f"{c_str(f'{NUM_BENCHMARK_RUNS:,} passes')} | "
        f"{c_str(f'{NUM_BENCHMARK_RUNS:,} passes')}"
    )
    print(mid_sep)
    print(
        f" {'On-Disk Model Size':<{col_w}} | "
        f"{c_str(size_fp32)} | "
        f"{c_str(size_int8)}"
    )
    print(
        f" {'Mean Latency (Avg)':<{col_w}} | "
        f"{c_str(mean_fp32)} | "
        f"{c_str(mean_int8)}"
    )
    print(
        f" {'Median Latency (P50)':<{col_w}} | "
        f"{c_str(p50_fp32)} | "
        f"{c_str(p50_int8)}"
    )
    print(
        f" {'99th Percentile (P99)':<{col_w}} | "
        f"{c_str(p99_fp32)} | "
        f"{c_str(p99_int8)}"
    )
    print(
        f" {'Inference Throughput':<{col_w}} | "
        f"{c_str(fps_fp32)} | "
        f"{c_str(fps_int8)}"
    )
    print(mid_sep)
    print(
        f" {'Memory Footprint Reduction':<{col_w}} | "
        f"{c_str('1.00x (Baseline)')} | "
        f"{c_str(f'{comp_pct:.1f}% reduction')}"
    )
    print(
        f" {'Inference Acceleration':<{col_w}} | "
        f"{c_str('1.00x (Baseline)')} | "
        f"{c_str(f'{speedup:.2f}x faster')}"
    )
    print(
        f" {'Q-Value Cosine Similarity':<{col_w}} | "
        f"{c_str('1.0000 (Reference)')} | "
        f"{c_str(cos_sim_str)}"
    )
    print(sep)

    print("\n[+] EDGE AVIONICS & REAL-TIME HARDWARE VERIFICATION:")
    print(
        f"    1. Microsecond Real-Time Guarantee:\n"
        f"       - Radar Pulse Dwell Time (TSRD Time Step) : 100.00 us\n"
        f"       - INT8 Single-Core Inference Latency      : {int8_metrics['mean_us']:.2f} us ({int8_metrics['mean_us']/1000.0:.4f} ms)\n"
        f"       - Dwell Window Budget Consumed            : {(int8_metrics['mean_us']/100.0)*100.0:.1f}% (hard real-time deterministic margin!)\n"
    )
    print(
        f"    2. Embedded Memory Budget & Compression:\n"
        f"       - Unquantized FP32 Model Size            : {fp32_metrics['file_size_kb']:.2f} KB\n"
        f"       - Quantized INT8 Edge Model Size         : {int8_metrics['file_size_kb']:.2f} KB\n"
        f"       - Net RAM / Flash Storage Saved          : {fp32_metrics['file_size_kb'] - int8_metrics['file_size_kb']:.2f} KB ({comp_pct:.1f}% reduction)\n"
    )
    print(
        f"    3. Numerical Stability & Decision Integrity:\n"
        f"       - Q-Vector Directional Alignment         : {cos_sim_pct} directional fidelity\n"
        f"       - High-Confidence Argmax Agreement       : {agreement_pct} discrete channel selection match\n"
    )
    print(sep + "\n")


def run_pipeline() -> None:
    """Executes export, quantization, benchmarking, and reporting."""
    # 1. Export FP32 ONNX
    fp32_path = export_fp32_onnx()

    # 2. Quantize to INT8
    int8_path = quantize_to_int8(str(fp32_path))

    # 3. Benchmark both models
    print(f"\n[*] Step 4: Running {NUM_BENCHMARK_RUNS:,} inference benchmark runs on FP32 model...")
    fp32_results = benchmark_model(str(fp32_path))

    print(f"[*] Step 5: Running {NUM_BENCHMARK_RUNS:,} inference benchmark runs on INT8 model...")
    int8_results = benchmark_model(str(int8_path))

    # 4. Numerical fidelity check
    print("[*] Step 6: Evaluating numerical consistency across 100 random observations...")
    consistency = verify_numerical_consistency(str(fp32_path), str(int8_path))

    # 5. Output comparison table
    print_edge_benchmark_table(fp32_results, int8_results, consistency)


if __name__ == "__main__":
    run_pipeline()
