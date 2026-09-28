"""
api_backend.py - Phase 3: Ultra-Low-Latency Edge REST API for EW Smart Scanning

FastAPI backend serving the exported ONNX model with sub-millisecond edge inference.
Hosts endpoints to reset the simulation environment and execute receiver scan steps
using either conventional linear raster scanning or the trained Agile ML Agent.
"""

from __future__ import annotations

import io
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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


ONNX_MODEL_PATH: str = "smart_scan_edge.onnx"


class EWSimulationEngine:
    """
    Simulator engine managing dual synchronized environments for side-by-side
    comparison of Linear Raster Baseline vs. Agile ML Agent, alongside the
    ONNX Runtime inference session.
    """

    def __init__(self, onnx_path: str = ONNX_MODEL_PATH):
        self.onnx_path = Path(onnx_path)
        self.session: ort.InferenceSession | None = None
        self.input_name: str = ""
        self.output_name: str = ""
        self.linear_env: SmartScanEnv | None = None
        self.ml_env: AgileEmitterPredictorWrapper | None = None
        self.ml_obs: np.ndarray | None = None
        self.num_bands: int = DEFAULT_NUM_BANDS
        self.total_time_steps: int = 0
        self.is_initialized: bool = False

    def initialize(self) -> None:
        """Loads ONNX model and initializes radar simulation environments."""
        if not self.onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model '{self.onnx_path}' not found. Please run export_onnx.py first."
            )

        # 1. Initialize ONNX Runtime Session
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 1  # Low-power edge execution

        self.session = ort.InferenceSession(
            str(self.onnx_path), opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # 2. Ingest radar data & discretize Truth Matrix
        dataset_path = find_default_dataset_path()
        if dataset_path and dataset_path.exists():
            pdw_df = load_radar_data(dataset_path, max_rows=DEFAULT_MAX_ROWS)
        else:
            pdw_df = generate_synthetic_pdw_dataset(num_pulses=DEFAULT_MAX_ROWS)

        truth_matrix, metadata = discretize_pdw_stream(
            pdw_df,
            num_bands=DEFAULT_NUM_BANDS,
            time_step_us=DEFAULT_TIME_STEP_US,
        )

        # Instantiate separate environments for fair side-by-side execution
        base_linear = SmartScanEnv(
            truth_matrix=truth_matrix,
            metadata=metadata,
            hardware_penalty_coeff=0.2,
        )
        base_ml = SmartScanEnv(
            truth_matrix=truth_matrix,
            metadata=metadata,
            hardware_penalty_coeff=0.2,
        )

        self.linear_env = base_linear
        self.ml_env = AgileEmitterPredictorWrapper(base_ml)
        self.num_bands = base_linear.N
        self.total_time_steps = base_linear.T

        # 3. Initial Reset
        self.reset()
        self.is_initialized = True

    def reset(self) -> dict[str, Any]:
        """Resets both simulation tracks to initial step 0."""
        if self.linear_env is None or self.ml_env is None:
            raise RuntimeError("Simulation engine not initialized.")

        _, linear_info = self.linear_env.reset()
        self.ml_obs, ml_info = self.ml_env.reset()

        return {
            "status": "reset_successful",
            "step": 0,
            "total_steps": self.total_time_steps,
            "num_bands": self.num_bands,
            "observation_dim": int(self.ml_obs.shape[0]),
            "linear_metrics": {
                "pd": float(linear_info["pd"]),
                "pfa": float(linear_info["pfa"]),
                "cumulative_hits": int(linear_info["cumulative_hits"]),
                "cumulative_misses": int(linear_info["cumulative_misses"]),
                "total_scans": int(linear_info["total_scans"]),
                "cumulative_reward": float(linear_info["cumulative_reward"]),
                "intercept_time_error_us": float(linear_info["intercept_time_error_us"]),
            },
            "ml_metrics": {
                "pd": float(ml_info["pd"]),
                "pfa": float(ml_info["pfa"]),
                "cumulative_hits": int(ml_info["cumulative_hits"]),
                "cumulative_misses": int(ml_info["cumulative_misses"]),
                "total_scans": int(ml_info["total_scans"]),
                "cumulative_reward": float(ml_info["cumulative_reward"]),
                "intercept_time_error_us": float(ml_info["intercept_time_error_us"]),
            },
        }

    def step(self, strategy: str = "ml_agent") -> dict[str, Any]:
        """
        Executes one receiver scan step using either 'linear' or 'ml_agent'.
        """
        if self.linear_env is None or self.ml_env is None or self.session is None or self.ml_obs is None:
            raise RuntimeError("Simulation engine not initialized.")

        strat = strategy.lower()
        if strat == "linear":
            target_env = self.linear_env
            if target_env.current_step >= target_env.T:
                raise HTTPException(
                    status_code=400,
                    detail=f"Linear episode complete (step {target_env.current_step} >= {target_env.T}). Call /reset to restart.",
                )
            chosen_band = target_env.current_step % self.num_bands
            _, reward, terminated, truncated, info = target_env.step(chosen_band)
            inference_latency_us = 0.0

        elif strat == "ml_agent":
            target_env = self.ml_env
            unwrapped = target_env.unwrapped
            if unwrapped.current_step >= unwrapped.T:
                raise HTTPException(
                    status_code=400,
                    detail=f"ML Agent episode complete (step {unwrapped.current_step} >= {unwrapped.T}). Call /reset to restart.",
                )
            # Edge ONNX Inference
            t0 = time.perf_counter_ns()
            ort_input = {self.input_name: self.ml_obs[np.newaxis, :].astype(np.float32)}
            q_values = self.session.run([self.output_name], ort_input)[0]
            chosen_band = int(np.argmax(q_values, axis=1)[0])
            t1 = time.perf_counter_ns()
            inference_latency_us = (t1 - t0) / 1000.0

            next_obs, reward, terminated, truncated, info = target_env.step(chosen_band)
            self.ml_obs = next_obs
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid strategy '{strategy}'. Must be 'linear' or 'ml_agent'.",
            )

        return {
            "step": int(info["step"]),
            "strategy": strat,
            "chosen_band": chosen_band,
            "is_hit": bool(info["is_hit"]),
            "reward": float(reward),
            "base_reward": float(info["base_reward"]),
            "hardware_penalty": float(info["hardware_penalty"]),
            "jump_distance": int(info["jump_distance"]),
            "terminated": bool(terminated or truncated),
            "inference_latency_us": round(inference_latency_us, 2),
            "metrics": {
                "pd": float(info["pd"]),
                "pfa": float(info["pfa"]),
                "cumulative_hits": int(info["cumulative_hits"]),
                "cumulative_misses": int(info["cumulative_misses"]),
                "total_scans": int(info["total_scans"]),
                "cumulative_transmissions": int(info["cumulative_transmissions"]),
                "cumulative_reward": float(info["cumulative_reward"]),
                "intercept_time_error_us": float(info["intercept_time_error_us"]),
                "step_intercept_time_error_us": float(info["step_intercept_time_error_us"]),
            },
        }

    def step_both(self) -> dict[str, Any]:
        """Steps both linear and ml_agent models synchronously in one call."""
        linear_res = self.step("linear")
        ml_res = self.step("ml_agent")
        return {
            "step": ml_res["step"],
            "linear": linear_res,
            "ml_agent": ml_res,
        }



# Singleton engine instance
engine = EWSimulationEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize engine and load ONNX runtime
    engine.initialize()
    print("[+] EW Edge API initialized. ONNX runtime loaded and ready.")
    yield
    print("[-] Shutting down EW Edge API.")


app = FastAPI(
    title="Electronic Warfare Smart Scan Edge API",
    description="Ultra-low-latency ONNX-accelerated radar receiver scanning engine for agile spectrum dominance.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StepRequest(BaseModel):
    strategy: Literal["linear", "ml_agent"] = Field(
        default="ml_agent",
        description="Receiver scanning strategy: 'linear' for open-loop raster scan, or 'ml_agent' for ONNX DQN policy.",
    )


@app.get("/")
def get_status():
    """Health check and engine status."""
    return {
        "status": "online",
        "service": "Electronic Warfare Smart Scan Edge API",
        "engine_ready": engine.is_initialized,
        "onnx_model": str(engine.onnx_path),
        "total_time_steps": engine.total_time_steps,
        "num_bands": engine.num_bands,
    }


@app.get("/reset")
def reset_simulation():
    """
    Resets the RF simulation environment to initial step 0 and clears all metrics.
    """
    return engine.reset()


@app.post("/step")
def step_simulation(
    payload: StepRequest | None = None,
    strategy: Literal["linear", "ml_agent"] | None = Query(
        default=None,
        description="Optional query param: 'linear' or 'ml_agent'",
    ),
):
    """
    Executes a receiver tuning step.

    - **linear**: Steps the sequential raster sweep baseline.
    - **ml_agent**: Passes the current 46-D state to the ONNX model, selects argmax Q-value,
      and steps the environment with sub-millisecond edge latency.
    """
    strat = strategy or (payload.strategy if payload else "ml_agent")
    return engine.step(strategy=strat)


@app.post("/step_both")
def step_both_simulation():
    """
    Simultaneously steps both the Open-Loop Baseline and Agile ML Agent
    in exact synchronization over the ground truth radar timeline.
    """
    return engine.step_both()


@app.get("/metrics")
def get_current_metrics():
    """Returns the current cumulative EW metrics for both simulation tracks."""
    if engine.linear_env is None or engine.ml_env is None:
        raise HTTPException(status_code=500, detail="Engine not initialized.")
    lin_u = engine.linear_env
    ml_u = engine.ml_env.unwrapped

    return {
        "step": ml_u.current_step,
        "total_steps": ml_u.T,
        "pd": float(ml_u.cumulative_hits / max(ml_u.cumulative_transmissions, 1)),
        "pfa": float(ml_u.cumulative_misses / max(ml_u.total_scans, 1)),
        "cumulative_hits": ml_u.cumulative_hits,
        "cumulative_misses": ml_u.cumulative_misses,
        "total_scans": ml_u.total_scans,
        "cumulative_reward": float(ml_u.cumulative_reward),
        "linear": {
            "current_band": lin_u.current_band,
            "pd": float(lin_u.cumulative_hits / max(lin_u.cumulative_transmissions, 1)),
            "pfa": float(lin_u.cumulative_misses / max(lin_u.total_scans, 1)),
            "cumulative_hits": lin_u.cumulative_hits,
            "cumulative_misses": lin_u.cumulative_misses,
            "total_scans": lin_u.total_scans,
            "cumulative_reward": float(lin_u.cumulative_reward),
        },
        "ml_agent": {
            "current_band": ml_u.current_band,
            "pd": float(ml_u.cumulative_hits / max(ml_u.cumulative_transmissions, 1)),
            "pfa": float(ml_u.cumulative_misses / max(ml_u.total_scans, 1)),
            "cumulative_hits": ml_u.cumulative_hits,
            "cumulative_misses": ml_u.cumulative_misses,
            "total_scans": ml_u.total_scans,
            "cumulative_reward": float(ml_u.cumulative_reward),
        },
    }


if __name__ == "__main__":
    print("=" * 75)
    print("   LAUNCHING ELECTRONIC WARFARE SMART SCAN EDGE API (UVICORN)   ")
    print("=" * 75)
    uvicorn.run("api_backend:app", host="127.0.0.1", port=8005, reload=False)
