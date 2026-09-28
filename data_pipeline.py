"""
data_pipeline.py - Electronic Warfare Radar Pulse Ingestion & Discretization Pipeline

Phase 1 EW Reinforcement Learning Simulation.
Provides memory-safe streaming of radar Pulse Descriptor Words (PDWs) using Polars
and PyArrow (zero pandas dependencies). Extracts Time of Arrival (ToA) and Centre Frequency,
discretizes continuous pulses into time steps and frequency bands, and outputs a 2D Truth Matrix.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any, Generator, Tuple

import h5py
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.dataset as pa_ds


# Default parameters for EW Spectrum & Simulation
DEFAULT_NUM_BANDS: int = 20
DEFAULT_TIME_STEP_US: float = 1000.0  # 1000 microseconds = 1 millisecond
DEFAULT_MAX_ROWS: int = 500_000
DEFAULT_CHUNK_SIZE: int = 50_000


def find_default_dataset_path(base_dir: str | Path = ".") -> Path | None:
    """
    Search for available Turing Synthetic Radar Dataset files in the workspace.
    Prefers stare mode training configs, then scan mode, then any .h5/.parquet/.csv file.
    """
    base_dir = Path(base_dir)
    search_candidates = [
        base_dir / "dataset" / "stare" / "train_stare" / "config_0.h5",
        base_dir / "dataset" / "scan" / "train_scan" / "config_0.h5",
    ]
    for candidate in search_candidates:
        if candidate.is_file():
            return candidate

    # Glob for any .h5 file in dataset/
    h5_matches = list(base_dir.glob("dataset/**/*.h5"))
    if h5_matches:
        return sorted(h5_matches)[0]

    # Glob for parquet or csv files
    tabular_matches = list(base_dir.glob("dataset/**/*.parquet")) + list(
        base_dir.glob("dataset/**/*.csv")
    )
    if tabular_matches:
        return sorted(tabular_matches)[0]

    return None


def stream_h5_chunks(
    file_path: str | Path,
    max_rows: int = DEFAULT_MAX_ROWS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Generator[pa.Table, None, None]:
    """
    Memory-safe generator that streams slices from an HDF5 dataset without loading
    the full file into memory. Converts each chunk into a PyArrow Table.

    Extracts:
        Column 0: Time of Arrival (ToA) in microseconds
        Column 1: Centre Frequency in MHz
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"HDF5 dataset not found at: {file_path}")

    with h5py.File(file_path, "r") as h5_file:
        if "data" not in h5_file:
            raise KeyError(
                f"Expected 'data' dataset in HDF5 file. Found keys: {list(h5_file.keys())}"
            )

        dset = h5_file["data"]
        total_available = dset.shape[0]
        rows_to_read = min(total_available, max_rows)

        rows_yielded = 0
        while rows_yielded < rows_to_read:
            current_chunk = min(chunk_size, rows_to_read - rows_yielded)
            start_idx = rows_yielded
            end_idx = rows_yielded + current_chunk

            # Extract only columns 0 (ToA) and 1 (Centre Frequency)
            raw_chunk = dset[start_idx:end_idx, :2]

            toa_col = pa.array(raw_chunk[:, 0].astype(np.float32), type=pa.float32())
            freq_col = pa.array(raw_chunk[:, 1].astype(np.float32), type=pa.float32())

            chunk_table = pa.Table.from_arrays(
                [toa_col, freq_col], names=["ToA", "Frequency"]
            )
            rows_yielded += current_chunk
            yield chunk_table


def stream_tabular_chunks(
    file_path: str | Path,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> pl.DataFrame:
    """
    Stream tabular datasets (Parquet, Arrow IPC, or CSV) using Polars lazy scanning
    and projection pushdown to extract only ToA and Frequency up to max_rows.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix in [".parquet", ".pq"]:
        lf = pl.scan_parquet(str(file_path))
    elif suffix in [".csv"]:
        lf = pl.scan_csv(str(file_path))
    elif suffix in [".arrow", ".ipc", ".feather"]:
        lf = pl.scan_ipc(str(file_path))
    else:
        raise ValueError(f"Unsupported tabular format: {suffix}")

    # Standardize column names
    col_names = lf.collect_schema().names()
    toa_col = next(
        (c for c in col_names if c.lower() in ["toa", "time of arrival (toa)", "time"]),
        None,
    )
    freq_col = next(
        (
            c
            for c in col_names
            if c.lower()
            in ["frequency", "centre frequency", "centre_frequency", "freq", "cf"]
        ),
        None,
    )

    if not toa_col or not freq_col:
        # Fallback to positional first two columns if specific names not matched
        toa_col, freq_col = col_names[0], col_names[1]

    df = (
        lf.select(
            [
                pl.col(toa_col).cast(pl.Float32).alias("ToA"),
                pl.col(freq_col).cast(pl.Float32).alias("Frequency"),
            ]
        )
        .head(max_rows)
        .collect(streaming=True)
    )
    return df


def load_radar_data(
    source_path: str | Path | None = None,
    max_rows: int = DEFAULT_MAX_ROWS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> pl.DataFrame:
    """
    Loads up to max_rows of radar pulses from the specified source path using Polars & PyArrow.
    If source_path is None, discovers local dataset files automatically.

    Returns:
        pl.DataFrame with columns ['ToA', 'Frequency']
    """
    if source_path is None:
        source_path = find_default_dataset_path()
        if source_path is None:
            raise FileNotFoundError(
                "No local radar dataset found in workspace. Please provide a valid file path."
            )

    source_path = Path(source_path)
    suffix = source_path.suffix.lower()

    if suffix in [".h5", ".hdf5"]:
        # Stream chunks from HDF5 and accumulate into a PyArrow Table, then wrap with Polars
        arrow_chunks = []
        for chunk in stream_h5_chunks(
            source_path, max_rows=max_rows, chunk_size=chunk_size
        ):
            arrow_chunks.append(chunk)

        combined_table = pa.concat_tables(arrow_chunks)
        df = pl.from_arrow(combined_table)
    else:
        df = stream_tabular_chunks(source_path, max_rows=max_rows)

    # Ensure chronological order by Time of Arrival
    df = df.sort("ToA")
    return df


def discretize_pdw_stream(
    pdw_data: pl.DataFrame | pa.Table | np.ndarray,
    num_bands: int = DEFAULT_NUM_BANDS,
    time_step_us: float = DEFAULT_TIME_STEP_US,
    min_freq_mhz: float | None = None,
    max_freq_mhz: float | None = None,
    max_time_steps: int | None = None,
) -> Tuple[np.ndarray, dict[str, Any]]:
    """
    Discretizes continuous radar pulse arrival data into:
    1. Fixed time steps of duration time_step_us.
    2. N discrete frequency bands covering the spectrum [min_freq_mhz, max_freq_mhz].

    Parameters:
        pdw_data: Polars DataFrame, PyArrow Table, or NumPy array containing ToA and Frequency.
        num_bands: Number of discrete frequency bins (N, e.g. 20).
        time_step_us: Discrete time bin width in microseconds (e.g. 1000.0 us = 1 ms).
        min_freq_mhz: Minimum spectrum frequency bound. If None, derived from data.
        max_freq_mhz: Maximum spectrum frequency bound. If None, derived from data.
        max_time_steps: Optional ceiling on total sequential time steps.

    Returns:
        truth_matrix: 2D NumPy array of shape (T, N) with 1 if transmitting, 0 if empty.
        metadata: Dictionary with detailed binning, timing, and pulse ToA mappings.
    """
    if isinstance(pdw_data, pl.DataFrame):
        toa_arr = pdw_data["ToA"].to_numpy()
        freq_arr = pdw_data["Frequency"].to_numpy()
    elif isinstance(pdw_data, pa.Table):
        toa_arr = pdw_data["ToA"].to_numpy()
        freq_arr = pdw_data["Frequency"].to_numpy()
    elif isinstance(pdw_data, np.ndarray):
        toa_arr = pdw_data[:, 0]
        freq_arr = pdw_data[:, 1]
    else:
        raise TypeError(f"Unsupported data type for pdw_data: {type(pdw_data)}")

    num_pulses = len(toa_arr)
    if num_pulses == 0:
        raise ValueError("Cannot discretize empty pulse train data.")

    min_toa = float(np.min(toa_arr))
    max_toa = float(np.max(toa_arr))

    # Establish frequency spectrum bounds
    if min_freq_mhz is None:
        min_freq_mhz = float(np.min(freq_arr))
    if max_freq_mhz is None:
        max_freq_mhz = float(np.max(freq_arr))

    freq_span = max(max_freq_mhz - min_freq_mhz, 1e-3)
    band_edges = np.linspace(min_freq_mhz, max_freq_mhz, num_bands + 1)

    # Compute discrete time step indices
    time_indices = ((toa_arr - min_toa) / time_step_us).astype(np.int64)

    # Compute discrete frequency band indices: clip to [0, num_bands - 1]
    norm_freq = (freq_arr - min_freq_mhz) / freq_span
    freq_indices = np.clip(np.floor(norm_freq * num_bands).astype(np.int64), 0, num_bands - 1)

    # Determine total time steps T
    total_time_steps = int(np.max(time_indices) + 1)
    if max_time_steps is not None:
        total_time_steps = min(total_time_steps, max_time_steps)
        valid_mask = time_indices < total_time_steps
        time_indices = time_indices[valid_mask]
        freq_indices = freq_indices[valid_mask]
        toa_arr = toa_arr[valid_mask]

    # Construct the 2D binary Truth Matrix: shape (T, N)
    truth_matrix = np.zeros((total_time_steps, num_bands), dtype=np.uint8)
    truth_matrix[time_indices, freq_indices] = 1

    # Exact Pulse ToA Matrix: stores earliest arrival ToA (us) in cell (t, b)
    # Used for accurate Intercept Time Error calculation
    pulse_toa_matrix = np.full((total_time_steps, num_bands), np.nan, dtype=np.float64)
    # Populate earliest ToA per bin
    for t_idx, f_idx, toa_val in zip(time_indices, freq_indices, toa_arr):
        if np.isnan(pulse_toa_matrix[t_idx, f_idx]):
            pulse_toa_matrix[t_idx, f_idx] = toa_val
        else:
            pulse_toa_matrix[t_idx, f_idx] = min(pulse_toa_matrix[t_idx, f_idx], toa_val)

    metadata: dict[str, Any] = {
        "num_bands": num_bands,
        "time_step_us": time_step_us,
        "total_time_steps": total_time_steps,
        "min_toa_us": min_toa,
        "max_toa_us": max_toa,
        "min_freq_mhz": min_freq_mhz,
        "max_freq_mhz": max_freq_mhz,
        "band_edges_mhz": band_edges,
        "total_pulses_processed": num_pulses,
        "active_cells_count": int(truth_matrix.sum()),
        "cell_density": float(truth_matrix.sum() / truth_matrix.size),
        "pulse_toa_matrix": pulse_toa_matrix,
    }

    return truth_matrix, metadata


def generate_synthetic_pdw_dataset(
    num_pulses: int = 500_000,
    num_emitters: int = 15,
    duration_us: float = 5_000_000.0,
    min_freq_mhz: float = 1000.0,
    max_freq_mhz: float = 18000.0,
    random_seed: int = 42,
) -> pl.DataFrame:
    """
    Self-contained synthetic radar pulse generator conforming to TSRD characteristics.
    Used for standalone unit testing and CI validation when the 70GB dataset is absent.
    """
    rng = np.random.default_rng(random_seed)

    # Random emitter carrier frequencies
    emitter_freqs = rng.uniform(min_freq_mhz + 200, max_freq_mhz - 200, size=num_emitters)
    # Pulse Repetition Intervals (PRIs) in microseconds
    emitter_pris = rng.uniform(50.0, 500.0, size=num_emitters)

    toas_list = []
    freqs_list = []

    pulses_per_emitter = num_pulses // num_emitters
    for i in range(num_emitters):
        cf = emitter_freqs[i]
        pri = emitter_pris[i]
        start_time = rng.uniform(0.0, 10_000.0)

        # Generate pulse train for this emitter
        pulse_toas = start_time + np.arange(pulses_per_emitter) * pri
        # Emitter frequency jitter / agile hopping
        freq_jitter = rng.normal(0.0, 5.0, size=pulses_per_emitter)
        pulse_freqs = np.clip(cf + freq_jitter, min_freq_mhz, max_freq_mhz)

        toas_list.append(pulse_toas)
        freqs_list.append(pulse_freqs)

    all_toas = np.concatenate(toas_list)
    all_freqs = np.concatenate(freqs_list)

    # Sort interleaved pulse train chronologically
    sort_idx = np.argsort(all_toas)
    sorted_toas = all_toas[sort_idx]
    sorted_freqs = all_freqs[sort_idx]

    arrow_tbl = pa.Table.from_arrays(
        [
            pa.array(sorted_toas.astype(np.float32)),
            pa.array(sorted_freqs.astype(np.float32)),
        ],
        names=["ToA", "Frequency"],
    )
    return pl.from_arrow(arrow_tbl)


if __name__ == "__main__":
    print("=" * 70)
    print("Electronic Warfare RL Simulation - Data Pipeline (Phase 1)")
    print("=" * 70)

    dataset_path = find_default_dataset_path()
    if dataset_path and dataset_path.exists():
        print(f"[*] Found local dataset at: {dataset_path}")
        print(f"[*] Memory-safe streaming first {DEFAULT_MAX_ROWS:,} rows...")
        radar_df = load_radar_data(dataset_path, max_rows=DEFAULT_MAX_ROWS)
    else:
        print("[!] Local dataset path not found. Generating synthetic TSRD pulse train...")
        radar_df = generate_synthetic_pdw_dataset(num_pulses=DEFAULT_MAX_ROWS)

    print(f"[*] Extracted {len(radar_df):,} pulses into Polars DataFrame.")
    print(f"[*] Columns: {radar_df.columns}")
    print(
        f"[*] ToA range: [{radar_df['ToA'].min():.2f} us, {radar_df['ToA'].max():.2f} us]"
    )
    print(
        f"[*] Frequency range: [{radar_df['Frequency'].min():.2f} MHz, {radar_df['Frequency'].max():.2f} MHz]"
    )

    print("\n[*] Discretizing data into Truth Matrix...")
    truth_mat, meta = discretize_pdw_stream(
        radar_df,
        num_bands=DEFAULT_NUM_BANDS,
        time_step_us=DEFAULT_TIME_STEP_US,
    )

    print(f"[+] Truth Matrix Shape: {truth_mat.shape} (Time Steps x Frequency Bands)")
    print(f"[+] Total Time Steps (T): {meta['total_time_steps']}")
    print(f"[+] Discrete Frequency Bands (N): {meta['num_bands']}")
    print(f"[+] Active Transmitting Cells: {meta['active_cells_count']:,} ({meta['cell_density']*100:.2f}% density)")
    print(f"[+] Memory footprint: {truth_mat.nbytes / 1024:.2f} KB")
    print("=" * 70)
