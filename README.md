# Smart Scan Strategy for Electronic Warfare (SIH 2026)
### Edge-Deployable, Threat-Aware Cooperative ML Scheduling Against Frequency-Agile Emitters

[![Inference Latency](https://img.shields.io/badge/ONNX_INT8_Latency-17.28_%C2%B5s-brightgreen)](#)
[![Model Footprint](https://img.shields.io/badge/Edge_Model_Size-185.87_KB-blue)](#)
[![Single Rx Pd](https://img.shields.io/badge/Single_Rx_Pd-12.13%25_(Ceiling_13.18%25)-orange)](#)
[![Cooperative Rx Pd](https://img.shields.io/badge/Dual_Cooperative_Pd-25.34%25-success)](#)
[![Tier 1 Missile Defense](https://img.shields.io/badge/Lethal_Threat_Pd-30.56%25-red)](#)

---

## 1. Operational Problem & Baseline Reality
Modern Electronic Warfare (EW) platforms operate under severe physical constraints: high-sensitivity heterodyne receivers typically possess instantaneous bandwidths an order of magnitude narrower than the contested RF spectrum. 

Legacy open-loop raster sweeps blindly cycle through frequency channels sequentially. Against agile, frequency-hopping emitters and low-duty-cycle missile fire-control radars:
* **Severe Scanning Waste:** Baseline raster scans spend **62.42%** of their mission time scanning empty spectrum ($P_{fa}$).
* **Poor Intercept Probability:** The baseline captures less than **4.95%** of active transmissions ($P_d$).
* **Physical Single-Receiver Ceiling:** Across 6,823 temporal steps containing 51,766 agile transmissions, a single sensor can scan at most 6,823 times. The theoretical mathematical ceiling for any single receiver is:
  $$\text{Physical Ceiling} = \frac{6,823}{51,766} \approx 13.18\%$$

---

## 2. Quantitative Performance Scorecard

| Operational Metric | Legacy Raster Sweep | Single-Agent DQN (Phase 2) | Cooperative Dual DQN (Upgrade 1) | Threat-Priority Dual DQN (Upgrade 2) |
| :--- | :--- | :--- | :--- | :--- |
| **Receiver Architecture** | 1 Channel Sequential | 1 Channel Agile | 2 Channels Joint ($20 \times 20$) | 2 Channels Threat-Weighted |
| **Probability of Detection ($P_d$)** | 4.95% | **12.13%** (92% of limit) | **25.34%** (98.4% of dual limit) | **13.22%** (Selective Defense) |
| **False Alarm Rate ($P_{fa}$)** | 62.42% | 7.94% | **3.86%** | 49.85% (Silent Vigilance) |
| **Tier 1 Missile Lock $P_d$** | 0.00% | 0.00% | 0.00% | **30.56%** (+30.56% Net) |
| **Hardware Collision Rate** | N/A | N/A | **0.00% (0 Collisions)** | **0.00% (0 Collisions)** |
| **Episode Mission Score** | 18,789.0 | 62,206.8 ($3.31\times$) | 129,890.0 ($3.41\times$) | **129,730.0** (Net Lethality) |

---

## 3. Key Architectural Innovations

### A. Memory-Safe Data Pipeline (70 GB TSRD Dataset)
* Implemented zero-copy streaming ingestion using **PyArrow** and **Polars** expressions.
* Discretized continuous Pulse Descriptor Words (PDWs) from the Turing Synthetic Radar Dataset into a $6,823 \times 20$ multi-emitter truth matrix without memory crashes.

### B. Hardware-Aware Cooperative Scheduling
* Integrated physical RF local oscillator (LO) slew penalties: $\text{Penalty} = -0.2 \times |a_{\text{current}} - a_{\text{chosen}}|$.
* Deployed a joint 400-action discrete space for dual receivers. The agent autonomously learned **spectral deconfliction with 0.00% channel overlap**, shattering the single-receiver 13.18% ceiling.

### C. Threat Priority Engine (Lethality-Weighted)
* Raw detection count does not equal mission survivability. Emitters are mapped into 3 tactical tiers:
  * **Tier 1 (Missile Lock / Fire-Control, Bands 16–19):** $+50$ Hit, $-20$ Unintercepted Penalty.
  * **Tier 2 (Agile Interceptor, Bands 10–15):** $+20$ Hit.
  * **Tier 3 (Routine Surveillance, Bands 0–9):** $+5$ Hit.
* Receiver 1 hunts high-density agile signals, while Receiver 2 maintains continuous overwatch on lethal channels, increasing critical missile detection from 0% to **30.56%**.

### D. Sub-20 µs INT8 Edge Quantization
* Graph export to ONNX followed by dynamic integer quantization (`QUInt8`):
  * **Memory Footprint:** Compressed from 712.86 KB down to **185.87 KB** (73.9% reduction).
  * **Inference Latency:** Mean execution time of **17.28 µs** (P99 at 24.11 µs).
  * **Real-Time Margin:** Consumes only 17.3% of the 100 µs radar pulse dwell window, leaving over 82 µs for RF synthesizer settling.

### E. Deterministic Periodic Scan Receiver (PSR) Synchronizer
* Non-RL Track-While-Scan (TWS) interception module using inter-pulse differencing and ordinary least squares phase regression.
* Recovers antenna rotation period ($T_{\text{rot}} = 2000$ steps) with **0.11-step residual error (0.005%)**.
* Employs 5-step advance parking to achieve a **100.0% intercept rate**, resolving the low-duty-cycle blind spot of purely reactive ML.

---

## 4. Repository Structure

```text
├── data_pipeline.py         # PyArrow streaming pipeline for 70GB TSRD
├── rf_env.py                # Single-receiver Gymnasium environment
├── train_dqn.py             # Stable-Baselines3 single-receiver trainer
├── evaluate_agents.py       # Comparative evaluator for single-receiver RL vs baseline
├── rf_env_dual.py           # Dual-receiver cooperative environment (400 actions)
├── train_dual_dqn.py        # Cooperative dual-receiver DQN trainer
├── evaluate_dual.py         # Comparative evaluator for dual-receiver RL vs baseline
├── rf_env_threat.py         # Threat-priority environment (Tiered lethality)
├── train_threat_dqn.py      # Transfer learning threat-weighted trainer
├── quantize_edge.py         # FP32 to INT8 dynamic quantization & micro-benchmarking
├── psr_synchronizer.py      # Phase-locked periodic scan synchronizer
├── api_backend.py           # FastAPI service serving INT8 ONNX inference
├── test_edge_api.py         # Automated test suite for edge inference API
├── smart_scan_dashboard/    # Flutter web tactical War Room UI
├── smart_scan_dqn.zip       # Pre-trained Single-Receiver DQN weights
├── smart_scan_dual_dqn.zip  # Pre-trained Cooperative Dual-Receiver DQN weights
├── smart_scan_threat_dqn.zip# Pre-trained Threat-Priority Dual-Receiver DQN weights
├── smart_scan_edge.onnx     # Single-Receiver FP32 ONNX model
├── threat_agent_fp32.onnx   # Threat-Priority Dual-Receiver FP32 ONNX model
├── threat_agent_int8.onnx   # Threat-Priority Dual-Receiver INT8 quantized model
└── requirements.txt         # Production Python dependencies
```

---

## 5. Quickstart & Evaluation Instructions

### A. Environment Setup
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### B. Execute INT8 Quantization Benchmark (Upgrade 3)
```powershell
python quantize_edge.py
```
*Outputs file size compression (712 KB -> 185 KB) and microsecond inference benchmark (17.28 µs latency).*

### C. Execute Periodic Scan Synchronizer (Upgrade 4)
```powershell
python psr_synchronizer.py
```
*Validates deterministic $T_{\text{rot}}$ extraction ($1999.89$ steps) and 100.0% post-lock intercept rate.*

### D. Execute Dual-Receiver Cooperative Benchmark (Upgrade 1)
```powershell
python evaluate_dual.py
```
*Validates 25.34% cooperative $P_d$ and 0.00% hardware collision rate.*

### E. Launch Live Edge API & Tactical War Room UI
1. **Start FastAPI Backend:**
   ```powershell
   uvicorn api_backend:app --port 8000
   ```
2. **Launch Flutter War Room Dashboard:**
   ```powershell
   cd smart_scan_dashboard
   flutter run -d chrome
   ```
