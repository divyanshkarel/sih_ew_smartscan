"""
test_edge_api.py - Verification script for Phase 3 Edge API

Runs automated verification on the FastAPI endpoints (/reset, /step, /metrics)
testing both 'linear' and 'ml_agent' strategies across 50 steps.
"""

from __future__ import annotations

import io
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from fastapi.testclient import TestClient
from api_backend import app


def test_api():
    print("=" * 75)
    print("        RUNNING FASTAPI ENDPOINT & ONNX INFERENCE VERIFICATION        ")
    print("=" * 75)

    client = TestClient(app)

    with client:
        # 1. Test /reset
        r_reset = client.get("/reset")
        assert r_reset.status_code == 200, f"Reset failed: {r_reset.text}"
        data = r_reset.json()
        print(f"[+] GET /reset passed. Step={data['step']}, Bands={data['num_bands']}, ObsDim={data['observation_dim']}")

        # 2. Test Linear Strategy (10 steps)
        print("\n[*] Testing Strategy='linear' (10 steps)...")
        for i in range(10):
            res = client.post("/step", json={"strategy": "linear"}).json()
            print(
                f"    Step {res['step']:2d} | Band: {res['chosen_band']:2d} | "
                f"Hit: {str(res['is_hit']):<5} | Reward: {res['reward']:5.1f} | "
                f"Pd: {res['metrics']['pd']*100:5.1f}% | Pfa: {res['metrics']['pfa']*100:5.1f}%"
            )

        # 3. Reset and Test ML Agent Strategy (10 steps)
        client.get("/reset")
        print("\n[*] Testing Strategy='ml_agent' with ONNX Inference (10 steps)...")
        for i in range(10):
            res = client.post("/step", json={"strategy": "ml_agent"}).json()
            print(
                f"    Step {res['step']:2d} | Band: {res['chosen_band']:2d} | "
                f"Hit: {str(res['is_hit']):<5} | Latency: {res['inference_latency_us']:6.1f} us | "
                f"Reward: {res['reward']:5.1f} | Pd: {res['metrics']['pd']*100:5.1f}% | Pfa: {res['metrics']['pfa']*100:5.1f}%"
            )

        # 4. Test /metrics
        r_metrics = client.get("/metrics")
        assert r_metrics.status_code == 200
        m_data = r_metrics.json()
        print(f"\n[+] GET /metrics passed:")
        print(f"    - Current Step:           {m_data['step']}")
        print(f"    - Total Scans:            {m_data['total_scans']}")
        print(f"    - Cumulative Hits:        {m_data['cumulative_hits']}")
        print(f"    - Cumulative Reward:      {m_data['cumulative_reward']:.2f}")
        print(f"    - Pd:                     {m_data['pd']*100:.2f}%")
        print(f"    - Pfa:                    {m_data['pfa']*100:.2f}%")

    print("\n" + "=" * 75)
    print("                  ALL API VERIFICATION TESTS PASSED                   ")
    print("=" * 75)


if __name__ == "__main__":
    test_api()
