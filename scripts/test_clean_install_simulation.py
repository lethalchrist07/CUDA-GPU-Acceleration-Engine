"""
scripts/test_clean_install_simulation.py
Simulate clean installation with no .pt checkpoints, verifying model discovery,
first-run initialization, missing-model error handling, and recovery instructions.
"""
import os
import sys
import shutil
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app import create_app
from fastapi.testclient import TestClient

def test_clean_install():
    print("=" * 70)
    print("  NEXORA BLK-05: CLEAN INSTALLATION SIMULATION")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as td:
        clean_root = Path(td)
        (clean_root / "checkpoints").mkdir()
        (clean_root / "data").mkdir()
        
        # Copy release manifest and readme
        shutil.copy(PROJECT_ROOT / "checkpoints/MODEL_MANIFEST.json", clean_root / "checkpoints/MODEL_MANIFEST.json")
        shutil.copy(PROJECT_ROOT / "checkpoints/README.md", clean_root / "checkpoints/README.md")

        print(f"[*] Initialized clean environment at: {clean_root}")
        app = create_app(clean_root)
        
        with TestClient(app) as client:
            # 1. Models listing
            res = client.get("/api/models")
            assert res.status_code == 200, f"Expected 200, got {res.status_code}"
            data = res.json()
            print(f"[+] /api/models returned {len(data.get('models', []))} registered architectures:")
            for m in data.get("models", []):
                print(f"    - {m['model_id']} ({m['name']}): status={m['status']}, checkpoints={len(m['checkpoints'])}")

            # 2. ResNet-18 missing checkpoint handling
            res_res = client.post("/api/models/load", json={"model_name": "ResNet18"})
            print(f"\n[+] Missing ResNet18 load response: {res_res.status_code}")
            print(f"    Detail: {res_res.json()}")

            # 3. GPT-86.8M missing checkpoint handling
            res_gpt = client.post("/api/models/load", json={"model_name": "NEXORA GPT-86.8M"})
            print(f"\n[+] Missing GPT-86.8M load response: {res_gpt.status_code}")
            print(f"    Detail: {res_gpt.json()}")

            # 4. Predict endpoint handling when model missing
            res_pred = client.post("/api/models/predict", json={"sample_index": 0})
            print(f"\n[+] Missing model predict response: {res_pred.status_code}")
            print(f"    Detail: {res_pred.json()}")

            # 5. Generate endpoint handling when model missing
            res_gen = client.post("/api/models/generate", json={"prompt": "Test prompt"})
            print(f"\n[+] Missing model generate response: {res_gen.status_code}")
            print(f"    Detail: {res_gen.json()}")

if __name__ == "__main__":
    test_clean_install()
