"""
scripts/test_missing_models.py
Verify exact error messages and recovery instructions when checkpoints and datasets are missing.
"""
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import backend.serving as serving

def test_missing_behavior():
    print("=" * 70)
    print("  TESTING MISSING MODEL & DATASET ERROR HANDLING")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as td:
        clean_dir = Path(td)
        (clean_dir / "checkpoints").mkdir()
        (clean_dir / "data").mkdir()

        # Point serving PROJECT_ROOT to clean_dir
        orig_root = serving.PROJECT_ROOT
        try:
            serving.PROJECT_ROOT = clean_dir

            # 1. Model discovery
            models = serving.ModelServer.list_available_models()
            print("[+] Model listing in sterile environment:")
            for m in models:
                print(f"    - {m['model_id']} ({m['name']}): {len(m['checkpoints'])} checkpoints found")

            # 2. ResNet18 load
            try:
                serving.ModelServer.load_model("ResNet18")
                print("[-] FAIL: ResNet18 loaded unexpectedly!")
            except FileNotFoundError as e:
                print(f"\n[+] ResNet18 FileNotFoundError: {e}")

            # 3. Transformer load
            try:
                serving.ModelServer.load_model("NEXORA GPT-86.8M")
                print("[-] FAIL: Transformer loaded unexpectedly!")
            except FileNotFoundError as e:
                print(f"\n[+] Transformer FileNotFoundError: {e}")

            # 4. Predict
            try:
                serving.ModelServer.predict(sample_index=0)
                print("[-] FAIL: Predict executed unexpectedly!")
            except FileNotFoundError as e:
                print(f"\n[+] Predict FileNotFoundError: {e}")

            # 5. Generate
            try:
                serving.ModelServer.generate(prompt="Hello")
                print("[-] FAIL: Generate executed unexpectedly!")
            except FileNotFoundError as e:
                print(f"\n[+] Generate FileNotFoundError: {e}")

        finally:
            serving.PROJECT_ROOT = orig_root

    print("\n" + "=" * 70)
    print("  MISSING BEHAVIOR TEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    test_missing_behavior()
