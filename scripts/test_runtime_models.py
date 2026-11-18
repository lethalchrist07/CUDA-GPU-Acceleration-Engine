"""
scripts/test_runtime_models.py
Empirical testing of model loading, real inference, VRAM footprint, and CPU fallback.
"""
import os
import sys
import time
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from backend.serving import ModelServer

def get_vram_mb():
    if torch.cuda.is_available():
        return {
            "allocated_mb": round(torch.cuda.memory_allocated() / (1024**2), 2),
            "reserved_mb": round(torch.cuda.memory_reserved() / (1024**2), 2),
            "max_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 2),
        }
    return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}

def main():
    print("=" * 70)
    print("  NEXORA BLK-05: FIRST-RUN MODEL RUNTIME VERIFICATION")
    print("=" * 70)
    
    cuda_avail = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    print(f"[*] CUDA Available: {cuda_avail} | Device: {dev_name}")
    if cuda_avail:
        print(f"[*] Total VRAM: {round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)} GB")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"[*] Baseline VRAM: {get_vram_mb()}")

    # -------------------------------------------------------------
    # 1. Vision Model (ResNet18) on CUDA
    # -------------------------------------------------------------
    print("\n--- TEST 1: ResNet18 Loading and Inference on CUDA ---")
    if cuda_avail:
        vram_pre_load = get_vram_mb()
        load_res = ModelServer.load_model(model_name="ResNet18", device_str="cuda:0")
        vram_post_load = get_vram_mb()
        print(f"[+] Loaded: {load_res['model_name']} on {load_res['device']} (Params: {load_res['parameters']:,})")
        print(f"    Checkpoint: {load_res['checkpoint']}")
        print(f"    VRAM added by weights: {round(vram_post_load['allocated_mb'] - vram_pre_load['allocated_mb'], 2)} MB")

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        pred_res = ModelServer.predict(sample_index=0)
        pred_time = (time.perf_counter() - t0) * 1000.0
        vram_post_infer = get_vram_mb()
        print(f"[+] Prediction: class={pred_res['predicted_class']} (idx={pred_res['predicted_index']})")
        print(f"    Confidence: {pred_res['confidence'] * 100:.2f}% | Latency: {pred_res['latency_ms']:.2f} ms")
        print(f"    Ground Truth: {pred_res['metadata'].get('ground_truth_class')} | Correct: {pred_res['correct']}")
        print(f"    Top predictions: {pred_res['top_predictions'][:3]}")
        print(f"    Peak VRAM during inference: {vram_post_infer['max_allocated_mb']} MB")

        # Unload
        unload_res = ModelServer.unload_model("vision")
        print(f"[+] Unloaded vision: {unload_res}")
        torch.cuda.empty_cache()
        print(f"    VRAM after unload & cache empty: {get_vram_mb()}")

    # -------------------------------------------------------------
    # 2. Vision Model (ResNet18) on CPU Fallback
    # -------------------------------------------------------------
    print("\n--- TEST 2: ResNet18 Loading and Inference on CPU Fallback ---")
    load_cpu = ModelServer.load_model(model_name="ResNet18", device_str="cpu")
    print(f"[+] Loaded CPU: {load_cpu['model_name']} on {load_cpu['device']} (Params: {load_cpu['parameters']:,})")
    pred_cpu = ModelServer.predict(sample_index=0)
    print(f"[+] CPU Prediction: class={pred_cpu['predicted_class']} (idx={pred_cpu['predicted_index']})")
    print(f"    Confidence: {pred_cpu['confidence'] * 100:.2f}% | Latency: {pred_cpu['latency_ms']:.2f} ms")
    print(f"    Top predictions: {pred_cpu['top_predictions'][:3]}")
    assert pred_cpu["predicted_class"] == "cat", "Expected cat for CIFAR-10 sample 0"
    ModelServer.unload_model("vision")

    # -------------------------------------------------------------
    # 3. Transformer Model (NEXORA GPT-86.8M) on CUDA
    # -------------------------------------------------------------
    print("\n--- TEST 3: NEXORA GPT-86.8M Loading and Inference on CUDA ---")
    if cuda_avail:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        vram_pre_gpt = get_vram_mb()
        load_gpt = ModelServer.load_model(model_name="NEXORA GPT-86.8M", device_str="cuda:0")
        vram_post_gpt = get_vram_mb()
        print(f"[+] Loaded: {load_gpt['model_name']} on {load_gpt['device']} (Params: {load_gpt['parameters']:,})")
        print(f"    Checkpoint: {load_gpt['checkpoint']}")
        print(f"    VRAM added by model: {round(vram_post_gpt['allocated_mb'] - vram_pre_gpt['allocated_mb'], 2)} MB")

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        gen_res = ModelServer.generate(
            prompt="Technical Foundations of GPU Computing:",
            max_new_tokens=30,
            temperature=0.8,
            device_str="cuda:0",
            seed=42
        )
        vram_post_gen = get_vram_mb()
        print(f"[+] Generation Status: {gen_res['status']}")
        print(f"    Tokens generated: {gen_res['generated_tokens']} in {gen_res['generation_duration_ms']:.2f} ms")
        print(f"    Throughput: {gen_res['tokens_per_second']:.1f} tokens/sec")
        print(f"    Generated text: {repr(gen_res['generated_text'])}")
        print(f"    Full output: {repr(gen_res['full_text'])}")
        print(f"    Peak VRAM: {vram_post_gen['max_allocated_mb']} MB (reported peak_vram_mb: {gen_res['peak_vram_mb']})")

        unload_gpt = ModelServer.unload_model("transformer")
        print(f"[+] Unloaded transformer: {unload_gpt}")
        torch.cuda.empty_cache()
        print(f"    VRAM after unload & cache empty: {get_vram_mb()}")

    # -------------------------------------------------------------
    # 4. Transformer Model (NEXORA GPT-86.8M) on CPU Fallback
    # -------------------------------------------------------------
    print("\n--- TEST 4: NEXORA GPT-86.8M Loading and Inference on CPU Fallback ---")
    load_gpt_cpu = ModelServer.load_model(model_name="NEXORA GPT-86.8M", device_str="cpu")
    print(f"[+] Loaded CPU: {load_gpt_cpu['model_name']} on {load_gpt_cpu['device']}")
    gen_cpu = ModelServer.generate(
        prompt="Technical Foundations of GPU Computing:",
        max_new_tokens=15,
        temperature=0.8,
        device_str="cpu",
        seed=42
    )
    print(f"[+] CPU Generation Status: {gen_cpu['status']}")
    print(f"    Tokens generated: {gen_cpu['generated_tokens']} in {gen_cpu['generation_duration_ms']:.2f} ms")
    print(f"    Throughput: {gen_cpu['tokens_per_second']:.1f} tokens/sec")
    print(f"    Generated text: {repr(gen_cpu['generated_text'])}")
    ModelServer.unload_model("all")

    print("\n" + "=" * 70)
    print("  RUNTIME VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
