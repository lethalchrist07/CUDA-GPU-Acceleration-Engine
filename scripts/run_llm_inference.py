import sys
import os
import argparse

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.llm.engine import LLMEngine
from engine.llm.safety import safe_display_output


def main():
    parser = argparse.ArgumentParser(description="CUDA AI Acceleration Engine - Local LLM Inference CLI")
    parser.add_argument("--prompt", type=str, default="", help="Prompt to generate a response for")
    parser.add_argument("--model", type=str, default="qwen3.5:4b", help="Model name (e.g. qwen3.5:4b, qwen2.5-coder:7b)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "compare"],
                        help="Inference device: cuda, cpu, or compare (runs on both and benchmarks)")
    parser.add_argument("--max-tokens", type=int, default=64, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--context-length", type=int, default=2048, help="Context length window")
    parser.add_argument("--list-models", action="store_true", help="List installed local models and exit")

    args = parser.parse_args()

    engine = LLMEngine(results_dir="results")

    if args.list_models:
        print("=" * 80)
        print("  CUDA AI Acceleration Engine - Available Local LLM Models")
        print("=" * 80)
        models = engine.list_available_models()
        if models:
            for m in models:
                info = engine.get_model_info(m)
                print(f"  - {m:<20} | Params: {info.get('parameter_size', 'N/A'):<8} | "
                      f"Quant: {info.get('quantization_level', 'N/A'):<10} | Format: {info.get('format', 'N/A')}")
        else:
            print("  No models found in the active runtime.")
        print("=" * 80)
        return 0

    prompt = args.prompt.strip()
    if not prompt:
        prompt = "Explain in two sentences how NVIDIA CUDA accelerates matrix operations in deep learning."

    print("=" * 90)
    print("      CUDA AI Acceleration Engine - Local LLM Inference Module")
    print("=" * 90)
    print(f"[*] Target Model:       {args.model}")
    print(f"[*] Hyperparameters:   MaxTokens={args.max_tokens}, Temp={args.temperature}, ContextLen={args.context_length}")
    print(f"[*] Input Prompt:       \"{prompt}\"\n")

    if args.device == "compare":
        print("-" * 90)
        print(">>> RUNNING HEAD-TO-HEAD CPU vs GPU INFERENCE BENCHMARK")
        print("-" * 90)

        # 1. CPU Run
        print("[*] Dispatching to CPU (100% Host Threads)...")
        cpu_resp = engine.generate(
            prompt=prompt,
            model_name=args.model,
            device="cpu",
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            context_length=args.context_length,
            experiment_id=f"infer_cpu_{args.model.replace(':', '_')}",
        )

        # 2. GPU Run
        print("[*] Dispatching to NVIDIA RTX 3050 GPU (CUDA Cores)...")
        gpu_resp = engine.generate(
            prompt=prompt,
            model_name=args.model,
            device="cuda",
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            context_length=args.context_length,
            experiment_id=f"infer_gpu_{args.model.replace(':', '_')}",
        )

        # Print outputs safely
        print("\n" + "=" * 90)
        print("                               GENERATED RESPONSES")
        print("=" * 90)
        print(f"[CPU Response ({cpu_resp.device_name})]:\n{safe_display_output(cpu_resp.text.strip())}\n")
        print(f"[GPU Response ({gpu_resp.device_name})]:\n{safe_display_output(gpu_resp.text.strip())}\n")

        # Comparison Table
        print("=" * 90)
        print("                           INFERENCE PERFORMANCE METRICS")
        print("=" * 90)
        print(f"{'Metric':<32} | {'CPU (Intel Core i5)':<25} | {'GPU (NVIDIA RTX 3050)':<25}")
        print("-" * 90)
        m_cpu = cpu_resp.metrics
        m_gpu = gpu_resp.metrics
        ttft_speedup = (m_cpu.ttft_ms / m_gpu.ttft_ms) if m_gpu.ttft_ms > 0 else 1.0
        tps_speedup = (m_gpu.tokens_per_second / m_cpu.tokens_per_second) if m_cpu.tokens_per_second > 0 else 1.0

        print(f"{'Time to First Token (TTFT)':<32} | {m_cpu.ttft_ms:.1f} ms{'':<18} | {m_gpu.ttft_ms:.1f} ms ({ttft_speedup:.2f}x Faster)")
        print(f"{'Generation Duration':<32} | {m_cpu.generation_duration_ms:.1f} ms{'':<18} | {m_gpu.generation_duration_ms:.1f} ms")
        print(f"{'Total End-to-End Latency':<32} | {m_cpu.total_duration_ms:.1f} ms{'':<18} | {m_gpu.total_duration_ms:.1f} ms")
        print(f"{'Prompt Tokens Processed':<32} | {m_cpu.prompt_tokens} tokens{'':<16} | {m_gpu.prompt_tokens} tokens")
        print(f"{'Output Tokens Generated':<32} | {m_cpu.output_tokens} tokens{'':<16} | {m_gpu.output_tokens} tokens")
        print(f"{'Throughput (Tokens / Sec)':<32} | {m_cpu.tokens_per_second:.2f} tok/s{'':<16} | {m_gpu.tokens_per_second:.2f} tok/s ({tps_speedup:.2f}x Faster)")
        print("-" * 90)
        print(f"{'LLM Backend VRAM Held':<32} | {m_cpu.backend_vram_mb:.1f} MB{'':<18} | {m_gpu.backend_vram_mb:.1f} MB")
        print(f"{'PyTorch Process VRAM':<32} | {m_cpu.pytorch_vram_mb:.1f} MB{'':<18} | {m_gpu.pytorch_vram_mb:.1f} MB")
        print(f"{'Total Hardware VRAM in Use':<32} | {m_cpu.global_vram_used_mb:.1f} MB{'':<18} | {m_gpu.global_vram_used_mb:.1f} MB")
        print("=" * 90)
        print(f"[+] Results saved to results/inference_experiments.csv and results/inference_experiments.json\n")

    else:
        # Single Device Run
        print(f"[*] Dispatching generation request to device: {args.device}...")
        resp = engine.generate(
            prompt=prompt,
            model_name=args.model,
            device=args.device,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            context_length=args.context_length,
        )

        m = resp.metrics
        print("\n" + "=" * 90)
        print("                               GENERATED RESPONSE")
        print("=" * 90)
        print(safe_display_output(resp.text.strip()))
        print("\n" + "=" * 90)
        print("                               PERFORMANCE METRICS")
        print("=" * 90)
        print(f"  Device:                 {resp.device} ({resp.device_name})")
        print(f"  Time to First Token:    {m.ttft_ms:.1f} ms")
        print(f"  Generation Duration:    {m.generation_duration_ms:.1f} ms")
        print(f"  Total Duration:         {m.total_duration_ms:.1f} ms")
        print(f"  Output Tokens:          {m.output_tokens} tokens ({m.tokens_per_second:.2f} tok/s)")
        print(f"  Backend VRAM:           {m.backend_vram_mb:.1f} MB")
        print(f"  PyTorch Process VRAM:   {m.pytorch_vram_mb:.1f} MB")
        print(f"  Global Hardware VRAM:   {m.global_vram_used_mb:.1f} MB")
        print("=" * 90 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
