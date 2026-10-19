import os
import json
import csv
import time
from typing import Dict, Any, List
from .types import InferenceResponse


class LLMTracker:
    """
    Exports LLM inference metrics, hardware latency, and VRAM breakdowns
    to JSON and CSV artifacts for benchmarking and dashboard integration.
    """

    def __init__(self, results_dir: str = "results") -> None:
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        self.json_file = os.path.join(self.results_dir, "inference_experiments.json")
        self.csv_file = os.path.join(self.results_dir, "inference_experiments.csv")

    def log_inference(self, response: InferenceResponse, experiment_id: str = "") -> Dict[str, Any]:
        """Convert InferenceResponse to record and log to disk."""
        meta = response.metadata or {}
        m = response.metrics

        record = {
            "experiment_id": experiment_id or f"infer_{response.device}_{int(time.time())}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": response.model_name,
            "quantization": meta.get("quantization_level", "unknown"),
            "parameter_size": meta.get("parameter_size", "unknown"),
            "device": response.device,
            "device_name": response.device_name,
            "prompt_tokens": m.prompt_tokens,
            "output_tokens": m.output_tokens,
            "ttft_ms": m.ttft_ms,
            "generation_duration_ms": m.generation_duration_ms,
            "total_duration_ms": m.total_duration_ms,
            "tokens_per_second": m.tokens_per_second,
            "backend_vram_mb": m.backend_vram_mb,
            "pytorch_vram_mb": m.pytorch_vram_mb,
            "global_vram_used_mb": m.global_vram_used_mb,
        }

        self._append_to_json(record)
        self._append_to_csv(record)
        return record

    def _append_to_json(self, record: Dict[str, Any]) -> None:
        records: List[Dict[str, Any]] = []
        if os.path.exists(self.json_file):
            try:
                with open(self.json_file, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    if isinstance(content, list):
                        records = content
                    elif isinstance(content, dict) and "inferences" in content:
                        records = content["inferences"]
            except Exception:
                records = []

        records.append(record)
        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump({"inferences": records}, f, indent=2)

    def _append_to_csv(self, record: Dict[str, Any]) -> None:
        file_exists = os.path.exists(self.csv_file)
        fieldnames = [
            "experiment_id",
            "timestamp",
            "model_name",
            "quantization",
            "parameter_size",
            "device",
            "device_name",
            "prompt_tokens",
            "output_tokens",
            "ttft_ms",
            "generation_duration_ms",
            "total_duration_ms",
            "tokens_per_second",
            "backend_vram_mb",
            "pytorch_vram_mb",
            "global_vram_used_mb",
        ]

        row = {k: record.get(k, "") for k in fieldnames}

        with open(self.csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
