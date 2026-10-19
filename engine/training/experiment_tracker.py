import os
import json
import csv
import time
from typing import Dict, Any, List, Optional


class ExperimentTracker:
    """
    Manages structured logging and export of AI model training experiments
    to JSON and CSV artifacts for benchmarking and dashboard visualization.
    """

    def __init__(self, results_dir: str = "results", filename_prefix: str = "") -> None:
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        self.json_file = os.path.join(self.results_dir, f"{filename_prefix}training_experiments.json")
        self.csv_file = os.path.join(self.results_dir, f"{filename_prefix}training_experiments.csv")

    def log_experiment(self, experiment_data: Dict[str, Any]) -> None:
        """Log a new experiment entry to both JSON and CSV files."""
        # Add timestamp if missing
        if "timestamp" not in experiment_data:
            experiment_data["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

        self._append_to_json(experiment_data)
        self._append_to_csv(experiment_data)

    def _append_to_json(self, record: Dict[str, Any]) -> None:
        experiments: List[Dict[str, Any]] = []
        if os.path.exists(self.json_file):
            try:
                with open(self.json_file, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    if isinstance(content, list):
                        experiments = content
                    elif isinstance(content, dict) and "experiments" in content:
                        experiments = content["experiments"]
            except Exception:
                experiments = []

        experiments.append(record)
        with open(self.json_file, "w", encoding="utf-8") as f:
            json.dump({"experiments": experiments}, f, indent=2)

    def _append_to_csv(self, record: Dict[str, Any]) -> None:
        file_exists = os.path.exists(self.csv_file)
        fieldnames = [
            "experiment_id",
            "timestamp",
            "model_name",
            "parameters",
            "device",
            "device_name",
            "epochs",
            "batch_size",
            "learning_rate",
            "optimizer",
            "seed",
            "use_amp",
            "total_train_time_ms",
            "avg_epoch_time_ms",
            "final_train_loss",
            "final_val_loss",
            "best_val_accuracy",
            "peak_gpu_memory_mb",
            "speedup_vs_cpu",
        ]

        row = {k: record.get(k, "") for k in fieldnames}

        with open(self.csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
