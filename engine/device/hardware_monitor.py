"""
Hardware Protection and Telemetry Monitoring Module for CUDA GPU Acceleration Engine (Phase 9).
Provides continuous GPU thermal monitoring, VRAM tracking, utilization profiling,
graceful OOM recovery with automatic batch size reduction, and emergency checkpointing.
"""
from __future__ import annotations

import csv
import gc
import io
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class HardwareMonitor:
    """
    Monitors and guards GPU hardware resources during training and inference.
    Enforces thermal limits without altering driver or clock configurations.
    """

    def __init__(
        self,
        device_id: int = 0,
        warn_temp_c: float = 75.0,
        max_temp_c: float = 83.0,
        log_file: Optional[str] = "results/hardware_telemetry.csv",
    ):
        self.device_id = device_id
        self.warn_temp_c = warn_temp_c
        self.max_temp_c = max_temp_c
        self.log_file = log_file
        self.history: List[Dict[str, Any]] = []

        if self.log_file:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_file)), exist_ok=True)

    @staticmethod
    def _run_nvidia_smi(query: str) -> Optional[List[str]]:
        """Query nvidia-smi safely with timeout and no console window."""
        command = shutil.which("nvidia-smi")
        if not command:
            return None
        try:
            output = subprocess.run(
                [command, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            lines = [line.strip() for line in output.stdout.strip().splitlines() if line.strip()]
            if lines:
                return [field.strip() for field in lines[0].split(",")]
            return None
        except Exception as exc:
            logger.debug(f"nvidia-smi query failed: {exc}")
            return None

    def get_telemetry(self) -> Dict[str, Any]:
        """
        Collect unified live hardware telemetry combining PyTorch CUDA metrics and nvidia-smi.
        """
        telemetry: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device_id": self.device_id,
            "cuda_available": torch.cuda.is_available(),
            "device_name": "CPU",
            "temperature_c": None,
            "utilization_gpu_pct": None,
            "power_draw_w": None,
            "power_limit_w": None,
            "vram_allocated_mb": 0.0,
            "vram_reserved_mb": 0.0,
            "vram_total_mb": 0.0,
            "vram_free_mb": 0.0,
        }

        if not torch.cuda.is_available():
            return telemetry

        telemetry["device_name"] = torch.cuda.get_device_name(self.device_id)

        # PyTorch Memory Stats
        try:
            allocated = torch.cuda.memory_allocated(self.device_id)
            reserved = torch.cuda.memory_reserved(self.device_id)
            total = torch.cuda.get_device_properties(self.device_id).total_memory
            telemetry["vram_allocated_mb"] = round(allocated / (1024.0 * 1024.0), 2)
            telemetry["vram_reserved_mb"] = round(reserved / (1024.0 * 1024.0), 2)
            telemetry["vram_total_mb"] = round(total / (1024.0 * 1024.0), 2)
            telemetry["vram_free_mb"] = round((total - reserved) / (1024.0 * 1024.0), 2)
        except Exception as exc:
            logger.debug(f"Could not read PyTorch memory stats: {exc}")

        # Live NVIDIA SMI Stats
        smi_fields = self._run_nvidia_smi("temperature.gpu,utilization.gpu,power.draw,power.limit")
        if smi_fields and len(smi_fields) >= 4:
            try:
                telemetry["temperature_c"] = float(smi_fields[0])
            except (ValueError, TypeError):
                pass
            try:
                telemetry["utilization_gpu_pct"] = float(smi_fields[1])
            except (ValueError, TypeError):
                pass
            try:
                telemetry["power_draw_w"] = float(smi_fields[2])
            except (ValueError, TypeError):
                pass
            try:
                telemetry["power_limit_w"] = float(smi_fields[3])
            except (ValueError, TypeError):
                pass

        return telemetry

    def check_thermal_safety(self) -> Tuple[bool, Optional[str]]:
        """
        Verify that GPU operating temperature remains within safe manufacturer boundaries.
        Returns: (is_safe, warning_or_error_message)
        """
        telemetry = self.get_telemetry()
        temp = telemetry.get("temperature_c")
        if temp is None:
            return True, None

        if temp >= self.max_temp_c:
            msg = (
                f"CRITICAL THERMAL ALERT: GPU temperature {temp:.1f}°C exceeds critical threshold "
                f"({self.max_temp_c:.1f}°C). Workload must be suspended to protect hardware!"
            )
            logger.error(msg)
            return False, msg

        if temp >= self.warn_temp_c:
            msg = (
                f"THERMAL WARNING: GPU temperature {temp:.1f}°C exceeds warning threshold "
                f"({self.warn_temp_c:.1f}°C)."
            )
            logger.warning(msg)
            return True, msg

        return True, None

    def handle_oom(self, current_batch_size: int, min_batch_size: int = 16) -> Optional[int]:
        """
        Graceful recovery from CUDA Out-Of-Memory condition:
        1. Clears PyTorch CUDA memory cache
        2. Performs Python garbage collection
        3. Halves the batch size if above min_batch_size
        Returns: new_batch_size or None if cannot be reduced.
        """
        logger.warning(f"OOM condition encountered at batch size {current_batch_size}. Initiating recovery...")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

        if current_batch_size <= min_batch_size:
            logger.error(
                f"Cannot reduce batch size below minimum limit {min_batch_size}. Recovery aborted."
            )
            return None

        new_batch_size = max(min_batch_size, current_batch_size // 2)
        logger.info(
            f"OOM Recovery Successful: Cache flushed, reducing batch size from {current_batch_size} -> {new_batch_size}."
        )
        return new_batch_size

    def record_snapshot(
        self,
        epoch: int,
        step: Optional[int] = None,
        phase: str = "train",
        loss: Optional[float] = None,
        accuracy: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Record a telemetry snapshot to history and append to telemetry CSV."""
        snapshot = self.get_telemetry()
        snapshot["epoch"] = epoch
        snapshot["step"] = step
        snapshot["phase"] = phase
        snapshot["loss"] = loss
        snapshot["accuracy"] = accuracy
        self.history.append(snapshot)

        if self.log_file:
            self._append_to_csv(snapshot)

        return snapshot

    def _append_to_csv(self, record: Dict[str, Any]) -> None:
        file_exists = os.path.exists(self.log_file)
        fieldnames = [
            "timestamp",
            "epoch",
            "step",
            "phase",
            "loss",
            "accuracy",
            "temperature_c",
            "utilization_gpu_pct",
            "power_draw_w",
            "vram_allocated_mb",
            "vram_reserved_mb",
            "vram_free_mb",
            "vram_total_mb",
        ]
        row = {k: record.get(k, "") for k in fieldnames}
        try:
            with open(self.log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as exc:
            logger.debug(f"Failed to append hardware telemetry CSV: {exc}")
