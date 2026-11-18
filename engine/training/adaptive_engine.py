"""
Adaptive Training Engine for NEXORA AI Platform.
Provides hardware capability discovery, VRAM budgeting, safe automatic batch-size
and precision selection, and dynamic OOM recovery for single-GPU architectures
(prioritizing NVIDIA GeForce RTX 3050 Laptop GPU with 6GB VRAM) and CPU fallback.
"""

from __future__ import annotations

import gc
import logging
import math
import os
import threading
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

logger = logging.getLogger("AdaptiveTrainingEngine")

# Supported models and architectural metadata
MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "cifar10_mlp": {
        "parameters": 3426250,
        "type": "vision_mlp",
        "default_batch_size": 128,
        "min_batch_size": 16,
        "max_batch_size": 512,
        "bytes_per_sample_act": 36864,  # Intermediate activations ~36KB per sample
        "optimizer": "sgd",
    },
    "baseline_mlp": {
        "parameters": 465172,
        "type": "vision_mlp",
        "default_batch_size": 128,
        "min_batch_size": 16,
        "max_batch_size": 512,
        "bytes_per_sample_act": 8192,
        "optimizer": "sgd",
    },
    "resnet18": {
        "parameters": 11173962,
        "type": "vision_cnn",
        "default_batch_size": 128,
        "min_batch_size": 16,
        "max_batch_size": 256,
        "bytes_per_sample_act": 1887436,  # ~1.8MB per sample for CIFAR stem
        "optimizer": "sgd",
    },
    "transformer_1m": {
        "parameters": 957184,
        "type": "causal_lm",
        "default_batch_size": 8,
        "min_batch_size": 1,
        "max_batch_size": 64,
        "layers": 4,
        "hidden_size": 128,
        "heads": 4,
        "default_seq_len": 256,
        "optimizer": "adamw",
    },
    "transformer_25m": {
        "parameters": 23063808,
        "type": "causal_lm",
        "default_batch_size": 4,
        "min_batch_size": 1,
        "max_batch_size": 32,
        "layers": 12,
        "hidden_size": 384,
        "heads": 6,
        "default_seq_len": 256,
        "optimizer": "adamw",
    },
    "transformer_50m": {
        "parameters": 86825472,  # NEXORA GPT-86.8M
        "type": "causal_lm",
        "default_batch_size": 2,
        "min_batch_size": 1,
        "max_batch_size": 16,
        "layers": 12,
        "hidden_size": 768,
        "heads": 12,
        "default_seq_len": 256,
        "optimizer": "adamw",
    },
    "transformer_100m": {
        "parameters": 92133888,  # NEXORA GPT-100M Flagship
        "type": "causal_lm",
        "default_batch_size": 2,
        "min_batch_size": 1,
        "max_batch_size": 16,
        "layers": 12,
        "hidden_size": 768,
        "heads": 12,
        "default_seq_len": 256,
        "optimizer": "adamw",
    },
}

# Alias mappings
MODEL_ALIASES: Dict[str, str] = {
    "nexora_gpt": "transformer_50m",
    "nexora_gpt_86m": "transformer_50m",
    "nexora_gpt_86.8m": "transformer_50m",
    "nexora_gpt_100m": "transformer_100m",
    "transformer": "transformer_100m",
}


@dataclass
class AdaptiveHardwareProfile:
    """Hardware capability specification for training runtime."""
    device_name: str
    device_type: str  # "cuda" or "cpu"
    major: int
    minor: int
    sm_count: int
    total_vram_mb: float
    free_vram_mb: float
    supports_ampere_tensor_cores: bool
    supports_fp16: bool
    supports_bf16: bool
    supports_custom_cuda: bool
    supports_cuda_graphs: bool
    temperature_c: Optional[float] = None
    utilization_gpu_pct: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AdaptiveTrainingDecision:
    """Auditable evidence-based adaptive training configuration decision."""
    model_name: str
    device: str
    applied_batch_size: int
    applied_precision: str
    gradient_accumulation_steps: int
    effective_batch_size: int
    execution_path: str
    estimated_peak_vram_mb: float
    available_vram_mb: float
    safety_margin_mb: float
    vram_safety_ratio: float
    is_safe: bool
    rationale: str
    adjustments_made: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AdaptiveTrainingEngine:
    """
    Central adaptive decision and resource management engine for AI model training.
    Provides hardware capability classification, pre-flight VRAM estimation,
    safe automatic batch-size & precision selection, and dynamic OOM recovery.
    """

    _instance: Optional[AdaptiveTrainingEngine] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._cache_lock = threading.Lock()
        self._recommendation_cache: Dict[str, AdaptiveTrainingDecision] = {}
        logger.info("AdaptiveTrainingEngine initialized successfully.")

    @staticmethod
    def normalize_model_name(model_name: str) -> str:
        clean = model_name.strip().lower().replace("-", "_")
        return MODEL_ALIASES.get(clean, clean)

    def get_hardware_profile(self, force_cpu: bool = False, device_id: int = 0) -> AdaptiveHardwareProfile:
        """Inspect and return current host hardware capabilities."""
        if force_cpu or not torch.cuda.is_available():
            return AdaptiveHardwareProfile(
                device_name="CPU Host",
                device_type="cpu",
                major=0,
                minor=0,
                sm_count=0,
                total_vram_mb=0.0,
                free_vram_mb=0.0,
                supports_ampere_tensor_cores=False,
                supports_fp16=False,
                supports_bf16=False,
                supports_custom_cuda=False,
                supports_cuda_graphs=False,
                temperature_c=None,
                utilization_gpu_pct=None,
            )

        props = torch.cuda.get_device_properties(device_id)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_id)
        is_ampere = (props.major == 8 and props.minor >= 6) or (props.major > 8)

        # Check native custom kernel availability
        has_custom_cuda = False
        try:
            import custom_kernels
            has_custom_cuda = custom_kernels.is_native_available() and torch.cuda.is_available()
        except Exception:
            has_custom_cuda = False

        # Strictly check BF16 support
        bf16_supported = False
        try:
            bf16_supported = torch.cuda.is_bf16_supported()
        except Exception:
            bf16_supported = False

        # Telemetry
        temp_c: Optional[float] = None
        util_pct: Optional[float] = None
        try:
            from engine.device.hardware_monitor import get_fast_gpu_telemetry
            fast = get_fast_gpu_telemetry(device_id)
            if fast:
                temp_c = fast.get("temperature_c")
                util_pct = fast.get("utilization_gpu_pct")
        except Exception:
            pass

        return AdaptiveHardwareProfile(
            device_name=props.name,
            device_type="cuda",
            major=props.major,
            minor=props.minor,
            sm_count=props.multi_processor_count,
            total_vram_mb=round(total_bytes / (1024**2), 1),
            free_vram_mb=round(free_bytes / (1024**2), 1),
            supports_ampere_tensor_cores=is_ampere,
            supports_fp16=(props.major >= 7),
            supports_bf16=bf16_supported,
            supports_custom_cuda=has_custom_cuda,
            supports_cuda_graphs=(os.name == "nt"),
            temperature_c=temp_c,
            utilization_gpu_pct=util_pct,
        )

    def estimate_training_vram(
        self,
        model_name: str,
        batch_size: int = 32,
        precision: str = "float32",
        seq_len: Optional[int] = None,
        backend: str = "GPU_NATIVE",
        force_cpu: bool = False,
    ) -> Dict[str, Any]:
        """
        Calculates mathematical pre-flight VRAM estimation:
        - Parameter storage (FP32=4 bytes, FP16/BF16=2 bytes)
        - Optimizer states (AdamW=8 bytes/param fp32 moments, SGD=4 bytes/param with momentum)
        - Gradients (same as parameters)
        - Activations (derived from layer dimensions, sequence length, and batch size)
        - PyTorch / cuBLAS runtime workspace (~256MB baseline)
        """
        norm_model = self.normalize_model_name(model_name)
        spec = MODEL_SPECS.get(norm_model, MODEL_SPECS["cifar10_mlp"])
        hw = self.get_hardware_profile(force_cpu=(force_cpu or backend == "CPU_NATIVE"))

        if hw.device_type == "cpu":
            return {
                "model_name": norm_model,
                "batch_size": batch_size,
                "precision": "float32",
                "param_memory_mb": 0.0,
                "optimizer_memory_mb": 0.0,
                "gradient_memory_mb": 0.0,
                "activation_memory_mb": 0.0,
                "workspace_memory_mb": 0.0,
                "total_estimated_vram_mb": 0.0,
                "available_vram_mb": 0.0,
                "headroom_mb": 0.0,
                "is_safe": True,
                "safety_ratio": 0.0,
            }

        num_params = spec["parameters"]
        is_mixed = precision in ("amp", "float16", "fp16", "bfloat16", "bf16")
        bytes_per_param = 2 if is_mixed else 4

        # Parameter and gradient storage
        param_mb = (num_params * bytes_per_param) / (1024**2)
        grad_mb = (num_params * bytes_per_param) / (1024**2)

        # Optimizer memory
        if spec.get("optimizer") == "adamw":
            # AdamW stores 2 x fp32 states (m, v) = 8 bytes per parameter
            optim_mb = (num_params * 8) / (1024**2)
        else:
            # SGD with momentum stores 1 x fp32 state = 4 bytes per parameter
            optim_mb = (num_params * 4) / (1024**2)

        # Activations
        if spec["type"] == "causal_lm":
            s_len = seq_len or spec.get("default_seq_len", 256)
            layers = spec.get("layers", 12)
            hidden = spec.get("hidden_size", 768)
            # Decoder transformer activation memory per token ~ 34 * hidden * layers bytes
            # across self-attention, MLP, norm, and residual paths
            bytes_per_token_act = 34 * hidden * layers * (2 if is_mixed else 4) / 4
            activation_bytes = batch_size * s_len * bytes_per_token_act
            act_mb = activation_bytes / (1024**2)
        else:
            bytes_per_sample = spec.get("bytes_per_sample_act", 36864)
            act_mb = (batch_size * bytes_per_sample * (0.5 if is_mixed else 1.0)) / (1024**2)

        workspace_mb = 256.0  # CUDA context, cuBLASLt workspace, allocator fragmentation headroom
        total_estimated_mb = round(param_mb + optim_mb + grad_mb + act_mb + workspace_mb, 2)
        avail_mb = hw.free_vram_mb if hw.free_vram_mb > 0 else hw.total_vram_mb
        headroom_mb = round(avail_mb - total_estimated_mb, 2)
        safety_ratio = round(total_estimated_mb / max(1.0, avail_mb), 3)

        return {
            "model_name": norm_model,
            "batch_size": batch_size,
            "precision": precision,
            "param_memory_mb": round(param_mb, 2),
            "optimizer_memory_mb": round(optim_mb, 2),
            "gradient_memory_mb": round(grad_mb, 2),
            "activation_memory_mb": round(act_mb, 2),
            "workspace_memory_mb": workspace_mb,
            "total_estimated_vram_mb": total_estimated_mb,
            "available_vram_mb": avail_mb,
            "headroom_mb": headroom_mb,
            "is_safe": (headroom_mb > 0 and safety_ratio <= 0.90),
            "safety_ratio": safety_ratio,
        }

    def select_precision(
        self,
        requested_precision: Optional[str] = None,
        hw: Optional[AdaptiveHardwareProfile] = None,
        force_cpu: bool = False,
    ) -> Tuple[str, str, List[str]]:
        """
        Safely selects supported precision mode:
        - FP32: always supported.
        - FP16/AMP: supported on GPU with Tensor Cores (Ampere SM 8.6). Falls back to FP32 on CPU.
        - BF16: strictly guarded by torch.cuda.is_bf16_supported(). Never selected when unsupported.
        """
        if hw is None:
            hw = self.get_hardware_profile(force_cpu=force_cpu)

        adjustments: List[str] = []
        clean_req = (requested_precision or "").strip().lower()

        # CPU mode
        if hw.device_type == "cpu" or force_cpu:
            if clean_req in ("amp", "float16", "fp16", "bfloat16", "bf16"):
                adjustments.append(f"Precision '{clean_req}' not supported on CPU. Reverted to float32.")
                return "float32", "CPU-only execution requires standard float32 precision.", adjustments
            return "float32", "Standard FP32 precision selected for CPU execution.", adjustments

        # BF16 request
        if clean_req in ("bfloat16", "bf16"):
            if hw.supports_bf16:
                return "bfloat16", "BFloat16 Tensor Core mixed precision verified and active.", adjustments
            else:
                adjustments.append("BFloat16 requested but unsupported by hardware/runtime. Reverted to amp (FP16).")
                return (
                    "amp",
                    "Device does not support BFloat16 natively. Safely fell back to FP16 AMP with Tensor Cores.",
                    adjustments,
                )

        # FP16 / AMP request
        if clean_req in ("amp", "float16", "fp16"):
            if hw.supports_fp16:
                return "amp", "PyTorch Native AMP (FP16) enabled with Ampere SM 8.6 Tensor Core acceleration.", adjustments
            else:
                adjustments.append("FP16 AMP unsupported on device. Reverted to float32.")
                return "float32", "Device lacks Tensor Core FP16 support. Reverted to float32.", adjustments

        # FP32 request
        if clean_req in ("float32", "fp32"):
            return "float32", "Full FP32 precision explicitly requested.", adjustments

        # Auto selection on GPU
        if hw.supports_ampere_tensor_cores:
            return (
                "amp",
                "Automatically selected PyTorch Native AMP (FP16) for optimal Ampere SM 8.6 Tensor Core throughput.",
                adjustments,
            )

        return "float32", "Standard FP32 precision selected by default.", adjustments

    def recommend_training_config(
        self,
        model_name: str,
        backend: str = "GPU_NATIVE",
        requested_batch_size: Optional[int] = None,
        requested_precision: Optional[str] = None,
        seq_len: Optional[int] = None,
        gradient_accumulation_steps: int = 1,
        target_vram_ratio: float = 0.85,
        force_cpu: bool = False,
    ) -> AdaptiveTrainingDecision:
        """
        Calculates optimal, evidence-based training hyperparameters:
        - Selects supported precision (FP32, FP16/AMP, BF16 only where supported).
        - Computes safe batch size based on available VRAM and safety margin.
        - Scales gradient accumulation steps if batch size had to be reduced, preserving effective batch size.
        - Provides full audit trail of adjustments and rationales.
        """
        norm_model = self.normalize_model_name(model_name)
        spec = MODEL_SPECS.get(norm_model, MODEL_SPECS["cifar10_mlp"])
        is_cpu = force_cpu or backend.upper() == "CPU_NATIVE"
        hw = self.get_hardware_profile(force_cpu=is_cpu)

        adjustments: List[str] = []

        # 1. Select Precision
        selected_prec, prec_rationale, prec_adj = self.select_precision(
            requested_precision=requested_precision,
            hw=hw,
            force_cpu=is_cpu,
        )
        adjustments.extend(prec_adj)

        # 2. Select Batch Size and Gradient Accumulation
        applied_batch_size: int
        applied_accum: int = max(1, gradient_accumulation_steps)

        if is_cpu:
            # CPU: memory is system RAM, bound by computational throughput
            default_bs = spec.get("default_batch_size", 32)
            applied_batch_size = requested_batch_size if (requested_batch_size and requested_batch_size > 0) else default_bs
            # Clamp to model valid limits
            applied_batch_size = max(spec.get("min_batch_size", 1), min(spec.get("max_batch_size", 512), applied_batch_size))
            estimated = self.estimate_training_vram(norm_model, applied_batch_size, selected_prec, seq_len, backend, force_cpu=True)
            rationale = f"CPU training path selected. {prec_rationale}"
            exec_path = "standard_cpu"
        else:
            avail_mb = hw.free_vram_mb if hw.free_vram_mb > 0 else hw.total_vram_mb
            budget_mb = avail_mb * target_vram_ratio

            # Candidate candidate batch sizes (power of 2 descending)
            if spec["type"] == "causal_lm":
                candidate_sizes = [16, 8, 4, 2, 1]
            elif spec["type"] == "vision_cnn":
                candidate_sizes = [256, 128, 64, 32, 16]
            else:
                candidate_sizes = [512, 256, 128, 64, 32, 16]

            # If user specified a batch size
            user_bs = requested_batch_size if (requested_batch_size and requested_batch_size > 0) else None

            if user_bs is not None:
                user_est = self.estimate_training_vram(norm_model, user_bs, selected_prec, seq_len, backend)
                if user_est["total_estimated_vram_mb"] <= budget_mb:
                    applied_batch_size = user_bs
                    estimated = user_est
                    rationale = f"Requested batch size {user_bs} fits safely within {budget_mb:.0f} MB VRAM budget ({user_est['total_estimated_vram_mb']:.0f} MB). {prec_rationale}"
                else:
                    # Unsafe: find highest safe batch size
                    safe_bs = spec.get("min_batch_size", 1)
                    safe_est = self.estimate_training_vram(norm_model, safe_bs, selected_prec, seq_len, backend)
                    for c_bs in candidate_sizes:
                        if c_bs <= user_bs:
                            c_est = self.estimate_training_vram(norm_model, c_bs, selected_prec, seq_len, backend)
                            if c_est["total_estimated_vram_mb"] <= budget_mb:
                                safe_bs = c_bs
                                safe_est = c_est
                                break

                    applied_batch_size = safe_bs
                    estimated = safe_est
                    # Automatically double/scale gradient accumulation to preserve effective throughput
                    scale_factor = math.ceil(user_bs / safe_bs)
                    applied_accum = max(applied_accum, scale_factor)
                    adjustments.append(
                        f"Requested batch size {user_bs} exceeded {budget_mb:.0f} MB VRAM budget (estimated {user_est['total_estimated_vram_mb']:.0f} MB). "
                        f"Adaptively reduced batch size to {safe_bs} and scaled gradient accumulation steps to {applied_accum} to preserve effective batch size."
                    )
                    rationale = (
                        f"VRAM safety constraint applied on {hw.device_name}: Reduced batch size from {user_bs} -> {safe_bs} "
                        f"(estimated {estimated['total_estimated_vram_mb']:.0f} MB / {avail_mb:.0f} MB available) with accum={applied_accum}. {prec_rationale}"
                    )
            else:
                # Auto batch size: find largest candidate fitting in budget
                safe_bs = spec.get("min_batch_size", 1)
                safe_est = self.estimate_training_vram(norm_model, safe_bs, selected_prec, seq_len, backend)
                for c_bs in candidate_sizes:
                    c_est = self.estimate_training_vram(norm_model, c_bs, selected_prec, seq_len, backend)
                    if c_est["total_estimated_vram_mb"] <= budget_mb:
                        safe_bs = c_bs
                        safe_est = c_est
                        break

                applied_batch_size = safe_bs
                estimated = safe_est
                adjustments.append(f"Auto-selected optimal batch size {applied_batch_size} based on {budget_mb:.0f} MB VRAM budget.")
                rationale = f"Optimal batch size {applied_batch_size} selected for {hw.device_name} ({estimated['total_estimated_vram_mb']:.0f} MB peak). {prec_rationale}"

            # Execution path
            if backend.upper() == "GPU_CUSTOM" and hw.supports_custom_cuda:
                exec_path = "custom_cuda"
            else:
                exec_path = "optimized_pytorch"

        effective_batch_size = applied_batch_size * applied_accum

        return AdaptiveTrainingDecision(
            model_name=norm_model,
            device=hw.device_name,
            applied_batch_size=applied_batch_size,
            applied_precision=selected_prec,
            gradient_accumulation_steps=applied_accum,
            effective_batch_size=effective_batch_size,
            execution_path=exec_path,
            estimated_peak_vram_mb=estimated["total_estimated_vram_mb"],
            available_vram_mb=estimated["available_vram_mb"],
            safety_margin_mb=estimated["headroom_mb"],
            vram_safety_ratio=estimated["safety_ratio"],
            is_safe=estimated["is_safe"],
            rationale=rationale,
            adjustments_made=adjustments,
        )

    def handle_oom(
        self,
        current_batch_size: int,
        current_accum_steps: int = 1,
        min_batch_size: Optional[int] = None,
        model_name: str = "cifar10_mlp",
    ) -> Optional[Dict[str, Any]]:
        """
        Graceful CUDA Out-Of-Memory Recovery Handler:
        1. Clears PyTorch CUDA memory cache and triggers garbage collection.
        2. Halves the batch size if above min_batch_size.
        3. Doubles gradient accumulation steps to preserve effective batch size.
        4. Preserves all checkpoints on disk without corruption.
        Returns: mitigation details dictionary, or None if batch size cannot be reduced further.
        """
        norm_model = self.normalize_model_name(model_name)
        spec = MODEL_SPECS.get(norm_model, MODEL_SPECS["cifar10_mlp"])
        floor_bs = min_batch_size or spec.get("min_batch_size", 1)

        logger.warning(
            f"OOM condition encountered during training of '{norm_model}' at batch_size={current_batch_size}. Initiating recovery..."
        )

        # 1. Flush memory caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

        # 2. Check if batch size can be reduced
        if current_batch_size <= floor_bs:
            logger.error(
                f"Cannot reduce batch size below minimum floor {floor_bs}. OOM recovery exhausted."
            )
            return None

        # 3. Halve batch size and double accumulation
        new_batch_size = max(floor_bs, current_batch_size // 2)
        new_accum_steps = current_accum_steps * 2
        effective_bs = new_batch_size * new_accum_steps

        mitigation = {
            "status": "RECOVERED",
            "model_name": norm_model,
            "previous_batch_size": current_batch_size,
            "new_batch_size": new_batch_size,
            "previous_accum_steps": current_accum_steps,
            "new_accum_steps": new_accum_steps,
            "effective_batch_size": effective_bs,
            "message": (
                f"Recovered from CUDA OOM: Cleared GPU cache, halved batch size ({current_batch_size} -> {new_batch_size}), "
                f"and doubled gradient accumulation ({current_accum_steps} -> {new_accum_steps}) to maintain {effective_bs} effective batch size."
            ),
        }
        logger.info(mitigation["message"])
        return mitigation
