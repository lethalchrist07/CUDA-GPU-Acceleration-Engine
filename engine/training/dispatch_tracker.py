"""
Phase 10 Stage 2 Milestone 4: Kernel Dispatch Tracker
Instruments and accounts for every invocation of compiled custom C++/CUDA kernels
during forward and backward training passes.
"""
from typing import Dict, Any, List
from contextlib import contextmanager
import custom_kernels as ck


TRACKED_OPS = [
    "vector_add",
    "vector_add_backward",
    "vector_scale",
    "vector_scale_backward",
    "relu",
    "relu_backward",
    "sigmoid",
    "sigmoid_backward",
    "gelu",
    "gelu_backward",
    "fused_add_relu",
    "fused_add_relu_backward",
    "fused_bias_activation",
    "fused_bias_activation_backward",
    "matmul_tiled",
    "matmul_wmma",
    "reduce_sum",
    "reduce_max",
]


class KernelDispatchTracker:
    """
    Context manager and monitoring interceptor for custom CUDA kernel execution.
    Counts exact invocations of compiled forward and backward C++/CUDA functions.
    """

    def __init__(self):
        self.counts: Dict[str, int] = {op: 0 for op in TRACKED_OPS}
        self.native_fallbacks: Dict[str, int] = {
            "cross_entropy_loss": 0,
            "data_loading": 0,
            "optimizer_step": 0,
        }
        self._original_fns: Dict[str, Any] = {}
        self._is_active = False

    def record_fallback(self, op_name: str, count: int = 1):
        if op_name in self.native_fallbacks:
            self.native_fallbacks[op_name] += count
        else:
            self.native_fallbacks[op_name] = count

    def start(self):
        if self._is_active:
            return
        mod = ck._get_module()
        self._original_fns.clear()

        for op in TRACKED_OPS:
            if hasattr(mod, op):
                orig_fn = getattr(mod, op)
                self._original_fns[op] = orig_fn

                def make_wrapper(op_name: str, fn: Any):
                    def wrapper(*args, **kwargs):
                        self.counts[op_name] += 1
                        return fn(*args, **kwargs)
                    return wrapper

                setattr(mod, op, make_wrapper(op, orig_fn))

        self._is_active = True

    def stop(self):
        if not self._is_active:
            return
        mod = ck._get_module()
        for op, orig_fn in self._original_fns.items():
            setattr(mod, op, orig_fn)
        self._original_fns.clear()
        self._is_active = False

    def reset(self):
        for op in self.counts:
            self.counts[op] = 0
        for op in self.native_fallbacks:
            self.native_fallbacks[op] = 0

    def get_summary(self) -> Dict[str, Any]:
        mod = ck._get_module()
        is_fallback = getattr(mod, "is_fallback", False)
        is_native = getattr(mod, "is_native", not is_fallback)

        forward_dispatches = {
            op: count for op, count in self.counts.items()
            if not op.endswith("_backward") and count > 0
        }
        backward_dispatches = {
            op: count for op, count in self.counts.items()
            if op.endswith("_backward") and count > 0
        }
        total_custom_forward = sum(forward_dispatches.values())
        total_custom_backward = sum(backward_dispatches.values())

        return {
            "custom_forward_dispatches": forward_dispatches,
            "custom_backward_dispatches": backward_dispatches,
            "total_custom_forward": total_custom_forward,
            "total_custom_backward": total_custom_backward,
            "total_custom_dispatches": total_custom_forward + total_custom_backward,
            "native_fallbacks": {k: v for k, v in self.native_fallbacks.items() if v > 0},
            "is_custom_dispatched": (total_custom_forward + total_custom_backward) > 0 and not is_fallback,
            "is_native_custom": is_native and (total_custom_forward + total_custom_backward) > 0,
            "is_fallback_active": is_fallback,
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
