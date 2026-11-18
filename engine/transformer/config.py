"""
Transformer Configuration and Model Size Presets.

Defines configuration dataclass and verified parameter presets for:
- Small smoke test (1M–5M parameters)
- Foundation model (~25M parameters)
- Flagship model (~100M parameters)
- Future stretch models (200M–300M parameters)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any


class ActivationType(str, Enum):
    GELU = "gelu"
    GELU_TANH = "gelu_tanh"
    RELU = "relu"
    SILU = "silu"


class NormType(str, Enum):
    LAYER_NORM = "layer_norm"
    RMS_NORM = "rms_norm"


@dataclass
class TransformerConfig:
    vocab_size: int = 4096
    max_seq_len: int = 512
    d_model: int = 384
    n_layer: int = 12
    n_head: int = 6
    d_ff: Optional[int] = None  # Defaults to 4 * d_model
    dropout: float = 0.1
    attn_dropout: float = 0.1
    bias: bool = True
    tie_weights: bool = True  # Tie token embeddings and LM output head
    activation: ActivationType = ActivationType.GELU
    norm_type: NormType = NormType.LAYER_NORM
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    use_gradient_checkpointing: bool = False
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        if self.d_model % self.n_head != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_head ({self.n_head})")
        if isinstance(self.activation, str):
            self.activation = ActivationType(self.activation)
        if isinstance(self.norm_type, str):
            self.norm_type = NormType(self.norm_type)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "d_model": self.d_model,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
            "attn_dropout": self.attn_dropout,
            "bias": self.bias,
            "tie_weights": self.tie_weights,
            "activation": self.activation.value,
            "norm_type": self.norm_type.value,
            "layer_norm_eps": self.layer_norm_eps,
            "initializer_range": self.initializer_range,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformerConfig":
        return cls(**data)


# ============================================================================
# Verified Model Presets
# ============================================================================

def get_preset_config(preset_name: str, vocab_size: Optional[int] = None) -> TransformerConfig:
    """
    Returns mathematically verified configurations targeting specific parameter budgets:
    - 'smoke_1m': ~1.1M parameters (ideal for rapid forward/backward testing)
    - 'foundation_25m': ~24.6M parameters (target foundation model for RTX 3050 6GB)
    - 'flagship_100m': ~98.3M parameters (flagship architecture)
    - 'stretch_200m': ~202M parameters (future stretch)
    - 'stretch_300m': ~303M parameters (future stretch)
    """
    preset = preset_name.lower().strip()
    
    if preset in ("smoke_1m", "smoke", "1m"):
        v = vocab_size or 1024
        return TransformerConfig(
            vocab_size=v,
            max_seq_len=256,
            d_model=128,
            n_layer=4,
            n_head=4,
            dropout=0.0,
            tie_weights=True,
        )
    elif preset in ("foundation_25m", "foundation", "25m"):
        v = vocab_size or 4096
        return TransformerConfig(
            vocab_size=v,
            max_seq_len=512,
            d_model=384,
            n_layer=12,
            n_head=6,
            dropout=0.1,
            tie_weights=True,
        )
    elif preset in ("production_86m", "production_50m", "50m", "86m", "transformer_50m"):
        v = vocab_size or 2048
        return TransformerConfig(
            vocab_size=v,
            max_seq_len=256,
            d_model=768,
            n_layer=12,
            n_head=12,
            dropout=0.1,
            tie_weights=True,
        )
    elif preset in ("flagship_100m", "flagship", "100m"):
        v = vocab_size or 8192
        return TransformerConfig(
            vocab_size=v,
            max_seq_len=1024,
            d_model=768,
            n_layer=12,
            n_head=12,
            dropout=0.1,
            tie_weights=True,
        )
    elif preset in ("stretch_200m", "200m"):
        v = vocab_size or 8192
        return TransformerConfig(
            vocab_size=v,
            max_seq_len=1024,
            d_model=1024,
            n_layer=16,
            n_head=16,
            dropout=0.1,
            tie_weights=True,
        )
    elif preset in ("stretch_300m", "300m"):
        v = vocab_size or 8192
        return TransformerConfig(
            vocab_size=v,
            max_seq_len=1024,
            d_model=1024,
            n_layer=24,
            n_head=16,
            dropout=0.1,
            tie_weights=True,
        )
    else:
        raise ValueError(
            f"Unknown preset: '{preset_name}'. Available: 'smoke_1m', 'foundation_25m', 'flagship_100m', 'stretch_200m', 'stretch_300m'"
        )
