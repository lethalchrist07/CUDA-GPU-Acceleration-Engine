"""
Modular Decoder-Only GPT-Style Transformer Architecture.

Implemented in pure native PyTorch (CUDA & CPU execution).
Features:
- Token and learned positional embeddings
- Causal multi-head self-attention with strict autoregressive masking
- Pre-LayerNorm residual architecture
- Configurable MLP with GELU
- Optional weight tying between token embedding and LM projection head
- Deterministic initialization with GPT-2 residual scaling
- Activation checkpointing support for memory conservation
- Autoregressive text generation (greedy & temperature/top-k sampling)
- Comprehensive parameter counting
"""

import math
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from engine.transformer.config import TransformerConfig, ActivationType, NormType


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


def get_norm_layer(norm_type: NormType, dim: int, eps: float = 1e-5) -> nn.Module:
    if norm_type == NormType.RMS_NORM:
        return RMSNorm(dim, eps=eps)
    return nn.LayerNorm(dim, eps=eps)


def get_activation_fn(act_type: ActivationType):
    if act_type == ActivationType.GELU_TANH:
        return lambda x: F.gelu(x, approximate="tanh")
    elif act_type == ActivationType.RELU:
        return F.relu
    elif act_type == ActivationType.SILU:
        return F.silu
    return F.gelu


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with strict autoregressive masking.
    
    Supports both PyTorch scaled_dot_product_attention (Flash/SDPA) and
    explicit manual attention matrix computation for mathematical inspection.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        self.attn_dropout = config.attn_dropout

        # Combined Q, K, V projection
        self.c_attn = nn.Linear(self.d_model, 3 * self.d_model, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        
        # Regularization
        self.attn_drop = nn.Dropout(self.attn_dropout)
        self.resid_drop = nn.Dropout(self.dropout)

        # Persistent causal mask: lower-triangular 1s, upper-triangular 0s
        # Registered as buffer so it moves with device but is not a trainable parameter
        self.register_buffer(
            "bias_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool)).view(
                1, 1, config.max_seq_len, config.max_seq_len
            ),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        use_sdpa: bool = True,
        return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            use_sdpa: If True, uses F.scaled_dot_product_attention when possible
            return_attn_weights: If True, forces manual computation to return attention weights
        Returns:
            Tuple of (output tensor, optional attention weights)
        """
        B, T, C = x.size()

        # Project to Q, K, V
        qkv = self.c_attn(x)  # (B, T, 3 * d_model)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention: (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        attn_weights = None

        if use_sdpa and not return_attn_weights:
            # Native PyTorch SDPA (dispatches to FlashAttention-2 / Memory-Efficient / C++ kernel)
            dropout_p = self.attn_dropout if self.training else 0.0
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True,
            )
        else:
            # Explicit manual attention with strict causal masking
            # (B, n_head, T, head_dim) @ (B, n_head, head_dim, T) -> (B, n_head, T, T)
            scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            
            # Apply causal mask: mask positions where bias_mask is False (upper triangle)
            mask = self.bias_mask[:, :, :T, :T]
            scores = scores.masked_fill(~mask, float("-inf"))
            
            probs = F.softmax(scores, dim=-1)
            probs = self.attn_drop(probs)
            if return_attn_weights:
                attn_weights = probs
            y = torch.matmul(probs, v)  # (B, n_head, T, head_dim)

        # Re-assemble head dimensions: (B, T, d_model)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Final output projection
        y = self.resid_drop(self.c_proj(y))
        return y, attn_weights


class TransformerMLP(nn.Module):
    """Feed-forward network with configurable expansion ratio and activation."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.act = get_activation_fn(config.activation)
        self.c_proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.c_fc(x))
        h = self.c_proj(h)
        return self.dropout(h)


class TransformerBlock(nn.Module):
    """
    Standard Pre-LayerNorm Transformer Decoder Block.
    x = x + Attention(LN1(x))
    x = x + MLP(LN2(x))
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln_1 = get_norm_layer(config.norm_type, config.d_model, config.layer_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = get_norm_layer(config.norm_type, config.d_model, config.layer_norm_eps)
        self.mlp = TransformerMLP(config)
        self.use_gradient_checkpointing = config.use_gradient_checkpointing

    def forward(
        self,
        x: torch.Tensor,
        use_sdpa: bool = True,
        return_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Pre-LN Self-Attention with residual connection
        normed_1 = self.ln_1(x)
        attn_out, weights = self.attn(normed_1, use_sdpa=use_sdpa, return_attn_weights=return_attn_weights)
        x = x + attn_out

        # Pre-LN MLP with residual connection
        normed_2 = self.ln_2(x)
        x = x + self.mlp(normed_2)
        return x, weights


class GPTTransformer(nn.Module):
    """
    Complete GPT-Style Decoder-Only Autoregressive Transformer.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embeddings = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        # Transformer Decoder Stack
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])

        # Final Layer Normalization
        self.ln_f = get_norm_layer(config.norm_type, config.d_model, config.layer_norm_eps)

        # Output LM Head Projection
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight Tying (Weight sharing between token embedding and LM head)
        if config.tie_weights:
            self.lm_head.weight = self.token_embeddings.weight

        # Initialize all weights deterministically
        self.apply(self._init_weights)

        # GPT-2 style scaled initialization for residual projection layers
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=config.initializer_range / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            if hasattr(module, "weight") and module.weight is not None:
                torch.nn.init.ones_(module.weight)

    def count_parameters(self) -> Dict[str, Any]:
        """
        Accurate parameter counting accounting for tied vs untied embeddings.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        token_emb_params = self.token_embeddings.weight.numel()
        pos_emb_params = self.position_embeddings.weight.numel()
        emb_params = token_emb_params + pos_emb_params

        # If weights are tied, lm_head shares token_embeddings storage
        # so non_embedding_params = total - embedding
        non_embedding_params = total_params - emb_params

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_embedding_parameters": non_embedding_params,
            "token_embedding_parameters": token_emb_params,
            "position_embedding_parameters": pos_emb_params,
            "total_embedding_parameters": emb_params,
            "tied_weights": self.config.tie_weights,
            "d_model": self.config.d_model,
            "n_layer": self.config.n_layer,
            "n_head": self.config.n_head,
            "vocab_size": self.config.vocab_size,
            "max_seq_len": self.config.max_seq_len,
        }

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        use_sdpa: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        Args:
            idx: LongTensor of shape (B, T) token indices
            targets: Optional LongTensor of shape (B, T) target next-token indices
            use_sdpa: Use PyTorch SDPA causal attention
        Returns:
            Tuple of (logits, loss). If targets is None, loss is None.
        """
        device = idx.device
        B, T = idx.size()
        if T > self.config.max_seq_len:
            raise ValueError(
                f"Cannot forward sequence of length {T}, max_seq_len is {self.config.max_seq_len}"
            )

        # Positions tensor: [0, 1, ..., T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)

        # Forward embeddings
        tok_emb = self.token_embeddings(idx)  # (B, T, d_model)
        pos_emb = self.position_embeddings(pos)  # (1, T, d_model)
        x = self.drop(tok_emb + pos_emb)

        # Forward Transformer blocks
        for block in self.blocks:
            if self.config.use_gradient_checkpointing and self.training:
                # Wrap block in activation checkpointing
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(inputs[0], use_sdpa=inputs[1])[0]
                    return custom_forward

                x = checkpoint(create_custom_forward(block), x, use_sdpa, use_reentrant=False)
            else:
                x, _ = block(x, use_sdpa=use_sdpa)

        # Final LayerNorm
        x = self.ln_f(x)  # (B, T, d_model)

        # LM Head Projection
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten to compute cross-entropy loss over all sequence positions
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.config.pad_token_id,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation via greedy or temperature/top-k/top-p sampling.
        Args:
            idx: Tensor of shape (B, T) token indices representing prompt
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (1.0 = normal, <1.0 = sharper, 0.0 = greedy)
            top_k: Optional top-k filter for sampling
            top_p: Optional top-p nucleus filter for sampling (0.0 < top_p < 1.0)
            eos_token_id: Stop generation if all batches generate this token
            generator: Optional torch.Generator for deterministic, thread-isolated sampling
        Returns:
            Tensor of shape (B, T + max_new_tokens)
        """
        self.eval()
        eos_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        for _ in range(max_new_tokens):
            # Crop to context window if needed
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            
            logits, _ = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :]  # (B, vocab_size)

            if temperature == 0.0:
                # Greedy decoding
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-5)
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1, generator=generator)

            idx = torch.cat((idx, idx_next), dim=1)

            if eos_id is not None and (idx_next == eos_id).all():
                break

        return idx
