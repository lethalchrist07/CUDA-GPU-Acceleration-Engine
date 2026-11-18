"""
Custom GPT-Style Transformer Package for CUDA GPU Acceleration Engine.
"""

from engine.transformer.config import (
    TransformerConfig,
    ActivationType,
    NormType,
    get_preset_config,
)
from engine.transformer.model import (
    GPTTransformer,
    TransformerBlock,
    CausalSelfAttention,
    TransformerMLP,
)
from engine.transformer.tokenizer import (
    BaseTokenizer,
    CharTokenizer,
    ByteTokenizer,
    BPETokenizer,
)
from engine.transformer.dataset import (
    TextChunkDataset,
    create_train_val_datasets,
    create_dataloaders,
)
from engine.transformer.data_pipeline import (
    DocumentCorpus,
    DatasetManifest,
    MemoryMappedChunkDataset,
    prepare_production_datasets,
)
from engine.transformer.trainer import (
    TransformerTrainer,
    TransformerTrainingConfig,
    get_cosine_schedule_with_warmup,
    safe_perplexity,
)
from engine.transformer.memory_profiler import (
    measure_actual_gpu_memory,
    compute_analytical_memory,
)
from engine.transformer.dataset_pipeline import (
    DatasetExpansionPipeline,
    TextCleaner,
    MinHashDeduplicator,
)
from engine.transformer.scalable_dataset import (
    ScalableMemmapDataset,
    create_scalable_dataloaders,
    seed_worker,
)
from engine.transformer.tokenizer_validator import TokenizerValidator
from engine.transformer.evaluator import (
    ModelEvaluator,
    compute_distinct_ngrams,
    compute_repetition_index,
)
from engine.transformer.adaptive_engine import (
    AdaptiveExecutionEngine,
    ExecutionPath,
    EngineWorkloadType,
    HardwareProfile,
    SelectionDecision,
    CustomFusedTransformerMLP,
)

__all__ = [
    "TransformerConfig",
    "ActivationType",
    "NormType",
    "get_preset_config",
    "GPTTransformer",
    "TransformerBlock",
    "CausalSelfAttention",
    "TransformerMLP",
    "BaseTokenizer",
    "CharTokenizer",
    "ByteTokenizer",
    "BPETokenizer",
    "TextChunkDataset",
    "create_train_val_datasets",
    "create_dataloaders",
    "DocumentCorpus",
    "DatasetManifest",
    "MemoryMappedChunkDataset",
    "prepare_production_datasets",
    "TransformerTrainer",
    "TransformerTrainingConfig",
    "get_cosine_schedule_with_warmup",
    "safe_perplexity",
    "measure_actual_gpu_memory",
    "compute_analytical_memory",
    "DatasetExpansionPipeline",
    "TextCleaner",
    "MinHashDeduplicator",
    "ScalableMemmapDataset",
    "create_scalable_dataloaders",
    "seed_worker",
    "TokenizerValidator",
    "ModelEvaluator",
    "compute_distinct_ngrams",
    "compute_repetition_index",
    "AdaptiveExecutionEngine",
    "ExecutionPath",
    "EngineWorkloadType",
    "HardwareProfile",
    "SelectionDecision",
    "CustomFusedTransformerMLP",
]
