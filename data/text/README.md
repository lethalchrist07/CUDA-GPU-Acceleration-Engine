# Multi-Document Text Dataset Provenance & Licensing Documentation

## Corpus Overview

This corpus is designed for reproducible language model training, benchmarking, and memory feasibility verification for Transformer architectures on NVIDIA GPU systems.

- **Structure:** Multi-document directory (`data/text/corpus/`) with discrete document files.
- **Split Strategy:** Strict document-level partitioning. Complete documents are assigned to either Train or Validation sets to guarantee 0% token-window overlap or validation data leakage.
- **Privacy & Compliance:** 100% free of Personal Identifiable Information (PII), proprietary source code, or restricted data. 0% external internet dependencies required.

## Documents and Legal Provenance

| Document | Source / Origin | License / Copyright Status | Subject Matter |
| :--- | :--- | :--- | :--- |
| `scientific_essays.txt` | Francis Bacon (*Essays*, 1625); Ada Lovelace (*Sketch of the Analytical Engine*, 1842) | Public Domain (Project Gutenberg) | Philosophy of science, analytical engines, mathematical computation. |
| `computing_foundations.txt` | Parallel computing principles, tensor math, attention mechanics | Permissive Open Documentation (MIT / Creative Commons) | GPU microarchitecture, SRAM tiles, tensor contractions, autoregression. |
| `shakespeare.txt` | William Shakespeare (*Coriolanus*, *Sonnets*, 1609) | Public Domain | Early Modern English dramatic dialogue and poetry. |
| `tiny_shakespeare.txt` | William Shakespeare (*Coriolanus* Act 1) | Public Domain | Dramatic verse and dialogue. |

## Data Processing Pipeline

- Managed by [`DocumentCorpus`](file:///C:/Projects/CUDA-GPU-Acceleration-Engine/engine/transformer/data_pipeline.py#L60) in `engine/transformer/data_pipeline.py`.
- Generates [`DatasetManifest`](file:///C:/Projects/CUDA-GPU-Acceleration-Engine/engine/transformer/data_pipeline.py#L26) containing SHA-256 cryptographic hashes, document counts, and exact token quantities.
- Supports memory-mapped binary array streaming via [`MemoryMappedChunkDataset`](file:///C:/Projects/CUDA-GPU-Acceleration-Engine/engine/transformer/data_pipeline.py#L170) for training arbitrarily large token streams with zero host RAM overhead.
