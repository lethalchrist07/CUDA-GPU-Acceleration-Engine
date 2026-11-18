"""
Execute Full Forensic BPE Tokenizer Validation.
Phase 13 Milestone 1: Produces results/tokenizer_validation_report.json.
"""

import json
import logging
import sys
from pathlib import Path

# Ensure repository root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from engine.transformer import BPETokenizer
from engine.transformer.tokenizer_validator import TokenizerValidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TokenizerAudit")


def run_tokenizer_audit():
    tok_path = PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_campaign" / "tokenizer" / "bpe_vocab.json"
    if not tok_path.is_file():
        raise FileNotFoundError(f"Active tokenizer not found at: {tok_path}")

    logger.info(f"Loading active BPE tokenizer from: {tok_path}")
    tokenizer = BPETokenizer.load(tok_path)
    logger.info(f"Loaded tokenizer with vocabulary size: {tokenizer.vocab_size}")

    validator = TokenizerValidator(tokenizer)
    ckpt_path = PROJECT_ROOT / "checkpoints" / "transformer" / "pretraining_campaign" / "best_model.pt"

    logger.info("Executing full tokenizer audit...")
    report = validator.run_full_audit(checkpoint_path=ckpt_path)

    out_file = PROJECT_ROOT / "results" / "tokenizer_validation_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("=" * 70)
    logger.info("TOKENIZER VALIDATION REPORT SUMMARY:")
    logger.info(f"  Vocabulary Size:            {report['vocab_size']}")
    logger.info(f"  Roundtrip Passed:           {report['roundtrip_validation']['passed']}")
    logger.info(f"  Overall Bytes/Token:        {report['compression_efficiency']['overall_bytes_per_token']}")
    logger.info(f"  Overall Chars/Token:        {report['compression_efficiency']['overall_chars_per_token']}")
    logger.info(f"  Encode Throughput:          {report['throughput_benchmark']['encode_tokens_per_sec']:,} tok/s ({report['throughput_benchmark']['encode_mb_per_sec']} MB/s)")
    logger.info(f"  Decode Throughput:          {report['throughput_benchmark']['decode_tokens_per_sec']:,} tok/s ({report['throughput_benchmark']['decode_mb_per_sec']} MB/s)")
    logger.info(f"  Checkpoint Compatibility:   {report['checkpoint_compatibility']['compatible']}")
    logger.info(f"  Checkpoint Embedding Shape: {report['checkpoint_compatibility']['embedding_matrix_shape']}")
    logger.info(f"Report saved to: {out_file}")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_tokenizer_audit()
