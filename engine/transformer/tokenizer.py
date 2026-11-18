"""
Modular Tokenizer for Transformer Language Modeling.

Provides deterministic, self-contained tokenizers:
1. CharTokenizer: Character-level vocabulary with explicit special tokens.
2. ByteTokenizer: Lossless UTF-8 byte-level tokenizer with zero OOV rate.
3. BPETokenizer: Byte-Pair Encoding subword tokenizer (GPT-2 style) with zero OOV rate,
   reproducible pair merges, and configurable vocabulary size.

All tokenizers support:
- Explicit special tokens: <pad>=0, <unk>=1, <bos>=2, <eos>=3
- JSON serialization and deserialization
- Corrupted character / input validation
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union


class BaseTokenizer:
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"

    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3

    def __init__(self):
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        raise NotImplementedError

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        raise NotImplementedError

    def save(self, filepath: Union[str, Path]) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "type": self.__class__.__name__,
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
            "special_tokens": {
                "pad": self.PAD_ID,
                "unk": self.UNK_ID,
                "bos": self.BOS_ID,
                "eos": self.EOS_ID,
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "BaseTokenizer":
        path = Path(filepath)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        tokenizer_type = data.get("type", "CharTokenizer")
        if tokenizer_type == "ByteTokenizer":
            tok = ByteTokenizer()
            tok.token_to_id = data["token_to_id"]
            tok.id_to_token = {int(v): k for k, v in tok.token_to_id.items()}
            return tok
        elif tokenizer_type == "BPETokenizer":
            tok = BPETokenizer()
            tok.token_to_id = data["token_to_id"]
            tok.id_to_token = {int(v): k for k, v in tok.token_to_id.items()}
            raw_merges = data.get("merges", [])
            tok.merges = {tuple(m["pair"]): m["new_id"] for m in raw_merges}
            tok.vocab_bytes = {int(k): bytes(v) for k, v in data.get("vocab_bytes", {}).items()}
            return tok
        else:
            tok = CharTokenizer()
            tok.token_to_id = data["token_to_id"]
            tok.id_to_token = {int(v): k for k, v in tok.token_to_id.items()}
            return tok


class CharTokenizer(BaseTokenizer):
    """
    Character-level tokenizer extracted from a training corpus.
    Guarantees reproducible character mapping and explicit special token IDs.
    """

    def __init__(self, vocab_chars: Optional[List[str]] = None):
        super().__init__()
        self.token_to_id = {
            self.PAD_TOKEN: self.PAD_ID,
            self.UNK_TOKEN: self.UNK_ID,
            self.BOS_TOKEN: self.BOS_ID,
            self.EOS_TOKEN: self.EOS_ID,
        }

        if vocab_chars:
            unique_chars = sorted(list(set(vocab_chars)))
            curr_id = 4
            for ch in unique_chars:
                if ch not in self.token_to_id:
                    self.token_to_id[ch] = curr_id
                    curr_id += 1

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    @classmethod
    def train_from_text(cls, text: str) -> "CharTokenizer":
        if not text:
            raise ValueError("Cannot train tokenizer from empty text.")
        chars = list(set(text))
        return cls(chars)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text)}")
        
        tokens: List[int] = []
        if add_special_tokens:
            tokens.append(self.BOS_ID)

        for ch in text:
            tokens.append(self.token_to_id.get(ch, self.UNK_ID))

        if add_special_tokens:
            tokens.append(self.EOS_ID)

        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        chars: List[str] = []
        specials = {self.PAD_ID, self.UNK_ID, self.BOS_ID, self.EOS_ID}
        
        for tid in token_ids:
            if skip_special_tokens and tid in specials:
                continue
            chars.append(self.id_to_token.get(tid, self.UNK_TOKEN))
        return "".join(chars)


class ByteTokenizer(BaseTokenizer):
    """
    Lossless UTF-8 byte-level tokenizer.
    Vocab size: 4 special tokens + 256 byte values = 260 total tokens.
    Guarantees 0% out-of-vocabulary rate on arbitrary UTF-8 text.
    """

    def __init__(self):
        super().__init__()
        self.token_to_id = {
            self.PAD_TOKEN: self.PAD_ID,
            self.UNK_TOKEN: self.UNK_ID,
            self.BOS_TOKEN: self.BOS_ID,
            self.EOS_TOKEN: self.EOS_ID,
        }
        for b in range(256):
            token_str = f"<byte_{b:02x}>"
            self.token_to_id[token_str] = 4 + b
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text)}")
        
        raw_bytes = text.encode("utf-8")
        tokens: List[int] = []
        if add_special_tokens:
            tokens.append(self.BOS_ID)

        for b in raw_bytes:
            tokens.append(4 + b)

        if add_special_tokens:
            tokens.append(self.EOS_ID)

        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        byte_list: List[int] = []
        specials = {self.PAD_ID, self.UNK_ID, self.BOS_ID, self.EOS_ID}
        
        for tid in token_ids:
            if tid in specials:
                if not skip_special_tokens:
                    byte_list.extend(self.id_to_token.get(tid, "").encode("utf-8"))
                continue
            byte_val = tid - 4
            if 0 <= byte_val < 256:
                byte_list.append(byte_val)

        return bytes(byte_list).decode("utf-8", errors="replace")


class BPETokenizer(BaseTokenizer):
    """
    Byte-Pair Encoding (BPE) subword tokenizer built on UTF-8 bytes.
    
    Properties:
    - Base vocabulary: 4 special tokens + 256 byte tokens = 260 base tokens.
    - Subsequent tokens: Learned merge pairs of most frequent byte sequences.
    - Zero out-of-vocabulary rate on arbitrary text or code.
    - Compact sequence compression for Transformer context efficiency.
    - Deterministic encoding via priority rank merges.
    """

    def __init__(self):
        super().__init__()
        self.token_to_id = {
            self.PAD_TOKEN: self.PAD_ID,
            self.UNK_TOKEN: self.UNK_ID,
            self.BOS_TOKEN: self.BOS_ID,
            self.EOS_TOKEN: self.EOS_ID,
        }
        self.vocab_bytes: Dict[int, bytes] = {}
        for b in range(256):
            tid = 4 + b
            token_str = f"<byte_{b:02x}>"
            self.token_to_id[token_str] = tid
            self.vocab_bytes[tid] = bytes([b])
        
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        # Map: (token_id_1, token_id_2) -> merged_token_id
        self.merges: Dict[Tuple[int, int], int] = {}

    @classmethod
    def train_from_text(
        cls,
        text: str,
        target_vocab_size: int = 1024,
        min_frequency: int = 2,
    ) -> "BPETokenizer":
        """
        Trains BPE tokenizer on text until target_vocab_size is reached or no pairs exceed min_frequency.
        """
        if not text:
            raise ValueError("Cannot train BPE tokenizer from empty text.")

        tok = cls()
        if target_vocab_size <= 260:
            return tok

        # Initial sequence of byte token IDs
        raw_bytes = text.encode("utf-8")
        tokens: List[int] = [4 + b for b in raw_bytes]

        curr_id = 260
        num_merges = target_vocab_size - 260

        for _ in range(num_merges):
            if len(tokens) < 2:
                break

            # Count all adjacent token pairs
            pair_counts: Counter = Counter()
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] += 1

            if not pair_counts:
                break

            best_pair, best_freq = pair_counts.most_common(1)[0]
            if best_freq < min_frequency:
                break

            # Register new merged token
            new_id = curr_id
            curr_id += 1
            tok.merges[best_pair] = new_id
            
            # Reconstruct byte representation
            merged_bytes = tok.vocab_bytes[best_pair[0]] + tok.vocab_bytes[best_pair[1]]
            tok.vocab_bytes[new_id] = merged_bytes
            
            # String representation
            try:
                repr_str = merged_bytes.decode("utf-8")
            except UnicodeDecodeError:
                repr_str = f"<bpe_{new_id}>"
            
            tok.token_to_id[repr_str] = new_id
            tok.id_to_token[new_id] = repr_str

            # Apply merge to current sequence
            new_tokens: List[int] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tok

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text)}")

        if not text:
            return [self.BOS_ID, self.EOS_ID] if add_special_tokens else []

        # If no merges exist, return base byte tokens
        if not self.merges:
            tokens = [4 + b for b in text.encode("utf-8")]
            if add_special_tokens:
                tokens = [self.BOS_ID] + tokens + [self.EOS_ID]
            return tokens

        # Precompute rank mapping if not cached
        if not hasattr(self, "_rank_map") or len(self._rank_map) != len(self.merges):
            self._rank_map = {pair: rank for rank, pair in enumerate(self.merges.keys())}
            self._word_cache: Dict[bytes, List[int]] = {}

        # Split text into regex tokens: words, whitespace, punctuation
        import re
        parts = re.findall(r"\w+|[^\w\s]|\s+", text)
        tokens: List[int] = []

        for part in parts:
            p_bytes = part.encode("utf-8")
            cached = self._word_cache.get(p_bytes)
            if cached is not None:
                tokens.extend(cached)
                continue

            word = [4 + b for b in p_bytes]
            while len(word) >= 2:
                # Find all adjacent pairs in current word that exist in learned merges
                valid = []
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    rank = self._rank_map.get(pair)
                    if rank is not None:
                        valid.append((rank, pair))
                if not valid:
                    break

                # Merge the pair with the earliest learned priority rank
                _, best_pair = min(valid, key=lambda x: x[0])
                new_id = self.merges[best_pair]
                new_word: List[int] = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                        new_word.append(new_id)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                word = new_word

            # Cache word tokens
            if len(self._word_cache) < 200000:
                self._word_cache[p_bytes] = word
            tokens.extend(word)

        if add_special_tokens:
            tokens = [self.BOS_ID] + tokens + [self.EOS_ID]

        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        byte_list: List[int] = []
        specials = {self.PAD_ID, self.UNK_ID, self.BOS_ID, self.EOS_ID}

        for tid in token_ids:
            if tid in specials:
                if not skip_special_tokens:
                    byte_list.extend(self.id_to_token.get(tid, "").encode("utf-8"))
                continue
            
            chunk = self.vocab_bytes.get(tid)
            if chunk is not None:
                byte_list.extend(chunk)
            elif 4 <= tid < 260:
                byte_list.append(tid - 4)

        return bytes(byte_list).decode("utf-8", errors="replace")

    def save(self, filepath: Union[str, Path]) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        merges_list = [{"pair": list(pair), "new_id": new_id} for pair, new_id in self.merges.items()]
        vocab_bytes_dict = {str(k): list(v) for k, v in self.vocab_bytes.items()}
        data = {
            "type": self.__class__.__name__,
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
            "special_tokens": {
                "pad": self.PAD_ID,
                "unk": self.UNK_ID,
                "bos": self.BOS_ID,
                "eos": self.EOS_ID,
            },
            "merges": merges_list,
            "vocab_bytes": vocab_bytes_dict,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
