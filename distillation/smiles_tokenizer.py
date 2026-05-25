"""
SMILES tokenizer for the distillation encoder.

Regex-based atom-wise tokenization (the convention used by ChemBERTa, MolBERT,
Regression-Transformer). Splits SMILES into atomic / bond / numeric / bracket
tokens — handles two-letter elements (`Cl`, `Br`) and bracketed atoms (`[NH+]`).

Vocab structure:
    0: <pad>      padding
    1: <cls>      classifier head pool position
    2: <unk>      out-of-vocab token
    3..: data tokens, sorted by frequency descending

Built once with `scripts/build_vocab.py` and persisted to JSON. The same vocab
is shared between training and inference.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch


# Regex pattern: bracketed atoms, two-letter elements, single-letter atoms,
# stereo markers, bond symbols, ring numbers, parentheses, dot.
SMILES_TOKEN_RE = re.compile(
    r"(\[[^\]]+\]"             # bracketed atom: [N+], [C@H], etc.
    r"|Br|Cl"                  # two-letter halogens
    r"|@@?"                    # chirality
    r"|/|\\"                   # double-bond stereo
    r"|=|#|-|\+|:"             # bond symbols
    r"|\(|\)|\."               # parens and disconnects
    r"|%\d{2}"                 # two-digit ring numbers (e.g. %10)
    r"|[0-9]"                  # single-digit ring numbers
    r"|[A-Za-z])"              # single-letter atoms
)


PAD_TOKEN = "<pad>"
CLS_TOKEN = "<cls>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
CLS_ID = 1
UNK_ID = 2


def tokenize(smiles: str) -> List[str]:
    return SMILES_TOKEN_RE.findall(smiles)


class SmilesTokenizer:
    def __init__(self, vocab: Dict[str, int], max_len: int = 128):
        self.vocab = vocab
        self.inv_vocab = {i: t for t, i in vocab.items()}
        self.max_len = max_len
        # Sanity: special-token ids must match the constants above.
        assert vocab.get(PAD_TOKEN) == PAD_ID, "vocab broken: missing/wrong <pad>"
        assert vocab.get(CLS_TOKEN) == CLS_ID, "vocab broken: missing/wrong <cls>"
        assert vocab.get(UNK_TOKEN) == UNK_ID, "vocab broken: missing/wrong <unk>"

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, smiles: str, pad_to_max: bool = True
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (ids, attn_mask). attn_mask: True for real tokens."""
        toks = tokenize(smiles)
        # Prepend <cls>; truncate if needed (leave room for cls)
        toks = toks[: self.max_len - 1]
        ids = [CLS_ID] + [self.vocab.get(t, UNK_ID) for t in toks]
        attn = [True] * len(ids)
        if pad_to_max:
            while len(ids) < self.max_len:
                ids.append(PAD_ID)
                attn.append(False)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(attn, dtype=torch.bool)

    def encode_batch(self, smiles_list: List[str]
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batched encode, all padded to `max_len`. Returns ([B, L], [B, L])."""
        ids_list, mask_list = zip(*[self.encode(s, pad_to_max=True) for s in smiles_list])
        return torch.stack(ids_list, dim=0), torch.stack(mask_list, dim=0)

    def decode(self, ids: torch.Tensor) -> str:
        """Best-effort detokenize (debug only — not guaranteed to round-trip
        through RDKit because we don't insert disambiguating brackets)."""
        out = []
        for i in ids.tolist():
            if i == PAD_ID:
                break
            if i in (CLS_ID, UNK_ID):
                continue
            out.append(self.inv_vocab.get(i, "?"))
        return "".join(out)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"vocab": self.vocab, "max_len": self.max_len}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SmilesTokenizer":
        with open(path) as f:
            d = json.load(f)
        return cls(vocab=d["vocab"], max_len=d.get("max_len", 128))


def build_vocab_from_smiles(smiles_list: List[str], min_freq: int = 1) -> Dict[str, int]:
    """Iterate every SMILES, tally token frequencies, build a {token → id}
    dict sorted by frequency descending (most common first)."""
    from collections import Counter
    counts: Counter = Counter()
    for s in smiles_list:
        for tok in tokenize(s):
            counts[tok] += 1
    sorted_toks = [t for t, c in counts.most_common() if c >= min_freq]
    vocab: Dict[str, int] = {PAD_TOKEN: PAD_ID, CLS_TOKEN: CLS_ID, UNK_TOKEN: UNK_ID}
    for tok in sorted_toks:
        vocab[tok] = len(vocab)
    return vocab
