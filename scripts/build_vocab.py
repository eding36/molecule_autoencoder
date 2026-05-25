"""
Build the SMILES vocabulary used by the distillation tokenizer/encoder.

    python scripts/build_vocab.py --csv data/zinc250k.csv --out runs/vocab.json
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from distillation.smiles_tokenizer import (SmilesTokenizer,
                                                build_vocab_from_smiles, tokenize)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--smiles-col", default="smiles")
    p.add_argument("--out", required=True)
    p.add_argument("--max-len", type=int, default=128,
                    help="Tokenizer max sequence length (truncate beyond this).")
    p.add_argument("--min-freq", type=int, default=1,
                    help="Drop tokens that appear fewer than this many times.")
    args = p.parse_args()

    with open(args.csv) as f:
        reader = csv.DictReader(f)
        smiles_list = [r[args.smiles_col].strip().strip('"') for r in reader]
    print(f"[vocab] read {len(smiles_list)} SMILES from {args.csv}")

    vocab = build_vocab_from_smiles(smiles_list, min_freq=args.min_freq)
    tok = SmilesTokenizer(vocab, max_len=args.max_len)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out))

    # Coverage stats — what fraction of mols fit under max_len?
    over = 0
    max_seen = 0
    for s in smiles_list[:50_000]:
        L = len(tokenize(s)) + 1                        # +1 for CLS
        max_seen = max(max_seen, L)
        if L > args.max_len:
            over += 1
    print(f"[vocab] size={len(vocab)} max_len={args.max_len} "
           f"max_seen_len={max_seen} oversized@50k={over}")
    print(f"[vocab] saved → {out}")


if __name__ == "__main__":
    main()
