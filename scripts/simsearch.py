"""
Similarity search using the distillation model.

No featurization at inference time — just tokenize the SMILES and run the
distillation transformer. 

Side-by-side comparison: cosine similarity on distill_model embeddings vs.
Morgan-FP Tanimoto (radius=2, 2048 bits) as the baseline.

Usage:
    python scripts/simsearch.py \\
        --distillation-ckpt runs/distill_remote/best.pt \\
        --vocab runs/vocab.json \\
        --library data/zinc250k.csv \\
        --queries "aspirin:CC(=O)Oc1ccccc1C(=O)O" "caffeine:Cn1cnc2n(C)c(=O)n(C)c(=O)c12" \\
        --top-k 10 \\
        --out runs/simsearch.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from distillation.smiles_encoder import SmilesEncoder, SmilesEncoderConfig
from distillation.smiles_tokenizer import SmilesTokenizer
from utils.feature_utils import morgan_fps, tanimoto_matrix


DEFAULT_QUERIES = [
    ("aspirin",       "CC(=O)Oc1ccccc1C(=O)O"),
    ("ibuprofen",     "CC(C)Cc1ccc(C(C)C(=O)O)cc1"),
    ("caffeine",      "Cn1cnc2n(C)c(=O)n(C)c(=O)c12"),
    ("diphenhydramine", "CN(C)CCOC(c1ccccc1)c1ccccc1"),
    ("sildenafil",    "CCCc1nn(C)c2c1nc(-c1cc(S(=O)(=O)N3CCN(C)CC3)ccc1OCC)[nH]c2=O"),
]


def parse_queries(argv) -> List[Tuple[str, str]]:
    out = []
    for s in argv:
        if ":" in s:
            name, smi = s.split(":", 1)
            out.append((name.strip(), smi.strip()))
        else:
            out.append((s[:24], s))
    return out


@torch.no_grad()
def embed_with_distill(smiles_list: List[str], distill_model: SmilesEncoder,
                        tokenizer: SmilesTokenizer, device: torch.device,
                        batch_size: int = 256) -> np.ndarray:
    """Tokenize + forward through the distillation. Returns [N, D] L2-normalized."""
    distill_model.eval()
    embs = []
    t0 = time.time()
    for i in range(0, len(smiles_list), batch_size):
        chunk = smiles_list[i:i + batch_size]
        ids, mask = tokenizer.encode_batch(chunk)
        ids, mask = ids.to(device), mask.to(device)
        e = distill_model(ids, mask)                                    # already L2-normed
        embs.append(e.cpu())
        if (i // batch_size + 1) % 20 == 0:
            rate = (i + len(chunk)) / max(time.time() - t0, 1e-6)
            print(f"[embed] {i + len(chunk)}/{len(smiles_list)} rate={rate:.0f}/s",
                   flush=True)
    return torch.cat(embs, dim=0).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--distillation-ckpt", required=True)
    p.add_argument("--vocab", required=True)
    p.add_argument("--library", required=True, help="CSV with a SMILES column")
    p.add_argument("--smiles-col", default="smiles")
    p.add_argument("--queries", nargs="*", default=None,
                    help="`name:SMILES` pairs (default: 5 well-known drugs)")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--limit", type=int, default=0, help="cap library size")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if args.device == "auto" else torch.device(args.device))
    print(f"Device: {device}")

    # ---- Tokenizer + distill_model
    tokenizer = SmilesTokenizer.load(args.vocab)
    ckpt = torch.load(args.distillation_ckpt, map_location="cpu", weights_only=False)
    enc_cfg = SmilesEncoderConfig(**ckpt["encoder_config"])
    distill_model = SmilesEncoder(enc_cfg).to(device)
    distill_model.load_state_dict(ckpt["model"])
    distill_model.eval()
    print(f"Distillation loaded: step {ckpt.get('step', '?')} val_loss="
           f"{ckpt.get('val_loss', '?')} hidden={enc_cfg.hidden_dim} "
           f"layers={enc_cfg.num_layers} out_dim={enc_cfg.output_dim}\n")

    # ---- Library SMILES
    with open(args.library) as f:
        rows = list(csv.DictReader(f))
    lib_smiles = [r[args.smiles_col].strip().strip('"') for r in rows]
    if args.limit > 0:
        lib_smiles = lib_smiles[:args.limit]
    lib_smiles = sorted(set(lib_smiles))
    print(f"Library: {len(lib_smiles)} unique SMILES from {args.library}")

    # ---- Embed library + compute Morgan-FP baseline
    lib_embs = embed_with_distill(lib_smiles, distill_model, tokenizer, device,
                                    args.batch_size)
    print(f"Library embedded: {lib_embs.shape}")

    lib_fps = morgan_fps(lib_smiles)
    keep_mask = np.array([s in lib_fps for s in lib_smiles])
    lib_smiles = [s for s, k in zip(lib_smiles, keep_mask) if k]
    lib_embs = lib_embs[keep_mask]
    lib_fp_matrix = np.stack([lib_fps[s] for s in lib_smiles]).astype(np.uint8)
    print(f"After FP filter: {len(lib_smiles)} mols\n")

    # ---- Queries
    queries = parse_queries(args.queries) if args.queries else DEFAULT_QUERIES
    print(f"Queries ({len(queries)}):")
    for n, s in queries:
        print(f"  {n}: {s}")
    print()

    q_smis = [q[1] for q in queries]
    q_embs = embed_with_distill(q_smis, distill_model, tokenizer, device, args.batch_size)
    q_fps = morgan_fps(q_smis)

    # ---- Search
    results = []
    for (name, smi), q_emb in zip(queries, q_embs):
        if smi not in q_fps:
            print(f"[skip] {name} — Morgan FP failed")
            continue
        q_fp = q_fps[smi]
        cos = lib_embs @ q_emb                                      # cosine (L2-normed)
        tan = tanimoto_matrix(q_fp, lib_fp_matrix)
        cos_top = np.argsort(cos)[::-1][:args.top_k]
        tan_top = np.argsort(tan)[::-1][:args.top_k]

        q_result = {
            "query_name": name,
            "query_smiles": smi,
            "model_top": [{"rank": r + 1, "smiles": lib_smiles[i], "cosine": float(cos[i])}
                            for r, i in enumerate(cos_top)],
            "tanimoto_top": [{"rank": r + 1, "smiles": lib_smiles[i], "tanimoto": float(tan[i])}
                              for r, i in enumerate(tan_top)],
        }
        results.append(q_result)

        print(f"=== {name} ({smi}) ===")
        print(f"{'rank':>4}  {'distill SMILES':<48} {'cos':>6}    "
               f"{'Tanimoto SMILES':<48} {'tan':>6}")
        for r in range(args.top_k):
            m = q_result["model_top"][r]
            t = q_result["tanimoto_top"][r]
            print(f"{m['rank']:>4}  {m['smiles']:<48.48} {m['cosine']:>6.3f}    "
                   f"{t['smiles']:<48.48} {t['tanimoto']:>6.3f}")
        print()

    # ---- Save
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "distillation_ckpt": args.distillation_ckpt,
            "library": args.library,
            "library_size": len(lib_smiles),
            "top_k": args.top_k,
            "results": results,
        }, f, indent=2)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
