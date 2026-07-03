"""
Builds the distillation model training set:
Run the mol_struct_ae (`MolStructAutoencoder`) over a SMILES library and save the
(SMILES, sim_embed) pairs that the distillation will be trained against.

Two input modes:
  * `--shard-dir` — pre-featurized shards with `.smiles` tags (fast path)
  * `--library`   — CSV → featurize on the fly (slow but always works)

Output: a `.pt` file containing:
    {
      "smiles":   List[str]            length L
      "embeds":   np.ndarray [L, D]    float32, L2-normalized
      "checkpoint": str
      "config":   dict
    }

Usage:
    python scripts/precompute_mol_struct_ae_embeds.py \\
        --checkpoint runs/zinc250k_remote/ckpt_step002000.pt \\
        --shard-dir data/shards \\
        --out runs/mol_struct_ae_embeds.pt
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mol_struct_ae import MolStructAutoencoder
from mol_struct_ae.model import MolAEConfig
from utils.feature_utils import (embed_all, embed_from_shards, featurize_all)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--shard-dir", default="", help="pre-featurized shards (fast)")
    p.add_argument("--library", default="", help="CSV (fallback if no shards)")
    p.add_argument("--smiles-col", default="smiles")
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--shard-start", type=int, default=0,
                    help="for parallel runs: process shard_paths[start::stride]")
    p.add_argument("--shard-stride", type=int, default=1)
    p.add_argument("--amp", action="store_true",
                    help="run forward in bfloat16 autocast (~1.7x faster, lossless for inference)")
    args = p.parse_args()

    if not args.shard_dir and not args.library:
        raise SystemExit("Provide --shard-dir OR --library")

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if args.device == "auto" else torch.device(args.device))
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg_args = ckpt.get("config", {})
    cfg = MolAEConfig(
        max_atoms=cfg_args.get("max_atoms", 48),
        hidden_dim=cfg_args.get("hidden", 96),
        latent_dim=cfg_args.get("latent", 256),
    )
    model = MolStructAutoencoder(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded mol_struct_ae: step {ckpt.get('step', '?')}, max_atoms={cfg.max_atoms}, "
           f"hidden={cfg.hidden_dim}, latent={cfg.latent_dim}\n")

    t0 = time.time()
    if args.shard_dir:
        print(f"Reading shards from {args.shard_dir}")
        smi_to_embed: Dict[str, np.ndarray] = embed_from_shards(
            args.shard_dir, model, device, args.batch_size, cfg.max_atoms,
            shard_start=args.shard_start, shard_stride=args.shard_stride,
            amp=args.amp,
        )
    else:
        print(f"Featurizing CSV {args.library}")
        with open(args.library) as f:
            rows = list(csv.DictReader(f))
        smis = sorted({r[args.smiles_col].strip().strip('"') for r in rows})
        if args.limit > 0:
            smis = smis[:args.limit]
        samples = featurize_all(smis, max_atoms=cfg.max_atoms, workers=args.workers)
        smi_to_embed = embed_all(samples, model, device, args.batch_size, cfg.max_atoms)
    print(f"\n{len(smi_to_embed)} pairs collected in {time.time()-t0:.0f}s")

    # Persist as parallel lists.
    smiles_list = list(smi_to_embed.keys())
    embeds = np.stack([smi_to_embed[s] for s in smiles_list]).astype(np.float32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "smiles": smiles_list,
        "embeds": embeds,
        "checkpoint": args.checkpoint,
        "config": cfg_args,
    }, out, pickle_protocol=4)   # protocol 4+ required for >4GB objects
    size_mb = out.stat().st_size / 2**20
    print(f"Saved → {out} ({size_mb:.1f} MB, embeds shape={embeds.shape})")


if __name__ == "__main__":
    main()
