"""
Train the SMILES-only `SmilesEncoder` to match the mol_struct_ae's `sim_embed` via
cosine distillation.

Inputs:
  * `--pairs runs/mol_struct_ae_embeds.pt` — file produced by
    `scripts/precompute_mol_struct_ae_embeds.py` (parallel SMILES + embed arrays).
  * `--vocab runs/vocab.json` — produced by `scripts/build_vocab.py`.

Loss: `1 - cosine(distillation_embed, mol_struct_ae_embed)`, both L2-normalized. Simple, stable, and
gives a direct interpretation (0 = perfect match, 2 = maximally opposite).

Saves a checkpoint that pairs with the tokenizer for SMILES-only inference.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from distillation.smiles_encoder import SmilesEncoder, SmilesEncoderConfig
from distillation.smiles_tokenizer import SmilesTokenizer


def cosine_warmup_lr(step: int, warmup_steps: int, total_steps: int,
                      base_lr: float, min_lr_ratio: float = 0.05) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    p = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    p = min(max(p, 0.0), 1.0)
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True, help="mol_struct_ae_embeds.pt")
    p.add_argument("--vocab", required=True)
    p.add_argument("--out-dir", default="runs/distill")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-frac", type=float, default=0.03)
    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=2000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ---- Data
    blob = torch.load(args.pairs, map_location="cpu", weights_only=False)
    smiles_list: List[str] = blob["smiles"]
    mol_ae_embeds: np.ndarray = blob["embeds"]                                  # [L, D]
    embed_dim = mol_ae_embeds.shape[1]
    print(f"[data] {len(smiles_list)} pairs, embed_dim={embed_dim}")

    # Train/val split (fixed; deterministic shuffle)
    idx = np.arange(len(smiles_list))
    rng = np.random.RandomState(args.seed); rng.shuffle(idx)
    n_val = max(1, int(len(idx) * args.val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    print(f"[data] train={len(train_idx)} val={len(val_idx)}")

    # ---- Tokenizer & model
    tok = SmilesTokenizer.load(args.vocab)
    if tok.max_len != args.max_len:
        # Allow override (the tokenizer pads to its own max_len; we re-init).
        tok = SmilesTokenizer(tok.vocab, max_len=args.max_len)
    cfg = SmilesEncoderConfig(
        vocab_size=tok.vocab_size, hidden_dim=args.hidden,
        num_layers=args.num_layers, num_heads=args.num_heads,
        dropout=args.dropout, max_len=args.max_len, output_dim=embed_dim,
    )
    model = SmilesEncoder(cfg).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params:,} hidden={cfg.hidden_dim} layers={cfg.num_layers}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)

    n_train = len(train_idx)
    steps_per_epoch = max(1, n_train // args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = max(1, int(args.warmup_frac * total_steps))
    print(f"[train] steps/epoch={steps_per_epoch} total={total_steps} warmup={warmup_steps}")

    def make_batch(indices: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        smis = [smiles_list[i] for i in indices]
        ids, mask = tok.encode_batch(smis)
        tgt = torch.from_numpy(mol_ae_embeds[indices]).float()
        # Re-normalize defensively in case mol_struct_ae embeds weren't unit-norm
        tgt = F.normalize(tgt, dim=-1)
        return ids.to(args.device), mask.to(args.device), tgt.to(args.device)

    step = 0
    t0 = time.time()
    log_path = out_dir / "train.log"
    log_f = open(log_path, "a")
    best_val = float("inf")

    for epoch in range(args.epochs):
        # Shuffle each epoch
        epoch_idx = train_idx.copy(); rng.shuffle(epoch_idx)
        for off in range(0, n_train, args.batch_size):
            batch_idx = epoch_idx[off:off + args.batch_size]
            if len(batch_idx) < 2:
                continue
            ids, mask, tgt = make_batch(batch_idx)

            lr = cosine_warmup_lr(step, warmup_steps, total_steps, args.lr)
            for g in opt.param_groups:
                g["lr"] = lr

            model.train()
            opt.zero_grad(set_to_none=True)
            pred = model(ids, mask)                                # already L2-normed
            loss = (1.0 - (pred * tgt).sum(dim=-1)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % args.log_every == 0:
                rate = (step + 1) / max(time.time() - t0, 1e-6)
                msg = (f"[step {step:>6d} ep {epoch}] cos_loss={float(loss):.4f} "
                        f"lr={lr:.2e} rate={rate:.1f}it/s")
                print(msg, flush=True)
                log_f.write(json.dumps({"step": step, "epoch": epoch, "lr": lr,
                                          "loss": float(loss)}) + "\n")
                log_f.flush()

            if step > 0 and step % args.ckpt_every == 0:
                ckpt_path = out_dir / f"ckpt_step{step:06d}.pt"
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                              "step": step, "config": vars(args),
                              "encoder_config": vars(cfg)}, ckpt_path)
                print(f"[ckpt] saved {ckpt_path}")

            step += 1

        # ---- Validation (cosine error on held-out set)
        model.eval()
        with torch.no_grad():
            val_losses = []
            for off in range(0, len(val_idx), args.batch_size):
                vi = val_idx[off:off + args.batch_size]
                ids, mask, tgt = make_batch(vi)
                pred = model(ids, mask)
                val_losses.append(float((1.0 - (pred * tgt).sum(dim=-1)).mean()))
        val_loss = float(np.mean(val_losses))
        print(f"[epoch {epoch}] val_cos_loss={val_loss:.4f}")
        log_f.write(json.dumps({"epoch_end": epoch, "val_loss": val_loss}) + "\n")
        log_f.flush()
        if val_loss < best_val:
            best_val = val_loss
            best_path = out_dir / "best.pt"
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                          "step": step, "config": vars(args),
                          "encoder_config": vars(cfg), "val_loss": val_loss}, best_path)
            print(f"[best] val={val_loss:.4f} saved → {best_path}")

    final = out_dir / "final.pt"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                  "config": vars(args), "encoder_config": vars(cfg),
                  "val_loss": val_loss}, final)
    print(f"[done] final saved → {final}")
    log_f.close()


if __name__ == "__main__":
    main()
