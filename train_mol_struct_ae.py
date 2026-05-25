"""
Train the Pairformer MolStructAutoencoder on precomputed shards.

Reconstruction losses (10 per-track) + KL regularizer. No contrastive head,
no cross-track-consistency module — both removed as part of the move to the
unified Pairformer backbone.

Run locally (for testing):
    python train_mol_struct_ae.py --shard-dir data/shards --epochs 1 --batch-size 16

Run on Modal: see modal_mol_struct_ae.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mol_struct_ae import MolStructAutoencoder, compute_total_loss
from mol_struct_ae.dataset import ShardedMolDataset, ShardSequentialSampler, make_collate_fn
from mol_struct_ae.losses import LossWeights
from mol_struct_ae.model import MolAEConfig


def cosine_warmup_lr(step: int, warmup_steps: int, total_steps: int,
                     base_lr: float, min_lr_ratio: float = 0.1) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard-dir", required=True)
    p.add_argument("--out-dir", default="runs/zinc250k")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-atoms", type=int, default=64)
    p.add_argument("--max-dihedrals", type=int, default=64)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--latent", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-frac", type=float, default=0.03)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ckpt-every", type=int, default=2000)
    p.add_argument("--amp", action="store_true", help="mixed-precision training")
    p.add_argument("--resume", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ---- Data
    dataset = ShardedMolDataset(args.shard_dir, max_atoms=args.max_atoms)
    sampler = ShardSequentialSampler(dataset, shuffle=True)
    collate_fn = make_collate_fn(args.max_atoms, args.max_dihedrals)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=(args.device == "cuda"), drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    print(f"[data] {len(dataset)} molecules, {len(loader)} steps/epoch")

    # ---- Model
    cfg = MolAEConfig(max_atoms=args.max_atoms, hidden_dim=args.hidden,
                       latent_dim=args.latent)
    model = MolStructAutoencoder(cfg).to(args.device)
    weights = LossWeights()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and args.device == "cuda")

    total_steps = args.epochs * len(loader)
    warmup_steps = max(1, int(args.warmup_frac * total_steps))
    print(f"[train] total_steps={total_steps} warmup_steps={warmup_steps} amp={args.amp}")

    start_step = 0
    if args.resume and Path(args.resume).is_file():
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_step = ckpt.get("step", 0)
        print(f"[train] resumed from step {start_step}")

    model.train()
    step = start_step
    t0 = time.time()
    log_path = out_dir / "train.log"
    log_f = open(log_path, "a")

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for batch in loader:
            batch = batch.to(args.device)
            lr = cosine_warmup_lr(step, warmup_steps, total_steps, args.lr)
            for g in opt.param_groups:
                g["lr"] = lr

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and args.device == "cuda"):
                out = model(batch, sample=True)
                losses = compute_total_loss(out, batch, weights=weights)
                total = losses["total"]

            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            if step % args.log_every == 0:
                terms = {k: float(v.detach().cpu()) for k, v in losses.items()}
                rate = (step - start_step + 1) / max(time.time() - t0, 1e-6)
                msg = (f"[step {step:>6d} ep {epoch}] total={terms['total']:.4f} "
                        f"recon[2d={terms['atom_feat']:.3f}/adj={terms['adj']:.3f}/"
                        f"len={terms['bond_len']:.3f}/ang={terms['bond_ang']:.3f}/"
                        f"chg={terms['partial_charges']:.4f}/"
                        f"tor={terms['torsion']:.3f}/pharma={terms['pharma']:.3f}] "
                        f"kl={terms['kl']:.3f} "
                        f"lr={lr:.2e} rate={rate:.1f}it/s")
                print(msg, flush=True)
                log_f.write(json.dumps({"step": step, "epoch": epoch, "lr": lr, **terms}) + "\n")
                log_f.flush()

            if step > 0 and step % args.ckpt_every == 0:
                ckpt_path = out_dir / f"ckpt_step{step:06d}.pt"
                torch.save({"model": model.state_dict(),
                              "opt": opt.state_dict(), "step": step, "config": vars(args)},
                            ckpt_path)
                print(f"[ckpt] saved {ckpt_path}", flush=True)

            step += 1

        # End-of-epoch checkpoint
        ckpt_path = out_dir / f"ckpt_epoch{epoch:03d}.pt"
        torch.save({"model": model.state_dict(),
                      "opt": opt.state_dict(), "step": step, "config": vars(args)},
                    ckpt_path)
        print(f"[epoch {epoch}] saved {ckpt_path}", flush=True)

    final = out_dir / "final.pt"
    torch.save({"model": model.state_dict(),
                  "opt": opt.state_dict(), "step": step, "config": vars(args)}, final)
    print(f"[done] saved {final}")
    log_f.close()


if __name__ == "__main__":
    main()
