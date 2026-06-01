#!/usr/bin/env python3
"""
Distillation pipeline on Modal.

Three-step parallel pipeline:
  1. `precompute_partition_modal` — spawned N× in parallel; each handles every
     Nth shard with bf16 autocast inference, writes its own partition file.
  2. `merge_pairs_modal`         — CPU-only, concatenates the N partitions
     into one (smiles, sim_embed) pairs file.
  3. `train_distill_modal`       — trains the SmilesEncoder via cosine
     distillation against the merged targets.

Entrypoint: `main_parallel` orchestrates all three. `resume_merge_and_train`
skips Stage 1 if partitions already exist on the volume.

Download the trained checkpoint when done:
    MODAL_PROFILE=<profile> python -m modal run \\
        modal_distillation.py::download_results
"""
from __future__ import annotations

import modal

APP_NAME = "molstructae-distill"
REMOTE_DIR = "/root/mol_struct_ae"
RUNS_DIR = "/root/runs"
PAIRS_OUT = f"{RUNS_DIR}/mol_struct_ae_embeds.pt"
DISTILLATION_DIR = f"{RUNS_DIR}/distill"
LOCAL_PROJECT = "/Users/eding36/VSCodeProjects/mol_struct_ae"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", extra_index_url="https://download.pytorch.org/whl/cu121")
    .pip_install(
        "rdkit>=2023.3.1",
        "numpy>=1.24",
        "scikit-learn>=1.3",
        "tqdm>=4.65",
    )
    .add_local_dir(
        LOCAL_PROJECT,
        remote_path=REMOTE_DIR,
        ignore=["runs", ".cache", "**/__pycache__", "**/*.pyc", ".git", "data/shards"],
    )
)

runs_vol = modal.Volume.from_name("molstructae-runs", create_if_missing=True)


# ── Per-partition precompute: 1 GPU container handles shard_paths[start::stride]
@app.function(
    image=image, gpu="A10G", cpu=4.0, memory=16 * 1024,
    timeout=60 * 60 * 24,
    volumes={RUNS_DIR: runs_vol},
)
def precompute_partition_modal(
    ckpt_path: str, shard_dir: str, partition_out: str,
    shard_start: int, shard_stride: int, batch_size: int = 256, amp: bool = True,
) -> str:
    """Run precompute on a subset of shards (every Nth, offset by start)."""
    import subprocess, sys
    cmd = [
        sys.executable, "-u", f"{REMOTE_DIR}/scripts/precompute_mol_struct_ae_embeds.py",
        "--checkpoint", ckpt_path,
        "--shard-dir", shard_dir,
        "--out", partition_out,
        "--batch-size", str(batch_size),
        "--shard-start", str(shard_start),
        "--shard-stride", str(shard_stride),
    ]
    if amp:
        cmd.append("--amp")
    print(f"[partition {shard_start}/{shard_stride}] running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REMOTE_DIR)
    runs_vol.commit()
    return partition_out


# ── Merge per-partition outputs into a single pairs file (CPU only)
@app.function(
    image=image, cpu=4.0, memory=16 * 1024,
    timeout=60 * 30, volumes={RUNS_DIR: runs_vol},
)
def merge_pairs_modal(partition_paths: list, merged_out: str) -> str:
    import numpy as np, torch
    smiles_all, embeds_all = [], []
    runs_vol.reload()
    for p in partition_paths:
        d = torch.load(p, map_location="cpu", weights_only=False)
        smiles_all.extend(d["smiles"])
        embeds_all.append(d["embeds"])
        print(f"[merge] loaded {p}: {len(d['smiles'])} pairs")
    embeds = np.concatenate(embeds_all, axis=0)
    print(f"[merge] writing {len(smiles_all)} pairs → {merged_out}")
    torch.save({"smiles": smiles_all, "embeds": embeds}, merged_out, pickle_protocol=4)
    runs_vol.commit()
    return merged_out


# ── Stage 2: train the SmilesEncoder via cosine distillation
@app.function(
    image=image, gpu="A10G", cpu=4.0, memory=16 * 1024,
    timeout=60 * 60 * 12,
    volumes={RUNS_DIR: runs_vol},
)
def train_distill_modal(
    pairs_path: str, epochs: int = 20, batch_size: int = 256,
    hidden: int = 256, num_layers: int = 6, num_heads: int = 8, lr: float = 3e-4,
) -> None:
    import os, subprocess, sys
    vocab_path = f"{REMOTE_DIR}/runs/vocab.json"
    if not os.path.exists(vocab_path):
        subprocess.run([
            sys.executable, "-u", f"{REMOTE_DIR}/scripts/build_vocab.py",
            "--csv", f"{REMOTE_DIR}/data/zinc250k.csv", "--out", vocab_path,
        ], check=True, cwd=REMOTE_DIR)
    os.makedirs(DISTILLATION_DIR, exist_ok=True)
    subprocess.run([
        sys.executable, "-u", f"{REMOTE_DIR}/train_distillation.py",
        "--pairs", pairs_path, "--vocab", vocab_path,
        "--out-dir", DISTILLATION_DIR,
        "--epochs", str(epochs), "--batch-size", str(batch_size),
        "--hidden", str(hidden), "--num-layers", str(num_layers),
        "--num-heads", str(num_heads), "--lr", str(lr),
    ], check=True, cwd=REMOTE_DIR)
    runs_vol.commit()


@app.local_entrypoint()
def main_parallel(
    ckpt_path: str = "",                       # absolute path on volume
    shard_dir: str = "",                       # absolute shard dir on volume
    pairs_out: str = "",                       # absolute output path for merged pairs
    n_partitions: int = 4,                     # parallel precompute workers
    epochs: int = 20, batch_size: int = 256,
    hidden: int = 256, num_layers: int = 6, lr: float = 3e-4,
):
    """Parallel distillation pipeline.

    Stage 1: spawn `n_partitions` A10G containers, each handles every Nth
    shard with AMP/bf16 inference (~4× wall-time win on Stage 1).
    Stage 2: merge per-partition outputs on CPU, then train distillation on
    one A10G container.

    Total wall-time: roughly Stage1/N + Stage2 (≈ 1.5h + 5h ≈ 6.5h for 5M
    pairs at N=4, vs ~15h for the serial pipeline).
    """
    if not ckpt_path:
        raise SystemExit("--ckpt-path required for main_parallel")
    if not shard_dir:
        raise SystemExit("--shard-dir required for main_parallel")
    pairs_out = pairs_out or PAIRS_OUT

    # Per-partition output paths (in same dir as merged output)
    base, ext = (pairs_out.rsplit(".", 1) + ["pt"])[:2]
    partition_paths = [f"{base}_part{i}of{n_partitions}.{ext}"
                        for i in range(n_partitions)]

    print(f"[main_parallel] {n_partitions}-way parallel precompute")
    print(f"  ckpt   : {ckpt_path}")
    print(f"  shards : {shard_dir}")
    print(f"  merged : {pairs_out}")

    # Stage 1 — spawn N parallel partitions, collect outputs.
    futures = [
        precompute_partition_modal.spawn(
            ckpt_path=ckpt_path, shard_dir=shard_dir,
            partition_out=partition_paths[i],
            shard_start=i, shard_stride=n_partitions,
            batch_size=batch_size, amp=True,
        )
        for i in range(n_partitions)
    ]
    print(f"[main_parallel] spawned {n_partitions} precompute partitions; waiting…")
    for i, f in enumerate(futures):
        print(f"  partition {i}: done → {f.get()}")

    # Stage 1.5 — merge.
    print(f"[main_parallel] merging partitions → {pairs_out}")
    merge_pairs_modal.remote(partition_paths=partition_paths, merged_out=pairs_out)

    # Stage 2 — train.
    print(f"[main_parallel] launching Stage 2 (train_distill)")
    train_distill_modal.spawn(
        pairs_path=pairs_out, epochs=epochs, batch_size=batch_size,
        hidden=hidden, num_layers=num_layers, lr=lr,
    )


@app.local_entrypoint()
def resume_merge_and_train(
    pairs_out: str = "",
    n_partitions: int = 4,
    epochs: int = 20, batch_size: int = 256,
    hidden: int = 256, num_layers: int = 6, lr: float = 3e-4,
):
    """Resume: skip Stage 1 (assumes partition files already on volume), run
    merge + Stage 2 training. Use when a previous main_parallel run completed
    Stage 1 partitions but the merge or Stage 2 failed.
    """
    pairs_out = pairs_out or PAIRS_OUT
    base, ext = (pairs_out.rsplit(".", 1) + ["pt"])[:2]
    partition_paths = [f"{base}_part{i}of{n_partitions}.{ext}"
                        for i in range(n_partitions)]
    print(f"[resume] merging {n_partitions} partitions → {pairs_out}")
    merge_pairs_modal.remote(partition_paths=partition_paths, merged_out=pairs_out)
    print(f"[resume] launching Stage 2 (train_distill)")
    train_distill_modal.spawn(
        pairs_path=pairs_out, epochs=epochs, batch_size=batch_size,
        hidden=hidden, num_layers=num_layers, lr=lr,
    )


# ── MoleculeNet fine-tuning on the distillation SmilesEncoder
@app.function(
    image=image, gpu="A10G", cpu=4.0, memory=16 * 1024,
    timeout=60 * 60 * 6,
    volumes={RUNS_DIR: runs_vol},
)
def moleculenet_finetune_distill_modal(
    distill_ckpt: str, vocab_path: str, datasets: list, seeds: list,
    epochs: int = 100, batch_size: int = 32, lr: float = 1e-3,
    dropout: float = 0.5, train_smiles_path: str = "",
    split_override: str = "",
) -> list:
    import os, sys, subprocess
    sys.path.insert(0, REMOTE_DIR)
    from utils.benchmark_moleculenet import DistillBackend, run_finetune_benchmark

    # Vocab fallback: `runs/` is excluded from the image mount, so we rebuild
    # from the bundled CSV if the file is missing on the container fs.
    if not os.path.exists(vocab_path):
        print(f"[ft-distill] vocab not found at {vocab_path}; building from CSV")
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        subprocess.run([
            sys.executable, "-u", f"{REMOTE_DIR}/scripts/build_vocab.py",
            "--csv", f"{REMOTE_DIR}/data/zinc250k.csv", "--out", vocab_path,
        ], check=True, cwd=REMOTE_DIR)

    backend = DistillBackend(distill_ckpt, vocab_path)
    return run_finetune_benchmark(
        backend, datasets=datasets, seeds=seeds, epochs=epochs,
        batch_size=batch_size, lr=lr, dropout=dropout,
        train_smiles_path=train_smiles_path or None,
        split_override=split_override or None,
    )


# Mole-BERT Table 1 reference numbers — same as in modal_mol_struct_ae.py.
MOLEBERT_REFERENCE = {
    "Tox21": 0.768, "ToxCast": 0.643, "SIDER": 0.628, "ClinTox": 0.789,
    "MUV": 0.786, "HIV": 0.782, "BBBP": 0.719, "BACE": 0.808,
}


@app.local_entrypoint()
def moleculenet_finetune_distill(
    distill_ckpt: str = f"{DISTILLATION_DIR}/best.pt",
    vocab_path: str = f"{REMOTE_DIR}/runs/vocab.json",
    datasets: str = "BBBP,BACE,SIDER,Tox21",
    seeds: str = "0,1,2",
    epochs: int = 100, batch_size: int = 32, lr: float = 1e-3,
    train_smiles_path: str = "",
    split_override: str = "",
):
    """Fine-tune the SMILES-only distillation encoder on MoleculeNet.

    Identical protocol to `modal_mol_struct_ae.py::moleculenet_finetune` (Hu
    et al. / Mole-BERT: end-to-end fine-tuning, best-val epoch, mean±std).
    Per-dataset split/metric come from `utils.benchmark_moleculenet.DATASET_PROTOCOL`
    (scaffold + ROC-AUC for BBBP/BACE/HIV; random + ROC-AUC for Tox21/ToxCast/
    SIDER/ClinTox; random + AUPRC for MUV). Uses the distillation
    `SmilesEncoder` as backbone — no RDKit featurization needed, ~100× faster
    per molecule than the AE backend; same fine-tuning protocol means numbers
    are directly comparable to the AE benchmark.
    """
    ds_list = [d.strip() for d in datasets.split(",") if d.strip()]
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    print(f"[moleculenet_finetune_distill] ckpt={distill_ckpt}")
    print(f"  vocab={vocab_path}   datasets={ds_list}   seeds={seed_list}")
    print(f"  split_override={split_override or '(per-dataset DATASET_PROTOCOL)'}")
    res = moleculenet_finetune_distill_modal.remote(
        distill_ckpt=distill_ckpt, vocab_path=vocab_path,
        datasets=ds_list, seeds=seed_list,
        epochs=epochs, batch_size=batch_size, lr=lr,
        train_smiles_path=train_smiles_path,
        split_override=split_override or None,
    )
    print("\n" + "=" * 78)
    print("  MoleculeNet — DISTILL SmilesEncoder END-TO-END FINE-TUNING")
    print("  (same protocol as the AE benchmark; no RDKit featurization)")
    print("=" * 78)
    print(f"  {'Dataset':<10}  {'Metric':<6}  {'N':>6}  {'OURS (mean±std)':>20}  {'Mole-BERT':>9}")
    for r in res:
        ref = MOLEBERT_REFERENCE.get(r["dataset"])
        ref_s = f"{ref:.3f}" if ref else "  —  "
        print(f"  {r['dataset']:<10}  {r['metric']:<6}  {r['n_kept']:>6}  "
               f"{r['mean']:.4f} ± {r['std']:.4f}  {ref_s:>9}")
    return res


@app.local_entrypoint()
def download_results():
    import os
    dest = f"{LOCAL_PROJECT}/runs/distill_remote"
    os.makedirs(dest, exist_ok=True)
    for entry in runs_vol.listdir("distill"):
        name = entry.path.split("/")[-1]
        if name.endswith((".pt", ".json", ".log")):
            with open(f"{dest}/{name}", "wb") as fout:
                for chunk in runs_vol.read_file(f"distill/{name}"):
                    fout.write(chunk)
            print(f"Downloaded: {name}")
    print(f"Saved to {dest}")
