#!/usr/bin/env python3
"""
Distillation pipeline on Modal:
  1. precompute_mol_struct_ae_embeds_modal — run the trained MolStructAutoencoder
     over the shards on the volume, dump (smiles, sim_embed) pairs.
  2. train_distillation_modal           — train the SmilesEncoder distillation to match.

Both run on a single A10G. Total job time ≈ 30 min on ~43k molecules.

Run end-to-end:
    MODAL_PROFILE=<profile> python -m modal run --detach modal_distillation.py

Download distillation checkpoint when done:
    MODAL_PROFILE=<profile> python -m modal run \\
        modal_distillation.py::download_results
"""
from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "molstructae-distill"
REMOTE_DIR = "/root/mol_struct_ae"
RUNS_DIR = "/root/runs"
CKPT_DIR = f"{RUNS_DIR}/zinc250k"
PAIRS_OUT = f"{RUNS_DIR}/mol_struct_ae_embeds.pt"
DISTILLATION_DIR = f"{RUNS_DIR}/distill"
LOCAL_PROJECT = "/Users/eding36/VSCodeProjects/mol_struct_ae"
LOCAL_CKPT = f"{LOCAL_PROJECT}/runs/zinc250k_remote/ckpt_step002000.pt"
LOCAL_VOCAB = f"{LOCAL_PROJECT}/runs/vocab.json"

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


@app.function(
    image=image, gpu="A10G", cpu=4.0, memory=16 * 1024,
    timeout=60 * 60 * 3,
    volumes={RUNS_DIR: runs_vol},
)
def pipeline_modal(
    ckpt_name: str = "ckpt_step002000.pt",
    epochs: int = 20,
    batch_size: int = 256,
    hidden: int = 256,
    num_layers: int = 6,
    num_heads: int = 8,
    lr: float = 3e-4,
) -> None:
    import os
    import subprocess
    import sys

    ckpt_path = f"{CKPT_DIR}/{ckpt_name}"
    vocab_path = f"{REMOTE_DIR}/runs/vocab.json"           # uploaded with add_local_dir
    if not os.path.exists(vocab_path):
        # Fallback: build it on the fly from the bundled CSV.
        print(f"[pipeline] vocab not found at {vocab_path}; building from CSV")
        subprocess.run([
            sys.executable, "-u", f"{REMOTE_DIR}/scripts/build_vocab.py",
            "--csv", f"{REMOTE_DIR}/data/zinc250k.csv",
            "--out", vocab_path,
        ], check=True, cwd=REMOTE_DIR)

    # ---- Stage 1: precompute mol_struct_ae embeds from shards on the volume
    print(f"[pipeline] STAGE 1: precompute mol_struct_ae embeddings")
    subprocess.run([
        sys.executable, "-u", f"{REMOTE_DIR}/scripts/precompute_mol_struct_ae_embeds.py",
        "--checkpoint", ckpt_path,
        "--shard-dir", f"{RUNS_DIR}/shards",
        "--out", PAIRS_OUT,
        "--batch-size", "128",
    ], check=True, cwd=REMOTE_DIR)
    runs_vol.commit()

    # ---- Stage 2: train the distillation
    print(f"\n[pipeline] STAGE 2: train SMILES-only distillation")
    os.makedirs(DISTILLATION_DIR, exist_ok=True)
    subprocess.run([
        sys.executable, "-u", f"{REMOTE_DIR}/train_distillation.py",
        "--pairs", PAIRS_OUT,
        "--vocab", vocab_path,
        "--out-dir", DISTILLATION_DIR,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--hidden", str(hidden),
        "--num-layers", str(num_layers),
        "--num-heads", str(num_heads),
        "--lr", str(lr),
    ], check=True, cwd=REMOTE_DIR)
    runs_vol.commit()


def _upload_if_missing(local_path: str, remote_rel: str) -> None:
    """Idempotently upload a local file to the volume."""
    p = Path(local_path)
    if not p.is_file():
        raise FileNotFoundError(f"local file not found: {local_path}")
    print(f"[upload] {p.name} → volume:{remote_rel}")
    with runs_vol.batch_upload(force=True) as upload:
        upload.put_file(str(p), remote_rel)


@app.local_entrypoint()
def main(
    epochs: int = 20,
    batch_size: int = 256,
    hidden: int = 256,
    num_layers: int = 6,
    lr: float = 3e-4,
):
    """Upload mol_struct_ae checkpoint to the volume (if local copy exists), then
    spawn the GPU pipeline."""
    if Path(LOCAL_CKPT).is_file():
        _upload_if_missing(LOCAL_CKPT, "zinc250k/ckpt_step002000.pt")
    else:
        print(f"[main] WARN local ckpt not at {LOCAL_CKPT}; expecting it on volume already")

    print("Launching distillation pipeline on Modal (A10G GPU)")
    print(f"  Stage 1: precompute mol_struct_ae embeds from /root/runs/shards/")
    print(f"  Stage 2: train SmilesEncoder ({num_layers}L × {hidden}d) for {epochs}ep")
    pipeline_modal.spawn(
        epochs=epochs, batch_size=batch_size, hidden=hidden,
        num_layers=num_layers, lr=lr,
    )


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
