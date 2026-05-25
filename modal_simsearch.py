#!/usr/bin/env python3
"""
Similarity search on Modal — uses the distilled SMILES-only distillation. No
featurization at inference; just tokenize → distill_model forward → cosine.
Includes Morgan-FP Tanimoto side-by-side as a baseline.

Run:
    MODAL_PROFILE=<profile> python -m modal run --detach modal_simsearch.py

Download results:
    MODAL_PROFILE=<profile> python -m modal run modal_simsearch.py::download_results
"""
from __future__ import annotations

import modal

APP_NAME = "molstructae-simsearch"
REMOTE_DIR = "/root/mol_struct_ae"
RUNS_DIR = "/root/runs"
DISTILLATION_DIR = f"{RUNS_DIR}/distill"
OUT = f"{RUNS_DIR}/simsearch.json"
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


@app.function(
    image=image,
    gpu="A10G",
    cpu=4.0,
    memory=8 * 1024,
    timeout=60 * 60 * 1,
    volumes={RUNS_DIR: runs_vol},
)
def run_simsearch(distillation_ckpt: str = "best.pt", vocab: str = "vocab.json",
                   top_k: int = 10, batch_size: int = 256) -> None:
    import os
    import subprocess
    import sys

    vocab_path = f"{REMOTE_DIR}/runs/{vocab}"
    if not os.path.exists(vocab_path):
        print(f"[simsearch] vocab missing at {vocab_path}; building from ZINC CSV")
        subprocess.run([
            sys.executable, "-u", f"{REMOTE_DIR}/scripts/build_vocab.py",
            "--csv", f"{REMOTE_DIR}/data/zinc250k.csv",
            "--out", vocab_path,
        ], check=True, cwd=REMOTE_DIR)

    cmd = [
        sys.executable, "-u", f"{REMOTE_DIR}/scripts/simsearch.py",
        "--distillation-ckpt", f"{DISTILLATION_DIR}/{distillation_ckpt}",
        "--vocab", vocab_path,
        "--library", f"{REMOTE_DIR}/data/zinc250k.csv",
        "--out", OUT,
        "--top-k", str(top_k),
        "--batch-size", str(batch_size),
    ]
    print(f"[simsearch] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REMOTE_DIR)
    runs_vol.commit()


@app.local_entrypoint()
def main(distillation_ckpt: str = "best.pt", top_k: int = 10):
    print(f"Launching simsearch on Modal — distillation_ckpt={distillation_ckpt} top_k={top_k}")
    run_simsearch.spawn(distillation_ckpt=distillation_ckpt, top_k=top_k)


@app.local_entrypoint()
def download_results():
    import os
    dest = f"{LOCAL_PROJECT}/runs/zinc250k_remote"
    os.makedirs(dest, exist_ok=True)
    out_local = f"{dest}/simsearch.json"
    with open(out_local, "wb") as fout:
        for chunk in runs_vol.read_file("simsearch.json"):
            fout.write(chunk)
    print(f"Saved → {out_local}")
