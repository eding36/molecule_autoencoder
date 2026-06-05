#!/usr/bin/env python3
"""
Modal entrypoint for the MoleculeNet MLP-from-scratch baseline.

Morgan-FP → 3-layer MLP, no pretraining. Provides the canonical MoleculeNet
"Singletask Network" floor — gives an honest "is pretraining helping?" delta
against the AE and Distill backbones evaluated by `modal_mol_struct_ae.py`
and `modal_distillation.py`. Same fine-tuning protocol (DATASET_PROTOCOL
split/metric, best-val epoch, multi-seed mean±std) across all three.

Usage:
    MODAL_PROFILE=<profile> python -m modal run --detach \\
        modal_mlp_baseline.py::moleculenet_finetune_mlp \\
        --datasets BBBP,BACE,HIV,Tox21,ToxCast,SIDER,ClinTox,MUV \\
        --seeds 0,1,2 --epochs 100 --split-override scaffold
"""
from __future__ import annotations

import modal

APP_NAME = "molstructae-mlp-baseline"
REMOTE_DIR = "/root/mol_struct_ae"
RUNS_DIR = "/root/runs"
LOCAL_PROJECT = "/Users/eding36/VSCodeProjects/mol_struct_ae"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", extra_index_url="https://download.pytorch.org/whl/cu121")
    .pip_install(
        "rdkit>=2023.3.1",
        "numpy>=1.24",
        "pandas>=2.0",
        "tqdm>=4.65",
        "scikit-learn>=1.3",
        "deepchem>=2.7",
    )
    .add_local_dir(
        LOCAL_PROJECT,
        remote_path=REMOTE_DIR,
        ignore=["runs", ".cache", "**/__pycache__", "**/*.pyc", ".git", "data/shards"],
    )
)

runs_vol = modal.Volume.from_name("molstructae-runs", create_if_missing=True)


# Mole-BERT Table 1 reference (ROC-AUC, scaffold split, fine-tuning, 10 seeds).
MOLEBERT_REFERENCE = {
    "Tox21": 0.768, "ToxCast": 0.643, "SIDER": 0.628, "ClinTox": 0.789,
    "MUV": 0.786, "HIV": 0.782, "BBBP": 0.719, "BACE": 0.808,
}


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 6,
    volumes={RUNS_DIR: runs_vol},
)
def moleculenet_finetune_mlp_modal(datasets: list, seeds: list,
                                    epochs: int = 100, batch_size: int = 32,
                                    lr: float = 1e-3, dropout: float = 0.5,
                                    train_smiles_path: str = "",
                                    split_override: str = "",
                                    fp_radius: int = 2, fp_nbits: int = 2048,
                                    hidden: int = 512) -> list:
    import sys
    sys.path.insert(0, REMOTE_DIR)
    from utils.mlp_baseline import MLPBackend
    from utils.benchmark_moleculenet import run_finetune_benchmark

    backend = MLPBackend(fp_radius=fp_radius, fp_nbits=fp_nbits, hidden=hidden)
    return run_finetune_benchmark(
        backend, datasets=datasets, seeds=seeds, epochs=epochs,
        batch_size=batch_size, lr=lr, dropout=dropout,
        train_smiles_path=train_smiles_path or None,
        split_override=split_override or None,
    )


@app.local_entrypoint()
def moleculenet_finetune_mlp(datasets: str = "BBBP,BACE,ClinTox,SIDER",
                              seeds: str = "0,1,2",
                              epochs: int = 100, batch_size: int = 32, lr: float = 1e-3,
                              train_smiles_path: str = "",
                              split_override: str = "",
                              fp_radius: int = 2, fp_nbits: int = 2048,
                              hidden: int = 512):
    """MoleculeNet MLP-from-scratch baseline (Morgan FP + 3-layer MLP).

    Same fine-tuning protocol (DATASET_PROTOCOL split/metric, best-val epoch,
    multi-seed mean±std) as the AE / Distill benchmarks — directly comparable.
    No pretraining: fresh MLP weights per seed. Use `--split-override scaffold`
    to force scaffold split across all datasets for fair comparison.
    """
    ds_list = [d.strip() for d in datasets.split(",") if d.strip()]
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    print(f"[moleculenet_finetune_mlp] Morgan r={fp_radius} bits={fp_nbits} hidden={hidden}")
    print(f"  datasets: {ds_list}   seeds: {seed_list}   epochs={epochs}")
    print(f"  split_override: {split_override or '(per-dataset DATASET_PROTOCOL)'}")
    res = moleculenet_finetune_mlp_modal.remote(
        datasets=ds_list, seeds=seed_list,
        epochs=epochs, batch_size=batch_size, lr=lr,
        train_smiles_path=train_smiles_path,
        split_override=split_override,
        fp_radius=fp_radius, fp_nbits=fp_nbits, hidden=hidden,
    )

    print("\n" + "=" * 78)
    print("  MoleculeNet — MLP-FROM-SCRATCH BASELINE (Morgan FP + 3-layer MLP)")
    print("  Same protocol as AE / Distill benchmarks — no pretraining.")
    print("=" * 78)
    print(f"  {'Dataset':<10}  {'Metric':<6}  {'#task':>5}  {'N':>6}  "
           f"{'BASELINE (mean±std)':>20}  {'Mole-BERT':>9}")
    print(f"  {'-'*10}  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*20}  {'-'*9}")
    for r in res:
        ref = MOLEBERT_REFERENCE.get(r["dataset"])
        ref_s = f"{ref:.3f}" if ref is not None else "  —  "
        ours = f"{r['mean']:.4f} ± {r['std']:.4f}"
        print(f"  {r['dataset']:<10}  {r['metric']:<6}  {r['n_tasks']:>5}  "
               f"{r['n_kept']:>6}  {ours:>20}  {ref_s:>9}")
