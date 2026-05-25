#!/usr/bin/env python3
"""
Modal training for MolStructAutoencoder on ZINC250k.

Two-phase pipeline, both attached to a single Modal volume so the expensive
featurization is done exactly once:

  1. `precompute_features`  (CPU, parallel RDKit featurization → sharded .pt)
  2. `train_mol_struct_ae_modal`       (GPU, mixed-precision training over the shards)

Run featurization once (it caches into the `molstructae-runs` volume):
    MODAL_PROFILE=chemicalbinding python3 -m modal run --detach \\
        modal_mol_struct_ae.py::precompute_features

Then launch training (detaches; survives client disconnect):
    MODAL_PROFILE=chemicalbinding python3 -m modal run --detach modal_mol_struct_ae.py

With overrides:
    MODAL_PROFILE=chemicalbinding python3 -m modal run --detach \\
        modal_mol_struct_ae.py --epochs 20 --batch-size 128 --hidden 192 --latent 384

Download checkpoints when training finishes:
    MODAL_PROFILE=chemicalbinding python3 -m modal run \\
        modal_mol_struct_ae.py::download_results
"""
from __future__ import annotations

import modal

APP_NAME = "molstructae-zinc"
REMOTE_DIR = "/root/mol_struct_ae"
RUNS_DIR = "/root/runs"
SHARD_DIR = f"{RUNS_DIR}/shards"
CKPT_DIR = f"{RUNS_DIR}/zinc250k"
LOCAL_PROJECT = "/Users/eding36/VSCodeProjects/mol_struct_ae"

app = modal.App(APP_NAME)

# ── Container image ────────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", extra_index_url="https://download.pytorch.org/whl/cu121")
    .pip_install(
        "rdkit>=2023.3.1",
        "numpy>=1.24",
        "pandas>=2.0",
        "tqdm>=4.65",
        "scikit-learn>=1.3",
        "PyTDC>=0.4.1",   # Therapeutics Data Commons — LIT-PCBA + other benchmarks
    )
    .add_local_dir(
        LOCAL_PROJECT,
        remote_path=REMOTE_DIR,
        ignore=["runs", ".cache", "**/__pycache__", "**/*.pyc", ".git", "data/shards"],
    )
)

# ── Persistent volume — holds precomputed shards + checkpoints ────────────────
runs_vol = modal.Volume.from_name("molstructae-runs", create_if_missing=True)


# ── Featurization (CPU, parallel) ─────────────────────────────────────────────
@app.function(
    image=image,
    cpu=32.0,
    memory=32 * 1024,
    timeout=60 * 60 * 24,  # 24h ceiling (Modal max) — sized for ZINC15-10M
    volumes={RUNS_DIR: runs_vol},
)
def precompute_features(
    max_atoms: int = 64,
    max_torsions: int = 64,
    shard_size: int = 1024,
    workers: int = 8,
    limit: int = 0,
    csv_path: str = "",            # if "", use bundled data/zinc250k.csv
    shard_dir: str = "",           # if "", use the legacy SHARD_DIR
) -> None:
    import os
    import subprocess
    import sys

    csv_path = csv_path or f"{REMOTE_DIR}/data/zinc250k.csv"
    shard_dir = shard_dir or SHARD_DIR

    os.makedirs(shard_dir, exist_ok=True)
    existing = [f for f in os.listdir(shard_dir) if f.endswith(".pt")]
    print(f"[precompute] {len(existing)} shards already in {shard_dir}; "
           f"utils/featurize.py will resume past them.")

    cmd = [
        sys.executable, "-u", f"{REMOTE_DIR}/utils/featurize.py",
        "--csv", csv_path,
        "--out", shard_dir,
        "--max-atoms", str(max_atoms),
        "--max-torsions", str(max_torsions),
        "--shard-size", str(shard_size),
        "--workers", str(workers),
    ]
    if limit > 0:
        cmd += ["--limit", str(limit)]
    print(f"[precompute] running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    runs_vol.commit()
    print(f"[precompute] committed shards to {shard_dir}.")


# ── Training (GPU, mixed precision) ───────────────────────────────────────────
@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 12,   # 12h ceiling
    volumes={RUNS_DIR: runs_vol},
)
def train_mol_struct_ae_modal(
    epochs: int = 20,
    batch_size: int = 64,
    num_workers: int = 4,
    max_atoms: int = 48,
    max_dihedrals: int = 64,
    hidden: int = 96,
    latent: int = 256,
    lr: float = 3e-4,
    amp: bool = True,
    resume: str = "",
    shard_dir: str = "",
    out_dir: str = "",
) -> None:
    import os
    import subprocess
    import sys

    shard_dir = shard_dir or SHARD_DIR
    out_dir = out_dir or CKPT_DIR
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        sys.executable, "-u", f"{REMOTE_DIR}/train_mol_struct_ae.py",
        "--shard-dir", shard_dir,
        "--out-dir", out_dir,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--num-workers", str(num_workers),
        "--max-atoms", str(max_atoms),
        "--max-dihedrals", str(max_dihedrals),
        "--hidden", str(hidden),
        "--latent", str(latent),
        "--lr", str(lr),
    ]
    if amp:
        cmd.append("--amp")
    if resume:
        cmd += ["--resume", resume]

    print(f"[train] running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REMOTE_DIR)
    runs_vol.commit()
    print(f"[train] committed checkpoints to {out_dir}.")


# ── Default entrypoint — kicks off training (assumes shards exist) ────────────
@app.local_entrypoint()
def main(
    epochs: int = 20,
    batch_size: int = 64,
    hidden: int = 96,
    latent: int = 256,
    max_atoms: int = 48,
    resume: str = "",
):
    print("Launching MolStructAutoencoder training on Modal (A10G GPU)")
    print(f"  Dataset      : ZINC250k (precomputed shards on volume)")
    print(f"  Epochs       : {epochs}    Batch size: {batch_size}")
    print(f"  Hidden/Latent: {hidden}/{latent}  max_atoms: {max_atoms}")
    print(f"  Output       : molstructae-runs volume → zinc250k/")
    train_mol_struct_ae_modal.spawn(
        epochs=epochs, batch_size=batch_size, hidden=hidden, latent=latent,
        max_atoms=max_atoms, resume=resume,
    )


# ── Precompute entrypoint (run this first, then `main`) ───────────────────────
@app.local_entrypoint()
def precompute(
    max_atoms: int = 64,
    shard_size: int = 1024,
    workers: int = 16,
    limit: int = 0,
):
    print(f"Featurizing ZINC250k on Modal — max_atoms={max_atoms} workers={workers}")
    precompute_features.spawn(
        max_atoms=max_atoms, shard_size=shard_size,
        workers=workers, limit=limit,
    )


# ── Watcher: polls the volume until shards are ready, then spawns training ───
# Runs on cheap CPU (~$0.05/h) — avoids burning A10G hours waiting for the
# precompute to commit. Once shards exist it calls train_mol_struct_ae_modal.remote()
# which blocks until training is done.
@app.function(
    image=image,
    cpu=1.0,
    memory=2 * 1024,
    timeout=60 * 60 * 18,
    volumes={RUNS_DIR: runs_vol},
)
def wait_and_train(
    min_shards: int = 200,
    poll_seconds: int = 60,
    epochs: int = 20,
    batch_size: int = 64,
    hidden: int = 96,
    latent: int = 256,
    max_atoms: int = 48,
    resume: str = "",
) -> None:
    import glob
    import os
    import time

    os.makedirs(SHARD_DIR, exist_ok=True)
    waited = 0
    while True:
        runs_vol.reload()
        shards = glob.glob(f"{SHARD_DIR}/shard_*.pt")
        if len(shards) >= min_shards:
            print(f"[wait] {len(shards)} shards visible after {waited}s — launching training",
                   flush=True)
            break
        print(f"[wait] {len(shards)} shards (need {min_shards}); waited {waited}s",
               flush=True)
        time.sleep(poll_seconds)
        waited += poll_seconds

    train_mol_struct_ae_modal.remote(
        epochs=epochs, batch_size=batch_size, hidden=hidden, latent=latent,
        max_atoms=max_atoms, resume=resume,
    )


@app.local_entrypoint()
def auto():
    """Spawn the watcher: blocks (on CPU) until precompute commits shards, then
    spawns GPU training. The most efficient end-to-end orchestration."""
    print("Spawning watcher — will trigger GPU training once shards land.")
    wait_and_train.spawn()


# ─── Index-cache builder (CPU only) ───────────────────────────────────────────
# Instantiating ShardedMolDataset(shard_dir, max_atoms=...) triggers a full
# torch.load over every shard to filter oversize molecules. On a 4700-shard
# directory that's ~90 min. Running it on CPU once and committing the
# .index.maxatoms_<N>.json sidecar makes subsequent training runs (and resumes)
# start instantly.
@app.function(
    image=image,
    cpu=8.0,
    memory=16 * 1024,
    timeout=60 * 60 * 3,    # 3h is plenty
    volumes={RUNS_DIR: runs_vol},
)
def build_index_cache(shard_dir: str, max_atoms: int = 64) -> None:
    import sys
    sys.path.insert(0, REMOTE_DIR)
    from mol_struct_ae.dataset import ShardedMolDataset

    print(f"[build_index_cache] shard_dir={shard_dir}  max_atoms={max_atoms}")
    ds = ShardedMolDataset(shard_dir, max_atoms=max_atoms)
    print(f"[build_index_cache] indexed {len(ds)} samples across {len(ds.shard_paths)} shards")
    runs_vol.commit()
    print("[build_index_cache] cache committed to volume.")


@app.local_entrypoint()
def index_cache(shard_dir: str = "", max_atoms: int = 64):
    """Build + commit the dataset index cache for a shard directory.

    Cheap CPU job (~$0.10) that does the slow torch.load scan once so training
    relaunches read the index instantly. Defaults to the ZINC15 5M shard dir.
    """
    sd = shard_dir or ZINC15_SHARD_DIR
    print(f"[index_cache] scheduling cache build for {sd} (max_atoms={max_atoms})")
    build_index_cache.spawn(shard_dir=sd, max_atoms=max_atoms)


# ─── Volume-side file copy (CPU only) ─────────────────────────────────────────
@app.function(
    image=image,
    cpu=1.0,
    memory=2 * 1024,
    timeout=60 * 10,
    volumes={RUNS_DIR: runs_vol},
)
def copy_on_volume_modal(src: str, dst: str) -> None:
    import shutil, os
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print(f"[copy] {src} → {dst}")
    shutil.copy(src, dst)
    runs_vol.commit()
    sz = os.path.getsize(dst) / 2**20
    print(f"[copy] done ({sz:.1f} MB) — committed to volume")


@app.local_entrypoint()
def save_checkpoint(src: str, dst: str):
    """Copy a checkpoint (or any file) to a new path on the Modal volume.

    Example:
        modal run modal_mol_struct_ae.py::save_checkpoint \\
            --src /root/runs/zinc15_5m_diverse_run/ckpt_step074000.pt \\
            --dst /root/runs/zinc15_5m_diverse_run/ORIGINAL_epoch0_backup.pt
    """
    print(f"[save_checkpoint] {src} → {dst}")
    copy_on_volume_modal.remote(src=src, dst=dst)


# ─── Similarity search (GPU) ───────────────────────────────────────────────────
@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 2,
    volumes={RUNS_DIR: runs_vol},
)
def simsearch_modal(
    checkpoint: str,
    library_shard_dir: str,
    query_name: str,
    query_smiles: str,
    top_k: int = 20,
    batch_size: int = 256,
    max_atoms: int = 64,
) -> dict:
    """Embed a query SMILES + a pre-featurized library, return top-K cosine hits.

    Returns a dict so the local entrypoint can print results cleanly.
    """
    import sys, time
    sys.path.insert(0, REMOTE_DIR)
    import numpy as np
    import torch
    import torch.nn.functional as F
    from mol_struct_ae import MolStructAutoencoder
    from mol_struct_ae.model import MolAEConfig
    from mol_struct_ae.data import collate
    from utils.featurize import featurize_smiles
    from utils.feature_utils import embed_from_shards

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[simsearch] device={device}")

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg_args = ckpt.get("config", {})
    cfg = MolAEConfig(
        max_atoms=cfg_args.get("max_atoms", max_atoms),
        hidden_dim=cfg_args.get("hidden", 96),
        latent_dim=cfg_args.get("latent", 256),
    )
    model = MolStructAutoencoder(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[simsearch] loaded checkpoint (step {ckpt.get('step', '?')}) — "
           f"max_atoms={cfg.max_atoms} hidden={cfg.hidden_dim} latent={cfg.latent_dim}")

    # 1) Embed query
    qs = featurize_smiles(query_smiles, max_atoms=cfg.max_atoms)
    if qs is None:
        raise ValueError(f"query SMILES failed featurization: {query_smiles}")
    qbatch = collate([qs], max_atoms=cfg.max_atoms).to(device)
    with torch.no_grad():
        out = model(qbatch, sample=False)
        q_embed = F.normalize(out["sim_embed"], dim=-1).cpu().numpy()[0]
    print(f"[simsearch] query='{query_name}' SMILES='{query_smiles}'  "
           f"embed.shape={q_embed.shape}")

    # 2) Embed library
    t0 = time.time()
    lib = embed_from_shards(library_shard_dir, model, device, batch_size, cfg.max_atoms)
    print(f"[simsearch] embedded {len(lib)} library SMILES in {time.time()-t0:.0f}s")

    smis = list(lib.keys())
    L = np.stack([lib[s] for s in smis]).astype(np.float32)        # [N, D] L2-normalized
    sims = L @ q_embed.astype(np.float32)                          # [N]

    topk_idx = np.argsort(-sims)[: top_k + 1]                      # +1 in case query is in library
    results = []
    for i in topk_idx:
        results.append({"smiles": smis[i], "cosine": float(sims[i])})
    return {
        "query_name": query_name,
        "query_smiles": query_smiles,
        "checkpoint": checkpoint,
        "library_size": len(smis),
        "top_k": results[: top_k + 1],
    }


@app.local_entrypoint()
def simsearch(
    query_smiles: str = "CC(=O)Oc1ccccc1C(=O)O",
    query_name: str = "aspirin",
    top_k: int = 20,
    checkpoint: str = "",
    library_shard_dir: str = "",
):
    """Embed `query_smiles` + the ZINC250k library, print top-K cosine hits."""
    ckpt = checkpoint or f"{ZINC15_CKPT_DIR}/final.pt"
    lib = library_shard_dir or SHARD_DIR
    print(f"[simsearch] query='{query_name}' ({query_smiles})")
    print(f"            checkpoint={ckpt}")
    print(f"            library={lib}")
    res = simsearch_modal.remote(
        checkpoint=ckpt, library_shard_dir=lib,
        query_name=query_name, query_smiles=query_smiles,
        top_k=top_k,
    )
    print(f"\n=== Top {top_k} hits vs {res['query_name']} "
           f"(library size: {res['library_size']:,}) ===")
    print(f"{'rank':>4}  {'cosine':>7}  smiles")
    rank = 0
    for hit in res["top_k"]:
        if hit["smiles"] == res["query_smiles"]:   # skip self if present
            continue
        rank += 1
        print(f"  {rank:>3}  {hit['cosine']:+.4f}  {hit['smiles']}")
        if rank >= top_k:
            break


# ─── MoleculeACE benchmark (GPU, multi-CPU) ────────────────────────────────────
@app.function(
    image=image,
    gpu="A10G",
    cpu=16.0,
    memory=32 * 1024,
    timeout=60 * 60 * 4,
    volumes={RUNS_DIR: runs_vol},
)
def moleculeace_benchmark_modal(checkpoint: str, targets: list,
                                  workers: int = 12,
                                  embed_batch_size: int = 256,
                                  train_smiles_path: str = "") -> list:
    import sys
    sys.path.insert(0, REMOTE_DIR)
    import torch
    from mol_struct_ae import MolStructAutoencoder
    from mol_struct_ae.model import MolAEConfig
    from mol_struct_ae.dataset import make_collate_fn
    from utils.moleculeace_benchmark import run_moleculeace

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg_args = ckpt.get("config", {})
    cfg = MolAEConfig(
        max_atoms=cfg_args.get("max_atoms", 64),
        hidden_dim=cfg_args.get("hidden", 96),
        latent_dim=cfg_args.get("latent", 256),
    )
    model = MolStructAutoencoder(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[moleculeace] loaded ckpt step={ckpt.get('step','?')}  workers={workers}")

    collate_fn = make_collate_fn(cfg.max_atoms, cfg_args.get("max_dihedrals", 64))
    return run_moleculeace(model, collate_fn, device, cfg.max_atoms,
                             targets=targets or None, workers=workers,
                             embed_batch_size=embed_batch_size,
                             train_smiles_path=train_smiles_path or None)


@app.local_entrypoint()
def moleculeace_benchmark(checkpoint: str = "", targets: str = "",
                           train_smiles_path: str = ""):
    """MoleculeACE activity-cliff benchmark on a subset of ChEMBL targets.

    Run a Ridge probe on (frozen embedding → pIC50). Compare RMSE on overall
    test set vs RMSE on the 'cliff_mol' subset. A chemistry-aware embedding
    has cliff RMSE ≈ overall RMSE; a Morgan-FP-like embedding has cliff RMSE
    much higher than overall.
    """
    ckpt = checkpoint or f"{ZINC15_CKPT_DIR}/final.pt"
    train_smis = train_smiles_path or f"{RUNS_DIR}/train_smiles_zinc15_5m.json.gz"
    tgt_list = [t.strip() for t in targets.split(",") if t.strip()] if targets else []
    print(f"[moleculeace_benchmark] ckpt={ckpt}")
    print(f"  leakage set: {train_smis}")
    print(f"  targets: {tgt_list or 'DEFAULT_SUBSET'}")
    res = moleculeace_benchmark_modal.remote(checkpoint=ckpt, targets=tgt_list,
                                                train_smiles_path=train_smis)

    print("\n" + "=" * 118)
    print("  MoleculeACE — Ridge probe on frozen embeddings, RMSE (lower=better) "
           "[leakage-filtered against ZINC15 5M train]")
    print("=" * 118)
    print(f"  {'Target':<22}  {'leak':>4}  {'n_test':>6}  {'cliff':>5}  "
           f"{'all (m / fp)':>16}  {'cliff (m / fp)':>16}  {'non-cliff (m / fp)':>20}  "
           f"{'cliff gap (m / fp)':>20}")
    print(f"  {'-'*22}  {'-'*4}  {'-'*6}  {'-'*5}  "
           f"{'-'*16}  {'-'*16}  {'-'*20}  {'-'*20}")
    for r in res:
        if "error" in r:
            print(f"  {r['target']:<22}  ERROR: {r['error']}")
            continue
        all_m = r['rmse_all_model']; all_fp = r['rmse_all_fp']
        cl_m  = r['rmse_cliff_model']; cl_fp  = r['rmse_cliff_fp']
        nc_m  = r['rmse_noncl_model']; nc_fp  = r['rmse_noncl_fp']
        gap_m = cl_m - nc_m
        gap_fp = cl_fp - nc_fp
        leak = r.get('n_dropped_leak', 0)
        print(f"  {r['target']:<22}  {leak:>4}  {r['n_test']:>6}  {r['n_test_cliff']:>5}  "
               f"{all_m:>6.3f} / {all_fp:>6.3f}  "
               f"{cl_m:>6.3f} / {cl_fp:>6.3f}  "
               f"{nc_m:>8.3f} / {nc_fp:>8.3f}  "
               f"{gap_m:>+8.3f} / {gap_fp:>+8.3f}")
    print()
    print("  Notes:")
    print("    • Cliff RMSE = error on test mols flagged as activity-cliff pairs")
    print("    • A chemistry-aware embedding should have small (cliff - non-cliff) gap")
    print("    • Morgan-FP-only models typically have large positive gap on cliffs")


# ─── LIT-PCBA benchmark (GPU, multi-CPU) ───────────────────────────────────────
@app.function(
    image=image,
    gpu="A10G",
    cpu=16.0,
    memory=32 * 1024,
    timeout=60 * 60 * 6,
    volumes={RUNS_DIR: runs_vol},
)
def litpcba_benchmark_modal(checkpoint: str, targets: list, max_decoys: int = 10000,
                              workers: int = 12, embed_batch_size: int = 256,
                              train_smiles_path: str = "") -> list:
    import sys
    sys.path.insert(0, REMOTE_DIR)
    import torch
    from mol_struct_ae import MolStructAutoencoder
    from mol_struct_ae.model import MolAEConfig
    from mol_struct_ae.dataset import make_collate_fn
    from utils.litpcba_benchmark import run_litpcba

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg_args = ckpt.get("config", {})
    cfg = MolAEConfig(
        max_atoms=cfg_args.get("max_atoms", 64),
        hidden_dim=cfg_args.get("hidden", 96),
        latent_dim=cfg_args.get("latent", 256),
    )
    model = MolStructAutoencoder(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[litpcba] loaded ckpt step={ckpt.get('step','?')}  workers={workers}")

    collate_fn = make_collate_fn(cfg.max_atoms, cfg_args.get("max_dihedrals", 64))
    return run_litpcba(model, collate_fn, device, cfg.max_atoms,
                        targets=targets or None, max_decoys=max_decoys,
                        workers=workers, embed_batch_size=embed_batch_size,
                        train_smiles_path=train_smiles_path or None)


@app.local_entrypoint()
def litpcba_benchmark(checkpoint: str = "", targets: str = "",
                       max_decoys: int = 10000,
                       train_smiles_path: str = ""):
    """LIT-PCBA virtual-screening benchmark on a subset of protein targets.

    For each target: embed actives + sampled decoys; pick 5 actives as queries;
    rank everything else by max cosine to a query. Report:
        EF@1%, EF@5%   (enrichment factor — higher better; random = 1.0)
        AUROC          (ranking-based)
    """
    ckpt = checkpoint or f"{ZINC15_CKPT_DIR}/final.pt"
    train_smis = train_smiles_path or f"{RUNS_DIR}/train_smiles_zinc15_5m.json.gz"
    tgt_list = [t.strip() for t in targets.split(",") if t.strip()] if targets else []
    print(f"[litpcba_benchmark] ckpt={ckpt}  max_decoys={max_decoys}")
    print(f"  leakage set: {train_smis}")
    print(f"  targets: {tgt_list or 'DEFAULT_SUBSET'}")
    res = litpcba_benchmark_modal.remote(checkpoint=ckpt, targets=tgt_list,
                                            max_decoys=max_decoys,
                                            train_smiles_path=train_smis)

    print("\n" + "=" * 115)
    print("  LIT-PCBA — virtual screening enrichment (higher = better; random = 1.0) "
           "[leakage-filtered against ZINC15 5M train]")
    print("=" * 115)
    print(f"  {'Target':<12}  {'N_act':>6}  {'N_dec':>6}  {'leak (a/d)':>10}  "
           f"{'EF@1% (m / fp)':>18}  {'EF@5% (m / fp)':>18}  {'AUROC (m / fp)':>18}")
    print(f"  {'-'*12}  {'-'*6}  {'-'*6}  {'-'*10}  "
           f"{'-'*18}  {'-'*18}  {'-'*18}")
    for r in res:
        if "error" in r:
            print(f"  {r['target']:<12}  ERROR: {r['error']}")
            continue
        leak_str = f"{r.get('leak_actives',0)}/{r.get('leak_decoys',0)}"
        print(f"  {r['target']:<12}  {r['n_actives']:>6}  {r['n_decoys']:>6}  {leak_str:>10}  "
               f"{r['model_EF1']:>7.2f} / {r['fp_EF1']:>7.2f}  "
               f"{r['model_EF5']:>7.2f} / {r['fp_EF5']:>7.2f}  "
               f"{r['model_AUROC']:>7.3f} / {r['fp_AUROC']:>7.3f}")
    print()
    print("  Notes:")
    print("    • (model / fp) → cosine on our embedding vs cosine on 1024-bit Morgan FPs")
    print("    • Decoys subsampled to max_decoys per target (default 10K)")
    print("    • Queries: 5 actives per draw, averaged over 10 random draws")


# ─── Perturbation smoke test (GPU) ─────────────────────────────────────────────
# Measure how the embedding moves under controlled structural perturbations:
# +CH2 (methylene), +CH3 (methyl), halogen swap, +phenyl ring.
@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 30,
    volumes={RUNS_DIR: runs_vol},
)
def pair_compare_modal(checkpoint: str, pairs: list) -> list:
    """For each (label, smi_a, smi_b) tuple, embed both and return cosine sim.

    `pairs` arrives as a JSON-serialisable list of [str, str, str] lists.
    """
    import sys
    sys.path.insert(0, REMOTE_DIR)
    import numpy as np
    import torch
    import torch.nn.functional as F
    from mol_struct_ae import MolStructAutoencoder
    from mol_struct_ae.model import MolAEConfig
    from mol_struct_ae.data import collate
    from utils.featurize import featurize_smiles

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg_args = ckpt.get("config", {})
    cfg = MolAEConfig(
        max_atoms=cfg_args.get("max_atoms", 64),
        hidden_dim=cfg_args.get("hidden", 96),
        latent_dim=cfg_args.get("latent", 256),
    )
    model = MolStructAutoencoder(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[pair_compare] loaded ckpt step={ckpt.get('step','?')} on {device}")

    # Featurize unique SMILES once
    seen: dict = {}
    for label, a, b in pairs:
        for smi in (a, b):
            if smi not in seen:
                s = featurize_smiles(smi, max_atoms=cfg.max_atoms)
                if s is None:
                    print(f"[pair_compare] WARN failed to featurize {smi}")
                seen[smi] = s

    # Embed in one batch where possible
    smi_list = [s for s, samp in seen.items() if samp is not None]
    samples = [seen[s] for s in smi_list]
    batch = collate(samples, max_atoms=cfg.max_atoms).to(device)
    with torch.no_grad():
        out = model(batch, sample=False)
        E = F.normalize(out["sim_embed"], dim=-1).cpu().numpy()
    emb = {s: E[i] for i, s in enumerate(smi_list)}

    results = []
    for label, a, b in pairs:
        if a not in emb or b not in emb:
            results.append({"label": label, "a": a, "b": b, "cosine": None})
            continue
        cos = float(np.dot(emb[a], emb[b]))
        results.append({"label": label, "a": a, "b": b, "cosine": cos})
    return results


# ─── Training-set SMILES extractor (CPU) ───────────────────────────────────────
# Iterates a shard directory, pulls the canonical SMILES out of every MolSample
# (stripped of stereo info so the leakage check is conservative), dedupes, and
# writes the set to a single .json.gz file on the volume. Used by the
# MoleculeNet benchmark to remove any benchmark mol that the model already saw
# during pre-training.
def _extract_shard_smiles(shard_path: str) -> list:
    """Worker: load one shard, return list of canonical (stereo-stripped) SMILES.

    Defined at module scope so it pickles cleanly across multiprocessing workers.
    """
    import sys, os
    sys.path.insert(0, REMOTE_DIR)
    import torch
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    try:
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  skip unreadable {os.path.basename(shard_path)}: {type(e).__name__}")
        return []
    out: list = []
    for s in shard:
        raw = getattr(s, "smiles", "") or ""
        if not raw:
            continue
        m = Chem.MolFromSmiles(raw)
        if m is None:
            continue
        out.append(Chem.MolToSmiles(m, isomericSmiles=False))
    return out


@app.function(
    image=image,
    cpu=8.0,
    memory=16 * 1024,
    timeout=60 * 60 * 2,
    volumes={RUNS_DIR: runs_vol},
)
def extract_train_smiles_modal(shard_dir: str, out_path: str, workers: int = 8) -> int:
    import sys, glob, gzip, json, os, time
    from multiprocessing import Pool
    sys.path.insert(0, REMOTE_DIR)

    shard_paths = sorted(glob.glob(os.path.join(shard_dir, "shard_*.pt")))
    print(f"[extract_train_smiles] {len(shard_paths)} shards in {shard_dir}; workers={workers}")

    smis: set = set()
    processed = 0
    t0 = time.time()
    with Pool(workers) as pool:
        for chunk in pool.imap_unordered(_extract_shard_smiles, shard_paths, chunksize=4):
            smis.update(chunk)
            processed += 1
            if processed % 200 == 0:
                rate = processed / max(time.time() - t0, 1e-6)
                eta = (len(shard_paths) - processed) / max(rate, 1e-6) / 60
                print(f"  processed {processed}/{len(shard_paths)} shards; unique mols: "
                       f"{len(smis):,}  ({rate:.1f} shards/s, ETA {eta:.1f}m)",
                       flush=True)

    print(f"[extract_train_smiles] {len(smis):,} unique canonical SMILES "
           f"in {(time.time()-t0)/60:.1f} min")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with gzip.open(out_path, "wt") as f:
        json.dump(sorted(smis), f)
    runs_vol.commit()
    print(f"[extract_train_smiles] wrote {out_path} (committed)")
    return len(smis)


@app.local_entrypoint()
def extract_train_smiles(shard_dir: str = "", out_path: str = ""):
    """Iterate training shards → write deduped canonical-SMILES set to the volume.

    Uses .spawn() so the job survives client disconnect (detached-safe).
    Monitor progress via `modal app logs <APP_ID>`.
    """
    sd = shard_dir or ZINC15_SHARD_DIR
    op = out_path or f"{RUNS_DIR}/train_smiles_zinc15_5m.json.gz"
    print(f"[extract_train_smiles] shard_dir={sd}  out={op}")
    print(f"[extract_train_smiles] spawning detached worker — tail logs via Modal app id")
    extract_train_smiles_modal.spawn(shard_dir=sd, out_path=op)


# ─── MoleculeNet benchmark (GPU) ───────────────────────────────────────────────
@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 2,
    volumes={RUNS_DIR: runs_vol},
)
def moleculenet_benchmark_modal(checkpoint: str, datasets: list,
                                  embed_batch_size: int = 128,
                                  train_smiles_path: str = "") -> list:
    """Run MoleculeNet linear-probe benchmark on the supplied datasets.

    If `train_smiles_path` is non-empty, drop every benchmark molecule whose
    canonical (stereo-stripped) SMILES is in that file.
    """
    import sys
    sys.path.insert(0, REMOTE_DIR)
    import torch
    from mol_struct_ae import MolStructAutoencoder
    from mol_struct_ae.model import MolAEConfig
    from mol_struct_ae.dataset import make_collate_fn
    from utils.featurize import featurize_smiles
    from utils.moleculenet_benchmark import run_benchmark

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg_args = ckpt.get("config", {})
    cfg = MolAEConfig(
        max_atoms=cfg_args.get("max_atoms", 64),
        hidden_dim=cfg_args.get("hidden", 96),
        latent_dim=cfg_args.get("latent", 256),
    )
    model = MolStructAutoencoder(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[bench] loaded ckpt step={ckpt.get('step','?')} max_atoms={cfg.max_atoms}")

    collate_fn = make_collate_fn(cfg.max_atoms, cfg_args.get("max_dihedrals", 64))
    results = run_benchmark(
        featurize_fn=featurize_smiles,
        model=model,
        collate_fn=collate_fn,
        device=device,
        max_atoms=cfg.max_atoms,
        datasets=datasets,
        embed_batch_size=embed_batch_size,
        train_smiles_path=train_smiles_path or None,
    )
    return results


# Reference numbers from the literature (scaffold split where reported).
# These are NOT directly comparable due to differing splits/probes/seeds,
# but they place our numbers in context.
LITERATURE_REFERENCE = {
    "BBBP":     {"MolBERT": 0.762, "ChemBERTa": 0.728, "MolCLR": 0.736, "Uni-Mol": 0.729, "Grover-Lg": 0.940},
    "BACE":     {"MolBERT": 0.866, "ChemBERTa": 0.799, "MolCLR": 0.824, "Uni-Mol": 0.857, "Grover-Lg": 0.878},
    "ClinTox":  {"MolBERT": 0.732, "ChemBERTa": 0.733, "MolCLR": 0.911, "Uni-Mol": 0.919, "Grover-Lg": 0.812},
    "HIV":      {"MolBERT": 0.783, "ChemBERTa": 0.622, "MolCLR": 0.780, "Uni-Mol": 0.808, "Grover-Lg": 0.802},
    "ESOL":     {"MolBERT": 0.531, "MolCLR": 1.110, "Uni-Mol": 0.788, "Grover-Lg": 0.831},
    "FreeSolv": {"MolBERT": 0.948, "MolCLR": 2.200, "Uni-Mol": 1.480, "Grover-Lg": 1.544},
    "Lipo":     {"MolBERT": 0.560, "MolCLR": 0.789, "Uni-Mol": 0.603, "Grover-Lg": 0.560},
}


@app.local_entrypoint()
def moleculenet_benchmark(checkpoint: str = "",
                           datasets: str = "BBBP,BACE,ClinTox,HIV,ESOL,FreeSolv,Lipo",
                           train_smiles_path: str = ""):
    """Linear-probe benchmark on MoleculeNet (scaffold split).

    Reports our model vs Morgan-FP baseline (apples-to-apples, computed in
    the same pipeline) alongside published literature numbers (which may use
    different splits/probes — see footnote).

    If `train_smiles_path` is non-empty (default uses the canonical-SMILES
    set under RUNS_DIR), benchmark molecules also seen during pre-training
    are dropped before embedding to prevent leakage.
    """
    ckpt = checkpoint or f"{ZINC15_CKPT_DIR}/final.pt"
    train_smis = train_smiles_path or f"{RUNS_DIR}/train_smiles_zinc15_5m.json.gz"
    ds_list = [d.strip() for d in datasets.split(",") if d.strip()]
    print(f"[moleculenet_benchmark] ckpt={ckpt}")
    print(f"  train SMILES leakage set: {train_smis}")
    print(f"  datasets: {ds_list}")
    res = moleculenet_benchmark_modal.remote(checkpoint=ckpt, datasets=ds_list,
                                                train_smiles_path=train_smis)

    # ── Print comparison table
    print("\n" + "=" * 105)
    print("  MoleculeNet — scaffold split, linear probe on frozen embeddings  "
           "(leakage-filtered against ZINC15 5M train set)")
    print("=" * 105)
    print(f"  {'Dataset':<10}  {'Task':<7}  {'Metric':<7}  {'N':>5}  {'leak':>4}  "
           f"{'OURS':>7}  {'Morgan':>7}  {'MolBERT':>7}  {'ChemBERTa':>9}  {'MolCLR':>7}  {'Uni-Mol':>7}")
    print(f"  {'-'*10}  {'-'*7}  {'-'*7}  {'-'*5}  {'-'*4}  "
           f"{'-'*7}  {'-'*7}  {'-'*7}  {'-'*9}  {'-'*7}  {'-'*7}")
    for r in res:
        ref = LITERATURE_REFERENCE.get(r["dataset"], {})
        def fmt(x): return f"{x:.4f}" if isinstance(x, (int, float)) else "  —  "
        score_model = r['model']
        score_fp = r['morgan_fp']
        sm = f"{score_model:>7.4f}" if score_model == score_model else "    —  "  # NaN check
        sf = f"{score_fp:>7.4f}" if score_fp == score_fp else "    —  "
        print(f"  {r['dataset']:<10}  {r['task'][:5]:<7}  {r['metric']:<7}  "
               f"{r['n_kept']:>5}  {r.get('n_dropped_leak', 0):>4}  "
               f"{sm}  {sf}  "
               f"{fmt(ref.get('MolBERT')):>7}  {fmt(ref.get('ChemBERTa')):>9}  "
               f"{fmt(ref.get('MolCLR')):>7}  {fmt(ref.get('Uni-Mol')):>7}")
    print()
    print("  Notes:")
    print("    • OURS = MolStructAutoencoder (1-epoch ZINC15 5M) + LogReg/Ridge probe")
    print("    • Morgan = 1024-bit r=2 fingerprint + same probe (apples-to-apples baseline)")
    print("    • Classification metric is AUROC (↑ better); regression is RMSE (↓ better)")
    print("    • Literature numbers from each paper's reported scaffold-split scores.")
    print("      Splits/probes/seeds differ; treat as ballpark, not direct comparison.")


@app.local_entrypoint()
def perturbation_smoke(checkpoint: str = ""):
    """Smoke-test the embedding under four classes of single-atom-group edits.

    Each row: (description, base SMILES, perturbed SMILES). The model should
    assign HIGH cosine for similar pairs and LOW cosine for unrelated baselines.
    """
    ckpt = checkpoint or f"{ZINC15_CKPT_DIR}/final.pt"

    pairs = [
        # ── METHYLENE (+CH2 in middle / chain) ─────────────────────────────────
        ["[CH2] pentane → hexane",
            "CCCCC", "CCCCCC"],
        ["[CH2] phenylacetic → 3-phenylpropanoic acid",
            "OC(=O)Cc1ccccc1", "OC(=O)CCc1ccccc1"],
        ["[CH2] aspirin → homoaspirin (extra CH2 in acetyl)",
            "CC(=O)Oc1ccccc1C(=O)O", "CCC(=O)Oc1ccccc1C(=O)O"],

        # ── METHYL (+CH3 at a terminal / on a ring) ────────────────────────────
        ["[CH3] benzene → toluene",
            "c1ccccc1", "Cc1ccccc1"],
        ["[CH3] theobromine → caffeine (N-methyl)",
            "CN1C=NC2=C1C(=O)NC(=O)N2C",
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"],
        ["[CH3] aspirin → 5-methylaspirin",
            "CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Oc1cc(C)ccc1C(=O)O"],

        # ── HALOGEN swap (F → Cl → Br) ────────────────────────────────────────
        ["[halogen] fluorobenzene → chlorobenzene",
            "Fc1ccccc1", "Clc1ccccc1"],
        ["[halogen] chlorobenzene → bromobenzene",
            "Clc1ccccc1", "Brc1ccccc1"],
        ["[halogen] 4-F-toluene → 4-Br-toluene",
            "Cc1ccc(F)cc1", "Cc1ccc(Br)cc1"],

        # ── PHENYL (+ benzene ring) ────────────────────────────────────────────
        ["[phenyl] toluene → biphenyl-methyl",
            "Cc1ccccc1", "Cc1ccc(-c2ccccc2)cc1"],
        ["[phenyl] ethanol → 2-phenylethanol",
            "CCO", "OCCc1ccccc1"],
        ["[phenyl] aspirin → 4-phenyl-aspirin",
            "CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Oc1ccc(-c2ccccc2)cc1C(=O)O"],

        # ── IDENTITY baseline (should be ≈ 1.0) ────────────────────────────────
        ["[identity] aspirin == aspirin",
            "CC(=O)Oc1ccccc1C(=O)O", "CC(=O)Oc1ccccc1C(=O)O"],

        # ── UNRELATED baseline (should be << any above) ────────────────────────
        ["[unrelated] aspirin vs glucose",
            "CC(=O)Oc1ccccc1C(=O)O", "OCC1OC(O)C(O)C(O)C1O"],
        ["[unrelated] aspirin vs ethanol",
            "CC(=O)Oc1ccccc1C(=O)O", "CCO"],
        ["[unrelated] caffeine vs glucose",
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "OCC1OC(O)C(O)C(O)C1O"],
    ]

    print(f"[perturbation_smoke] ckpt={ckpt}  pairs={len(pairs)}")
    res = pair_compare_modal.remote(checkpoint=ckpt, pairs=pairs)

    # Group by category prefix for readability
    print()
    print(f"  {'category':<10}  {'cosine':>7}  {'description':<55}")
    print(f"  {'-'*10}  {'-'*7}  {'-'*55}")
    last_cat = None
    for r in res:
        cat = r["label"].split("]")[0].strip("[")
        if last_cat is not None and cat != last_cat:
            print()
        last_cat = cat
        desc = r["label"].split("] ", 1)[-1]
        cos_s = f"{r['cosine']:+.4f}" if r["cosine"] is not None else "  N/A "
        print(f"  {cat:<10}  {cos_s:>7}  {desc:<55}")


# ─── ZINC15 10M pipeline (fetch → featurize → train, all on Modal) ──────────────

ZINC15_CSV = f"{RUNS_DIR}/zinc15_10m.csv"                   # raw 10M from DeepChem
ZINC15_DIVERSE_CSV = f"{RUNS_DIR}/zinc15_5m_diverse.csv"     # scaffold-balanced 5M
ZINC15_SHARD_DIR = f"{RUNS_DIR}/shards_zinc15_5m_diverse"
ZINC15_CKPT_DIR = f"{RUNS_DIR}/zinc15_5m_diverse_run"


@app.function(
    image=image,
    cpu=16.0,
    memory=32 * 1024,
    timeout=60 * 60 * 6,         # 6h — downloading + canonicalizing 10–30M SMILES
    volumes={RUNS_DIR: runs_vol},
)
def fetch_zinc15_modal(size: str = "10M", out_path: str = "") -> None:
    """Download DeepChem's pre-curated ZINC15 {size} CSV onto the volume."""
    import os
    import subprocess
    import sys

    out_path = out_path or ZINC15_CSV
    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / 2**20
        print(f"[fetch] {out_path} already exists ({size_mb:.0f} MB); skipping. "
               f"Delete it to re-download.")
        return

    cmd = [
        sys.executable, "-u", f"{REMOTE_DIR}/scripts/fetch_zinc15.py",
        "--size", size,
        "--out", out_path,
    ]
    print(f"[fetch] running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REMOTE_DIR)
    runs_vol.commit()
    size_mb = os.path.getsize(out_path) / 2**20
    print(f"[fetch] wrote {out_path} ({size_mb:.0f} MB), committed to volume.")


# ── ZINC15-10M pipeline runs in 3 detached jobs because Modal caps any single
# function at 24h and the full pipeline takes ~30+ hours. Each entry point
# spawns one stage; user runs them in sequence as each completes.


@app.function(
    image=image,
    cpu=16.0,
    memory=32 * 1024,
    timeout=60 * 60 * 2,
    volumes={RUNS_DIR: runs_vol},
)
def diversify_zinc15_modal(
    target_count: int = 5_000_000,
    workers: int = 16,
    in_csv: str = "",
    out_csv: str = "",
) -> None:
    """Scaffold-aware subsample of `in_csv` → `out_csv` on volume."""
    import os
    import subprocess
    import sys

    in_csv = in_csv or ZINC15_CSV
    out_csv = out_csv or ZINC15_DIVERSE_CSV
    if os.path.exists(out_csv):
        size_mb = os.path.getsize(out_csv) / 2**20
        print(f"[diversify] {out_csv} already exists ({size_mb:.0f} MB); "
               f"skipping. Delete it to re-run.")
        return
    cmd = [
        sys.executable, "-u", f"{REMOTE_DIR}/scripts/diversify_smiles.py",
        "--in-csv", in_csv,
        "--out-csv", out_csv,
        "--target-count", str(target_count),
        "--workers", str(workers),
    ]
    print(f"[diversify] running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REMOTE_DIR)
    runs_vol.commit()


@app.local_entrypoint()
def zinc15_fetch(size: str = "10M"):
    """Stage 1 / 4 — download DeepChem's pre-curated ZINC15 {size} CSV. ~3 min."""
    print(f"[zinc15_fetch] size={size}")
    fetch_zinc15_modal.spawn(size=size, out_path=ZINC15_CSV)


@app.local_entrypoint()
def zinc15_diversify(target_count: int = 5_000_000, workers: int = 16):
    """Stage 2 / 4 — Bemis-Murcko scaffold-balanced subsample → diverse CSV. ~5 min."""
    print(f"[zinc15_diversify] {ZINC15_CSV} → {ZINC15_DIVERSE_CSV} "
           f"(target={target_count:,})")
    diversify_zinc15_modal.spawn(target_count=target_count, workers=workers)


@app.local_entrypoint()
def zinc15_precompute(
    workers: int = 32,
    max_atoms: int = 64,
    shard_size: int = 1024,
    shard_dir: str = "",
):
    """Stage 3 / 4 — featurize the scaffold-diversified CSV. ~12-15h on cpu=32.

    If `shard_dir` is given, shards land there; otherwise the default
    ZINC15_SHARD_DIR is used. Useful when re-featurizing at a different
    max_atoms cap without overwriting an existing shard set.
    """
    sd = shard_dir or ZINC15_SHARD_DIR
    print(f"[zinc15_precompute] workers={workers} max_atoms={max_atoms} csv={ZINC15_DIVERSE_CSV}")
    print(f"  → shards land at {sd}")
    precompute_features.spawn(
        max_atoms=max_atoms, shard_size=shard_size,
        workers=workers, csv_path=ZINC15_DIVERSE_CSV, shard_dir=sd,
    )


@app.local_entrypoint()
def zinc15_train(
    epochs: int = 1,
    batch_size: int = 64,
    hidden: int = 96,
    latent: int = 256,
    max_atoms: int = 48,
    lr: float = 3e-4,
    resume: str = "",
):
    """Stage 3 / 3 — train mol_struct_ae on the 10M shards. ~17h for 1 epoch on A10G.

    If `resume` is given (path to a .pt checkpoint on the Modal volume),
    training continues from that checkpoint's model + optimizer state.
    """
    print(f"[zinc15_train] epochs={epochs} hidden={hidden}/{latent} batch={batch_size} lr={lr}")
    print(f"  shards = {ZINC15_SHARD_DIR}")
    print(f"  output → {ZINC15_CKPT_DIR}")
    if resume:
        print(f"  resume = {resume}")
    train_mol_struct_ae_modal.spawn(
        epochs=epochs, batch_size=batch_size, hidden=hidden, latent=latent,
        max_atoms=max_atoms, lr=lr, resume=resume,
        shard_dir=ZINC15_SHARD_DIR, out_dir=ZINC15_CKPT_DIR,
    )


@app.local_entrypoint()
def resume_from_local_ckpt(
    local_ckpt: str = "runs/zinc250k_remote/ckpt_step002000.pt",
    epochs: int = 1,
    workers: int = 8,
):
    """End-to-end re-run in the current Modal profile:
      1. Upload `local_ckpt` to the volume so the remote training can resume.
      2. Spawn `precompute_features` (will be a no-op if shards already exist).
      3. Spawn `wait_and_train(resume=...)` — polls the volume on cheap CPU,
         triggers GPU training when shards land.

    Run with:
        MODAL_PROFILE=chemicalbinding python -m modal run --detach \\
            modal_mol_struct_ae.py::resume_from_local_ckpt
    """
    from pathlib import Path

    p = Path(local_ckpt)
    if not p.is_file():
        raise FileNotFoundError(f"checkpoint not found: {local_ckpt}")
    remote_ckpt_rel = f"zinc250k/{p.name}"
    remote_ckpt_abs = f"{CKPT_DIR}/{p.name}"
    print(f"[resume] uploading {local_ckpt} → volume:{remote_ckpt_rel}")
    with runs_vol.batch_upload(force=True) as upload:
        upload.put_file(str(p), remote_ckpt_rel)
    print(f"[resume] uploaded.")

    print(f"[resume] spawning precompute_features (no-op if shards already on volume)")
    precompute_features.spawn(workers=workers)

    print(f"[resume] spawning wait_and_train(resume={remote_ckpt_abs}, epochs={epochs})")
    wait_and_train.spawn(resume=remote_ckpt_abs, epochs=epochs)


# ── Download checkpoints to local ─────────────────────────────────────────────
@app.local_entrypoint()
def download_results():
    import os
    dest = f"{LOCAL_PROJECT}/runs/zinc250k_remote"
    os.makedirs(dest, exist_ok=True)
    for entry in runs_vol.listdir("zinc250k"):
        name = entry.path.split("/")[-1]
        if name.endswith((".pt", ".json", ".log")):
            with open(f"{dest}/{name}", "wb") as fout:
                for chunk in runs_vol.read_file(f"zinc250k/{name}"):
                    fout.write(chunk)
            print(f"Downloaded: {name}")
    print(f"Saved to {dest}")
