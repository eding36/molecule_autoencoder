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
    shard_dir: str = "",           # default to SHARD_DIR when empty
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
    shard_dir: str = "",
):
    print(f"Featurizing ZINC250k on Modal — max_atoms={max_atoms} workers={workers}")
    precompute_features.spawn(
        max_atoms=max_atoms, shard_size=shard_size,
        workers=workers, limit=limit, shard_dir=shard_dir,
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


# ─── MoleculeNet FINE-TUNING benchmark (GPU) ──────────────────────────────────
# Replicates the Hu et al. (2020) / Mole-BERT (ICLR 2023) protocol: fine-tune
# the pretrained encoder END-TO-END with a task head, scaffold 80/10/10, report
# test ROC-AUC/RMSE at the best-validation epoch, mean±std over N seeds.
# This is the apples-to-apples harness for comparing against Mole-BERT Table 1.
@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 24,
    volumes={RUNS_DIR: runs_vol},
)
def moleculenet_finetune_modal(checkpoint: str, datasets: list, seeds: list,
                                epochs: int = 100, batch_size: int = 32,
                                lr: float = 1e-3, dropout: float = 0.5,
                                train_smiles_path: str = "",
                                split_strategy: str = "deterministic") -> list:
    import sys
    sys.path.insert(0, REMOTE_DIR)
    from utils.benchmark_moleculenet import AEBackend, run_finetune_benchmark

    backend = AEBackend(checkpoint)
    return run_finetune_benchmark(
        backend, datasets=datasets, seeds=seeds, epochs=epochs,
        batch_size=batch_size, lr=lr, dropout=dropout,
        train_smiles_path=train_smiles_path or None,
        split_strategy=split_strategy,
    )


# Mole-BERT Table 1 reference (ROC-AUC, scaffold split, fine-tuning, 10 seeds).
MOLEBERT_REFERENCE = {
    "Tox21": 0.768, "ToxCast": 0.643, "SIDER": 0.628, "ClinTox": 0.789,
    "MUV": 0.786, "HIV": 0.782, "BBBP": 0.719, "BACE": 0.808,
}


@app.local_entrypoint()
def moleculenet_finetune(checkpoint: str = "",
                         datasets: str = "BBBP,BACE,ClinTox,SIDER",
                         seeds: str = "0,1,2",
                         epochs: int = 100, batch_size: int = 32, lr: float = 1e-3,
                         train_smiles_path: str = "",
                         split_strategy: str = "deterministic"):
    """Fine-tuning benchmark replicating Mole-BERT's protocol (Table 1).

    Fine-tunes the pretrained encoder END-TO-END per dataset (scaffold 80/10/10,
    test score at best-val epoch, mean±std over seeds). Directly comparable to
    Mole-BERT's published fine-tuning numbers (Table 1).

    NOTE: our encoder needs 3D ETKDG featurization per molecule, so the big
    sets (HIV 41K, MUV 93K, ToxCast 8.5K×617-task) are SLOW to featurize. Start
    with the small/medium sets; add the big ones explicitly when ready. Defaults
    to 3 seeds for turnaround — pass seeds=0,1,...,9 to match the paper's 10.
    """
    ckpt = checkpoint or f"{ZINC15_CKPT_DIR}/final.pt"
    ds_list = [d.strip() for d in datasets.split(",") if d.strip()]
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    train_smis = train_smiles_path  # default empty: leakage filter only for single-task
    print(f"[moleculenet_finetune] ckpt={ckpt}")
    print(f"  datasets: {ds_list}   seeds: {seed_list}   epochs={epochs}")
    print(f"  split_strategy: {split_strategy}")
    res = moleculenet_finetune_modal.remote(
        checkpoint=ckpt, datasets=ds_list, seeds=seed_list,
        epochs=epochs, batch_size=batch_size, lr=lr,
        train_smiles_path=train_smis,
        split_strategy=split_strategy,
    )

    print("\n" + "=" * 78)
    print("  MoleculeNet — END-TO-END FINE-TUNING (scaffold 80/10/10, best-val epoch)")
    print("  Directly comparable to Mole-BERT (ICLR 2023) Table 1.")
    print("=" * 78)
    print(f"  {'Dataset':<10}  {'Metric':<6}  {'#task':>5}  {'N':>6}  "
           f"{'OURS (mean±std)':>18}  {'Mole-BERT':>9}")
    print(f"  {'-'*10}  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*18}  {'-'*9}")
    for r in res:
        ref = MOLEBERT_REFERENCE.get(r["dataset"])
        ref_s = f"{ref:.3f}" if ref is not None else "  —  "
        ours = f"{r['mean']:.4f} ± {r['std']:.4f}"
        print(f"  {r['dataset']:<10}  {r['metric']:<6}  {r['n_tasks']:>5}  "
               f"{r['n_kept']:>6}  {ours:>18}  {ref_s:>9}")
    print()
    print("  Notes:")
    print("    • OURS = pretrained Pairformer encoder + linear head, fine-tuned end-to-end.")
    print("    • Mole-BERT col = their Table 1 ROC-AUC (GIN backbone, 10 seeds).")
    print("    • Classification = AUROC (↑); regression = RMSE (↓).")
    print("    • Same protocol (scaffold 80/10/10, best-val epoch) — directly comparable,")
    print("      modulo backbone (Pairformer vs GIN) and pretraining set (ZINC15 5M vs 2M).")


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
    shard_dir: str = "",
    out_dir: str = "",
):
    """Stage 3 / 3 — train mol_struct_ae on the 10M shards. ~17h for 1 epoch on A10G.

    If `resume` is given (path to a .pt checkpoint on the Modal volume),
    training continues from that checkpoint's model + optimizer state.

    `shard_dir` / `out_dir` default to the ZINC15 5M paths but can be
    overridden when training on a re-featurized shard set (e.g. max_atoms=96).
    """
    sd = shard_dir or ZINC15_SHARD_DIR
    od = out_dir or ZINC15_CKPT_DIR
    print(f"[zinc15_train] epochs={epochs} hidden={hidden}/{latent} batch={batch_size} "
           f"max_atoms={max_atoms} lr={lr}")
    print(f"  shards = {sd}")
    print(f"  output → {od}")
    if resume:
        print(f"  resume = {resume}")
    train_mol_struct_ae_modal.spawn(
        epochs=epochs, batch_size=batch_size, hidden=hidden, latent=latent,
        max_atoms=max_atoms, lr=lr, resume=resume,
        shard_dir=sd, out_dir=od,
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
