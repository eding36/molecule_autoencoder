"""
MoleculeNet benchmark for frozen molecular embeddings.

Pipeline (per dataset):
  1. Download CSV from DeepChem's S3 mirror.
  2. Parse SMILES + label column, drop rows with bad SMILES or oversize atoms.
  3. Featurize via utils.featurize.featurize_smiles → MolSample.
  4. Embed via the supplied MolStructAutoencoder.
  5. Compute the Morgan-FP baseline (1024-bit radius=2) for the same molecules.
  6. Scaffold-split via Bemis-Murcko: train/test = 80/20, scaffolds disjoint.
  7. Probe:
        classification → sklearn LogisticRegression, metric = ROC-AUC
        regression     → sklearn Ridge,             metric = RMSE
  8. Report both our model and the FP baseline.

Datasets handled (DeepChem's MoleculeNet S3):
  BBBP, BACE, ClinTox, HIV       (classification)
  ESOL, FreeSolv, Lipo           (regression)
"""
from __future__ import annotations

import csv
import gzip
import io
import os
import time
import urllib.request
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score

RDLogger.DisableLog("rdApp.*")


# ──────────────────────────────────────────────────────────────────────────────
# Dataset configs
# ──────────────────────────────────────────────────────────────────────────────
DATASETS: Dict[str, dict] = {
    "BBBP": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        "smi_col": "smiles", "y_col": "p_np", "task": "classification",
    },
    "BACE": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
        "smi_col": "mol", "y_col": "Class", "task": "classification",
    },
    "ClinTox": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
        "smi_col": "smiles", "y_col": "CT_TOX", "task": "classification",
    },
    "HIV": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
        "smi_col": "smiles", "y_col": "HIV_active", "task": "classification",
    },
    "ESOL": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
        "smi_col": "smiles", "y_col": "measured log solubility in mols per litre",
        "task": "regression",
    },
    "FreeSolv": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv",
        "smi_col": "smiles", "y_col": "expt", "task": "regression",
    },
    "Lipo": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
        "smi_col": "smiles", "y_col": "exp", "task": "regression",
    },
}


def _download(url: str) -> str:
    print(f"[download] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "moleculenet-benchmark/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    if url.endswith(".gz"):
        raw = gzip.decompress(raw)
    return raw.decode("utf-8")


def load_train_smiles_set(path: str) -> set:
    """Load the JSON-gz set of canonical (stereo-stripped) training SMILES."""
    import json
    with gzip.open(path, "rt") as f:
        return set(json.load(f))


def filter_against_train_set(smiles: List[str], y: np.ndarray,
                              train_set: set) -> Tuple[List[str], np.ndarray, int]:
    """Drop any row whose canonical (stereo-stripped) SMILES is in `train_set`.

    Mirrors the canonicalization used by extract_train_smiles_modal so the
    comparison is consistent on both sides.
    """
    keep_smis: List[str] = []
    keep_y: List[float] = []
    dropped = 0
    for smi, label in zip(smiles, y):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        canon = Chem.MolToSmiles(m, isomericSmiles=False)
        if canon in train_set:
            dropped += 1
            continue
        keep_smis.append(smi)
        keep_y.append(label)
    return keep_smis, np.array(keep_y, dtype=np.float32), dropped


def load_dataset(name: str) -> Tuple[List[str], np.ndarray, str]:
    cfg = DATASETS[name]
    text = _download(cfg["url"])
    reader = csv.DictReader(io.StringIO(text))
    smis: List[str] = []
    ys: List[float] = []
    for row in reader:
        smi = row.get(cfg["smi_col"], "").strip().strip('"')
        if not smi:
            continue
        v = row.get(cfg["y_col"], "").strip()
        if v == "" or v.lower() in ("nan", "na"):
            continue
        try:
            ys.append(float(v))
        except ValueError:
            continue
        smis.append(smi)
    return smis, np.array(ys, dtype=np.float32), cfg["task"]


# ──────────────────────────────────────────────────────────────────────────────
# Scaffold split (Bemis-Murcko)
# ──────────────────────────────────────────────────────────────────────────────
def _murcko_scaffold(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return ""
    try:
        scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
        return scaf
    except Exception:
        return ""


def scaffold_split(smiles: List[str], frac_train: float = 0.8,
                    seed: int = 0) -> Tuple[List[int], List[int]]:
    """Group molecules by scaffold; sort groups largest-first (deterministic
    for reproducibility); place groups into train until we hit frac_train.
    Returns (train_idx, test_idx).
    """
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(smiles):
        groups[_murcko_scaffold(s)].append(i)
    # Sort by group size (descending) then by scaffold string for determinism
    sorted_groups = sorted(groups.values(), key=lambda g: (-len(g), smiles[g[0]]))
    n = len(smiles)
    cut = int(frac_train * n)
    train, test = [], []
    for g in sorted_groups:
        if len(train) + len(g) <= cut:
            train.extend(g)
        else:
            test.extend(g)
    return train, test


# ──────────────────────────────────────────────────────────────────────────────
# Morgan-FP baseline
# ──────────────────────────────────────────────────────────────────────────────
def morgan_fp_matrix(smiles: List[str], n_bits: int = 1024, radius: int = 2) -> np.ndarray:
    out = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        out[i] = arr.astype(np.float32)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Model-embedding helper (uses already-loaded MolStructAutoencoder)
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def embed_smiles_with_model(smiles: List[str], featurize_fn: Callable, model,
                              collate_fn: Callable, device: torch.device,
                              max_atoms: int, batch_size: int = 128) -> Tuple[np.ndarray, List[int]]:
    """Returns embeddings [n_kept, D] L2-normalized, and the indices of kept rows."""
    samples = []
    keep_idx: List[int] = []
    for i, smi in enumerate(smiles):
        s = featurize_fn(smi, max_atoms=max_atoms)
        if s is None:
            continue
        samples.append(s)
        keep_idx.append(i)

    embs: List[torch.Tensor] = []
    for i in range(0, len(samples), batch_size):
        chunk = samples[i : i + batch_size]
        batch = collate_fn(chunk).to(device)
        out = model(batch, sample=False)
        e = F.normalize(out["sim_embed"], dim=-1).cpu()
        embs.append(e)
    if not embs:
        return np.zeros((0, 256), dtype=np.float32), []
    E = torch.cat(embs, dim=0).numpy().astype(np.float32)
    return E, keep_idx


# ──────────────────────────────────────────────────────────────────────────────
# Probe
# ──────────────────────────────────────────────────────────────────────────────
def probe(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
           y_test: np.ndarray, task: str) -> Tuple[float, str]:
    """Returns (score, metric_name)."""
    if task == "classification":
        # Skip degenerate test labels
        if len(np.unique(y_test)) < 2:
            return float("nan"), "AUROC"
        clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=1)
        clf.fit(X_train, y_train)
        # decision_function preferred over predict_proba (faster, no calibration needed)
        try:
            scores = clf.decision_function(X_test)
        except Exception:
            scores = clf.predict_proba(X_test)[:, 1]
        return float(roc_auc_score(y_test, scores)), "AUROC"
    else:
        # Standardize y for Ridge
        mu, sd = float(y_train.mean()), float(y_train.std() or 1.0)
        rgr = Ridge(alpha=1.0)
        rgr.fit(X_train, (y_train - mu) / sd)
        y_pred = rgr.predict(X_test) * sd + mu
        return float(np.sqrt(mean_squared_error(y_test, y_pred))), "RMSE"


# ──────────────────────────────────────────────────────────────────────────────
# Top-level: run all datasets
# ──────────────────────────────────────────────────────────────────────────────
def run_benchmark(featurize_fn: Callable, model, collate_fn: Callable,
                   device: torch.device, max_atoms: int,
                   datasets: Optional[List[str]] = None,
                   embed_batch_size: int = 128,
                   train_smiles_path: Optional[str] = None) -> List[dict]:
    train_set: Optional[set] = None
    if train_smiles_path:
        print(f"[bench] loading train-smiles leakage set from {train_smiles_path} …")
        train_set = load_train_smiles_set(train_smiles_path)
        print(f"[bench] loaded {len(train_set):,} unique training SMILES")

    datasets = datasets or list(DATASETS.keys())
    results: List[dict] = []
    for name in datasets:
        t0 = time.time()
        print(f"\n[{name}] downloading + parsing CSV …")
        try:
            smis, y, task = load_dataset(name)
        except Exception as e:
            print(f"[{name}] FAILED to load: {type(e).__name__}: {e} — skipping")
            results.append({
                "dataset": name, "task": "?", "metric": "—",
                "n_total": 0, "n_kept": 0, "n_train": 0, "n_test": 0,
                "n_dropped_leak": 0,
                "model": float("nan"), "morgan_fp": float("nan"),
                "error": f"{type(e).__name__}: {e}",
            })
            continue
        n_raw = len(smis)
        n_leak = 0
        if train_set is not None:
            smis, y, n_leak = filter_against_train_set(smis, y, train_set)
            print(f"[{name}] leakage filter: dropped {n_leak}/{n_raw} "
                   f"({100*n_leak/max(n_raw,1):.1f}%) mols also seen in training")
        print(f"[{name}] {len(smis)} rows after leakage filter; task={task}")

        try:
            print(f"[{name}] embedding with model …")
            E_model, keep = embed_smiles_with_model(
                smis, featurize_fn, model, collate_fn, device, max_atoms,
                batch_size=embed_batch_size,
            )
            smis_kept = [smis[i] for i in keep]
            y_kept = y[keep]
            print(f"[{name}] kept {len(smis_kept)}/{len(smis)} after featurize "
                   f"(dropped: bad SMILES, >max_atoms, embed fails)")

            print(f"[{name}] computing Morgan FPs …")
            E_fp = morgan_fp_matrix(smis_kept)

            print(f"[{name}] scaffold-splitting …")
            tr, te = scaffold_split(smis_kept)
            print(f"[{name}] train={len(tr)}  test={len(te)}")

            score_model, metric = probe(E_model[tr], y_kept[tr], E_model[te], y_kept[te], task)
            score_fp, _ = probe(E_fp[tr], y_kept[tr], E_fp[te], y_kept[te], task)

            dt = time.time() - t0
            print(f"[{name}] {metric}: model={score_model:.4f}   morgan-fp={score_fp:.4f}   "
                   f"({dt:.0f}s)")
            results.append({
                "dataset": name, "task": task, "metric": metric,
                "n_total": n_raw, "n_kept": len(smis_kept),
                "n_train": len(tr), "n_test": len(te),
                "n_dropped_leak": n_leak,
                "model": score_model, "morgan_fp": score_fp,
            })
        except Exception as e:
            print(f"[{name}] FAILED during embed/probe: {type(e).__name__}: {e} — skipping")
            results.append({
                "dataset": name, "task": task, "metric": "—",
                "n_total": n_raw, "n_kept": 0, "n_train": 0, "n_test": 0,
                "n_dropped_leak": n_leak,
                "model": float("nan"), "morgan_fp": float("nan"),
                "error": f"{type(e).__name__}: {e}",
            })
    return results
