"""
MoleculeACE — Activity Cliff Estimation (Tilborg, Alenicheva, Grisoni 2022).

Tests whether your embedding handles activity cliffs: pairs of molecules that
are nearly identical in structure but have very different bioactivities.
Morgan-FP-based models tend to FAIL on cliff molecules (because they look
similar in FP space). A good chemistry-aware embedding should handle them.

Protocol:
  1. For each of 30 ChEMBL targets (we run a subset for speed):
     - Download canonical CSV from MoleculeACE repo
     - Parse: smiles, pIC50 (y), cliff_mol (1/0), split (train/test)
     - Featurize multithreaded; drop unfeaturizable
     - Embed via supplied model; Morgan FP for baseline
     - Train Ridge on train split → predict pIC50 on test
     - Compute RMSE on all test mols + RMSE on cliff-only subset
  2. Report per-target table.

The headline metric is **RMSE_cliff**. A model that just learns Morgan patterns
has RMSE_cliff > RMSE_overall by a wide margin. A model that captures genuine
chemistry has the two roughly equal.
"""
from __future__ import annotations

import csv
import io
import time
import urllib.request
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

RDLogger.DisableLog("rdApp.*")


# 30 ChEMBL targets from the MoleculeACE paper. We run a representative subset
# for tractability (each adds ~5-10 min); the full sweep is left as an option.
ALL_TARGETS = [
    "CHEMBL1862_Ki", "CHEMBL1871_Ki", "CHEMBL2034_Ki", "CHEMBL2047_EC50",
    "CHEMBL204_Ki",  "CHEMBL2147_Ki", "CHEMBL214_Ki",  "CHEMBL218_EC50",
    "CHEMBL219_Ki",  "CHEMBL228_Ki",  "CHEMBL231_Ki",  "CHEMBL233_Ki",
    "CHEMBL234_Ki",  "CHEMBL235_EC50","CHEMBL236_Ki",  "CHEMBL237_EC50",
    "CHEMBL237_Ki",  "CHEMBL238_Ki",  "CHEMBL239_EC50","CHEMBL244_Ki",
    "CHEMBL262_Ki",  "CHEMBL264_Ki",  "CHEMBL2835_Ki", "CHEMBL287_Ki",
    "CHEMBL2971_Ki", "CHEMBL3979_EC50","CHEMBL4005_Ki","CHEMBL4203_Ki",
    "CHEMBL4616_EC50","CHEMBL4792_Ki",
]

# A diverse subset of 8 targets covering different protein families
DEFAULT_SUBSET = [
    "CHEMBL204_Ki",     # Thrombin
    "CHEMBL214_Ki",     # 5-HT1a serotonin receptor
    "CHEMBL219_Ki",     # Dopamine D4
    "CHEMBL228_Ki",     # Serotonin 2C
    "CHEMBL233_Ki",     # μ-opioid
    "CHEMBL234_Ki",     # Dopamine D3
    "CHEMBL236_Ki",     # δ-opioid
    "CHEMBL287_Ki",     # σ opioid
]


def _download_csv(target: str) -> str:
    url = (f"https://raw.githubusercontent.com/molML/MoleculeACE/main/"
           f"MoleculeACE/Data/benchmark_data/{target}.csv")
    print(f"  [download] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "moleculeace-benchmark/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read().decode("utf-8")


def load_target(target: str) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """Returns (smiles, y_pIC50, is_cliff, is_train_split). All length-N."""
    text = _download_csv(target)
    reader = csv.DictReader(io.StringIO(text))
    smis, ys, cliffs, splits = [], [], [], []
    # Column names per MoleculeACE: smiles, exp_mean [nM] (or 'y'), cliff_mol, split
    rows = list(reader)
    if not rows:
        raise RuntimeError(f"empty CSV for {target}")
    first = rows[0]
    # Detect column names
    smi_key = "smiles" if "smiles" in first else "Smiles"
    if "y" in first:
        y_key = "y"
    elif "exp_mean [nM]" in first:
        y_key = "exp_mean [nM]"
    else:
        y_key = next(k for k in first.keys() if k.lower() in ("y", "pic50", "pchembl_value"))
    cliff_key = "cliff_mol" if "cliff_mol" in first else next(
        (k for k in first.keys() if "cliff" in k.lower()), "")
    split_key = "split" if "split" in first else next(
        (k for k in first.keys() if k.lower() in ("split", "set")), "")

    for row in rows:
        smi = row.get(smi_key, "").strip().strip('"')
        if not smi:
            continue
        v = row.get(y_key, "").strip()
        try:
            y = float(v)
        except ValueError:
            continue
        cliff = int(float(row.get(cliff_key, "0").strip() or "0")) if cliff_key else 0
        sp = (row.get(split_key, "train").strip().lower()) if split_key else "train"
        is_train = sp in ("train", "training")

        smis.append(smi)
        ys.append(y)
        cliffs.append(cliff)
        splits.append(is_train)

    return (
        smis,
        np.array(ys, dtype=np.float32),
        np.array(cliffs, dtype=np.int8),
        np.array(splits, dtype=bool),
    )


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


@torch.no_grad()
def embed_with_model(samples: list, model, collate_fn: Callable, device,
                      batch_size: int = 256) -> np.ndarray:
    embs: List[torch.Tensor] = []
    for i in range(0, len(samples), batch_size):
        batch = collate_fn(samples[i:i + batch_size]).to(device)
        out = model(batch, sample=False)
        e = F.normalize(out["sim_embed"], dim=-1).cpu()
        embs.append(e)
    return torch.cat(embs, dim=0).numpy().astype(np.float32)


def _ridge_rmse(X_train, y_train, X_test, y_test):
    mu, sd = float(y_train.mean()), float(y_train.std() or 1.0)
    rgr = Ridge(alpha=1.0)
    rgr.fit(X_train, (y_train - mu) / sd)
    y_pred = rgr.predict(X_test) * sd + mu
    return float(np.sqrt(mean_squared_error(y_test, y_pred))), y_pred


def run_moleculeace(model, collate_fn: Callable, device, max_atoms: int,
                     targets: Optional[List[str]] = None,
                     workers: int = 8, embed_batch_size: int = 256,
                     train_smiles_path: Optional[str] = None) -> List[dict]:
    """Run the MoleculeACE benchmark on the supplied targets.

    If `train_smiles_path` is provided (a .json.gz file produced by
    extract_train_smiles_modal), drop every benchmark mol whose canonical
    (stereo-stripped) SMILES is in the pretraining set. This prevents
    leakage from inflating scores. Cliff flags + train/test labels are
    preserved on the surviving subset.
    """
    from utils.parallel_featurize import featurize_smiles_parallel
    from utils.moleculenet_benchmark import load_train_smiles_set

    train_set: Optional[set] = None
    if train_smiles_path:
        print(f"[bench] loading leakage set from {train_smiles_path} …")
        train_set = load_train_smiles_set(train_smiles_path)
        print(f"[bench] loaded {len(train_set):,} unique training SMILES")

    targets = targets or DEFAULT_SUBSET
    results: List[dict] = []
    for target in targets:
        t0 = time.time()
        print(f"\n[{target}] downloading + parsing CSV …")
        try:
            smis, y, cliffs, is_train = load_target(target)
        except Exception as e:
            print(f"[{target}] FAILED to load: {type(e).__name__}: {e}")
            results.append({"target": target, "error": str(e)})
            continue
        n_raw = len(smis)
        n_leak = 0
        if train_set is not None:
            keep_mask = np.zeros(n_raw, dtype=bool)
            for i, smi in enumerate(smis):
                m = Chem.MolFromSmiles(smi)
                if m is None:
                    continue
                canon = Chem.MolToSmiles(m, isomericSmiles=False)
                if canon not in train_set:
                    keep_mask[i] = True
            n_leak = int(n_raw - keep_mask.sum())
            smis = [s for i, s in enumerate(smis) if keep_mask[i]]
            y = y[keep_mask]; cliffs = cliffs[keep_mask]; is_train = is_train[keep_mask]
            print(f"[{target}] leakage filter: dropped {n_leak}/{n_raw} "
                   f"({100*n_leak/max(n_raw,1):.1f}%) mols also in training")
        print(f"[{target}] {len(smis)} rows; train={is_train.sum()} test={int((~is_train).sum())} "
               f"cliff_test={int(cliffs[~is_train].sum())}")

        print(f"[{target}] featurizing in parallel (workers={workers}) …")
        samples, keep_idx = featurize_smiles_parallel(
            smis, max_atoms=max_atoms, workers=workers,
        )
        if not samples:
            print(f"[{target}] no featurizable molecules — skipping")
            results.append({"target": target, "error": "no featurizable"})
            continue
        kept_y = y[keep_idx]
        kept_cliff = cliffs[keep_idx]
        kept_train = is_train[keep_idx]
        kept_smis = [smis[i] for i in keep_idx]

        print(f"[{target}] embedding {len(samples)} mols …")
        E_model = embed_with_model(samples, model, collate_fn, device, embed_batch_size)
        E_fp = morgan_fp_matrix(kept_smis)

        tr_mask = kept_train
        te_mask = ~kept_train
        n_te_cliff = int(kept_cliff[te_mask].sum())
        if int(tr_mask.sum()) < 10 or int(te_mask.sum()) < 5:
            print(f"[{target}] insufficient train/test rows after featurize — skipping")
            results.append({"target": target, "error": "tiny split"})
            continue

        # Ridge on model + on Morgan FP
        rmse_all_m, ypred_m = _ridge_rmse(E_model[tr_mask], kept_y[tr_mask],
                                            E_model[te_mask], kept_y[te_mask])
        rmse_all_fp, ypred_fp = _ridge_rmse(E_fp[tr_mask], kept_y[tr_mask],
                                              E_fp[te_mask], kept_y[te_mask])

        # Cliff-only and non-cliff RMSE within the test set
        te_y = kept_y[te_mask]
        te_cliff = kept_cliff[te_mask].astype(bool)
        def sub_rmse(y_true, y_hat, mask):
            if mask.sum() == 0:
                return float("nan")
            return float(np.sqrt(mean_squared_error(y_true[mask], y_hat[mask])))
        rmse_cliff_m = sub_rmse(te_y, ypred_m, te_cliff)
        rmse_cliff_fp = sub_rmse(te_y, ypred_fp, te_cliff)
        rmse_noncl_m = sub_rmse(te_y, ypred_m, ~te_cliff)
        rmse_noncl_fp = sub_rmse(te_y, ypred_fp, ~te_cliff)

        dt = time.time() - t0
        print(f"[{target}] RMSE all/cliff/non-cliff   "
               f"model={rmse_all_m:.3f}/{rmse_cliff_m:.3f}/{rmse_noncl_m:.3f}   "
               f"morgan={rmse_all_fp:.3f}/{rmse_cliff_fp:.3f}/{rmse_noncl_fp:.3f}   "
               f"({dt:.0f}s)")
        results.append({
            "target": target,
            "n_raw": n_raw, "n_dropped_leak": n_leak,
            "n_train": int(tr_mask.sum()), "n_test": int(te_mask.sum()),
            "n_test_cliff": n_te_cliff,
            "rmse_all_model": rmse_all_m, "rmse_all_fp": rmse_all_fp,
            "rmse_cliff_model": rmse_cliff_m, "rmse_cliff_fp": rmse_cliff_fp,
            "rmse_noncl_model": rmse_noncl_m, "rmse_noncl_fp": rmse_noncl_fp,
        })
    return results
