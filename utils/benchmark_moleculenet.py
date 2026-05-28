"""
MoleculeNet *fine-tuning* benchmark — replicates the Hu et al. (2020) /
Mole-BERT (Xia et al., ICLR 2023) protocol: the pretrained encoder is
fine-tuned end-to-end per dataset (no frozen probe).

Protocol (per dataset, matching Mole-BERT §5.2 / Appendix E):
  1. Download + parse the dataset CSV.
  2. Optional leakage filter against the pre-training SMILES set.
  3. Featurize once → list[MolSample] (the expensive ETKDG step; cached and
     reused across all seeds).
  4. Bemis-Murcko scaffold split 80/10/10 (train/val/test).
  5. Attach a task head to the *pretrained encoder* and FINE-TUNE END-TO-END
     (encoder + head), 100 epochs, batch 32, Adam lr=1e-3, dropout=0.5.
  6. Model selection: report TEST ROC-AUC (cls) / RMSE (reg) at the epoch with
     the best VALIDATION score (early-stopping-by-checkpoint protocol).
  7. Repeat over N seeds (default 10); report mean (std).

Defining traits:
  * end-to-end fine-tuning of the encoder, not a frozen sklearn probe;
  * 80/10/10 split with a validation set used for model selection;
  * multi-seed mean/std;
  * native multi-task support (Tox21/ToxCast/SIDER/MUV/ClinTox) via masked BCE.

This is the apples-to-apples harness for comparing against Mole-BERT Table 1.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import time
import urllib.request
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_squared_error, roc_auc_score


# ──────────────────────────────────────────────────────────────────────────────
# Download + leakage-filter helpers.
# ──────────────────────────────────────────────────────────────────────────────
def _download(url: str) -> str:
    print(f"[download] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "molstruct-bench/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    if url.endswith(".gz"):
        raw = gzip.decompress(raw)
    return raw.decode("utf-8")


def load_train_smiles_set(path: str) -> set:
    """Load the JSON-gz set of canonical (stereo-stripped) training SMILES."""
    with gzip.open(path, "rt") as f:
        return set(json.load(f))


def filter_against_train_set(smiles: List[str], y: np.ndarray,
                             train_set: set) -> Tuple[List[str], np.ndarray, int]:
    """Drop any row whose canonical (stereo-stripped) SMILES is in `train_set`.

    Mirrors the canonicalization used by extract_train_smiles_modal so the
    leakage comparison is consistent on both sides.
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

RDLogger.DisableLog("rdApp.*")


# ──────────────────────────────────────────────────────────────────────────────
# Dataset configs — single AND multi-task. `labels="auto"` means "every column
# except the listed non-label columns is a binary task" (used for the big
# multi-task assay sets). NaN / empty cells are treated as missing labels.
# ──────────────────────────────────────────────────────────────────────────────
DATASETS: Dict[str, dict] = {
    # ── single-task classification ──────────────────────────────────────────
    "BBBP": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        "smi_col": "smiles", "labels": ["p_np"], "task": "classification",
    },
    "BACE": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
        "smi_col": "mol", "labels": ["Class"], "task": "classification",
    },
    "HIV": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
        "smi_col": "smiles", "labels": ["HIV_active"], "task": "classification",
    },
    # ── multi-task classification ───────────────────────────────────────────
    "ClinTox": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
        "smi_col": "smiles", "labels": ["FDA_APPROVED", "CT_TOX"],
        "task": "classification",
    },
    "Tox21": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
        "smi_col": "smiles", "labels": "auto", "task": "classification",
        "non_label": {"smiles", "mol_id"},
    },
    "ToxCast": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/toxcast_data.csv.gz",
        "smi_col": "smiles", "labels": "auto", "task": "classification",
        "non_label": {"smiles"},
    },
    "SIDER": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz",
        "smi_col": "smiles", "labels": "auto", "task": "classification",
        "non_label": {"smiles"},
    },
    "MUV": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz",
        "smi_col": "smiles", "labels": "auto", "task": "classification",
        "non_label": {"smiles", "mol_id"},
    },
    # ── single-task regression ──────────────────────────────────────────────
    "ESOL": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
        "smi_col": "smiles",
        "labels": ["measured log solubility in mols per litre"], "task": "regression",
    },
    "FreeSolv": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv",
        "smi_col": "smiles", "labels": ["expt"], "task": "regression",
    },
    "Lipo": {
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",
        "smi_col": "smiles", "labels": ["exp"], "task": "regression",
    },
}


def load_dataset_multitask(name: str) -> Tuple[List[str], np.ndarray, str, List[str]]:
    """Returns (smiles, Y[N, T], task, task_names). Missing labels are NaN."""
    cfg = DATASETS[name]
    text = _download(cfg["url"])
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = reader.fieldnames or []

    if cfg["labels"] == "auto":
        non_label = cfg.get("non_label", {cfg["smi_col"]})
        task_cols = [c for c in fieldnames if c not in non_label]
    else:
        task_cols = list(cfg["labels"])

    smis: List[str] = []
    rows_y: List[List[float]] = []
    for row in reader:
        smi = (row.get(cfg["smi_col"], "") or "").strip().strip('"')
        if not smi:
            continue
        yvals: List[float] = []
        for c in task_cols:
            v = (row.get(c, "") or "").strip()
            if v == "" or v.lower() in ("nan", "na"):
                yvals.append(float("nan"))
            else:
                try:
                    yvals.append(float(v))
                except ValueError:
                    yvals.append(float("nan"))
        # Skip rows with no usable label at all
        if all(np.isnan(v) for v in yvals):
            continue
        smis.append(smi)
        rows_y.append(yvals)
    Y = np.array(rows_y, dtype=np.float32)
    return smis, Y, cfg["task"], task_cols


# ──────────────────────────────────────────────────────────────────────────────
# 3-way Bemis-Murcko scaffold split (80/10/10).
# Deterministic largest-first scaffold assignment (Hu et al. style); the seed
# only perturbs tie ordering so the split is essentially fixed across seeds and
# the std reflects init/training stochasticity — matching the Hu et al. protocol.
# ──────────────────────────────────────────────────────────────────────────────
def _murcko(smi: str) -> str:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    except Exception:
        return ""


def scaffold_split_3way(smiles: List[str], frac_train: float = 0.8,
                         frac_val: float = 0.1
                         ) -> Tuple[List[int], List[int], List[int]]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(smiles):
        groups[_murcko(s)].append(i)
    sorted_groups = sorted(groups.values(), key=lambda g: (-len(g), smiles[g[0]]))
    n = len(smiles)
    n_train, n_val = int(frac_train * n), int(frac_val * n)
    train, val, test = [], [], []
    for g in sorted_groups:
        if len(train) + len(g) <= n_train:
            train.extend(g)
        elif len(val) + len(g) <= n_val:
            val.extend(g)
        else:
            test.extend(g)
    return train, val, test


# ──────────────────────────────────────────────────────────────────────────────
# Fine-tune model: pretrained encoder (encode→mu) + a dropout+linear task head.
# All encoder params + head params are optimized end-to-end.
# ──────────────────────────────────────────────────────────────────────────────
class FineTuneModel(nn.Module):
    def __init__(self, ae: nn.Module, latent_dim: int, n_tasks: int,
                 dropout: float = 0.5):
        super().__init__()
        self.ae = ae
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(latent_dim, n_tasks)

    def forward(self, batch) -> torch.Tensor:
        enc = self.ae.encode(batch, sample=False)     # uses mu (deterministic)
        h = self.dropout(enc["mu"])
        return self.head(h)                            # [B, n_tasks]

    def trainable_parameters(self):
        # Encoder path that actually contributes to encode(): input embedder,
        # pairformer stack, global aggregator. Decoders are excluded (no grad).
        for mod in (self.ae.input_embedder, self.ae.pairformer, self.ae.aggregator):
            yield from mod.parameters()
        yield from self.dropout.parameters()
        yield from self.head.parameters()


def _multitask_bce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Masked BCE over non-NaN labels. logits/y: [B, T]."""
    mask = ~torch.isnan(y)
    if mask.sum() == 0:
        return logits.sum() * 0.0
    yt = torch.nan_to_num(y, nan=0.0)
    loss = F.binary_cross_entropy_with_logits(logits, yt, reduction="none")
    return (loss * mask.float()).sum() / mask.float().sum()


@torch.no_grad()
def _evaluate(model: FineTuneModel, samples, Y, idx, collate_fn, device,
              task: str, max_atoms: int, batch_size: int = 128) -> float:
    model.eval()
    preds: List[np.ndarray] = []
    for i in range(0, len(idx), batch_size):
        chunk = [samples[j] for j in idx[i:i + batch_size]]
        batch = collate_fn(chunk).to(device)
        out = model(batch).cpu().numpy()
        preds.append(out)
    P = np.concatenate(preds, axis=0) if preds else np.zeros((0, Y.shape[1]))
    Yt = Y[idx]
    if task == "regression":
        # single-task regression in practice
        return float(np.sqrt(mean_squared_error(Yt[:, 0], P[:, 0])))
    # classification: mean ROC-AUC over tasks that have both classes in this split
    aucs: List[float] = []
    for t in range(Y.shape[1]):
        yt = Yt[:, t]
        m = ~np.isnan(yt)
        if m.sum() == 0 or len(np.unique(yt[m])) < 2:
            continue
        aucs.append(roc_auc_score(yt[m], P[m, t]))
    return float(np.mean(aucs)) if aucs else float("nan")


def finetune_once(make_ae: Callable[[], nn.Module], latent_dim: int,
                  samples, Y, splits, task: str, collate_fn, device,
                  max_atoms: int, seed: int, epochs: int = 100,
                  batch_size: int = 32, lr: float = 1e-3,
                  dropout: float = 0.5) -> float:
    """One fine-tuning run. Returns TEST score at the best-VAL epoch."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr, va, te = splits

    ae = make_ae()                                # fresh pretrained encoder
    model = FineTuneModel(ae, latent_dim, Y.shape[1], dropout=dropout).to(device)
    opt = torch.optim.Adam(model.trainable_parameters(), lr=lr)

    y_t = torch.from_numpy(Y).to(device)
    higher_better = (task == "classification")
    best_val = -np.inf if higher_better else np.inf
    best_test = float("nan")

    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(tr))
        for i in range(0, len(tr), batch_size):
            bidx = [tr[j] for j in order[i:i + batch_size]]
            batch = collate_fn([samples[j] for j in bidx]).to(device)
            logits = model(batch)
            yb = y_t[bidx]
            if task == "classification":
                loss = _multitask_bce(logits, yb)
            else:
                loss = F.mse_loss(logits[:, 0], yb[:, 0])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        val = _evaluate(model, samples, Y, va, collate_fn, device, task, max_atoms)
        improved = (val > best_val) if higher_better else (val < best_val)
        if not np.isnan(val) and improved:
            best_val = val
            best_test = _evaluate(model, samples, Y, te, collate_fn, device,
                                  task, max_atoms)
    return best_test


def featurize_dataset(smis: List[str], Y: np.ndarray, featurize_fn: Callable,
                      max_atoms: int) -> Tuple[list, np.ndarray, List[str]]:
    """ETKDG-featurize every molecule once. Returns (samples, Y_kept, smis_kept)."""
    samples, keep = [], []
    for i, smi in enumerate(smis):
        s = featurize_fn(smi, max_atoms=max_atoms)
        if s is None:
            continue
        samples.append(s)
        keep.append(i)
    return samples, Y[keep], [smis[i] for i in keep]


def run_finetune_benchmark(make_ae: Callable[[], nn.Module], latent_dim: int,
                           featurize_fn: Callable, collate_fn: Callable,
                           device: torch.device, max_atoms: int,
                           datasets: List[str], seeds: List[int],
                           epochs: int = 100, batch_size: int = 32,
                           lr: float = 1e-3, dropout: float = 0.5,
                           train_smiles_path: Optional[str] = None) -> List[dict]:
    train_set = None
    if train_smiles_path:
        print(f"[ft] loading leakage set from {train_smiles_path} …")
        train_set = load_train_smiles_set(train_smiles_path)
        print(f"[ft] {len(train_set):,} training SMILES")

    results: List[dict] = []
    for name in datasets:
        t0 = time.time()
        print(f"\n[{name}] downloading + parsing …")
        smis, Y, task, task_names = load_dataset_multitask(name)
        n_raw = len(smis)

        if train_set is not None and Y.shape[1] == 1:
            smis, y1, n_leak = filter_against_train_set(smis, Y[:, 0], train_set)
            Y = y1.reshape(-1, 1)
            print(f"[{name}] leakage filter: dropped {n_leak}/{n_raw}")

        print(f"[{name}] {n_raw} mols, {len(task_names)} task(s); featurizing (ETKDG) …")
        samples, Yk, smis_k = featurize_dataset(smis, Y, featurize_fn, max_atoms)
        print(f"[{name}] kept {len(samples)}/{n_raw} after featurize")

        tr, va, te = scaffold_split_3way(smis_k)
        print(f"[{name}] scaffold 80/10/10 → train={len(tr)} val={len(va)} test={len(te)}")

        scores: List[float] = []
        for sd in seeds:
            s = finetune_once(make_ae, latent_dim, samples, Yk, (tr, va, te),
                              task, collate_fn, device, max_atoms, seed=sd,
                              epochs=epochs, batch_size=batch_size, lr=lr,
                              dropout=dropout)
            scores.append(s)
            print(f"[{name}] seed={sd} test={s:.4f}")
        arr = np.array(scores, dtype=np.float64)
        metric = "AUROC" if task == "classification" else "RMSE"
        print(f"[{name}] {metric} = {np.nanmean(arr):.4f} ± {np.nanstd(arr):.4f} "
               f"over {len(seeds)} seeds  ({time.time()-t0:.0f}s)")
        results.append({
            "dataset": name, "task": task, "metric": metric,
            "n_tasks": len(task_names), "n_kept": len(samples),
            "n_train": len(tr), "n_val": len(va), "n_test": len(te),
            "mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr)),
            "scores": scores,
        })
    return results
