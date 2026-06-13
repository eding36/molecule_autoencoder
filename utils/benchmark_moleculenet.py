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
from sklearn.metrics import (average_precision_score, mean_squared_error,
                              roc_auc_score)


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


# Per-dataset canonical MoleculeNet protocols (split + primary metric):
#  * scaffold split for the structurally-driven tasks (BBBP, BACE, HIV) — labels
#    depend on 2D/3D structure, so scaffold-disjoint splits test generalization
#    to novel cores.
#  * random split for the broad-assay multi-task sets (Tox21, ToxCast, SIDER,
#    ClinTox, MUV) — labels are receptor/toxicity-driven, scaffold-disjointness
#    over-penalizes; random split matches how MoleculeNet originally reported.
#  * AUPRC for MUV (extreme imbalance, AUROC is misleading near 1.0 on rare
#    positives); AUROC for everything else.
DATASET_PROTOCOL = {
    # classification — scaffold split, ROC-AUC
    "BBBP":    {"split": "scaffold", "metric": "auroc"},
    "BACE":    {"split": "scaffold", "metric": "auroc"},
    "HIV":     {"split": "scaffold", "metric": "auroc"},
    # classification — random split, ROC-AUC
    "Tox21":   {"split": "random",   "metric": "auroc"},
    "ToxCast": {"split": "random",   "metric": "auroc"},
    "SIDER":   {"split": "random",   "metric": "auroc"},
    "ClinTox": {"split": "random",   "metric": "auroc"},
    # classification — random split, AUPRC (extreme imbalance)
    "MUV":     {"split": "random",   "metric": "auprc"},
    # regression — scaffold split, RMSE (sane default for the regression sets)
    "ESOL":    {"split": "scaffold", "metric": "rmse"},
    "FreeSolv":{"split": "scaffold", "metric": "rmse"},
    "Lipo":    {"split": "scaffold", "metric": "rmse"},
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
    """DeepChem `ScaffoldSplitter` — deterministic largest-first scaffold split.

    Matches the MoleculeNet paper protocol exactly (DeepChem default).
    """
    import deepchem as dc
    frac_test = 1.0 - frac_train - frac_val
    dataset = dc.data.NumpyDataset(
        X=np.zeros((len(smiles), 1)), ids=np.array(smiles, dtype=object),
    )
    splitter = dc.splits.ScaffoldSplitter()
    tr, va, te = splitter.split(
        dataset, frac_train=frac_train, frac_valid=frac_val, frac_test=frac_test,
    )
    return list(map(int, tr)), list(map(int, va)), list(map(int, te))


def random_split_3way(n: int, seed: int,
                       frac_train: float = 0.8, frac_val: float = 0.1
                       ) -> Tuple[List[int], List[int], List[int]]:
    """Pure-random 80/10/10 index permutation — NOT scaffold-disjoint.

    The MoleculeNet-recommended protocol for datasets where the labels are
    not driven by structural-novelty generalization: Tox21, ToxCast, SIDER,
    ClinTox, MUV. For these, scaffold-disjointness over-penalizes — random
    splitting matches the way these labels are actually used in practice.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n).tolist()
    n_train = int(frac_train * n)
    n_val = int(frac_val * n)
    return (perm[:n_train],
            perm[n_train:n_train + n_val],
            perm[n_train + n_val:])


def random_scaffold_split_3way(
    smiles: List[str], seed: int,
    frac_train: float = 0.8, frac_val: float = 0.1,
) -> Tuple[List[int], List[int], List[int]]:
    """Scaffold-disjoint split with per-seed-randomized group order.

    Same scaffold disjointness as `scaffold_split_3way` (no scaffold appears in
    two splits), but the order in which scaffold groups are assigned to
    train/val/test is shuffled with `seed`. Different seeds produce different
    splits, so positives/negatives naturally distribute across train/val/test
    across the seed range — fixes the extreme-imbalance pathology that the
    deterministic largest-first variant exhibits on ClinTox.

    This matches DeepChem's `RandomGroupSplitter` behavior, which appears to be
    Mole-BERT's actual MoleculeNet split (their ClinTox = 0.789 is incompatible
    with deterministic largest-first under the natural label distribution).
    """
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(smiles):
        groups[_murcko(s)].append(i)
    group_list = list(groups.values())
    rng = np.random.default_rng(seed)
    rng.shuffle(group_list)

    n = len(smiles)
    n_train, n_val = int(frac_train * n), int(frac_val * n)
    train, val, test = [], [], []
    for g in group_list:
        if len(train) + len(g) <= n_train:
            train.extend(g)
        elif len(val) + len(g) <= n_val:
            val.extend(g)
        else:
            test.extend(g)
    return train, val, test


def stratified_scaffold_split_3way(
    smiles: List[str], Y: np.ndarray,
    frac_train: float = 0.8, frac_val: float = 0.1,
) -> Tuple[List[int], List[int], List[int]]:
    """Scaffold-disjoint split that additionally ensures each binary task has
    both classes present in val and test (when feasible).

    Starts from the deterministic largest-first scaffold split, then for each
    task checks val/test class diversity; if a split is missing a class, moves
    the smallest train scaffold group containing the missing class into that
    split. No-op for datasets whose splits already have class diversity (e.g.
    BBBP, BACE, Tox21) — preserves prior results bit-for-bit when no swap is
    needed. Designed for ClinTox-style extreme class imbalance.
    """
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(smiles):
        groups[_murcko(s)].append(i)
    sorted_groups = sorted(groups.values(), key=lambda g: (-len(g), smiles[g[0]]))

    n = len(smiles)
    n_train, n_val = int(frac_train * n), int(frac_val * n)
    train, val, test = [], [], []
    train_groups: List[List[int]] = []
    val_groups:   List[List[int]] = []
    test_groups:  List[List[int]] = []
    for g in sorted_groups:
        if len(train) + len(g) <= n_train:
            train.extend(g); train_groups.append(g)
        elif len(val) + len(g) <= n_val:
            val.extend(g); val_groups.append(g)
        else:
            test.extend(g); test_groups.append(g)

    Y = np.asarray(Y)
    n_tasks = Y.shape[1] if Y.ndim == 2 else 1

    def _has_both_classes(idx_list, t):
        yt = Y[idx_list, t]
        yt = yt[~np.isnan(yt)]
        if len(yt) == 0:
            return True  # no labels — nothing to stratify
        return len(np.unique(yt)) >= 2

    def _smallest_train_group_with(t, target_class):
        best, best_size = None, float("inf")
        for tg in train_groups:
            yt = Y[tg, t]
            yt = yt[~np.isnan(yt)]
            if (yt == target_class).any() and len(tg) < best_size:
                best, best_size = tg, len(tg)
        return best

    swaps = 0
    for t in range(n_tasks):
        for split_idx, split_groups in ((val, val_groups), (test, test_groups)):
            if _has_both_classes(split_idx, t):
                continue
            yt = Y[split_idx, t]
            yt = yt[~np.isnan(yt)]
            present = int(yt[0]) if len(yt) > 0 else 1
            missing = 1 - present
            donor = _smallest_train_group_with(t, missing)
            if donor is None:
                continue
            for idx in donor:
                train.remove(idx)
                split_idx.append(idx)
            train_groups.remove(donor)
            split_groups.append(donor)
            swaps += 1

    if swaps:
        print(f"[stratify] moved {swaps} scaffold group(s) from train to "
               f"val/test to ensure class diversity")

    return train, val, test


# ──────────────────────────────────────────────────────────────────────────────
# Shared loss + eval — backend-agnostic.
# ──────────────────────────────────────────────────────────────────────────────
def _multitask_bce(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Masked BCE over non-NaN labels. logits/y: [B, T]."""
    mask = ~torch.isnan(y)
    if mask.sum() == 0:
        return logits.sum() * 0.0
    yt = torch.nan_to_num(y, nan=0.0)
    loss = F.binary_cross_entropy_with_logits(logits, yt, reduction="none")
    return (loss * mask.float()).sum() / mask.float().sum()


@torch.no_grad()
def _evaluate(model: nn.Module, prepared, Y: np.ndarray, idx, task: str,
              metric: str = "auroc", batch_size: int = 128) -> Tuple[float, float]:
    """Returns (primary_metric, val_loss).

    `metric` selects the classification metric: "auroc" (default) or "auprc"
    (used for MUV, where extreme imbalance makes AUROC near-trivially high).
    For regression, both returns are RMSE.

    val_loss is always-defined masked BCE (cls) / MSE (reg) — used as fallback
    selection when the primary metric is undefined on a degenerate val split.
    `model(prepared, indices)` must return logits.
    """
    model.eval()
    preds: List[np.ndarray] = []
    for i in range(0, len(idx), batch_size):
        bidx = idx[i:i + batch_size]
        logits = model(prepared, bidx).cpu().numpy()
        preds.append(logits)
    P = np.concatenate(preds, axis=0) if preds else np.zeros((0, Y.shape[1]))
    Yt = Y[idx]
    if task == "regression":
        rmse = float(np.sqrt(mean_squared_error(Yt[:, 0], P[:, 0])))
        return rmse, rmse
    # classification
    mask = ~np.isnan(Yt)
    y_safe = np.nan_to_num(Yt, nan=0.0)
    z = P
    bce = np.maximum(z, 0) - z * y_safe + np.log1p(np.exp(-np.abs(z)))
    loss = float((bce * mask).sum() / max(mask.sum(), 1))
    scorer = average_precision_score if metric == "auprc" else roc_auc_score
    scores: List[float] = []
    for t in range(Y.shape[1]):
        yt = Yt[:, t]; m = ~np.isnan(yt)
        if m.sum() == 0 or len(np.unique(yt[m])) < 2:
            continue
        scores.append(scorer(yt[m], P[m, t]))
    val = float(np.mean(scores)) if scores else float("nan")
    return val, loss


# ──────────────────────────────────────────────────────────────────────────────
# Backends — each handles its own data prep + model construction. The driver
# is generic: it calls `backend.prepare(smiles, device)` once per dataset and
# `backend.build_finetune_model(n_tasks)` once per seed. Models built by a
# backend must implement `forward(prepared, indices) -> logits` so the train
# loop is data-shape-agnostic.
# ──────────────────────────────────────────────────────────────────────────────
class _AEFinetuneWrapper(nn.Module):
    """Pairformer encoder + dropout + linear task head, end-to-end trainable."""
    def __init__(self, ae: nn.Module, latent_dim: int, n_tasks: int,
                 dropout: float, collate_fn: Callable):
        super().__init__()
        self.ae = ae
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(latent_dim, n_tasks)
        self.collate_fn = collate_fn

    def forward(self, prepared, indices):
        chunk = [prepared["samples"][i] for i in indices]
        batch = self.collate_fn(chunk).to(prepared["device"])
        mu = self.ae.encode(batch, sample=False)["mu"]
        return self.head(self.dropout(mu))

    def trainable_parameters(self):
        # Encoder path only — decoders are not in the forward graph but live
        # in `ae.parameters()`; excluding them avoids spurious weight decay.
        for mod in (self.ae.input_embedder, self.ae.pairformer, self.ae.aggregator):
            yield from mod.parameters()
        yield from self.dropout.parameters()
        yield from self.head.parameters()


class _DistillFinetuneWrapper(nn.Module):
    """SmilesEncoder + dropout + linear task head, end-to-end trainable."""
    def __init__(self, encoder: nn.Module, output_dim: int, n_tasks: int,
                 dropout: float):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(output_dim, n_tasks)

    def forward(self, prepared, indices):
        ids = prepared["ids"][indices]
        mask = prepared["mask"][indices]
        h = self.encoder(ids, mask, normalize=False)
        return self.head(self.dropout(h))


class AEBackend:
    """Pairformer mol_struct_ae fine-tune backend (RDKit ETKDG featurization)."""
    def __init__(self, checkpoint_path: str):
        import torch as _torch
        from mol_struct_ae import MolStructAutoencoder
        from mol_struct_ae.model import MolAEConfig
        from mol_struct_ae.dataset import make_collate_fn
        ckpt = _torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg_args = ckpt.get("config", {})
        self.cfg = MolAEConfig(
            max_atoms=cfg_args.get("max_atoms", 96),
            hidden_dim=cfg_args.get("hidden", 128),
            latent_dim=cfg_args.get("latent", 256),
        )
        self.state = ckpt["model"]
        self.collate_fn = make_collate_fn(self.cfg.max_atoms,
                                           cfg_args.get("max_dihedrals", 64))
        self._MolStructAutoencoder = MolStructAutoencoder
        print(f"[AEBackend] ckpt step={ckpt.get('step', '?')} "
               f"max_atoms={self.cfg.max_atoms} hidden={self.cfg.hidden_dim} "
               f"latent={self.cfg.latent_dim}")

    def prepare(self, smiles: List[str], device: torch.device
                ) -> Tuple[List[int], dict]:
        """ETKDG-featurize each SMILES; return (kept_indices, opaque_state)."""
        from utils.featurize import featurize_smiles
        samples, keep = [], []
        for i, smi in enumerate(smiles):
            s = featurize_smiles(smi, max_atoms=self.cfg.max_atoms)
            if s is None:
                continue
            samples.append(s); keep.append(i)
        return keep, {"samples": samples, "device": device}

    def build_finetune_model(self, n_tasks: int, dropout: float = 0.5) -> nn.Module:
        ae = self._MolStructAutoencoder(self.cfg)
        ae.load_state_dict(self.state)
        return _AEFinetuneWrapper(ae, self.cfg.latent_dim, n_tasks, dropout,
                                   self.collate_fn)


class DistillBackend:
    """SMILES-only distillation fine-tune backend (no featurization)."""
    def __init__(self, checkpoint_path: str, vocab_path: str):
        import torch as _torch
        from distillation.smiles_encoder import SmilesEncoder, SmilesEncoderConfig
        from distillation.smiles_tokenizer import SmilesTokenizer
        ckpt = _torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.cfg = SmilesEncoderConfig(**ckpt["encoder_config"])
        self.state = ckpt["model"]
        self.tok = SmilesTokenizer.load(vocab_path)
        if self.tok.max_len != self.cfg.max_len:
            self.tok = SmilesTokenizer(self.tok.vocab, max_len=self.cfg.max_len)
        self._SmilesEncoder = SmilesEncoder
        print(f"[DistillBackend] ckpt step={ckpt.get('step', '?')} "
               f"hidden={self.cfg.hidden_dim} layers={self.cfg.num_layers} "
               f"output_dim={self.cfg.output_dim}  val_cos_loss(best)="
               f"{ckpt.get('val_loss', '?')}")

    def prepare(self, smiles: List[str], device: torch.device
                ) -> Tuple[List[int], dict]:
        """Tokenize all SMILES once; reuse across all seeds."""
        keep = [i for i, s in enumerate(smiles) if s and isinstance(s, str)]
        smis_kept = [smiles[i] for i in keep]
        ids, mask = self.tok.encode_batch(smis_kept)
        return keep, {"ids": ids.to(device), "mask": mask.to(device)}

    def build_finetune_model(self, n_tasks: int, dropout: float = 0.5) -> nn.Module:
        enc = self._SmilesEncoder(self.cfg)
        enc.load_state_dict(self.state)
        return _DistillFinetuneWrapper(enc, self.cfg.output_dim, n_tasks, dropout)


# ──────────────────────────────────────────────────────────────────────────────
# Generic train loop — works with any backend whose model implements
# `forward(prepared, indices) -> logits`.
# ──────────────────────────────────────────────────────────────────────────────
def finetune_once(backend, prepared, Y: np.ndarray, splits, task: str,
                  device: torch.device, seed: int, n_tasks: int,
                  epochs: int = 100, batch_size: int = 32, lr: float = 1e-3,
                  dropout: float = 0.5, metric: str = "auroc") -> float:
    """One fine-tuning run. Returns TEST score (in `metric`) at the best-VAL epoch."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr, va, te = splits

    model = backend.build_finetune_model(n_tasks, dropout).to(device)
    # AE wrapper exposes a trimmed param iterator (skips decoders); the distill
    # wrapper doesn't have that distinction and uses .parameters().
    params = (model.trainable_parameters() if hasattr(model, "trainable_parameters")
              else model.parameters())
    opt = torch.optim.Adam(list(params), lr=lr)

    y_t = torch.from_numpy(Y).to(device)
    best_auc = -np.inf
    best_loss = np.inf
    best_test = float("nan")
    auc_ever_defined = False

    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(tr))
        for i in range(0, len(tr), batch_size):
            bidx = [tr[j] for j in order[i:i + batch_size]]
            logits = model(prepared, bidx)
            yb = y_t[bidx]
            if task == "classification":
                loss = _multitask_bce(logits, yb)
            else:
                loss = F.mse_loss(logits[:, 0], yb[:, 0])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        val_metric, val_loss = _evaluate(model, prepared, Y, va, task, metric=metric)
        if task == "classification" and not np.isnan(val_metric):
            auc_ever_defined = True
            if val_metric > best_auc:
                best_auc = val_metric
                test_metric, _ = _evaluate(model, prepared, Y, te, task, metric=metric)
                best_test = test_metric
        else:
            if not auc_ever_defined and val_loss < best_loss:
                best_loss = val_loss
                test_metric, _ = _evaluate(model, prepared, Y, te, task, metric=metric)
                best_test = test_metric
    return best_test


def run_finetune_benchmark(backend, datasets: List[str], seeds: List[int],
                           device: Optional[torch.device] = None,
                           epochs: int = 100, batch_size: int = 32,
                           lr: float = 1e-3, dropout: float = 0.5,
                           train_smiles_path: Optional[str] = None,
                           split_override: Optional[str] = None,
                           metric_override: Optional[str] = None) -> List[dict]:
    """Each dataset is run under its canonical MoleculeNet protocol from
    `DATASET_PROTOCOL`. `split_override` / `metric_override` force the same
    protocol on every dataset for ablation studies (e.g. "all scaffold"
    comparisons).

    Splits:
      "scaffold" — Bemis-Murcko scaffold groups, sorted largest-first then
                   greedily allocated to train/val/test. Deterministic across
                   seeds (only model init / minibatch order varies). Matches
                   DeepChem's `ScaffoldSplitter` and the MoleculeNet paper.
      "random"   — pure-random index permutation per seed. Not
                   scaffold-disjoint. The canonical MoleculeNet protocol
                   for the broad-assay multi-task sets.

    Metrics: "auroc" (default for classification), "auprc" (MUV), "rmse"
             (regression — selected automatically when task=="regression").
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        # Resolve protocol for this dataset
        proto = DATASET_PROTOCOL.get(name, {"split": "scaffold", "metric": "auroc"})
        ds_split = split_override or proto["split"]
        ds_metric = (metric_override or proto["metric"]) if task == "classification" else "rmse"
        print(f"[{name}] {n_raw} mols, {len(task_names)} task(s); "
               f"protocol: split={ds_split} metric={ds_metric}")

        keep_idx, prepared = backend.prepare(smis, device)
        smis_kept = [smis[i] for i in keep_idx]
        Y_kept = Y[keep_idx]
        print(f"[{name}] kept {len(smis_kept)}/{n_raw} after prepare")

        scores: List[float] = []
        last_tr = last_va = last_te = []
        for sd in seeds:
            if ds_split == "scaffold":
                tr, va, te = random_scaffold_split_3way(smis_kept, seed=sd)
            elif ds_split == "random":
                tr, va, te = random_split_3way(len(smis_kept), seed=sd)
            else:
                raise ValueError(f"unknown split {ds_split!r} for {name}")
            print(f"[{name}] 80/10/10 ({ds_split}, seed={sd}) → "
                   f"train={len(tr)} val={len(va)} test={len(te)}")
            last_tr, last_va, last_te = tr, va, te
            s = finetune_once(backend, prepared, Y_kept, (tr, va, te), task,
                              device, seed=sd, n_tasks=Y_kept.shape[1],
                              epochs=epochs, batch_size=batch_size, lr=lr,
                              dropout=dropout, metric=ds_metric)
            scores.append(s)
            print(f"[{name}] seed={sd} test={s:.4f}")
        arr = np.array(scores, dtype=np.float64)
        metric_label = {"auroc": "AUROC", "auprc": "AUPRC", "rmse": "RMSE"}[ds_metric]
        print(f"[{name}] {metric_label} = {np.nanmean(arr):.4f} ± {np.nanstd(arr):.4f} "
               f"over {len(seeds)} seeds  ({time.time()-t0:.0f}s)")
        results.append({
            "dataset": name, "task": task, "metric": metric_label,
            "split": ds_split,
            "n_tasks": Y_kept.shape[1], "n_kept": len(smis_kept),
            "n_train": len(last_tr), "n_val": len(last_va), "n_test": len(last_te),
            "mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr)),
            "scores": scores,
        })
    return results
