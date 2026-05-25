"""
LIT-PCBA — Virtual screening benchmark (Tran-Nguyen, Jacquemard, Rognan 2020).

15 protein targets with experimentally verified actives + property-matched
decoys. The standard "did your embedding learn enough chemistry to do real
screening?" test.

Protocol per target:
  1. Download actives.smi + inactives.smi from LIT-PCBA mirror.
  2. Subsample decoys to a tractable size (default 10,000) — uniform random.
  3. Featurize actives + decoys in parallel; embed with model + Morgan FP.
  4. Pick K query actives (default 5) uniformly at random; the rest are held-out
     actives.
  5. For every NON-query molecule, score = max cosine sim to ANY query.
  6. Rank descending. Report:
        EF@1%, EF@5%   — enrichment over random
        AUROC          — ranking-based area under ROC
  7. Repeat for several random query draws; average across draws to stabilize.

Headlines: a random model gives EF@1% ≈ 1.0. Good models on tractable
LIT-PCBA targets land in 3-20× range. SOTA virtual-screening models can hit
30-100× on easier targets like ESR1_ago.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from sklearn.metrics import roc_auc_score

RDLogger.DisableLog("rdApp.*")


# NOTE: PyTDC does not include the exact LIT-PCBA dataset. We use PyTDC's HTS
# (high-throughput screening) datasets instead — these are also PubChem BioAssay
# screens with binary actives/decoys, and answer the same scientific question
# ("given known actives, can the embedding rank more actives above decoys?").
# Available HTS datasets (from PyTDC v0.4.x):
ALL_TARGETS = [
    "hiv",
    "sarscov2_3clpro_diamond",
    "sarscov2_vitro_touret",
    "orexin1_receptor_butkiewicz",
    "m1_muscarinic_receptor_agonists_butkiewicz",
    "m1_muscarinic_receptor_antagonists_butkiewicz",
    "potassium_ion_channel_kir2.1_butkiewicz",
    "kcnq2_potassium_channel_butkiewicz",
    "cav3_t-type_calcium_channels_butkiewicz",
    "choline_transporter_butkiewicz",
    "serine_threonine_kinase_33_butkiewicz",
    "tyrosyl-dna_phosphodiesterase_butkiewicz",
]

# Tractable subset covering 4 diverse biological targets
DEFAULT_SUBSET = [
    "sarscov2_vitro_touret",                     # SARS-CoV-2 antiviral (small, fast)
    "orexin1_receptor_butkiewicz",                # GPCR
    "m1_muscarinic_receptor_agonists_butkiewicz", # GPCR
    "kcnq2_potassium_channel_butkiewicz",         # ion channel
]


def _load_via_tdc_hts(target: str, tdc_path: str):
    """Load a virtual-screening dataset by name via PyTDC's HTS interface."""
    import os
    os.makedirs(tdc_path, exist_ok=True)
    from tdc.single_pred import HTS
    data = HTS(name=target, path=tdc_path)
    df = data.get_data()
    smi_col = "Drug" if "Drug" in df.columns else df.columns[0]
    y_col = "Y" if "Y" in df.columns else df.columns[-1]
    actives = df.loc[df[y_col] == 1, smi_col].astype(str).tolist()
    decoys = df.loc[df[y_col] == 0, smi_col].astype(str).tolist()
    print(f"  [tdc] loaded HTS({target!r}): actives={len(actives)} decoys={len(decoys)}")
    return actives, decoys


def load_target(target: str, max_decoys: int = 10000,
                 seed: int = 0,
                 tdc_path: str = "./tdc_data") -> Tuple[List[str], List[str]]:
    """Returns (actives_smis, decoys_smis) for one LIT-PCBA target via PyTDC.

    Tries TDC's HTS loader (the only working programmatic source for LIT-PCBA).
    Decoys are uniformly subsampled to `max_decoys` for tractability.
    """
    actives, decoys = _load_via_tdc_hts(target, tdc_path)
    if len(decoys) > max_decoys:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(decoys), size=max_decoys, replace=False)
        decoys = [decoys[i] for i in sorted(idx)]
    return actives, decoys


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
    # L2 normalize so cosine == dot product
    norms = np.linalg.norm(out, axis=1, keepdims=True).clip(min=1e-8)
    return out / norms


@torch.no_grad()
def embed_with_model(samples, model, collate_fn: Callable, device,
                      batch_size: int = 256) -> np.ndarray:
    embs: List[torch.Tensor] = []
    for i in range(0, len(samples), batch_size):
        batch = collate_fn(samples[i:i + batch_size]).to(device)
        out = model(batch, sample=False)
        e = F.normalize(out["sim_embed"], dim=-1).cpu()
        embs.append(e)
    return torch.cat(embs, dim=0).numpy().astype(np.float32)


def _ef_at_k(scores: np.ndarray, labels: np.ndarray, k_frac: float) -> float:
    """Enrichment factor at top k_frac of the ranking."""
    n = len(scores)
    n_actives = int(labels.sum())
    if n_actives == 0:
        return float("nan")
    k = max(1, int(np.ceil(n * k_frac)))
    order = np.argsort(-scores)[:k]
    hits_in_topk = int(labels[order].sum())
    return (hits_in_topk / k) / (n_actives / n)


def _eval_target(E_act: np.ndarray, E_dec: np.ndarray, n_queries: int = 5,
                  n_draws: int = 10, seed: int = 0) -> Dict[str, float]:
    """Repeated leave-N-out: pick `n_queries` actives as queries, rank everything
    else by max cosine to a query. Average EF/AUROC over draws."""
    rng = np.random.default_rng(seed)
    Na = len(E_act); Nd = len(E_dec)
    metrics = {"EF@1%": [], "EF@5%": [], "AUROC": []}
    for _ in range(n_draws):
        q_idx = rng.choice(Na, size=min(n_queries, Na - 1), replace=False)
        q_mask = np.zeros(Na, dtype=bool); q_mask[q_idx] = True
        Q = E_act[q_mask]                      # [n_q, D]

        # pool = remaining actives + all decoys
        pool_act = E_act[~q_mask]              # [Na-nq, D]
        pool = np.concatenate([pool_act, E_dec], axis=0)
        labels = np.concatenate([np.ones(len(pool_act)), np.zeros(Nd)]).astype(int)

        # cosine matrix → max over queries → similarity score per pool item
        sims = pool @ Q.T                       # [pool, n_q]
        scores = sims.max(axis=1)

        metrics["EF@1%"].append(_ef_at_k(scores, labels, 0.01))
        metrics["EF@5%"].append(_ef_at_k(scores, labels, 0.05))
        try:
            metrics["AUROC"].append(float(roc_auc_score(labels, scores)))
        except Exception:
            metrics["AUROC"].append(float("nan"))
    return {k: float(np.nanmean(v)) for k, v in metrics.items()}


def run_litpcba(model, collate_fn: Callable, device, max_atoms: int,
                 targets: Optional[List[str]] = None,
                 max_decoys: int = 10000,
                 n_queries: int = 5,
                 n_draws: int = 10,
                 workers: int = 8,
                 embed_batch_size: int = 256,
                 train_smiles_path: Optional[str] = None) -> List[dict]:
    """Run LIT-PCBA VS benchmark.

    If `train_smiles_path` is provided, drop any active or decoy whose
    canonical (stereo-stripped) SMILES is in the pretraining set, to
    prevent leakage from inflating enrichment.
    """
    from utils.parallel_featurize import featurize_smiles_parallel
    from utils.moleculenet_benchmark import load_train_smiles_set

    train_set: Optional[set] = None
    if train_smiles_path:
        print(f"[bench] loading leakage set from {train_smiles_path} …")
        train_set = load_train_smiles_set(train_smiles_path)
        print(f"[bench] loaded {len(train_set):,} unique training SMILES")

    def _filter_leak(smis: List[str]) -> Tuple[List[str], int]:
        if train_set is None:
            return list(smis), 0
        kept = []; n_leak = 0
        for s in smis:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            if Chem.MolToSmiles(m, isomericSmiles=False) in train_set:
                n_leak += 1
                continue
            kept.append(s)
        return kept, n_leak

    targets = targets or DEFAULT_SUBSET
    results: List[dict] = []
    for target in targets:
        t0 = time.time()
        print(f"\n[{target}] downloading actives + inactives …")
        try:
            actives_smis, decoys_smis = load_target(target, max_decoys=max_decoys)
        except Exception as e:
            print(f"[{target}] FAILED to load: {type(e).__name__}: {e}")
            results.append({"target": target, "error": str(e)})
            continue
        n_act_raw = len(actives_smis)
        n_dec_raw = len(decoys_smis)
        actives_smis, leak_act = _filter_leak(actives_smis)
        decoys_smis, leak_dec = _filter_leak(decoys_smis)
        if train_set is not None:
            print(f"[{target}] leakage filter: dropped {leak_act}/{n_act_raw} actives, "
                   f"{leak_dec}/{n_dec_raw} decoys")
        print(f"[{target}] actives={len(actives_smis)}  decoys(sampled)={len(decoys_smis)}")

        # Featurize actives + decoys together so we batch-embed once
        all_smis = actives_smis + decoys_smis
        labels_orig = np.concatenate(
            [np.ones(len(actives_smis)), np.zeros(len(decoys_smis))]
        ).astype(int)

        print(f"[{target}] featurizing in parallel (workers={workers}) …")
        samples, keep_idx = featurize_smiles_parallel(
            all_smis, max_atoms=max_atoms, workers=workers,
        )
        if not samples:
            print(f"[{target}] no featurizable molecules — skipping")
            results.append({"target": target, "error": "no featurizable"})
            continue
        kept_smis = [all_smis[i] for i in keep_idx]
        kept_labels = labels_orig[keep_idx]
        n_act_kept = int(kept_labels.sum())
        n_dec_kept = int((1 - kept_labels).sum())
        if n_act_kept < n_queries + 1:
            print(f"[{target}] too few actives kept ({n_act_kept}) — skipping")
            results.append({"target": target, "error": f"only {n_act_kept} actives"})
            continue

        print(f"[{target}] embedding {len(samples)} mols …")
        E_all = embed_with_model(samples, model, collate_fn, device, embed_batch_size)
        E_fp = morgan_fp_matrix(kept_smis)

        E_act_m = E_all[kept_labels == 1]
        E_dec_m = E_all[kept_labels == 0]
        E_act_fp = E_fp[kept_labels == 1]
        E_dec_fp = E_fp[kept_labels == 0]

        m_metrics = _eval_target(E_act_m, E_dec_m, n_queries=n_queries, n_draws=n_draws)
        fp_metrics = _eval_target(E_act_fp, E_dec_fp, n_queries=n_queries, n_draws=n_draws)

        dt = time.time() - t0
        print(f"[{target}] model EF@1%={m_metrics['EF@1%']:.2f}  EF@5%={m_metrics['EF@5%']:.2f}  "
               f"AUROC={m_metrics['AUROC']:.3f}")
        print(f"[{target}] morgan EF@1%={fp_metrics['EF@1%']:.2f}  EF@5%={fp_metrics['EF@5%']:.2f}  "
               f"AUROC={fp_metrics['AUROC']:.3f}   ({dt:.0f}s)")

        results.append({
            "target": target,
            "n_actives": n_act_kept, "n_decoys": n_dec_kept,
            "n_act_raw": n_act_raw, "n_dec_raw": n_dec_raw,
            "leak_actives": leak_act, "leak_decoys": leak_dec,
            "model_EF1": m_metrics["EF@1%"], "model_EF5": m_metrics["EF@5%"],
            "model_AUROC": m_metrics["AUROC"],
            "fp_EF1": fp_metrics["EF@1%"], "fp_EF5": fp_metrics["EF@5%"],
            "fp_AUROC": fp_metrics["AUROC"],
        })
    return results
