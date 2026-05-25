"""
Subsample a SMILES CSV with **scaffold-aware** diversity selection.

Algorithm:
  1. Compute the Bemis-Murcko scaffold for every SMILES (parallel RDKit).
  2. Group SMILES by scaffold → dict[scaffold_smiles → list[smi]].
  3. Round-robin pick across scaffolds until `--target-count` is reached:
       round 1: 1 mol from each scaffold     (≈ all unique scaffolds)
       round 2: 2nd mol from each scaffold (where available)
       round 3: 3rd mol from each scaffold ...
     This caps how many molecules any single scaffold can contribute before
     under-represented scaffolds get a chance — giving a more uniform spread
     across chemical space than uniform random sampling.

Why scaffolds (Bemis-Murcko) and not random sampling:
  ZINC15-10M is a curated lead-like set, but scaffold frequency is highly
  skewed (a handful of common cores dominate). Random sampling preserves that
  skew; scaffold round-robin actively flattens it. The result trains a more
  generally-applicable representation.

CLI:
    python scripts/diversify_smiles.py \\
        --in-csv runs/zinc15_10m.csv \\
        --out-csv runs/zinc15_5m_diverse.csv \\
        --target-count 5000000 \\
        --workers 32
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


def get_scaffold(smi: str) -> tuple[str, str]:
    """Return (scaffold_smiles, original_smiles).
    Acyclic molecules → empty-string scaffold bucket.
    Invalid SMILES   → return (None, smi) so caller can drop them.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, smi
    try:
        scaff_mol = MurckoScaffold.GetScaffoldForMol(mol)
        scaff_smi = Chem.MolToSmiles(scaff_mol) if scaff_mol.GetNumAtoms() else ""
    except Exception:
        scaff_smi = ""
    return scaff_smi, smi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-csv", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--smiles-col", default="smiles")
    p.add_argument("--target-count", type=int, default=5_000_000)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # ---- Read SMILES
    print(f"[diversify] reading {args.in_csv}", flush=True)
    with open(args.in_csv) as f:
        reader = csv.DictReader(f)
        smiles_list = [r[args.smiles_col].strip().strip('"') for r in reader
                        if r.get(args.smiles_col)]
    print(f"[diversify] read {len(smiles_list):,} SMILES", flush=True)

    # ---- Scaffold computation (parallel)
    print(f"[diversify] computing Bemis-Murcko scaffolds on {args.workers} workers",
           flush=True)
    by_scaffold: dict[str, list[str]] = defaultdict(list)
    t0 = time.time()
    n_invalid = 0
    with Pool(args.workers) as pool:
        for i, (scaff, smi) in enumerate(pool.imap_unordered(get_scaffold,
                                                              smiles_list,
                                                              chunksize=500)):
            if scaff is None:
                n_invalid += 1
            else:
                by_scaffold[scaff].append(smi)
            if (i + 1) % 200_000 == 0:
                rate = (i + 1) / max(time.time() - t0, 1e-6)
                print(f"[scaffold] {i+1:>10,}/{len(smiles_list):,} "
                       f"scaffolds={len(by_scaffold):>9,} rate={rate:.0f}/s",
                       flush=True)
    n_scaffolds = len(by_scaffold)
    avg_per = sum(len(v) for v in by_scaffold.values()) / max(n_scaffolds, 1)
    print(f"[diversify] {n_scaffolds:,} unique scaffolds, avg {avg_per:.1f} mols/scaffold, "
           f"{n_invalid:,} invalid SMILES dropped\n", flush=True)

    # ---- Round-robin pick across scaffolds
    random.seed(args.seed)
    scaffolds = list(by_scaffold.keys())
    random.shuffle(scaffolds)
    for s in scaffolds:
        random.shuffle(by_scaffold[s])
    pick_idx = {s: 0 for s in scaffolds}

    selected: list[str] = []
    t1 = time.time()
    target = min(args.target_count, sum(len(v) for v in by_scaffold.values()))
    round_no = 0
    while len(selected) < target:
        round_no += 1
        added_this_round = 0
        for s in scaffolds:
            if pick_idx[s] < len(by_scaffold[s]):
                selected.append(by_scaffold[s][pick_idx[s]])
                pick_idx[s] += 1
                added_this_round += 1
                if len(selected) >= target:
                    break
        rate = len(selected) / max(time.time() - t1, 1e-6)
        print(f"[round {round_no}] picked {added_this_round:,} → total {len(selected):,}/"
               f"{target:,} rate={rate:.0f}/s", flush=True)
        if added_this_round == 0:
            print(f"[diversify] exhausted all scaffolds at {len(selected):,}")
            break

    # ---- Write CSV
    print(f"\n[diversify] writing {len(selected):,} SMILES → {args.out_csv}",
           flush=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["smiles"])
        for s in selected:
            w.writerow([s])
    print("[diversify] done.")


if __name__ == "__main__":
    main()
