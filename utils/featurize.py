"""
Offline featurization of an arbitrary SMILES CSV → sharded `MolSample` lists.

Works on any CSV with a SMILES column (default name `smiles`, override with `--smiles-col`). Supports a `--limit`
flag for partial runs and an idempotent resume (counts existing shards and
skips that many input rows).

For each SMILES:
  1. RDKit parse + canonicalize + ETKDG conformer + MMFF optimize.
  2. Atom features (32-D): one-hot atomic num (top-10) + formal charge +
     hybridization one-hot + aromaticity + num H + chirality + in-ring.
  3. Bond features (8-D): bond-type one-hot + stereo + in-ring + conjugated.
  4. SE(3)-invariant 3D scalars: atom-type vector (16-D — atomic num + radii +
     mass + degree + ring flags), bond lengths, bond angles (sparse triples),
     dihedral angles (sparse quadruples — rotatable + ring torsions). Raw xyz
     coordinates are used to derive these scalars and then discarded — the
     model itself is coordinate-free.
  5. Pharmacophore (8-D): donor, acceptor, aromatic, hydrophobic, +charge,
     -charge, halogen, ring.
  6. Gasteiger partial charges (per atom).
  7. Canonical SMILES string stored on the sample so the embedding pipeline can
     recover molecule identity from shards without re-featurizing.

Each shard is a .pt file containing a list of `MolSample` objects (sparse —
no padding). The training-time DataLoader does the collate→dense step.

CLI:
    python utils/featurize.py \
        --csv data/zinc250k.csv --out data/shards \
        --max-atoms 64 --shard-size 1024 --workers 8
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, ChemicalFeatures
from rdkit import RDConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mol_struct_ae.data import (ATOM_FEAT_DIM, ATYPE_DIM, BOND_FEAT_DIM,
                                  PHARMA_FEAT_DIM, MolSample)

RDLogger.DisableLog("rdApp.*")


# ---- Feature vocab ----------------------------------------------------------
TOP_ATOMS = [6, 7, 8, 9, 15, 16, 17, 35, 53, 1]    # C, N, O, F, P, S, Cl, Br, I, H
HYBRIDIZATIONS = [
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]

# Periodic-table table lookups for atom-type vector (period, group, radii, EN)
PERIODIC_TABLE = Chem.GetPeriodicTable()


# ---- Per-atom feature extraction --------------------------------------------
def atom_feature_vec(atom: Chem.Atom) -> np.ndarray:
    """Pad/truncate to ATOM_FEAT_DIM (=32) — see mol_struct_ae.data."""
    feat = np.zeros(ATOM_FEAT_DIM, dtype=np.float32)
    z = atom.GetAtomicNum()
    if z in TOP_ATOMS:
        feat[TOP_ATOMS.index(z)] = 1.0                              # 0..9
    feat[10] = atom.GetFormalCharge()
    h_idx = HYBRIDIZATIONS.index(atom.GetHybridization()) if atom.GetHybridization() in HYBRIDIZATIONS else -1
    if h_idx >= 0:
        feat[11 + h_idx] = 1.0                                      # 11..15
    feat[16] = float(atom.GetIsAromatic())
    feat[17] = atom.GetTotalNumHs()
    feat[18] = float(atom.IsInRing())
    chir = atom.GetChiralTag()
    feat[19] = float(chir == Chem.ChiralType.CHI_TETRAHEDRAL_CW)
    feat[20] = float(chir == Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
    feat[21] = atom.GetDegree()
    feat[22] = atom.GetExplicitValence()
    feat[23] = atom.GetImplicitValence()
    feat[24] = float(atom.GetIsotope() != 0)
    # 25..31 reserved for future flags — stays zero
    return feat


def atom_type_vec(atom: Chem.Atom) -> np.ndarray:
    """16-D numeric atom-type vector (continuous, generalizes better than one-hot)."""
    feat = np.zeros(ATYPE_DIM, dtype=np.float32)
    z = atom.GetAtomicNum()
    feat[0] = z / 50.0                                              # normalized atomic num
    feat[1] = PERIODIC_TABLE.GetRcovalent(z) or 0.0
    feat[2] = PERIODIC_TABLE.GetRvdw(z) or 0.0
    feat[3] = float(atom.GetIsAromatic())
    feat[4] = atom.GetFormalCharge()
    feat[5] = atom.GetTotalNumHs()
    feat[6] = atom.GetMass() / 100.0
    feat[7] = atom.GetDegree()
    feat[8] = float(atom.IsInRing())
    feat[9] = float(atom.IsInRingSize(5))
    feat[10] = float(atom.IsInRingSize(6))
    # 11..15 reserved
    return feat


def bond_feature_vec(bond: Chem.Bond) -> np.ndarray:
    feat = np.zeros(BOND_FEAT_DIM, dtype=np.float32)
    bt = bond.GetBondType()
    if bt in BOND_TYPES:
        feat[BOND_TYPES.index(bt)] = 1.0                            # 0..3
    feat[4] = float(bond.GetStereo() != Chem.BondStereo.STEREONONE)
    feat[5] = float(bond.IsInRing())
    feat[6] = float(bond.GetIsConjugated())
    feat[7] = float(bond.GetIsAromatic())
    return feat


# ---- Pharmacophore extraction -----------------------------------------------
_FEAT_FACTORY: Optional[ChemicalFeatures.MolChemicalFeatureFactory] = None


def get_factory():
    global _FEAT_FACTORY
    if _FEAT_FACTORY is None:
        fdef = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        _FEAT_FACTORY = ChemicalFeatures.BuildFeatureFactory(fdef)
    return _FEAT_FACTORY


def pharma_feature_matrix(mol: Chem.Mol) -> np.ndarray:
    """8-D binary labels per atom: donor, acceptor, aromatic, hydrophobic,
    +charge, -charge, halogen, ring."""
    N = mol.GetNumAtoms()
    feat = np.zeros((N, PHARMA_FEAT_DIM), dtype=np.float32)
    factory = get_factory()
    label_map = {"Donor": 0, "Acceptor": 1, "Aromatic": 2, "Hydrophobe": 3,
                 "PosIonizable": 4, "NegIonizable": 5}
    for f in factory.GetFeaturesForMol(mol):
        col = label_map.get(f.GetFamily())
        if col is None:
            continue
        for aid in f.GetAtomIds():
            feat[aid, col] = 1.0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in (9, 17, 35, 53):
            feat[atom.GetIdx(), 6] = 1.0
        if atom.IsInRing():
            feat[atom.GetIdx(), 7] = 1.0
    return feat


# ---- Bond angles + dihedrals from topology + coords -------------------------
def compute_angles(mol: Chem.Mol, coords: np.ndarray):
    """Bond angles at every bonded atom j with two distinct bonded neighbors i, k."""
    triples = []
    angles = []
    for j_atom in mol.GetAtoms():
        j = j_atom.GetIdx()
        neighbors = [nb.GetIdx() for nb in j_atom.GetNeighbors()]
        for a in range(len(neighbors)):
            for b in range(a + 1, len(neighbors)):
                i, k = neighbors[a], neighbors[b]
                vij = coords[i] - coords[j]
                vkj = coords[k] - coords[j]
                cos = np.dot(vij, vkj) / max(np.linalg.norm(vij) * np.linalg.norm(vkj), 1e-6)
                triples.append((i, j, k))
                angles.append(math.acos(min(1 - 1e-6, max(-1 + 1e-6, cos))))
    if not triples:
        return torch.zeros(3, 0, dtype=torch.long), torch.zeros(0)
    ai = torch.tensor(triples, dtype=torch.long).t()                # [3, A]
    return ai, torch.tensor(angles, dtype=torch.float32)


def compute_dihedrals(mol: Chem.Mol, coords: np.ndarray, max_torsions: int = 64):
    """Signed dihedrals over every chain i-j-k-l of bonded atoms."""
    quads = []
    for j_atom in mol.GetAtoms():
        j = j_atom.GetIdx()
        for k_atom in j_atom.GetNeighbors():
            k = k_atom.GetIdx()
            if k <= j:
                continue                                            # avoid duplicates
            for i_atom in j_atom.GetNeighbors():
                i = i_atom.GetIdx()
                if i == k:
                    continue
                for l_atom in k_atom.GetNeighbors():
                    l = l_atom.GetIdx()
                    if l == j or l == i:
                        continue
                    quads.append((i, j, k, l))
                    if len(quads) >= max_torsions:
                        break
                if len(quads) >= max_torsions:
                    break
            if len(quads) >= max_torsions:
                break
        if len(quads) >= max_torsions:
            break
    if not quads:
        return torch.zeros(4, 0, dtype=torch.long), torch.zeros(0)
    arr = np.array(quads)
    p0, p1, p2, p3 = coords[arr[:, 0]], coords[arr[:, 1]], coords[arr[:, 2]], coords[arr[:, 3]]
    b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
    n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2, axis=-1, keepdims=True).clip(min=1e-6))
    x = (n1 * n2).sum(-1); y = (m1 * n2).sum(-1)
    angles = np.arctan2(y, x)
    return torch.tensor(arr.T, dtype=torch.long), torch.tensor(angles, dtype=torch.float32)


# ---- Top-level: SMILES → MolSample ------------------------------------------
def featurize_smiles(smiles: str, max_atoms: int = 64,
                      max_torsions: int = 64) -> Optional[MolSample]:
    smiles = smiles.strip().strip('"')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canonical_smiles = Chem.MolToSmiles(mol)
    mol = Chem.AddHs(mol)
    if mol.GetNumAtoms() > max_atoms:
        return None
    # Embed 3D conformer; bail if it fails.
    if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) < 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        return None
    N = mol.GetNumAtoms()
    coords = mol.GetConformer().GetPositions().astype(np.float32)

    # Atom-level features
    atom_feats = np.stack([atom_feature_vec(a) for a in mol.GetAtoms()])
    atom_types = np.stack([atom_type_vec(a) for a in mol.GetAtoms()])
    pharma = pharma_feature_matrix(mol)
    charges = np.array([float(a.GetProp("_GasteigerCharge")) if a.HasProp("_GasteigerCharge") else 0.0
                         for a in mol.GetAtoms()], dtype=np.float32)
    charges = np.nan_to_num(charges, nan=0.0, posinf=0.0, neginf=0.0)

    # Bond-level features (both directions for the undirected graph)
    src, dst, bond_feats_list, bond_lengths = [], [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_feature_vec(bond)
        d = float(np.linalg.norm(coords[i] - coords[j]))
        for s, t in ((i, j), (j, i)):
            src.append(s); dst.append(t)
            bond_feats_list.append(bf); bond_lengths.append(d)
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros(2, 0, dtype=torch.long)
    bond_feats = torch.tensor(np.stack(bond_feats_list) if bond_feats_list else np.zeros((0, BOND_FEAT_DIM)),
                               dtype=torch.float32)
    bond_lengths_t = torch.tensor(bond_lengths, dtype=torch.float32) if bond_lengths else torch.zeros(0)

    # Angles + dihedrals (derived scalars; coords themselves are not stored)
    angle_index, bond_angles = compute_angles(mol, coords)
    dihedral_index, dihedral_angles = compute_dihedrals(mol, coords, max_torsions=max_torsions)

    return MolSample(
        atom_feats_2d=torch.tensor(atom_feats, dtype=torch.float32),
        bond_feats_2d=bond_feats,
        edge_index_2d=edge_index,
        atom_types=torch.tensor(atom_types, dtype=torch.float32),
        bond_lengths=bond_lengths_t,
        angle_index=angle_index,
        bond_angles=bond_angles,
        partial_charges=torch.tensor(charges, dtype=torch.float32),
        pharma_feats=torch.tensor(pharma, dtype=torch.float32),
        dihedral_index=dihedral_index,
        dihedral_angles=dihedral_angles,
        smiles=canonical_smiles,
    )


# ---- CLI -------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--smiles-col", default="smiles")
    p.add_argument("--max-atoms", type=int, default=64)
    p.add_argument("--max-torsions", type=int, default=64)
    p.add_argument("--shard-size", type=int, default=1024)
    p.add_argument("--limit", type=int, default=0, help="0 = no limit")
    p.add_argument("--workers", type=int, default=0,
                    help="0 = serial. >0 = multiprocessing pool with this many workers")
    p.add_argument("--start", type=int, default=0, help="line offset into CSV")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Resume support ----
    # If shards already exist, count kept molecules and skip that many input
    # rows (× ~1.01 to account for the small fraction dropped at featurize time).
    # Resumes from `next_shard_id = len(existing)`; mild duplication of a few
    # boundary molecules is harmless.
    existing_shards = sorted(out.glob("shard_*.pt"))
    skip_rows = 0
    next_shard_id = 0
    if existing_shards:
        kept_so_far = 0
        for path in existing_shards:
            kept_so_far += len(torch.load(path, weights_only=False))
        skip_rows = int(kept_so_far * 1.01) + 50
        next_shard_id = len(existing_shards)
        print(f"[featurize] RESUME: {len(existing_shards)} existing shards, "
               f"{kept_so_far} kept; skipping first ~{skip_rows} input rows, "
               f"next shard_id={next_shard_id}")

    # Stream smiles to keep memory low even at 250k rows.
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        rows = [r[args.smiles_col] for r in reader]
    base_offset = args.start + skip_rows
    if args.limit > 0:
        rows = rows[base_offset:base_offset + args.limit]
    else:
        rows = rows[base_offset:]
    print(f"[featurize] {len(rows)} SMILES from {args.csv} (offset={base_offset})")

    shard: List[MolSample] = []
    shard_id, kept, dropped, t0 = next_shard_id, 0, 0, time.time()

    if args.workers > 0:
        from multiprocessing import Pool
        worker_kwargs = dict(max_atoms=args.max_atoms, max_torsions=args.max_torsions)
        from functools import partial
        fn = partial(featurize_smiles, **worker_kwargs)
        with Pool(args.workers) as pool:
            iterator = pool.imap_unordered(fn, rows, chunksize=32)
            for sample in iterator:
                if sample is None:
                    dropped += 1; continue
                shard.append(sample); kept += 1
                if len(shard) >= args.shard_size:
                    torch.save(shard, out / f"shard_{shard_id:05d}.pt")
                    shard_id += 1; shard = []
                if (kept + dropped) % 1000 == 0:
                    rate = (kept + dropped) / (time.time() - t0 + 1e-6)
                    print(f"[featurize] processed={kept+dropped} kept={kept} dropped={dropped}"
                           f" rate={rate:.1f}/s")
    else:
        for i, smi in enumerate(rows):
            sample = featurize_smiles(smi, max_atoms=args.max_atoms, max_torsions=args.max_torsions)
            if sample is None:
                dropped += 1; continue
            shard.append(sample); kept += 1
            if len(shard) >= args.shard_size:
                torch.save(shard, out / f"shard_{shard_id:05d}.pt")
                shard_id += 1; shard = []
            if (i + 1) % 500 == 0:
                rate = (i + 1) / (time.time() - t0 + 1e-6)
                print(f"[featurize] processed={i+1}/{len(rows)} kept={kept} dropped={dropped}"
                       f" rate={rate:.1f}/s")

    if shard:
        torch.save(shard, out / f"shard_{shard_id:05d}.pt")

    print(f"[featurize] DONE: kept={kept} dropped={dropped} shards={shard_id + (1 if shard else 0)}")


if __name__ == "__main__":
    main()
