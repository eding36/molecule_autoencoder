"""
Data containers for a multi-track molecular autoencoder.

A `MolSample` is one molecule with all four track inputs:
  * 2D graph  – atom node features, bond features, edge_index (sparse)
  * 3D track  – atom types, bond_lengths (per edge), bond_angles (per angle
                triple), angle_index (which atoms form the angle). All scalars;
                raw xyz is used during featurization to derive these and then
                discarded — the model is SE(3)-invariant by construction.
  * Charge    – per-atom Gasteiger partial charges
  * Pharma    – per-atom pharmacophore labels (donor/acceptor/aromatic/...)

A `MolBatch` is the batched, dense form fed to the model:
  * atom_feats_2d   [B, N, F_atom]
  * bond_feats_2d   [B, N, N, F_bond]      (zeros for non-bonded pairs)
  * adj_2d          [B, N, N]              (0/1)
  * atom_types      [B, N, F_atype]
  * bond_lengths    [B, N, N]              (0 for non-bonded)
  * bond_angles     [B, N, N, N]           (angle at j formed by i-j-k) — sparse, optional
  * angle_mask      [B, N, N, N]
  * partial_charges [B, N]
  * pharma_feats    [B, N, F_pharma]
  * dihedral_index  [B, T, 4]              (i,j,k,l atom indices per torsion)
  * dihedral_angles [B, T]                 (signed dihedral in radians)
  * dihedral_mask   [B, T]
  * atom_mask       [B, N]                 (True for real atoms)
  * num_atoms       [B]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


# ---- Feature dimensions (defaults; override in featurization) ----
ATOM_FEAT_DIM = 32      # one-hot atomic number (top elements) + degree + charge + hybrid + arom + nH + chir
BOND_FEAT_DIM = 8       # type one-hot + stereo + in_ring + conjugated
ATYPE_DIM = 16          # 3D atom-type embedding input (atomic_num + period/group + radius/EN)
PHARMA_FEAT_DIM = 8     # donor, acceptor, aromatic, hydrophobic, +charge, -charge, halogen, ring
MAX_DIHEDRALS = 64      # padded count of torsions per molecule


@dataclass
class MolSample:
    """A single molecule's multi-track raw inputs (variable atom count).

    SE(3)-invariant by construction: only scalar geometric features (bond
    lengths / angles / dihedrals) are stored — no raw coordinates and no
    orientation-dependent voxel grid.
    """
    atom_feats_2d: torch.Tensor       # [N, F_atom]
    bond_feats_2d: torch.Tensor       # [E, F_bond]
    edge_index_2d: torch.Tensor       # [2, E]
    atom_types: torch.Tensor          # [N, F_atype]
    bond_lengths: torch.Tensor        # [E]
    angle_index: torch.Tensor         # [3, A]
    bond_angles: torch.Tensor         # [A]
    partial_charges: torch.Tensor     # [N]
    pharma_feats: torch.Tensor        # [N, F_pharma]
    dihedral_index: torch.Tensor      # [4, T] (i,j,k,l rows)
    dihedral_angles: torch.Tensor     # [T]
    smiles: str = ""                  # canonical SMILES (optional; lets shard
                                       # users recover molecule identity without
                                       # re-featurizing). Default keeps existing
                                       # pre-tagged shards loadable.


@dataclass
class MolBatch:
    """Padded, dense batch tensors. `atom_mask` marks real atoms."""
    atom_feats_2d: torch.Tensor       # [B, N, F_atom]
    bond_feats_2d: torch.Tensor       # [B, N, N, F_bond]
    adj_2d: torch.Tensor              # [B, N, N]  (1 if bond exists)
    atom_types: torch.Tensor          # [B, N, F_atype]
    bond_lengths: torch.Tensor        # [B, N, N]
    bond_angles: torch.Tensor         # [B, N, N, N]  (angle at j for triple i-j-k)
    angle_mask: torch.Tensor          # [B, N, N, N]  bool
    partial_charges: torch.Tensor     # [B, N]
    pharma_feats: torch.Tensor        # [B, N, F_pharma]
    dihedral_index: torch.Tensor      # [B, T, 4]  long
    dihedral_angles: torch.Tensor     # [B, T]
    dihedral_mask: torch.Tensor       # [B, T]  bool
    atom_mask: torch.Tensor           # [B, N]  bool
    num_atoms: torch.Tensor           # [B]

    def to(self, device) -> "MolBatch":
        return MolBatch(
            **{k: (v.to(device) if torch.is_tensor(v) else v)
               for k, v in self.__dict__.items()}
        )


def collate(samples: List[MolSample], max_atoms: int = 64,
             max_dihedrals: int = MAX_DIHEDRALS) -> MolBatch:
    """Pad a list of `MolSample` into a dense `MolBatch`.

    Pads to `max_atoms` (raises if any molecule exceeds it).
    """
    B = len(samples)
    N = max_atoms
    F_atom = samples[0].atom_feats_2d.shape[-1]
    F_bond = samples[0].bond_feats_2d.shape[-1]
    F_atype = samples[0].atom_types.shape[-1]
    F_pharma = samples[0].pharma_feats.shape[-1]

    atom_feats_2d = torch.zeros(B, N, F_atom)
    bond_feats_2d = torch.zeros(B, N, N, F_bond)
    adj_2d = torch.zeros(B, N, N)
    atom_types = torch.zeros(B, N, F_atype)
    bond_lengths = torch.zeros(B, N, N)
    bond_angles = torch.zeros(B, N, N, N)
    angle_mask = torch.zeros(B, N, N, N, dtype=torch.bool)
    partial_charges = torch.zeros(B, N)
    pharma_feats = torch.zeros(B, N, F_pharma)
    T = max_dihedrals
    dihedral_index = torch.zeros(B, T, 4, dtype=torch.long)
    dihedral_angles = torch.zeros(B, T)
    dihedral_mask = torch.zeros(B, T, dtype=torch.bool)
    atom_mask = torch.zeros(B, N, dtype=torch.bool)
    num_atoms = torch.zeros(B, dtype=torch.long)

    for b, s in enumerate(samples):
        n = s.atom_feats_2d.shape[0]
        if n > N:
            raise ValueError(f"Molecule has {n} atoms > max_atoms={N}")
        atom_mask[b, :n] = True
        num_atoms[b] = n
        atom_feats_2d[b, :n] = s.atom_feats_2d
        atom_types[b, :n] = s.atom_types
        partial_charges[b, :n] = s.partial_charges
        pharma_feats[b, :n] = s.pharma_feats

        # Bonds (undirected sparse → dense symmetric)
        ei = s.edge_index_2d
        if ei.numel() > 0:
            src, dst = ei[0], ei[1]
            adj_2d[b, src, dst] = 1.0
            bond_feats_2d[b, src, dst] = s.bond_feats_2d
            bond_lengths[b, src, dst] = s.bond_lengths

        # Angles
        ai = s.angle_index
        if ai.numel() > 0:
            i, j, k = ai[0], ai[1], ai[2]
            bond_angles[b, i, j, k] = s.bond_angles
            angle_mask[b, i, j, k] = True

        # Dihedrals
        di = s.dihedral_index
        if di.numel() > 0:
            t = di.shape[1]
            if t > T:
                raise ValueError(f"Molecule has {t} dihedrals > max_dihedrals={T}")
            dihedral_index[b, :t] = di.t()        # [T, 4]
            dihedral_angles[b, :t] = s.dihedral_angles
            dihedral_mask[b, :t] = True

    return MolBatch(
        atom_feats_2d=atom_feats_2d,
        bond_feats_2d=bond_feats_2d,
        adj_2d=adj_2d,
        atom_types=atom_types,
        bond_lengths=bond_lengths,
        bond_angles=bond_angles,
        angle_mask=angle_mask,
        partial_charges=partial_charges,
        pharma_feats=pharma_feats,
        dihedral_index=dihedral_index,
        dihedral_angles=dihedral_angles,
        dihedral_mask=dihedral_mask,
        atom_mask=atom_mask,
        num_atoms=num_atoms,
    )
