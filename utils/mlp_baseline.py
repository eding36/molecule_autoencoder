"""Morgan-FP + 3-layer MLP from-scratch baseline for MoleculeNet.

The canonical MoleculeNet "Singletask Network" / ECFP-MLP floor — same
fine-tuning protocol as the pretrained AE / Distill backbones, but with no
pretraining. Provides an honest "is pretraining helping?" delta when compared
side-by-side under matching split/metric/seeds.

Backend protocol (matches `AEBackend` / `DistillBackend` in
`utils.benchmark_moleculenet`):
    prepare(smiles, device) -> (kept_indices, {"fps": tensor})
    build_finetune_model(n_tasks, dropout) -> nn.Module
        whose forward(prepared, indices) -> logits
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


class _MLPFinetuneWrapper(nn.Module):
    def __init__(self, fp_dim: int, hidden: int, n_tasks: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fp_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_tasks),
        )

    def forward(self, prepared, indices):
        idx = torch.as_tensor(indices, device=prepared["fps"].device, dtype=torch.long)
        return self.net(prepared["fps"][idx])


class MLPBackend:
    """Morgan-FP MLP baseline. No pretraining — fresh weights per seed."""

    def __init__(self, fp_radius: int = 2, fp_nbits: int = 2048,
                  hidden: int = 512):
        self.fp_radius = fp_radius
        self.fp_nbits = fp_nbits
        self.hidden = hidden
        print(f"[MLPBackend] Morgan FP r={fp_radius} bits={fp_nbits} "
               f"hidden={hidden}  (trained from scratch)")

    def prepare(self, smiles: List[str], device: torch.device
                ) -> Tuple[List[int], dict]:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        fps, keep = [], []
        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is None:
                continue
            bv = AllChem.GetMorganFingerprintAsBitVect(mol, self.fp_radius,
                                                       nBits=self.fp_nbits)
            arr = (np.frombuffer(bv.ToBitString().encode(), dtype="u1")
                   - ord("0")).astype(np.float32)
            fps.append(arr); keep.append(i)
        fps_tensor = torch.from_numpy(np.stack(fps)).to(device)
        return keep, {"fps": fps_tensor}

    def build_finetune_model(self, n_tasks: int, dropout: float = 0.5) -> nn.Module:
        return _MLPFinetuneWrapper(self.fp_nbits, self.hidden, n_tasks, dropout)
