"""
Losses for the Pairformer autoencoder.

Per-track reconstruction (10 terms):
  * atom_mask:       BCE on slot-existence logits
  * atom_feat:       MSE on the 32-D 2D atom feature vector
  * adj:             BCE on the symmetric N×N adjacency
  * bond_feat:       MSE on per-bond features (masked by adjacency)
  * atype:           MSE on the 16-D continuous atom-type vector
  * bond_len:        MSE on bond lengths (masked by adjacency)
  * bond_ang:        MSE on bond angles (masked by angle_mask)
  * partial_charges: MSE on Gasteiger partial charges (masked by atom_mask)
  * pharma:          BCE on the 8-D pharmacophore label vector
  * torsion:         MSE on (sin φ, cos φ) per dihedral (masked by dihedral_mask)

Regularizer (1 term):
  * kl:              −½ Σ(1+logσ²−μ²−exp(logσ²))    (β-VAE style, light weight)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .data import MolBatch


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """mask broadcasts over the last dim of pred/target."""
    m = mask.float()
    if m.dim() < pred.dim():
        m = m.unsqueeze(-1)
    diff = (pred - target) ** 2 * m
    return diff.sum() / m.sum().clamp(min=1.0) / pred.shape[-1]


def masked_bce(pred_logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.float()
    if m.dim() < pred_logits.dim():
        m = m.unsqueeze(-1)
    loss = F.binary_cross_entropy_with_logits(pred_logits, target.float(), reduction="none")
    return (loss * m).sum() / m.sum().clamp(min=1.0) / pred_logits.shape[-1]


# ---------------------------------------------------------------------------
# Loss weights
# ---------------------------------------------------------------------------
@dataclass
class LossWeights:
    # Tier-1: per-atom identity + bond topology
    atom_mask: float = 1.0
    atom_feat: float = 1.0
    adj: float = 1.0

    # Geometry — tier-1 weight on every 3D scalar so the latent must preserve shape
    atype: float = 0.75
    bond_len: float = 1.0
    bond_ang: float = 1.0
    torsion: float = 1.0

    # Tier-2: supporting signals
    bond_feat: float = 0.5
    partial_charges: float = 0.5
    pharma: float = 0.5

    # Regularizer
    # Bumped 1e-3 → 2e-3 to combat latent re-collapse. β=5e-3 fixed the worst
    # collapse (aspirin/glucose 0.92→0.79) but over-spread small edits
    # (benzene/toluene 0.88→0.64); 2e-3 is the targeted middle ground.
    kl: float = 2e-3


def compute_total_loss(out: Dict[str, torch.Tensor], batch: MolBatch,
                       weights: Optional[LossWeights] = None) -> Dict[str, torch.Tensor]:
    w = weights or LossWeights()
    losses: Dict[str, torch.Tensor] = {}

    atom_mask = batch.atom_mask                                                # [B, N]
    pair_mask = atom_mask.unsqueeze(1) & atom_mask.unsqueeze(2)                # [B, N, N]
    adj_mask = pair_mask                                                       # for the symmetric adjacency target
    bond_mask = (batch.adj_2d > 0) & pair_mask
    triple_mask = batch.angle_mask                                             # already gated

    # --- Atom existence ---
    losses["atom_mask"] = F.binary_cross_entropy_with_logits(
        out["atom_mask_logits"], atom_mask.float()
    )

    # --- 2D track ---
    losses["atom_feat"] = masked_mse(out["atom_pred"], batch.atom_feats_2d, atom_mask)
    losses["adj"] = masked_bce(out["adj_logits"].unsqueeze(-1), batch.adj_2d.unsqueeze(-1), adj_mask)
    losses["bond_feat"] = masked_mse(out["bond_pred"], batch.bond_feats_2d, bond_mask)

    # --- 3D track ---
    losses["atype"] = masked_mse(out["atype"], batch.atom_types, atom_mask)
    losses["bond_len"] = masked_mse(out["bond_len"].unsqueeze(-1),
                                     batch.bond_lengths.unsqueeze(-1), bond_mask)
    losses["bond_ang"] = masked_mse(out["bond_ang"].unsqueeze(-1),
                                     batch.bond_angles.unsqueeze(-1), triple_mask)

    # --- Charge track ---
    losses["partial_charges"] = masked_mse(out["partial_charges"].unsqueeze(-1),
                                            batch.partial_charges.unsqueeze(-1), atom_mask)

    # --- Pharma track ---
    losses["pharma"] = masked_bce(out["pharma_pred"], batch.pharma_feats, atom_mask)

    # --- Torsion track --- 36-bin classification over [-π, π].
    # The TorsionDecoder outputs n_bins logits per dihedral. We bin the
    # target angles into 10°-wide buckets and use cross-entropy. This is
    # robust to ETKDG's ~10–20° conformer noise (small angle differences
    # land in the same bin) and naturally captures multimodal targets
    # (a rotatable bond can have gauche+, trans, gauche- minima).
    #
    # Loss scale: random chance ≈ log(36) ≈ 3.58; correct-bin most of the
    # time ≈ 0.5; near-perfect ≈ 0.1.
    torsion_logits = out["torsion_logits"]                                     # [B, T, n_bins]
    n_bins = torsion_logits.shape[-1]
    # Bin index = floor((φ + π) / (2π / n_bins))
    bin_idx = ((batch.dihedral_angles + math.pi) * n_bins
               / (2.0 * math.pi)).long().clamp(0, n_bins - 1)                  # [B, T]
    ce = F.cross_entropy(
        torsion_logits.reshape(-1, n_bins),
        bin_idx.reshape(-1),
        reduction="none",
    ).reshape(bin_idx.shape)                                                   # [B, T]
    mask_f = batch.dihedral_mask.float()
    losses["torsion"] = (ce * mask_f).sum() / mask_f.sum().clamp(min=1.0)

    # --- KL (β-VAE) ---
    mu, logvar = out["mu"], out["logvar"]
    losses["kl"] = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total = (
        w.atom_mask * losses["atom_mask"]
        + w.atom_feat * losses["atom_feat"]
        + w.adj * losses["adj"]
        + w.bond_feat * losses["bond_feat"]
        + w.atype * losses["atype"]
        + w.bond_len * losses["bond_len"]
        + w.bond_ang * losses["bond_ang"]
        + w.partial_charges * losses["partial_charges"]
        + w.pharma * losses["pharma"]
        + w.torsion * losses["torsion"]
        + w.kl * losses["kl"]
    )
    losses["total"] = total
    return losses
