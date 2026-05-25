"""
Losses for the multi-track autoencoder.

Per-track reconstruction:
  * 2D atom features: MSE on continuous, BCE on binary slices — we treat the
    whole vector as a regression target for simplicity.
  * 2D adjacency:    BCE
  * 2D bond features: MSE (masked by adjacency)
  * 3D atom types:   MSE
  * Bond lengths:    MSE (masked by adjacency)
  * Bond angles:     MSE (masked by angle_mask)
  * Partial charges: MSE (masked by atom_mask)
  * Charge grid:     MSE
  * Pharma:          BCE
  * Atom-existence:  BCE on slot mask
Auxiliary:
  * KL divergence on mu/logvar (β-VAE style, light weight)
  * Contrastive (NT-Xent) on `sim_embed` when pair labels are present
  * Cross-track consistency: each fused track token must linearly predict the
    others — encourages the cross-attention to actually mix information.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
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
# Contrastive (NT-Xent) on similarity-projected latents
# ---------------------------------------------------------------------------
def nt_xent_loss(embeds: torch.Tensor, pair_indices: torch.Tensor, pair_labels: torch.Tensor,
                  temperature: float = 0.1) -> torch.Tensor:
    """Compute NT-Xent over labeled pairs.

    `pair_indices`: [P, 2] indices into batch. `pair_labels`: [P] 1=positive (similar),
    0=negative. For each positive pair (a, b) we want sim(a,b) high while all other
    in-batch negatives are pushed down — standard supervised contrastive.
    """
    if pair_indices is None or pair_indices.numel() == 0:
        return embeds.sum() * 0.0
    z = F.normalize(embeds, dim=-1)
    sim = z @ z.t() / temperature                                              # [B, B]
    # Build positive mask from labeled pairs (symmetric)
    B = z.shape[0]
    pos_mask = torch.zeros(B, B, device=z.device, dtype=torch.bool)
    pos_idx = pair_indices[pair_labels.bool()]
    pos_mask[pos_idx[:, 0], pos_idx[:, 1]] = True
    pos_mask[pos_idx[:, 1], pos_idx[:, 0]] = True
    if not pos_mask.any():
        return embeds.sum() * 0.0
    # Mask out self-similarity (use large finite neg to avoid -inf * 0 = NaN
    # when later multiplying log_prob by the pos_mask).
    self_mask = torch.eye(B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(self_mask, -1e9)
    log_prob = sim - torch.logsumexp(sim, dim=-1, keepdim=True)
    # Mean of log-prob over positives per anchor
    pos_count = pos_mask.sum(dim=-1).clamp(min=1)
    loss = -(log_prob * pos_mask.float()).sum(dim=-1) / pos_count
    # Average over anchors that actually have positives
    anchors = pos_mask.any(dim=-1).float()
    return (loss * anchors).sum() / anchors.sum().clamp(min=1.0)


# ---------------------------------------------------------------------------
# Cross-track consistency: predict track i's fused token from track j's
# ---------------------------------------------------------------------------
class CrossTrackConsistency(nn.Module):
    """Learn linear maps token_i → token_j for every i,j pair; penalize their
    error. Encourages each track to be predictable from the others, which is the
    point of the cross-attention block."""

    def __init__(self, hidden_dim: int, num_tracks: int = 4):
        super().__init__()
        self.maps = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_tracks * (num_tracks - 1))
        ])
        self.num_tracks = num_tracks

    def forward(self, tokens) -> torch.Tensor:
        loss = 0.0
        idx = 0
        for i in range(self.num_tracks):
            for j in range(self.num_tracks):
                if i == j:
                    continue
                pred = self.maps[idx](tokens[i])
                loss = loss + F.mse_loss(pred, tokens[j].detach())
                idx += 1
        return loss / (self.num_tracks * (self.num_tracks - 1))


# ---------------------------------------------------------------------------
# Aggregate loss
# ---------------------------------------------------------------------------
@dataclass
class LossWeights:
    # Tier-1 (the "what is each atom + how is it bonded" signal)
    atom_mask: float = 1.0
    atom_feat: float = 1.0
    adj: float = 1.0

    # Geometry — bumped to tier-1 to make 3D structure carry more weight in z
    atype: float = 0.75       # 3D atom-type vector (input to Geometry3DEncoder)
    bond_len: float = 1.0     # bond lengths (SE(3)-invariant scalar)
    bond_ang: float = 1.0     # bond angles
    torsion: float = 1.0      # dihedrals (sin/cos) — was the stickiest term, deserves more push

    # Tier-2 — supporting signals
    bond_feat: float = 0.5
    partial_charges: float = 0.5   # dropped: Gasteiger charges are highly correlated with
                                   #   atom_feat and atype, so 0.5 is enough.
    pharma: float = 0.5

    # Regularizers
    kl: float = 1e-3
    contrastive: float = 1.0       # disabled at runtime (passed 0.0 from train script)
    cross_track: float = 0.1


def compute_total_loss(out: Dict[str, torch.Tensor], batch: MolBatch,
                       cross_consistency: Optional[CrossTrackConsistency] = None,
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

    # --- Torsion track --- regress sin/cos to respect angular topology.
    sincos_pred = out["torsion_sincos"]                                        # [B, T, 2]
    target_sincos = torch.stack([batch.dihedral_angles.sin(),
                                  batch.dihedral_angles.cos()], dim=-1)        # [B, T, 2]
    losses["torsion"] = masked_mse(sincos_pred, target_sincos, batch.dihedral_mask)

    # --- KL (β-VAE) ---
    mu, logvar = out["mu"], out["logvar"]
    losses["kl"] = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # --- Contrastive similarity ---
    if batch.pair_indices is not None and batch.pair_indices.numel() > 0:
        losses["contrastive"] = nt_xent_loss(out["sim_embed"], batch.pair_indices, batch.pair_labels)
    else:
        losses["contrastive"] = out["sim_embed"].sum() * 0.0

    # --- Cross-track consistency ---
    if cross_consistency is not None:
        losses["cross_track"] = cross_consistency(out["fused_tokens"])
    else:
        losses["cross_track"] = out["z"].sum() * 0.0

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
        + w.contrastive * losses["contrastive"]
        + w.cross_track * losses["cross_track"]
    )
    losses["total"] = total
    return losses
