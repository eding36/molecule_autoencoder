"""
Per-track decoders.

All decoders take the latent z [B, d_latent] plus the fused track token, and
produce dense reconstructions of that track up to `max_atoms`. The atom mask is
also reconstructed (predicted from atom count) so loss can ignore padded slots.

Decoders are intentionally simple — the goal of the autoencoder is to learn a
useful latent for similarity, not to ace de-novo molecular generation.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class AtomQueryDecoder(nn.Module):
    """Generates N atom slots from a global latent via learned queries.

    Each of the `max_atoms` slot positions has a learned query that cross-attends
    to the latent, producing a per-slot hidden vector. A separate MLP predicts
    whether each slot is "active" (occupied by a real atom).
    """

    def __init__(self, latent_dim: int, hidden_dim: int, max_atoms: int, num_layers: int = 2, heads: int = 4):
        super().__init__()
        self.max_atoms = max_atoms
        self.queries = nn.Parameter(torch.randn(max_atoms, hidden_dim) * 0.02)
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        layer = nn.TransformerDecoderLayer(hidden_dim, heads, dim_feedforward=hidden_dim * 2,
                                            batch_first=True, activation="gelu")
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.mask_head = nn.Linear(hidden_dim, 1)

    def forward(self, z: torch.Tensor, ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # z: [B, d_latent], ctx: [B, K, hidden] (e.g. fused track tokens)
        B = z.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)                    # [B, N, hidden], reshape query to batch dimension
        memory = torch.cat([self.latent_proj(z).unsqueeze(1), ctx], dim=1)  # [B, 1+K, hidden]
        h = self.decoder(q, memory)                                         # [B, N, hidden]
        atom_logits = self.mask_head(h).squeeze(-1)                         # [B, N]
        return h, atom_logits


class Graph2DDecoder(nn.Module):
    """Reconstructs atom features, bond features, and adjacency."""

    def __init__(self, hidden_dim: int, atom_feat_dim: int, bond_feat_dim: int):
        super().__init__()
        self.atom_head = nn.Linear(hidden_dim, atom_feat_dim)
        # Bond features and adjacency from pairwise hidden states
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.adj_head = nn.Linear(hidden_dim, 1)
        self.bond_head = nn.Linear(hidden_dim, bond_feat_dim)

    def forward(self, slot_h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # slot_h: [B, N, hidden]
        B, N, _ = slot_h.shape
        atom_pred = self.atom_head(slot_h)                                  # [B, N, F_atom]
        hi = slot_h.unsqueeze(2).expand(-1, -1, N, -1)
        hj = slot_h.unsqueeze(1).expand(-1, N, -1, -1)
        pair = self.pair_mlp(torch.cat([hi, hj], dim=-1))
        pair = 0.5 * (pair + pair.transpose(1, 2))                          # symmetric
        adj_logits = self.adj_head(pair).squeeze(-1)                        # [B, N, N]
        bond_pred = self.bond_head(pair)                                    # [B, N, N, F_bond]
        return atom_pred, adj_logits, bond_pred


class Geometry3DDecoder(nn.Module):
    """Reconstructs atom-type vectors, bond lengths and bond angles.

    Raw xyz coords are not reconstructed — the encoder is SE(3)-invariant
    (uses only distances and angles), so there is no orientation signal in the
    latent to decode back to absolute coordinates.
    """

    def __init__(self, hidden_dim: int, atom_type_dim: int, max_atoms: int):
        super().__init__()
        self.atype_head = nn.Linear(hidden_dim, atom_type_dim)
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.length_head = nn.Linear(hidden_dim, 1)
        # Angles predicted from triple slot states
        self.triple_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, slot_h: torch.Tensor, angle_mask: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reconstruct atom types, bond lengths and bond angles.

        Bond angles are predicted only at the masked triple positions; the dense
        [B,N,N,N,3D] cat that the previous version allocated was multi-GB.
        Output `bond_ang` is still dense [B,N,N,N] for API stability — non-mask
        entries stay zero and are ignored by the loss (which already masks).
        """
        B, N, D = slot_h.shape
        atype = self.atype_head(slot_h)                                     # [B, N, atype_dim]
        hi = slot_h.unsqueeze(2).expand(-1, -1, N, -1)
        hj = slot_h.unsqueeze(1).expand(-1, N, -1, -1)
        pair = self.pair_mlp(torch.cat([hi, hj], dim=-1))
        pair = 0.5 * (pair + pair.transpose(1, 2))
        bond_len = self.length_head(pair).squeeze(-1)                       # [B, N, N]

        bond_ang = torch.zeros(B, N, N, N, device=slot_h.device, dtype=slot_h.dtype)
        angle_coords = angle_mask.nonzero(as_tuple=False)                    # [A, 4]
        if angle_coords.shape[0] > 0:
            b_idx, i_idx, j_idx, k_idx = angle_coords.unbind(dim=-1)
            h_i = slot_h[b_idx, i_idx]                                       # [A, D]
            h_j = slot_h[b_idx, j_idx]
            h_k = slot_h[b_idx, k_idx]
            pred = self.triple_mlp(torch.cat([h_i, h_j, h_k], dim=-1)).squeeze(-1)
            bond_ang[b_idx, i_idx, j_idx, k_idx] = pred.to(bond_ang.dtype)
        return atype, bond_len, bond_ang


class ChargeDecoder(nn.Module):
    """Reconstructs per-atom partial charges from atom slot features."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.charge_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                          nn.Linear(hidden_dim, 1))

    def forward(self, slot_h: torch.Tensor) -> torch.Tensor:
        return self.charge_head(slot_h).squeeze(-1)                          # [B, N]


class TorsionDecoder(nn.Module):
    """Predicts each dihedral as a 36-class distribution over 10° bins.

    Two important design changes vs the original:

    1. **Direct pair_repr read (Fix A):** in addition to the 4 atom slot
       features, we gather `pair[i, l]` and `pair[j, k]` from post-Pairformer
       pair_repr. The dihedral was scattered into those pair entries at the
       input embedder, so the relevant signal is sitting right there — no
       need to make it round-trip through z and back through slot_decoder.

    2. **36-bin classification (Fix D):** instead of regressing (sin φ, cos φ),
       we output 36 logits per dihedral (10° bins covering [-π, π]). This is
       robust to ETKDG's ~10–20° conformer noise (small angle differences
       land in the same bin) and naturally captures multimodal targets
       (a rotatable bond can have gauche+, trans, gauche- minima).

    Output shape: [B, T, n_bins].
    """

    def __init__(self, d_single: int, d_pair: int, n_bins: int = 36):
        super().__init__()
        self.n_bins = n_bins
        in_dim = 4 * d_single + 2 * d_pair
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_single * 2), nn.GELU(),
            nn.Linear(d_single * 2, d_single), nn.GELU(),
            nn.Linear(d_single, n_bins),
        )

    def forward(
        self,
        slot_h: torch.Tensor,                  # [B, N, d_single]
        pair: torch.Tensor,                    # [B, N, N, d_pair]
        dihedral_index: torch.Tensor,          # [B, T, 4]
    ) -> torch.Tensor:
        B, N, D = slot_h.shape
        T = dihedral_index.shape[1]
        idx = dihedral_index.clamp(min=0, max=N - 1)                            # [B,T,4]
        # Gather slot features at i, j, k, l
        b_idx = torch.arange(B, device=slot_h.device).view(B, 1, 1).expand_as(idx)
        gathered = slot_h[b_idx, idx]                                           # [B,T,4,D]
        flat = gathered.view(B, T, -1)                                          # [B,T,4*D]
        # Gather pair entries at (i, l) and (j, k) where the dihedral was scattered
        b1 = torch.arange(B, device=slot_h.device).view(B, 1).expand(B, T)
        i_idx = idx[..., 0]
        j_idx = idx[..., 1]
        k_idx = idx[..., 2]
        l_idx = idx[..., 3]
        p_il = pair[b1, i_idx, l_idx]                                           # [B,T,d_pair]
        p_jk = pair[b1, j_idx, k_idx]                                           # [B,T,d_pair]
        x = torch.cat([flat, p_il, p_jk], dim=-1)                                # [B,T,4*D+2*d_pair]
        return self.mlp(x)                                                       # [B,T,n_bins]


class PharmaDecoder(nn.Module):
    """Reconstructs per-atom pharmacophore labels (binary)."""

    def __init__(self, hidden_dim: int, pharma_dim: int):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                   nn.Linear(hidden_dim, pharma_dim))

    def forward(self, slot_h: torch.Tensor) -> torch.Tensor:
        return self.head(slot_h)                                             # [B, N, pharma_dim]
