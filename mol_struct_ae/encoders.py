"""
Per-track encoders.

All encoders work on dense, padded tensors with an `atom_mask` and produce:
  * per-atom features  [B, N, d]   (None for the voxel charge track)
  * a pooled "track token" [B, d]

Dense GATConv re-implementation: PyG's GATConv is for sparse PyG batches; we work
in dense [B,N,N] adjacency space so the decoders are simpler. The math is identical
(multi-head additive attention over neighbors, masked by adjacency).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x: [B, N, d], mask: [B, N] bool → [B, d]."""
    m = mask.unsqueeze(-1).float()
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


class GaussianRBF(nn.Module):
    """Smear scalar values (distances/angles) into a fixed RBF basis."""

    def __init__(self, num_centers: int = 32, low: float = 0.0, high: float = 5.0):
        super().__init__()
        centers = torch.linspace(low, high, num_centers)
        self.register_buffer("centers", centers)
        self.gamma = 1.0 / (centers[1] - centers[0]).item() ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [...], out: [..., K]
        diff = x.unsqueeze(-1) - self.centers
        return torch.exp(-self.gamma * diff.pow(2))


# ---------------------------------------------------------------------------
# 2D graph encoder: dense GAT
# ---------------------------------------------------------------------------
class DenseGATLayer(nn.Module):
    """One multi-head GAT layer over a dense adjacency, with edge features.

    Mirrors `GATConv` (Velickovic et al.) but on `[B,N,N]` adjacency tensors so
    the rest of the pipeline stays dense.
    """

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert out_dim % heads == 0
        self.heads = heads
        self.head_dim = out_dim // heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.W_edge = nn.Linear(edge_dim, out_dim, bias=False)
        # Attention vectors per Velickovic et al.
        self.a_src = nn.Parameter(torch.empty(heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(heads, self.head_dim))
        self.a_edge = nn.Parameter(torch.empty(heads, self.head_dim))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        nn.init.xavier_uniform_(self.a_edge)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,        # [B, N, in_dim]
        edge_attr: torch.Tensor,  # [B, N, N, edge_dim]
        adj: torch.Tensor,      # [B, N, N]
        atom_mask: torch.Tensor,  # [B, N]
    ) -> torch.Tensor:
        B, N, _ = h.shape
        Wh = self.W(h).view(B, N, self.heads, self.head_dim)            # [B,N,H,D]
        We = self.W_edge(edge_attr).view(B, N, N, self.heads, self.head_dim)  # [B,N,N,H,D]

        e_src = (Wh * self.a_src).sum(-1)                                # [B,N,H]
        e_dst = (Wh * self.a_dst).sum(-1)                                # [B,N,H]
        e_edge = (We * self.a_edge).sum(-1)                              # [B,N,N,H]

        # Attention logits at edge (i→j) = e_src[i] + e_dst[j] + e_edge[i,j]
        logits = e_src.unsqueeze(2) + e_dst.unsqueeze(1) + e_edge        # [B,N,N,H]
        logits = F.leaky_relu(logits, 0.2)

        # Mask: only attend across real bonds (adj=1) AND real atoms (self-loops added).
        # Use a large finite negative so padded rows (no valid keys) produce finite
        # softmax outputs instead of NaN; the output is re-masked by atom_mask below.
        eye = torch.eye(N, device=h.device).unsqueeze(0).expand(B, -1, -1)
        adj_self = (adj + eye).clamp(max=1.0)
        pair_mask = adj_self.bool() & atom_mask.unsqueeze(1) & atom_mask.unsqueeze(2)
        logits = logits.masked_fill(~pair_mask.unsqueeze(-1), -1e9)

        attn = F.softmax(logits, dim=2)                                  # softmax over j
        attn = self.dropout(attn)

        # Aggregate: [B,N,H,D] = sum_j attn[i,j,h] * Wh[j,h,d]   (+ edge contribution)
        h_msg = torch.einsum("bijh,bjhd->bihd", attn, Wh)                # neighbor messages
        e_msg = torch.einsum("bijh,bijhd->bihd", attn, We)               # edge-aware messages
        out = (h_msg + e_msg).reshape(B, N, self.heads * self.head_dim)
        out = out * atom_mask.unsqueeze(-1).float()
        return out


class Graph2DEncoder(nn.Module):
    """GATConv-style 2D-graph encoder over atoms + bond features."""

    def __init__(self, atom_dim: int, bond_dim: int, hidden_dim: int = 128, heads: int = 4, num_layers: int = 4):
        super().__init__()
        self.atom_embed = nn.Linear(atom_dim, hidden_dim)
        self.bond_embed = nn.Linear(bond_dim, hidden_dim)
        self.layers = nn.ModuleList([
            DenseGATLayer(hidden_dim, hidden_dim, hidden_dim, heads=heads)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

    def forward(self, atom_feats: torch.Tensor, bond_feats: torch.Tensor,
                adj: torch.Tensor, atom_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.atom_embed(atom_feats)
        e = self.bond_embed(bond_feats)
        for layer, norm in zip(self.layers, self.norms):
            h_new = layer(h, e, adj, atom_mask)
            h = norm(h + h_new)
        h = h * atom_mask.unsqueeze(-1).float()
        token = masked_mean(h, atom_mask)
        return h, token


# ---------------------------------------------------------------------------
# 3D geometry + bond-angle encoder
# ---------------------------------------------------------------------------
class Geometry3DEncoder(nn.Module):
    """SchNet-style 3D encoder that additionally consumes bond angles.

    Each message-passing layer mixes:
      * pair messages weighted by RBF(bond_length)
      * angle messages weighted by RBF(bond_angle) gathered over triples (i,j,k)
    """

    def __init__(self, atom_type_dim: int, hidden_dim: int = 128, num_layers: int = 4,
                 num_rbf: int = 32, dist_cutoff: float = 5.0):
        super().__init__()
        self.atom_embed = nn.Linear(atom_type_dim, hidden_dim)
        self.dist_rbf = GaussianRBF(num_rbf, 0.0, dist_cutoff)
        self.angle_rbf = GaussianRBF(num_rbf, 0.0, math.pi)
        self.dist_proj = nn.Linear(num_rbf, hidden_dim)
        self.angle_proj = nn.Linear(num_rbf, hidden_dim)

        self.pair_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
            for _ in range(num_layers)
        ])
        self.angle_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
            for _ in range(num_layers)
        ])
        self.update_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

    def forward(self, atom_types: torch.Tensor, adj: torch.Tensor,
                bond_lengths: torch.Tensor, bond_angles: torch.Tensor, angle_mask: torch.Tensor,
                atom_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = atom_types.shape
        h = self.atom_embed(atom_types)
        h = h * atom_mask.unsqueeze(-1).float()

        # Edge filter: bonded AND both atoms real
        bond_mask = (adj > 0) & atom_mask.unsqueeze(1) & atom_mask.unsqueeze(2)  # [B,N,N]

        # RBF expansion for pair distances (kept dense — only 4D, ~MBs)
        e_dist = self.dist_proj(self.dist_rbf(bond_lengths))                     # [B,N,N,d]
        e_dist = e_dist * bond_mask.unsqueeze(-1).float()

        # Sparse angle indices and angles, gathered once for the whole stack.
        # angle_mask: [B,N,N,N] bool — True at real angle triples (typically ~100/mol).
        angle_coords = angle_mask.nonzero(as_tuple=False)                        # [A, 4]
        has_angles = angle_coords.shape[0] > 0
        if has_angles:
            b_idx, i_idx, j_idx, k_idx = angle_coords.unbind(dim=-1)
            ang_vals = bond_angles[b_idx, i_idx, j_idx, k_idx]                   # [A]
            e_ang_sparse = self.angle_proj(self.angle_rbf(ang_vals))             # [A, d]

        for pair_mlp, angle_mlp, upd_mlp, norm in zip(self.pair_mlps, self.angle_mlps,
                                                       self.update_mlps, self.norms):
            # Pair messages — still dense, allocation is small (B*N*N*d ≈ few MB)
            hi = h.unsqueeze(2).expand(-1, -1, N, -1)
            hj = h.unsqueeze(1).expand(-1, N, -1, -1)
            pair_msg = pair_mlp(torch.cat([hi, hj], dim=-1)) * e_dist            # [B,N,N,d]
            pair_msg = pair_msg * bond_mask.unsqueeze(-1).float()
            m_pair = pair_msg.sum(dim=2)                                          # [B,N,d]

            # Angle messages — sparse over actual triples (i,j,k), scattered to center j.
            # Avoids the [B,N,N,N,d] tensors that would otherwise be O(GB).
            m_angle = torch.zeros_like(h)
            if has_angles:
                h_i_a = h[b_idx, i_idx]                                          # [A, d]
                h_k_a = h[b_idx, k_idx]                                          # [A, d]
                msg_a = angle_mlp(torch.cat([h_i_a, h_k_a], dim=-1)) * e_ang_sparse.to(h_i_a.dtype)
                # autocast can leave RBF/exp outputs as float32 while h is half;
                # match dtypes at the scatter boundary.
                m_angle.index_put_((b_idx, j_idx), msg_a.to(m_angle.dtype), accumulate=True)

            update = upd_mlp(m_pair + m_angle)
            h = norm(h + update)
            h = h * atom_mask.unsqueeze(-1).float()

        token = masked_mean(h, atom_mask)
        return h, token


# ---------------------------------------------------------------------------
# Charge distribution encoder: per-atom partial-charge MLP
# ---------------------------------------------------------------------------
class ChargeEncoder(nn.Module):
    """Per-atom partial-charge encoder.

    Embeds each atom's Gasteiger partial charge into a hidden vector via a small
    MLP.  The voxel-grid CNN path was removed: PCA-canonicalized grids are not
    consistently oriented across structurally similar molecules (one-atom
    differences shift the PCA frame), so the grid provided a noisy training
    signal.  Charge locality is already captured here; through-space charge
    proximity can be added later via non-bonded pairwise distances.
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.charge_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, partial_charges: torch.Tensor,
                atom_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_atom = self.charge_mlp(partial_charges.unsqueeze(-1))   # [B, N, hidden]
        h_atom = self.norm(h_atom) * atom_mask.unsqueeze(-1).float()
        token = masked_mean(h_atom, atom_mask)
        return h_atom, token


# ---------------------------------------------------------------------------
# Pharmacophore encoder (extra track)
# ---------------------------------------------------------------------------
class TorsionEncoder(nn.Module):
    """Encodes dihedral angles defined on 4-atom quadruples (i,j,k,l).

    Per-quadruple feature = MLP([h_i, h_j, h_k, h_l, RBF(cos φ), RBF(sin φ)]).
    Sin/cos used together so the angle's circular topology is respected.
    The per-quadruple features are scattered back to the involved atoms
    (mean-pooled), producing an atom-aligned representation that can join the
    per-atom cross-attention with the other structural tracks.
    """

    def __init__(self, atom_type_dim: int, hidden_dim: int = 128, num_rbf: int = 16):
        super().__init__()
        self.atom_embed = nn.Linear(atom_type_dim, hidden_dim)
        self.cos_rbf = GaussianRBF(num_rbf, -1.0, 1.0)
        self.sin_rbf = GaussianRBF(num_rbf, -1.0, 1.0)
        self.tor_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4 + num_rbf * 2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, atom_types: torch.Tensor, dihedral_index: torch.Tensor,
                dihedral_angles: torch.Tensor, dihedral_mask: torch.Tensor,
                atom_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = atom_types.shape
        T = dihedral_index.shape[1]
        h_atom = self.atom_embed(atom_types) * atom_mask.unsqueeze(-1).float()  # [B,N,d]

        # Gather atom features at each of the 4 positions per torsion
        idx = dihedral_index.clamp(min=0, max=N - 1)                            # [B,T,4]
        # h_atom[b, idx[b,t,p], :] for each (b,t,p)
        b_idx = torch.arange(B, device=h_atom.device).view(B, 1, 1).expand_as(idx)
        gathered = h_atom[b_idx, idx]                                           # [B,T,4,d]
        flat = gathered.view(B, T, -1)                                          # [B,T,4d]

        ang = dihedral_angles
        feat = torch.cat([flat, self.cos_rbf(ang.cos()), self.sin_rbf(ang.sin())], dim=-1)
        tor_feat = self.tor_mlp(feat) * dihedral_mask.unsqueeze(-1).float()     # [B,T,d]

        # Scatter back to atoms (mean over torsions each atom appears in)
        h_scatter = torch.zeros(B, N, tor_feat.shape[-1], device=h_atom.device, dtype=tor_feat.dtype)
        count = torch.zeros(B, N, 1, device=h_atom.device, dtype=tor_feat.dtype)
        for p in range(4):
            ip = idx[..., p].unsqueeze(-1).expand(-1, -1, tor_feat.shape[-1])   # [B,T,d]
            mask_t = dihedral_mask.unsqueeze(-1).float()
            h_scatter.scatter_add_(1, ip, tor_feat * mask_t)
            count.scatter_add_(1, idx[..., p].unsqueeze(-1), mask_t)
        h_per_atom = h_scatter / count.clamp(min=1.0)
        h_per_atom = h_per_atom * atom_mask.unsqueeze(-1).float()

        # Track token = mean over real torsions
        denom = dihedral_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        token = (tor_feat * dihedral_mask.unsqueeze(-1).float()).sum(dim=1) / denom
        return h_per_atom, token, tor_feat


class PharmaEncoder(nn.Module):
    """Per-atom pharmacophore features → small transformer over atoms.

    Captures donor/acceptor/aromatic/hydrophobic/charge labels, complementary to
    the explicit graph + geometry tracks.
    """

    def __init__(self, pharma_dim: int, hidden_dim: int = 128, num_layers: int = 2, heads: int = 4):
        super().__init__()
        self.embed = nn.Linear(pharma_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, nhead=heads, dim_feedforward=hidden_dim * 2,
                                            batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, pharma_feats: torch.Tensor, atom_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.embed(pharma_feats)
        key_padding = ~atom_mask
        h = self.transformer(h, src_key_padding_mask=key_padding)
        h = h * atom_mask.unsqueeze(-1).float()
        token = masked_mean(h, atom_mask)
        return h, token
