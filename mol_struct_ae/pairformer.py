"""
Pairformer-lite for molecular structure autoencoding.

Replaces the 5-separate-encoder + post-hoc cross-attention design with a
unified backbone modelled on AlphaFold-3's Pairformer (and RoseTTAFold-3's
analogous stack). Two representations:

    single_repr  [B, N, d_single]   - per-atom features
    pair_repr    [B, N, N, d_pair]  - per-(atom, atom) features

Every input track (2D atom features, atom types, partial charges, pharma flags,
torsion-derived per-atom features) gets *added into* `single_repr` at the
input embedder; every per-pair input (adjacency, bond features, bond lengths)
gets added into `pair_repr`. The 5 tracks are no longer separate computations
— they share a unified backbone that updates `single` and `pair` jointly at
every block.

A `PairformerBlock` does, at each layer:
    1. pair += OuterProductMean(single)              # single contributes to pair
    2. pair += TriangleAttention(pair)               # pair self-update (AF3)
    3. pair += PairTransition(pair)                  # pair MLP
    4. single += AttnWithPairBias(single, pair)      # pair biases single attention
    5. single += SingleTransition(single)            # single MLP

This is the "Pairformer-lite" — drops AF3's triangle multiplication and
ending-node triangle attention (we keep only starting-node triangle attention)
to fit in our compute budget at N=96. The remaining operations are sufficient
to propagate information across all 5 tracks at every layer.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# Small helpers — inlined here because they're the only shared bits the
# Pairformer needs.
# ──────────────────────────────────────────────────────────────────────────────
def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x: [B, N, d], mask: [B, N] bool → [B, d] (masked-mean over N)."""
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
        diff = x.unsqueeze(-1) - self.centers
        return torch.exp(-self.gamma * diff.pow(2))


# ──────────────────────────────────────────────────────────────────────────────
# Input embedding: MolBatch tensors → (single_repr, pair_repr)
# ──────────────────────────────────────────────────────────────────────────────
class InputEmbedder(nn.Module):
    """Maps the multi-track MolBatch inputs to (single, pair) representations.

    Per-atom inputs (atom_feats_2d, atom_types, partial_charges, pharma_feats)
    are summed into `single` after individual embeddings.

    `pair_repr` receives FOUR kinds of geometric/topological signal:
        1. 1-bond:   adj_2d, bond_feats_2d, bond_lengths (RBF)        → pair[i,j] for bonded pairs
        2. 2-bond:   bond_angles (RBF) scattered from triples (i,j,k) → pair[i,k] and pair[k,i]
        3. 4-bond:   dihedral angles (sin/cos RBF) from quads (i,j,k,l) → pair[i,l] and pair[l,i]

    This means triangle attention has real geometric information at every
    relevant hop distance, not just 1-hop bond lengths. AF3-style: pair_repr
    is the "geometry tensor" that the rest of the network reasons over.
    """

    def __init__(
        self,
        atom_feat_dim: int,
        atom_type_dim: int,
        pharma_feat_dim: int,
        bond_feat_dim: int,
        d_single: int = 128,
        d_pair: int = 32,
        num_rbf: int = 16,
        dist_cutoff: float = 5.0,
    ):
        super().__init__()
        # ── single (per-atom) embeddings ────────────────────────────────────
        self.atom_feat_emb = nn.Linear(atom_feat_dim, d_single)
        self.atom_type_emb = nn.Linear(atom_type_dim, d_single)
        self.charge_emb = nn.Linear(1, d_single)
        self.pharma_emb = nn.Linear(pharma_feat_dim, d_single)
        self.single_norm = nn.LayerNorm(d_single)

        # ── pair (per-(atom, atom)) embeddings ──────────────────────────────
        # 1-bond signal: adjacency + bond features + bond-length RBF
        self.adj_emb = nn.Embedding(2, d_pair)
        self.bond_feat_emb = nn.Linear(bond_feat_dim, d_pair)
        self.dist_rbf = GaussianRBF(num_rbf, 0.0, dist_cutoff)
        self.dist_emb = nn.Linear(num_rbf, d_pair)

        # 2-bond signal: bond angles (i-j-k → pair[i,k])
        # Angles live in [0, π]; cover the full range with RBFs.
        import math as _math
        self.angle_rbf = GaussianRBF(num_rbf, 0.0, _math.pi)
        self.angle_emb = nn.Linear(num_rbf, d_pair)

        # 4-bond signal: signed dihedrals via multi-frequency Fourier RBFs.
        # Three harmonics (1ω, 2ω, 3ω) natively encode the 3-fold symmetry
        # of σ-bond rotation landscapes (gauche+, trans, gauche- minima).
        self.dih_cos_rbf = GaussianRBF(num_rbf, -1.0, 1.0)
        self.dih_sin_rbf = GaussianRBF(num_rbf, -1.0, 1.0)
        self.dih_emb = nn.Linear(num_rbf * 6, d_pair)   # 3 freqs × (cos, sin)

        self.pair_norm = nn.LayerNorm(d_pair)

    def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        atom_mask_f = batch.atom_mask.unsqueeze(-1).float()                  # [B, N, 1]

        # ── single_repr from all per-atom tracks ─────────────────────────────
        # Dihedrals do NOT contribute here — they're 4-atom features that live
        # in pair_repr at pair[i, l] (the terminal atoms of the 4-atom chain).
        h = self.atom_feat_emb(batch.atom_feats_2d)
        h = h + self.atom_type_emb(batch.atom_types)
        h = h + self.charge_emb(batch.partial_charges.unsqueeze(-1))
        h = h + self.pharma_emb(batch.pharma_feats)
        single = self.single_norm(h) * atom_mask_f                            # [B, N, d_single]

        # ── pair_repr: start with 1-bond signal ──────────────────────────────
        adj_long = batch.adj_2d.long().clamp(0, 1)                           # [B, N, N]
        pair = self.adj_emb(adj_long)                                         # [B, N, N, d_pair]
        pair = pair + self.bond_feat_emb(batch.bond_feats_2d)
        # Bond lengths are 0 for non-bonded pairs; only weight RBF on bonded
        dist_feat = self.dist_emb(self.dist_rbf(batch.bond_lengths))         # [B, N, N, d_pair]
        pair = pair + dist_feat * batch.adj_2d.unsqueeze(-1)

        # ── pair_repr: 2-bond signal from bond angles (i-j-k → pair[i,k]) ────
        # Sparse scatter: only the actual angle triples contribute; we never
        # materialise the dense [B, N, N, N] tensor.
        if batch.angle_mask.any():
            ang_idx = batch.angle_mask.nonzero(as_tuple=False)               # [A, 4]: (b, i, j, k)
            b_a, i_a, j_a, k_a = ang_idx.unbind(dim=-1)
            ang_vals = batch.bond_angles[b_a, i_a, j_a, k_a]                 # [A]
            ang_feats = self.angle_emb(self.angle_rbf(ang_vals))             # [A, d_pair]
            # Symmetric scatter: pair[i,k] += feat AND pair[k,i] += feat.
            # accumulate=True so multiple angles through different j's sum.
            pair = pair.index_put((b_a, i_a, k_a), ang_feats, accumulate=True)
            pair = pair.index_put((b_a, k_a, i_a), ang_feats, accumulate=True)

        # ── pair_repr: dihedrals scattered into BOTH terminal and axis pairs ─
        # For each dihedral (i, j, k, l):
        #   pair[i, l] / pair[l, i]  — the 4-bond terminal pair
        #   pair[j, k] / pair[k, j]  — the rotation-axis pair
        # Giving the model two complementary entry points for the same signal.
        if batch.dihedral_mask.any():
            valid = batch.dihedral_mask.nonzero(as_tuple=False)              # [P, 2]: (b, t)
            b_d, t_d = valid.unbind(dim=-1)
            quads = batch.dihedral_index[b_d, t_d]                            # [P, 4]
            i_d, j_d, k_d, l_d = quads.unbind(dim=-1)
            phi = batch.dihedral_angles[b_d, t_d]                             # [P]
            # Multi-frequency Fourier features: 1ω, 2ω, 3ω each as (cos, sin)
            # → 6 RBF blocks. Captures 3-fold rotational symmetry natively.
            dih_in = torch.cat([
                self.dih_cos_rbf(phi.cos()),         self.dih_sin_rbf(phi.sin()),
                self.dih_cos_rbf((2 * phi).cos()),   self.dih_sin_rbf((2 * phi).sin()),
                self.dih_cos_rbf((3 * phi).cos()),   self.dih_sin_rbf((3 * phi).sin()),
            ], dim=-1)                                                        # [P, num_rbf*6]
            dih_feats = self.dih_emb(dih_in)                                  # [P, d_pair]
            # Terminal-pair scatter
            pair = pair.index_put((b_d, i_d, l_d), dih_feats, accumulate=True)
            pair = pair.index_put((b_d, l_d, i_d), dih_feats, accumulate=True)
            # Axis-pair scatter (the j-k bond around which the rotation happens)
            pair = pair.index_put((b_d, j_d, k_d), dih_feats, accumulate=True)
            pair = pair.index_put((b_d, k_d, j_d), dih_feats, accumulate=True)

        # ── normalise + mask padded atom rows/columns ────────────────────────
        pair_mask = (
            batch.atom_mask.unsqueeze(1) & batch.atom_mask.unsqueeze(2)
        ).unsqueeze(-1).float()                                               # [B, N, N, 1]
        pair = self.pair_norm(pair) * pair_mask

        return single, pair


# ──────────────────────────────────────────────────────────────────────────────
# single → pair update (OuterProductMean)
# ──────────────────────────────────────────────────────────────────────────────
class OuterProductMean(nn.Module):
    """Standard AF3-style single → pair update.

    pair_update[i, j] = Linear( flatten( a[i] outer b[j] ) )

    where a, b are low-dim projections of single. Cheap relative to pair
    self-attention because the outer product happens in low dim (default 8x8).
    """

    def __init__(self, d_single: int, d_pair: int, d_inner: int = 16):
        super().__init__()
        self.norm = nn.LayerNorm(d_single)
        self.proj_a = nn.Linear(d_single, d_inner)
        self.proj_b = nn.Linear(d_single, d_inner)
        self.out_proj = nn.Linear(d_inner * d_inner, d_pair)

    def forward(self, single: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
        x = self.norm(single)
        mask_f = atom_mask.unsqueeze(-1).float()
        a = self.proj_a(x) * mask_f                                          # [B, N, dh]
        b = self.proj_b(x) * mask_f                                          # [B, N, dh]
        # Outer product per (i, j) pair
        outer = torch.einsum("bid,bje->bijde", a, b)                         # [B, N, N, dh, dh]
        outer = outer.flatten(-2)                                            # [B, N, N, dh*dh]
        # Mean-normalize by valid count (so this stays scale-stable across batch sizes)
        n_valid = mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)             # [B, 1, 1]
        outer = outer / n_valid.unsqueeze(-1)
        return self.out_proj(outer)                                          # [B, N, N, d_pair]


# ──────────────────────────────────────────────────────────────────────────────
# pair self-update: triangle attention (starting node only — AF3-lite)
# ──────────────────────────────────────────────────────────────────────────────
class TriangleAttentionStartingNode(nn.Module):
    """For each (i, j), attend over k using Q from pair[i, j], K/V from
    pair[i, k], with a bias from pair[k, j]. This is AF3's triangle-attention
    around-starting-node operation.

    Cost is O(B * N^3 * heads * d_h) — the triangle term. At N=96 with default
    heads=4, d_h=8, this is ~1.4 GFLOP per batch element per layer; tractable.
    """

    def __init__(self, d_pair: int, heads: int = 4):
        super().__init__()
        assert d_pair % heads == 0, f"d_pair={d_pair} not divisible by heads={heads}"
        self.heads = heads
        self.d_h = d_pair // heads
        self.norm = nn.LayerNorm(d_pair)
        self.q_proj = nn.Linear(d_pair, d_pair, bias=False)
        self.k_proj = nn.Linear(d_pair, d_pair, bias=False)
        self.v_proj = nn.Linear(d_pair, d_pair, bias=False)
        self.bias_proj = nn.Linear(d_pair, heads, bias=False)
        self.gate_proj = nn.Linear(d_pair, d_pair)
        self.out_proj = nn.Linear(d_pair, d_pair)

    def forward(self, pair: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
        x = self.norm(pair)
        B, N, _, _ = x.shape
        H, D = self.heads, self.d_h

        # Q[i, j], K[i, k], V[i, k]
        Q = self.q_proj(x).view(B, N, N, H, D)                              # [B, i, j, h, d]
        K = self.k_proj(x).view(B, N, N, H, D)                              # [B, i, k, h, d]
        V = self.v_proj(x).view(B, N, N, H, D)                              # [B, i, k, h, d]
        # Pair bias[k, j] — feeds into score for triangle (i, j, k)
        bias = self.bias_proj(x)                                            # [B, k, j, h]

        # scores[b, i, j, k, h] = (Q[b, i, j, h, :] · K[b, i, k, h, :]) / sqrt(d)
        scores = torch.einsum("bijhd,bikhd->bijkh", Q, K) / math.sqrt(D)
        # Add the (k, j) bias, broadcast over i
        # bias is indexed (k, j); transpose to (j, k) and add as [B, 1, j, k, h]
        scores = scores + bias.permute(0, 2, 1, 3).unsqueeze(1)             # broadcast over i

        # Mask: k positions corresponding to padded atoms should not attend
        # atom_mask: [B, N]. We want to mask scores[b, i, j, k] where atom k is padded.
        # Use the dtype-aware "min" so this stays safe under AMP / fp16.
        key_pad = (~atom_mask).view(B, 1, 1, N, 1)                          # [B, 1, 1, k, 1]
        scores = scores.masked_fill(key_pad, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores, dim=3)                                     # softmax over k
        out = torch.einsum("bijkh,bikhd->bijhd", attn, V)                   # [B, N, N, H, D]
        out = out.reshape(B, N, N, H * D)                                   # [B, N, N, d_pair]

        # Sigmoid gate (AF3-style)
        gate = torch.sigmoid(self.gate_proj(x))
        out = out * gate

        return self.out_proj(out)


# ──────────────────────────────────────────────────────────────────────────────
# pair MLP
# ──────────────────────────────────────────────────────────────────────────────
class PairTransition(nn.Module):
    def __init__(self, d_pair: int, mult: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(d_pair)
        self.fc = nn.Sequential(
            nn.Linear(d_pair, d_pair * mult), nn.GELU(),
            nn.Linear(d_pair * mult, d_pair),
        )

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        return self.fc(self.norm(pair))


# ──────────────────────────────────────────────────────────────────────────────
# single self-attention with a pair-derived bias
# ──────────────────────────────────────────────────────────────────────────────
class SingleAttentionWithPairBias(nn.Module):
    """single self-attention where the (i, j) attention score gets an additive
    bias from pair_repr[i, j]. This is how pair information flows back into
    the per-atom representation at every block.
    """

    def __init__(self, d_single: int, d_pair: int, heads: int = 4):
        super().__init__()
        assert d_single % heads == 0
        self.heads = heads
        self.d_h = d_single // heads
        self.norm_single = nn.LayerNorm(d_single)
        self.norm_pair = nn.LayerNorm(d_pair)
        self.q_proj = nn.Linear(d_single, d_single, bias=False)
        self.k_proj = nn.Linear(d_single, d_single, bias=False)
        self.v_proj = nn.Linear(d_single, d_single, bias=False)
        self.bias_proj = nn.Linear(d_pair, heads, bias=False)
        self.gate_proj = nn.Linear(d_single, d_single)
        self.out_proj = nn.Linear(d_single, d_single)

    def forward(
        self, single: torch.Tensor, pair: torch.Tensor, atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.norm_single(single)
        B, N, _ = x.shape
        H, D = self.heads, self.d_h

        Q = self.q_proj(x).view(B, N, H, D)
        K = self.k_proj(x).view(B, N, H, D)
        V = self.v_proj(x).view(B, N, H, D)

        scores = torch.einsum("bihd,bjhd->bijh", Q, K) / math.sqrt(D)        # [B, N, N, H]
        # Add pair bias
        pair_bias = self.bias_proj(self.norm_pair(pair))                    # [B, N, N, H]
        scores = scores + pair_bias

        # Mask padded atoms at the key side (dtype-aware for AMP / fp16)
        key_pad = (~atom_mask).view(B, 1, N, 1)
        scores = scores.masked_fill(key_pad, torch.finfo(scores.dtype).min)

        attn = F.softmax(scores, dim=2)
        out = torch.einsum("bijh,bjhd->bihd", attn, V).reshape(B, N, H * D)

        # Sigmoid gating
        gate = torch.sigmoid(self.gate_proj(x))
        out = out * gate

        # Re-mask query side
        return self.out_proj(out) * atom_mask.unsqueeze(-1).float()


# ──────────────────────────────────────────────────────────────────────────────
# single MLP
# ──────────────────────────────────────────────────────────────────────────────
class SingleTransition(nn.Module):
    def __init__(self, d_single: int, mult: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(d_single)
        self.fc = nn.Sequential(
            nn.Linear(d_single, d_single * mult), nn.GELU(),
            nn.Linear(d_single * mult, d_single),
        )

    def forward(self, single: torch.Tensor) -> torch.Tensor:
        return self.fc(self.norm(single))


# ──────────────────────────────────────────────────────────────────────────────
# One pairformer block: 5 residual updates interleaving single & pair
# ──────────────────────────────────────────────────────────────────────────────
class PairformerBlock(nn.Module):
    def __init__(
        self,
        d_single: int,
        d_pair: int,
        heads_single: int = 4,
        heads_pair: int = 4,
        pair_inner: int = 16,
    ):
        super().__init__()
        # Pair updates
        self.outer = OuterProductMean(d_single, d_pair, d_inner=pair_inner)
        self.tri_attn = TriangleAttentionStartingNode(d_pair, heads=heads_pair)
        self.pair_trans = PairTransition(d_pair)
        # Single updates
        self.single_attn = SingleAttentionWithPairBias(d_single, d_pair, heads=heads_single)
        self.single_trans = SingleTransition(d_single)

    def forward(
        self, single: torch.Tensor, pair: torch.Tensor, atom_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ① pair += single → pair
        pair = pair + self.outer(single, atom_mask)
        # ② pair self-update
        pair = pair + self.tri_attn(pair, atom_mask)
        # ③ pair MLP
        pair = pair + self.pair_trans(pair)
        # ④ single ← pair-biased self-attention
        single = single + self.single_attn(single, pair, atom_mask)
        # ⑤ single MLP
        single = single + self.single_trans(single)
        return single, pair


# ──────────────────────────────────────────────────────────────────────────────
# Stack
# ──────────────────────────────────────────────────────────────────────────────
class PairformerStack(nn.Module):
    def __init__(
        self,
        d_single: int = 128,
        d_pair: int = 32,
        num_blocks: int = 4,
        heads_single: int = 4,
        heads_pair: int = 4,
        pair_inner: int = 16,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            PairformerBlock(d_single, d_pair, heads_single, heads_pair, pair_inner)
            for _ in range(num_blocks)
        ])

    def forward(
        self, single: torch.Tensor, pair: torch.Tensor, atom_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            single, pair = block(single, pair, atom_mask)
        return single, pair


# ──────────────────────────────────────────────────────────────────────────────
# Aggregator: single_repr → (z, fused_tokens for decoders)
# ──────────────────────────────────────────────────────────────────────────────
class GlobalAggregator(nn.Module):
    """Pool single_repr via learned-query attention to produce:
      • A CLS-like vector → μ, log σ²  → z (the latent)
      • K context tokens that the decoders consume as cross-attention memory

    Implementation note: K + 1 learned queries share one MultiheadAttention.
    Query index 0 is the CLS query (drives the latent); queries 1…K are the
    context tokens (decoder memory). This matches the 5-track architecture's
    pattern where the latent came from a CLS-token attending over the
    pooled track tokens — the discrimination came from learned attention
    weights, not arithmetic mean.

    Previous design used arithmetic masked-mean over per-atom features for
    the latent. That averaged out molecular distinctions: aspirin and
    glucose (similar atom-feature distributions) ended up with similar
    means → very high cosine similarity → poor downstream discrimination.
    """

    def __init__(
        self,
        d_single: int = 128,
        latent_dim: int = 256,
        num_context_tokens: int = 5,
        heads: int = 4,
    ):
        super().__init__()
        self.num_context = num_context_tokens
        self.norm = nn.LayerNorm(d_single)
        # K + 1 learned queries: index 0 = CLS (drives latent), 1..K = context
        self.queries = nn.Parameter(
            torch.randn(num_context_tokens + 1, d_single) * 0.02
        )
        self.attn = nn.MultiheadAttention(
            d_single, num_heads=heads, batch_first=True,
        )
        # Latent heads (consume the CLS-attention output)
        self.mu_head = nn.Linear(d_single, latent_dim)
        self.logvar_head = nn.Linear(d_single, latent_dim)

    def forward(
        self, single: torch.Tensor, atom_mask: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        x = self.norm(single)
        B = x.shape[0]
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)                  # [B, K+1, d]
        key_pad = ~atom_mask
        out, _ = self.attn(queries, x, x, key_padding_mask=key_pad)            # [B, K+1, d]

        # First query is CLS — drives the latent through learned attention
        cls_out = out[:, 0, :]                                                  # [B, d]
        mu = self.mu_head(cls_out)
        logvar = self.logvar_head(cls_out)

        # Remaining K queries are decoder context tokens
        ctx = out[:, 1:, :]                                                     # [B, K, d]
        fused_tokens = [ctx[:, i, :] for i in range(self.num_context)]

        return fused_tokens, mu, logvar
