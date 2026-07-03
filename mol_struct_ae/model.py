"""
Top-level multi-track structural autoencoder, pairformer-based.

Forward pass:
    1. InputEmbedder maps all multi-track inputs (atom features, atom types,
       partial charges, pharma flags, torsions, adjacency, bond features,
       bond lengths) into two unified representations:
           single_repr  [B, N, d_single]   per-atom
           pair_repr    [B, N, N, d_pair]  per-(atom, atom)
    2. PairformerStack runs N blocks; at every block, single and pair update
       jointly via OuterProductMean (single→pair), TriangleAttention (pair
       self-update), and AttentionWithPairBias (single←pair). Bond-angle /
       through-space geometric info propagates through the pair tensor at
       every layer instead of being collected in a separate cross-attention
       stage at the end.
    3. GlobalAggregator pools single_repr to produce a global latent z (with
       μ/log σ² for light VAE regularization) plus 5 context tokens for the
       decoders.
    4. Each track's decoder reconstructs the original inputs from z + the
       context tokens from GlobalAggregator

The model returns a dict of predictions; losses are computed in losses.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from .data import (ATOM_FEAT_DIM, ATYPE_DIM, BOND_FEAT_DIM, MolBatch,
                    PHARMA_FEAT_DIM)
from .pairformer import (InputEmbedder, PairformerStack, GlobalAggregator)
from .decoders import (AtomQueryDecoder, ChargeDecoder, Geometry3DDecoder,
                        Graph2DDecoder, PharmaDecoder, TorsionDecoder)


@dataclass
class MolAEConfig:
    max_atoms: int = 64
    # Pairformer (single + pair representation) sizing
    hidden_dim: int = 128          # d_single
    pair_dim: int = 32             # d_pair
    latent_dim: int = 256
    heads: int = 4
    num_pairformer_blocks: int = 4
    # Decoder cross-attention memory tokens produced by GlobalAggregator
    num_context_tokens: int = 5
    # Input feature dims (override if you change featurization)
    atom_feat_dim: int = ATOM_FEAT_DIM
    bond_feat_dim: int = BOND_FEAT_DIM
    atom_type_dim: int = ATYPE_DIM
    pharma_feat_dim: int = PHARMA_FEAT_DIM


class MolStructAutoencoder(nn.Module):
    def __init__(self, cfg: Optional[MolAEConfig] = None):
        super().__init__()
        cfg = cfg or MolAEConfig()
        self.cfg = cfg
        d_single = cfg.hidden_dim
        d_pair = cfg.pair_dim

        # ── Pairformer encoder stack ────────────────────────────────────────
        self.input_embedder = InputEmbedder(
            atom_feat_dim=cfg.atom_feat_dim,
            atom_type_dim=cfg.atom_type_dim,
            pharma_feat_dim=cfg.pharma_feat_dim,
            bond_feat_dim=cfg.bond_feat_dim,
            d_single=d_single, d_pair=d_pair,
        )
        self.pairformer = PairformerStack(
            d_single=d_single, d_pair=d_pair,
            num_blocks=cfg.num_pairformer_blocks,
            heads_single=cfg.heads, heads_pair=cfg.heads,
        )
        self.aggregator = GlobalAggregator(
            d_single=d_single, latent_dim=cfg.latent_dim,
            num_context_tokens=cfg.num_context_tokens, heads=cfg.heads,
        )

        # ── Decoders (unchanged API: consume z + list of context tokens) ────
        self.slot_decoder = AtomQueryDecoder(
            cfg.latent_dim, d_single, cfg.max_atoms,
            num_layers=2, heads=cfg.heads,
        )
        self.dec_2d = Graph2DDecoder(d_single, cfg.atom_feat_dim, cfg.bond_feat_dim)
        self.dec_3d = Geometry3DDecoder(d_single, cfg.atom_type_dim, cfg.max_atoms)
        self.dec_charge = ChargeDecoder(d_single)
        self.dec_pharma = PharmaDecoder(d_single, cfg.pharma_feat_dim)
        self.dec_torsion = TorsionDecoder(d_single, cfg.pair_dim)

        # NOTE: a `sim_proj` MLP used to wrap z into a separate similarity
        # embedding (SimCLR-style projection head). We removed it because
        # there's no contrastive loss to train it; it stayed at random init
        # and just degraded similarity by adding noise. `sim_embed` is now
        # `z` directly. The old `sim_proj.*` keys are kept in the state dict
        # only as a dummy module so legacy checkpoints continue to load
        # without strict=False; the module is never called.
        self.sim_proj = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim), nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
        )

    def encode(self, batch: MolBatch, sample: bool = False) -> Dict[str, torch.Tensor]:
        # 1) input embedding: all tracks → (single, pair)
        single, pair = self.input_embedder(batch)

        # 2) pairformer stack: N blocks of (outer-product, triangle-attn,
        #    pair-MLP, single-attn-with-pair-bias, single-MLP)
        single, pair = self.pairformer(single, pair, batch.atom_mask)

        # 3) global pool → latent + context tokens for decoders
        fused_tokens, mu, logvar = self.aggregator(single, batch.atom_mask)

        if sample and self.training:
            std = (0.5 * logvar).exp()
            z = mu + std * torch.randn_like(std)
        else:
            z = mu

        return {
            "z": z, "mu": mu, "logvar": logvar,
            "fused_tokens": fused_tokens,
            "single": single, "pair": pair,
        }

    def decode(self, z: torch.Tensor, fused_tokens, pair: torch.Tensor,
                dihedral_index: torch.Tensor, angle_mask: torch.Tensor,
                ) -> Dict[str, torch.Tensor]:
        ctx = torch.stack(fused_tokens, dim=1)                              # [B, 5, hidden]
        slot_h, atom_mask_logits = self.slot_decoder(z, ctx)

        atom_pred, adj_logits, bond_pred = self.dec_2d(slot_h)
        atype, bond_len, bond_ang = self.dec_3d(slot_h, angle_mask)
        partial = self.dec_charge(slot_h)
        pharma_pred = self.dec_pharma(slot_h)
        # TorsionDecoder now reads post-Pairformer pair_repr at the (i,l) and
        # (j,k) entries for each dihedral, and outputs 36-bin classification
        # logits instead of (sin, cos) regression.
        torsion_logits = self.dec_torsion(slot_h, pair, dihedral_index)

        return {
            "atom_mask_logits": atom_mask_logits,
            "atom_pred": atom_pred,
            "adj_logits": adj_logits,
            "bond_pred": bond_pred,
            "atype": atype,
            "bond_len": bond_len,
            "bond_ang": bond_ang,
            "partial_charges": partial,
            "pharma_pred": pharma_pred,
            "torsion_logits": torsion_logits,
        }

    def forward(self, batch: MolBatch, sample: bool = True) -> Dict[str, torch.Tensor]:
        enc = self.encode(batch, sample=sample)
        dec = self.decode(enc["z"], enc["fused_tokens"], enc["pair"],
                          batch.dihedral_index, batch.angle_mask)
        # sim_embed aliases z directly — see note in __init__ about sim_proj.
        return {**enc, **dec, "sim_embed": enc["z"]}
