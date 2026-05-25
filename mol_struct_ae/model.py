"""
Top-level multi-track structural autoencoder.

Forward pass:
    1. Each track's encoder produces per-atom features (where applicable) and a
       global track token.
    2. Per-atom cross-attention mixes the three atom-aligned tracks
       (2D, 3D, pharma).
    3. Track-token transformer fuses all four tracks into a latent z (mu/logvar
       for light VAE regularization).
    4. Each track's decoder reconstructs the original inputs from z + the fused
       track tokens.

The model returns a dict of predictions; losses are computed in losses.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from .data import (ATOM_FEAT_DIM, ATYPE_DIM, BOND_FEAT_DIM, MolBatch,
                    PHARMA_FEAT_DIM)
from .encoders import (ChargeEncoder, Geometry3DEncoder, Graph2DEncoder,
                        PharmaEncoder, TorsionEncoder)
from .cross_attention import PerAtomCrossAttention, TrackTokenFusion
from .decoders import (AtomQueryDecoder, ChargeDecoder, Geometry3DDecoder,
                        Graph2DDecoder, PharmaDecoder, TorsionDecoder)


@dataclass
class MolAEConfig:
    max_atoms: int = 64
    hidden_dim: int = 128
    latent_dim: int = 256
    heads: int = 4
    gat_layers: int = 4
    geom_layers: int = 4
    pharma_layers: int = 2
    cross_layers: int = 2
    fusion_layers: int = 2
    atom_feat_dim: int = ATOM_FEAT_DIM
    bond_feat_dim: int = BOND_FEAT_DIM
    atom_type_dim: int = ATYPE_DIM
    pharma_feat_dim: int = PHARMA_FEAT_DIM


class MolStructAutoencoder(nn.Module):
    def __init__(self, cfg: Optional[MolAEConfig] = None):
        super().__init__()
        cfg = cfg or MolAEConfig()
        self.cfg = cfg
        d = cfg.hidden_dim

        # Encoders
        self.enc_2d = Graph2DEncoder(cfg.atom_feat_dim, cfg.bond_feat_dim, d, cfg.heads, cfg.gat_layers)
        self.enc_3d = Geometry3DEncoder(cfg.atom_type_dim, d, cfg.geom_layers)
        self.enc_charge = ChargeEncoder(d)
        self.enc_pharma = PharmaEncoder(cfg.pharma_feat_dim, d, cfg.pharma_layers, cfg.heads)
        self.enc_torsion = TorsionEncoder(cfg.atom_type_dim, d)

        # Cross-attention (per-atom, across four atom-aligned tracks: 2D, 3D, pharma, torsion).
        self.cross_atom = nn.ModuleList([
            PerAtomCrossAttention(d, num_tracks=4, heads=cfg.heads)
            for _ in range(cfg.cross_layers)
        ])

        # Track-token fusion → latent. 5 tracks: 2D, 3D, charge, pharma, torsion.
        self.fusion = TrackTokenFusion(d, num_tracks=5, heads=cfg.heads,
                                        num_layers=cfg.fusion_layers, latent_dim=cfg.latent_dim)

        # Decoders share a single atom-query decoder that maps z + fused tokens → atom slots.
        self.slot_decoder = AtomQueryDecoder(cfg.latent_dim, d, cfg.max_atoms, num_layers=2, heads=cfg.heads)
        self.dec_2d = Graph2DDecoder(d, cfg.atom_feat_dim, cfg.bond_feat_dim)
        self.dec_3d = Geometry3DDecoder(d, cfg.atom_type_dim, cfg.max_atoms)
        self.dec_charge = ChargeDecoder(d)
        self.dec_pharma = PharmaDecoder(d, cfg.pharma_feat_dim)
        self.dec_torsion = TorsionDecoder(d)

        # Similarity head (Tanimoto-like projection over the latent for contrastive loss)
        self.sim_proj = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim), nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
        )

    def encode(self, batch: MolBatch, sample: bool = False) -> Dict[str, torch.Tensor]:
        # Per-track encoders
        h2d, t2d = self.enc_2d(batch.atom_feats_2d, batch.bond_feats_2d,
                                batch.adj_2d, batch.atom_mask)
        h3d, t3d = self.enc_3d(batch.atom_types, batch.adj_2d,
                                batch.bond_lengths, batch.bond_angles, batch.angle_mask,
                                batch.atom_mask)
        hch, tch = self.enc_charge(batch.partial_charges, batch.atom_mask)
        hph, tph = self.enc_pharma(batch.pharma_feats, batch.atom_mask)
        htor, ttor, _ = self.enc_torsion(batch.atom_types, batch.dihedral_index,
                                          batch.dihedral_angles, batch.dihedral_mask,
                                          batch.atom_mask)

        # Per-atom cross-attention across (2D, 3D, Pharma, Torsion).
        # Charge is voxel-shaped and joins only at the global-token stage.
        atom_tracks = [h2d, h3d, hph, htor]
        for layer in self.cross_atom:
            atom_tracks = layer(atom_tracks, batch.atom_mask)
        h2d, h3d, hph, htor = atom_tracks

        # Refresh pooled tokens after per-atom mixing
        mask_f = batch.atom_mask.unsqueeze(-1).float()
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        t2d = (h2d * mask_f).sum(dim=1) / denom
        t3d = (h3d * mask_f).sum(dim=1) / denom
        tph = (hph * mask_f).sum(dim=1) / denom
        ttor = (htor * mask_f).sum(dim=1) / denom

        # Track-token transformer + latent head (5 tracks: 2D, 3D, charge, pharma, torsion)
        fused_tokens, z_det, mu, logvar = self.fusion([t2d, t3d, tch, tph, ttor])

        if sample and self.training:
            std = (0.5 * logvar).exp()
            z = mu + std * torch.randn_like(std)
        else:
            z = z_det

        return {
            "z": z, "mu": mu, "logvar": logvar,
            "fused_tokens": fused_tokens,
            "h2d": h2d, "h3d": h3d, "hch": hch, "hph": hph, "htor": htor,
        }

    def decode(self, z: torch.Tensor, fused_tokens, dihedral_index: torch.Tensor,
                angle_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        ctx = torch.stack(fused_tokens, dim=1)                              # [B, 5, hidden]
        slot_h, atom_mask_logits = self.slot_decoder(z, ctx)

        atom_pred, adj_logits, bond_pred = self.dec_2d(slot_h)
        atype, bond_len, bond_ang = self.dec_3d(slot_h, angle_mask)
        partial = self.dec_charge(slot_h)
        pharma_pred = self.dec_pharma(slot_h)
        torsion_sincos = self.dec_torsion(slot_h, dihedral_index)

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
            "torsion_sincos": torsion_sincos,
        }

    def forward(self, batch: MolBatch, sample: bool = True) -> Dict[str, torch.Tensor]:
        enc = self.encode(batch, sample=sample)
        dec = self.decode(enc["z"], enc["fused_tokens"], batch.dihedral_index, batch.angle_mask)
        return {**enc, **dec, "sim_embed": self.sim_proj(enc["z"])}
