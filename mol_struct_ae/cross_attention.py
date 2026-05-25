"""
Cross-track information passing.

Two stages:
  1. Per-atom cross-attention between the three structural tracks that share an
     atom-index axis (2D-graph, 3D-geometry, pharmacophore). Each atom's
     representation in one track attends to its counterparts in the others.
  2. Global track-token fusion: pooled tokens from all four tracks (the three
     above + the voxel-charge track) are stacked and run through a transformer
     so every track sees every other.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


class PerAtomCrossAttention(nn.Module):
    """Bidirectional cross-attention between K aligned per-atom track tensors.

    All inputs have shape [B, N, d] sharing the same atom indices. For each
    track we let it attend to the concatenation of the *other* tracks at the
    same atom positions, mixed by MultiheadAttention.
    """

    def __init__(self, hidden_dim: int, num_tracks: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_tracks = num_tracks
        self.attns = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
            for _ in range(num_tracks)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_tracks)])
        self.ffns = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
                          nn.Linear(hidden_dim * 2, hidden_dim))
            for _ in range(num_tracks)
        ])
        self.norms2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_tracks)])

    def forward(self, tracks: List[torch.Tensor], atom_mask: torch.Tensor) -> List[torch.Tensor]:
        assert len(tracks) == self.num_tracks
        key_padding = ~atom_mask  # [B, N]
        out = []
        for i, h in enumerate(tracks):
            others = [tracks[j] for j in range(self.num_tracks) if j != i]
            kv = torch.cat(others, dim=1)                                          # [B, (K-1)*N, d]
            # Build key-padding by tiling atom-mask over the concatenated tracks.
            kv_pad = key_padding.repeat(1, self.num_tracks - 1)
            attended, _ = self.attns[i](h, kv, kv, key_padding_mask=kv_pad)
            h2 = self.norms[i](h + attended)
            h2 = self.norms2[i](h2 + self.ffns[i](h2))
            h2 = h2 * atom_mask.unsqueeze(-1).float()
            out.append(h2)
        return out


class TrackTokenFusion(nn.Module):
    """Transformer over the (small) sequence of per-track global tokens.

    Input: list of K tensors each [B, d]. Stack to [B, K, d] and run a
    transformer encoder — every track ends up cross-conditioned on every other.
    Returns the fused tokens (same shape) and a single bottleneck vector.
    """

    def __init__(self, hidden_dim: int, num_tracks: int, heads: int = 4,
                 num_layers: int = 2, latent_dim: int = 256):
        super().__init__()
        self.num_tracks = num_tracks
        layer = nn.TransformerEncoderLayer(hidden_dim, heads, dim_feedforward=hidden_dim * 2,
                                            batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.track_pos = nn.Parameter(torch.randn(num_tracks, hidden_dim) * 0.02)
        # CLS-style aggregation → latent
        self.cls = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.latent_head = nn.Linear(hidden_dim, latent_dim)
        # VAE-style mu/logvar (light regularization; can be ignored at inference)
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, tokens: List[torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        B = tokens[0].shape[0]
        stack = torch.stack(tokens, dim=1) + self.track_pos                # [B, K, d]
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, stack], dim=1)                                  # [B, 1+K, d]
        x = self.transformer(x)
        cls_out = x[:, 0]                                                   # [B, d]
        fused_tokens = list(x[:, 1:].unbind(dim=1))                         # K × [B, d]
        mu = self.mu_head(cls_out)
        logvar = self.logvar_head(cls_out)
        z = mu  # deterministic by default; sampling done in losses if VAE on
        return fused_tokens, z, mu, logvar
