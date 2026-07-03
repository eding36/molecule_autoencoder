"""
SMILES-only encoder (distillation model for distillation).

Inputs: token IDs + attention mask from `SmilesTokenizer`.
Output: an L2-normalized embedding of dimension `output_dim`, matching the
mol_struct_ae's `sim_embed`.

Architecture: standard transformer encoder over the SMILES sequence
with a learned `<cls>` token (id=1) at position 0. The CLS hidden state is
passed through a small MLP head → L2-normalized → output.

"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SmilesEncoderConfig:
    vocab_size: int
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.1
    max_len: int = 128
    output_dim: int = 256


class SmilesEncoder(nn.Module):
    def __init__(self, cfg: SmilesEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.hidden_dim, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.randn(cfg.max_len, cfg.hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * cfg.ffn_mult,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,                   # pre-norm — more stable for deeper stacks
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.output_dim),
        )

    def forward(self, ids: torch.Tensor, attn_mask: torch.Tensor,
                normalize: bool = True) -> torch.Tensor:
        """
        ids:       [B, L] long
        attn_mask: [B, L] bool — True for real tokens
        returns:   [B, output_dim]
        """
        L = ids.size(1)
        h = self.tok_embed(ids) + self.pos_embed[:L].unsqueeze(0)
        # nn.Transformer expects True = "ignore", i.e. inverse of our mask.
        key_padding = ~attn_mask
        h = self.transformer(h, src_key_padding_mask=key_padding)
        h = self.norm(h)
        cls = h[:, 0]                                # CLS hidden state
        out = self.head(cls)
        if normalize:
            out = F.normalize(out, dim=-1)
        return out
