# mol_struct_ae

A two-stage pipeline for **structural similarity search** over small molecules.

**Stage 1 — `mol_struct_ae`**: a multi-track autoencoder that learns a single
latent from five complementary views of each molecule:

   1. **2D graph** — atom and bond features, encoded with GAT
   2. **3D geometry** — bond lengths, bond angles, encoded with an SE(3)-invariant MPNN (RBF on distances/angles only — no raw xyz)
   3. **Charge** — per-atom Gasteiger partial charges, encoded with an MLP
   4. **Pharmacophore** — per-atom donor/acceptor/aromatic/etc. labels, encoded with MLP + transformer
   5. **Torsion / dihedral angles** — signed dihedrals over 4-atom quadruples, encoded with an MLP that scatters back to atoms

The tracks talk to each other through per-atom cross-attention + a global
track-token transformer. The result is one 256-D latent `z` (and its
L2-normalized projection `sim_embed`) per molecule. Reconstruction losses on
every track + cross-track-consistency keep the latent informative.

**Stage 2 — `distillation`**: a small SMILES-only transformer trained to
reproduce `mol_struct_ae`'s `sim_embed` directly from the SMILES string. Once
trained, inference is just `tokenize(smiles) → distillation → embedding` — no
RDKit, no featurization, ~1 ms per molecule.

```
                  ┌──── train once ────┐    ┌──── train once ────┐
SMILES → RDKit → MolSample → mol_struct_ae → sim_embed
                                                  │
                                                  ▼ (target)
SMILES ───────────────────────────→ distillation → embedding (≈ sim_embed)
                                          ▲
                                          └─── trained against the target via
                                               cosine-distillation loss
```

**Inference uses only the distillation model.** No conformer generation, no
charge calc — just a SMILES tokenizer + a small transformer.

This document covers:

1. [Architecture](#architecture) — the five `mol_struct_ae` tracks, the cross-attention fusion, the latent, and the per-track decoders. Plus the `distillation` model.
2. [Data format](#data-format) — what a `MolSample` and a `MolBatch` contain.
3. [Featurization](#featurization) — how raw molecules are turned into `MolSample`s and sharded to disk.
4. [Training](#training) — stage 1 (mol_struct_ae) and stage 2 (distillation).
5. [Inference and similarity](#inference) — embedding any SMILES with the distillation model.
6. [Repo layout and running](#repo-layout) — also covers the four bundled
   benchmark suites (MoleculeNet, MoleculeACE, HTS virtual-screening,
   perturbation smoke test), all leakage-filtered against the training set.

---

## Architecture

```
  ┌──────────────────────────── INPUT ────────────────────────────────┐
  │  2D graph  |  bond lengths+angles  |  partial charges  |  pharma  |  torsion  │
  └───────────────────────────────────────────────────────────────────┘
       │                   │                   │              │            │
       ▼                   ▼                   ▼              ▼            ▼
      GAT           SE(3)-invariant          MLP            MLP +        MLP
                       MPNN                               Transformer-
                  (RBF distances                           encoder
                   + angles)
       │                   │                   │              │            │
       └───────────────────┴──────┬────────────┴──────────────┴────────────┘
                                   ▼
               Per-atom cross-attention  (×L)          ← 4 atom-aligned tracks
               (2D, 3D, pharma, torsion attend to each other per atom)
                                   │
        ┌──────────────────────────┴────────────────┐
        ▼                                           ▼
    Pool per track → 5 track tokens        (charge token joins here)
                                   │
                                   ▼
               Track-token transformer (CLS+5)          ← global fusion
                                   │
                                   ▼
                         μ, log σ²  → reparam → z (latent)
                                   │
        ┌──────────────┬───────────┴──────┬──────────────┬─────────────┐
        ▼              ▼                  ▼              ▼             ▼
    2D recon      3D recon           charge recon    pharma recon   torsion recon
    (atom feats,  (atype, bond len,  (per-atom q)    (per-atom)     (sin φ, cos φ)
     adj, bond)    bond angle)
```

### The five tracks

Each track is an encoder–decoder pair that consumes and reconstructs one view
of the molecule. They are run **in parallel** and stitched together through
cross-attention.

| Track | Inputs | Encoder | Per-atom features? | Decoder output |
|---|---|---|---|---|
| **2D graph** | atom features (atomic num, charge, hybridization, …), bond features (type, stereo, in-ring), edge index | Dense **GATConv** with edge features (`DenseGATLayer`), multi-head additive attention with atoms only attending to connected atoms | yes | atom features, adjacency (BCE), bond features |
| **3D geometry** | atom types, bond lengths, bond angles, angle index | SE(3)-invariant MPNN: pair messages weighted by `RBF(bond_length)`, angle messages weighted by `RBF(bond_angle)` aggregated over atom triplets (i,j,k). Raw xyz coords are **not** used in the encoder — they would break rotation invariance. | yes | atom types, bond lengths, bond angles |
| **Charge** | per-atom Gasteiger partial charges `[N]` | Small MLP on the scalar charge per atom. A voxel-grid CNN path was considered but removed: PCA-canonicalized grids are not consistently oriented across structurally similar molecules (adding one atom shifts the PCA frame arbitrarily), so the grid introduced noisy training signal. Per-atom charges are fully SE(3)-invariant by construction. | yes | per-atom charges (MSE) |
| **Pharmacophore** | per-atom binary labels: donor, acceptor, aromatic, hydrophobic, ±charge, halogen, ring | Small transformer over atoms | yes | per-atom binary labels (BCE) |
| **Torsion / dihedral** | dihedral index `[T,4]`, signed dihedral angles `[T]` | MLP that takes inputs `(h_i, h_j, h_k, h_l, RBF(cos φ), RBF(sin φ))`, then scatters output back to atoms (mean over torsions each atom is in) | yes (after scatter) | (sin φ, cos φ) per torsion |

**Why these five?** They cover the natural hierarchy of molecular descriptors:
2-atom (bonds + lengths), 3-atom (bond angles), 4-atom (dihedrals), per-atom
electronic (partial charges), and binary chemical labels (pharmacophore). The
2D graph is the topological backbone; the rest are geometric/electronic/chemical
layers on top of it. Together they encode far more than Tanimoto fingerprints.

**SE(3) invariance.** All encoder inputs are either topology-based (2D graph,
pharmacophore labels) or SE(3)-invariant scalars (bond lengths, bond angles,
dihedral angles, partial charges). No raw xyz coordinates enter any encoder.
This means two conformers of the same molecule that differ only by a global
rotation or translation produce identical latent vectors.

### Cross-attention and fusion

Two stages, both in [cross_attention.py](mol_struct_ae/cross_attention.py).

1. **Per-atom cross-attention** (`PerAtomCrossAttention`, ×L layers). The four
   tracks that are atom-aligned — 2D, 3D, pharma, torsion — all have shape
   `[B, N, d]` with the *same atom indexing*. For each track in turn, every
   atom queries the same atom's representations in the other three tracks
   (concatenated as keys/values). This is followed by a feed-forward + norm.
   The result: each track's per-atom hidden states are aware of what the other
   tracks "saw" at that atom.

   The charge track is also atom-aligned (per-atom MLP output) and participates
   in this stage.

2. **Track-token fusion** (`TrackTokenFusion`). Each of the five tracks is
   pooled to a `[B, d]` token. A CLS token is prepended and a small transformer
   encoder mixes everything. The CLS output goes through a μ / log σ² head →
   reparameterized → latent `z`. The post-fusion track tokens are kept around
   and fed back into the decoders as cross-attention memory.

The fusion is **light VAE-style**: KL is in the loss but at a very small weight
(`1e-3`), so the latent is mostly deterministic. The point is regularization,
not generation.

### Latent and similarity head

The latent `z ∈ ℝ^d_latent` is passed through a small `sim_proj` MLP to produce
`sim_embed`, the embedding used for contrastive learning and inference-time
cosine similarity. Keeping the projection separate from `z` lets the latent
serve reconstruction without being collapsed to a contrastive surface.

### Decoders

All decoders condition on `z` via a shared `AtomQueryDecoder`: `max_atoms`
learned query vectors cross-attend to `[z, fused_track_tokens]` and produce
`max_atoms` "slot" hidden vectors plus a per-slot atom-existence logit. Each
track decoder then maps slots to its reconstruction target — atom-level heads
read individual slots, pair-level heads read symmetric outer products
(`hi ⊕ hj`), triple-level heads (bond angles) read three-way concatenations.

The 3D decoder reconstructs **atom types, bond lengths, and bond angles** — not
raw xyz coordinates. This is consistent with the encoder: since no orientation
information enters the latent, there is nothing to decode back to absolute
coordinates.

### Distillation model

Defined in [distillation/](distillation/). Architecturally simple:

```
SMILES string
   │
   ▼  regex tokenizer (~65 tokens, atoms + bonds + brackets)
   │
   ▼  Embedding + learned positional encoding
   │
   ▼  TransformerEncoder × N (pre-norm, 8-head)
   │
   ▼  LayerNorm
   │
   ▼  take CLS hidden state
   │
   ▼  MLP head → L2-normalize
   │
   ▼  256-D embedding (same dim as mol_struct_ae's sim_embed)
```

Defaults: 6 layers × 256 hidden × 8 heads ≈ 5 M params. Trained with cosine
distillation loss `1 - cos(distill_embed, sim_embed)` against pre-computed
`mol_struct_ae` outputs (see [Training](#training)).

**Why distill instead of using `mol_struct_ae` directly at inference?**
`mol_struct_ae` requires ~100 ms of RDKit work per molecule (ETKDG embedding,
MMFF optimization, Gasteiger charges). The distillation model inputs raw SMILES
and skips all of it — ~1 ms per molecule on GPU, suitable for screening 10⁶+
libraries.

---

## Data format

Two dataclasses in [data.py](mol_struct_ae/data.py):

### `MolSample` (one molecule, variable size, stored in shards)

The sample is **SE(3)-invariant by construction**: only scalar geometric
features (bond lengths / angles / dihedrals) are stored — no raw xyz
coordinates and no voxel grids. xyz is used during featurization to derive
those scalars and then discarded.

| Field | Shape | Notes |
|---|---|---|
| `atom_feats_2d` | `[N, F_atom]` | per-atom 2D features (default 32-dim) |
| `bond_feats_2d` | `[E, F_bond]` | per-edge bond features |
| `edge_index_2d` | `[2, E]` | directed edge list (both directions for undirected bonds) |
| `atom_types` | `[N, F_atype]` | numeric atom-type embedding (atomic num, radii, mass, …) |
| `bond_lengths` | `[E]` | Å, aligned with `edge_index_2d` |
| `angle_index` | `[3, A]` | rows are (i, j, k) — angle at j |
| `bond_angles` | `[A]` | radians, 0..π |
| `partial_charges` | `[N]` | Gasteiger partial charges |
| `pharma_feats` | `[N, F_pharma]` | binary pharmacophore labels |
| `dihedral_index` | `[4, T]` | rows are (i, j, k, l) |
| `dihedral_angles` | `[T]` | signed dihedrals in (−π, π] |
| `smiles` | `str` (default `""`) | canonical SMILES — lets shard-based pipelines recover molecule identity without re-featurizing |

### `MolBatch` (padded, dense, model-ready)

`collate(samples, max_atoms, max_dihedrals)` produces dense batched tensors.
No `coords` or `charge_grid` — the encoders are coordinate-free.

| Field | Shape |
|---|---|
| `atom_feats_2d` | `[B, N, F_atom]` |
| `bond_feats_2d` | `[B, N, N, F_bond]` |
| `adj_2d` | `[B, N, N]` |
| `atom_types` | `[B, N, F_atype]` |
| `bond_lengths` | `[B, N, N]` |
| `bond_angles` | `[B, N, N, N]` |
| `angle_mask` | `[B, N, N, N]` bool |
| `partial_charges` | `[B, N]` |
| `pharma_feats` | `[B, N, F_pharma]` |
| `dihedral_index` | `[B, T, 4]` |
| `dihedral_angles` | `[B, T]` |
| `dihedral_mask` | `[B, T]` bool |
| `atom_mask` | `[B, N]` bool |
| `num_atoms` | `[B]` |

`MolBatch.pair_indices [P, 2]` and `MolBatch.pair_labels [P]` are optional and
drive the contrastive head — see [Training](#training).

---

## Featurization

The ready-to-run pipeline is in [utils/featurize.py](utils/featurize.py).
It's generic — point it at any CSV with a SMILES column:

```bash
python utils/featurize.py \
    --csv data/zinc250k.csv --out data/shards \
    --max-atoms 64 --shard-size 1024 --workers 8
```

What it produces: one `.pt` file per shard, each containing a Python
`list[MolSample]` (up to `--shard-size` molecules, sparse — no padding).
Resume-safe: if interrupted mid-run, re-invoking with the same `--out` directory
skips the input rows already processed.

For each row in the CSV, [featurize_smiles()](utils/featurize.py) does:

| Field | How it's computed |
|---|---|
| (parse) | `Chem.MolFromSmiles`, then `canonical_smiles = Chem.MolToSmiles(mol)` stored on the sample |
| (3D) | `AddHs` → `EmbedMolecule` (ETKDG v3) → `MMFFOptimizeMolecule` — xyz is used only to derive the scalars below, then discarded |
| `atom_feats_2d` (32-D) | one-hot top-10 atomic number, formal charge, one-hot hybridization, aromaticity, total H, in-ring, chirality CW/CCW, degree, explicit/implicit valence, is-isotope |
| `atom_types` (16-D) | continuous: atomic-num/50, covalent + vdW radii, aromatic, formal charge, total H, mass/100, degree, in-ring, in 5-ring, in 6-ring |
| `bond_feats_2d` (8-D) | one-hot bond type, stereo, in-ring, conjugated, aromatic |
| `bond_lengths`, `bond_angles`, `dihedral_angles` | derived from the post-MMFF xyz; sparse triples / quadruples over real bonded atoms — SE(3)-invariant scalars |
| `pharma_feats` (8-D) | RDKit `BaseFeatures.fdef` → donor / acceptor / aromatic / hydrophobic / ±ionizable / halogen / in-ring |
| `partial_charges` | Gasteiger charges via `ComputeGasteigerCharges` |

For large benchmark runs, [utils/parallel_featurize.py](utils/parallel_featurize.py)
wraps `featurize_smiles` in a `multiprocessing.Pool` (default 8 workers) so
one-off SMILES batches don't bottleneck on RDKit's serial ETKDG + MMFF.

---

## Training

Two stages, run sequentially:

```
                  (RDKit, parallel CPU,    (load shards, sparse multi-track
                   ~1 h on 8 cores)         losses, GPU, ~20 h on A10G)
SMILES CSV ──── utils/featurize.py ───── train_mol_struct_ae.py
                  data/shards/          → runs/zinc250k/best.pt    (stage 1)
                                                  │
                                                  │ (load mol_struct_ae ckpt,
                                                  ▼  run over shards, save
                                                     SMILES + sim_embed pairs)
                                  scripts/precompute_mol_struct_ae_embeds.py
                                  → runs/mol_struct_ae_embeds.pt
                                                  │
                                                  ▼     (cosine distillation loss,
                                                         GPU, ~20 min)
                                          train_distillation.py
                                          → runs/distill/best.pt   (stage 2)
```

### Stage 1 — `mol_struct_ae` (5-track autoencoder)

Train with reconstruction over all 5 tracks + KL + cross-track consistency.

```bash
# Featurize once (CSV → sharded MolSample list[.pt])
python utils/featurize.py --csv data/zinc250k.csv --out data/shards \
                            --max-atoms 64 --shard-size 1024 --workers 8

# Train (resume-safe via --resume <ckpt>)
python train_mol_struct_ae.py --shard-dir data/shards \
                                --out-dir runs/zinc250k \
                                --epochs 20 --batch-size 64 --max-atoms 48 \
                                --hidden 96 --latent 256 --lr 3e-4 --amp
```

[train_mol_struct_ae.py](train_mol_struct_ae.py) reads sharded `MolSample`s
via [`ShardedMolDataset`](mol_struct_ae/dataset.py), dense-collates each batch,
and runs [`MolStructAutoencoder`](mol_struct_ae/model.py) with AMP and a
cosine-warmup LR schedule.

**Two performance features in the data path:**

1. **`ShardSequentialSampler`** (also in [dataset.py](mol_struct_ae/dataset.py)).
   Walks samples shard-by-shard so the single-slot shard cache stays warm. With
   `shuffle=True` on a global random sampler, each batch of 64 would pull from
   ~64 different shards and force a `torch.load` per molecule (~100× slowdown
   on a network volume). The sampler shuffles *across* shards each epoch and
   *within* each shard, giving equivalent statistical mixing at a fraction of
   the I/O cost.

2. **Index cache.** First instantiation of `ShardedMolDataset` walks every
   shard to filter oversize molecules and build `(shard_idx, local_idx)` pairs;
   this is then written to `<shard_dir>/.index.maxatoms_<N>.json`. Subsequent
   runs (resumes, multi-epoch chains) load the index from JSON in ~2 seconds
   instead of repeating the multi-minute scan.

### Stage 2 — `distillation` (SMILES → embedding)

Three substeps:

```bash
# (a) Build SMILES vocab from the same SMILES library
python scripts/build_vocab.py --csv data/zinc250k.csv --out runs/vocab.json

# (b) Run the trained mol_struct_ae over the shards → (SMILES, sim_embed) pairs
python scripts/precompute_mol_struct_ae_embeds.py \
    --checkpoint runs/zinc250k/best.pt \
    --shard-dir data/shards \
    --out runs/mol_struct_ae_embeds.pt

# (c) Train the distillation transformer to match those embeddings
python train_distillation.py --pairs runs/mol_struct_ae_embeds.pt \
                               --vocab runs/vocab.json \
                               --out-dir runs/distill \
                               --epochs 20 --batch-size 256 \
                               --hidden 256 --num-layers 6 --num-heads 8 \
                               --lr 3e-4
```

The pairs file (`mol_struct_ae_embeds.pt`) is:

```python
{"smiles": List[str],            # length L (the canonical SMILES of each mol)
 "embeds": np.ndarray (L, 256)}  # the L2-normalized sim_embed targets
```

The distillation loss is the simplest thing that works:
**`1 − cos(distill_embed, sim_embed)`**, both already L2-normalized.

### `mol_struct_ae` losses (stage 1)

Implemented in [losses.py](mol_struct_ae/losses.py), aggregated by
`compute_total_loss(...)` and weighted by `LossWeights`.

| Loss | Function | Weight (default) | Notes |
|---|---|---|---|
| Atom existence | `BCE(atom_mask_logits, atom_mask)` | 1.0 | predicts which slots are real |
| 2D atom features | masked MSE | 1.0 | per-atom feature vector |
| Adjacency | masked BCE | 1.0 | symmetric pairwise |
| Bond features | masked MSE | 0.5 | only over bonded pairs |
| Atom types (3D) | masked MSE | 0.5 | |
| Bond lengths | masked MSE | 0.5 | only over bonded pairs |
| Bond angles | masked MSE | 0.5 | only over angle triples |
| Partial charges | masked MSE | 1.0 | |
| Pharma | masked BCE | 0.5 | |
| Torsion | masked MSE on `(sin φ, cos φ)` | 0.5 | sin/cos to respect angular topology |
| KL (β-VAE) | `−½ Σ(1+logσ²−μ²−exp(logσ²))` | 1e-3 | very light regularization |
| Contrastive | NT-Xent on `sim_embed` | 1.0 | uses `pair_indices` / `pair_labels` |
| Cross-track consistency | mean MSE of all `f_ij(token_i) → token_j.detach()` | 0.1 | forces the global fusion to actually mix tracks |

Note: raw xyz coordinate reconstruction and voxel grid reconstruction were
removed. The 3D encoder is SE(3)-invariant (uses only distances and angles),
so there is no orientation signal in the latent to reconstruct coordinates from.
The voxel grid was removed because PCA-canonicalized grids are not consistently
oriented across structurally similar molecules — adding a single atom shifts the
PCA frame enough to produce very different grids for nearly identical molecules.

### Contrastive head (optional, not used in the default pipeline)

`mol_struct_ae` includes an NT-Xent contrastive term that activates when
`MolBatch.pair_indices` / `pair_labels` are supplied — labelled pairs of
similar (`1`) or dissimilar (`0`) molecules (e.g. Tanimoto > 0.7, same
Bemis-Murcko scaffold, same target/IC50). In the current pipeline this term
is **left at weight 0** because the distillation model is what actually serves
similarity at inference; `mol_struct_ae` is trained as a pure-reconstruction
autoencoder.

Gradient clipping at `clip_grad_norm=1.0` is on by default.

### `distillation` loss (stage 2)

Just one term:

| Loss | Function | Notes |
|---|---|---|
| Cosine distillation | `1 − cos(distill_embed, sim_embed)` | both L2-normalized; equals 0 when the distillation matches `mol_struct_ae` exactly, 2 at maximum opposition |

Trained with AdamW + cosine-warmup LR. Val split is 2% by default. Best
checkpoint by held-out cosine loss is saved to `runs/distill/best.pt`.

---

## Inference

Inference uses **only the distillation model** — no RDKit, no MolSample, no
featurization. The distillation model takes SMILES strings directly:

```python
import torch
from distillation import SmilesEncoder, SmilesEncoderConfig, SmilesTokenizer

tokenizer = SmilesTokenizer.load("runs/vocab.json")
ckpt      = torch.load("runs/distill/best.pt", weights_only=False)
cfg       = SmilesEncoderConfig(**ckpt["encoder_config"])
model     = SmilesEncoder(cfg).eval()
model.load_state_dict(ckpt["model"])

ids, mask = tokenizer.encode_batch(["CC(=O)Oc1ccccc1C(=O)O",     # aspirin
                                     "Cn1cnc2n(C)c(=O)n(C)c(=O)c12"])  # caffeine
with torch.no_grad():
    embeds = model(ids, mask)     # [2, 256], L2-normalized
print(embeds.shape, embeds.norm(dim=-1))
```

~1 ms/mol on GPU. Cosine similarity is just `embeds @ embeds.T`.

### Library similarity search

[scripts/simsearch.py](scripts/simsearch.py) is the end-to-end CLI: embed
every SMILES in a CSV with the distillation model, then for each query SMILES
print the top-K matches. Includes a Morgan-FP Tanimoto column side-by-side as
a sanity baseline.

```bash
python scripts/simsearch.py \
    --distillation-ckpt runs/distill/best.pt \
    --vocab runs/vocab.json \
    --library data/zinc250k.csv \
    --queries "aspirin:CC(=O)Oc1ccccc1C(=O)O" \
              "caffeine:Cn1cnc2n(C)c(=O)n(C)c(=O)c12" \
    --top-k 10 --out runs/simsearch.json
```

### Embedding with `mol_struct_ae` directly (advanced)

The 5-track autoencoder can be invoked for inference too — useful for
debugging, for the precompute step in stage 2, or if you want the
reconstruction outputs alongside the embedding. ~100 ms/mol because of RDKit.

```python
import torch, torch.nn.functional as F
from mol_struct_ae import MolStructAutoencoder
from mol_struct_ae.data import collate
from mol_struct_ae.model import MolAEConfig
from utils.featurize import featurize_smiles

ckpt  = torch.load("runs/zinc250k/best.pt", map_location="cpu", weights_only=False)
cfg   = MolAEConfig(**{k: v for k, v in ckpt["config"].items()
                         if k in {"max_atoms", "hidden_dim", "latent_dim"}})
model = MolStructAutoencoder(cfg).eval()
model.load_state_dict(ckpt["model"])

sample = featurize_smiles("CC(=O)Oc1ccccc1C(=O)O", max_atoms=cfg.max_atoms)
batch  = collate([sample], max_atoms=cfg.max_atoms)
with torch.no_grad():
    z = F.normalize(model(batch, sample=False)["sim_embed"], dim=-1)
print(z.shape)   # [1, latent_dim]
```

### How similarity is computed

Both `MolStructAutoencoder.sim_embed` and the distillation model's output are
L2-normalized, so cosine similarity falls out as a dot product. Self-similarity
is 1.0; pair similarities live in roughly [−1, 1].

For >1M molecules: embed once with the distillation model, stack into `[L, d]`,
and hand to FAISS (`IndexFlatIP` up to ~1M, `IndexIVFPQ` beyond). The 5-track
decoders are never needed at inference — they exist so the encoder learns a
representation that *can* reconstruct each track, acting as a strong inductive
prior against latent collapse.

---

## Repo layout

```
mol_struct_ae/                            ← repo root
├── README.md
├── data/zinc250k.csv
├── runs/                                  ← runtime outputs (ckpts, vocab, results)
│
├── mol_struct_ae/                         ← MOL_STRUCT_AE package (5-track autoencoder)
│   ├── __init__.py
│   ├── data.py            ← MolSample (carries .smiles), MolBatch, collate
│   ├── dataset.py         ← ShardedMolDataset + ShardSequentialSampler + index cache
│   ├── encoders.py        ← Graph2DEncoder, Geometry3DEncoder, ChargeEncoder,
│   │                        PharmaEncoder, TorsionEncoder, DenseGATLayer, GaussianRBF
│   ├── cross_attention.py ← PerAtomCrossAttention, TrackTokenFusion
│   ├── decoders.py        ← AtomQueryDecoder + per-track decoders
│   ├── model.py           ← MolStructAutoencoder, MolAEConfig
│   └── losses.py          ← compute_total_loss, LossWeights, NT-Xent,
│                            CrossTrackConsistency
│
├── distillation/                          ← DISTILLATION package (SMILES-only model)
│   ├── __init__.py
│   ├── smiles_tokenizer.py ← SmilesTokenizer + regex-based atom-wise tokenization
│   └── smiles_encoder.py   ← SmilesEncoder (transformer with CLS-pool head)
│
├── utils/                                 ← shared helpers
│   ├── __init__.py
│   ├── featurize.py                ← CSV → sharded MolSample list[.pt]  (CLI runnable)
│   ├── parallel_featurize.py       ← multithreaded featurize_smiles_parallel()
│   ├── feature_utils.py            ← featurize_all / embed_all / embed_from_shards /
│   │                                 morgan_fps / tanimoto_matrix
│   ├── moleculenet_benchmark.py    ← MoleculeNet scaffold-split linear probe
│   │                                 + leakage filter against training-SMILES set
│   ├── moleculeace_benchmark.py    ← MoleculeACE activity-cliff RMSE probe
│   └── litpcba_benchmark.py        ← Virtual-screening EF@K via PyTDC HTS datasets
│
├── scripts/
│   ├── build_vocab.py                          ← build SMILES vocab from CSV
│   ├── precompute_mol_struct_ae_embeds.py      ← run mol_struct_ae over a library →
│   │                                             (SMILES, embed) pairs for distillation
│   └── simsearch.py                            ← distillation cosine vs. Tanimoto top-K
│
├── train_mol_struct_ae.py                ← train the 5-track autoencoder
├── train_distillation.py                 ← train the distillation model on pre-computed
│                                           mol_struct_ae embeddings
│
├── modal_mol_struct_ae.py                ← Modal pipeline: featurize, train, simsearch,
│                                           benchmarks (MoleculeNet, MoleculeACE, LIT-PCBA
│                                           via TDC HTS), index cache, leakage extractor
├── modal_distillation.py                 ← Modal pipeline: precompute_embeds + train
└── modal_simsearch.py                    ← Modal simsearch runner (uses distillation)
```

### Running

**Train mol_struct_ae locally (single GPU):**

```bash
# 1. Featurize once (CPU, parallel)
python utils/featurize.py --csv data/zinc250k.csv --out data/shards \
                            --max-atoms 64 --workers 8

# 2. Train mol_struct_ae (resume-safe via --resume)
python train_mol_struct_ae.py --shard-dir data/shards --out-dir runs/zinc250k \
                                --epochs 20 --batch-size 64 --max-atoms 48 \
                                --hidden 96 --latent 256 --lr 3e-4 --amp
```

**Distill mol_struct_ae → SMILES-only model locally:**

```bash
# 1. Build SMILES vocab
python scripts/build_vocab.py --csv data/zinc250k.csv --out runs/vocab.json

# 2. Run mol_struct_ae over the library → (smiles, embed) pairs
python scripts/precompute_mol_struct_ae_embeds.py \
    --checkpoint runs/zinc250k/ckpt_epochN.pt \
    --shard-dir data/shards --out runs/mol_struct_ae_embeds.pt

# 3. Train the distillation model
python train_distillation.py --pairs runs/mol_struct_ae_embeds.pt \
                               --vocab runs/vocab.json \
                               --out-dir runs/distill --epochs 20
```

**On Modal (orchestrated CPU featurize → A10G train on a shared volume):**

```bash
# Train mol_struct_ae:
MODAL_PROFILE=<profile> python -m modal run --detach modal_mol_struct_ae.py::precompute
MODAL_PROFILE=<profile> python -m modal run --detach modal_mol_struct_ae.py::auto

# Resume mol_struct_ae from a local checkpoint:
MODAL_PROFILE=<profile> python -m modal run --detach \
    modal_mol_struct_ae.py::resume_from_local_ckpt \
    --local-ckpt runs/zinc250k_remote/ckpt_step002000.pt --epochs 1

# Continue training from a checkpoint on the volume (resume + lower LR):
MODAL_PROFILE=<profile> python -m modal run --detach modal_mol_struct_ae.py::zinc15_train \
    --epochs 3 --max-atoms 64 --lr 1e-4 \
    --resume /root/runs/zinc15_5m_diverse_run/ckpt_epoch000.pt

# Distillation pipeline (uploads ckpt + precomputes embeds + trains distillation):
MODAL_PROFILE=<profile> python -m modal run --detach modal_distillation.py

# Pull checkpoints back when training finishes
MODAL_PROFILE=<profile> python -m modal run modal_mol_struct_ae.py::download_results
MODAL_PROFILE=<profile> python -m modal run modal_distillation.py::download_results
```

**Similarity search** (Modal wrapper around `simsearch.py`):

```bash
MODAL_PROFILE=<profile> python -m modal run --detach modal_simsearch.py
MODAL_PROFILE=<profile> python -m modal run modal_simsearch.py::download_results

# Or the inline mol_struct_ae simsearch (no distillation needed):
MODAL_PROFILE=<profile> python -m modal run modal_mol_struct_ae.py::simsearch \
    --query-smiles "CC(=O)Oc1ccccc1C(=O)O" --query-name aspirin --top-k 20
```

### Utility jobs on Modal

```bash
# Build the index cache for a shard dir (one-time, ~15 min on 8 CPUs):
MODAL_PROFILE=<profile> python -m modal run --detach \
    modal_mol_struct_ae.py::index_cache --max-atoms 64

# Extract the deduplicated canonical-SMILES set of the training data
# (used by all benchmark suites for leakage filtering):
MODAL_PROFILE=<profile> python -m modal run --detach \
    modal_mol_struct_ae.py::extract_train_smiles
```

### Benchmarking

Four benchmark suites are wired up as Modal entrypoints — all use **frozen
embeddings** from the trained `final.pt` and run a Morgan-FP baseline side by
side for apples-to-apples comparison. Each one is **leakage-filtered** against
the canonical-SMILES set extracted from the training shards.

```bash
# 1. MoleculeNet — scaffold split, linear probe, AUROC/RMSE per dataset.
#    BBBP, BACE, ClinTox, HIV, ESOL, FreeSolv, Lipo.
MODAL_PROFILE=<profile> python -m modal run \
    modal_mol_struct_ae.py::moleculenet_benchmark

# 2. MoleculeACE — activity-cliff RMSE on 8 ChEMBL targets.
#    Tests whether the embedding handles structurally-similar / activity-different
#    pairs that fingerprint-only models traditionally fail on.
MODAL_PROFILE=<profile> python -m modal run \
    modal_mol_struct_ae.py::moleculeace_benchmark

# 3. Virtual-screening EF@1% / EF@5% / AUROC on PyTDC's HTS PubChem-bioassay
#    datasets. (PyTDC does not ship LIT-PCBA itself; these are the
#    scientifically-equivalent HTS screens it does ship.)
MODAL_PROFILE=<profile> python -m modal run \
    modal_mol_struct_ae.py::litpcba_benchmark

# 4. Perturbation smoke test — cosine similarity for curated pairs differing
#    by one methyl / methylene / halogen / phenyl. Sanity check that the
#    embedding's local neighborhood structure makes chemical sense.
MODAL_PROFILE=<profile> python -m modal run \
    modal_mol_struct_ae.py::perturbation_smoke
```

The benchmark utilities (`utils/moleculenet_benchmark.py`,
`utils/moleculeace_benchmark.py`, `utils/litpcba_benchmark.py`) all share the
same featurization-and-leakage-filter pipeline. They use
`utils/parallel_featurize.py` to multi-process RDKit ETKDG + MMFF across
12 workers, then dense-batch the resulting `MolSample`s through the model
on GPU.

Requirements: `torch >= 2.0`, `rdkit >= 2023.3.1`. No PyG dependency — the
GATConv is implemented in dense form
([encoders.py:`DenseGATLayer`](mol_struct_ae/encoders.py)) so the whole pipeline
runs on batched tensors with `atom_mask`. If you'd rather use PyG's sparse
`GATConv` for very large molecules, swap `Graph2DEncoder` — the rest of the
pipeline is decoupled.
