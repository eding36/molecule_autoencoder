# mol_struct_ae

A two-stage pipeline for **structural similarity search** over small molecules.

**Stage 1 — `mol_struct_ae`**: a **Pairformer-based** structural autoencoder
modelled after AlphaFold-3. Two unified representations are updated jointly at
every block:

   - **`single_repr [B, N, d_single]`** — one vector per atom. Carries atom
     identity (atomic number, hybridization, …), atom types (radii, mass,
     …), Gasteiger partial charges, and pharmacophore flags.
   - **`pair_repr [B, N, N, d_pair]`** — one vector per atomic bond. Carries
     **geometric information** at every relevant bond-hop distance:
     1-bond (adjacency, bond features, bond lengths), 2-bond (bond angle information
     scattered into the terminal atoms of the bond angle), and 4-bond (signed dihedrals encoded and scattered into both
     the terminal atoms of the dihedral `pair[i, l]` *and* the central, rotational axis bond `pair[j, k]`).

The Pairformer processes both atom and bond representations through 4 blocks of OuterProductMean (single → pair), TriangleAttention (pair self-
update), and AttentionWithPairBias (pair → single). All input geometry, atom
identity, and chemical labels live in one of two tensors instead of having
their own encoders.

The Pairformer used here  is a stack of 4 blocks where the per-atom
`single_repr` and per-pair `pair_repr` update each other at every layer
(OuterProductMean, TriangleAttention, AttentionWithPairBias).
The post-Pairformer `single_repr` is pooled to a 256-D latent `z` per molecule. Reconstruction losses computed after decoding the latent keep the latent informative.

**Stage 2 — `distillation`**: a small SMILES-only transformer trained to
reproduce `mol_struct_ae`'s `z` latent embedding directly from the SMILES string. Once
trained, users can `tokenize(smiles)` and then run the distillation model on the tokenized smiles to generate an embedding. No RDKit or featurization required, ~1 ms per molecule.

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

This README covers:

1. [Architecture](#architecture) — the Pairformer-based encoder (mol_struct_ae) (single + pair representations updated jointly at every block, AF3-style), the latent, and the per-track decoders. Plus the `distillation` model.
2. [Data format](#data-format) — what a `MolSample` and a `MolBatch` contain.
3. [Featurization](#featurization) — how raw molecules are turned into `MolSample`s and sharded to disk.
4. [Training](#training) — stage 1 (mol_struct_ae) and stage 2 (distillation).
5. [Inference and similarity](#inference) — embedding any SMILES with the distillation model.
6. [Repo layout and running](#repo-layout) — also covers the MoleculeNet
   fine-tuning benchmark, leakage-filtered against the training set.

---

## Architecture

The encoder is a **Pairformer** modelled on AlphaFold-3's Pairformer. Two
unified representations — *single* (per-atom) and *pair* (per-atom-pair) —
update jointly at every block. 

```
  ┌────────────────────────── INPUT TRACKS ───────────────────────────┐
  │  atom_feats   atom_type   partial_charge  pharmacokinetic props   │
  │  adj_2d   bond_feats   bond_lengths   bond_angles   dihedrals     │
  └───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                        InputEmbedder
              (per-atom inputs → single_repr [B, N, d_single])
              (per-pair inputs → pair_repr   [B, N, N, d_pair])
                              │
                              ▼
                   ┌── PairformerBlock × 4 ───┐
                   │                          │
                   │  1. pair_repr += OuterProductMean(single_repr)
                   │  2. pair_repr += TriangleAttention(pair_repr)
                   │  3. pair_repr += PairTransition(pair_repr)
                   │  4. single_repr += AttnWithPairBias(single_repr, pair_repr)
                   │  5. single_repr += SingleTransition(single_repr)
                   │                          │
                   └──────────────────────────┘
                              │
                              ▼
                     GlobalAggregator (single_repr)
              ├──→ CLS-pool(single_repr) → μ, log σ² → reparam(μ, σ²) → z
              └──→ 5 other learned context tokens for decoder input
                              │
                              ▼
                  AtomQueryDecoder (z + 5 CLS_tokens → decoded_molecule [B, N, d])
                              │
                        Decoder Stack
       ┌──────────────┬───────┴──────────┬──────────────┬─────────────┐
       ▼              ▼                  ▼              ▼             ▼
2D Feature        3D Feature           Charge   Pharmacokinetic    Dihedral/torsion 
    (atom feats,  (atype, bond     (per-atom q)   (per-atom)     (36-bin
     adj, bond)    len, bond ang)                                 classification of
                                                                  dihedral angle)
```

### The Pairformer block

At every block, single and pair representations exchange information bidirectionally:

| Step | Op | Function |
|------|----|----------|
| ① | `OuterProductMean` | single → pair: for each pair (i,j), use outer-product of low-dim projections of single[i] and single[j] |
| ② | `TriangleAttentionStartingNode` | pair self-update: for each (i,j), attend over k with bias from pair[k,j] — captures 3-atom geometry through pair_repr |
| ③ | `PairTransition` | pair MLP — feed-forward on the pair tensor |
| ④ | `SingleAttentionWithPairBias` | pair → single: standard single self-attention where pair_repr biases attention logits — pair information flows back into per-atom representation |
| ⑤ | `SingleTransition` | single MLP — feed-forward on the single tensor |

Each step is a **residual update with pre-LayerNorm** and (for the attention
modules) a sigmoid gate on the output, following AF3's design. The whole
block is one "communication round" between the two representations.

### Input embedding

Every input from the featurization is mapped into either `single_repr` (per
atom) or `pair_repr` (per atom pair) at a single input embedder:

| Input | Goes into | How |
|-------|-----------|-----|
| `atom_feats_2d` (32-D) | single | linear embedding |
| `atom_types` (16-D) | single | linear embedding |
| `partial_charges` (scalar) | single | linear embedding of unsqueezed scalar |
| `pharma_feats` (8-D binary) | single | linear embedding |
| `adj_2d` (0/1) | pair (1-bond) | small embedding table |
| `bond_feats_2d` (8-D) | pair (1-bond) | linear embedding |
| `bond_lengths` (scalar Å) | pair (1-bond) | RBF(0–5 Å) → linear, gated by adj_2d |
| `bond_angles` (radians) | pair (2-bond) | Each pair[i, k] (and pair[k, i]) obtains a feature RBF(0–π) for every angle triple (i, j, k). Multiple j's connecting the same (i, k) accumulate additively. |
| `dihedral_angles` (signed φ) | pair (4-bond) | Given a dihedral (i,j,k,l), the terminal pair `pair[i,l] & pair[l,i]` and the rotational bond`pair[j,k] & pair[k,j]` all get features: `[RBF(cos kφ), RBF(sin kφ)]` for `k ∈ {1, 2, 3}`. The 3 harmonics natively encode the 3-fold symmetry of σ-bond rotation (gauche+, trans, gauche-). |

All five complementary views of the molecule (topology, 3D geometry, partial
charges, pharmacophore labels, dihedrals) are summed into these two tensors,
then processed jointly by the Pairformer backbone.

**SE(3) invariance.** All Pairformer inputs are either topology-based (2D
graph, pharmacophore labels) or SE(3)-invariant scalars (bond lengths, bond
angles, dihedral angles, partial charges). No raw xyz coordinates enter the
model. Two conformers of the same molecule that differ only by a global
rotation or translation produce identical latent vectors.


### Latent and similarity head

After the Pairformer outputs an atomic representation of the molecule `single_repr`, `GlobalAggregator` contains a MultiHeadAttention layer that does two things:
   1. Transforms `single_repr [1,96,128]` into a `[1,256]` latent vector `z` by attending to each atom in `single_repr` . 
   2. Generates 5 other context tokens that produce other types of summaries of the molecule (e.g. charge, polarity, 3D structure, torsional features)

- **Pooling of `single_repr`.** Its attention output is fed into a `μ` and `log σ²` Linear() layer, then reparameterized into the latent `z ∈ ℝ^256`. KL is in the loss at a small weight (default 2e-3) so `z` is mostly deterministic; the VAE regularizer
  is for shape and spread, not generation. 
- **Generation of 5 other context tokens** Their attention outputs are the 5 decoder context tokens**, which are passed alongside `z` into the per-track decoders.

### Decoders
`AtomQueryDecoder` reads in `z` and the 5 context tokens, and returns two vectors: `[B,max_atoms,128]` (the molecular vector) and `[B,max_atoms]` (a per atom existence logit) through cross & self attention. The cross attention is done through `z` attending to the context tokens. 

Each decoder then reads in the molecular vector and is trained to match its reconstruction target. 
`Graph2DDecoder` attempts to reconstruct adjacency and bond information, `Geometry3DDecoder` attempts to reconstruct atom type, bond length, and bond angle information. ChargeDecoder attempts to reconstruct charge values per atom. `PharmaDecoder` attempts to reconstruct pharmacokinetic properties of the molecule. 

`TorsionDecoder` is unique from the others. It receives the molecular vector, the pair_representation vector from the Pairformer, 4-atom indices of all dihedrals, and  attempts to reconstruct dihedral angles in the molecule. The output is a 36-bin
classification logit vector over [−π, π] (10°-wide buckets). This method is robust to
our conformation generator's ~10–20° conformer noise.

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

### Stage 1 — `mol_struct_ae` (Pairformer autoencoder)

Train with reconstruction losses over all per-track decoder outputs (atom
features, adjacency, bond features, atom types, bond lengths, bond angles,
partial charges, pharmacophore flags, dihedrals as 36-bin classification)
plus a small KL regularizer on the latent.

```bash
# Featurize once (CSV → sharded MolSample list[.pt])
# max_atoms=96 is the recommended default — covers >95% of drug-like molecules.
python utils/featurize.py --csv data/zinc250k.csv --out data/shards \
                            --max-atoms 96 --shard-size 1024 --workers 8

# Train (resume-safe via --resume <ckpt>)
# batch_size=24 fits comfortably at max_atoms=96 with AMP on a 24-GB GPU; the
# triangle-attention scores tensor [B, N, N, N, H] is the memory ceiling.
python train_mol_struct_ae.py --shard-dir data/shards \
                                --out-dir runs/zinc250k \
                                --epochs 3 --batch-size 24 --max-atoms 96 \
                                --hidden 128 --latent 256 --lr 3e-4 --amp
```

`--hidden` controls `d_single` (per-atom hidden width). `d_pair` (32) and
`num_pairformer_blocks` (4) are `MolAEConfig` defaults set in code rather
than via CLI.

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
| Atom types (3D) | masked MSE | 0.75 | |
| Bond lengths | masked MSE | 1.0 | only over bonded pairs |
| Bond angles | masked MSE | 1.0 | only over angle triples |
| Partial charges | masked MSE | 0.5 | |
| Pharma | masked BCE | 0.5 | |
| Torsion | masked cross-entropy over 36 bins of `[−π, π]` | 1.0 | classification, not regression — robust to ETKDG noise and multimodal targets (gauche+/trans/gauche-) |
| KL (β-VAE) | `−½ Σ(1+logσ²−μ²−exp(logσ²))` | 2e-3 | small but non-trivial — keeps μ from collapsing onto a single drug-like manifold during long training |

Note: raw xyz coordinate reconstruction and voxel grid reconstruction were
removed. The 3D encoder is SE(3)-invariant (uses only distances and angles),
so there is no orientation signal in the latent to reconstruct coordinates from.
The voxel grid was removed because PCA-canonicalized grids are not consistently
oriented across structurally similar molecules — adding a single atom shifts the
PCA frame enough to produce very different grids for nearly identical molecules.

Gradient clipping at `clip_grad_norm=1.0` is on by default. There is no
contrastive loss — the older NT-Xent head was removed; the latent is
shaped entirely by reconstruction + KL.

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

The Pairformer autoencoder can be invoked for inference too — useful for
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

---

## Repo layout

```
mol_struct_ae/                            ← repo root
├── README.md
├── data/zinc250k.csv
├── runs/                                  ← runtime outputs (ckpts, vocab, results)
│
├── mol_struct_ae/                         ← MOL_STRUCT_AE package (pairformer autoencoder)
│   ├── __init__.py
│   ├── data.py            ← MolSample (carries .smiles), MolBatch, collate
│   ├── dataset.py         ← ShardedMolDataset + ShardSequentialSampler + index cache
│   ├── pairformer.py      ← InputEmbedder, OuterProductMean, TriangleAttention,
│   │                        SingleAttentionWithPairBias, PairformerBlock × N,
│   │                        GlobalAggregator (the full encoder backbone)
│   ├── decoders.py        ← AtomQueryDecoder + per-track decoders
│   ├── model.py           ← MolStructAutoencoder, MolAEConfig
│   └── losses.py          ← compute_total_loss, LossWeights (reconstruction
│                            tracks + KL only)
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
│   └── benchmark_moleculenet.py    ← MoleculeNet END-TO-END FINE-TUNING benchmark
│                                     (Hu et al. / Mole-BERT protocol: scaffold
│                                     80/10/10, best-val epoch, multi-seed, multi-task;
│                                     self-contained — download + leakage helpers inline)
│
├── scripts/
│   ├── build_vocab.py                          ← build SMILES vocab from CSV
│   ├── precompute_mol_struct_ae_embeds.py      ← run mol_struct_ae over a library →
│   │                                             (SMILES, embed) pairs for distillation
│   └── simsearch.py                            ← distillation cosine vs. Tanimoto top-K
│
├── train_mol_struct_ae.py                ← train the Pairformer autoencoder
├── train_distillation.py                 ← train the distillation model on pre-computed
│                                           mol_struct_ae embeddings
│
├── modal_mol_struct_ae.py                ← Modal pipeline: featurize, train, simsearch,
│                                           MoleculeNet fine-tuning benchmark, index
│                                           cache, leakage extractor
├── modal_distillation.py                 ← Modal pipeline: precompute_embeds + train
└── modal_simsearch.py                    ← Modal simsearch runner (uses distillation)
```

### Running

**Train mol_struct_ae locally (single GPU):**

```bash
# 1. Featurize once (CPU, parallel)
python utils/featurize.py --csv data/zinc250k.csv --out data/shards \
                            --max-atoms 96 --workers 8

# 2. Train mol_struct_ae (resume-safe via --resume)
python train_mol_struct_ae.py --shard-dir data/shards --out-dir runs/zinc250k \
                                --epochs 3 --batch-size 24 --max-atoms 96 \
                                --hidden 128 --latent 256 --lr 3e-4 --amp
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
# (used by the MoleculeNet benchmark for leakage filtering):
MODAL_PROFILE=<profile> python -m modal run --detach \
    modal_mol_struct_ae.py::extract_train_smiles
```

### Benchmarking

One benchmark is wired up as a Modal entrypoint: **MoleculeNet end-to-end
fine-tuning** (the Mole-BERT / Hu et al. protocol). It is **leakage-filtered**
against the canonical-SMILES set from the training shards. 

```bash
# MoleculeNet — END-TO-END FINE-TUNING (scaffold 80/10/10, best-val epoch,
# multi-seed mean±std). Replicates Mole-BERT (ICLR 2023) Table 1, so the
# numbers are directly comparable. Single- and multi-task datasets:
# BBBP, BACE, ClinTox, SIDER, Tox21, ToxCast, MUV, HIV.
MODAL_PROFILE=<profile> python -m modal run \
    modal_mol_struct_ae.py::moleculenet_finetune \
    --datasets BBBP,BACE,ClinTox,SIDER --seeds 0,1,2,3,4,5,6,7,8,9 --epochs 100
```

The benchmark featurizes once per dataset (RDKit ETKDG + MMFF) and reuses the
`MolSample`s across all seeds. It's self-contained in
`utils/benchmark_moleculenet.py` (dataset download + leakage-filter helpers
are inline).

Requirements: `torch >= 2.0`, `rdkit >= 2023.3.1`. No PyG dependency — the
whole pipeline runs on dense batched tensors with `atom_mask`, with the
triangle attention in `pair_repr` handling all graph-structured updates.
