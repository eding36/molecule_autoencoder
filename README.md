# mol_struct_ae

This README covers:
1. [Overview](#overview)  
2. [Architectures & Input Features](#architectures) — the Pairformer-based encoder (mol_struct_ae) (single + pair representations updated jointly at every block, AF3-style), the latent, and the per-track decoders. Plus the `distillation` model.3. [Featurization](#featurization) — how raw molecules are turned into `MolSample`s and sharded to disk.
4. [Training](#training) — stage 1 (mol_struct_ae) and stage 2 (distillation).
5. [Inference](#inference) — embedding any SMILES with the distillation model. Similarity searching.
6. [Benchmarking](#benchmarking) —Using the MoleculeNet
   fine-tuning benchmark, leakage-filtered against the training set.

## Overview

This repo contains an ML framework that can be used for efficient **structural similarity search** over small molecules. It can also be fine-tuned for property prediction 
tasks, which will be detailed below. The ML framework consists firstly of a molecular autoencoder that requires initial featurization of your SMILES which can be 
expensive depending on your molecular library size. Thus, after the molecular autoencoder 
is trained to a deisred point, we train a distillation model as the second part of the framework which learns to match the molecular autoencoder's embeddings 
for purely SMILES (no featurization required). 

**Part 1 — `Molecular Autoencoder (mol_struct_ae)`**: a structural autoencoder modelled after 
AlphaFold-3's pairformer trunk. Two unified representations are updated 
jointly at every block:

   - **`single_repr [B, N, d_single]`** — Per atom representation. Carries atom identity information (atomic number, hybridization, …), 
   and atom attributes (radii, mass, Gasteiger partial charges, and pharmacophore properties).
   - **`pair_repr [B, N, N, d_pair]`** — Per bond representation. Carries **geometric information** at every relevant bond-hop distance:
     - 1-bond (adjacency, bond features, bond lengths)
     - 2-bond (bond angle information, assigned to terminal 
     atoms of bond angle)
     - 4-bond (dihedral angle between atoms `[i,j,k,l]` assigned to 
     both the terminal atoms of the dihedral `pair[i, l]` *and* the central, rotational axis bond `pair[j, k]`).

The Pairformer processes both atom and bond representations jointly 
through OuterProductMean, TriangleAttention, and AttentionWithPairBias. 
The post-Pairformer `single_repr` is pooled to a 256-D latent `z` per molecule. 

**Part 2 — `Distillation Model (distillation)`**: a lightweight transformer trained to reproduce 
`mol_struct_ae`'s `z` latent embedding directly from the SMILES string. Once trained, 
users can `tokenize(smiles)` and then run the distillation model on the tokenized smiles 
to generate an embedding. No RDKit or featurization required.

A visual workflow is depicted below.
```
                  ┌──── train once ────┐  
SMILES → RDKit →  → mol_struct_ae → sim_embed 
                                       ▲
                                       │                                     
SMILES ───────────→ distillation → embedding
                                          ▲
                                          └─── trained to recapitulate sim_embed for a given smiles via cosine-distillation loss                                            
```

---

## Architectures

### Molecular Autoencoder (mol_struct_ae)

The encoder is modelled after AlphaFold-3's Pairformer architecture. Two
unified representations: *single_repr* (per-atom) and *pair_repr* (per-bond) are
updated jointly at every block. 

```
  ┌────────────────────────── INPUT TRACKS ───────────────────────────┐
  │  atom_feats   atom_type   partial_charge  pharmacokinetic_props   │
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
                  AtomQueryDecoder (z + 5 context tokens → decoded_molecule [B, N, d])
                              │
                  Molecule Structural Decoders
       ┌──────────────┬───────┴──────────┬──────────────┬─────────────┐
       ▼              ▼                  ▼              ▼             ▼
2D Feature        3D Feature           Charge   Pharmacokinetic    Dihedral/torsion 
    (atom feats,  (atype, bond     (per-atom q)   (per-atom)     (36-bin
     adj, bond)    len, bond ang)                                 classification of
                                                                  dihedral angle)
```

#### The Pairformer block

At every block, single and pair representations exchange information bidirectionally:

| Step | Op | Function |
|------|----|----------|
| ① | `OuterProductMean` | single_repr updates pair_repr: update pair (i,j) by the outer-product of low-dim projections of single_repr[i] and single_repr[j] |
| ② | `TriangleAttentionStartingNode` | pair_repr updates itself: for each (i,j), attend over a third atom k with bias from pair[k,j] — captures 3-atom geometry through pair_repr |
| ③ | `PairTransition` | pair_repr updates iself, as each pair_repr is fed into an MLP |
| ④ | `SingleAttentionWithPairBias` | pair_repr updates single_repr : self attention is first performed on each single_repr (atom). Then, each atom receives a bias obtained from all its pair_reprs, with bonded atoms receiving higher bias |
| ⑤ | `SingleTransition` | single_repr updates itself, once again through an MLP | 

#### Featurization

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

#### Input Features

Every input from the featurization is read in as a `MolSample`, batched into `MolBatch`, and then mapped into either `single_repr` (per atom) or `pair_repr` (per atom pair):

N = atom # of small molecule, normalized to max_atoms in this workflow through padding / truncation
B = batch size
T = # of dihedrals, normalized to max_dihedrals

| Input | Goes into | Dimensionality |
|-------|-----------|-----|
| `atom_feats_2d` | single_repr | (B,N,32) |
| `atom_types`  | single_repr | (B,N,16) |
| `partial_charges` | single_repr | (B,N) |
| `pharma_feats` | single_repr | (B,N,8) |
| `adj_2d` | pair (1-bond) | 2D embedding: (B,N,N) |
| `bond_feats_2d` | pair_repr | (B,N,N,8) |
| `bond_lengths`  | pair_repr | (B,N,N) |
| `bond_angles` (radians) | pair_repr | (B,N,N,N) |
| `dihedral_angles` (signed φ) | pair_repr | (B,T,4)|

All five complementary views of the molecule (topology, 3D geometry, partial charges, pharmacophore labels, dihedrals) are summed into these two tensors, then processed jointly by the Pairformer backbone.

**SE(3) invariance.** All Pairformer inputs are either topology-based (2D graph, pharmacophore labels) or SE(3)-invariant scalars (bond lengths, bond angles, dihedral angles, partial charges). No raw xyz coordinates enter the model. Two conformers of the same molecule that differ only by a global rotation or translation produce identical latent vectors.


#### Latent and similarity head

After the Pairformer outputs an atomic representation of the molecule `single_repr`, `GlobalAggregator` contains a MultiHeadAttention (n=6) layer that does two things:
   1. Transforms `single_repr [1,max_atoms,128]` into a `[1,256]` latent vector `z` by attending to each atom in `single_repr` . 
   2. Generates 5 other context tokens that produce other types of summaries of the molecule (charge, polarity, 2D graph, 3D structure, torsional features)

- **Pooling of `single_repr`.** Its attention output is fed into a `μ` and `log σ²` Linear() layer, then reparameterized into the latent `z ∈ ℝ^256`. KL is in the loss at a small weight (default 2e-3) so `z` is mostly deterministic; the VAE regularizer
  is for shape and spread, not generation. 

#### Decoders
`AtomQueryDecoder` reads in `z` and the 5 other context tokens, and returns two vectors: `[B,max_atoms,128]` (the molecular vector) and `[B,max_atoms]` (a per atom existence logit) through cross & self attention. The cross attention is done through `z` attending to the context tokens. 

Each decoder then reads in the molecular vector and is trained to match its reconstruction target. 
`Graph2DDecoder` attempts to reconstruct adjacency and bond information, `Geometry3DDecoder` attempts to reconstruct atom type, bond length, and bond angle information. ChargeDecoder attempts to reconstruct charge values per atom. `PharmaDecoder` attempts to reconstruct pharmacokinetic properties of the molecule. 

`TorsionDecoder` is unique from the others. It receives the molecular vector, the pair_representation vector from the Pairformer, 4-atom indices of all dihedrals, and  attempts to reconstruct dihedral angles in the molecule. The output is a 36-bin
classification logit vector over [−π, π] (10°-wide buckets). Bins were used because conformers are somewhat stochastic and dihedral angles are always slightly variant for the same molecule.

### Distillation model

Defined in [distillation/](distillation/). Architecturally simple:

```
SMILES string
   │
   ▼  smiles tokenizer (~65 tokens for each atom type + bond_type + brackets)
   │
   ▼  Embedding + learned positional encoding
   │
   ▼  TransformerEncoder × N layers (8 multiattention heads)
   │
   ▼  LayerNorm
   │
   ▼  fetch CLS token hidden state
   │
   ▼  MLP head → L2-normalize
   │
   ▼  256-D embedding (same dim as mol_struct_ae's sim_embed)
```

Trained with cosine distillation loss `1 - cos(distill_embed, sim_embed)` against pre-computed
`mol_struct_ae` outputs (see [Training](#training)).

**Why distill instead of using `mol_struct_ae` directly at inference?**
`mol_struct_ae` requires ~100 ms of RDKit work per molecule (ETKDG embedding,
MMFF optimization, Gasteiger charges). The distillation model inputs raw SMILES
and skips all of it — ~1 ms per molecule on GPU, suitable for screening 10⁶+
libraries.


## Training

Two stages, run sequentially. The full pipeline runs on [Modal](https://modal.com).

### Stage 1 — `mol_struct_ae` (Pairformer autoencoder)

Four detached Modal jobs run sequentially, all existing in [modal_mol_struct_ae.py](modal_mol_struct_ae.py):

```bash
# 1/4 — fetch DeepChem's pre-curated ZINC15 10M lead-like SMILES → CSV (~3 min)
modal run modal_mol_struct_ae.py::zinc15_fetch --size 10M

# 2/4 — Bemis-Murcko scaffold-balanced 5M subsample (~5 min)
#        diversify_smiles.py caps per-scaffold counts so the training set isn't
#        dominated by a few over-represented frameworks.
modal run modal_mol_struct_ae.py::zinc15_diversify --target-count 5000000

# 3/4 — featurize the 5M CSV → MolSample .pt file (~12-15h on cpu=32)
#        max_atoms=96 at featurize time; molecules over the cap are dropped.
modal run modal_mol_struct_ae.py::zinc15_precompute --max-atoms 96 --workers 32

# 4/4 — train the Pairformer autoencoder
modal run --detach modal_mol_struct_ae.py::zinc15_train \
    --batch-size 24 --max-atoms 96 \
    --hidden 128 --latent 256 --lr 3e-4
```

Training can be resumed via `--resume <ckpt-on-volume>`. The
checkpoint used for benchmarking is under `runs/zinc15_5m_diverse_run_pairformer_v3_kl2e3/ckpt_step102000.pt` (`max_atoms=96`,
`hidden=128`, `latent=256`, KL weight `2e-3`).


**Running locally** (on 250k drug-like molecules) — the same two scripts the Modal stages wrap:

```bash
python utils/featurize.py --csv data/zinc250k.csv --out data/shards \
                            --max-atoms 96 --shard-size 1024 --workers 8
python train_mol_struct_ae.py --shard-dir data/shards \
                                --out-dir runs/zinc250k \
                                --batch-size 24 --max-atoms 96 \
                                --hidden 128 --latent 256 --lr 3e-4 --amp
```

#### `mol_struct_ae` losses (stage 1)

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
| Torsion | masked cross-entropy over 36 bins of `[−π, π]` | 1.0 | classification, not regression — robust to conformer sampling noise and multimodal targets (gauche+/trans/gauche-) |
| KL (β-VAE) | `−½ Σ(1+logσ²−μ²−exp(logσ²))` | 2e-3 | small: keeps μ from collapsing onto a single drug-like manifold during long training |


### Stage 2 — `distillation` (SMILES → embedding)

A SMILES-only transformer is trained to reproduce `mol_struct_ae`
encoder's `sim_embed`, so inference needs no RDKit featurization. Three steps are
orchestrated by [modal_distillation.py](modal_distillation.py):

1. **Precompute targets** — `n_partitions` (default 4) to parallelize precompute four ways. 
2. **Merge** — a CPU container concatenates the partitions into one
   `mol_struct_ae_embeds.pt` pairs file (the default name; the shipped
   `runs/distill/best.pt` was trained against this file saved as
   `teacher_embeds.pt` — same contents, name set via `--pairs-out`).
3. **Train** — one A10G container trains the [`SmilesEncoder`](distillation/smiles_encoder.py)
   via cosine distillation. The SMILES vocab is built once from
   [build_vocab.py](scripts/build_vocab.py) if not already on the volume.

```bash
# Runs all three stages; point it at the AE checkpoint + shard dir on the volume.
# These are the artifacts the shipped runs/distill/best.pt was trained from.
modal run --detach modal_distillation.py::main_parallel \
    --ckpt-path /root/runs/zinc15_5m_diverse_run_pairformer_v3_kl2e3/ckpt_step102000.pt \
    --shard-dir /root/runs/shards_zinc15_5m_diverse_96 \
    --n-partitions 4 \
    --epochs 20 --batch-size 256 \
    --hidden 256 --num-layers 6 --lr 3e-4
```

The trained encoder is written to `runs/distill/` on the volume. `resume_merge_and_train` skips stage 1 if the
partition files already exist (use this when merge or training failed mid-run).

**Running locally**:

```bash
python scripts/build_vocab.py --csv data/zinc250k.csv --out runs/vocab.json

python scripts/precompute_mol_struct_ae_embeds.py \
    --checkpoint runs/zinc250k/best.pt \
    --shard-dir data/shards \
    --out runs/mol_struct_ae_embeds.pt --amp

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

---

## Inference

For datasets with >250k molecules, use **only the distillation model**. The distillation model takes SMILES strings directly:

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
Since the embeddings for both the molecular autoencoder and distillation model are both L2 normalized, similar embeddings (molecules) should have high cosine similarity. 

[scripts/simsearch.py](scripts/simsearch.py) embeds every SMILES in a CSV with the distillation model, then for each query SMILES print the top-K matches. Includes a Morgan-FP Tanimoto column side-by-side as a sanity baseline.

```bash
python scripts/simsearch.py \
    --distillation-ckpt runs/distill/best.pt \
    --vocab runs/vocab.json \
    --library data/zinc250k.csv \
    --queries "aspirin:CC(=O)Oc1ccccc1C(=O)O" \
              "caffeine:Cn1cnc2n(C)c(=O)n(C)c(=O)c12" \
    --top-k 10 --out runs/simsearch.json
```

### Embedding with `mol_struct_ae` directly

The Pairformer autoencoder can be invoked for inference too — useful for
embedding quality analysis, required for distillation training.

```python
import torch, torch.nn.functional as F
from mol_struct_ae import MolStructAutoencoder
from mol_struct_ae.data import collate
from mol_struct_ae.model import MolAEConfig
from utils.featurize import featurize_smiles

ckpt  = torch.load("runs/zinc15_5m_diverse_run_pairformer_v3_kl2e3/ckpt_step102000.pt", map_location="cpu", weights_only=False)
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

## Benchmarking

Both stages share a single benchmark harness (**MoleculeNet end-to-end
fine-tuning**, Hu et al. / Mole-BERT protocol), wired up as two Modal
entrypoints — one per backend — both backed by the same code in
`utils/benchmark_moleculenet.py`. Each backend implements a small protocol
(`prepare(smiles)` → opaque featurized blob; `build_finetune_model()` → a
wrapper that maps `(prepared, indices) → logits`), so adding a new encoder is
a ~20-line class:

  - **`AEBackend`** — runs RDKit featurization, end-to-end fine-tunes the
    Pairformer encoder + an MLP head (decoders excluded) on top of the latent
    μ vector.
  - **`DistillBackend`** — tokenizes SMILES with `SmilesTokenizer`, end-to-end
    fine-tunes the distillation transformer + an MLP head on the CLS embedding.
    ~50× faster end-to-end than the AE backend (no RDKit, no conformer gen).

Both backends are **leakage-filtered** against the canonical-SMILES set from
the training shards.

### Split & metric

All benchmarks use **scaffold split across every dataset** (`--split-override scaffold`):
Bemis-Murcko scaffold groups are assigned to train/val/test in a per-seed-randomized
order (`random_scaffold_split_3way`, equivalent to DeepChem's `RandomGroupSplitter`).
Different seeds produce different scaffold assignments, so positives and negatives
distribute naturally across splits — this avoids the degenerate AUROC that the
deterministic largest-first variant produces on imbalanced sets like ClinTox.

| Dataset | Split | Metric |
|---------|-------|--------|
| BBBP, BACE, HIV, Tox21, ToxCast, SIDER, ClinTox | scaffold 80/10/10 (random per seed) | ROC-AUC |
| MUV | scaffold 80/10/10 (random per seed) | **AUPRC** (extreme imbalance ~0.2% positives — AUROC uninformative) |
| ESOL, FreeSolv, Lipo | scaffold 80/10/10 | RMSE |

The `DATASET_PROTOCOL` table in `utils/benchmark_moleculenet.py` still records per-dataset
defaults (random split for Tox21/SIDER/ClinTox/MUV) — pass `--split-override scaffold`
at the CLI to override the whole run, as done for all reported numbers.

### Running the benchmark

```bash
# Stage 1 — AE-backbone fine-tune (slow: RDKit featurization dominates).
MODAL_PROFILE=<profile> python -m modal run --detach \
    modal_mol_struct_ae.py::moleculenet_finetune \
    --datasets BBBP,BACE,HIV,Tox21,ToxCast,SIDER,ClinTox,MUV \
    --seeds 0,1,2 --epochs 100 --split-override scaffold

# Stage 2 — distillation-backbone fine-tune (fast: no featurization, ~1 ms/mol).
MODAL_PROFILE=<profile> python -m modal run --detach \
    modal_distillation.py::moleculenet_finetune_distill \
    --distill-ckpt /root/runs/distill/best.pt \
    --datasets BBBP,BACE,HIV,Tox21,ToxCast,SIDER,ClinTox,MUV \
    --seeds 0,1,2 --epochs 100 --split-override scaffold
```

The AE backend featurizes once per dataset (RDKit ETKDG + MMFF) and reuses
the `MolSample`s across all seeds. The distillation backend tokenizes once
and reuses the token IDs. Both are self-contained in
`utils/benchmark_moleculenet.py` (dataset download + leakage-filter helpers
are inline).
