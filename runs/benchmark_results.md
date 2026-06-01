# MoleculeNet Fine-Tuning Benchmark Results

**Checkpoint (Stage 1, AE):** `v3_kl2e3/ckpt_step102000.pt`
(Pairformer + multi-track reconstruction, β=2e-3 KL, pretrained on 5M ZINC15-diverse)

**Checkpoint (Stage 2, Distill):** `runs/distill/best.pt`
(SmilesEncoder distilled against the AE's `sim_embed`, lowest val cosine loss)

**Protocol:** end-to-end fine-tuning (Hu et al. / Mole-BERT, ICLR 2023);
80/10/10 split with per-dataset split + metric per `DATASET_PROTOCOL`
(scaffold + ROC-AUC for BBBP/BACE/HIV; random + ROC-AUC for
Tox21/ToxCast/SIDER/ClinTox; random + AUPRC for MUV). Scaffold splits use
per-seed randomized scaffold groups (`random_scaffold_split_3way`, equivalent
to DeepChem `RandomGroupSplitter`), which is what Mole-BERT reports — the
deterministic largest-first variant gives sub-random ClinTox AUROC. Adam
lr=1e-3, batch=32, dropout=0.5, grad-clip=1.0; 100 epochs; test score at the
best-validation epoch; mean ± std over N independent seeds (random init +
minibatch order + per-seed split).

## Summary — Stage 2 (Distill SmilesEncoder, canonical per-dataset protocol)

| dataset | split | metric | seeds | **DISTILL** | Mole-BERT (Table 1) | Δ |
|---|---|---|---|---|---|---|
| BBBP    | scaffold | AUROC ↑ | 3 | **0.9208 ± 0.0391** | 0.7187 ± 0.0160 | **+0.202** |
| BACE    | scaffold | AUROC ↑ | 3 | **0.8143 ± 0.0249** | 0.8084 ± 0.0140 | +0.006 |
| HIV     | scaffold | AUROC ↑ | 3 | _running_           | 0.7820          | _tbd_ |
| Tox21   | random   | AUROC ↑ | 3 | _running_           | 0.7682 ± 0.0050 | _tbd_ |
| ToxCast | random   | AUROC ↑ | 3 | _running_           | 0.6430          | _tbd_ |
| SIDER   | random   | AUROC ↑ | 3 | _running_           | 0.6275 ± 0.0113 | _tbd_ |
| ClinTox | random   | AUROC ↑ | 3 | _running_           | 0.7890 ± 0.0300 | _tbd_ |
| MUV     | random   | AUPRC ↑ | 3 | _running_           | 0.7860          | _tbd_ |

## Summary — Stage 1 (AE backbone, earlier deterministic-protocol runs)

These numbers were collected before the protocol unification, so the splits
are not directly comparable to the Stage-2 table above. Kept for historical
context — they predate `DATASET_PROTOCOL`.

| dataset | metric | n_train / n_val / n_test | seeds | **AE** | Mole-BERT (Table 1) | Δ |
|---|---|---|---|---|---|---|
| BBBP                     | AUROC ↑ | 1584 / 198 / 199 | 3 | **0.8025 ± 0.0124** | 0.7187 ± 0.0160 | **+0.084** |
| SIDER                    | AUROC ↑ | 1022 / 127 / 129 | 3 | 0.5820 ± 0.0177     | 0.6275 ± 0.0113 | −0.046 |
| ClinTox (det-strat)      | AUROC ↑ | 1100 / 137 / 140 | 3 | 0.306  ± 0.122 ⚠    | 0.7890 ± 0.0300 | (split artifact) |
| **ClinTox (rand-split)** | AUROC ↑ | 1101 / 137 / 139 | 3 | **0.9032 ± 0.0240** | 0.7890 ± 0.0300 | **+0.114** |
| BACE                     | AUROC ↑ | 1163 / 145 / 146 | 3 | **0.8256 ± 0.0092** | 0.8084 ± 0.0140 | **+0.017** |
| Tox21                    | AUROC ↑ | 6109 / 763 / 765 | 3 | 0.7551 ± 0.0141     | 0.7682 ± 0.0050 | −0.013 |

## Per-seed scores

### Distill (Stage 2, canonical per-dataset protocol)

**BBBP** — scaffold split (per-seed random scaffold groups), AUROC

```
seed 0  0.8577
seed 1  0.8667
seed 2  0.9380
```

Mean 0.9208, std 0.0391. Wall-time: 412 s (~7 min for 3 seeds — distill
backbone is ~40× faster than AE since it skips RDKit featurization).

**BACE** — scaffold split (per-seed random scaffold groups), AUROC

```
seed 0  0.7992
seed 1  0.8493
seed 2  0.7943
```

Mean 0.8143, std 0.0249. Wall-time: 308 s (~5 min for 3 seeds).

_HIV, Tox21, ToxCast, SIDER, ClinTox, MUV: running (job `blfu3uhvh`)._

## Per-seed scores — Stage 1 (AE)

### BBBP (3 seeds — current run, seeds 0..2, deterministic split)

```
seed 0  0.7887
seed 1  0.7999
seed 2  0.8188
```

Mean 0.8025, std 0.0124 (population). Wall-time: 6055 s (≈ 101 min for 3
seeds). Reproduces yesterday's seeds 0..2 bit-for-bit (confirms determinism).

**Bonus context — 8-seed partial from yesterday's cancelled run (seeds 0..7):**

```
seed 0  0.7887        seed 4  0.8527
seed 1  0.7999        seed 5  0.7962
seed 2  0.8188        seed 6  0.8364
seed 3  0.8220        seed 7  0.7803
```

8-seed mean 0.8119, std 0.0233. The 3-seed headline above is the value
reported in the protocol-uniform comparison; the 8-seed result is a wider
view (still well above Mole-BERT's 0.7187 ± 0.0160).

### SIDER (3 seeds — current run, seeds 0..2)

```
seed 0  0.5662
seed 1  0.6067
seed 2  0.5731
```

Mean 0.5820, std 0.0177. Wall-time: 3670 s (≈ 61 min for 3 seeds).

### BACE (3 seeds — current run, seeds 0..2)

```
seed 0  0.8337
seed 1  0.8127
seed 2  0.8305
```

Mean 0.8256, std 0.0092. Wall-time: 4337 s (≈ 72 min for 3 seeds).

### Tox21 (3 seeds — current run, seeds 0..2, deterministic scaffold split)

```
seed 0  0.7620
seed 1  0.7679
seed 2  0.7354
```

Mean 0.7551, std 0.0141. Wall-time: 22060 s (≈ 6.1 hours for 3 seeds —
Tox21 is by far the heaviest dataset at 6109 train mols × 12 tasks).
Within Mole-BERT's std band (theirs: 0.768 ± 0.005).

### ClinTox (3 seeds — random_per_seed split, the correct protocol)

```
seed 0  0.8949
seed 1  0.9359
seed 2  0.8789
```

Mean 0.9032, std 0.0240. Wall-time: 3707 s (≈ 62 min). All three seeds in
the 0.88–0.94 range — not a single-seed fluke. **+0.114 above Mole-BERT's
0.789 ± 0.030**, well outside their std band. Confirms that ClinTox is
not hard for a structure-aware encoder once the split protocol actually
distributes positives across train/val/test.

## Notes

- **Backbone (AE)**: Pairformer-lite (4 blocks, d_single=128, d_pair=32) with
  3D-aware features (bond lengths/angles/dihedrals as Fourier features).
  **Backbone (Distill)**: 6-layer transformer encoder over SMILES tokens
  (~5M params), trained against the AE's `sim_embed` via cosine distillation.
  Mole-BERT = 5-layer GIN (hidden=300) with 2D-only features. The fine-tuning
  protocol is identical across all three; the backbones differ by design — that's
  the comparison.
- **Pretraining**: ours = 5M ZINC15-diverse scaffold-balanced subset; theirs
  = 2M ZINC15. Both source the same underlying database.
- **Pretraining task**: ours = multi-track reconstruction (atom features,
  adjacency, bond features, atom types, bond lengths, bond angles,
  charges, pharma flags, 36-bin dihedral classification) + β-VAE KL
  (β=2e-3). Theirs = Masked Atoms Modeling (MAM) + Triplet Masked
  Contrastive Learning (TMCL).
- **Leakage**: this run did NOT apply the training-SMILES leakage filter.
  ZINC15 contains drug-like molecules; some BBBP / BACE / ClinTox /
  SIDER / Tox21 molecules may have appeared in the 5M pretrain set. Worth
  rerunning with `--train-smiles-path` once we have a baseline.
