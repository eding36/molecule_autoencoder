"""
Reusable helpers extracted for any benchmark or similarity-search script:

  * `_featurize_one`         — Pool-worker version of featurize_smiles that
    returns numpy/list values (bypasses the torch.multiprocessing fd-sharing
    bug observed at scale).
  * `_dict_to_sample`        — main-process reconstructor: dict → MolSample.
  * `featurize_all`          — parallel featurization with one-line progress
    logging.
  * `embed_all`              — batched model encoding from in-memory samples.
    Returns dict[SMILES → np] of L2-normalized `sim_embed` vectors.
  * `embed_from_shards`      — same output, but reads pre-featurized shards
    instead of featurizing. ~25× faster for large libraries (no RDKit step).
    Only works if shards were written with the `smiles` field populated.
  * `morgan_fps`             — RDKit Morgan-FP bitvector per SMILES.
  * `tanimoto_matrix`        — q_fp vs ref_fps matrix Tanimoto.

Nothing in this module references DUD/MUV/ChEMBL — everything here works on any
SMILES list.
"""
from __future__ import annotations

import sys
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.multiprocessing as torch_mp
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mol_struct_ae import MolStructAutoencoder
from mol_struct_ae.data import MolSample, collate
from utils.featurize import featurize_smiles

RDLogger.DisableLog("rdApp.*")

try:
    torch_mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass


FP_RADIUS = 2
FP_NBITS = 2048

#Helper func that turns pytorch tensors into a dict of numpy arrays, required for multiprocessing to work
def _featurize_one(smi: str, max_atoms: int):
    sample = featurize_smiles(smi, max_atoms=max_atoms)
    if sample is None:
        return smi, None
    d = {
        "atom_feats_2d": sample.atom_feats_2d.numpy(),
        "bond_feats_2d": sample.bond_feats_2d.numpy(),
        "edge_index_2d": sample.edge_index_2d.numpy(),
        "atom_types": sample.atom_types.numpy(),
        "bond_lengths": sample.bond_lengths.numpy(),
        "angle_index": sample.angle_index.numpy(),
        "bond_angles": sample.bond_angles.numpy(),
        "partial_charges": sample.partial_charges.numpy(),
        "pharma_feats": sample.pharma_feats.numpy(),
        "dihedral_index": sample.dihedral_index.numpy(),
        "dihedral_angles": sample.dihedral_angles.numpy(),
    }
    return smi, d

#Converts numpy dict to molsample (with torch tensors)
def _dict_to_sample(d) -> MolSample:
    return MolSample(
        atom_feats_2d=torch.from_numpy(d["atom_feats_2d"]),
        bond_feats_2d=torch.from_numpy(d["bond_feats_2d"]),
        edge_index_2d=torch.from_numpy(d["edge_index_2d"]).long(),
        atom_types=torch.from_numpy(d["atom_types"]),
        bond_lengths=torch.from_numpy(d["bond_lengths"]),
        angle_index=torch.from_numpy(d["angle_index"]).long(),
        bond_angles=torch.from_numpy(d["bond_angles"]),
        partial_charges=torch.from_numpy(d["partial_charges"]),
        pharma_feats=torch.from_numpy(d["pharma_feats"]),
        dihedral_index=torch.from_numpy(d["dihedral_index"]).long(),
        dihedral_angles=torch.from_numpy(d["dihedral_angles"]),
    )


def featurize_all(smiles_list: List[str], max_atoms: int,
                    workers: int = 8) -> Dict[str, MolSample]:
    fn = partial(_featurize_one, max_atoms=max_atoms)
    t0 = time.time()
    out: Dict[str, MolSample] = {}
    if workers <= 1:
        for i, smi in enumerate(smiles_list):
            _, d = fn(smi)
            if d is not None:
                out[smi] = _dict_to_sample(d)
            if (i + 1) % 1000 == 0:
                rate = (i + 1) / max(time.time() - t0, 1e-6)
                print(f"[featurize] {i+1}/{len(smiles_list)} rate={rate:.1f}/s", flush=True)
    else:
        with Pool(workers) as pool:
            for i, (smi, d) in enumerate(pool.imap_unordered(fn, smiles_list, chunksize=8)):
                if d is not None:
                    out[smi] = _dict_to_sample(d)
                if (i + 1) % 1000 == 0:
                    rate = (i + 1) / max(time.time() - t0, 1e-6)
                    print(f"[featurize] {i+1}/{len(smiles_list)} kept={len(out)} rate={rate:.1f}/s",
                           flush=True)
    print(f"[featurize] DONE — {len(out)}/{len(smiles_list)} kept in {time.time()-t0:.0f}s")
    return out


# ─── Autoencoder Pairformer embedding (batched, but loads all featurized smiles at once, memory intensive, only use for <250K smiles) ───────────────────────────────────────────────
@torch.no_grad()
def embed_all(samples_by_smiles: Dict[str, MolSample], model: MolStructAutoencoder,
              device: torch.device, batch_size: int, max_atoms: int) -> Dict[str, np.ndarray]:
    model.eval()
    smis = list(samples_by_smiles.keys())
    embs: List[torch.Tensor] = []
    t0 = time.time()
    for i in range(0, len(smis), batch_size):
        chunk = smis[i:i + batch_size]
        batch = collate([samples_by_smiles[s] for s in chunk], max_atoms=max_atoms).to(device)
        out = model(batch, sample=False)
        e = F.normalize(out["sim_embed"], dim=-1).cpu()
        embs.append(e)
        if (i // batch_size + 1) % 20 == 0:
            rate = (i + len(chunk)) / max(time.time() - t0, 1e-6)
            print(f"[embed] {i+len(chunk)}/{len(smis)} rate={rate:.1f}/s", flush=True)
    E = torch.cat(embs, dim=0).numpy()
    return {smi: E[i] for i, smi in enumerate(smis)}


# ─── Shard-based embedding (streams batches of shards, memory efficient, use for >250k smiles) ──────────────────────────
@torch.no_grad()
def embed_from_shards(shard_dir: str, model: MolStructAutoencoder,
                       device: torch.device, batch_size: int, max_atoms: int,
                       pattern: str = "shard_*.pt",
                       shard_start: int = 0, shard_stride: int = 1,
                       amp: bool = False) -> Dict[str, np.ndarray]:
    """Load each shard, embed in batches.

    Args:
      shard_start, shard_stride: process `shard_paths[shard_start::shard_stride]`.
      N parallel workers, with each worker passing
        `shard_start=i, shard_stride=N` for i in 0..N-1; together they cover
        every shard exactly once.
      amp: if True, run the forward pass under `torch.autocast(..., bfloat16)`.
        ~1.7× speedup on A10G for inference with no accuracy hit. Safe because
        the AE was trained with AMP and weights are happy in bf16.
    """
    import glob
    import os

    model.eval()
    all_shards = sorted(glob.glob(os.path.join(shard_dir, pattern)))
    if not all_shards:
        raise FileNotFoundError(f"no shards matching {pattern} in {shard_dir}")
    shard_paths = all_shards[shard_start::shard_stride]
    print(f"[embed_from_shards] processing {len(shard_paths)} / {len(all_shards)} "
           f"shards (start={shard_start}, stride={shard_stride}); amp={amp}", flush=True)

    autocast_ctx = (torch.autocast(device.type, dtype=torch.bfloat16) if amp
                     else _NullCtx())

    out: Dict[str, np.ndarray] = {}
    skipped_no_smiles = 0
    skipped_oversize = 0
    total_seen = 0
    t0 = time.time()
    for sp in shard_paths:
        try:
            shard = torch.load(sp, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[embed_from_shards] skip unreadable {sp}: {type(e).__name__}", flush=True)
            continue
        valid = []
        for s in shard:
            total_seen += 1
            if not getattr(s, "smiles", ""):
                skipped_no_smiles += 1
                continue
            if s.atom_feats_2d.shape[0] > max_atoms:
                skipped_oversize += 1
                continue
            valid.append(s)
        for i in range(0, len(valid), batch_size):
            chunk = valid[i:i + batch_size]
            batch = collate(chunk, max_atoms=max_atoms).to(device)
            with autocast_ctx:
                o = model(batch, sample=False)
            e = F.normalize(o["sim_embed"].float(), dim=-1).cpu().numpy()
            for s, vec in zip(chunk, e):
                out[s.smiles] = vec
        rate = total_seen / max(time.time() - t0, 1e-6)
        print(f"[embed_from_shards] {os.path.basename(sp)} seen={total_seen} "
               f"kept={len(out)} rate={rate:.1f}/s", flush=True)
    if skipped_no_smiles:
        print(f"[embed_from_shards] WARN skipped {skipped_no_smiles} samples lacking "
               f".smiles — re-run utils/featurize.py to regenerate with SMILES tags")
    if skipped_oversize:
        print(f"[embed_from_shards] skipped {skipped_oversize} samples > {max_atoms} atoms")
    return out


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


# ─── Morgan fingerprints (Tanimoto baseline) ─────────────────────────────────
def morgan_fps(smiles_list: List[str]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_NBITS)
        out[smi] = (np.frombuffer(bv.ToBitString().encode(), dtype="u1") - ord("0")).astype(np.uint8)
    return out


def tanimoto_matrix(q_fp: np.ndarray, ref_fps: np.ndarray) -> np.ndarray:
    inter = ref_fps @ q_fp
    union = ref_fps.sum(axis=1) + q_fp.sum() - inter
    return inter / np.clip(union, 1e-8, None)
