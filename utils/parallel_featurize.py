"""
Parallel featurization helper.

featurize_smiles is CPU-bound (RDKit ETKDG + MMFF) and serial. A 40K-molecule
benchmark takes ~40 min single-threaded. multiprocessing.Pool over a fixed
worker count yields ~N× speedup, dominating any GIL/torch overhead.

Usage:
    samples, keep_idx = featurize_smiles_parallel(smis, max_atoms=64, workers=8)
"""
from __future__ import annotations

from functools import partial
from multiprocessing import Pool
from typing import List, Optional, Tuple

from utils.featurize import featurize_smiles


def _worker(smi_with_idx, max_atoms: int):
    i, smi = smi_with_idx
    s = featurize_smiles(smi, max_atoms=max_atoms)
    return i, s


def featurize_smiles_parallel(smiles: List[str], max_atoms: int = 64,
                               workers: int = 8, chunksize: int = 16,
                               verbose: bool = True):
    """Featurize a list of SMILES in parallel.

    Returns (samples, keep_idx) where samples preserves the input order
    (with None entries filtered) and keep_idx[j] = original index of samples[j].
    """
    if workers <= 1:
        samples = []
        keep = []
        for i, smi in enumerate(smiles):
            s = featurize_smiles(smi, max_atoms=max_atoms)
            if s is not None:
                samples.append(s); keep.append(i)
        return samples, keep

    indexed = list(enumerate(smiles))
    fn = partial(_worker, max_atoms=max_atoms)
    results: List[Optional[Tuple[int, object]]] = [None] * len(smiles)
    n_done = 0
    import time
    t0 = time.time()
    with Pool(workers) as pool:
        for i, s in pool.imap_unordered(fn, indexed, chunksize=chunksize):
            results[i] = s
            n_done += 1
            if verbose and n_done % 2000 == 0:
                rate = n_done / max(time.time() - t0, 1e-6)
                eta = (len(smiles) - n_done) / max(rate, 1e-6) / 60
                print(f"  [featurize] {n_done}/{len(smiles)}  ({rate:.1f}/s, ETA {eta:.1f}m)",
                       flush=True)

    samples = []
    keep_idx = []
    for i, s in enumerate(results):
        if s is not None:
            samples.append(s); keep_idx.append(i)
    if verbose:
        print(f"  [featurize] done in {(time.time()-t0)/60:.1f}m — "
               f"kept {len(samples)}/{len(smiles)}")
    return samples, keep_idx
