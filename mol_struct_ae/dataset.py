"""
Sharded ZINC dataset: each shard is a `.pt` file holding a list[MolSample].

Shards are loaded lazily into memory by the DataLoader workers, then individual
`MolSample`s are yielded. Dense collation to `MolBatch` happens in `collate_fn`.
This keeps disk small (sparse on disk) while batches stay model-ready.

Key performance feature: pair the dataset with `ShardSequentialSampler` to walk
samples shard-by-shard. With many small shards on a network volume, random
access would force each batch to load 64 separate shards (~100× slowdown).
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Iterator, List, Optional

import torch
from torch.utils.data import Dataset, Sampler

from .data import MolSample, collate


def _index_cache_path(shard_dir: str, max_atoms: Optional[int]) -> str:
    key = "any" if max_atoms is None else str(max_atoms)
    return os.path.join(shard_dir, f".index.maxatoms_{key}.json")


class ShardedMolDataset(Dataset):
    """Indexes all MolSamples across all shards for easy sampling.

    On first construction, builds an index `(shard_idx, local_idx)` for every
    sample passing the `max_atoms` filter, then caches it to a JSON sidecar so
    subsequent runs skip the multi-minute torch.load scan.

    Each worker keeps a single-shard cache. Use this dataset with
    `ShardSequentialSampler` so consecutive __getitem__ calls hit the same shard.
    """

    def __init__(self, shard_dir: str, pattern: str = "shard_*.pt",
                  max_atoms: Optional[int] = None, use_cache: bool = True):
        all_paths = sorted(glob.glob(os.path.join(shard_dir, pattern))) #fetch all shards
        if not all_paths:
            raise FileNotFoundError(f"No shards matching {pattern} in {shard_dir}")

        cache_path = _index_cache_path(shard_dir, max_atoms)
        loaded_from_cache = False
        if use_cache and os.path.isfile(cache_path): #try loading from cache using json file of all the cached shards
            try:
                with open(cache_path) as f:
                    payload = json.load(f)
                cached_paths = payload["shard_paths"]
                cached_counts = payload["valid_counts"]   # per-shard valid sample count
                if cached_paths == all_paths and len(cached_paths) == len(cached_counts):
                    self.shard_paths = cached_paths
                    self.index = [(s, l) for s, n in enumerate(cached_counts) for l in range(n)]
                    print(f"[ShardedMolDataset] loaded cached index "
                           f"({len(self.shard_paths)} shards, {len(self.index)} samples) "
                           f"from {os.path.basename(cache_path)}")
                    loaded_from_cache = True
            except Exception as e:
                print(f"[ShardedMolDataset] index cache present but unreadable "
                       f"({type(e).__name__}: {e}); rebuilding")

        if not loaded_from_cache:
            self.shard_paths: List[str] = []
            self.index: List[tuple] = []
            valid_counts: List[int] = []
            skipped_shards: List[str] = []
            oversize_dropped = 0
            for i, path in enumerate(all_paths):
                try:
                    data = torch.load(path, map_location="cpu", weights_only=False)
                except Exception as e:
                    skipped_shards.append(f"{os.path.basename(path)} ({type(e).__name__}: {e})") #append shard path to skipped shards if error
                    continue
                shard_pos = len(self.shard_paths) #shard index in all paths
                self.shard_paths.append(path)
                kept_in_shard = 0
                for local_idx, sample in enumerate(data):
                    if max_atoms is not None and sample.atom_feats_2d.shape[0] > max_atoms: #count how many shards are dropped due to max_atom violation
                        oversize_dropped += 1
                        continue
                    self.index.append((shard_pos, local_idx)) #append which shard and which molecule index to the total molecule indices
                    kept_in_shard += 1
                valid_counts.append(kept_in_shard)
                ###---Bookkeeping####
                if (i + 1) % 200 == 0: 
                    print(f"[ShardedMolDataset] indexed {i+1}/{len(all_paths)} shards", flush=True)
            if skipped_shards:
                print(f"[ShardedMolDataset] WARN skipped {len(skipped_shards)} unreadable shard(s):")
                for s in skipped_shards:
                    print(f"  - {s}")
            if max_atoms is not None and oversize_dropped:
                print(f"[ShardedMolDataset] dropped {oversize_dropped} mols with >{max_atoms} atoms")
            if not self.index:
                raise RuntimeError(f"No valid samples in {shard_dir}")

            if use_cache:
                try:
                    with open(cache_path, "w") as f:
                        json.dump({"shard_paths": self.shard_paths,
                                    "valid_counts": valid_counts}, f)
                    print(f"[ShardedMolDataset] wrote index cache → {os.path.basename(cache_path)}")
                except Exception as e:
                    print(f"[ShardedMolDataset] WARN could not write index cache: {e}")

        self.total = len(self.index) #total shards
        self._cache_path: Optional[str] = None
        self._cache_data: Optional[List[MolSample]] = None

    def __len__(self):
        return self.total

    def _load_shard(self, shard_idx: int):
        path = self.shard_paths[shard_idx]
        if self._cache_path != path:
            self._cache_data = torch.load(path, map_location="cpu", weights_only=False)
            self._cache_path = path
        return self._cache_data

    def __getitem__(self, idx: int) -> MolSample: #fetches a single MolSample at a specified idx
        shard_idx, local_idx = self.index[idx]
        return self._load_shard(shard_idx)[local_idx]


class ShardSequentialSampler(Sampler[int]):
    """Returns randomized dataset indices grouped by shard. Shards are accessed in a shuffled order, and samples
    inside each shard are shuffled too.
    """

    def __init__(self, dataset: ShardedMolDataset, shuffle: bool = True, seed: int = 0):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        # Group dataset indices by shard once at construction
        by_shard: dict[int, List[int]] = {} #keys = shard indices, values = dataset indices for each shard
        for ds_idx, (shard_idx, _) in enumerate(dataset.index):
            by_shard.setdefault(shard_idx, []).append(ds_idx) #for each molsample, populate a dictionary in which keys are the shard index of the molsample, and value is the molsample index itself
        self.shard_to_indices = by_shard

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch) #initialize random seed
        shard_keys = list(self.shard_to_indices.keys()) #list of all molsample indices, ordered by shard index
        if self.shuffle: #shuffle shard indices
            shard_keys = [shard_keys[i] for i in torch.randperm(len(shard_keys), generator=g).tolist()]
        for shard_idx in shard_keys:
            indices = self.shard_to_indices[shard_idx] #fetch molsample indices for specific shard index
            if self.shuffle: #now shuffle molsample indices within this shard index
                perm = torch.randperm(len(indices), generator=g).tolist()
                for p in perm:
                    yield indices[p] #shuffled indices
            else:
                yield from indices #unshuffled MolSamples AND shards

    def __len__(self) -> int:
        return self.dataset.total


def make_collate_fn(max_atoms: int, max_dihedrals: int): #padding op, sparse --> dense array
    def collate_fn(samples: List[MolSample]):
        return collate(samples, max_atoms=max_atoms, max_dihedrals=max_dihedrals)
    return collate_fn
