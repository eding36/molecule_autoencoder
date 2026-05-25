"""
Download DeepChem's pre-curated ZINC15 lead-like SMILES subset and emit a
single-column CSV ready for `utils/featurize.py`.

DeepChem hosts the 250K/1M/10M lead-like ZINC15 subsets as gzipped tarballs.
We pull the tarball, extract the inner CSV, and write a clean `smiles`-only
CSV. No tranche-stitching, no canonical-SMILES dedup pass needed (DeepChem
already canonicalized the upstream subset).

Available sizes:
  * 250K  ≈  ~25 MB tarball
  * 1M    ≈ ~110 MB tarball
  * 10M   ≈ ~280 MB tarball
  * 270M  ≈ ~23 GB tarball (requires interactive confirmation upstream — we
                            don't support it here; not needed for pretraining)

Reference: https://github.com/deepchem/deepchem/blob/master/deepchem/molnet/
            load_function/zinc15_datasets.py

CLI:
    python scripts/fetch_zinc15.py \\
        --size 10M --out runs/zinc15_10m.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import tarfile
import time
import urllib.request
from pathlib import Path


DEEPCHEM_BASE = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; mol_struct_ae fetch_zinc15.py)",
    "Accept": "*/*",
}

VALID_SIZES = {"250K", "1M", "10M"}


def download_to(url: str, dst: str) -> None:
    """Streamed download with progress logging."""
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", "0"))
        chunk_size = 4 * 2**20                  # 4 MB chunks
        done = 0
        t0 = time.time()
        with open(dst, "wb") as out:
            while True:
                buf = resp.read(chunk_size)
                if not buf:
                    break
                out.write(buf)
                done += len(buf)
                if total > 0:
                    pct = 100 * done / total
                    mbps = (done / 2**20) / max(time.time() - t0, 1e-6)
                    print(f"[download] {done/2**20:>6.1f} / {total/2**20:>6.1f} MB "
                           f"({pct:>5.1f}%)  {mbps:.1f} MB/s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", choices=sorted(VALID_SIZES), default="10M")
    p.add_argument("--out", required=True, help="output CSV (single `smiles` column)")
    p.add_argument("--smiles-col", default="smiles",
                   help="name of the SMILES column to read from DeepChem's CSV")
    p.add_argument("--limit", type=int, default=0,
                   help="cap rows for smoke tests (0 = no cap)")
    p.add_argument("--keep-tarball", action="store_true",
                   help="don't delete the downloaded .tar.gz after extraction")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    name = f"zinc15_{args.size}_2D"
    tarball_url = f"{DEEPCHEM_BASE}{name}.tar.gz"
    tarball_local = out_path.parent / f"{name}.tar.gz"

    # ---- Download (skip if present)
    if tarball_local.is_file():
        print(f"[download] tarball already present at {tarball_local}; skipping.",
               flush=True)
    else:
        print(f"[download] {tarball_url} → {tarball_local}", flush=True)
        download_to(tarball_url, str(tarball_local))

    # ---- Extract + emit single-column SMILES CSV
    print(f"[extract] opening {tarball_local}", flush=True)
    t0 = time.time()
    rows_written = 0
    with tarfile.open(tarball_local, "r:gz") as tf:
        # Pick the first CSV inside the tarball — DeepChem ships one csv per size.
        members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".csv")]
        if not members:
            raise RuntimeError(f"No CSV inside {tarball_local}")
        inner = members[0]
        print(f"[extract] inner csv: {inner.name} ({inner.size/2**20:.0f} MB)", flush=True)

        with tf.extractfile(inner) as f, open(out_path, "w", newline="") as out:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
            if args.smiles_col not in reader.fieldnames:
                raise RuntimeError(
                    f"Column '{args.smiles_col}' not in CSV; available: {reader.fieldnames}"
                )
            writer = csv.writer(out)
            writer.writerow(["smiles"])
            for row in reader:
                smi = (row.get(args.smiles_col) or "").strip()
                if smi:
                    writer.writerow([smi])
                    rows_written += 1
                    if args.limit and rows_written >= args.limit:
                        break
                    if rows_written % 500_000 == 0:
                        rate = rows_written / max(time.time() - t0, 1e-6)
                        print(f"[extract] wrote {rows_written:>10,} rows  "
                               f"rate={rate:.0f}/s", flush=True)

    if not args.keep_tarball:
        os.remove(tarball_local)
        print(f"[cleanup] removed {tarball_local}")
    out_size_mb = out_path.stat().st_size / 2**20
    print(f"[done] {rows_written:,} SMILES → {out_path} ({out_size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
