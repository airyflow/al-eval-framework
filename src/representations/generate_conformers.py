#!/usr/bin/env python3
"""Stage 1 of the split Uni-Mol pipeline: conformer generation only.

Previously, `extract_unimol()` in extract.py did conformer generation
(RDKit ETKDG, CPU-bound, ~2-3h for 50K molecules observed directly on a
DOCKSTRING pool) and embedding computation (Uni-Mol transformer forward
pass, GPU-accelerable) as one continuous, single-process function. That
doesn't scale to a billion-molecule pool: conformer generation alone
would take years on one process, and it's the wrong resource shape for
one big GPU job anyway -- it's embarrassingly parallel per-molecule CPU
work, not something a GPU helps with.

This script runs *only* the conformer-generation step, over one chunk of
a (possibly huge) SMILES pool, and writes the result to an LMDB shard.
It has no GPU/model dependency at all, so it can be launched as a SLURM
array job across many CPU-only nodes in parallel (see
scripts/submit_conformer_gen.sh) -- e.g. splitting 1.3B molecules into
1000 chunks of ~1.3M molecules each, one per array task.

Once every chunk is done, run this script again with --merge to
consolidate all shards into the single `{partition}.lmdb` file that
DatasetUniMol.create_features() (muben/muben/dataset/dataset_unimol/dataset.py)
already knows how to load -- so extract_unimol() in extract.py picks up
the pre-generated conformers with *zero* changes to its own code, as
long as the merged file lands at the exact path its MubenRuntimeConfig
expects: `{out_dir}/_unimol_cache_{pool_hash}/train.lmdb`.

Storage note: measured directly (5000 real Enamine REAL molecules), a
single conformer record is ~9.15KB pickled at the muben-default
--n-conformer 10 (10 3D conformers + 1 2D fallback), or ~4.01KB at
--n-conformer 1. The latter is recommended for frozen feature extraction
specifically: extract_unimol() always prepares with partition="train",
which discards all but one *randomly chosen* conformer before a single
forward pass (see muben/muben/dataset/dataset_unimol/process.py's
conformer_sampling()), so generating 10 and discarding 9 is equivalent
in expectation to generating 1 directly -- and ~9x faster, measured. At
1.56B molecules that is on the order of ~6.1TB of on-disk cache at
--n-conformer 1 (~13.9TB at the muben default of 10) --
budget storage accordingly before launching a billion-scale array job.

Usage
-----
# One chunk of a DOCKSTRING target's pool (small-scale, mostly useful for
# testing the split path end-to-end):
python -m src.representations.generate_conformers --target PARP1 \\
    --chunk-id 0 --num-chunks 10

# One chunk of an arbitrary billion-scale library (one SMILES per line):
python -m src.representations.generate_conformers --smiles-file library.smi \\
    --total-count 1300000000 --chunk-id 0 --num-chunks 1000

# After all chunks finish, merge shards into the final cache:
python -m src.representations.generate_conformers --target PARP1 --num-chunks 10 --merge
"""
from __future__ import annotations

import argparse
import itertools
import os
import pickle
import time
from functools import partial
from multiprocessing import get_context
from multiprocessing import TimeoutError as MPTimeoutError
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import lmdb

from src.representations.extract import ROOT, OUTPUT_DIR, compute_pool_hash, compute_pool_identity

_KEY_WIDTH = 13  # zero-padded decimal width; comfortably covers > 1.3B indices


def _global_index_key(idx: int) -> bytes:
    """LMDB iterates keys in byte-lexicographic order, and DatasetUniMol's
    non-random_split load path (create_features()) trusts that order to
    match the SMILES list positionally. Zero-padding guarantees correct
    numeric ordering regardless of index magnitude."""
    return f"{idx:0{_KEY_WIDTH}d}".encode()


def _chunk_bounds(n: int, chunk_id: int, num_chunks: int) -> tuple[int, int]:
    chunk_size = (n + num_chunks - 1) // num_chunks
    start = chunk_id * chunk_size
    end = min(start + chunk_size, n)
    return start, end


def count_lines(path: str) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def read_smiles_chunk(path: str, start: int, end: int) -> list:
    """Reads only lines [start, end) of a SMILES file. Uses islice over a
    lazily-iterated file handle rather than loading the whole file into a
    list first -- memory stays bounded to one chunk even for a billion-line
    pool (the O(start) time to skip preceding lines is accepted here; if
    that skip cost becomes the bottleneck, a precomputed byte-offset index
    would remove it)."""
    with open(path) as f:
        lines = list(itertools.islice(f, start, end))
    return [line.strip() for line in lines if line.strip()]


def generate_conformers_for_chunk(
    smiles: list, shard_path: Path, start: int, map_size_gb: float,
    n_conformer: int = 10, num_workers: int = 4, timeout_s: int = 30,
    commit_every: int = 2000,
) -> int:
    """Same timeout-protected generation as
    muben/muben/dataset/dataset_unimol/dataset.py's create_features() --
    apply_async with a per-molecule timeout, falling back to 2D
    coordinates on a hang, since plain pool.imap has no per-item timeout
    and one bad molecule can block an entire chunk forever (observed
    directly: a worker pegged at 100% CPU for 40+ minutes on one molecule).

    Writes results to the LMDB shard incrementally (every `commit_every`
    molecules) instead of accumulating a chunk's full ~1.5M results in
    memory and writing once at the end. This matters specifically because
    a SLURM walltime kill (or any other kill) previously lost an entire
    chunk's progress with nothing written to disk -- at ~20 hours/chunk,
    that is not a rare edge case at this scale. Also resumable: molecules
    whose global index already has an entry in `shard_path` (e.g. from a
    prior attempt that got killed) are skipped rather than regenerated.
    """
    from muben.utils.chem import smiles_to_coords, smiles_to_2d_coords
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from tqdm.auto import tqdm

    shard_path.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(shard_path), subdir=False, map_size=int(map_size_gb * (1024 ** 3)))

    with env.begin() as txn:
        done_keys = set(txn.cursor().iternext(values=False))

    todo = [(i, smi) for i, smi in enumerate(smiles) if _global_index_key(start + i) not in done_keys]
    n_done_already = len(smiles) - len(todo)
    if n_done_already:
        print(f"[resume] {n_done_already:,}/{len(smiles):,} molecules already present in {shard_path} -- skipping")

    if not todo:
        env.close()
        return 0

    s2c = partial(smiles_to_coords, n_conformer=n_conformer)
    buffer = []

    def flush():
        if not buffer:
            return
        with env.begin(write=True) as txn:
            for key, value in buffer:
                txn.put(key, value)
        buffer.clear()

    with get_context("fork").Pool(num_workers) as pool:
        pending = [(i, smi, pool.apply_async(s2c, (smi,))) for i, smi in todo]
        for i, smi, async_result in tqdm(pending, total=len(pending)):
            try:
                atoms, coordinates = async_result.get(timeout=timeout_s)
            except MPTimeoutError:
                print(f"[timeout] {smi!r} -- falling back to 2D coordinates")
                mol = Chem.MolFromSmiles(smi)
                coordinates = [smiles_to_2d_coords(smi)] * (n_conformer + 1)
                mol = AllChem.AddHs(mol)
                atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]

            buffer.append((_global_index_key(start + i), pickle.dumps({"atoms": atoms, "coordinates": coordinates})))
            if len(buffer) >= commit_every:
                flush()

        flush()

    env.close()
    return len(todo)


def merge_shards(shards_dir: Path, out_path: Path, expected_num_chunks: int, map_size_gb: float, allow_partial: bool) -> None:
    """Commits one write transaction per shard (not one giant transaction
    for the whole merge) and tracks completed shards in a sidecar
    `.merge_progress` file, so a merge across ~1000 shards / several TB
    can be killed and resumed without losing everything and restarting --
    the same reasoning that motivated incremental commits in
    generate_conformers_for_chunk, applied to the merge step."""
    shard_paths = sorted(shards_dir.glob("chunk_*.lmdb"))
    if not allow_partial and len(shard_paths) != expected_num_chunks:
        raise SystemExit(
            f"Expected {expected_num_chunks} shards in {shards_dir}, found {len(shard_paths)}. "
            f"Pass --allow-partial to merge an incomplete set anyway."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = out_path.with_suffix(out_path.suffix + ".merge_progress")
    done_shards = set()
    if progress_path.exists():
        done_shards = set(progress_path.read_text().splitlines())
        print(f"[resume] {len(done_shards)}/{len(shard_paths)} shards already merged, skipping")

    out_env = lmdb.open(str(out_path), subdir=False, map_size=int(map_size_gb * (1024 ** 3)))

    n_written = 0
    n_merged_now = 0
    with open(progress_path, "a") as progress_f:
        for shard_path in shard_paths:
            if shard_path.name in done_shards:
                continue

            shard_env = lmdb.open(
                str(shard_path), subdir=False, readonly=True, lock=False,
                readahead=False, meminit=False, max_readers=256,
            )
            with out_env.begin(write=True) as out_txn:
                with shard_env.begin() as txn:
                    cursor = txn.cursor()
                    for key, value in cursor.iternext(keys=True, values=True):
                        out_txn.put(key, value)
                        n_written += 1
            shard_env.close()

            progress_f.write(shard_path.name + "\n")
            progress_f.flush()
            n_merged_now += 1

    out_env.close()

    print(f"[merge] {n_merged_now} shards merged this run ({len(shard_paths)} total) -> {n_written:,} new records -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=None, help="DOCKSTRING target (uses src.data.dockstring.load_pool)")
    parser.add_argument("--smiles-file", default=None, help="Plain text file, one SMILES per line (for arbitrary/billion-scale pools)")
    parser.add_argument("--total-count", type=int, default=None, help="Total pool size, to skip an O(N) line-count scan (strongly recommended for large --smiles-file array jobs, otherwise every chunk task rescans the whole file)")
    parser.add_argument("--out-dir", default=None, help="Override the output directory (default: results/embed/{target-or-file-stem}/)")
    parser.add_argument("--chunk-id", type=int, default=None, help="Required unless --merge")
    parser.add_argument("--num-chunks", type=int, required=True)
    parser.add_argument("--n-conformer", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--timeout-s", type=int, default=30)
    parser.add_argument("--commit-every", type=int, default=2000, help="Flush this many completed molecules to the LMDB shard at a time, rather than holding a whole ~1.5M-molecule chunk in memory until the end. Bounds how much work a kill (SLURM walltime, preemption, crash) can lose.")
    parser.add_argument("--partition", default="train", help="Must match the partition extract_unimol() prepares with (always 'train' today)")
    parser.add_argument("--map-size-gb", type=float, default=30.0, help="LMDB map_size for a single shard. Measured directly against 5000 real Enamine REAL molecules: ~8.9KB/molecule -> a full ~1.56M-molecule chunk (1000-way split of the 1.56B pool) needs ~14GB; 30GB leaves >2x headroom for larger-than-average molecules. The previous 10GB default was undersized and would have hit MDB_MAP_FULL partway through every chunk, not just slow ones.")
    parser.add_argument("--merge", action="store_true", help="Consolidate all chunk shards into the final cache instead of generating one")
    parser.add_argument("--merge-map-size-gb", type=float, default=200.0, help="LMDB map_size for the merged output (must fit the whole pool). At ~8.9KB/molecule (measured), a 1.56B-molecule pool needs ~14TB -- for that scale, override this explicitly (e.g. --merge-map-size-gb 30000) rather than relying on the small default sized for DOCKSTRING-scale (~50K-molecule) pools.")
    parser.add_argument("--allow-partial", action="store_true", help="Merge even if fewer than --num-chunks shards are present")
    args = parser.parse_args()

    if not args.target and not args.smiles_file:
        raise SystemExit("Must pass either --target or --smiles-file")

    if args.target:
        from src.data.dockstring import load_pool
        smiles_in_memory = load_pool(args.target)
        total = len(smiles_in_memory)
        pool_hash = compute_pool_identity(target_smiles=smiles_in_memory)
        default_out_dir = OUTPUT_DIR / args.target
    else:
        smiles_in_memory = None
        total = args.total_count if args.total_count is not None else count_lines(args.smiles_file)
        pool_hash = compute_pool_identity(smiles_file=args.smiles_file)
        default_out_dir = OUTPUT_DIR / Path(args.smiles_file).stem

    out_dir = Path(args.out_dir) if args.out_dir else default_out_dir
    cache_dir = out_dir / f"_unimol_cache_{pool_hash}"
    shards_dir = cache_dir / "_shards"
    final_path = cache_dir / f"{args.partition}.lmdb"

    print(f"[pool] {total:,} molecules total, identity={pool_hash}")

    if args.merge:
        merge_shards(shards_dir, final_path, args.num_chunks, args.merge_map_size_gb, args.allow_partial)
        return

    if args.chunk_id is None:
        raise SystemExit("--chunk-id is required unless --merge is set")

    start, end = _chunk_bounds(total, args.chunk_id, args.num_chunks)
    if smiles_in_memory is not None:
        chunk_smiles = smiles_in_memory[start:end]
    else:
        chunk_smiles = read_smiles_chunk(args.smiles_file, start, end)

    print(f"[chunk {args.chunk_id}/{args.num_chunks}] {len(chunk_smiles):,} molecules (indices [{start}, {end}))")

    shard_path = shards_dir / f"chunk_{args.chunk_id:05d}.lmdb"

    t0 = time.perf_counter()
    n_written = generate_conformers_for_chunk(
        chunk_smiles, shard_path, start, args.map_size_gb,
        n_conformer=args.n_conformer, num_workers=args.num_workers, timeout_s=args.timeout_s,
        commit_every=args.commit_every,
    )
    elapsed = time.perf_counter() - t0

    print(f"[done] chunk {args.chunk_id}: {n_written:,} molecules written in {elapsed:.1f}s -> {shard_path}")


if __name__ == "__main__":
    main()
