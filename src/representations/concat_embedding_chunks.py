#!/usr/bin/env python3
"""Concatenates per-chunk unimol_embeddings_chunk_{id}.npz files (from
compute_unimol_embeddings_chunk.py) into one final embeddings.npz --
cheap array-stacking (a few TB of float32 read+write, one pass), nothing
like the per-record LMDB I/O cost of merging raw conformer shards.

Usage
-----
python -m src.representations.concat_embedding_chunks \\
    --chunks-dir /path/to/_conformer_output/_embeddings \\
    --num-chunks 1000 \\
    --out /path/to/results/embed/enamine_real_1.56B/unimol_embeddings.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-dir", required=True)
    parser.add_argument("--num-chunks", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-partial", action="store_true", help="Concatenate even if fewer than --num-chunks chunk files are present")
    args = parser.parse_args()

    chunks_dir = Path(args.chunks_dir)
    chunk_paths = sorted(chunks_dir.glob("unimol_embeddings_chunk_*.npz"))

    if not args.allow_partial and len(chunk_paths) != args.num_chunks:
        raise SystemExit(
            f"Expected {args.num_chunks} chunk files in {chunks_dir}, found {len(chunk_paths)}. "
            f"Pass --allow-partial to concatenate an incomplete set anyway."
        )

    embeddings_parts, smiles_parts = [], []
    for p in chunk_paths:
        data = np.load(p, allow_pickle=False)
        embeddings_parts.append(data["embeddings"])
        smiles_parts.append(data["smiles"])
        print(f"[loaded] {p.name}: {data['embeddings'].shape}")

    embeddings = np.vstack(embeddings_parts)
    smiles = np.concatenate(smiles_parts)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, embeddings=embeddings, smiles=smiles)

    print(f"[done] {len(chunk_paths)} chunks -> {embeddings.shape[0]:,} molecules, {embeddings.shape[1]}d -> {out_path}")


if __name__ == "__main__":
    main()
