#!/usr/bin/env python3
"""Stage 2 of the split Uni-Mol pipeline, chunked: compute embeddings
directly from one conformer shard produced by generate_conformers.py,
without ever merging shards into one file.

No merge is needed because embedding computation only needs, for each
molecule, its SMILES plus its conformer (atoms/coordinates) in matching
order -- both are already available per-chunk: read_smiles_chunk() gives
the same SMILES slice generate_conformers.py used for this chunk-id, and
that chunk's shard (chunk_{id:05d}.lmdb) holds exactly those molecules'
conformers, in the same order (global LMDB keys sort correctly within a
contiguous chunk range -- see generate_conformers.py's _global_index_key).

This also solves a real scaling problem a merge-then-embed design would
not: GPU nodes are far scarcer than the CPU nodes used for conformer
generation (likely single digits to a few dozen concurrent on BigRed200,
not hundreds), so embedding computation needs its own chunk-level
parallelization anyway -- this reuses the same chunk boundaries already
on disk rather than requiring one large embedding job to read from
whatever the merge would have produced.

Output per chunk: {out_dir}/unimol_embeddings_chunk_{chunk_id:05d}.npz
(embeddings, smiles). Concatenate all chunks' outputs with
concat_embedding_chunks.py once done -- cheap array-stacking, nothing
like the I/O cost of merging raw conformer shards.

Usage
-----
python -m src.representations.compute_unimol_embeddings_chunk \\
    --smiles-file /path/to/library.smi --total-count 1559853242 \\
    --shards-dir /path/to/_conformer_output/_unimol_cache_HASH/_shards \\
    --chunk-id 0 --num-chunks 1000 \\
    --out-dir /path/to/_conformer_output/_embeddings
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([argparse.Namespace])

from src.representations.extract import MODEL_ZOO, ROOT, MubenRuntimeConfig
from src.representations.generate_conformers import _chunk_bounds, count_lines, read_smiles_chunk

_muben_root = ROOT / "muben"
if str(_muben_root) not in sys.path:
    sys.path.insert(0, str(_muben_root))

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    DEVICE = torch.device("cpu")


def _verify_alignment(chunk_smiles: list, atoms: list, sample_size: int = 20, seed: int = 0) -> None:
    """A count match (len(atoms) == len(chunk_smiles)) does not prove
    molecule i's conformer actually came from chunk_smiles[i] -- if
    generation and this script were ever run with a different
    --num-chunks/--total-count, _chunk_bounds() would silently compute
    different global-index boundaries, potentially preserving the count
    while pairing every molecule with the wrong conformer. Re-derive each
    sampled molecule's expected all-hydrogen atom count from its own
    SMILES (matching this project's own 2D/3D-fallback atom-counting
    convention: AllChem.AddHs(mol).GetNumAtoms()) and compare against
    what the shard actually stored at that position -- this is a real
    content check, not just a count, and fails loudly rather than
    silently producing misaligned embeddings."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(chunk_smiles), size=min(sample_size, len(chunk_smiles)), replace=False)

    mismatches = []
    for i in idxs:
        smi = chunk_smiles[i]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue  # unparseable SMILES can't be independently re-verified; not a misalignment signal
        expected_n_atoms = AllChem.AddHs(mol).GetNumAtoms()
        actual_n_atoms = len(atoms[i])
        if expected_n_atoms != actual_n_atoms:
            mismatches.append((i, smi, expected_n_atoms, actual_n_atoms))

    if mismatches:
        detail = "; ".join(f"index {i}: {smi!r} expected {exp} atoms, shard has {act}" for i, smi, exp, act in mismatches[:5])
        raise RuntimeError(
            f"Conformer/SMILES alignment check FAILED for {len(mismatches)}/{len(idxs)} sampled molecules "
            f"({detail}) -- do not trust these embeddings. Almost certainly a --num-chunks/--total-count "
            f"mismatch between the generate_conformers.py run that produced this shard and this invocation."
        )


def load_chunk_dataset(chunk_smiles: list, shard_path: Path, config):
    """Builds a DatasetUniMol directly from one conformer shard, bypassing
    DatasetUniMol.prepare()'s directory/partition-based file lookup (which
    expects one merged {partition}.lmdb, not many per-chunk shards) --
    everything that lookup would have set up is set here directly instead."""
    from muben.dataset import DatasetUniMol
    from muben.dataset.dataset_unimol.dictionary import DictionaryUniMol
    from muben.dataset.dataset_unimol.process import ProcessingPipeline
    from muben.utils.io import load_lmdb

    dictionary = DictionaryUniMol.load()
    dictionary.add_symbol("[MASK]", is_special=True)

    dataset = DatasetUniMol()
    dataset._partition = "train"
    dataset.processing_pipeline = ProcessingPipeline(
        dictionary=dictionary, max_atoms=config.max_atoms, max_seq_len=config.max_seq_len,
        remove_hydrogen_flag=config.remove_hydrogen, remove_polar_hydrogen_flag=config.remove_polar_hydrogen,
    )
    dataset.set_processor_variant("training")

    n = len(chunk_smiles)
    dataset._smiles = chunk_smiles
    dataset._lbs = np.zeros((n, 1), dtype=np.float32)
    dataset._masks = np.ones((n, 1), dtype=np.float32)
    dataset._ori_ids = None
    dataset._atoms, dataset._coordinates = load_lmdb(str(shard_path), ["atoms", "coordinates"])
    assert len(dataset._atoms) == n, (
        f"Shard {shard_path} has {len(dataset._atoms)} records but the chunk has {n} SMILES -- "
        f"mismatched chunk boundaries (wrong --num-chunks/--total-count?) or an incomplete shard."
    )
    _verify_alignment(chunk_smiles, dataset._atoms)
    dataset.data_instances = dataset.get_instances()
    return dataset, dictionary


def compute_embeddings_for_chunk(
    chunk_smiles: list, shard_path: Path, checkpoint_path: Path,
    batch_size: int = 256, num_workers: int = 4,
) -> np.ndarray:
    from muben.dataset.dataset_unimol import CollatorUniMol
    from muben.model.unimol.unimol import UniMol

    config = MubenRuntimeConfig(
        model_name="unimol", target="chunked", out_dir="/tmp", pool_hash="nohash",
        feature_type="unimol", checkpoint_path=checkpoint_path,
    )

    dataset, unimol_dict = load_chunk_dataset(chunk_smiles, shard_path, config)

    collator = CollatorUniMol(config, unimol_dict)
    pad_idx = unimol_dict.pad()
    collator._atom_pad_idx = pad_idx
    collator.pad_idx = pad_idx
    collator.atom_pad_idx = pad_idx

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collator,
        num_workers=num_workers, pin_memory=True,
    )

    model = UniMol(config=config, dictionary=unimol_dict).to(DEVICE)

    def _get_embeddings(self, batch):
        src_tokens, src_distance, src_edge_type = batch.atoms, batch.distances, batch.edge_types
        padding_mask = src_tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None

        x = self.embed_tokens(src_tokens)
        n_node = src_distance.size(-1)
        gbf_feat = self.gbf(src_distance, src_edge_type)
        gbf_result = self.gbf_proj(gbf_feat)
        attn_bias = gbf_result.permute(0, 3, 1, 2).contiguous().view(-1, n_node, n_node)

        encoder_rep, _, _, _, _ = self.encoder(x, padding_mask=padding_mask, attn_mask=attn_bias)
        return self.hidden_layer(encoder_rep[:, 0, :])  # CLS token -> (B, 512)

    model.get_embeddings = types.MethodType(_get_embeddings, model)
    model.eval()

    embeddings = []
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    autocast_kwargs = dict(device_type="cuda", dtype=amp_dtype) if torch.cuda.is_available() else dict(device_type="cpu", enabled=False)

    with torch.no_grad():
        for batch in loader:
            batch.to(DEVICE)
            with torch.autocast(**autocast_kwargs):
                feat = model.get_embeddings(batch)
            embeddings.append(feat.float().cpu().numpy())

    return np.vstack(embeddings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles-file", required=True)
    parser.add_argument("--total-count", type=int, default=None, help="Skip an O(N) line-count scan; strongly recommended at scale")
    parser.add_argument("--shards-dir", required=True, help="Directory containing chunk_XXXXX.lmdb shards from generate_conformers.py")
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--num-chunks", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    total = args.total_count if args.total_count is not None else count_lines(args.smiles_file)
    start, end = _chunk_bounds(total, args.chunk_id, args.num_chunks)
    chunk_smiles = read_smiles_chunk(args.smiles_file, start, end)

    shard_path = Path(args.shards_dir) / f"chunk_{args.chunk_id:05d}.lmdb"
    if not shard_path.exists():
        raise SystemExit(f"Shard not found: {shard_path}")

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else MODEL_ZOO / "unimol" / "mol_pre_all_h_220816.pt"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"unimol_embeddings_chunk_{args.chunk_id:05d}.npz"

    if out_path.exists():
        print(f"[skip] {out_path} already exists")
        return

    print(f"[chunk {args.chunk_id}/{args.num_chunks}] {len(chunk_smiles):,} molecules (indices [{start}, {end}))")

    t0 = time.perf_counter()
    matrix = compute_embeddings_for_chunk(
        chunk_smiles, shard_path, checkpoint_path,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    elapsed = time.perf_counter() - t0

    np.savez(out_path, embeddings=matrix, smiles=np.array(chunk_smiles))
    print(f"[done] chunk {args.chunk_id}: {matrix.shape[0]:,} embeddings ({matrix.shape[1]}d) in {elapsed:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
