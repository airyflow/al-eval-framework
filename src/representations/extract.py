#!/usr/bin/env python3
"""Frozen-encoder embedding extraction, retargeted at DOCKSTRING.

Adapted from FusionAL's extract_embeddings.py (DESIGN.md Sec 2.1/4.2): the
MUBen backbone-loading code (GROVER/Uni-Mol/MoLFormer model loading,
batching, AMP) is dataset-agnostic and reused unchanged. The only retarget is
the SMILES source -- `src.data.dockstring.load_pool(target)` instead of a
molpal library CSV -- and the output location, `results/embed/{target}/`,
matching configs/representation/*.yaml's `embed_path` convention.

Output: results/embed/{target}/{backbone}_embeddings.npz, containing both
'embeddings' (N, D) and 'smiles' (N,) so row alignment is self-documenting
(consumed by src/representations/feature_source.py::EmbeddingFeatureSource).

Usage
-----
python -m src.representations.extract --target PARP1 --backbone molformer
python -m src.representations.extract --target PARP1 --backbone unimol
python -m src.representations.extract --target PARP1 --backbone all
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([argparse.Namespace])

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_ZOO = ROOT / "models"
OUTPUT_DIR = ROOT / "results" / "embed"

_muben_root = ROOT / "muben"
if str(_muben_root) not in sys.path:
    sys.path.insert(0, str(_muben_root))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    DEVICE = torch.device("cpu")


# ==============================================================================
# MONKEY-PATCH -- replace MUBen's read_csv with a DOCKSTRING-aware version
# ==============================================================================

def _make_pool_reader(smiles_list: list):
    """A read_csv replacement that injects the DOCKSTRING pool SMILES
    directly into the MUBen Dataset object, bypassing any CSV path logic.
    Labels/masks are dummies -- only the SMILES matter for embedding
    extraction; no head is trained here.
    """

    def pool_read_csv(self, data_dir: str, partition: str):
        n = len(smiles_list)
        self._smiles = smiles_list
        self._lbs = np.zeros((n, 1), dtype=np.float32)
        self._masks = np.ones((n, 1), dtype=np.float32)
        self._ori_ids = None
        print(f"[dataset] Injected {n:,} SMILES from DOCKSTRING pool "
              f"(partition='{partition}' ignored)")
        return self

    return pool_read_csv


def patch_muben_dataset(smiles_list: list):
    """Apply the monkey-patch before instantiating any MUBen Dataset subclass."""
    import muben.dataset.dataset as _ds_module

    _ds_module.Dataset.read_csv = _make_pool_reader(smiles_list)


# ==============================================================================
# SHARED CONFIG
# ==============================================================================

def compute_pool_hash(smiles: list) -> str:
    """Short content hash of a SMILES pool, used to key on-disk caches so
    that two different SMILES sets (e.g. a smoke test vs. the real Phase 0
    pool) sharing the same `target` name never collide on the same cache
    path -- the bug that previously forced disabling caching entirely.

    Hashes incrementally rather than `"".join(smiles)` first -- joining
    materializes the whole pool as one Python string, which is fine at
    50K molecules but not at billion-molecule scale (~50GB+ for a 1.3B
    pool). Only used for in-memory pools (e.g. DOCKSTRING targets); see
    `compute_pool_identity` for the file-based, billion-scale case.
    """
    h = hashlib.sha1()
    for smi in smiles:
        h.update(smi.encode())
        h.update(b"\n")
    return h.hexdigest()[:10]


def compute_pool_identity(smiles_file: str = None, target_smiles: list = None) -> str:
    """Cache key for a pool, from either a file path (billion-scale custom
    libraries) or an in-memory SMILES list (DOCKSTRING targets).

    A file is identified by (resolved path, size, mtime) rather than its
    content hash -- content-hashing a 1.3B-line file is a full sequential
    read (tens of GB), and `generate_conformers.py` needs to derive this
    same identity independently in every chunk task (potentially
    thousands of them for a billion-scale array job), so it must be O(1)
    metadata rather than O(N) content, in exchange for the (accepted)
    weakness that touching the file without changing size/mtime won't be
    detected as a change.
    """
    if smiles_file is not None:
        st = os.stat(smiles_file)
        fingerprint = f"{Path(smiles_file).resolve()}:{st.st_size}:{st.st_mtime_ns}"
        return hashlib.sha1(fingerprint.encode()).hexdigest()[:10]
    return compute_pool_hash(target_smiles)


class MubenRuntimeConfig:
    def __init__(
        self, model_name, target, out_dir, pool_hash: str = "nohash",
        feature_type="none", checkpoint_path="",
    ):
        cache_key = f"{target}_{pool_hash}"
        self.data_dir = str(ROOT / "muben" / "data" / "files" / cache_key)
        self.model_name = model_name
        self.feature_type = feature_type
        self.checkpoint_path = str(checkpoint_path)
        # Conformer/feature cache lives in its own hash-keyed subdirectory,
        # separate from the final .npz outputs saved directly under out_dir.
        self.unimol_feature_dir = str(Path(out_dir) / f"_unimol_cache_{pool_hash}")
        self.num_preprocess_workers = 4

        # Cache key now includes a content hash (see pool_hash above), so a
        # different SMILES set can no longer collide with an existing cache
        # under the same target name -- safe to reuse across repeated runs
        # on the same (target, pool) pair, e.g. conformer generation for
        # Uni-Mol, which is the single slowest step in extraction.
        self.ignore_preprocessed_dataset = False
        self.disable_dataset_saving = False
        self.disable_checkpoint_loading = False

        # DatasetUniMol.create_features() branches on this when a
        # pre-generated `unimol_feature_dir/{partition}.lmdb` exists (see
        # src/representations/generate_conformers.py, which now populates
        # that path for the split conformer-generation stage). Required --
        # without it, that branch raises AttributeError, since it was
        # previously dead code (nothing ever wrote to unimol_feature_dir).
        self.random_split = False

        # GROVER
        self.hidden_size = 128
        self.dropout = 0.1
        self.bias = False
        self.num_mt_block = 1
        self.num_attn_head = 4
        self.embedding_output_type = "both"

        # Uni-Mol
        self.max_atoms = 64
        self.max_seq_len = 80
        self.only_polar_hydrogens = False
        self.remove_hydrogen = True
        self.remove_polar_hydrogen = False
        self.encoder_embed_dim = 512
        self.encoder_layers = 15
        self.encoder_attention_heads = 64
        self.encoder_ffn_embed_dim = 2048
        self.activation_fn = "gelu"
        self.pooler_stride = 1
        self.pooler_dropout = 0.0
        self.emb_dropout = 0.1
        self.attention_dropout = 0.1
        self.activation_dropout = 0.0
        self.delta_pair_repr_norm_loss = -1
        self.masked_coord_loss = 0.0
        self.masked_dist_loss = 0.0
        self.masked_type_loss = 0.0
        self.pooler_activation_fn = "Tanh"

        # MoLFormer
        self.pretrained_model_name_or_path = "ibm-research/MoLFormer-XL-both-10pct"
        self.tokenizer_trust_remote_code = True

        # Task (unused -- no head is trained during extraction)
        self.uncertainty_method = "none"
        self.task_type = "regression"
        self.bbp_prior_sigma = 0.5
        self.n_lbs = 1
        self.n_tasks = 1
        self.activation = "ReLU"
        self.ffn_num_layers = 2
        self.ffn_hidden_size = 128


# ==============================================================================
# SAVE -- always bundle SMILES + embeddings together
# ==============================================================================

def save_embeddings(name: str, matrix: np.ndarray, smiles: list, out_dir: Path) -> dict:
    out_npz = out_dir / f"{name}_embeddings.npz"
    np.savez(out_npz, embeddings=matrix, smiles=np.array(smiles))

    print(f"[saved] {out_npz.name}  shape={matrix.shape}")
    print(f"        smiles[0]  = {smiles[0]}")
    print(f"        smiles[-1] = {smiles[-1]}")

    return {"n_molecules": matrix.shape[0], "embedding_dim": matrix.shape[1], "output_path": str(out_npz)}


# ==============================================================================
# EXTRACTORS
# ==============================================================================

def extract_grover(smiles: list, target: str, out_dir: Path, pool_hash: str = "nohash") -> dict:
    print("\n>>> GROVER 2D Graph Representations...")

    from muben.dataset import DatasetGrover
    from muben.dataset.dataset_grover import CollatorGrover
    from muben.model import GROVER

    config = MubenRuntimeConfig(model_name="grover", target=target, out_dir=out_dir, pool_hash=pool_hash)
    dataset = DatasetGrover()
    dataset.prepare(config=config, partition="train")

    collator = CollatorGrover(config)
    loader = DataLoader(
        dataset, batch_size=128, shuffle=False, collate_fn=collator,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2,
    )

    ckpt_path = MODEL_ZOO / "grover" / "grover_base.pt"
    assert ckpt_path.exists(), f"Missing: {ckpt_path}"

    model_cfg = MubenRuntimeConfig(
        model_name="grover", target=target, out_dir=out_dir, pool_hash=pool_hash, checkpoint_path=ckpt_path
    )
    model = GROVER(model_cfg).to(DEVICE)
    model.eval()

    embeddings = []
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    autocast_kwargs = dict(device_type="cuda", dtype=amp_dtype) if torch.cuda.is_available() else dict(device_type="cpu", enabled=False)

    with torch.no_grad():
        for batch in loader:
            batch.to(DEVICE)
            components = batch.molecule_graphs.components
            _, _, _, _, _, a_scope, _, _ = components

            with torch.autocast(**autocast_kwargs):
                output = model.grover(components)
                mol_from_bond = model.readout(output["atom_from_bond"], a_scope)
                mol_from_atom = model.readout(output["atom_from_atom"], a_scope)
                combined = torch.cat([mol_from_bond, mol_from_atom], dim=1)

            embeddings.append(combined.float().cpu().numpy())

    matrix = np.vstack(embeddings)
    return save_embeddings("grover", matrix, smiles, out_dir)


def extract_unimol(smiles: list, target: str, out_dir: Path, pool_hash: str = "nohash") -> dict:
    print("\n>>> Uni-Mol 3D Conformational Representations...")

    from muben.dataset import DatasetUniMol
    from muben.dataset.dataset_unimol import CollatorUniMol
    from muben.dataset.dataset_unimol.dictionary import DictionaryUniMol
    from muben.model.unimol.unimol import UniMol

    unimol_ckpt = MODEL_ZOO / "unimol" / "mol_pre_all_h_220816.pt"
    config = MubenRuntimeConfig(
        model_name="unimol", target=target, out_dir=out_dir, pool_hash=pool_hash,
        feature_type="unimol", checkpoint_path=unimol_ckpt,
    )

    dataset = DatasetUniMol()
    dataset.prepare(config=config, partition="train")

    unimol_dict = DictionaryUniMol.load()
    unimol_dict.add_symbol("[MASK]", is_special=True)
    print(f"[dict] vocab size: {len(unimol_dict)}")

    collator = CollatorUniMol(config, unimol_dict)
    pad_idx = unimol_dict.pad()
    collator._atom_pad_idx = pad_idx
    collator.pad_idx = pad_idx
    collator.atom_pad_idx = pad_idx

    loader = DataLoader(
        dataset, batch_size=256, shuffle=False, collate_fn=collator,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2,
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

    matrix = np.vstack(embeddings)
    return save_embeddings("unimol", matrix, smiles, out_dir)


def extract_molformer(smiles: list, target: str, out_dir: Path, pool_hash: str = "nohash") -> dict:
    # pool_hash unused -- MolFormer doesn't go through muben's Dataset cache.
    print("\n>>> MoLFormer 1D Chemical Language Representations...")

    from transformers import AutoModel, AutoTokenizer

    molformer_path = str(MODEL_ZOO / "molformer")
    tokenizer = AutoTokenizer.from_pretrained(molformer_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(molformer_path, trust_remote_code=True).to(DEVICE)
    model.eval()

    embeddings = []
    batch_size = 256
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    autocast_kwargs = dict(device_type="cuda", dtype=amp_dtype) if torch.cuda.is_available() else dict(device_type="cpu", enabled=False)

    for i in range(0, len(smiles), batch_size):
        batch_smi = smiles[i:i + batch_size]
        inputs = tokenizer(batch_smi, padding=True, truncation=True, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            with torch.autocast(**autocast_kwargs):
                outputs = model(**inputs)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                emb = outputs.pooler_output.float()
            else:
                emb = outputs.last_hidden_state.mean(dim=1).float()
        embeddings.append(emb.cpu().numpy())

    matrix = np.vstack(embeddings)
    return save_embeddings("molformer", matrix, smiles, out_dir)


_ALL_EXTRACTORS = [("grover", extract_grover), ("unimol", extract_unimol), ("molformer", extract_molformer)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="DOCKSTRING target, e.g. PARP1")
    parser.add_argument(
        "--backbone", default="all", choices=["all", "grover", "unimol", "molformer"],
        help="Which backbone to extract (default: all). Phase 0 only needs molformer + unimol.",
    )
    args = parser.parse_args()

    from src.data.dockstring import load_pool

    out_dir = OUTPUT_DIR / args.target
    out_dir.mkdir(parents=True, exist_ok=True)

    smiles = load_pool(args.target)
    print(f"[pool] {args.target}: {len(smiles):,} SMILES loaded")

    phash = compute_pool_hash(smiles)
    print(f"[pool] content hash: {phash} (keys the on-disk conformer/feature cache)")

    patch_muben_dataset(smiles)

    extractors = _ALL_EXTRACTORS if args.backbone == "all" else [
        (bb, fn) for bb, fn in _ALL_EXTRACTORS if bb == args.backbone
    ]

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    run_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    records = []
    total_t0 = time.perf_counter()

    for backbone, fn in extractors:
        t0 = time.perf_counter()
        meta = fn(smiles, args.target, out_dir, pool_hash=phash)
        elapsed = time.perf_counter() - t0
        records.append({
            "timestamp": run_ts, "target": args.target, "backbone": backbone,
            "n_molecules": meta["n_molecules"], "embedding_dim": meta["embedding_dim"],
            "elapsed_s": round(elapsed, 1), "device": device_str, "output_path": meta["output_path"],
        })
        print(f"  [{backbone}] done in {elapsed:.1f}s")

    total_elapsed = time.perf_counter() - total_t0
    print(f"\n[done] Total extraction time: {total_elapsed:.1f}s")

    log_path = OUTPUT_DIR / "extraction_log.txt"
    with open(log_path, "a") as f:
        f.write(f"\n=== {run_ts}  target={args.target}  device={device_str} ===\n")
        for r in records:
            f.write(
                f"  {r['backbone']:<12}  {r['n_molecules']:>6,} mol  {r['embedding_dim']:>5}d"
                f"  {r['elapsed_s']:>8.1f}s  ->  {r['output_path']}\n"
            )
        f.write(f"  {'total':<12}  {' ':>6}     {' ':>5}   {total_elapsed:>8.1f}s\n")

    print(f"[log]  {log_path}")
    print(f"\n[done] Embeddings in {out_dir}")


if __name__ == "__main__":
    main()
