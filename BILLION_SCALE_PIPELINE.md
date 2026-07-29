# Billion-Scale Representation Pipeline: Design

Extends the DOCKSTRING-scale pipeline in `DESIGN.md` (~50K molecules/target,
single-process extraction) to arbitrary-size libraries -- concretely, the
1.56B-molecule Enamine REAL lead-like library (SurA and GAK Glide HTVS
docking scores). The core problem DESIGN.md's pipeline doesn't address:
conformer generation and embedding computation each take too long and
produce too much data for a single process, so both need to be split into
independent, parallelizable shards -- while staying correctly aligned to
the same molecules throughout.

## The one invariant everything else depends on

**A canonical SMILES file, one molecule per line, line number = global
index, never reordered.** Every downstream artifact (conformer shard,
embedding chunk) is keyed by this same global index. Nothing is ever
joined by string/hash lookup during normal operation -- alignment is
positional, guaranteed by construction, not by convention. The one place
this breaks down is combining data *across* two different per-target
source files (see "Cross-target alignment" below), which is why it gets
special handling rather than being treated as the general case.

Concretely: `surA_smiles.txt`, line `i` (0-indexed) = global index `i`,
extracted once from the source docking-score file and never rewritten.

## Stage 0: Source extraction

Docking-score files ship as whitespace-separated `SMILES ID score` (no
header), sorted by that file's own docking score (best first) -- see
"Using the sort order" below for why that sort matters. Extraction pulls
one column via a streaming `unzip -p | zcat | awk` pipeline (no
intermediate full decompression to disk), preserving row order exactly.
For the current SurA library: 1,559,853,242 molecules, `awk '{print $1}'`
for SMILES only (no scores/IDs extracted yet -- see "Cross-target
alignment").

Staged on `/N/project/SingleCell_Image/mengjing/enamine_real_1.56B/`, not
`/N/slate` (800GB personal quota, ~150-250x too small) or `/N/scratch`
(100TB, but a confirmed 30-day purge -- real risk for data meant to
outlive one job). Project/condo storage has neither limit; verify RW
access and quota headroom before reusing this pattern for a different lab
allocation.

## Stage 1: Conformer generation (sharded)

`src/representations/generate_conformers.py` + `scripts/submit_conformer_gen.sh`.

- **Chunking**: `_chunk_bounds(total, chunk_id, num_chunks)` deterministically
  splits `[0, total)` into `num_chunks` contiguous global-index ranges.
  Must be called with identical `total`/`num_chunks` in every stage that
  touches this pool -- a mismatch silently shifts chunk boundaries (see
  the alignment-verification note in Stage 2).
- **Per-molecule keying**: within a shard, molecule at global index `i`
  is stored under LMDB key `_global_index_key(i)` (zero-padded to 13
  digits, so byte-lexicographic order == numeric order). LMDB's B+tree
  guarantees sorted-key iteration, so reading a shard back always yields
  molecules in ascending global-index order -- this is what makes Stage
  2 able to skip merging.
- **`--n-conformer 1`**, not muben's default of 10: measured directly,
  ~9.1x faster and ~2.3x smaller on-disk than generating 10. Not a
  quality tradeoff here -- `extract_unimol()`-style embedding extraction
  discards all but one *randomly chosen* conformer before a single
  frozen forward pass anyway (no multi-epoch training for the other 9 to
  help with). Required patching a hardcoded `assert len(coordinates) ==
  11` in vendored `muben/muben/dataset/dataset_unimol/process.py`.
- **Crash safety**: `generate_conformers_for_chunk()` commits to the LMDB
  shard every `--commit-every` (default 2000) molecules and skips
  already-present indices on restart -- a kill (walltime, preemption)
  costs at most one commit interval, not the whole ~1.5M-molecule chunk.
  Verified directly: killed mid-run, confirmed partial data persisted,
  resumed, confirmed it skipped completed work and finished correctly.
- **`--map-size-gb 30`** per shard: measured ~8.9-9.15KB/molecule
  (n_conformer=10) or ~4.01KB/molecule (n_conformer=1) against 5000 real
  molecules -- a full ~1.56M-molecule chunk needs ~6-14GB depending on
  n_conformer; 30GB leaves >2x margin. This is a sparse virtual ceiling
  on Lustre, not real disk usage, so generous headroom is free.
- **SLURM**: array job, one task per chunk, `--time=36:00:00` (measured
  worst-case observed ~26h at n_conformer=10; the n_conformer=1 switch
  should shrink this considerably but wasn't re-measured at full BigRed200
  scale), 16 workers/task. `%K` array suffix caps concurrency to a known
  number rather than the scheduler's opportunistic burst (observed once:
  427 concurrent unprompted).
- **Real bug fixed here**: `generate_conformers.py`'s default `--out-dir`
  for `--smiles-file` mode is `ROOT/results/embed/{stem}` -- always under
  the repo on Slate, regardless of where the input SMILES file lives.
  This silently sent ~714GB to the 800GB Slate quota before being caught.
  `submit_conformer_gen.sh` now passes `--out-dir` explicitly, derived
  from the input file's own directory, every time.

Output: `{out_dir}/_unimol_cache_{pool_hash}/_shards/chunk_{id:05d}.lmdb`,
one per chunk. For the current SurA run: 1000/1000 shards, 5.5TB, zero
failures.

## Stage 2: Embedding computation (sharded, no merge)

`src/representations/compute_unimol_embeddings_chunk.py` +
`concat_embedding_chunks.py` (the latter deliberately usually unused --
see below).

**No merge step exists between Stage 1 and Stage 2.** Embedding
computation only needs one chunk's SMILES slice and that same chunk's
shard, in matching order -- both already satisfy that by construction
(same `_chunk_bounds`, same global-index ordering). `load_chunk_dataset()`
populates a `DatasetUniMol` directly from one shard via `load_lmdb()`,
bypassing `DatasetUniMol.prepare()`'s directory/partition-based file
lookup (built for one merged file, not many chunk shards).

This also solves a real resource-shape problem a merge-then-embed design
would not: GPU nodes are far scarcer than the CPU nodes used for Stage 1
(likely single digits to a few dozen concurrent on BigRed200, not
hundreds) -- chunk-level parallelism here reuses Stage 1's boundaries
rather than requiring one large embedding job to read from whatever a
merge would have produced.

**Alignment verification, not just a count check**: a shard/SMILES-slice
count match does not prove molecule `i`'s conformer actually came from
SMILES `i` -- if Stage 1 and Stage 2 are ever run with different
`--num-chunks`/`--total-count`, `_chunk_bounds()` silently computes
different boundaries, potentially preserving count while pairing every
molecule with the wrong conformer. `_verify_alignment()` re-derives each
of a random sample's expected all-hydrogen atom count from its own SMILES
(`AllChem.AddHs(mol).GetNumAtoms()`, confirmed to match this project's own
atom-counting convention in both the normal and 2D-fallback generation
paths) and compares against what the shard actually stored. Fails loudly
on mismatch rather than silently producing misaligned embeddings.

**Output stays chunked deliberately.** `concat_embedding_chunks.py`
exists but is not run for the full 1.56B set: one ~3TB embeddings.npz
would recreate, on the read side, the same problem Stage 1's merge would
have created on the write side -- `EmbeddingFeatureSource` (the existing
AL-loop feature source) loads one file into an in-memory dict, which
doesn't fit any more than the LookupObjective problem below does. Final
output: `unimol_embeddings_chunk_{id:05d}.npz` (`embeddings`, `smiles`)
per chunk, left as-is.

## Cross-target alignment (adding GAK, or any second target)

**Different targets' docking-score files are not in the same row order.**
Confirmed directly: the first 3 rows of SurA and GAK are entirely
different compounds. Each file is independently sorted by its own
docking score (best-first for that target), so "row 1" means "top
scorer for that target," which differs by target. Positional alignment
(the Stage 0-2 invariant) holds *within* one target's extraction; it does
not hold *across* targets.

To use a second target's scores against embeddings/conformers already
computed for the first target's ordering:

1. Extract the first target's Enamine ID column too (dropped so far for
   SurA -- SMILES-only was extracted originally), preserving its row
   order, giving `id[i]` alongside `smiles[i]`.
2. Build an ID -> global-index mapping as an **LMDB** (not an in-memory
   dict -- 1.56B string keys would face the same multi-hundred-GB problem
   as `LookupObjective` below).
3. Stream the second target's file; for each row, look up its ID in that
   LMDB; on a hit, record `(first_target_global_index, second_target_score)`.
   Rows with no match are dropped for that target, not treated as errors
   -- coverage between the two files isn't yet confirmed to be identical
   (worth checking total row counts before assuming so).

Not yet built -- needed only when actually using a second target, not
before.

## Using the docking-score sort order

Legitimate: since each file is pre-sorted by score, the true top-*k* for
Recall@*k*/AUDC ground truth is just the first *k* rows (e.g., true
top-1% = first ~15.6M rows) -- free, no ranking computation needed.
Score-threshold queries are a binary search, not a full scan.

Illegitimate, and would silently invalidate any AL evaluation built on
it: never construct the *searchable pool* (what the AL loop selects
from) from the top of a score-sorted file. That manufactures a pool
where every molecule is already a near-hit, making the discovery problem
degenerate and Recall@k trivially ~100%. If a smaller pool is ever drawn
from the full 1.56B set (mirroring DOCKSTRING's 50K-per-target
subsamples), it must be drawn at random across the full file; the sort
order is for computing ground truth afterward, never for choosing which
rows are in the pool.

## Known scaling gaps -- not yet solved, don't block current work

Two pieces of the existing (DOCKSTRING-scale) codebase were designed
around a `~50K`-`~260K`-molecule pool and do not scale to 1.56B as-is.
Neither blocks conformer generation or embedding computation, which are
done here; both matter only once actually running AL against this pool.

- **`molpal/objectives/lookup.py`'s `LookupObjective`** loads its entire
  lookup file into one Python dict, `{smiles_string: score}`, at
  initialization. At ~150-250 bytes/entry x 1.56B molecules, that's
  roughly 250-350GB of RAM for the oracle alone. Needs a position-indexed,
  disk-backed redesign (e.g., keyed by the same global index this whole
  pipeline already uses, not by SMILES string) before it can serve as the
  oracle for a billion-scale AL run.
- **`src/representations/feature_source.py`'s `EmbeddingFeatureSource`**
  loads one `.npz` into memory and builds a SMILES->row dict, same shape
  of problem. This is exactly why Stage 2's output is left chunked rather
  than concatenated -- concatenating would just move the memory problem
  from "building the file" to "reading the file," not solve it. A
  billion-scale-compatible feature source needs to read from many chunk
  files (or something disk-backed/positional) directly.
- MolPAL's `MoleculePool`/`Explorer` more generally haven't been audited
  for billion-scale assumptions at all -- unknown scope of work, flagged
  here so it isn't forgotten, not because anything specific is confirmed
  broken.

## Reference numbers (all measured directly, not estimated)

| Quantity | Value |
|---|---|
| SurA pool size | 1,559,853,242 molecules |
| SMILES-only extraction size | ~67GB (71,888,092,106 bytes) |
| Conformer generation rate (16 workers, n_conformer=10) | 16.6-33.5 it/s (sampled, 5 tasks) |
| Conformer size, n_conformer=10 | ~8.9-9.15 KB/molecule |
| Conformer size, n_conformer=1 | ~4.01 KB/molecule |
| n_conformer=1 vs. 10 speedup | ~9.1x (measured, 300 molecules) |
| Full conformer cache (1000 shards, n_conformer=1, SurA) | 5.5TB, 1000/1000 complete, 0 failures |
| Final embeddings, full 1.56B (512-dim float32) | ~3.0TB |
| Slate personal quota | 800GB |
| Scratch quota / purge | 100TB / 30 days (confirmed) |
| SingleCell_Image project quota | ~60TB allocation, ~49TB free when checked |
