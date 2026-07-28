#!/bin/bash

#SBATCH -J conformer_gen
#SBATCH -p general
#SBATCH -o logs/conformer_gen_%A_%a.txt
#SBATCH -e logs/conformer_gen_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=36:00:00
#SBATCH --mem=32G
#SBATCH -A r00939

# CPU-only Stage 1 of the split Uni-Mol pipeline (conformer generation
# only -- see src/representations/generate_conformers.py). No GPU and no
# model loading, so this scales across many parallel array tasks/nodes
# for a billion-scale pool, independent of the GPU-based embedding
# stage that runs afterward (submit_unimol_extract.sh).
#
# Usage (must be submitted as a SLURM array job so $SLURM_ARRAY_TASK_ID
# is set -- pass --array explicitly, it is not baked in here since the
# right chunk count depends on the pool size). Optionally cap concurrent
# tasks with SLURM's %K array suffix (e.g. --array=0-999%50 runs at most
# 50 at once) -- worth using regardless of your account's real fair-share
# limit, so you get a known, predictable concurrency rather than finding
# it out by trial.
#   sbatch --array=0-<NUM_CHUNKS-1>[%MAX_CONCURRENT] scripts/submit_conformer_gen.sh <TARGET-or-smiles-file> <NUM_CHUNKS> [TOTAL_COUNT]
#
# e.g. a small DOCKSTRING target, mostly for testing the split path:
#   sbatch --array=0-9 scripts/submit_conformer_gen.sh PARP1 10
#
# e.g. the 1.56B-molecule Enamine REAL SurA library, 1000 chunks
# (~1.56M mol/chunk), passing the exact pre-counted total so none of the
# 1000 tasks redundantly re-scans the 67GB SMILES file just to size its
# chunk boundaries. Staged under /N/project/SingleCell_Image/mengjing,
# not /N/slate (800GB personal quota -- the ~6.1TB conformer cache at
# --n-conformer 1, see below, is ~7.6x too big) and not /N/scratch
# (100TB, but purges after 30 days, confirmed directly). This lab
# condo allocation has confirmed RW access, ~49TB free of a 60TB
# allocation, and no purge policy -- also where the source Enamine REAL
# zip already lives (SingleCell_Image/Yang/AI Drug/LargeData/), so this
# is data the same lab already stages, not an unrelated use of shared
# storage. Also solves where the final ~3TB of embeddings
# (produced by the separate embedding-computation stage in extract.py,
# not this script) should live -- no need to wait on the pending Slate
# quota increase for this pipeline specifically. Run the merge step
# (below) as soon as most/all chunks finish:
#   sbatch --array=0-999 scripts/submit_conformer_gen.sh /N/project/SingleCell_Image/mengjing/enamine_real_1.56B/surA_smiles.txt 1000 1559853242
#
# --n-conformer 1 (down from muben's default of 10): measured directly,
# generating 10 conformers and generating 1 differ by ~9.1x in wall-clock
# time (38.2s vs 4.2s for the same 300 real molecules, 8 workers) and
# ~2.3x in on-disk size (9.15KB/molecule vs 4.01KB/molecule). This is not
# a quality tradeoff for our use case: extract_unimol() always prepares
# with partition="train", which routes through muben's
# process_training() -> conformer_sampling(), which discards all but one
# *randomly chosen* conformer before a single frozen forward pass --
# there is no multi-epoch training here for the extra conformers to add
# robustness to, so generating 10 and discarding 9 was equivalent in
# expectation to generating 1 directly. The vendored
# muben/muben/dataset/dataset_unimol/process.py conformer_sampling()
# had a hardcoded `assert len(coordinates) == 11` blocking this -- now
# relaxed to `>= 1` and verified end-to-end (real DatasetUniMol.prepare()
# + process_training() call, correct tensor shapes out). Checked and
# ruled out as a factor: OMP_NUM_THREADS=4 x 16 workers on 16 cores
# (4x oversubscription in principle) made no measurable difference
# (12.7s vs 12.6s for 2000 molecules) -- RDKit's ETKDG generation isn't
# leaning on multithreaded BLAS enough for this to matter here.
#
# Combined effect on the 1.56B pool: real per-task throughput sampled
# from job 7800807 at n_conformer=10 (5 tasks, ~15min in) ranged
# 16.6-33.5 it/s, i.e. ~12.9-26.1h per ~1.56M-molecule chunk. Dividing by
# the measured ~9.1x speedup, n_conformer=1 puts that at roughly
# 1.4-2.9h/chunk -- --time is set to 6h for real margin over that new
# range (not the old 24-48h figures from when this was n_conformer=10).
# If actual throughput on BigRed200 comes in slower than this local
# estimate and tasks still hit the 6h wall, remember
# generate_conformers_for_chunk() commits to its LMDB shard every
# --commit-every (default 2000) molecules and skips already-done indices
# on restart -- a walltime kill costs at most one commit interval of
# work, not the whole chunk; just resubmit the same chunk-id.
#
# After every array task completes (or once enough have -- see above),
# merge the shards into the final cache (single process, not an array
# job). --merge-map-size-gb must fit the WHOLE pool, not one shard.
# --out-dir must match what generation used -- for --smiles-file mode
# this script now passes --out-dir explicitly (next to the input file),
# so the merge command must repeat that same --out-dir or it will look
# under Slate's results/embed/ and find nothing:
#   python -m src.representations.generate_conformers --target PARP1 --num-chunks 10 --merge
#   python -m src.representations.generate_conformers --smiles-file /N/project/SingleCell_Image/mengjing/enamine_real_1.56B/surA_smiles.txt --out-dir /N/project/SingleCell_Image/mengjing/enamine_real_1.56B/_conformer_output --num-chunks 1000 --merge --allow-partial --merge-map-size-gb 15000
#
# Storage: at the measured ~4.01KB/molecule (n_conformer=1), the full
# 1.56B pool needs ~6.1TB, not the ~13.9TB estimated for n_conformer=10.
# This script passes --map-size-gb 30 per shard (comfortable margin over
# the ~6GB/chunk this now implies) -- the original 10GB default (sized
# for n_conformer=10 chunks) was already undersized for that case and
# would have hit MDB_MAP_FULL partway through every chunk, not just slow
# ones; kept generous here since it's a sparse virtual ceiling on this
# Lustre filesystem, not real disk usage. Final Uni-Mol embeddings
# (512-dim float32) are unaffected by n_conformer (one embedding/molecule
# either way): ~3.0TB for the full 1.56B -- fits comfortably on the same
# SingleCell_Image project storage; no need to wait on the (still worth
# filing separately) Slate quota increase for this pipeline.

set -euo pipefail

POOL_ARG="${1:?Usage: sbatch --array=0-N scripts/submit_conformer_gen.sh <TARGET-or-smiles-file> <NUM_CHUNKS> [TOTAL_COUNT]}"
NUM_CHUNKS="${2:?Usage: sbatch --array=0-N scripts/submit_conformer_gen.sh <TARGET-or-smiles-file> <NUM_CHUNKS> [TOTAL_COUNT]}"
TOTAL_COUNT="${3:-}"
CHUNK_ID="${SLURM_ARRAY_TASK_ID:?This script must be submitted with --array=0-$((NUM_CHUNKS - 1)) (SLURM_ARRAY_TASK_ID is unset)}"

source /N/slate/mengjing/miniconda3/etc/profile.d/conda.sh
conda activate al-eval

cd /N/slate/mengjing/repos/al-eval-framework
mkdir -p logs

# Same thread-oversubscription guard as run_al.py/submit_unimol_extract.sh --
# without it, BLAS/OpenMP threads compete with the num-workers conformer
# processes for CPU (observed directly: a 49-thread near-hang at <1% CPU
# utilization on an unrelated run).
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# BigRed200 (HPE Cray) srun does not reliably inherit --cpus-per-task from
# the sbatch allocation for CPU-binding purposes without this -- srun
# otherwise aborts at launch with "CPU binding outside of job step
# allocation" (observed directly on job 7766000).
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

OUT_DIR_FLAG=()
if [[ -f "$POOL_ARG" ]]; then
    POOL_FLAG=(--smiles-file "$POOL_ARG")
    # generate_conformers.py's default --out-dir for --smiles-file mode is
    # NOT next to the input file -- it's ROOT/results/embed/{stem}, i.e.
    # always under this repo on Slate, regardless of where POOL_ARG lives.
    # That silently sent ~714GB of shards to the 800GB Slate quota in a
    # prior run of this exact script (job 7803319, "Disk quota exceeded"
    # partway through) even though POOL_ARG pointed at Scratch. Force
    # --out-dir to sit next to the input file explicitly so this can't
    # silently recur for any future large library staged off-Slate.
    OUT_DIR_FLAG=(--out-dir "$(dirname "$POOL_ARG")/_conformer_output")
else
    POOL_FLAG=(--target "$POOL_ARG")
fi

COUNT_FLAG=()
if [[ -n "$TOTAL_COUNT" ]]; then
    COUNT_FLAG=(--total-count "$TOTAL_COUNT")
fi

srun --cpu-bind=none python -m src.representations.generate_conformers \
    "${POOL_FLAG[@]}" "${COUNT_FLAG[@]}" "${OUT_DIR_FLAG[@]}" --chunk-id "$CHUNK_ID" --num-chunks "$NUM_CHUNKS" \
    --num-workers 16 --map-size-gb 30 --n-conformer 1
