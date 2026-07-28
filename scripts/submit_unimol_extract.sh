#!/bin/bash

#SBATCH -J unimol_extract
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH -o unimol_extract_%j.txt
#SBATCH -e unimol_extract_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH -A r00939

# -p gpu / --gpus-per-node=1 confirmed against IU KB0022436 directly (Big
# Red 200's single GPU partition is named "gpu", 4x A100/node, plain
# --gpus-per-node=N with no type prefix needed since BigRed200 only has
# one GPU type, unlike Quartz's gpu/hopper split). Check partition status
# first with `sinfo -p gpu` if a submission hangs in queue.
#
# Conformer generation itself is still CPU-bound RDKit work (~3h,
# unaffected by the GPU) -- the GPU only accelerates the Uni-Mol
# transformer forward pass that runs after conformer generation
# completes, observed to otherwise run silently (no progress bar) and
# slowly on CPU-only.
#
# Usage: sbatch scripts/submit_unimol_extract.sh <TARGET>
# e.g.:  sbatch scripts/submit_unimol_extract.sh PARP1

set -euo pipefail

TARGET="${1:?Usage: sbatch submit_unimol_extract.sh <TARGET>, e.g. PARP1}"

source /N/slate/mengjing/miniconda3/etc/profile.d/conda.sh
conda activate al-eval

cd /N/slate/mengjing/repos/al-eval-framework

# --cpus-per-task=8 gives headroom for extract.py's 4 conformer-generation
# worker processes (num_preprocess_workers=4) without oversubscribing --
# each worker's own BLAS/OpenMP threads are separately capped to 4 inside
# src/al/run_al.py-style thread limiting; set the same guard here since
# extract.py doesn't import that module.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# BigRed200 (HPE Cray) srun does not reliably inherit --cpus-per-task from
# the sbatch allocation for CPU-binding purposes -- without this, srun
# aborts at launch with "CPU binding outside of job step allocation"
# (observed directly: job 7766000 failed here before any real work ran).
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

srun --cpu-bind=none python -m src.representations.extract --target "$TARGET" --backbone unimol
