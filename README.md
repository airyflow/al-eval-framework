# al-eval-framework

Evaluation framework for testing whether static prediction performance
(RMSE) reliably predicts active-learning (AL) sample efficiency, across
frozen molecular representation families (fingerprints, sequence, graph
topology, 3D geometry) on DOCKSTRING docking targets. See [DESIGN.md](DESIGN.md)
for the full design rationale, and [BILLION_SCALE_PIPELINE.md](BILLION_SCALE_PIPELINE.md)
for the separate billion-scale (Enamine REAL) conformer/embedding pipeline,
which is independent of the DOCKSTRING pipeline documented below.

## 1. Clone

```bash
git clone git@github.com:airyflow/al-eval-framework.git
cd al-eval-framework
```

## 2. Environment

Requires conda and a CUDA-capable GPU (CPU works but is slow for the
embedding-extraction step).

```bash
conda env create -f environment.yml
conda activate al-eval
pip install -e .
```

`environment.yml` pins `torch==2.3.1+cu121` and `transformers==4.44.2`
deliberately — `transformers>=4.45` requires `torch>=2.4` and silently
disables its PyTorch backend against 2.3.1 (observed directly in a sibling
repo's shared environment). Don't bump `transformers` without also bumping
torch, or MoLFormer loading will fail with "PyTorch was not found."

## 3. Pretrained backbone checkpoints

`models/` is gitignored (large binaries) and must be populated before
running embedding extraction:

```
models/
├── grover/grover_base.pt
├── molformer/                    # full HF snapshot dir
└── unimol/mol_pre_all_h_220816.pt
```

If you already have these downloaded elsewhere on the same machine (e.g.
in a sibling `FusionAL` checkout), the fastest path is to symlink or copy
that `models/` directory here rather than re-downloading. Otherwise, fetch
each backbone directly:

```bash
# GROVER (Google Drive, via gdown)
pip install gdown
python -c "import gdown; gdown.download(id='1hiGwOzoRfbJQPWj0V_mtOffsqIIAMgjl', output='models/grover/grover_base.pt')"

# Uni-Mol (Hugging Face)
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='dptech/Uni-Mol-Models', filename='mol_pre_all_h_220816.pt', local_dir='models/unimol')
"

# MoLFormer (Hugging Face, full snapshot)
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='ibm-research/MoLFormer-XL-both-10pct', local_dir='models/molformer', local_dir_use_symlinks=False)
"
```

## 4. DOCKSTRING data

`data/` is also gitignored. DOCKSTRING pool/oracle CSVs are built on first
use by `src/data/dockstring.py`, which normally auto-downloads the raw
dataset from figshare via the `dockstring` pip package.

**On this machine, that auto-download is blocked by an AWS WAF
bot-challenge.** If you hit that, download the raw TSV manually in a
browser (the `dockstring` package's figshare link) and stage it before the
first run:

```bash
mkdir -p data/dockstring
# place the downloaded file as either of:
#   data/dockstring/dockstring-dataset.tsv
#   data/dockstring/dockstring-dataset.tsv.gz
```

`src/data/dockstring.py` checks for this staged file automatically and
skips the blocked download if present. Once built, each target gets its
own cached, subsampled pool at `data/dockstring/{target}.csv.gz` (built
once, reused after).

## 5. Quickstart: one target end-to-end

Four DOCKSTRING targets are configured (`configs/target/`): `EGFR`,
`ESR2`, `F2`, `PARP1`. Six representations are configured
(`configs/representation/`): `morgan`, `molformer`, `unimol`, `grover`,
`fusional_lite`, `fusional_full`. Sweep parameters (rounds, budget, seeds,
acquisition function, surrogate hyperparameters, metrics) live in
`configs/experiment.yaml`.

```bash
# 1. Extract frozen embeddings for a target (all 3 backbones; add
#    --backbone molformer|unimol|grover to run just one)
python -m src.representations.extract --target PARP1

# 2. Run the AL loop for one (representation, seed) pair against that target
python -m src.al.run_al --representation morgan --seed 0
# smoke-test overrides, to check the plumbing without waiting on a full
# sweep: --max-iters 1 --epochs 5

# 3. Compute static-eval metrics (RMSE/ECE) + C1-C4 diagnostic criteria
python -m src.metrics.run_static_and_criteria --representation morgan --seed 0

# 4. Generate the decoupling figure (static RMSE rank vs. AUDC rank) and
#    recall curves, once step 2-3 have been run across the representation
#    grid in configs/experiment.yaml
python -m src.analysis.decoupling
python -m src.analysis.recall_curves

# 5. (optional) Per-(target, representation, seed) embedding-geometry
#    diagnostics (PCA/TwoNN/density/neighborhood stability)
python -m src.analysis.run_geometry --target PARP1 --representation morgan --seed 0
```

Each script's `--help` documents its full flag set; `run_al.py`'s
`--target`/`--acquisition`/`--tag` let you sweep beyond
`configs/experiment.yaml`'s fixed protocol without overwriting the
baseline results other analyses depend on.

## 6. Repository layout

```
al-eval-framework/
├── configs/
│   ├── target/              # one yaml per DOCKSTRING target
│   ├── representation/      # one yaml per representation family
│   └── experiment.yaml      # rounds, budget, seeds, acquisition, surrogate hparams
├── data/dockstring/         # cached pool/oracle CSVs (gitignored, built on first use)
├── models/                  # pretrained backbone checkpoints (gitignored, see §3)
├── src/
│   ├── data/dockstring.py           # load_pool(), load_oracle(target)
│   ├── representations/extract.py   # frozen-encoder embedding extraction
│   ├── al/run_al.py                 # AL loop driver (wraps molpal.explorer.Explorer)
│   ├── metrics/                     # static RMSE/ECE + AUDC + C1-C4 criteria
│   └── analysis/                    # decoupling figure, recall curves, embedding geometry
├── molpal/                  # vendored MolPAL (Explorer, acquisition metrics, criteria.py)
├── muben/                   # vendored MUBen backbone library
└── results/                 # run outputs (gitignored)
```

See [DESIGN.md](DESIGN.md) for why each vendored piece was reused as-is
vs. rebuilt, and what the decoupling figure / criteria table are actually
meant to show.
