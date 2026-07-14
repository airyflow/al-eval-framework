# AL Evaluation Framework — Design Doc

Status: draft v1 — 2026-07-14
Scope: implement the evaluation framework described in the paper draft
("What Makes a Molecular Representation Good for Active Learning?"),
starting fresh in a new repo, reusing proven pieces from `FusionAL` and
`benchmarking_molecular_models`.

---

## 1. Goal

Test one hypothesis: **static prediction performance does not reliably
predict active-learning sample efficiency**, across representation
families (fingerprints, sequence, graph topology, 3D geometry), and use
four diagnostic properties (C1 Local Smoothness, C2 Predictive
Calibration, C3 Exploration Diversity, C4 Target-aware Organization) to
explain why. FusionAL (frozen MolFormer + Uni-Mol + GROVER concatenation)
is the baseline that should win under the AL evaluation despite not
necessarily winning under static RMSE.

The output is not just FusionAL — it's the **decoupling figure**
(static RMSE rank vs. AUDC rank) plus the criteria table that explains
it. Everything in this doc is designed to produce those two artifacts
correctly before anything else.

## 2. What already exists (and what to do with it)

Two repos at `/N/slate/mengjing/repos/` contain reusable code. Neither is
the framework itself — both were built for narrower purposes, and this
project will consciously simplify rather than inherit their complexity.

### 2.1 `FusionAL` — AL loop, surrogates, diagnostics (primary source)

| Piece | File | Verdict |
|---|---|---|
| Diagnostic criteria C1–C4 | `molpal/analysis/criteria.py` | **Reuse near-verbatim.** Already matches the paper's equations exactly (Spearman smoothness, regression ECE, latent + Tanimoto diversity, kNN target precision). This is the single most valuable existing asset. |
| Acquisition functions | `molpal/acquirer/metrics.py` | **Reuse as-is.** UCB, EI, PI, Thompson, greedy, borda are all implemented and match the paper's `α(x) = μ + βσ`. |
| Real MolPAL orchestration | `molpal/explorer.py`, `molpal/acquirer/`, `molpal/pools/`, `molpal/objectives/`, `molpal/models/base.py` | **Reuse as the AL loop backbone** (per your "fork/wrap MolPAL" choice). This is the actual MolPAL `Explorer` — checkpointing, batch acquisition, `LookupObjective` for CSV-based oracles, pluggable `Model` ABC. It is *not* used by `run_al.py`, which reimplements a parallel, simpler loop from scratch (`MVEExplorer`/`MolPALExplorer`). Building on the real `Explorer` gets you checkpointing and a config file for free and avoids maintaining two AL loops. |
| Embedding→surrogate bridge | `molpal/models/mvemodels.py` (`EmbeddingMVEModel`) | **Reuse the pattern.** Clean `Model` ABC implementation: maps SMILES→embedding rows, delegates fit/predict to a surrogate object. This is exactly the adapter needed to plug frozen embeddings into the real `Explorer`. |
| Surrogate architectures | `surrogates.py` | **Do not port wholesale.** 10 ALSU variants (bigfusion/Borda, learned/Ridge meta, nonlinear/attention fusion, 4 scheduled fine-tuning variants) — all research scaffolding for a *different* prior question (which fusion strategy wins). The paper's spec is one architecture: MLP `d→512→128→1`, MC Dropout p=0.2, T=50, trained per-round. Build that one surrogate; the ALSU zoo becomes an optional Phase 2+ ablation, not launch scope. |
| Embedding extraction | `extract_embeddings.py`, `preprocess_unimol.py` | **Reuse the MUBen-backed extraction code, retarget the SMILES source.** Currently hardcoded to Enamine10k/50k/HTS libraries. Needs a thin adapter so it runs over a DOCKSTRING SMILES list instead (see §4.1). |
| Static RMSE/ECE eval, AUDC | — | **Does not exist yet.** `run_al.py` only tracks top-1% recall and best score per round; there is no static held-out snapshot, no AUDC integral, and no per-round static-vs-AL comparison. This is the actual gap this project fills — build it fresh (§4.5). |
| Oracle/pool format | `run_al.py` `load_oracle`/`load_library_smiles` | Enamine + custom docking CSVs, not DOCKSTRING. Replace with a DOCKSTRING adapter (§4.1); keep the `{smiles: score}` dict convention since `LookupObjective` already expects exactly this shape via CSV.

### 2.2 `benchmarking_molecular_models` — static representation benchmarking (secondary source)

This is Praski et al.'s own benchmark code (the paper this work positions
against). Relevant pattern, not code to import directly:

- **Pluggable `Embedder` interface** (`src/common/types.py`, `model_wrappers/*/wrapper.py`) — one wrapper per model family (fingerprints, ChemBERTa, MolFormer, GROVER/MAT via huggingmolecules, Uni-Mol, GraphMVP, CLAMP, COATI, GEM, ...), each in its own venv with its own `init.sh`. Config-driven via Hydra (`config/experiment/*.yaml`, `config/model/*.yaml`).
- **Embedding caching**: `joblib`-serialized `EmbeddedDataset` keyed by `(dataset, model)`, skips recomputation if cached (`src/embedding/embedding.py`).
- Worth adopting the *caching discipline and config-driven experiment naming* (dataset × representation × seed as a Hydra multirun grid) — not the 15-model wrapper zoo, since this project only needs 4 representation families.

### 2.3 Net decision

Build a new, smaller repo that:
1. Vendors/reuses `criteria.py` and `acquirer/metrics.py` unchanged.
2. Wraps the real `molpal.explorer.Explorer` (not `run_al.py`'s reimplementation) via a new `EmbeddingMLPModel` (MC Dropout, single architecture) modeled on `EmbeddingMVEModel`.
3. Retargets embedding extraction at DOCKSTRING instead of Enamine.
4. Adds the missing static-eval + AUDC + decoupling-analysis layer from scratch.
5. Uses Hydra-style configs (dataset/target × representation × seed) for the experiment grid, following the pattern in `benchmarking_molecular_models`, not `run_al.py`'s flat argparse CLI — this matters once you're running N targets × 5 representations × 5 seeds.

## 3. Repository layout

```
al-eval-framework/
├── configs/
│   ├── target/              # one yaml per DOCKSTRING target (PARP1.yaml, F2.yaml, ...)
│   ├── representation/      # morgan.yaml, molformer.yaml, unimol.yaml, grover.yaml, fusional.yaml
│   └── experiment.yaml      # rounds, budget, seeds, acquisition, surrogate hparams
├── data/
│   └── dockstring/          # cached DOCKSTRING SMILES + per-target scores (gzip csv)
├── src/
│   ├── data/
│   │   └── dockstring.py            # load_pool(), load_oracle(target) -> {smi: score}
│   ├── representations/
│   │   ├── extract.py               # frozen-encoder extraction, DOCKSTRING-targeted
│   │   ├── morgan.py                # fingerprint featurizer (reuse molpal.featurizer)
│   │   └── fusion.py                # l2-normalize + concat -> FusionAL embedding
│   ├── surrogate/
│   │   ├── mc_dropout_mlp.py        # d -> 512 -> 128 -> 1, p=0.2, T=50 (the ONE architecture)
│   │   └── model_adapter.py         # Model ABC bridge, modeled on EmbeddingMVEModel
│   ├── al/
│   │   ├── run_al.py                # drives molpal.explorer.Explorer with LookupObjective
│   │   └── objective_dockstring.py  # LookupObjective config generator per target
│   ├── metrics/
│   │   ├── static.py                # held-out RMSE, static ECE
│   │   ├── al_metrics.py            # Recall@k(t), AUDC, per-round AL-ECE
│   │   └── criteria.py              # vendored from FusionAL molpal/analysis/criteria.py
│   └── analysis/
│       ├── decoupling.py            # static-RMSE-rank vs AUDC-rank, Spearman rho, Figure 1
│       └── error_correlation.py     # pairwise surrogate error correlation, Figure 2
├── scripts/
│   ├── extract_embeddings.sh
│   ├── run_experiment_grid.sh       # sweeps target x representation x seed
│   └── make_figures.sh
├── results/
│   └── <target>/<representation>/<seed>/   # history.json, static_snapshot.json, criteria.json
└── molpal/  (vendored subset: explorer.py, acquirer/, pools/, objectives/, models/base.py)
```

`molpal/` is vendored (copied, not symlinked to `FusionAL`) so this repo
has no runtime dependency on the research repo — `FusionAL` keeps
evolving independently for the fusion-strategy exploration it's already
doing.

## 4. Component design

### 4.1 Data & oracle: DOCKSTRING

DOCKSTRING ships ~260K molecules docked against 58 targets with a fixed
train/test split (`pip install dockstring` gives programmatic access;
the raw TSV is also downloadable directly).

```python
# src/data/dockstring.py
def load_pool() -> list[str]: ...              # canonical SMILES, fixed order
def load_oracle(target: str) -> dict[str, float]:  # {smiles: docking_score}, lower=better
```

This mirrors `run_al.py`'s `load_oracle`/`load_library_smiles` contract
exactly, so `LookupObjective` (which expects a `{smiles: score}` CSV)
needs only a small `objective_dockstring.py` that writes a per-target
lookup CSV from `load_oracle(target)` and generates the config file
`LookupObjective.__init__` expects.

**Open question (needs your input before Phase 0 starts):** which
DOCKSTRING target(s) for the MVP — a well-studied one like PARP1 or F2 is
a reasonable default since docking score distributions there are
well-characterized in the DOCKSTRING paper.

### 4.2 Representations

| Family | Encoder | Source |
|---|---|---|
| Fingerprint | Morgan FP, 2048-bit | `molpal.featurizer.Featurizer` — already dataset-agnostic, no extraction step |
| Sequence | MolFormer, 768-d mean pool | `extract_embeddings.py::extract_molformer`, retargeted to DOCKSTRING SMILES |
| 3D geometry | Uni-Mol, 512-d CLS | `extract_embeddings.py::extract_unimol` + `preprocess_unimol.py` for conformers |
| Graph topology | GROVER, 300–1600-d mean pool | `extract_embeddings.py::extract_grover` |
| Multi-view | FusionAL | l2-normalize each of MolFormer/Uni-Mol/GROVER, concatenate |

Retargeting `extract_embeddings.py` means replacing
`load_molpal_smiles()` (which reads `molpal/libraries/{dataset}.csv.gz`)
with `load_pool()` from `src/data/dockstring.py`, and pointing
`OUTPUT_DIR` at `results/embed/dockstring/`. The MUBen backbone-loading
code (GROVER/Uni-Mol/MoLFormer model loading, batching, AMP) is otherwise
unchanged — it's dataset-agnostic already, it just needs a different
SMILES source injected.

Note `surrogates.py`'s `_LightweightBackbone` is sized for
GROVER=1600-d (both bond- and atom-view concatenated, per
`extract_grover`'s `torch.cat([mol_from_bond, mol_from_atom])`) —
carry that dimension convention forward, not the paper draft's stated
300-d GROVER, or the FusionAL concat dimension in Table 2 of the paper
will be wrong.

### 4.3 Surrogate

One architecture, matching the paper spec exactly (§3.2 of the draft):

```
d -> Linear(512) -> ReLU -> Dropout(0.2)
   -> Linear(128) -> ReLU -> Dropout(0.2)
   -> Linear(1)
```

MC Dropout at inference: T=50 stochastic forward passes, dropout left
active, `mu = mean(preds)`, `sigma = std(preds)`. This is simpler than
`surrogates.py`'s dual-MVE-head + Spearman-loss architecture
(`_DualMVEModel`/`CombinedLoss`) — that's a legitimate design choice from
the ALSU work, but it isn't the paper's stated architecture, and
introducing it here would confound "does representation matter" with
"does surrogate architecture matter." Keep the surrogate fixed and
minimal so every representation is compared through an identical model.

Wire it into the real `Explorer` via a `Model` ABC adapter
(`src/surrogate/model_adapter.py`), copying `EmbeddingMVEModel`'s
`_get_X`/`train`/`get_means_and_vars` pattern but backed by the MC
Dropout MLP instead of an ALSU surrogate object.

### 4.4 Acquisition

Reuse `molpal/acquirer/metrics.py::ucb` unchanged: `α(x) = μ(x) + β·σ(x)`,
β=1, matching the paper. The real `Explorer` already calls into this
through `molpal.acquirer.Acquirer`.

### 4.5 Metrics (the actual gap to fill)

**Static** (`src/metrics/static.py`) — new code:
- Static RMSE: train surrogate once on `|D0|=0.5%`, evaluate held-out RMSE.
- Static ECE: `criteria.py::expected_calibration_error(y, mu, sigma)` on the same held-out set.

**AL** (`src/metrics/al_metrics.py`) — extends what `run_al.py` already
tracks (`top1pct_recall`, `best_score`) with:
- `Recall@k(t)` generalized to arbitrary k, not just 1%.
- **AUDC** — does not exist anywhere in either source repo. Trapezoidal integral of the Recall@k(t) curve over `t ∈ [0, 0.05|X|]`. This is the paper's headline AL metric and needs to be implemented from scratch.
- **AL-ECE per round** — call `criteria.py::expected_calibration_error` once per AL round on that round's held-out predictions (the function already exists; it just isn't currently called inside an AL loop anywhere).

**Diagnostics C1–C4** (`src/metrics/criteria.py`) — vendor
`FusionAL/molpal/analysis/criteria.py` verbatim; it already implements
exactly the equations in the paper (§4.1 of the draft: Eq. smoothness,
regression ECE, latent/Tanimoto diversity, kNN precision@10).

### 4.6 Decoupling analysis (Figure 1 — the paper's central claim)

New (`src/analysis/decoupling.py`): for each representation, pair
(static RMSE, AUDC), rank both axes, compute Spearman ρ between the two
rankings, plot side-by-side rank panels connected by lines. This is the
one artifact that doesn't exist in any form in either source repo and is
the figure the whole paper hinges on — build and sanity-check it early,
even on the MVP's 3 representations, rather than leaving it until the
full sweep is done.

## 5. Phasing

### Phase 0 — MVP (prove the plumbing)

- **1 DOCKSTRING target** (TBD — recommend PARP1 or F2).
- **3 representations**: Morgan FP, MolFormer, FusionAL (MolFormer + Uni-Mol only, to cut GROVER extraction/conformer-generation cost out of the critical path).
- **1 acquisition function**: UCB, β=1.
- **3 seeds**, 5 AL rounds, `|D0|`=0.5%, batch=1%.
- **Metrics**: static RMSE/ECE, Recall@1%(t), AUDC, decoupling scatter (even at N=3 it validates the pipeline), C1 (smoothness) + C2 (calibration) only — skip C3/C4 until Phase 1.
- **Exit criterion**: the decoupling scatter renders with real numbers, AUDC is computed correctly (spot-check by hand against a printed Recall@k(t) curve), and the real `Explorer` checkpoints/resumes correctly.

### Phase 1 — scale out

- Add Uni-Mol standalone, GROVER standalone, and full 3-encoder FusionAL.
- Add C3 (exploration diversity) and C4 (target-aware organization).
- Expand to the full target set (N targets — TBD) and 5 seeds.
- Populate Table 1 (criteria) and Table `tab:main` (static vs. sequential) for real.

### Phase 2 — the rest of the paper

- Figure 2 (pairwise surrogate error correlation heatmap).
- Figure 3 (AL-ECE tracked across rounds).
- View ablation table (which of MolFormer/Uni-Mol/GROVER contributes).
- Cold-start sensitivity sweep (`|D0| ∈ {0.1%, 0.5%, 1.0%}`).

## 6. Open decisions before writing code

1. **DOCKSTRING target(s) for Phase 0** — pick one now so extraction/oracle code can be written against a concrete target instead of an abstract interface.
2. **Compute budget** — Uni-Mol conformer generation (`preprocess_unimol.py`) took "~10–40 min on CPU" for Enamine50k (50K molecules); DOCKSTRING's pool is ~260K, so this is the likely Phase 0 bottleneck. Worth deciding whether Phase 0 subsamples the DOCKSTRING pool (e.g. 50K) rather than using the full library, purely to keep iteration fast.
3. **GROVER checkpoint** — `models/grover/grover_{base,large}.pt` already exist in `FusionAL/models/`; confirm reuse vs. re-download for the new repo (they're large binary files — a symlink to `FusionAL/models/` avoids duplicating gigabytes, at the cost of a cross-repo dependency for weights only, not code).

## 7. Milestone checklist

- [ ] Confirm target(s) + compute budget (open decisions above)
- [ ] `src/data/dockstring.py`: pool + oracle loaders
- [ ] Retarget `extract_embeddings.py` at DOCKSTRING SMILES; extract Morgan (instant), MolFormer, Uni-Mol for the Phase 0 pool
- [ ] `src/surrogate/mc_dropout_mlp.py` + `model_adapter.py`
- [ ] Vendor `criteria.py`, `acquirer/metrics.py`, and the real `explorer.py` + its `acquirer/pools/objectives/models.base` deps
- [ ] `src/al/objective_dockstring.py` — LookupObjective config generator
- [ ] Run one full AL loop end-to-end (Morgan FP, 1 seed) through the real `Explorer` — verify checkpointing works
- [ ] `src/metrics/static.py`, `al_metrics.py` (AUDC is the new piece)
- [ ] `src/analysis/decoupling.py` — render Figure 1 on 3 representations
- [ ] Phase 0 exit criteria met → proceed to Phase 1
