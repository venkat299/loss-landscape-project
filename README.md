## Loss Landscape Geometry Project

Objective: build a small but complete experimentation pipeline to study how the **loss landscape geometry** of MLP classifiers on synthetic datasets relates to **optimization dynamics**, **generalization**, and **architecture / optimizer choices**.  

High level:
- Train a matrix of MLPs (depth/width/activation/optimizer × datasets).
- Save checkpoints and metrics.
- Probe the loss landscape (interpolation, random slices, Hessian spectrum, sharpness, PCA, connectivity).
- Generate figures and Markdown reports summarizing the geometry and performance.

---

## 1. Setup

This project is designed to use [uv](https://github.com/astral-sh/uv) and Python 3.11+.

From the repo root:

```bash
# 1) Initialize uv project (if not already done)
uv init

# 2) Install dependencies
uv add torch numpy matplotlib scipy tqdm pytest
```

All commands below assume you run them from the repository root.

---

## 2. Running Experiments

### 2.1 Train the full experiment matrix

Train all (architecture × activation × optimizer × dataset) combinations with multiple seeds and save checkpoints + metrics:

```bash
uv run python -m project.experiments.run_full_matrix \
  --output-root reports/experiments
```

Key options (defaults are usually fine):
- `--n-train`, `--n-test`: dataset sizes per split.
- `--noise`: input noise for synthetic data.
- `--device`: `"cpu"` or `"cuda"` (auto-detected by default).
- `--epochs`, `--learning-rate`, `--batch-size`, `--lr-step-size`, `--lr-gamma`, `--momentum`.
- `--include-circles`, `--include-xor`: add extra datasets beyond moons.

Outputs are written under `reports/experiments/…`, one directory per configuration and seed, containing:
- `checkpoints/`: `init_epoch0.pt`, `final_epoch<E>.pt` (and optional mid-epoch checkpoints).
- `metrics.json`: per-epoch train/test loss & accuracy.
- `summary.json`: serialized `DatasetConfig`, `ModelConfig`, `TrainingConfig`, and seed.

### 2.2 Run all landscape probes + reports

Once training is done, run the analysis pass:

```bash
uv run python -m project.experiments.run_probes_and_reports \
  --experiments-root reports/experiments \
  --figures-root reports/figures \
  --reports-root reports
```

This will:
- For each trained run:
  - Regenerate the dataset from its saved config.
  - Load initial and final checkpoints.
  - Run:
    - linear interpolation (init → final) with per-layer normalized directions,
    - 1D & 2D random-direction slices,
    - Hessian spectrum (top‑k eigenvalues + Hutchinson trace estimate),
    - ε‑sharpness probes.
  - Save figures under:
    - `reports/figures/dataset=…/arch=…/act=…/opt=…/seed=…/…`
- For each (dataset, architecture, activation, optimizer) group across seeds:
  - Run linear connectivity paths between seeds (with neuron permutation alignment for 1-hidden-layer models).
  - Compute PCA over weights from initial/final checkpoints and evaluate a 2D loss grid in the PCA plane.
  - Save connectivity and PCA figures under:
    - `reports/figures/dataset=…/arch=…/act=…/opt=…/connectivity/`
    - `reports/figures/dataset=…/arch=…/act=…/opt=…/pca/`
- Generate top-level Markdown reports under `reports/`:
  - `summary.md`
  - `depth_study.md`
  - `width_study.md`
  - `activation_study.md`
  - `optimizer_study.md`
  - `connectivity_study.md`

The script is **idempotent**:
- Per-run probes are skipped if `hessian/spectrum.json` already exists for that run.
- Group-level connectivity/PCA are skipped if `pca/pca_surface.png` exists for that configuration.

You can re-run the command safely after interruptions to resume remaining work.

---

## 3. Project Structure (High Level)

```text
project/
  data/           # Synthetic datasets (moons, circles, Gaussians, XOR)
  models/         # MLP classifier and predefined architecture variants
  experiments/    # Training + orchestration scripts
  landscape/      # Loss landscape probing methods + visualizations
  utils/          # Shared configs, seeding, plotting helpers
  reports/        # Markdown report generation utilities

reports/          # Generated experiments, figures, and Markdown reports
tests/            # Unit and smoke tests
```

### 3.1 `project/data/`

- `datasets.py`:
  - Synthetic 2D classification datasets:
    - moons
    - circles
    - Gaussian clusters (2–4 clusters)
    - XOR-like
  - Each returns normalized `(x_train, y_train, x_test, y_test)` tensors, using training statistics.

### 3.2 `project/models/`

- `mlp.py`:
  - `MLPClassifier`: configurable fully-connected MLP with:
    - variable depth and width,
    - activations: ReLU, Tanh, GELU,
    - Xavier or He initialization depending on activation.
  - `get_predefined_model_config(variant, …)`: arch variants from `Tasks.md`:
    - `shallow-small` (1 × 50),
    - `shallow-wide` (1 × 500),
    - `deep-small` (4 × 100),
    - `deep-large` (4 × 250),
    - `medium` (2 × 100).

### 3.3 `project/experiments/`

- `train_model.py`:
  - Core training utilities:
    - optimizers: SGD (with momentum) and Adam,
    - StepLR learning-rate schedules,
    - deterministic seeding via `TrainingConfig.seed`,
    - logging of train/test loss & accuracy per epoch,
    - checkpoint saving (`init`, optional mid-epochs, `final`).
  - `train_model` and `evaluate_model` are reusable across scripts.

- `run_full_matrix.py`:
  - Builds the full experiment matrix over:
    - architectures,
    - activations,
    - optimizers,
    - datasets (moons + optional circles / XOR),
    - multiple seeds.
  - Generates datasets, trains models via `train_model`, and writes:
    - metrics (JSON),
    - summaries (JSON),
    - checkpoints.

- `run_probes_and_reports.py`:
  - Orchestrates all landscape probes and report generation, as described in §2.2.
  - Uses progress bars (via `tqdm`) and skip-logic to avoid recomputing finished runs/groups.

### 3.4 `project/landscape/`

Core probing methods:
- `interpolation.py`:
  - `linear_interpolation_curve`: loss/accuracy along θ_A → θ_B with optional per-layer normalized directions.
- `random_slice.py`:
  - `random_1d_loss_slice`: 1D random direction slice L(θ + αd), α ∈ [−δ, δ].
  - `random_2d_loss_surface`: 2D random directions (orthonormalized), grid over (α, β).
- `hessian.py`:
  - Hessian–vector products,
  - power iteration for top‑k eigenvalues,
  - Hutchinson estimator for trace.
- `sharpness.py`:
  - ε‑sharpness via randomized per-layer normalized perturbations inside a radius ball.
- `connectivity.py`:
  - linear connectivity barriers between modes,
  - simple neuron permutation alignment for 1-hidden-layer MLPs,
  - optional quadratic Bézier paths.
- `pca.py`:
  - collect weight trajectories from checkpoints,
  - compute PCA components,
  - project training paths into the PCA plane,
  - evaluate loss over a 2D PCA grid.

Visualizations:
- `visualizations/visualize.py`:
  - Turn numerical probe outputs into figures using `utils.plotting`:
    - interpolation curves,
    - slice surfaces + contours,
    - Hessian eigenvalue stem plots,
    - sharpness histograms,
    - PCA trajectories and surfaces,
    - connectivity curves with barrier annotations.

### 3.5 `project/utils/`

- `configs.py`: dataclasses for:
  - `DatasetConfig`, `ModelConfig`, `TrainingConfig`,
  - `InterpolationConfig`, `SliceConfig`, `HessianConfig`,
  - `SharpnessConfig`, `PCAConfig`, `ExperimentConfig`.
- `seed.py`:
  - `set_global_seed`: deterministic seeding for Python, NumPy, and PyTorch (including CUDA flags).
- `plotting.py`:
  - Small helpers for line plots, 3D surfaces, contours, histograms, stem plots, and 2D trajectories.

### 3.6 `project/reports/`

- `markdown.py`:
  - `generate_summary_report`: builds `summary.md` from experiment metrics.
  - `generate_study_reports`: builds:
    - `summary.md`,
    - `depth_study.md`,
    - `width_study.md`,
    - `activation_study.md`,
    - `optimizer_study.md`,
    - `connectivity_study.md`.
  - Aggregates run summaries and groups them by depth, width, activation, optimizer for concise tables.

---

## 4. Testing

Lightweight tests live under `tests/`:

```bash
uv run python -m pytest -q
```

Included tests cover:
- direction normalization and parameter flatten/unflatten,
- Hessian spectrum estimation on a tiny model,
- basic interpolation curve properties.

---

## 5. How the Pieces Fit Together

End-to-end flow:
1. **Data generation** (`project/data`) creates small synthetic classification problems.
2. **Model definition** (`project/models`) provides flexible MLP architectures in the 20k–100k parameter range.
3. **Training** (`project/experiments/train_model.py` + `run_full_matrix.py`) sweeps the experiment matrix and records metrics/checkpoints.
4. **Geometry probes** (`project/landscape`) analyze the loss landscape around trained solutions via interpolation, slices, Hessian, sharpness, PCA, and connectivity.
5. **Visualization** (`project/landscape/visualizations` + `project/utils/plotting.py`) converts probe outputs into interpretable plots.
6. **Reporting** (`project/reports`) aggregates metrics and figures into Markdown reports that summarize:
   - performance,
   - geometry of the loss landscape,
   - qualitative and quantitative differences across architectures and optimizers.

This structure makes it easy to rerun experiments, plug in new models or datasets, and extend the probing toolkit while keeping training, analysis, visualization, and reporting cleanly separated.
