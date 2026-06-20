# Project Recreation Manual

> **Last updated**: Phase 5 prep complete (benchmark runner, scaling, report generator)
> This manual is updated after every phase.

## Prerequisites

- Linux or macOS (Windows with WSL also works)
- Python 3.11+ available on the system
- [uv](https://github.com/astral-sh/uv) package manager installed
- Docker (optional, for containerized runs)

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Step 1: Create project directory and initialize

```bash
mkdir mgr && cd mgr
uv init --no-readme --python 3.11
rm -f main.py
```

## Step 2: Create directory structure

```bash
mkdir -p src/tabular_thesis/data
mkdir -p src/tabular_thesis/models/non_dl
mkdir -p src/tabular_thesis/models/dl
mkdir -p src/tabular_thesis/training
mkdir -p src/tabular_thesis/evaluation
mkdir -p src/tabular_thesis/utils
mkdir -p configs/dataset configs/model
mkdir -p scripts notebooks tests
mkdir -p docker docs
mkdir -p data/raw data/processed
mkdir -p results/mlruns results/figures
mkdir -p plans ans
```

## Step 3: Create empty `__init__.py` files

```bash
# src/tabular_thesis/__init__.py should contain version string (see below)
touch src/tabular_thesis/data/__init__.py
touch src/tabular_thesis/models/__init__.py
touch src/tabular_thesis/models/non_dl/__init__.py
touch src/tabular_thesis/models/dl/__init__.py
touch src/tabular_thesis/training/__init__.py
touch src/tabular_thesis/evaluation/__init__.py
touch src/tabular_thesis/utils/__init__.py
touch tests/__init__.py
```

`src/tabular_thesis/__init__.py`:
```python
"tabular_thesis — Deep Learning Algorithms for Tabular Data."
__version__ = "0.1.0"
```

## Step 4: Files to rewrite manually

### 4.1 Root configuration files

| File | Description | Lines |
|------|-------------|-------|
| `pyproject.toml` | Project metadata and all dependencies | ~60 |
| `.python-version` | Contains: `3.11` | 1 |
| `.gitignore` | Git ignore rules | ~50 |
| `.env.example` | Environment variable template | ~3 |
| `Makefile` | Build/run commands | ~45 |
| `README.md` | Project overview | ~70 |

### 4.2 Data layer

| File | Description | Lines |
|------|-------------|-------|
| `src/tabular_thesis/data/registry.py` | DatasetMeta, PreparedData dataclasses, enums | ~70 |
| `src/tabular_thesis/data/loader.py` | Dataset catalogue (6 datasets), OpenML loader | ~150 |
| `src/tabular_thesis/data/preprocessor.py` | MinimalPreprocessor — imputation + type detection only | ~155 |
| `src/tabular_thesis/data/splitter.py` | SplitRatios, prepare_data function | ~85 |

### 4.3 Model layer — base and registry

| File | Description | Lines |
|------|-------------|-------|
| `src/tabular_thesis/models/base.py` | BaseModel ABC with preprocessing_time_sec tracking | ~115 |
| `src/tabular_thesis/models/registry.py` | Lazy model registry, get_model() factory | ~75 |

### 4.4 Non-DL model wrappers (6 models)

| File | Description | Preprocessing |
|------|-------------|--------------|
| `src/tabular_thesis/models/non_dl/logistic_regression.py` | LogisticRegression | StandardScaler + OrdinalEncoder |
| `src/tabular_thesis/models/non_dl/knn.py` | KNN | StandardScaler + OrdinalEncoder |
| `src/tabular_thesis/models/non_dl/random_forest.py` | RandomForest | OrdinalEncoder only |
| `src/tabular_thesis/models/non_dl/xgboost_model.py` | XGBoost | OrdinalEncoder only |
| `src/tabular_thesis/models/non_dl/lightgbm_model.py` | LightGBM | OrdinalEncoder only |
| `src/tabular_thesis/models/non_dl/catboost_model.py` | CatBoost | Raw DataFrames (native categoricals) |

### 4.5 DL model wrappers (8 models — ALL implemented)

| File | Description | Lines | Status |
|------|-------------|-------|--------|
| `src/tabular_thesis/models/dl/mlp.py` | MLP — vanilla DL baseline with PyTorch training loop | ~210 | ✅ |
| `src/tabular_thesis/models/dl/ft_transformer.py` | FT-Transformer — Feature Tokenizer + Transformer encoder | ~280 | ✅ |
| `src/tabular_thesis/models/dl/tabnet_model.py` | TabNet — pytorch-tabnet wrapper | ~110 | ✅ |
| `src/tabular_thesis/models/dl/dcn_v2.py` | DCN v2 — deepctr-torch wrapper | ~155 | ✅ |
| `src/tabular_thesis/models/dl/scarf.py` | SCARF — contrastive pretraining + fine-tuning | ~260 | ✅ |
| `src/tabular_thesis/models/dl/node.py` | NODE — differentiable oblivious decision tree ensembles | ~230 | ✅ |
| `src/tabular_thesis/models/dl/igtd.py` | IGTD — feature-to-pixel mapping + CNN | ~260 | ✅ |
| `src/tabular_thesis/models/dl/reg_dnns.py` | Regularized DNNs — MLP with regularization cocktail | ~250 | ✅ |

### 4.6 Evaluation (includes local results storage)

| File | Description | Lines |
|------|-------------|-------|
| `src/tabular_thesis/evaluation/metrics.py` | MetricResult (with all timing), compute_metrics, Timer | ~100 |
| `src/tabular_thesis/evaluation/statistical.py` | Wilcoxon, Friedman tests | ~65 |
| `src/tabular_thesis/evaluation/visualization.py` | Comparison plots, scaling plots | ~85 |
| `src/tabular_thesis/evaluation/results_store.py` | Local CSV + JSON results storage (always saved) | ~95 |

### 4.6 Training infrastructure

| File | Description | Lines |
|------|-------------|-------|
| `src/tabular_thesis/training/tuner.py` | Optuna HP tuner | ~105 |
| `src/tabular_thesis/training/trainer.py` | Unified trainer with timing | ~155 |

### 4.7 Evaluation

| File | Description | Lines |
|------|-------------|-------|
| `src/tabular_thesis/evaluation/metrics.py` | MetricResult (with timing), compute_metrics, Timer | ~100 |
| `src/tabular_thesis/evaluation/statistical.py` | Wilcoxon, Friedman tests | ~65 |
| `src/tabular_thesis/evaluation/visualization.py` | Comparison plots, scaling plots | ~85 |

### 4.8 Utilities

| File | Description | Lines |
|------|-------------|-------|
| `src/tabular_thesis/utils/reproducibility.py` | set_seed, get_device | ~55 |
| `src/tabular_thesis/utils/logging.py` | setup_logging | ~30 |
| `src/tabular_thesis/utils/mlflow_utils.py` | MLFlow helpers (optional) | ~60 |
| `src/tabular_thesis/utils/config.py` | Path constants | ~10 |

### 4.9 Hydra configs (21 files)

| Directory | Files |
|-----------|-------|
| `configs/` | `config.yaml` |
| `configs/dataset/` | `adult.yaml`, `credit_fraud.yaml`, `higgs.yaml`, `mushroom.yaml`, `diabetes.yaml`, `helena.yaml` |
| `configs/model/` | `mlp.yaml`, `ft_transformer.yaml`, `tabnet.yaml`, `dcn_v2.yaml`, `scarf.yaml`, `node.yaml`, `igtd.yaml`, `reg_dnns.yaml`, `logistic_regression.yaml`, `knn.yaml`, `random_forest.yaml`, `xgboost.yaml`, `lightgbm.yaml`, `catboost.yaml` |

### 4.10 Docker

| File | Description |
|------|-------------|
| `docker/Dockerfile` | Single-stage Python 3.11 image with uv |
| `docker/docker-compose.yml` | App + MLFlow services |

### 4.11 Scripts (all working)

| File | Description | Lines |
|------|-------------|-------|
| `scripts/run_experiment.py` | Single experiment runner (placeholder) | ~15 |
| `scripts/run_benchmark.py` | Full benchmark with resume support, CLI args | ~130 |
| `scripts/run_scaling.py` | HIGGS scaling experiment at 5 sizes | ~130 |
| `scripts/generate_report.py` | Summary table + comparison plots from CSV | ~160 |

### 4.12 Tests

| File | Description |
|------|-------------|
| `tests/conftest.py` | Shared fixtures |
| `tests/test_data.py` | MinimalPreprocessor, DatasetMeta, SplitRatios tests |
| `tests/test_metrics.py` | Metric computation, Timer tests |
| `tests/test_models.py` | Model registry tests |
| `tests/test_integration.py` | End-to-end: all 6 non-DL models train/predict/evaluate |

### 4.13 Documentation

| File | Description |
|------|-------------|
| `docs/setup.md` | Setup and usage guide |
| `docs/recreation_manual.md` | This file |

## Step 5: Install dependencies

```bash
uv sync --prerelease=allow --all-extras
```

## Step 6: Verify installation

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: **30 tests passed**.

## Step 7: Initialize git (optional, local only)

```bash
git init
git add -A
git commit -m "Project recreation"
```

> **Note**: Do NOT push to any remote repository without explicit permission.

---

## File count summary (code only, excluding .md files)

| Category | Files | Lines |
|----------|-------|-------|
| Python source (`src/`) | 26 | 3,942 |
| Scripts (`scripts/`) | 4 | 446 |
| Tests (`tests/`) | 7 | 610 |
| Hydra configs (YAML) | 21 | 188 |
| Docker | 2 | 57 |
| Root files (pyproject.toml, Makefile, .gitignore, .env.example) | 4 | 168 |
| **Total code to rewrite** | **64** | **5,411** |

> Plus 2 documentation files (.md) and `uv.lock` (auto-generated).
> The `.python-version` file contains just `3.11`.

---

## Quick verification checklist

- [ ] `uv sync --prerelease=allow --all-extras` completes without errors
- [ ] `.venv/bin/python -m pytest tests/ -v` shows **30 tests passed**
- [ ] `.venv/bin/python -c "from tabular_thesis.models.registry import list_models; print(list_models())"` prints 14 model names
- [ ] `.venv/bin/python -c "from tabular_thesis.data.loader import DATASET_CATALOGUE; print(list(DATASET_CATALOGUE))"` prints 6 dataset names
- [ ] `.venv/bin/python -c "from tabular_thesis.models.registry import get_model; m = get_model('xgboost'); print(m)"` prints `XGBoostModel(name='xgboost')`
- [ ] `.venv/bin/python -c "from tabular_thesis.models.registry import get_model; m = get_model('mlp'); print(m)"` prints `MLPModel(name='mlp')`