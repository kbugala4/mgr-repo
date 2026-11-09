# Deep Learning for Tabular Data Classification Tasks

See the [documentation index](docs/README.md) for the accepted study, worker instructions and implementation limits. [AGENTS.md](AGENTS.md) records repository development rules.

Master's thesis experiment environment. The accepted pool is seven neural methods (MLP, TabNet, FT-Transformer, RealMLP, ModernNCA, TabM, TabPFN-3.5) and four conventional references (logistic regression, KNN, XGBoost, CatBoost), evaluated on twelve source datasets.

The runner implements the accepted TabArena Lite adaptation: one outer partition, eight inner folds, default plus 25 random configurations, and default/tuned/tuned-plus-ensemble reports from cached predictions. KNN uses native leave-one-out validation on IID tasks and group-constrained folds on grouped tasks. TabPFN uses one pinned local checkpoint with the full outer development context.

## Run on a worker

Use Linux x86-64, Python 3.11, [uv](https://docs.astral.sh/uv/getting-started/installation/), and a working NVIDIA driver for the complete study. The locked PyTorch distribution includes CUDA; the host still needs a compatible driver. GPU requests fail explicitly when CUDA is unavailable.

After acquiring Santander's competition CSV and configuring any required model access, run from the repository root:

```bash
bash scripts/bootstrap.sh
uv run --frozen thesis prepare
uv run --frozen thesis weights
uv run --frozen thesis plan
uv run --frozen thesis run
```

`prepare` downloads eleven public datasets from explicit OpenML dataset IDs/versions or live UCI archives. Santander requires `data/raw/santander/train.csv` from the official competition. Downloads become verified Parquet caches with source manifests and SHA-256 hashes. `weights` fetches the exact TabPFN-3.5 revision and verifies its published file hash. Neither operation runs training.

`run` writes `report.json` and `report.csv` and prints the study directory when all configuration jobs succeed. Repeating the same command reuses compatible completed jobs and retries failed jobs. Failed configurations retain tracebacks; any failure makes the command exit nonzero and skips automatic reporting. Manual reporting can exclude explicitly failed random configurations once every job was attempted; it rejects failed defaults or incomplete work. Changing the resolved configuration, data identity, code, dependencies, or recorded hardware produces another study directory.

See [setup](docs/setup.md) for CPU smoke checks, overrides, checkpoint access, containers, and moving caches between workers. The complete GPU matrix, all raw-data audits, and the two targeted mechanism studies still require qualification. Current evidence is in [implementation status](docs/implementation-status.md).

## Structure

```text
configs/                         Authoritative typed study/model/dataset YAML
src/tabular_thesis/data/          Verified source cache and task/outer-partition ownership
src/tabular_thesis/models/        Local OOP adapters and fold-local preprocessing
src/tabular_thesis/experiments/   Configuration/search-space service, TabArena bridge,
                                 resource policy, immutable artifacts, runner, reports
src/tabular_thesis/evaluation/    Validated metrics and analysis helpers
scripts/                         Bootstrap and compatibility entry points
tests/                          Offline regression and integration checks
docker/                         Locked worker image and GPU Compose service
docs/                           Study decisions, evidence, and operational instructions
```

The central parameter layer is an in-process service, with no network server required. TabArena/AutoGluon own fold training and optimized KNN validation. Reference packages supply FT-Transformer, RealMLP, ModernNCA, TabM, and TabPFN; local adapters handle task-specific integration.

## Development

```bash
make setup
make check
uv run --frozen thesis --set runtime.device=cpu smoke
```

With the default fold setting, smoke uses 180 synthetic rows, LR and a two-epoch MLP, default plus one random configuration, and eight folds: four jobs, 32 fold fits and six report rows. It downloads no datasets or pretrained weights. For a CPU worker, use `bash scripts/bootstrap.sh --set runtime.device=cpu`; this selects CPU execution for doctor while installing the same locked dependencies, including the CUDA-enabled PyTorch wheel on Linux x86-64. Results appear under `results/runs/smoke/STUDY_HASH`; rerunning identical settings resumes them. These checks are infrastructure evidence, not model-ranking results.

[Thesis topic](docs/thesis-topic.md), [model selection](docs/study-model-selection.md), [datasets](docs/dataset-selection.md) and [evaluation protocol](docs/tabarena-evaluation.md) define the study. [Implementation status](docs/implementation-status.md) separates implemented behavior from remaining qualification.
