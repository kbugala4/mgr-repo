# Worker setup and experiment execution

Updated 2026-11-07. Start with the [documentation index](README.md) and [implementation status](implementation-status.md) for accepted decisions, architecture and qualification boundaries. The intended complete-study worker is Linux x86-64 with Python 3.11 and an NVIDIA GPU. CPU execution supports small pilots and the ten trainable methods; the accepted TabPFN integration requires CUDA. No machine is assumed capable of every dataset/configuration without a resource pilot.

## Install and check

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), clone the chosen committed revision, and enter its root. Create a fresh environment on each worker; do not copy a virtual environment from another path or machine. The environment was resolved using uv 0.11.16; keep `uv.lock` and use `--frozen` rather than independently upgrading scientific packages.

```bash
bash scripts/bootstrap.sh
make check
```

Bootstrap installs all eleven methods and development checks, then runs `thesis doctor`. The configuration defaults to CUDA. For a CPU-only development machine:

```bash
bash scripts/bootstrap.sh --set runtime.device=cpu
uv run --frozen thesis --set runtime.device=cpu --set runtime.num_cpus=1 smoke
```

With the default fold setting, the smoke command uses 180 synthetic rows, logistic regression and a two-epoch MLP, with one default plus one random configuration and eight folds. Expect four configuration jobs, 32 fold fits and six D/T/T+E rows. It fetches no real datasets or pretrained weights. It prints `results/runs/smoke/STUDY_HASH`; repeat identical settings to verify resume, or pass that actual directory to `thesis report DIRECTORY`. Smoke replaces the study name, model/dataset selection, all model overrides and random-configuration count with its synthetic preset. It retains `inner_folds`, seeds, runtime and paths; changing `inner_folds` changes the fit count. The complete dependency environment is still required. The CPU bootstrap flag affects doctor execution, not dependency selection or saved YAML defaults. A successful CPU smoke/doctor does not qualify the full eleven-method GPU matrix.

`doctor` performs a CUDA tensor allocation and matrix operation when CUDA is requested. It does not repair drivers. The 2026-11-06 verification host could not initialize its installed CUDA runtime; that historical observation does not diagnose another worker. Actual GPU training/timing qualification remains open. The tested PyTorch wheel is 2.13.0+cu130. CUDA 13 requires an R580-or-newer compatible driver; NVIDIA's compatibility guide documents limits and PTX caveats. Verify the final worker with `doctor` rather than assuming a version number alone guarantees compatibility. [NVIDIA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html), [PyTorch CUDA support matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md).

The default resource policy assigns one GPU to neural methods and four CPU threads to each job. Conventional references use CPU. Folds and configurations execute sequentially. Set `CUDA_VISIBLE_DEVICES` before starting a worker to select its GPU. CPU thread limits apply to Torch and existing BLAS/OpenMP pools. `runtime.prediction_batch_size` controls local MLP/FT/TabNet inference. Local MLP/FT validation and inference batches are capped by both that limit and the fit batch size, avoiding a larger attention-memory requirement at evaluation time. Native KNN/RealMLP/ModernNCA/TabM and fixed TabPFN do not receive this setting; qualify their own batching and memory policies separately. The 3,600-second configuration limit is a configurable, provisional library budget, not a hard process deadline; individual native fits may overrun it. Calibrate it before the final freeze.

Local MLP, FT-Transformer and TabNet retain completed training when their advisory fold budget stops further epochs. Each fold's `hyperparameters_fit` records `budget_truncated`, `completed_epochs`, `granted_seconds`, `fit_seconds_wrapper` and `effective_batch_size` in `records/result.json`. A deadline reached before any completed epoch still raises `TimeoutError`. TabNet honours its stop only after completing the current epoch and validation, so it may overrun that fold's grant. Preprocessing, validation, inference and native bagging checks can still exhaust the available configuration budget; truncation does not guarantee that every configuration succeeds.

FT-Transformer also caps the fit batch to a power of two using a 4 GiB allowance for one float32 attention-score tensor (`heads × (features + 1)² × 4` bytes per row), with a minimum cap of eight rows. Validation and inference inherit the effective fit-batch ceiling. This heuristic does not bound total VRAM: attention blocks, gradients and other workers also consume memory, and the eight-row floor can exceed the allowance on extremely wide inputs. Inspect each fold's `effective_batch_size` and qualify the GPU workload before rerunning it.

## Prepare datasets

Dataset definitions live in `configs/dataset/*.yaml`; the study selection lives in `configs/config.yaml`.

| Source | Datasets | Acquisition |
| --- | --- | --- |
| OpenML | Adult, Covertype, Credit Fraud, HIGGS subset, Amazon, Helena, Bioresponse | Explicit dataset ID and version; original feature metadata preserved |
| UCI | German Credit, Ionosphere, Online Shoppers, Diabetes 130-US | Source archive downloaded once and parsed without extracting arbitrary paths |
| Kaggle competition | Santander Customer Satisfaction | Official training CSV supplied locally after the user satisfies competition access requirements |

Acquire Santander from the [official competition](https://www.kaggle.com/competitions/santander-customer-satisfaction/data). Place the extracted file at `data/raw/santander/train.csv`; the runner excludes `ID` and uses `TARGET`. Competition access may require a Kaggle account and acceptance of its rules. The runner does not accept those rules or substitute a different dataset.

```bash
uv run --frozen thesis prepare
```

For a small public-data check:

```bash
uv run --frozen thesis --set 'datasets=[ionosphere]' prepare
```

Preparation writes `data/prepared/NAME/data.parquet` and `manifest.json`. The manifest records source settings, normalized-file SHA-256, rows, and columns. Subsequent preparation verifies and reuses the cache. A changed definition or corrupt cache fails explicitly. Files are committed atomically under a source lock. `prepare` checks required columns, class counts and outer-split feasibility; actual inner-fold validation is checked during the backend run. These checks do not certify absence of duplicates, target leakage or cohort errors. Diabetes uses `patient_nbr` for grouped splitting and removes both patient and encounter identifiers from predictors.

OpenML, Hugging Face, and library caches live under `data/cache/`. Prepared data contain raw feature values and declared feature types. Imputers, encoders, scaling, and learned model preprocessing fit the relevant training fold. IID KNN optimized LOO fits preprocessing on the outer development population while excluding each row's label from its neighbor prediction. Grouped KNN uses eight training-fold-specific transformations. The fixed TabPFN path may fit preprocessing on the entire outer development set because it has no inner training/search stage.

For final reproducibility, copy the verified prepared directories to other workers or freeze their `expected_sha256` values in dataset YAML. An `expected_sha256` policy can be added after preparation without redefining the cached source; it verifies the normalized Parquet file. UCI YAML `version: 1` is not an immutable upstream archive version. A first download without an expected hash establishes a local snapshot, not a guarantee that an upstream UCI archive can never change. Full duplicate, provenance, leakage, and cohort audits remain required before thesis execution.

## Prepare the foundation checkpoint

```bash
uv run --frozen thesis weights
```

This fetches only `Prior-Labs/tabpfn_3_5` revision `06bf2ba35c80a92a3b9abb436b99cf49e7a0365e`, file `tabpfn-v3.5-20260909.safetensors` (876,027,932 bytes). Its SHA-256 is:

```text
ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3
```

The default destination is `data/weights/`. Relative checkpoint filenames resolve under `paths.data/weights`; an absolute configured checkpoint path stays absolute. The `weights` command requires `tabpfn` in the selected model list. The package and checkpoint have separate access/licensing requirements. If access is requested, complete the provider's process yourself and provide `HF_TOKEN` / `TABPFN_TOKEN` as applicable. Do not put credentials in YAML. Fetching a file does not establish that a license-gated inference call has been qualified. [Pinned model repository](https://huggingface.co/Prior-Labs/tabpfn_3_5/tree/06bf2ba35c80a92a3b9abb436b99cf49e7a0365e), [TabPFN package access instructions](https://pypi.org/project/tabpfn/).

The runner verifies checkpoint presence/hash before any model fitting. It never silently replaces TabPFN-3.5 with another checkpoint. The integration's declared limits are one million context rows, 20,000 predictors, and 160 classes; passing those bounds does not establish that a particular context fits available VRAM. Full-size and 100-class GPU pilots remain mandatory.

## Plan, execute, resume, report

```bash
uv run --frozen thesis plan
uv run --frozen thesis run
uv run --frozen thesis report results/runs/thesis-lite/STUDY_HASH
```

`run` expects prepared, verified dataset caches and the selected foundation checkpoint; it does not implicitly download them.

`plan` prints the resolved typed configuration and nominal counts. It does not fetch data/weights, instantiate every random candidate or verify dataset/model feasibility. The main `run` does not include the two targeted mechanism studies: their runners remain pending.

The default twelve-dataset matrix contains 3,132 configuration jobs: 12 × (10 × 26 + 1). Most jobs train eight fold models; IID KNN uses native child LOO, and fixed TabPFN uses the complete outer development context. This is not a count of independent neural networks: RealMLP/TabM may internally train packed ensembles.

Global options must precede the command. Example pilot:

```bash
uv run --frozen thesis \
  --set 'models=[logistic_regression,mlp]' \
  --set 'datasets=[ionosphere]' \
  --set random_configurations=1 \
  --set runtime.device=cpu \
  --set runtime.num_cpus=1 \
  --set model_overrides.mlp.epochs=2 \
  run
```

This shortened pilot is for infrastructure validation, not a final model comparison. To change paths or budgets, pass repeated `--set KEY=VALUE` options or edit YAML. Unknown settings/model parameters fail validation. All resolved settings enter the manifest. Native KNN/RealMLP/ModernNCA/TabM search spaces come from pinned TabArena generators; their YAML `search_space` entries are not consumed by the catalog, even if populated. Fixed TabPFN also ignores `search_space`. Only the six local methods use YAML search spaces. For the local six methods, sampled searched values overwrite same-key defaults/overrides; for the native four, configured parameters are applied after sampling and stay fixed. A `model_overrides` value therefore does not freeze a searched parameter uniformly across methods. Inspect [the catalog](../src/tabular_thesis/experiments/catalog.py) before fixing a searched parameter.

Each configuration owns a locked job directory containing its identity, effective model/child parameters, seeds, timing/memory metadata, compressed aligned OOF/test probabilities, and completion checksums. Unique attempt records preserve retry/interruption history. Inner validation row IDs are saved. A failed attempt writes an explicit status and traceback. The runner removes temporary fit files and releases unused memory after each attempt, including failures. The runner continues independent configuration jobs and returns nonzero if any fail. Repeating an unchanged command verifies completed artifacts and retries unfinished/failed work. Interrupted work never counts as complete. Resume restarts the whole unfinished configuration; it does not resume individual fold checkpoints. Fitted model files are discarded, and changing study identity does not reuse previous models automatically.

Automatic reporting occurs only if `run` returns successfully. After failures, invoke `thesis report DIRECTORY` manually once all jobs have been attempted and required defaults succeeded. The directory is under `results/runs/STUDY/STUDY_HASH`; its artifact layout is described below.

Reporting requires every declared job to be completed or explicitly failed. Failed random configurations are excluded from selection and counted in the report. A failed required default prevents a complete D/T/T+E report. Tuned parameters and greedy ensemble weights are chosen from successful OOF predictions only. Fixed TabPFN has a default result, with no manufactured tuning or fold-ensemble result. Reports preserve task and condition identities; no old shared CSV is appended.

Study identity includes source/lock digest, selected library versions, resolved settings, input identities and hardware. Git commit and image digest are not recorded automatically: preserve them separately. Main secondary metrics are implemented, but threshold selection, calibration experiments, significance/uncertainty analysis, Elo and thesis plots are not supplied by the reporting CLI.

Do not combine inner folds as independent observations. Across-task inference uses datasets, and separate reporting conditions remain separate comparisons. Confirmatory contrasts, multiplicity rules, effect sizes, uncertainty, and the two targeted studies still need their final scientific freeze.

### Report costs and memory boundaries

The [report implementation](../src/tabular_thesis/experiments/reporting.py) defines these fields:

| Field | Meaning |
| --- | --- |
| `search_fit_seconds` | Sum of fit times from successful configuration records; excludes failed fits and previous attempts. Repeated on every condition row for the same task/method. |
| `search_wall_seconds` | Sum of recorded attempt wall times, including failures and retries. Null if any attempt remains `running` after interruption because its duration is unknown. Excludes study setup, final cleanup and report generation. |
| `selected_fit_seconds` | Sum of successful fit times for configurations with positive condition weights; not multiplied by those weights. |
| `selected_inference_seconds` | Sum of the selected configurations' timed test prediction passes. Does not include cached probability combination or metric computation. |
| `selection_seconds` | Time for the whole task/method's cached selection and greedy ensemble construction, repeated on its condition rows. Recomputed when regenerating reports. |

Do not sum repeated search/selection costs across D/T/T+E. `weights` aligns with `configuration_indices`, which can have gaps after failed random candidates. The inherited metric key `log_loss_val` holds outer-test log loss.

Memory stays in each job's `records/result.json`; it is not a report CSV column. TabArena 0.1.0 tracks memory around `fit`, not the test prediction pass: CPU RSS and PyTorch allocated/reserved GPU memory, with `gpu_tracking_enabled` indicating availability. These are not full-process VRAM or whole-attempt peak measurements. Qualify inference memory separately, especially for KNN and TabPFN. Evidence: the pinned `tabarena.benchmark.exec_models.base.AbstractExecModel.fit_custom` and `_collect_memory_usage`, plus `tabarena.utils.memory_utils`.

## Containers

The worker image installs the project after copying source and includes all model dependencies. Compose enables GPU access and mounts data/results persistently. Use a recent Docker Compose supporting `gpus`, a working NVIDIA driver, and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). [Docker GPU instructions](https://docs.docker.com/compose/how-tos/gpu-support/).

```bash
mkdir -p data results
export THESIS_UID=$(id -u) THESIS_GID=$(id -g)
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml run --rm worker doctor
docker compose -f docker/docker-compose.yml run --rm worker smoke
docker compose -f docker/docker-compose.yml run --rm worker prepare
docker compose -f docker/docker-compose.yml run --rm worker weights
docker compose -f docker/docker-compose.yml run --rm worker run
```

Mounts preserve the same cache/result layout as native execution. The image uses pinned Python/uv tags and the dependency lock; freeze its resulting digest for final experiments. Container build/execution remains unqualified. The 2026-11-06 verification host lacked Docker daemon access and the Compose plugin; check these prerequisites independently on each worker. These definitions are a deployment candidate, not a certified worker image.

For multiple machines, use independent output directories and explicit dataset/model selections. The runner prevents concurrent ownership of an identical local job; it is not a distributed scheduler or an artifact-merging service. Preserve each worker's manifest and hardware identity. Start with one representative GPU pilot, then size the full matrix from measured runtime, RAM, VRAM, and artifact volume.

## Compatibility and remaining gates

A dataset with a configured `time` field is rejected with `NotImplementedError`; temporal splitting is not implemented. Diabetes patient-group constraints are implemented. Any additional split adaptation requires an explicit design and implementation before final execution.

The old holdout/TPE trainer and heterogeneous CSV writer were retired. Existing legacy results are left untouched and are not automatically imported into the new schema. `scripts/run_benchmark.py` delegates to the new CLI; old flags must be replaced with the commands above.

Before final experiments: qualify all twelve raw datasets, actual TabPFN inference/access, representative GPU paths, wide/high-cardinality memory use, native preprocessing reproducibility, the container, measured budgets, and the two targeted intervention runners. See [implementation status](implementation-status.md).

`make doctor`, `make prepare`, `make weights`, `make plan`, `make run` and `make smoke` pass `ARGS` before the subcommand. Example: `make smoke ARGS="--set runtime.device=cpu"`. Reporting uses `make report STUDY=results/runs/smoke/STUDY_HASH`. `scripts/run_benchmark.py` accepts the new CLI surface; `scripts/generate_report.py DIRECTORY` regenerates reports.

## Artifact layout

```text
results/runs/STUDY/STUDY_HASH/
  manifest.json, index.json
  splits/DATASET.json
  jobs/DATASET-MODEL-000/
    manifest.json, status.json
    attempts/UUID.json
    records/result.json
    predictions/result.npz
    failure.json                 # if an attempt failed
  report.json, report.csv         # derived
  failures.json                   # if configurations failed
```

Prediction caches contain aligned ground truth, probabilities, row IDs and inner validation identities. Fixed TabPFN has no OOF block. Temporary fitted models are removed after each attempt. A failed or interrupted configuration is retrained as a whole; completed configurations are checksum-verified before reuse.
