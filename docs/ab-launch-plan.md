# A/B launch plan — 2026-11-16

All approved configurations are now available. A has 36 evaluations, B has 108. This document prepares execution; no real pilot or full study was started locally.

## Inputs and provenance

The delivered `results/final/final_summary.csv` contains 372 unique dataset/model/condition entries across 12 datasets and 11 methods. Its tuned weights identify Credit Fraud/CatBoost configuration 17 and Santander/CatBoost configuration 2. `final_configurations.csv` contains 261 unique model/index entries, including full-precision JSON parameters. These two configurations now populate `configs/studies/study-a.yaml`, with the corresponding source run IDs. All four previously supplied A configurations and B's base configuration 21 match the export exactly.

The final CSVs are supplied evidence. They do not independently establish source artifact checksums, historical selection-rule timing, or the reported mean rank 12.75. Original run directories are not present locally. Verify their manifests and OOF on the worker using the source-audit commands in [mechanism studies](mechanism-studies.md). The A launcher's `--verify-source results/runs/thesis-lite` checks exact parameters against source job manifests; use it when those directories are available. B requires its separate eight-source audit; do not pass A's `--verify-source` option to B.

The CSV exports remain unchanged. `final_sources.csv` includes trailing tab-only rows and tabs in its last column name; readers must strip whitespace and ignore empty dataset rows. There are 132 nonempty source rows. No test metric values were used to choose the A/B settings.

SHA-256 of the supplied exports used for this amendment:

| File | SHA-256 |
| --- | --- |
| `final_summary.csv` | `81797f29ee33bae270468320fe3c92b073d82a4d66da1038c565f8024272ca17` |
| `final_configurations.csv` | `4247caeb76ea72b6d25a45e9b7748ce9ae8f269c242e53656e3e0684d147d4c3` |
| `final_sources.csv` | `c5681fa9cbba83445385a76c80d70df32fe8b5044b9e4148e87b5a3183c6813d` |

## Worker preparation and pilots

Transfer the completed A/B implementation, plans and unchanged dependency lock to the worker. Record the exact source digest, Git state and script/plan hashes; HEAD alone does not identify these uncommitted changes. Preserve original main-study code and results. Use the existing frozen environment and verified prepared datasets, without downloading data or replacing caches.

From the repository root, check the worker and expand both plans:

```bash
uv run --frozen thesis doctor
uv run --frozen python scripts/run_mechanism_studies.py configs/studies/study-a.yaml --dry-run
uv run --frozen python scripts/run_mechanism_studies.py configs/studies/study-b.yaml --dry-run
```

Expected dry-run counts are A=36/blocked=0 and B=108/blocked=0. Before the full launch, run these pilots sequentially, using a new results root for this amended A plan:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/run_mechanism_studies.py configs/studies/study-a.yaml --only amazon/mlp --variant weighted --seed 0 --results results/ab-2026-11-16
uv run --frozen python scripts/run_mechanism_studies.py configs/studies/study-a.yaml --only amazon/catboost --variant weighted --seed 0 --device cpu --results results/qualification-ab-cpu-2026-11-16
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/run_mechanism_studies.py configs/studies/study-b.yaml --only ionosphere/mlp --variant full --seed 0 --results results/ab-2026-11-16
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/run_mechanism_studies.py configs/studies/study-b.yaml --only ionosphere/mlp --variant no_both --seed 0 --results results/ab-2026-11-16
```

CatBoost is assigned CPU by the backend even when the launcher requests CUDA. The explicit CPU pilot uses a separate root because device visibility and run identity are part of provenance. In the full A process, MLP uses CUDA and CatBoost uses CPU. B pilots cover all three fractions; `--fraction 0.25` can shorten the first diagnostic pass.

Check completed histories, class weights, semantic rare-class mapping (Amazon code 0), native validation row IDs against the main run, subset-only preprocessing, and absence of time truncation. B checks actual preflight counts and post-fit updates, including German Credit/25%=1002 updates. Pilot success on Ionosphere does not establish memory feasibility of Amazon or Bioresponse; inspect memory during the first full-size conditions and retain failures explicitly.

## Concurrent execution

A and B have distinct registries (`runs/study-a/conditions.json` and `runs/study-b/conditions.json`) and different run identities. They can run in separate processes using one common results root and prepared-data root. Within either study, one launcher owns the registry lock: do not split A or B into simultaneous launchers using that same registry.

With two qualified GPUs, use separate terminals:

```bash
# Terminal A
CUDA_VISIBLE_DEVICES=0 uv run --frozen python scripts/run_mechanism_studies.py configs/studies/study-a.yaml --results results/ab-2026-11-16
```

```bash
# Terminal B
CUDA_VISIBLE_DEVICES=1 uv run --frozen python scripts/run_mechanism_studies.py configs/studies/study-b.yaml --results results/ab-2026-11-16
```

Use the same GPU visibility for a study's pilot, full execution and resume: the registry records it. If B will run on GPU 1, run its pilots with `CUDA_VISIBLE_DEVICES=1` too. Repeating the same command resumes completed conditions with checksum verification. Do not edit the plan or launcher after starting a registry; changed provenance requires another results root.

With one GPU, run A then B sequentially by default. Sharing one GPU is technically possible but requires a concurrent memory/throughput pilot; GPU allocation is not coordinated across the two processes. Each process also requests four CPU threads by default, so simultaneous runs need CPU and RAM headroom. Separate CPUs for A's CatBoost work are already managed within its sequential launcher; no new scheduler is needed.

## After training

Run metadata control and OOF diagnostics for A, then any triggered three-seed selection controls. Run B history/budget diagnostics. See [mechanism studies](mechanism-studies.md) for exact commands, conditional controls and reviewer-freeze schema. Full A/B training does not automatically publish test metrics.

The main-study report already contains test metrics. Record who inspected them and when; do not assert that there was no prior test access. The author/reviewer must establish how that history relates to the predeclared A/B protocol before test analysis. The existing analysis gate must not be bypassed or satisfied with an invented declaration.

Nominal work is 144 evaluations, 1008 neural fold fits and 144 CPU CatBoost fits. Conditional A controls add at most 18 evaluations, 72 neural and 72 CPU fits. Runtime depends on the worker and must be measured; no duration forecast is established here.

## Local verification

The updated A plan and export were compared by parsed JSON/YAML values, without opening test prediction arrays. Relevant plan-count regression now expects zero blocked A conditions. Full offline checks and synthetic CPU smoke are recorded after verification in the implementation status. These checks do not qualify GPU execution or independently reproduce original OOF selection.
