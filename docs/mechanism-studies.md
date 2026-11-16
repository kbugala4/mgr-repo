# Mechanism studies: execution and qualification

The [evaluation protocol](tabarena-evaluation.md#required-targeted-studies) owns the accepted A/B design and numerical analysis rules. The plans in `configs/studies/` do not change main-study YAML, searches, dependency/checkpoint pins or method scope.

**State on 2026-11-15:** implementation and synthetic CPU checks are available. Real-data pilots, GPU feasibility, campaign-worker identity, original OOF source-selection evidence and full A/B execution are not locally qualified. Credit Fraud/CatBoost and Santander/CatBoost have no supplied T configuration and remain blocked. Configuration 21 and the other four A configurations use full-precision values supplied in the handoff; their historical selection is not claimed to have been independently reproduced.

## Plan and qualify

Use the existing frozen environment. Do not install or synchronize another environment without authorization. Standard commands below use `uv run --frozen`; in a pre-existing environment where synchronization is prohibited, use `PYTHONPATH=src .venv/bin/python` in place of `uv run --frozen python`.

```bash
uv run --frozen python scripts/run_mechanism_studies.py configs/studies/study-a.yaml --dry-run
uv run --frozen python scripts/run_mechanism_studies.py configs/studies/study-b.yaml --dry-run
make check
uv run --frozen thesis --set runtime.device=cpu smoke
```

Dry run reads no datasets and trains nothing. A expands to 36 evaluations (12 blocked); B expands to 108. B dry-run epochs are declared expectations, not evidence from data. Before real B execution, the launcher loads only an already-prepared dataset, reconstructs pinned native development folds, checks subset rounding and step budgets, and compares actual fit counters afterward. Missing data raises an error; the launcher never downloads it.

Next worker qualification, requiring separately authorized real pilots:

```bash
uv run --frozen thesis doctor
uv run --frozen python scripts/run_mechanism_studies.py configs/studies/study-a.yaml \
  --only amazon/mlp --variant weighted --seed 0 --results results/qualification-ab
uv run --frozen python scripts/run_mechanism_studies.py configs/studies/study-a.yaml \
  --only amazon/catboost --variant weighted --seed 0 --results results/qualification-ab
uv run --frozen python scripts/run_mechanism_studies.py configs/studies/study-b.yaml \
  --only ionosphere/mlp --variant full --seed 0 --results results/qualification-ab
uv run --frozen python scripts/run_mechanism_studies.py configs/studies/study-b.yaml \
  --only ionosphere/mlp --variant no_both --seed 0 --results results/qualification-ab
```

The B commands cover all three fractions; `--fraction 0.25` restricts a pilot further. Record `CUDA_VISIBLE_DEVICES`, GPU load/sharing, driver, hardware, source revision and measured memory/time. Inspect real row counts, steps, selected states and absence of truncation. Compare A validation row IDs to the corresponding original main-run OOF IDs. CPU synthetic results do not establish these worker properties.

For source verification, `--verify-source /path/to/results/runs/thesis-lite` checks A parameters against the declared original job manifest. It does not itself prove that the candidate won OOF selection. To reconstruct selection without requesting test arrays:

```bash
uv run --frozen python scripts/select_mechanism_sources.py a \
  /path/to/main/single-dataset-run --model catboost --output results/analysis/source-a.json
uv run --frozen python scripts/select_mechanism_sources.py b \
  /path/to/adult-run /path/to/credit-fraud-run /path/to/higgs-run /path/to/amazon-run \
  /path/to/santander-run /path/to/online-shoppers-run /path/to/diabetes-run /path/to/helena-run \
  --output results/analysis/source-b.json
```

The A audit follows the main reporter: all 26 candidates must be attempted, the default must succeed, and explicitly failed random candidates are listed and excluded. The B rank reconstruction requires all 26 successful candidates on every source dataset. Both verify source artifact hashes. They access only `oof_y` and `oof_probabilities` members of legacy prediction archives; they do not request test labels/probabilities. B also verifies identical candidate parameters across datasets. An incomplete source candidate, a failed default, or a missing B candidate leaves the source audit unresolved; never guess T. The audit cannot reconstruct the date at which a historical selection rule was frozen. Review that evidence separately before presenting a predeclared-selection claim.

The refreshed handoff now includes the historical selection, budget and CatBoost scripts. Availability of their source code does not establish reproduction of their reported numbers. The historical B selector assigns every non-completed candidate the worst score (multiple failures share the average bottom rank), ignores `batch_size` when comparing candidate parameters, and pins eight full run IDs. The current audit deliberately requires completed candidates and exact parameter equality, accepts explicit run paths, and verifies artifact hashes. It is equivalent on that narrower valid input domain, not a drop-in replay of historical failures or differing batch settings. For historical reconstruction, match the supplied run IDs and inspect the post-batch-revision manifests first; do not relax these guards merely to obtain configuration 21. The [refresh revision report](archive/badania-ab-2026-11-15/raport-rewizji-po-odswiezeniu.md) records source IDs, comparisons and remaining evidence gaps.

Paired training seeds are recorded in `model_info.children_info.S1Fk.hyperparameters_fit.random_state`. Synthetic integration checks establish child seeds, unchanged validation row IDs across training seeds, and nested training subsets. The same numeric seed alone does not guarantee identical initialized tensors or batches across different folds/fractions: preprocessing can change input width and different row populations change batch contents. For an initialization comparison, tensor shapes and random-call order must also match; for batch pairing, row identities/order and batch settings must match. Do not infer GPU determinism from these CPU checks.

## Full execution after qualification and separate authorization

```bash
uv run --frozen python scripts/run_mechanism_studies.py configs/studies/study-a.yaml
uv run --frozen python scripts/run_mechanism_studies.py configs/studies/study-b.yaml
```

The A launcher completes available independent conditions, lists blocked/failed ones and exits nonzero when any remain. Use `--only DATASET/MODEL` for a selected pair. A source-plan amendment changes its provenance: use a separately named results root and retain the original registry rather than overwriting it. The launcher rejects changed provenance or a changed run identity on resume. Each registry maps a condition to at most one run. Completed job checksums are verified and incomplete attempts are retrained as whole configurations.

Do not resume the main campaign from the modified A/B source tree: source identity changes even when default numerical behavior remains equivalent. Preserve original main results and their manifests. The baseline equivalence check uses isolated temporary synthetic results, not campaign directories.

## Artifacts and failure evidence

Each condition uses `ExperimentRunner` directly, with no call to the main reporter. A/B's scoped TabArena hooks retain finite OOF/probability validation while deferring the outer-test metric (`metric_error: null`). The native pipeline still produces and validates test prediction arrays for later evaluation; it does not select from them or publish test scores.

- `results/runs/study-a/conditions.json` and `study-b/conditions.json`: locked, atomically replaced registries, plan/launcher SHA-256, Git revision, device visibility, conditions, states, directories and B preflight counters.
- `RUN/jobs/DATASET-MODEL-000/records/result.json`: effective per-child parameters, seed, class weights/counts, task-space training/validation row IDs, history, selection criterion, stopping reason and counters.
- `records/fold-ATTEMPT-SEED.json`: each completed fit's history written **before prediction**, surviving subsequent prediction/evaluation failure and removal of `work/`. The attempt ID links to `attempts/ATTEMPT.json`. Failed attempts remain identifiable; they are not complete results.
- `predictions/oof.npz`: OOF labels/probabilities/IDs and actual inner-validation IDs, with no test members. OOF diagnostics use this archive only.
- `predictions/result.npz`: aligned OOF and sealed test predictions for explicit later analysis.
- Completion checksums cover records, prediction archives and attempts. Metadata-only control validates JSON records without opening any prediction file.

The MLP history is a list of epoch records (`epoch`, `optimizer_steps`, `train_loss`, `val_loss`, `val_loss_weighted`, `param_norm`). `best_epoch` is one-based. CatBoost persists `validation_curves`, zero-based `best_iteration`, `tree_count`, selected loss and budget evidence. Nonfinite history entries become JSON `null`; probability validation still rejects unusable predictions. Raw histories are retained even when a later condition check fails. No fitted model is retained for new inference.

## Diagnose before opening test

```bash
uv run --frozen python scripts/analyze_study_a.py results/runs/study-a/conditions.json --control-check
uv run --frozen python scripts/analyze_study_a.py results/runs/study-a/conditions.json --diagnostics
uv run --frozen python scripts/analyze_study_b.py results/runs/study-b/conditions.json
```

A control reads only fold metadata. Its decision is unresolved unless all three seeds and 24 folds exist. OOF diagnostics also measure AP spread before test access. If control is triggered:

```bash
uv run --frozen python scripts/run_mechanism_studies.py configs/studies/study-a.yaml \
  --control amazon/mlp --control-decision results/analysis/study-a/control.json
```

The launcher binds the decision to the exact weighted registry entries. It cannot run an untriggered control. After controls, rerun OOF diagnostics on the final registry. Keep the original control decision; adding controls does not invalidate its unchanged weighted inputs. B analysis checks full budgets, nested/matched subsets and fold histories, then exports losses, gaps, plateau diagnostics and paired norms. No default analysis mode opens test members.

## Freeze and explicitly analyze test

An author/reviewer must establish whether test results have already been accessed. This session neither verifies that history nor approves a final freeze. A freeze JSON contains:

```json
{
  "analysis_version": "ab-v1",
  "reviewer": "REVIEWER_ID",
  "frozen_at": "ISO_TIMESTAMP",
  "prior_test_access": null,
  "registry_sha256": "SHA256_OF_FINAL_CONDITIONS_JSON",
  "script_sha256": {
    "analyze_study_a.py": "SHA256_OF_SCRIPT",
    "mechanism_analysis.py": "SHA256_OF_SHARED_ANALYSIS"
  }
}
```

Replace placeholders with reviewed evidence. `null` is intentionally not approval. The current pre-test gate requires a verified `false` for prior test access; an actual `true` requires an author/reviewer protocol decision distinguishing exploratory and predeclared analysis. For B, replace the script key with `analyze_study_b.py`. Freeze the numerical rules and script hashes before the first test analysis; do not regenerate a freeze automatically after an analysis edit.

```bash
uv run --frozen python scripts/analyze_study_a.py results/runs/study-a/conditions.json \
  --test --freeze /path/to/reviewed-a-freeze.json \
  --control-decision results/analysis/study-a/control.json \
  --diagnostics-file results/analysis/study-a/diagnostics.json
uv run --frozen python scripts/analyze_study_b.py results/runs/study-b/conditions.json \
  --test --freeze /path/to/reviewed-b-freeze.json
```

JSON and derived CSV outputs go to `results/analysis/study-a|study-b/`. Missing conditions, undefined metrics and undefined bootstrap replicates are explicit; no zero-filling or silent reduction to fewer paired seeds occurs. PR/histogram and learning-curve arrays are exported for later figures. Each output identifies its analysis version, scripts, registry and reviewer freeze. Review code changes before generating manuscript figures; manuscripts remain outside this repository.
