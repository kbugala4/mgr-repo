# Accepted evaluation plan: TabArena lightweight approach

Accepted and revised on **2026-11-05**. The initial main study uses the **twelve datasets in the accepted [dataset pool](dataset-selection.md) and seven DL plus four conventional methods**, with **one outer split, eight inner folds, and one default plus 25 random configurations per tunable method**. Two targeted studies of rare-class sensitivity and overfitting are also accepted. This replaces the earlier 16-dataset planning target. OpenML IDs/versions, dependency versions, checkpoint revision/hash, main seeds and ensemble settings are configured and pinned. Prepared input hashes, cohort rules, exact intervention settings and measured resource budgets still need final freezing; pinned integrations still need empirical qualification. The main runtime is now implemented; see [implementation status](implementation-status.md) for evidence and remaining qualification. The two targeted study runners are still pending.

## Accepted initial study

| Decision | Accepted plan |
| --- | --- |
| Dataset scope | Twelve named classification datasets are accepted in the [dataset pool](dataset-selection.md). Source routes, OpenML versions and executable split rules are configured. Full raw-data audits, final prepared hashes and cohort decisions remain pending; runs persist actual split manifests. |
| Methods | MLP, TabNet, FT-Transformer, RealMLP, ModernNCA, TabM, TabPFN-3.5; logistic regression, KNN, XGBoost, CatBoost. See [model selection](study-model-selection.md). |
| Outer evaluation | One predefined outer training/test split per dataset, shared by all methods. For eligible IID classification tasks, use the first fold of a declared stratified three-fold partition: approximately two-thirds development data and one-third test data. |
| Inner validation | Eight stratified folds within the outer training portion for ordinary tunable methods. Fit all learned preprocessing inside the respective inner training fold. |
| Search | One declared default plus 25 randomly sampled configurations: 26 candidates per tunable method. Pin search spaces and sampling seeds. The active runner uses seeded random sampling; the old TPE execution path was retired. |
| Reporting | Default, tuned, and tuned-plus-ensemble conditions, selected using inner validation predictions only. |
| KNN exception | IID tasks use native child LOO with preprocessing fitted on all outer development features; each row’s label is excluded from its neighbor prediction. This is not exact per-row preprocessing refitting. Diabetes uses eight patient-group folds. |
| Foundation-model exception | Evaluate one fixed TabPFN-3.5 local base checkpoint and declared inference settings. Use the outer training population as context for outer test prediction; do not apply the 26-configuration search or eight-fold prediction ensemble to this condition. |
| Primary metrics | ROC AUC for binary classification and log loss for multiclass classification. Report per-dataset scores and costs; aggregate rankings supplement these results. |
| Additional repetitions | No additional outer splits or full seed grid in the initial matrix. Any extension must be declared and budgeted before using test results to choose it. |
| Required mechanism studies | Bounded class-weighting and regularization/training-size experiments below. Their extra training-seed repetitions are separate from the main benchmark. |

This is a custom twelve-dataset study using TabArena's lightweight evaluation approach. It is not the complete official TabArena-Lite benchmark, nor an exact reproduction of every original model condition. In particular, the fixed TabPFN-3.5 condition differs from the original paper's tuned TabPFNv2 condition.

Revised **2026-11-11**: local MLP, FT-Transformer and TabNet stop at their advisory fold budgets while retaining usable completed training. They predict the cost of another epoch after each completed epoch; MLP/FT-Transformer also abandon a later partial epoch and restore the best completed checkpoint. TabNet finishes the current epoch before its native stop and best-weight restoration, so it can overrun by the remaining epoch and validation work. A deadline reached before any completed epoch still fails explicitly. Fold `hyperparameters_fit` records `budget_truncated`, `completed_epochs`, `granted_seconds`, `fit_seconds_wrapper` and `effective_batch_size`. Native methods, dependency pins, seeds, fold counts and the configuration budget are unchanged. This budget fix changes source identity; existing results remain in their original directories.

Revised **2026-11-12**: MLP, FT-Transformer and TabNet use `batch_size: auto`, resolved independently within each inner training fold. A searched multiplier `{0.5, 1, 2}` replaces the former fixed batch lists; the declared default multiplier is 1. MLP/FT-Transformer use training-row count. TabNet weights row count by the sum of its fold-local encoder cardinalities (including reserved unknown-category slots): `n′ = n × (1 + total_cardinality / 5000)`. These are declared engineering heuristics, not empirically qualified optimal batches.

| MLP/FT-Transformer training rows n | Base batch |
| --- | ---: |
| n < 2,000 | 64 |
| 2,000 ≤ n < 8,000 | 128 |
| 8,000 ≤ n < 32,000 | 256 |
| 32,000 ≤ n < 128,000 | 512 |
| n ≥ 128,000 | 1024 |

| TabNet effective rows n′ | Base batch | Automatic virtual batch |
| --- | ---: | ---: |
| n′ < 2,000 | 128 | 128 |
| 2,000 ≤ n′ < 8,000 | 256 | 128 |
| 8,000 ≤ n′ < 32,000 | 1024 | 128 |
| 32,000 ≤ n′ < 128,000 | 2048 | 256 |
| n′ ≥ 128,000 | 4096 | 512 |

Automatic batches are multiplied, rounded down to a power of two and bounded below by eight. Explicit integer batch requests ignore the multiplier. FT's attention cap and TabNet's singleton-batch correction still apply. TabNet's automatic virtual batch comes from the table without multiplying it; explicit virtual batches must be integers ≥2. The virtual batch is clamped to the effective physical batch, then increased if needed to avoid singleton BatchNorm chunks in full or remainder batches. Each fold records `effective_batch_size` and, for TabNet, the corrected `effective_virtual_batch_size`.

FT-Transformer caps its effective batch using a 4 GiB allowance for a single float32 attention-score tensor, rounded down to a power of two with an eight-row floor. At 1,776 predictors this caps an eight-head model at 32 rows and a four-head model at 64 rows; validation/inference use the same ceiling. This is a memory heuristic, not a total-VRAM guarantee. The revised spaces replace random configurations for all three local methods on all twelve datasets (36 method/dataset pairings). Reruns are to be launched separately by the owner; they have not been executed as part of this change. Other methods, dependency pins, seeds, folds and configuration budgets are unchanged. Source/configuration identity changes, so existing cached results remain in their original directories.

## Selection, ensembling, and caching

For each ordinary tunable method and dataset:

1. Reserve the outer test population and persist its row identities. Do not use its features to fit preprocessing, or its labels to tune configurations, select checkpoints, set thresholds, or learn ensemble weights.
2. Evaluate each of the 26 configurations on the same eight inner folds. Cache aligned out-of-fold validation probabilities, fold-averaged outer test probabilities and actual validation row identities. Fold models are temporary and are deleted after the attempt. Keep ModernNCA's reference bank and labels inside each inner training fold.
3. **Default:** average the eight default fold models' test probabilities.
4. **Tuned:** choose the configuration with the best inner validation score, including the default among the candidates. Average that configuration's eight fold models' test probabilities.
5. **Tuned plus ensemble:** select configurations and learn ensemble weights using inner out-of-fold validation predictions only. Apply the frozen weights to their corresponding fold-averaged test probabilities. The implementation pins AutoGluon greedy ensemble selection to size 100 and seed 42; validate resource use before final execution.
6. Report all conditions without selecting the preferred condition from outer test scores. Reuse cached predictions and metadata; these reporting conditions do not require three independent training sweeps or an additional final refit for ordinary fold ensembles.

IID KNN uses one native child-LOO fit per configuration and no added final-refit stage. Grouped KNN uses eight fold fits. One-hot encoding is fixed, while native time-budget subsampling may reduce the neighbor bank; effective rows and neighbor counts are recorded. Report the fixed TabPFN condition once; tuned and tuned-plus-ensemble conditions are not applicable to it under this plan. Any checkpoint-internal inference ensemble is part of its recorded fixed configuration.

Cache dataset and split identities, ground truth, validation/test probabilities, full resolved configurations, search seeds, fold seeds, model/checkpoint versions and hashes, executable source/lock digest, selected package versions, hardware, timings, peak memory, and trial completion/failure status. Record the Git revision and container image digest separately at the final freeze; the runner does not capture them automatically. Resume only compatible completed work. Report failures and timeouts explicitly; do not turn them into successful zero-score trials.

## Dataset validity and interpretation

- Source revision **2026-11-12**: Santander uses OpenML dataset 46859 version 2 (`TARGET`, 369 predictors, no `ID` column), replacing local-CSV acquisition. All twelve accepted datasets now download through OpenML or UCI. Source hashes, row order and split identities must be audited for the new mirror; existing local-source predictions are not automatically equivalent or reused. The method pool, folds, seeds and search budgets are unchanged.
- Audit source identity, duplicates, target leakage, class counts, and group/temporal structure for all twelve accepted datasets before freezing executable inputs. Each class must have enough examples for the declared outer and inner partitions.
- Diabetes implements patient-group constraints at both outer and inner levels. Temporal splitting is not implemented: a configured `time` field raises `NotImplementedError`. If audits require another group or temporal prediction claim, implement and freeze that adaptation before execution; do not silently apply IID stratification.
- Fit learned preprocessing on the relevant training partition, with the declared native IID KNN feature-preprocessing exception. Fixed TabPFN uses the full outer development context. Use justified model-specific transformations.
- The main report implements accuracy, balanced accuracy, macro/weighted F1, log loss, AUC, AP and per-class metrics/confusion matrices where defined. Binary aggregate AP uses encoded class 1; targeted studies must identify the rare outcome explicitly. Main hard predictions use `argmax`. Threshold selection and calibration studies remain pending. The inherited report key `log_loss_val` contains outer-test log loss.
- Use datasets as the comparison units across tasks. Fold models, configurations, and reporting conditions are not additional independent datasets. One outer split does not measure sensitivity to alternative outer splits.
- Predeclare primary comparisons, multiplicity handling, effect-size summaries, and uncertainty methods. If Elo is reported, define its reference and settings rather than assuming the official benchmark's reference model is in this pool.

## Required targeted studies

The [thesis topic](thesis-topic.md) explicitly requires studying rare-class sensitivity and resistance to overfitting. Both studies below are accepted in scope. The bounded execution design specifies an initial task/model roster and counts; exact seeds, regularization values, operating points, and adapter support must be qualified and frozen before final evaluation. These are controlled variants of existing methods, not additions to the eleven-method pool.

### A. Rare-class sensitivity

- **Tasks:** Credit Card Fraud, Santander Customer Satisfaction, and Amazon Employee Access. Define the rare outcome and probability column explicitly; Amazon's rare outcome may be the negative raw label.
- **Methods:** MLP and TabM, with CatBoost as the conventional reference, subject to qualifying their loss/weight interfaces.
- **Conditions:** ordinary training versus class-weighted training. For each training fold compute balanced weights as \(w_c=N/(K n_c)\), where N is its training-row count, K is the number of classes, and n_c is its count for class c. Apply the same declared weighting rule across compatible methods. Weight training losses only; retain the original validation/test class distributions and use unweighted evaluation metrics.
- **Control:** fix each method/task's non-weighting hyperparameters using its main benchmark inner-validation selection. Refit both conditions with matched folds and three paired training seeds. Do not select a configuration using outer test scores or run another 26-configuration search for each variant. This measures the effect of weighting conditional on those fixed hyperparameters; it does not compare separately optimized weighted pipelines.
- **Outcomes:** average precision for the declared rare class, minority precision/recall at validation-selected operating points, ROC AUC, and probability/calibration effects. Freeze the operating-point rule before final testing and apply it equally to both conditions.
- **Nominal work:** 3 tasks x 3 methods x 2 conditions x 3 seeds = **54 model-condition evaluations**. With eight inner folds, this is 432 fold fits: **288 neural GPU fits** and **144 CatBoost CPU fits**. Reuse the main outer split and inner fold identities, not outer test feedback.

### B. Small-data generalization and overfitting

- **Tasks:** Statlog German Credit, Ionosphere, and Bioresponse, covering naturally small mixed/numerical data and a wide source.
- **Method:** MLP with a pinned, competent published recipe fixed independently of these tasks' full-label tuning results. This prevents full-budget hyperparameter selection from silently informing a smaller training-size condition.
- **Conditions:** the competent regularized baseline, the same baseline with dropout disabled, and the same baseline with weight decay disabled. Choose a reference baseline with nonzero dropout and weight decay; freeze its values after implementation qualification. Change one regularizer at a time, keeping architecture, preprocessing, stopping rule, and other settings matched. The ablated networks do not replace the main benchmark's competent MLP.
- **Training sizes:** nested stratified subsets using 25%, 50%, and 100% of each inner training fold, with three paired training seeds. Keep inner validation and outer test populations unchanged, and record actual rows and per-class counts. Report the fixed validation-label budget: this is training-size sensitivity, not total annotation-budget efficiency or active learning.
- **Outcomes:** primary task performance, learning curves, and training/held-out generalization gaps measured with the same evaluation metric. Interpret each fold model's gap separately from the eight-fold ensemble's test performance.
- **Nominal work:** 3 tasks x 3 regularization conditions x 3 training sizes x 3 seeds = **81 model-condition evaluations**, or **648 neural GPU fold fits**. No extra hyperparameter search is planned for these fixed-recipe ablations.

Together the initial targeted design adds **135 model-condition evaluations**, **936 neural GPU fold fits**, and **144 CPU fold fits** if every condition completes. These are planning counts; reuse is permitted only for exactly matching completed work. Freeze any pilot-driven feasibility adjustment before final test evaluation and document the change.

High-dimensionality evidence comes from Bioresponse and its training-size conditions, complemented by the main benchmark. Extra irrelevant-feature or representation experiments are optional extensions. Active learning remains optional/future work. Neither is an automatic third required experiment block.

## Workload and resource qualification

For ten tunable methods, the main matrix contains **12 x 10 x 26 = 3,120 configuration jobs**, plus **12 fixed TabPFN outer evaluations**. Each ordinary configuration job contains eight fold fits; optimized KNN leave-one-out validation is an exception. The six trainable neural methods account for **12 x 6 x 26 x 8 = 14,976 GPU fold fits** if all jobs complete. There are **132 method/dataset pairings**. Default reporting, including fixed TabPFN, contributes 132 records; tuned and tuned-plus-ensemble each contribute 120, for **372 reporting records** if all applicable conditions succeed. These records reuse training work.

For planning, assign the six trainable neural methods to one GPU per job and the four conventional methods to CPUs. GPU use for tree methods would require a separate declared allocation and cost estimate. Keep TabPFN inference costs separate from per-task neural training; do not imply that its pretrained weights were trained within this budget.

| Mean runtime for an entire eight-fold neural configuration job on one GPU | Neural GPU-hours | Continuous time on one GPU |
| --- | ---: | ---: |
| 1 minute | 31.2 | 1.3 days |
| 5 minutes | 156 | 6.5 days |
| 15 minutes | 468 | 19.5 days |

These are conditional arithmetic scenarios, not measurements or a hardware-specific forecast. At a five-to-fifteen-minute mean, a provisional 20% operational reserve gives approximately **187-562 GPU-hours (8-23 continuous days), plus TabPFN inference** for the main benchmark. CPU work, the required targeted studies, and additional outer splits are excluded from this hour estimate. The targeted studies add 936 neural fold fits, bringing nominal planned neural fitting to **15,912 fold fits** before pilots/retries; their job runtimes need separate measurement. The discussed RTX PRO 6000 Blackwell Max-Q has 96 GB VRAM according to NVIDIA; its runtime on this workload has not been measured.

Before booking the full run, pilot three contrasting datasets with all six trainable neural methods, using the default plus two random configurations: **54 eight-fold configuration jobs**. Also qualify the fixed TabPFN condition on representative sizes. Measure end-to-end runtime, actual GPU use, peak VRAM, CPU/RAM requirements, inference time, and artifact size on the intended machine. Replace the scenarios with a dataset- and model-weighted estimate.

The configured provisional TabArena-style limit is one hour per configuration job across all eight inner folds, not one hour per fold or per entire search. Qualify and freeze its implementation after the pilot. Pin the framework version, search spaces, ensemble settings, and split manifest; current TabArena code may differ from the original paper's repetition policy below.

## Implementation status and next steps

Updated 2026-11-06: the main runner uses TabArena 0.1.0 and the pinned compatible AutoGluon release, actual fold models, seeded random configurations, validated cached predictions, and OOF-only selection/greedy ensembling. The thesis profile explicitly disables automatic tiny-data 5×5 validation to keep eight folds and one repeat. Grouped KNN uses eight group-constrained fits; IID KNN uses native optimized LOO with preprocessing fitted on outer development data. KNN fixes nominal categoricals to one-hot encoding rather than searching arbitrary ordinal/drop encodings. Its time-budget neighbor-bank subsampling, when used, is recorded.

Outer/model seed 42 and search seed 42 are configured. Native inner split/child seeds and actual validation rows are persisted. TabPFN package/checkpoint revision/hash are pinned. The one-hour configuration budget is provisional and advisory rather than a hard process deadline. Current source, tests, and remaining scientific/hardware gates are documented in [implementation status](implementation-status.md).

1. Audit the twelve accepted datasets, pin exact versions/hashes and feature roles, and freeze their prediction/split manifests.
2. Preserve the tested framework/dependency pins and qualify the actual worker/container.
3. Complete the remaining raw-data/checkpoint/hardware gates in the [implementation status](implementation-status.md).
4. Retain the passing small binary/multiclass, group-isolation, failure, cache, and report regressions; extend qualification to representative full-size tasks.
5. Qualify the two targeted studies, including weighting, regularization controls, subset isolation, paired seeds, and declared metrics. Freeze their exact execution settings and separate budget.
6. Run the target-machine pilot, freeze the execution manifest and resource allocation, then execute the main matrix and the two targeted studies.

## TabArena reference: original full and lightweight protocols

The following summarizes reference literature and is not the active thesis configuration. The accepted defaults and exceptions above control this repository. This documentation audit did not refresh the external sources.

TabArena combines curated tabular datasets, predefined splits, evaluation code, and a leaderboard. The original TabArena v0.1 suite contains 51 datasets. Its protocol targets independent and identically distributed (IID) prediction tasks; temporal or grouped data may require different splits.

### Reference only: full TabArena protocol

For each dataset and model:

1. Evaluate on repeated, 3-fold outer cross-validation. Datasets with fewer than 2,500 rows use 10 repeats (30 outer splits); larger datasets use 3 repeats (9 outer splits). Classification splits are stratified.
2. Within each outer training split, evaluate hyperparameters using 8-fold inner cross-validation. The original study generally evaluated one default and 200 randomly sampled configurations per model, with model-specific exceptions. Each configuration had a one-hour limit per outer training split.
3. Select configurations using inner validation predictions. Average the eight inner-fold models' predictions on the outer test split. The tuned-and-ensembled variant also combines configurations using weights learned from inner validation predictions.
4. Score the outer test predictions: ROC AUC for binary classification, log loss for multiclass classification, and RMSE for regression. Aggregate pairwise model comparisons across datasets using Elo; also report per-dataset scores, because Elo does not show effect sizes.

Never use outer test data for preprocessing, tuning, or ensemble selection. Use the same outer splits for all models.

### Reference only: TabArena-Lite

The first version of TabArena-Lite uses all 51 datasets but only the first outer split of each. It retains the official inner validation protocol. For cheaper initial evaluation, the maintainers suggest the default configuration and optionally about 25 random configurations. A run restricted to fewer official-suite datasets is an official-suite subset. This thesis instead uses its own selected twelve sources, so it is described as a custom adaptation, not an official subset claim.

### Reference only: approximate fit counts

One fit trains one configuration on one inner fold. Counts assume all configurations complete, with no extra refits:

| Protocol | Outer splits | Configurations | Inner folds | Fits per model and dataset |
| --- | ---: | ---: | ---: | ---: |
| Full, fewer than 2,500 rows | 30 | 201 | 8 | 48,240 |
| Full, 2,500 rows or more | 9 | 201 | 8 | 14,472 |
| Lite, default only | 1 | 1 | 8 | 8 |
| Lite, default + 25 random | 1 | 26 | 8 | 208 |
| Lite, default + 200 random | 1 | 201 | 8 | 1,608 |

Actual counts vary with model support, failures, time limits, and implementation. For multiple datasets, sum their counts for each model.

### Reference only: using a dataset subset

Select datasets before comparing results and document selection criteria, dataset IDs, versions, and splits. Record benchmark and model versions, tuning budgets, hardware, and deviations from the official protocol. Report per-dataset metrics and runtime alongside any aggregate. Claims apply to the selected datasets; fewer datasets reduce coverage, while fewer splits increase sensitivity to the particular split.

Sources: [TabArena paper](https://papers.neurips.cc/paper_files/paper/2025/file/1697e3fb412da11dc9488249f9e7bbc9-Paper-Datasets_and_Benchmarks_Track.pdf), [TabArena repository and current submission guidance](https://github.com/autogluon/tabarena), [custom-dataset runner example](https://github.com/autogluon/tabarena/blob/main/examples/benchmarking/run_quickstart_tabarena_custom_datasets.py), [current repetition helper](https://github.com/autogluon/tabarena/blob/main/packages/tabarena/src/tabarena/benchmark/task/metadata/fetch_metadata.py), [NVIDIA RTX PRO 6000 Blackwell Max-Q specifications](https://www.nvidia.com/en-eu/products/workstations/professional-desktop-gpus/rtx-pro-6000-max-q/).
