# Accepted dataset pool for thesis experiments

Accepted on **2026-11-05**: **twelve classification datasets**, paired with the accepted seven DL and four conventional methods. This replaces the earlier 16-dataset planning target and the earlier twelve-candidate catalogue. The names, configured source routes and OpenML versions below are accepted. Full raw-data audits, final prepared hashes and cohort policies remain execution gates; executable feature roles and split rules are already configured.

The controlling scope is the [thesis topic](thesis-topic.md). The [scientific scope](implementation-status.md#scientific-scope) explains the coverage decision; the [evaluation plan](tabarena-evaluation.md) owns validation, tuning, reporting, and the two required mechanism studies. Runtime update 2026-11-06: all twelve source routes are configured. Four UCI download/parsing routes have been exercised, including patient-group outer preflight. Dataset acquisition remains distinct from the full scientific audit; see [implementation status](implementation-status.md).

## Accepted twelve datasets

Sizes below are source/reference expectations except for German Credit, Ionosphere, Online Shoppers and Diabetes, whose prepared row/predictor counts are observed in the [verification summary](implementation-status.md#verification). Predictor counts exclude the target and intended identifier exclusions; actual retained columns must be recorded after the schema audit. A known row count does not establish model compatibility, difficulty, or independence.

| # | Dataset | Intended source | Expected rows | Expected predictors | Classes | Main scientific role |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | Adult Income | OpenML dataset 1590 | 48,842 | 14, mixed | 2 | Mixed-feature income classification reference |
| 2 | Covertype | OpenML dataset 1596 | 581,012 | 54, numerical/binary | 7 | Large multiclass task and scalability |
| 3 | Credit Card Fraud | OpenML dataset 1597; audit source equivalence | 284,807 | 29 numerical (`Time` excluded) | 2 | Severe minority-class imbalance |
| 4 | HIGGS subset | OpenML dataset 23512 | 98,050 | 28, numerical | 2 | Balanced numerical physics reference |
| 5 | Amazon Employee Access | OpenML dataset 4135 | 32,769 | 9, categorical | 2 | High-cardinality categoricals and imbalance |
| 6 | Santander Customer Satisfaction | OpenML dataset **46859**, version 2 | 76,020 | 369, numerical | 2 | Wider numerical data and imbalance |
| 7 | Online Shoppers Purchasing Intention | UCI dataset 468 | 12,330 | 17, mixed | 2 | Mixed commerce data and imbalance |
| 8 | Diabetes 130-US Hospitals | UCI dataset 296 (configured source) | 101,766 | 47 intended predictors, mixed | 3 | Missingness, patient structure, multiclass readmission |
| 9 | Helena | OpenML dataset **41169** | 65,196 | 27, numerical | 100 | Many-class prediction |
| 10 | Statlog German Credit | UCI dataset **144**, original mixed-feature representation | 1,000 | 20, mixed | 2 | Naturally small mixed-feature classification |
| 11 | Ionosphere | UCI dataset **52** | 351 | 34, numerical before constant-feature removal | 2 | Naturally small numerical classification |
| 12 | Bioresponse | OpenML dataset **4134** | 3,751 | 1,776, molecular descriptors | 2 | High dimensionality relative to training size |

Use one declared source/version per task in final experiments. All twelve accepted datasets download automatically: four UCI archive routes and eight pinned OpenML dataset IDs/versions. Source revision **2026-11-12**: Santander now uses public OpenML dataset 46859 version 2, mirroring the original Customer Satisfaction competition data. Its target remains `TARGET`; the mirror already drops `ID` and rows with missing target labels. OpenML reports 76,020 labeled rows, 369 numerical predictors and two classes. This is not a substitution with another Santander competition. The metadata was checked against OpenML; full raw-data equivalence, row order and prepared hashes remain audit gates.

## Configured source identities

These values mirror `configs/dataset/*.yaml`, not OpenML task identifiers. All selected datasets retain their original classification targets; no regression threshold is introduced.

| Configuration name | Source and ID | Version setting | Target | Classes |
| --- | --- | ---: | --- | ---: |
| `adult` | OpenML 1590 | 2 | `class` | 2 |
| `covertype` | OpenML 1596 | 4 | `class` | 7 |
| `credit_fraud` | OpenML 1597 | 1 | `Class` | 2 |
| `higgs` | OpenML 23512 | 2 | `class` | 2 |
| `amazon` | OpenML 4135 | 1 | `target` | 2 |
| `santander` | OpenML 46859 | 2 | `TARGET` | 2 |
| `online_shoppers` | UCI archive 468 | 1 | `Revenue` | 2 |
| `diabetes` | UCI archive 296 | 1 | `readmitted` | 3 |
| `helena` | OpenML 41169 | 1 | `class` | 100 |
| `german_credit` | UCI archive 144 | 1 | `class` | 2 |
| `ionosphere` | UCI archive 52 | 1 | `class` | 2 |
| `bioresponse` | OpenML 4134 | 1 | `target` | 2 |

UCI `version: 1` is an internal declaration, not an immutable upstream archive revision. UCI URLs are live; freeze prepared hashes or distribute verified prepared files. OpenML IDs/versions are explicit. The current [repository](../src/tabular_thesis/data/tasks.py) checks cached source settings and hashes, target/classes and partition feasibility; it does not automatically certify source independence, leakage or feature availability.

Diabetes declares coded identifiers/diagnoses as categorical, preserves `patient_nbr` only as split metadata and excludes `encounter_id`. Its observed task has 101,766 rows, 47 predictors and 71,518 patient groups. It uses patient-disjoint outer and inner partitions. Santander's OpenML mirror already omits `ID`, so its configuration has no identifier exclusion. OpenML loading explicitly requests `include_row_id=True`; Credit Fraud excludes `Time` (revision **2026-11-12**): it is seconds elapsed since the first transaction in the two-day collection window, monotone in row order, and not a transferable predictor. No other selected dataset contains such a timestamp; Online Shoppers' `Month`, `Weekend` and `SpecialDay` are calendar attributes of each session and remain predictors. HIGGS is the 98,050-row source subset, not the original 11-million-row source.

Preparation stores normalized Parquet plus its manifest under `data/prepared/NAME/`. `expected_sha256` may be frozen after preparation; it is then checked against that normalized file. No implicit row truncation is applied. Label encoding and its class mapping are persisted, so rare-class analysis must resolve the raw outcome explicitly.

The former Santander local-CSV cache has a different source identity and is not reused for the mirror. Preserve it and prepare the OpenML source in a separate data directory as described in [setup](setup.md#prepare-datasets); existing results are not rewritten or treated as equivalent automatically.

## Accepted substitutions

| Removed from the main pool | Accepted replacement | Reason |
| --- | --- | --- |
| Binarized California Housing | Statlog German Credit | Add a naturally small classification task without introducing a derived regression threshold. |
| Bank Marketing | Ionosphere | Add a naturally small numerical task. |
| Jannis | Bioresponse | Add a substantially wider task with fewer observations. |

Bank Marketing and Jannis remain valid literature references, but are outside the main matrix. Binarized California Housing is also excluded. Do not execute those tasks merely because historical configurations or notes still mention them.

## Coverage and interpretation

- **Small data:** German Credit and Ionosphere provide naturally small sources. Training-size experiments add controlled small-training regimes without counting subsets as new datasets.
- **High dimensionality:** Bioresponse provides many predictors relative to available training observations. Santander supplies a separate wider numerical task at a much larger sample size. Report predictor-to-training-example ratios for the actual folds and subsets.
- **Imbalance:** Credit Card Fraud, Santander, and Amazon are the intended three tasks for the rare-class study. Define the rare outcome explicitly: it may not have the positive label in the raw file.
- **Feature types:** Amazon emphasizes categoricals; Adult, German Credit, Online Shoppers, and Diabetes provide mixed features; other tasks emphasize numerical descriptors.
- **Multiclass prediction:** Covertype, Diabetes, and Helena retain distinct class-count regimes.
- **Domains:** income/credit, ecology, transaction fraud, physics, employee access, customer satisfaction, online commerce, clinical care, radar, and molecular response. Helena's anonymous challenge provenance does not establish an additional named application domain.

Do not treat these axes as independently controlled factors. Do not assign unmeasured labels such as “high correlation” or assume that a selected task will be difficult. Measure class frequencies, missingness, cardinality, sparsity, correlation, duplicates, dimensionality, and performance saturation during qualification, before interpreting results.

## Dataset and split qualification gates

1. Freeze source IDs, versions, raw-file hashes, target definitions, feature roles, duplicate policy, and acquisition instructions before final evaluation.
2. Verify that the twelve sources are distinct. Subsets, perturbations, folds, and alternative uploads are not additional independent datasets.
3. Use the shared outer/inner identities in the [evaluation plan](tabarena-evaluation.md). For eligible IID tasks, use one outer split and eight inner folds. Verify sufficient per-class counts at both levels.
4. Audit grouping and time before choosing partitions. Preserve Diabetes patient identifiers for grouping, remove encounter/patient identifiers from predictors, and constrain both outer and inner partitions for the intended unseen-patient claim.
5. Audit entity repetition for Amazon, time and duplicate structure for fraud, geographic structure for Covertype, and feature availability at prediction time for every application. State any group/temporal adaptation rather than calling it an unchanged IID protocol. Patient-group splitting is implemented; a configured temporal field currently raises `NotImplementedError` because temporal splitting is not implemented.
6. Keep HIGGS at the declared OpenML subset. The 11-million-row original source is not this selected task.
7. Verify OpenML feature roles explicitly. The current loader includes row-ID features, so do not assume upstream metadata removed `Time` or other identifiers. Preserve required split metadata before any deliberate predictor filtering.
8. Verify native class support, valid probabilities, fit/prediction memory, and runtime for all eleven methods, especially Helena's 100 classes and the largest/widest tasks. A documented model limit is not a successful integration test.
9. Fit learned preprocessing inside each current training partition, except for the declared native IID KNN feature-preprocessing/LOO path. Fixed TabPFN uses the complete outer development context. Any added constant/duplicate-column removal or feature selection must obey the declared partition boundary.
10. Report failures transparently. Replacing a task or changing one method's training population requires an explicit study-scope revision before using outer test results to choose it.

For controlled training-size experiments, form outer and inner partitions first, then draw nested, stratified subsets of the inner training rows. Keep validation/test populations fixed across the matched conditions, record actual class counts, and disclose the fixed validation-label budget. This measures training-size sensitivity; it is not an active-learning annotation-budget evaluation.

## Source and reference evidence

- [OpenML Santander 46859 metadata](https://www.openml.org/api/v1/json/data/46859), [features](https://www.openml.org/api/v1/json/data/features/46859), and [qualities](https://www.openml.org/api/v1/json/data/qualities/46859): version 2, target `TARGET`, 76,020 rows, 370 columns including target; the mirror omits `ID`.
- [UCI German Credit](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data): original mixed and numeric representations; use the mixed source.
- [UCI Ionosphere](https://archive.ics.uci.edu/dataset/52/ionosphere): radar classification, 351 rows and 34 features.
- [OpenML Bioresponse](https://www.openml.org/search?type=data&id=4134&sort=runs): molecular descriptors; exclude the target from predictor counts.
- [UCI Online Shoppers](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset): 12,330 sessions, with different users described for the sessions.
- [UCI Diabetes](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008): encounter and patient identifiers are ID roles rather than ordinary predictors.
- [OpenML Helena](https://www.openml.org/search?type=data&status=any&exact_name=helena): dataset 41169, not the previously catalogued task identifier 168329.
- [Published benchmark table](https://vldb.org/pvldb/vol14/p2059-zogaj.pdf): HIGGS dataset 23512 has 98,050 rows; Jannis dataset 41168 belongs to the removed candidate.
- [Scientific scope](implementation-status.md#scientific-scope): scientific rationale, sources, and unresolved raw-data checks.

No raw-data audit, machine qualification, or full experiment execution is marked complete by acceptance of this pool.
