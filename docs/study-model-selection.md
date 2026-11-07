# Accepted model selection for the thesis

Literature assessment and cutoff: 2026-10-27 (Europe/Warsaw). Model pool accepted: 2026-11-05.

Status: the user accepted the seven DL methods and four conventional references below. This document is the authoritative model-scope decision and supersedes earlier provisional pools in the repository and chat. The [twelve-dataset pool](dataset-selection.md) and initial [evaluation plan](tabarena-evaluation.md) are also accepted: one outer split, eight inner folds, default plus 25 random configurations, with model-specific exceptions. Two required targeted studies examine class weighting and matched regularization/training-size effects; these are variants of existing methods, not additions to the pool. Raw-data audits, final prepared hashes, exact intervention settings and measured budgets remain open. Main dependency/checkpoint pins, seeds and ensemble settings are already configured; actual full-size/GPU behavior still needs qualification. Runtime update 2026-11-06: the eleven-method backend and dependency/configuration pins are implemented; [implementation status](implementation-status.md) records interface checks and remaining qualification.

## Decision

Use **seven neural methods and four conventional references: eleven main methods in total**.

Neural methods:

1. MLP.
2. TabNet.
3. FT-Transformer.
4. RealMLP.
5. ModernNCA.
6. TabM.
7. TabPFN-3.5, using a pinned local base checkpoint.

Conventional references:

1. Logistic regression.
2. KNN.
3. XGBoost.
4. CatBoost.

Use one declared configuration family per method, with its declared tuning protocol. Do not turn every package variant, historical checkpoint, or ensembling option into an additional main method.

KNN is included as the conventional neighborhood reference for ModernNCA. This addition gives the conventional pool linear classification, neighborhood prediction, and boosted-tree ensembles. SVM remains outside the accepted pool.

## Paper dates and references

Dates identify the first public paper/preprint, or the canonical historical publication marked with a dagger. They are not package or checkpoint release dates. Historical methods predate the cited canonical papers; these references do not establish their original invention dates.

| Method | Paper | Paper date |
| --- | --- | --- |
| MLP † | Rumelhart, Hinton and Williams, [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0) | 1986-10-09 |
| TabNet | Arik and Pfister, [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442) | 2019-08-20 |
| FT-Transformer | Gorishniy et al., [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959) | 2021-06-22 |
| RealMLP | Holzmüller et al., [Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data](https://arxiv.org/abs/2407.04491) | 2024-07-05 |
| ModernNCA | Ye et al., [Revisiting Nearest Neighbor for Tabular Data: A Deep Tabular Baseline Two Decades Later](https://arxiv.org/abs/2407.03257) | 2024-07-03 |
| TabM | Gorishniy et al., [TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling](https://arxiv.org/abs/2410.24210) | 2024-10-31 |
| TabPFN-3.5 | Jäger et al., [TabPFN-3.5: Technical Report](https://arxiv.org/abs/2609.17895) | 2026-09-15 |
| Logistic regression † | Cox, [The Regression Analysis of Binary Sequences](https://doi.org/10.1111/j.2517-6161.1958.tb00292.x) | 1958-07 |
| KNN † | Cover and Hart, [Nearest Neighbor Pattern Classification](https://doi.org/10.1109/TIT.1967.1053964) | 1967-01 |
| XGBoost | Chen and Guestrin, [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754) | 2016-03-09 |
| CatBoost | Prokhorenkova et al., [CatBoost: unbiased boosting with categorical features](https://arxiv.org/abs/1706.09516) | 2017-06-28 |

Gorishniy et al. [S1] is also a tabular MLP reference. The current main MLP is a declared local ReLU/dropout/AdamW baseline with bounded one-hot input, not an exact reproduction of that paper’s recipe. The separate overfitting study still needs an independently frozen competent published recipe. Runs record exact effective implementations/settings and checkpoint identity.

This pool covers early neural baselines, historically important tabular-specific architectures, improved dense networks, neighborhood-based prediction, efficient ensembling, and a contemporary pretrained foundation model. It does not experimentally cover every historical architecture family. The literature review can be broader than the experiment matrix.

## Selection rules

The accepted selection follows an inference from the literature and the thesis objectives. No surveyed paper prescribes this exact list.

- Every method must answer a distinct question about tabular classification or the evolution of neural approaches.
- Include historical references deliberately, without labeling them current state of the art.
- Include strong modern neural pipelines, rather than comparing tree models only against older or weak neural implementations.
- Require accessible implementations and a feasible route to binary and multiclass experiments.
- Avoid several methods filling the same primary explanatory role when another family would otherwise be absent.
- Reserve one foundation-model slot. Evaluating the whole foundation-model leaderboard is outside the accepted scope.
- Distinguish mechanisms from measured causal effects: comparing two complete pipelines does not isolate their architectural differences.

The current repository is evidence of available integration work, not a reason to prefer a method scientifically. A wrapper's presence does not prove that it faithfully implements the named paper.

## Literature basis

The review prioritized original method papers, benchmark studies, and official implementations. Source versions matter: results from 2021, 2025, and current release documentation do not describe one unchanged competitive landscape.

| Study | Evidence relevant to selection | Consequence for this thesis |
| --- | --- | --- |
| **Revisiting Deep Learning Models for Tabular Data**, NeurIPS 2021 [S1] | Establishes MLP/ResNet references and FT-Transformer under a shared protocol. | Keep a basic dense reference and a faithful feature-token attention model. Discuss ResNet in the history; do not automatically add another dense baseline. |
| **Why do tree-based models still outperform deep learning on tabular data?**, NeurIPS 2022 [S2] | Benchmarks 45 datasets and investigates irrelevant features, orientation, and irregular target functions. | Organize explanation experiments around controlled data properties. Treat its performance conclusions as evidence for its tested regime and time period. |
| **When Do Neural Nets Outperform Boosted Trees on Tabular Data?**, NeurIPS 2023 [S3] | Compares 19 methods on 176 datasets and finds dataset properties and tuning important to relative performance. | Choose heterogeneous tasks and credible tuning. Do not assume that one family must win universally. |
| **Better by Default**, NeurIPS 2024 [S4] | Proposes RealMLP; separates meta-training datasets for defaults from meta-test evaluation. | Include a strong, reproducible modern dense-network recipe instead of relying on an ad hoc regularization cocktail. |
| **A Closer Look at Deep Learning Methods on Tabular Datasets**, version 4, 2025 [S5] | Evaluates a 300-dataset collection. It reports strong RealMLP, ModernNCA, ensemble, and foundation-model results, while earlier NODE/TabNet models generally rank lower. | Modern dense, neighborhood, and ensemble methods deserve slots. TabNet's role is historical and explanatory. These aggregate results include regression; they are not guaranteed classification outcomes for this thesis. |
| **TabArena**, NeurIPS 2025 [S6] | Curates datasets, model implementations, tuning, and validation. Its initial pool includes RealMLP, ModernNCA, TabM, and several foundation models. | These additions are supported beyond their originating papers. Record tuning, ensembling, and hardware conditions when using its findings. |
| **TabReD**, ICLR 2025 [S7] | Investigates benchmark realism, including temporal drift and feature characteristics. | State whether this thesis evaluates random-split prediction or deployment under shift. Do not claim the latter from the former. |

Benchmark agreement is supporting evidence, not proof of universal superiority. These studies have different data, metrics, protocols, and overlapping authors. Their dataset/model overlap also means that their conclusions are not independent replications by default.

## The seven neural methods

| Method | Era and role | Question it helps answer | Implementation route |
| --- | --- | --- | --- |
| **MLP** | Historical neural baseline | What does a conventional feed-forward network achieve under competent preprocessing and training? | Configured local ReLU/dropout/AdamW implementation, fold-local scaling and bounded one-hot input. See [implementation status](implementation-status.md); the targeted published recipe is separate. |
| **TabNet** | Early specialized architecture; AAAI 2021, preprint 2019 | Can sequential sparse feature selection help on tables with irrelevant or heterogeneous features? | Pinned `pytorch-tabnet==4.1.0` adapter with categorical embeddings and task-aware evaluation. Method reference: [S8]. |
| **FT-Transformer** | Established attention architecture; NeurIPS 2021 | How does feature-specific tokenization plus attention compare with dense networks? | Use the authors' implementation [I1]; the corrected adapter delegates feature-specific tokenization to that implementation. |
| **RealMLP** | Developed dense-network approach; NeurIPS 2024 | How much of the gap can a stronger preprocessing/training/regularization recipe close? | Use a pinned authors' `pytabkit` implementation [I2]. Record the exact TD/HPO variant and effective configuration. |
| **ModernNCA** | Learned-neighborhood approach; ICLR 2025, preprint 2024 | Does learning a useful similarity space improve prediction through neighboring training examples? | Use pinned TabArena’s TALENT-derived implementation; TALENT [I3] is its research lineage, not a direct runtime installation. Method reference: [S9]. |
| **TabM** | Efficient neural ensembling; ICLR 2025 | What does parameter-efficient ensembling add to an MLP-based approach, and at what cost? | Use the authors' implementation [I4]. Fix and record architecture variant, ensemble size, numerical embeddings, and training/inference aggregation. Method reference: [S10]. |
| **TabPFN-3.5** | Latest selected generation; September 2026 technical report | How does cross-dataset pretraining change performance and per-task cost? | Use a pinned local base checkpoint and package version [I5]. Do not silently substitute a hosted Plus/Thinking service. Method report: [S11]. |

The MLP is a reference, not a deliberately weak opponent. RealMLP adds a full training/preprocessing recipe; TabM adds ensembling. These are meaningful experimental comparisons even though all three have dense-network foundations. They should not be interpreted as three unrelated architectures.

TabNet provides an early specialized milestone even if newer models outperform it. Its comparison with FT-Transformer also separates sequential feature selection from general attention over feature tokens. Those complete-model comparisons do not, by themselves, prove that attention type caused a difference.

ModernNCA fills a missing role in the earlier proposals. It is a neural representation-learning method whose prediction uses neighbors; it is not ordinary KNN placed in the wrong category. Keep candidate neighbors and their labels within the training population. Validation/test labels must never enter the reference bank.

TabPFN-3.5 is a contemporary representative of the family whose earlier version was evaluated in Nature [S12]. The 2026 technical report is a different evidence source and release. Earlier peer-reviewed results do not validate every later checkpoint or establish that version 3.5 wins on this thesis's datasets.

## Why one foundation model, and why this one?

The thesis covers the evolution of deep learning for classification, not an exhaustive competition among pretrained tabular models. One representative keeps the experiment matrix bounded.

TabPFN is selected for its recognizable prior-fitted-network lineage, earlier peer-reviewed classification evidence [S12], and a current locally available release [I5]. This is a coverage and scope decision, not a verified performance win over TabICLv2.

TabICLv2 is a credible alternative with an ICML 2026 paper and public pretraining code [S13, I6]. Its paper's comparison with RealTabPFN-2.5 does not establish superiority over TabPFN-3.5. It remains in related work rather than becoming an automatic second experimental entry.

The literature scan also found later-generation candidates such as LimiX-2 [S14] and Google's TabFM [S15]. They show why the latest leaderboard cannot be treated as a final list of obligatory models. Including every contemporary release would defeat the requested scope.

Before freezing the matrix, qualify the chosen local TabPFN checkpoint on the intended task sizes and worker hardware. Verify access, class support, memory use, and deterministic configuration where supported. If it fails a predeclared feasibility condition, document the failure and revise the accepted scope explicitly before selecting a replacement or inspecting final test scores. Do not truncate only this model's training data silently or choose the replacement by test accuracy.

## Methods excluded from the main matrix

Exclusion means lower priority for this particular bounded study, not that a method is invalid or incapable of strong results.

| Candidate | Reason for exclusion |
| --- | --- |
| **ResNet** | Historically important and worth explaining through [S1]. RealMLP occupies the improved-dense-network slot in the main matrix. No automatic extra ResNet experiment is planned. |
| **Local regularized DNNs** | The local cocktail changes several factors and is not an exact reproduction of a published method. Replace its main-table role with RealMLP. Small controlled MLP ablations may still test particular changes. |
| **NODE** | An important differentiable-tree family. To keep the pool bounded, TabNet occupies the early tabular-specific slot. NODE is discussed in related work; experiments will not claim to cover differentiable trees comprehensively. Its lower aggregate rankings in [S5] and implementation-fidelity work also reduce its priority. |
| **SCARF** | A valuable self-supervised approach [S16]. The accepted core studies supervised classification plus one cross-dataset pretrained comparator. A serious SCARF study would require matched pretraining controls and label-budget/noise experiments, expanding the scope. Its absence is not evidence against self-supervision. |
| **TabR** | A strong retrieval-based alternative [S17]. ModernNCA occupies the learned-neighborhood slot and is represented in both [S5] and [S6]. This is not a universal performance claim that ModernNCA beats TabR. |
| **TabTransformer, SAINT, other feature-token attention variants** | FT-Transformer occupies the established attention slot. Additional variants would consume budget without adding an uncovered primary question in this design. |
| **DCN v2, DeepFM, FiBiNET** | Strongly associated with feature interaction/CTR settings. A dedicated interaction or recommendation study would justify them; that is not the accepted core. |
| **IGTD and other tabular-to-image approaches** | Introduce a separate representation/layout and CNN evaluation problem. Discuss historically without committing to a second conversion pipeline. |
| **TabICLv2 and additional foundation models** | The main matrix reserves one foundation-model representative. Keep their evidence and limitations in the literature chapter. |
| **Random forest, LightGBM, SVM, additional conventional methods** | The main comparison uses four conventional references. These remain valid methods, but extra baselines would widen a thesis whose emphasis is DL. Do not interpret four references as exhaustive conventional-ML coverage. |

## Conventional references

- **Logistic regression:** identifies tasks that a simple linear decision rule can solve. Use suitable categorical encoding and regularization.
- **KNN:** supplies the conventional control for ModernNCA. Tune neighbor count and weighting on validation data, and use justified scaling and categorical encoding. Compare complete pipelines honestly; isolating the effect of a learned representation requires matched controls. Record prediction cost and memory, not only its inexpensive fitting step.
- **XGBoost:** provides a strong boosted-tree reference. The repaired adapter uses task-aware classification and explicit CPU/thread settings; tuning uses the central catalog.
- **CatBoost:** provides a distinct boosted-tree implementation with native categorical handling. Preserve this capability in preprocessing.

Do not label a comparison against these models as a comparison against every possible conventional pipeline. Avoid weakening them to make the neural methods look favorable.

## Experimental story and boundaries

The progression is:

1. **Basic neural prediction:** MLP.
2. **Early tabular specialization:** TabNet and FT-Transformer.
3. **Developed solutions:** RealMLP, ModernNCA, and TabM.
4. **Current cross-dataset pretraining:** TabPFN-3.5.

This supports a chapter structure around what each development attempts to solve, whether it helps under the selected conditions, and what computational cost it introduces. It does not establish a universal chronological improvement: a newer method can lose on a particular dataset.

Useful controlled experiments, using subsets of the existing main methods rather than introducing another large model pool:

The required bounded studies are now defined in the [evaluation plan](tabarena-evaluation.md#required-targeted-studies): rare-class sensitivity through class weighting, and overfitting through matched MLP regularization/training-size experiments. Additional experiments below remain optional unless included in that plan.

- **Training size:** identical held-out populations and nested stratified training subsets. Include genuinely small training regimes; the old blanket rule excluding datasets below 10,000 rows is unsuitable for examining modern foundation models.
- **Irrelevant features:** add declared amounts of noise to the training and held-out representations under a fixed protocol. Examine which methods resist the added burden.
- **Numerical and categorical representation:** evaluate a small, predeclared representation ablation on MLP. Do not attribute a complete RealMLP/MLP difference to any single preprocessing component.
- **Ensembling:** if testing the mechanism explicitly, compare appropriately matched single and ensemble configurations. MLP versus TabM alone is a pipeline comparison with multiple differences.

Every extra configuration is an extra experiment and must count toward compute, even when called an ablation. Do not schedule a complete Cartesian product of models, datasets, perturbations, and tuning variants by default.

The dataset set should span the properties needed to distinguish these methods. Historical model coverage alone cannot establish why methods differ if all tasks share similar size, feature types, or difficulty.

## Fairness and reproducibility requirements

- Declare primary metrics and threshold-selection rules before final evaluation. Include PR/average-precision analysis for extreme imbalance. Do not select models by test performance.
- Run defaults and budgeted tuning as separately labeled conditions where included. Equal trial counts do not equal equal computation. Pretrained methods need an inference/adaptation protocol, not artificial repeated training trials.
- Report complete pipelines and their learned preprocessing. Fair comparison does not require identical encodings for every family; it requires justified model-specific treatment and no held-out fitting leakage.
- Record the variant behind each name. For example, the initial TabArena paper uses a particular TabM mini/embedding configuration; that evidence must not be silently attributed to every TabM configuration.
- For RealMLP, record known overlap between evaluation datasets and the datasets used to choose its defaults. Meta-tuned defaults encode cross-dataset information even without foundation-model weights. This is distinct from leaking held-out labels into a particular run.
- For pretrained methods, disclose checkpoint training information and any known benchmark exposure. Report per-task adaptation/inference cost separately from known external pretraining cost.
- Keep training, validation and test identities fixed and auditable. Diabetes group constraints are implemented; temporal splitting is currently rejected. Qualify the prediction claim before changing the split policy.
- Record hardware, threads, memory, fit/search, inference and attempt costs. Current fit timing includes model preprocessing; there is no universal separate preparation timer or extra final-refit stage. Keep fixed foundation inference and known external pretraining costs separate. Verify actual GPU use on the target machines.
- Pin code, dependencies, checkpoint hashes, inference ensemble settings, and the literature cutoff. Do not change the pool during the final run whenever a newer release appears.

## Implementation implications

The main backend now integrates all eleven accepted methods through one AutoGluon model interface. The local MLP/TabNet/FT and conventional adapters were repaired; RealMLP and TabM use pinned reference packages through TabArena; ModernNCA uses TabArena’s TALENT-derived code; fixed TabPFN-3.5 uses its pinned local checkpoint/package. KNN retains optimized validation while fixing one-hot handling for nominal features. [Implementation status](implementation-status.md) specifies exact executed variants, deviations, source/tests, and limits.

Use reference implementations through thin adapters. Do not reimplement four research models from descriptions just to retain uniform-looking local code. Reference packages still require pinned versions, task checks, correct validation handling, and result-provenance tests.

Before a full external run, a pilot must cover binary and multiclass prediction, mixed and numeric-only features, valid probabilities, the actual training-label reference bank for ModernNCA, stable persistence, interruption/resume, and measured worker memory. CPU interface, grouped-split, persistence/resume, and reporting checks now cover part of this gate. Full-size data, actual foundation inference, and target-GPU qualification remain open.

## Sources

### Benchmark and methodological evidence

- **[S1]** Gorishniy et al. *Revisiting Deep Learning Models for Tabular Data*. NeurIPS 2021. https://arxiv.org/abs/2106.11959
- **[S2]** Grinsztajn et al. *Why do tree-based models still outperform deep learning on tabular data?* NeurIPS 2022. https://arxiv.org/abs/2207.08815
- **[S3]** McElfresh et al. *When Do Neural Nets Outperform Boosted Trees on Tabular Data?* NeurIPS 2023. https://arxiv.org/abs/2305.02997
- **[S4]** Holzmüller et al. *Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data*. NeurIPS 2024; arXiv v3, January 2025. https://arxiv.org/abs/2407.04491
- **[S5]** Ye et al. *A Closer Look at Deep Learning Methods on Tabular Datasets*. arXiv v4, November 2025. The version is important; the initial preprint dates to 2024. https://arxiv.org/html/2407.00956v4
- **[S6]** Erickson et al. *TabArena: A Living Benchmark for Machine Learning on Tabular Data*. NeurIPS 2025. Original benchmark protocol/results: https://arxiv.org/html/2506.16791v1 ; current system: https://github.com/autogluon/tabarena
- **[S7]** Rubachev et al. *TabReD: Analyzing Pitfalls and Filling the Gaps in Tabular Deep Learning Benchmarks*. ICLR 2025. https://arxiv.org/abs/2406.19380

### Method papers and current reports

- **[S8]** Arik and Pfister. *TabNet: Attentive Interpretable Tabular Learning*. AAAI 2021; preprint 2019. https://arxiv.org/abs/1908.07442
- **[S9]** Ye et al. *Revisiting Nearest Neighbor for Tabular Data: A Deep Tabular Baseline Two Decades Later*. ICLR 2025. https://arxiv.org/abs/2407.03257
- **[S10]** Gorishniy et al. *TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling*. ICLR 2025. https://arxiv.org/abs/2410.24210
- **[S11]** *TabPFN-3.5: Technical Report*. September 2026, arXiv v2. A current technical report, not the earlier Nature paper. https://arxiv.org/abs/2609.17895
- **[S12]** Hollmann et al. *Accurate predictions on small data with a tabular foundation model*. Nature 637, 319–326, 2025. https://www.nature.com/articles/s41586-024-08328-6
- **[S13]** Qu et al. *TabICLv2: A better, faster, scalable, and open tabular foundation model*. ICML 2026. https://arxiv.org/abs/2602.11139
- **[S14]** *LimiX-2: A Contextual Mechanism Network Towards General Structured-Data Intelligence*. September 2026 preprint. https://arxiv.org/abs/2609.17488
- **[S15]** Google Research. *Introducing TabFM: A zero-shot foundation model for tabular data*. June 2026 author article. https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data/
- **[S16]** Bahri et al. *SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption*. ICLR 2022. https://arxiv.org/abs/2106.15147
- **[S17]** Gorishniy et al. *TabR: Tabular Deep Learning Meets Nearest Neighbors*. ICLR 2024. https://openreview.net/pdf?id=rhgIgTSSxW

### Implementation and availability checks

- **[I1]** Authors' reference MLP/ResNet/FT-Transformer implementation: https://github.com/yandex-research/rtdl-revisiting-models
- **[I2]** Authors' RealMLP implementation: https://github.com/dholzmueller/pytabkit
- **[I3]** Authors' ModernNCA implementation, linked by its paper, distributed in TALENT: https://github.com/LAMDA-Tabular/TALENT
- **[I4]** Authors' TabM implementation: https://github.com/yandex-research/tabm
- **[I5]** TabPFN local model/version availability: https://docs.priorlabs.ai/models ; implementation: https://github.com/PriorLabs/TabPFN
- **[I6]** Authors' TabICLv2 inference and pretraining implementation: https://github.com/soda-inria/tabicl

The original literature selection analysis was read-only and trained no candidates. Later implementation/interface checks and the shortened CPU pilot are recorded separately in [implementation status](implementation-status.md). The bibliography has not been refreshed by the documentation reconciliation. The accepted pool must be qualified by the corrected implementation and hardware pilots before results can support thesis claims.
