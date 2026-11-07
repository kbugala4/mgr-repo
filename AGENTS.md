# Experiment repository instructions

Read [README](README.md), [setup](docs/setup.md), [implementation status](docs/implementation-status.md) and the [evaluation protocol](docs/tabarena-evaluation.md) before making changes. The original [thesis topic](docs/thesis-topic.md) controls scientific scope.

## Accepted study

- Seven DL methods: MLP, TabNet, FT-Transformer, RealMLP, ModernNCA, TabM and fixed local TabPFN-3.5.
- Four conventional references: logistic regression, KNN, XGBoost and CatBoost.
- Twelve datasets defined in `configs/config.yaml` and documented in `docs/dataset-selection.md`.
- Main protocol: one outer split, eight inner folds, default plus 25 random configurations and cached default/tuned/tuned-ensemble reports. IID KNN uses native child LOO; Diabetes KNN uses group folds. Fixed TabPFN has no HPO or fold ensemble.
- Required weighting and matched MLP regularization/training-size studies are specified but not implemented. Active learning is optional. Do not expand the accepted pool automatically.

## Engineering and verification

Preserve OOP adapters, typed YAML and the central in-process configuration/search-space layer. The runner executes locally with pinned TabArena/AutoGluon libraries. `models/registry.py` contains only local wrappers; the complete method catalog is in `experiments/catalog.py` and `models/specification.py`.

Use `uv.lock` and frozen installation. Global CLI options precede subcommands. Run `make check` for relevant code changes and `uv run --frozen thesis --set runtime.device=cpu smoke` for a small synthetic end-to-end check. Configure real data and model access separately as described in setup.

GPU training, actual TabPFN inference, full-size task feasibility and container execution require worker qualification. Do not describe a CPU smoke result as that evidence. Keep accepted scientific settings and dependency/checkpoint pins unchanged unless the task authorizes a revision.

Do not commit credentials, datasets, weights, generated results, virtual environments, agent state, local plans or session notes. Keep manuscripts outside this experiment repository. Preserve existing work and results; do not run the full campaign without explicit authorization.
