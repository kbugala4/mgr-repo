# Experiment documentation

The documents here describe the executable platform and accepted scientific design. Local planning notes, session handoffs, historical reviews, generated verification snapshots and manuscript drafts are outside the published repository.

| Document | Purpose |
| --- | --- |
| [Setup](setup.md) | Installation, CPU demo, GPU worker, data/weights, configuration, resume and reporting. |
| [Implementation status](implementation-status.md) | Executed variants, protocol exceptions, verification scope and remaining qualification. |
| [Original thesis topic](thesis-topic.md) | Original Polish research objectives; controls scientific scope. |
| [Model selection](study-model-selection.md) | Accepted 7+4 methods, papers, rationale, exclusions and literature cutoff. |
| [Dataset selection](dataset-selection.md) | Accepted twelve sources, configured identities and audit requirements. |
| [Evaluation protocol](tabarena-evaluation.md) | Main comparison, exceptions, required mechanism studies and nominal workload. |

Executable defaults come from [study YAML](../configs/config.yaml), model/dataset YAML, [typed configuration](../src/tabular_thesis/experiments/configuration.py), source and [dependency lock](../uv.lock). The scientific documents distinguish implemented behavior from required studies whose runners remain pending. Resolve future discrepancies explicitly; do not infer a scope change from code alone.

The method-selection literature cutoff is 2026-10-27. Repository publication did not refresh the literature or alter the accepted method pool, checkpoint or evaluation protocol. Pin code, prepared inputs, dependencies, checkpoint, hardware and image digest before final experiments.
