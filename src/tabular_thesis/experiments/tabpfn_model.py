"""Pin the accepted local TabPFN-3.5 checkpoint; never substitute an older model."""

from pathlib import Path

from tabarena.models.tabpfn_3.model import TabPFN3Model

from tabular_thesis.experiments.artifacts import file_sha256


class TabPFN35Model(TabPFN3Model):
    ag_key, ag_name = "THESIS-TABPFN35", "ThesisTabPFN35"

    def _fit(self, X, y, **kwargs):
        checkpoint = Path(self.params.pop("checkpoint")).expanduser().resolve()
        expected = self.params.pop("checkpoint_sha256", None)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Missing pinned TabPFN checkpoint: {checkpoint}; run 'thesis weights'"
            )
        if expected and file_sha256(checkpoint) != expected:
            raise ValueError("TabPFN checkpoint checksum mismatch")
        if len(X) > 1_000_000 or X.shape[1] > 20_000 or self.num_classes > 160:
            raise ValueError("Task exceeds the declared TabPFN-3.5 limits")
        self.params["checkpoint_per_problem_type"] = {"classification": str(checkpoint)}
        self.params["ignore_pretraining_limits"] = False
        return super()._fit(X, y, **kwargs)
