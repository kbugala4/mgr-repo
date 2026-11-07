"""Explicit device, thread, deadline, and cache ownership."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FitContext:
    device: str = "cpu"
    num_cpus: int = 1
    time_limit: float | None = None
    seed: int = 42
    class_weights: tuple[float, ...] | None = None
    prediction_batch_size: int = 4096

    def validate(self) -> None:
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.num_cpus < 1 or self.prediction_batch_size < 1:
            raise ValueError("CPU count and prediction batch size must be positive")
        if self.time_limit is not None and self.time_limit <= 0:
            raise ValueError("time_limit must be positive")
        if self.device == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested but is unavailable. Check the NVIDIA driver and "
                    "container GPU access, or explicitly select runtime.device=cpu."
                )
            # Exercise a kernel as well as allocation before any expensive job.
            probe = torch.ones((8, 8), device="cuda")
            (probe @ probe).sum().item()

    def activate(self) -> None:
        self.validate()
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[key] = str(self.num_cpus)
        import torch

        torch.set_num_threads(self.num_cpus)
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=self.num_cpus)

    def deadline(self) -> float:
        return float("inf") if self.time_limit is None else time.monotonic() + self.time_limit


def configure_caches(root: Path) -> None:
    """Set writable, relocatable caches before importing OpenML or TabArena."""
    root = root.resolve()
    for name, directory in {
        "OPENML_CACHE_DIR": root / "openml",
        "HF_HOME": root / "huggingface",
        "XDG_CONFIG_HOME": root / "config",
        "XDG_CACHE_HOME": root / "libraries",
        "NUMBA_CACHE_DIR": root / "numba",
        "SKB_DATA_DIRECTORY": root / "skrub",
    }.items():
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(directory)
