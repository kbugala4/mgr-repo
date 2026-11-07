"""Reproducibility utilities — seed management and deterministic settings."""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducible experiments.

    Configures Python, NumPy, and (if available) PyTorch random
    number generators. Enables deterministic mode in PyTorch.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        logger.debug("PyTorch seeds set to %d (deterministic mode enabled)", seed)
    except ImportError:
        logger.debug("PyTorch not available — skipping torch seed setup")

    logger.info("Random seed set to %d", seed)


def get_device() -> str:
    """Detect the best available compute device.

    Returns:
        ``'cuda'`` if an NVIDIA GPU is available, otherwise ``'cpu'``.
    """
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info("Using GPU: %s", device_name)
            return "cuda"
    except ImportError:
        pass

    logger.info("Using CPU")
    return "cpu"
