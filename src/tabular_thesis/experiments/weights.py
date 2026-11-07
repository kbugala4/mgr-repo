"""Fetch only the accepted foundation checkpoint, outside timed execution."""

from pathlib import Path

from tabular_thesis.experiments.artifacts import file_sha256

REPOSITORY = "Prior-Labs/tabpfn_3_5"
REVISION = "06bf2ba35c80a92a3b9abb436b99cf49e7a0365e"
FILENAME = "tabpfn-v3.5-20260909.safetensors"
SHA256 = "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3"


def prepare_weights(path: Path) -> Path:
    if not path.exists():
        from huggingface_hub import hf_hub_download

        downloaded = Path(
            hf_hub_download(REPOSITORY, filename=FILENAME, revision=REVISION, local_dir=path.parent)
        )
        if downloaded.resolve() != path.resolve():
            raise ValueError(f"Configured checkpoint filename must be {FILENAME}")
    if file_sha256(path) != SHA256:
        raise ValueError("Downloaded checkpoint checksum does not match the pinned revision")
    return path
