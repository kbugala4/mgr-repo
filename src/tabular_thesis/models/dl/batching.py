"""Size-aware batch resolution shared by the three local neural wrappers."""

_DENSE = ((2_000, 64), (8_000, 128), (32_000, 256), (128_000, 512))
_DENSE_MAX = 1024

# Categorical cardinality weights the TabNet row-count heuristic, not a memory estimate.
_TABNET = ((2_000, 128, 128), (8_000, 256, 128), (32_000, 1024, 128), (128_000, 2048, 256))
_TABNET_MAX = (4096, 512)
_CARDINALITY_SCALE = 5_000


def _pow2_floor(value: int, minimum: int = 8) -> int:
    value = max(minimum, int(value))
    return 1 << (value.bit_length() - 1)


def dense_batch(n_rows: int) -> int:
    for limit, batch in _DENSE:
        if n_rows < limit:
            return batch
    return _DENSE_MAX


def tabnet_batch(n_rows: int, total_cardinality: int) -> tuple[int, int]:
    effective = n_rows * (1 + total_cardinality / _CARDINALITY_SCALE)
    for limit, batch, virtual in _TABNET:
        if effective < limit:
            return batch, virtual
    return _TABNET_MAX


def tabnet_virtual_batch(n_rows: int, batch_size: int, virtual: int) -> int:
    """Avoid singleton chunks in pinned TabNet's full and remainder virtual batches."""
    virtual = min(virtual, batch_size)
    while True:
        for rows in (min(n_rows, batch_size), n_rows % batch_size):
            if rows < 2:
                continue
            chunks = (rows + virtual - 1) // virtual
            chunk_size = (rows + chunks - 1) // chunks
            if rows % chunk_size == 1:
                break
        else:
            return virtual
        virtual += 1


def resolve(configured: int | str, auto_value: int, multiplier: float = 1.0) -> int:
    """Scale only automatic batches; explicit integer requests retain their value."""
    if configured != "auto":
        return int(configured)
    return _pow2_floor(round(auto_value * multiplier))
