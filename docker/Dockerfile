FROM python:3.11.14-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV UV_LINK_MODE=copy PYTHONUNBUFFERED=1
# Install dependencies without building a source-less project wheel.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src/ src/
COPY configs/ configs/
RUN uv sync --frozen --no-dev
ENTRYPOINT ["/app/.venv/bin/thesis"]
CMD ["doctor"]
