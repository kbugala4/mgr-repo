.PHONY: setup check doctor prepare weights plan run report smoke test
setup:
	uv sync --frozen
check:
	uv run --frozen ruff check src scripts tests
	uv run --frozen pytest -q

doctor prepare weights plan run smoke:
	uv run --frozen thesis $(ARGS) $@
report:
	uv run --frozen thesis report $(STUDY)
test:
	uv run --frozen pytest -q
