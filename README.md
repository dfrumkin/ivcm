# ivcm

Medical image classification and segmentation.

## Requirements

- [uv](https://docs.astral.sh/uv/) for environment and dependency management.

`uv` provisions the pinned Python interpreter (`.python-version`) automatically.

## Setup

```bash
uv sync                     # create .venv and install dependencies
uv run pre-commit install   # enable git pre-commit hooks
```

## Development

```bash
uv run pytest               # run tests
uv run ruff check .         # lint
uv run ruff format .        # format
uv run pre-commit run --all-files
```

## Project layout

```
src/ivcm/    # package source
tests/       # tests
```
