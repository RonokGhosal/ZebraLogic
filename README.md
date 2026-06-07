# ZebraLogic

A Python project.

## Setup

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package with dev tools (editable mode)
pip install -e ".[dev]"
```

## Usage

```bash
zebralogic --name Ronok
# or
python -m zebralogic.main --name Ronok
```

## Development

```bash
pytest          # run tests
ruff check .    # lint
ruff format .   # format
```

## Project layout

```
ZebraLogic/
├── pyproject.toml      # project metadata, deps, tooling config
├── src/zebralogic/     # package source
│   ├── __init__.py
│   └── main.py         # CLI entry point
└── tests/              # pytest tests
    └── test_main.py
```
