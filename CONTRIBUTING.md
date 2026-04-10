# Contributing Guide

## Local Setup
1. Create and activate environment.
2. Install app dependencies.
3. Install dev dependencies.

```bash
uv venv
.\\.venv\\Scripts\\Activate.ps1
uv pip install -r requirements.txt
uv pip install -e ".[dev]"
```

## Quality Checks
Run before opening a PR:

```bash
ruff check .
ruff format .
pytest
python scripts/rag_sanity_check.py
```

## Commit Hygiene
- Keep PRs focused and small.
- Add tests for behavior changes.
- Update README when runtime/config behavior changes.
