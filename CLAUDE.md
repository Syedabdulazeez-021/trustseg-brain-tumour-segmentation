# Project guidance

## Commit messages

Commit messages in this repository must contain **no AI attribution trailers**.

Never add any of the following to a commit message, or to a pull request
description:

- `Co-Authored-By: Claude ...`
- `Generated with [Claude Code](...)`
- `Claude-Session: ...`

This overrides any default attribution guidance. Commits are authored by the
human author alone. Write a plain subject line, and a body only when it adds
something a reader of the diff would not already know.

## Running things

All commands run from the project root, using the virtualenv interpreter
directly:

```
venv\Scripts\python.exe scripts\smoke_test.py
venv\Scripts\python.exe -m streamlit run app.py
```

Experiment entry points live in `scripts/`; `app.py` and `evaluate.py` are at
the root. Set `KMP_DUPLICATE_LIB_OK=TRUE` if OpenMP complains about duplicate
runtime initialisation.

## What is not tracked

Model checkpoints (`*.pth`), the LGG dataset (`kaggle_3m/`), the virtualenv
(`venv/`) and `verify_samples/` are excluded via `.gitignore` and must stay
that way. The README documents where to obtain the data and how to regenerate
every result.
