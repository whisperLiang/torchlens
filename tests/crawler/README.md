# Crawler acceptance checks

The ordinary acceptance gate checks frozen prompts, tracked-path boundaries,
release-lock provenance, and dry-run admission without starting a live crawler.
Committed release locks are allowed only with the exact provenance, hashes,
specifications, and probe receipts checked by the lifecycle suite. Untracked
companions cannot attest a tracked lock; local ignored solves are not releases.

```bash
python -m pytest tests/crawler/ -m "not rare and not slow" --tb=short
```

The full suite additionally runs the repository-wide secret scan and the real
isolated-worker composition. The secret scan uses `detect-secrets==1.5.0`, matching
the pre-commit pin, through the active Python interpreter. Its measured runtime
exceeds 40 seconds, so it is a separate `slow` test rather than part of the
under-five-second static boundary check.

```bash
python -m pytest tests/crawler/ --tb=short -ra
```

The real-worker dry-run/resume test shares the release suite's authenticated
composition: receipts and model awards, checkpoint/resume, append-only ledgers,
and progress notifications remain checked there. It requires a lock-built conda
prefix selected with `MENAGERIE_REAL_ENV_PREFIX` (or `CONDA_PREFIX`) and validated
by `menagerie/crawler/tests/conftest.py`. Missing prerequisites skip visibly in a
local test run and fail under `MENAGERIE_RELEASE_GATE=1`; such skips are not
evidence that the real-worker composition passed. Never substitute a fabricated
worker or an arbitrary existing Python prefix to make that test execute.
