# .github/ — CI/CD Configuration

The workflow FILES are the authority; this inventory summarizes them. If they
disagree, the YAML wins — update this doc in the same change.

## Workflows (all seven)

| File | Trigger | What It Does |
|------|---------|-------------|
| `workflows/lint.yml` | PR to main + `workflow_call` | ruff `format --check` + `check` (CHECK-ONLY — see below), full-tree `pre-commit run --all-files` parity job, full-tree gitleaks scan, actionlint + cross-file pin-lockstep gates (torch, pydot, pip-audit). |
| `workflows/tests.yml` | PR to main + `workflow_call` | Smoke matrix over exact CPU torch pins (floor 2.1.2 / canonical 2.8.0 / newest-admitted, py 3.10–3.12) with executed-floor attestations and the render-byte-oracle row; plus var-gated crawler round21 release-proof jobs (`MENAGERIE_RELEASE_RUNNERS`). |
| `workflows/quality.yml` | PR to main + `workflow_call` | mypy on py3.11 + newest-admitted torch; PR-blocking wheel/sdist manifest tripwires; pip-audit with NO suppressions. |
| `workflows/release.yml` | Push to main | Calls lint/tests/quality via `workflow_call`, then python-semantic-release (pinned) versions, builds reproducible artifacts, publishes to PyPI via OIDC trusted publishing and to GitHub Releases with a minimal-scope App token. |
| `workflows/nightly.yml` | Cron + `workflow_dispatch` | Perf regression gate, full fast tier, coverage floor, capture byte oracle, preview-backend matrix (tf/jax/tinygrad/paddle/mlx), wheel+sdist double-build reproducibility gate, PEP 561 consumer smoke. |
| `workflows/weekly.yml` | Cron + `workflow_dispatch` | Slow and rare tiers with executed-floor attestations. |
| `workflows/latest-canary.yml` | Cron + `workflow_dispatch` | Smoke tier against latest released torch/torchvision (ecosystem-drift isolation). |

## CI must NEVER auto-commit (locked)

Lint is a pure gate: it checks and fails; it never rewrites files, commits, or
pushes. An earlier design ran `ruff format` + `ruff check --fix` in CI and
pushed the fixes to main — that push raced the Release workflow (both trigger
on push to main) and cascaded release runs (the runaway-release incident
class). Do not reintroduce any workflow that commits or pushes, except the
release job's own semantic-release commit, which is guarded by `[skip ci]`
AND a `chore(release):` head-commit condition.

## Release Pipeline Details

- python-semantic-release pinned exactly in release.yml; conventional commits.
- Three never-ship-a-major defense layers: commit-msg hook, pre-push hook,
  custom parser that refuses `LevelBump.MAJOR` (see pyproject
  `[tool.semantic_release]` and `scripts/`). Major bumps require explicit
  owner authorization.
- PyPI: OIDC trusted publishing (no API tokens). GitHub auth: App token minted
  with `permission-contents: write` only, installed before checkout persists
  credentials.
- Artifacts are bit-reproducible (SOURCE_DATE_EPOCH wheel;
  `scripts/normalize_sdist.py` sdist); the nightly double-build gate attests
  both. Reproduce-from-tag recipe: pyproject build_command comment.
- Release-notes body uses the capped `templates/.release_notes.md.j2`
  (hard 100K-character budget; the uncapped builtin caused the 422-blocks-PyPI
  incident).

## Pinning & Permissions

- Third-party actions are SHA-pinned. ONE documented exception:
  `pypa/gh-action-pypi-publish@release/v1` (PyPA's own guidance; rationale in
  release.yml).
- Pre-commit hook repos are SHA-pinned (ruff-pre-commit stays on its tag,
  lockstep-parsed by a test).
- Every workflow declares `permissions: contents: read`; checkout uses
  `persist-credentials: false` outside the release job.

## Conventions

- Conventional commits required: `fix(scope):`, `feat(scope):`, `chore(scope):`
- `fix:` → patch bump, `feat:` → minor bump; major-bump markers (`feat!:`,
  `BREAKING CHANGE:`) are BLOCKED by the three defense layers above
- `chore:`, `docs:`, `ci:`, `test:` → no release
