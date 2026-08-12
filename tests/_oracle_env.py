"""Environment-fingerprinted golden resolution for the god-lane oracles (M13).

The byte-identity oracles (surface-v1, viz identity, legacy artifact, state
keysets) freeze REGRESSION baselines, not cross-environment identities: a
different torch/python build can legitimately change qualnames, float
formatting inside reprs, or source-line content without any TorchLens
behavior change. Cluster portability therefore keys goldens per environment:

* The checked-in canonical goldens carry the ``ENV`` marker file naming the
  fingerprint they were recorded under. On a matching environment they are
  enforced exactly as before.
* On any OTHER environment, goldens live under ``env-<fingerprint>/`` inside
  the same goldens directory. The FIRST run on a new environment records the
  baseline and SKIPS with an explicit reason (visible in CI output — never a
  silent green); every later run enforces byte-identity against it.
"""

from __future__ import annotations

import sys
from pathlib import Path


def env_fingerprint() -> str:
    """Return the golden-environment fingerprint for this interpreter."""

    import torch

    torch_version = torch.__version__.split("+", 1)[0]
    return f"py{sys.version_info.major}.{sys.version_info.minor}-torch{torch_version}"


def resolve_env_golden(golden_dir: Path, name: str) -> tuple[Path, bool]:
    """Resolve one golden file for the current environment.

    Returns
    -------
    tuple[Path, bool]
        The golden path to use and whether a MISSING file should be recorded
        as this environment's first-run baseline (True only off the canonical
        environment; a missing canonical golden stays a hard failure).
    """

    marker = golden_dir / "ENV"
    canonical_env = marker.read_text().strip() if marker.exists() else None
    current_env = env_fingerprint()
    if canonical_env is None or current_env == canonical_env:
        return golden_dir / name, False
    return golden_dir / f"env-{current_env}" / name, True
