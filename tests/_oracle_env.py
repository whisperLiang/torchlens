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
  the same goldens directory and are enforced byte-exactly when COMMITTED.
* A MISSING off-canonical golden is FAIL-CLOSED (b10 R78-4): the historical
  record-and-skip behavior silently self-baselined every ephemeral CI leg
  forever (no cache, no committed ``env-*`` dir, so "every later run
  enforces" never happened) and auto-rebaselined fresh dev boxes. Recording
  a first-run baseline now requires the EXPLICIT ``TORCHLENS_ORACLE_RECORD_ENV=1``
  opt-in (deliberate provisioning of a new long-lived box); ephemeral CI
  environments (``CI`` set) skip with a visible reason and never write.

``env_fingerprint`` deliberately stays NARROW (py-major.minor + torch version
sans build tag). It is known-blind to build variant, CPU ISA, thread count,
and BLAS (SF-51): do NOT try to close cross-machine float drift by widening
this key — the fail-closed missing-golden policy above is the guard, and a
structural/subprocess oracle is the fix direction of record.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

#: Explicit opt-in for recording a first-run baseline on a new (off-canonical,
#: non-ephemeral) environment. One deliberate provisioning run, then commit
#: the recorded ``env-*`` directory if the environment is meant to enforce.
RECORD_ENV_VAR = "TORCHLENS_ORACLE_RECORD_ENV"

#: Declares the running leg the ENFORCING leg for the byte-oracle goldens.
#: On an enforcing leg every missing-golden outcome is a hard FAILURE — the
#: CI skip and the record opt-in are both refused — so the one leg whose
#: environment is supposed to match the committed ENV marker can never drift
#: off-canonical (e.g. a matrix torch bump without a golden rebaseline) and
#: silently skip every golden case while staying green (grind-p3 T13.1).
ENFORCE_ENV_VAR = "TORCHLENS_ORACLE_ENFORCE"


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
        The golden path to use and whether this environment is off-canonical
        (True exactly when the path is env-keyed). Callers enforcing a golden
        should prefer :func:`require_env_golden`, which owns the fail-closed
        missing-file policy.
    """

    marker = golden_dir / "ENV"
    canonical_env = marker.read_text().strip() if marker.exists() else None
    current_env = env_fingerprint()
    if canonical_env is None or current_env == canonical_env:
        return golden_dir / name, False
    return golden_dir / f"env-{current_env}" / name, True


def require_env_golden(golden_dir: Path, name: str, update_env: str) -> Path:
    """Return the enforceable golden path for this environment, fail-closed.

    Policy for a MISSING golden (b10 R78-4):

    * canonical environment — hard failure (unchanged historical behavior);
    * ``TORCHLENS_ORACLE_ENFORCE=1`` (the declared enforcing leg) — hard
      failure, taking precedence over the record opt-in and the CI skip: an
      enforcing leg that finds itself off-canonical has drifted from the
      committed ENV marker and must go red, never quietly skip (T13.1);
    * off-canonical with ``TORCHLENS_ORACLE_RECORD_ENV=1`` and no ``CI`` —
      the caller may record a first-run baseline: the path is returned with
      its parent created, and the caller writes it then SKIPS;
    * off-canonical under ``CI`` — skip with a visible reason, never write
      (an ephemeral checkout can never satisfy "every later run enforces");
    * off-canonical otherwise — FAIL with recording instructions, so a fresh
      box never silently self-baselines while a root cause is open.

    Parameters
    ----------
    golden_dir:
        Directory holding the canonical goldens and the ``ENV`` marker.
    name:
        Golden file name.
    update_env:
        The owning family's update flag, named in failure messages.

    Returns
    -------
    Path
        Path of an EXISTING golden to enforce, or (record opt-in only) the
        path to record.
    """

    golden_path, off_canonical = resolve_env_golden(golden_dir, name)
    if golden_path.exists():
        return golden_path
    if not off_canonical:
        pytest.fail(
            f"missing canonical golden {golden_path}; generate deliberately with "
            f"{update_env}=1 (the update run reports SKIP, then re-run to verify)"
        )
    if os.environ.get(ENFORCE_ENV_VAR) == "1":
        pytest.fail(
            f"this leg declares {ENFORCE_ENV_VAR}=1 (byte-oracle enforcement) but "
            f"runs off-canonical environment {env_fingerprint()!r} with no committed "
            f"golden ({golden_path} missing). The enforcing leg has drifted from the "
            "committed ENV marker — rebaseline the goldens deliberately (one "
            f"{RECORD_ENV_VAR}=1 run on the new environment, reviewed and committed) "
            "or restore the leg's environment; an enforcing leg never skips"
        )
    if os.environ.get(RECORD_ENV_VAR) == "1" and not os.environ.get("CI"):
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        return golden_path
    if os.environ.get("CI"):
        pytest.skip(
            f"no committed golden for environment {env_fingerprint()!r} "
            f"({golden_path} missing); byte enforcement runs on environments "
            "with committed baselines only"
        )
    pytest.fail(
        f"no golden for environment {env_fingerprint()!r} ({golden_path} missing). "
        f"Refusing to self-baseline: record ONE deliberate first-run baseline with "
        f"{RECORD_ENV_VAR}=1, review it, and commit the env-* directory if this "
        "environment should enforce byte identity"
    )


def golden_mutation_flags_armed_under_ci(environ: Mapping[str, str]) -> list[str]:
    """Return golden update/regen/record flags armed in a CI environment.

    Consumed by the root conftest's session guard (b7 R53-3): a CI run with
    any of these armed would rebaseline instead of verifying.

    Parameters
    ----------
    environ:
        Environment mapping to inspect.

    Returns
    -------
    list[str]
        Sorted offending variable names; empty outside CI or when none armed.
    """

    if not environ.get("CI"):
        return []
    return sorted(
        name
        for name in environ
        if name.startswith(("TORCHLENS_UPDATE_", "TORCHLENS_REGEN_"))
        or name in {RECORD_ENV_VAR, "TL_SELECTOR_MATRIX_REGEN"}
    )


def write_provenance(golden_dir: Path, generator: str, update_env: str) -> None:
    """Record how the goldens in ``golden_dir`` were (re)generated.

    Written by update/record runs only — a sidecar, never compared, so it
    documents regeneration without perturbing golden bytes (b10 R78-8a).
    """

    import datetime

    import torch

    (golden_dir / "PROVENANCE").write_text(
        f"generator: {generator}\n"
        f"flag: {update_env}=1\n"
        f"env: {env_fingerprint()}\n"
        f"torch: {torch.__version__}\n"
        f"recorded: {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n"
    )
