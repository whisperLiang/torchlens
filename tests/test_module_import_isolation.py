"""Every torchlens module must be importable STANDALONE (no circular-import trap).

``import torchlens.bundle`` -- a documented PUBLIC module -- raised
``ImportError: cannot import name 'AmbiguousLabelError' from partially initialized
module 'torchlens.bundle'`` whenever it was the first torchlens module imported in a
process. ``torchlens/bundle/__init__.py`` imports
``torchlens.intervention._metrics``, which first executes
``torchlens/intervention/__init__.py``, which eagerly imported ``.bundle`` -- back
into the still-executing ``torchlens.bundle``. Two test modules
(``test_intervention_hardening_cert7/8.py``) were consequently uncollectable when run
alone and passed in the full suite only because an earlier module happened to import
``torchlens.intervention`` first: a green suite hiding a broken public import.

This module is the standing gate for that CLASS. The cheap arm pins the exact members
of the class (fixed vs still-pending); the exhaustive arm walks every module in the
package. Each candidate is imported in a FORKED child so it observes a genuinely cold
``sys.modules`` -- the parent's already-imported ``torch`` is inherited, which is what
makes the sweep affordable at all.

Adding a fix elsewhere is expected to FAIL the ledger test until the module is moved
out of ``PENDING_IMPORT_CYCLES``; that is the point (a residual cannot be quietly
forgotten, and a fixed module cannot silently regress).
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import pytest

import torchlens as tl

_PACKAGE_ROOT = Path(tl.__file__).resolve().parent

# Modules whose standalone import is FIXED and must stay fixed.
FIXED_IMPORT_CYCLES = (
    "torchlens.bundle",
    "torchlens._chunking",
    "torchlens._user_public_impls",
)

# Modules that still fail a standalone import. Each lives outside this lane's write
# territory (the r3 ops-split and IR packages), so the cycle is DISCLOSED here rather
# than silently tolerated. Fixing one means deleting its row.
PENDING_IMPORT_CYCLES = {
    "torchlens.ir.selector_eval": (
        "cycle through torchlens.ir.__init__ -> selector_eval -> ir.* back-edge; "
        "torchlens/ir/ is outside the artifact-boundary lane's territory"
    ),
    "torchlens.backends.torch._ops_autograd": (
        "the four _ops_* split modules import back into their sibling ops.py facade; "
        "backends/torch/ops.py is owned by the r3 ops-split lane"
    ),
    "torchlens.backends.torch._ops_capture_records": (
        "same _ops_* split-module back-edge as _ops_autograd"
    ),
    "torchlens.backends.torch._ops_emission": (
        "same _ops_* split-module back-edge as _ops_autograd"
    ),
    "torchlens.backends.torch._ops_retention": (
        "same _ops_* split-module back-edge as _ops_autograd"
    ),
}


def _package_modules() -> list[str]:
    """Return every importable module name in the installed torchlens package."""

    names: list[str] = []
    for source_path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        parts = list(source_path.relative_to(_PACKAGE_ROOT).with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        names.append(".".join(["torchlens", *parts]))
    return names


def _standalone_import_error(module_name: str) -> str | None:
    """Import ``module_name`` in a forked child and return its failure, if any.

    Parameters
    ----------
    module_name:
        Dotted module path to import cold.

    Returns
    -------
    str | None
        The child's final traceback line, or ``None`` when the import succeeded.
    """

    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - child process, never measured by coverage
        os.close(read_fd)
        try:
            for key in [k for k in sys.modules if k == "torchlens" or k.startswith("torchlens.")]:
                del sys.modules[key]
            __import__(module_name)
            message = b""
        except BaseException:
            message = traceback.format_exc().strip().splitlines()[-1].encode()[:400]
        try:
            os.write(write_fd, message)
            os.close(write_fd)
        finally:
            os._exit(0)
    os.close(write_fd)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(read_fd, 4096)
        if not chunk:
            break
        chunks.append(chunk)
    os.close(read_fd)
    os.waitpid(pid, 0)
    failure = b"".join(chunks).decode(errors="replace")
    return failure or None


pytestmark = pytest.mark.skipif(
    not hasattr(os, "fork"),
    reason="the cold-import probe needs os.fork to give each module a fresh sys.modules",
)


@pytest.mark.heavy
def test_known_cycle_members_import_standalone() -> None:
    """The fixed members of the circular-import class import cold, individually."""

    failures = {
        name: error
        for name in FIXED_IMPORT_CYCLES
        if (error := _standalone_import_error(name)) is not None
    }
    assert not failures, f"standalone import regressed: {failures}"


@pytest.mark.heavy
def test_pending_cycles_are_exactly_the_ledger() -> None:
    """The disclosed residual set is EXACT: no silent fix, no silent new cycle."""

    still_failing = {
        name for name in PENDING_IMPORT_CYCLES if _standalone_import_error(name) is not None
    }
    fixed = set(PENDING_IMPORT_CYCLES) - still_failing
    assert not fixed, (
        "these modules now import standalone -- delete their PENDING_IMPORT_CYCLES rows: "
        f"{sorted(fixed)}"
    )


@pytest.mark.slow
def test_every_package_module_imports_standalone() -> None:
    """Exhaustive class gate: every torchlens module imports cold.

    One forked import per module across the whole package (~470 modules), so this is
    the ``slow`` tier by construction; the ``heavy`` tests above are the per-step arm.
    """

    modules = _package_modules()
    assert len(modules) > 300, f"module discovery found only {len(modules)} modules"
    failures = {
        name: error
        for name in modules
        if name not in PENDING_IMPORT_CYCLES
        and (error := _standalone_import_error(name)) is not None
    }
    assert not failures, (
        "modules that cannot be imported standalone (fix the cycle, or disclose it in "
        f"PENDING_IMPORT_CYCLES with a reason): {failures}"
    )
