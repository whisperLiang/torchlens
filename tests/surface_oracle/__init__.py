"""Public-object-surface byte-identity oracle for the god-object re-plumbing.

This package snapshots every declared FIELD_ORDER field and every public
``@property`` on the primary record classes (``Trace``, ``Op``, ``Layer``,
``Module``, ``ModuleCall``, ``Param``, ``Buffer``, ``GradFn``, ``GradFnCall``,
``BackwardPass``, ``FuncCallLocation``) across the capture, pickle round-trip,
``.tlspec`` save/load, fork, and run lifecycle stages, and compares the
canonical serialization byte-for-byte against committed goldens after masking
only floating tensor byte hashes, which vary with CPU kernels across machines.
Digest presence, non-floating values, and the rest of the public surface
remain exact; independent same-run workers must still agree on the full raw
snapshot. Storage re-plumbing diffs must be root-caused, never re-snapshotted
merely to pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))
