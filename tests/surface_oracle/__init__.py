"""Public-object-surface byte-identity oracle for the god-object re-plumbing.

This package snapshots every declared FIELD_ORDER field and every public
``@property`` on the primary record classes (``Trace``, ``Op``, ``Layer``,
``Module``, ``ModuleCall``, ``Param``, ``Buffer``, ``GradFn``, ``GradFnCall``,
``BackwardPass``, ``FuncCallLocation``) across the capture, pickle round-trip,
``.tlspec`` save/load, fork, and run lifecycle stages, and compares the
canonical serialization byte-for-byte against committed goldens. Storage
re-plumbing phases are judged against these goldens: any diff is a public
behavior change and must be root-caused, never re-snapshotted to pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))
