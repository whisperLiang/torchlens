"""Single authority for which benchmark operations TorchLens owns.

``perf_gate.py`` and ``perf_suite.py`` previously carried two independent
copies of the same 10-prefix classifier with no lockstep test (b2 R41 round
5, wave-born): after fixwave-4 the classifier decides missing-row blocking,
status-failure blocking, uncomparable blocking, and the wall-clock-fallback
block, so a new TorchLens op family named outside the prefixes (or a prefix
edit landing in one copy) silently became FOREIGN on every axis — it could
vanish, fail, or regress with the gate green. Both modules now import from
here, and ``tests/test_perf_ownership.py`` asserts every operation
``perf_runner`` can emit classifies as owned or declared-foreign, never
unknown.
"""

from __future__ import annotations

#: Operation-name prefixes for rows exercising TorchLens code: failures and
#: regressions on these rows are gate-blocking.
TORCHLENS_OPERATION_PREFIXES: tuple[str, ...] = (
    "aux_",
    "fastlog_",
    "first_capture",
    "global_wrap",
    "raw_global",
    "raw_target",
    "raw_tl",
    "rerun_",
    "tl_",
    "trace_",
)

#: Deliberately foreign rows: pure-torch baselines and peer-tool comparisons.
#: These inform ratios but never block the gate. Every operation the runner
#: emits must match exactly one of the two prefix families.
FOREIGN_OPERATION_PREFIXES: tuple[str, ...] = (
    "peer_",
    "raw_forward",
    "raw_inference_mode",
)


def is_torchlens_operation(operation: str) -> bool:
    """Return whether ``operation`` is owned by TorchLens (gate-blocking).

    Parameters
    ----------
    operation:
        Benchmark operation identifier.

    Returns
    -------
    bool
        True when failures on this row should block the gate.
    """

    return operation.startswith(TORCHLENS_OPERATION_PREFIXES)


def classify_operation(operation: str) -> str:
    """Classify one operation as ``"owned"``, ``"foreign"``, or ``"unknown"``.

    ``"unknown"`` is the census-failure signal: an operation neither prefix
    family claims would silently escape all gate-blocking authority.

    Parameters
    ----------
    operation:
        Benchmark operation identifier.

    Returns
    -------
    str
        ``"owned"``, ``"foreign"``, or ``"unknown"``.
    """

    if operation.startswith(TORCHLENS_OPERATION_PREFIXES):
        return "owned"
    if operation.startswith(FOREIGN_OPERATION_PREFIXES):
        return "foreign"
    return "unknown"
