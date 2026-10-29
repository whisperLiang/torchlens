"""Native address-space protection for allocation-bomb regression tests."""

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest


@contextmanager
def address_space_limit(extra: int = 1 << 30) -> Iterator[None]:
    """Limit virtual memory on systems exposing Linux process memory and RLIMIT_AS.

    Parameters
    ----------
    extra
        Additional bytes allowed above the process's current virtual memory size.

    Yields
    ------
    None
        An allocation-bomb check protected by the native address-space limit.

    Notes
    -----
    Only checks entering this context require these native facilities. Other
    portable tests in the same modules remain collectable and runnable.
    """
    if sys.platform != "linux":
        pytest.skip("requires Linux RLIMIT_AS protection")
    import resource

    status = Path("/proc/self/status")
    if not status.is_file() or not hasattr(resource, "RLIMIT_AS"):
        pytest.skip("requires Linux /proc memory accounting and RLIMIT_AS protection")
    with status.open(encoding="ascii") as handle:
        vmsize_kb = next(int(line.split()[1]) for line in handle if line.startswith("VmSize"))
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (vmsize_kb * 1024 + extra, hard))
    try:
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
