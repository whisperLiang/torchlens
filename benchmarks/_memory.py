"""Native memory measurements without importing the package being benchmarked."""

import sys


def peak_rss_bytes() -> int:
    """Return the process's native lifetime peak resident memory in bytes.

    Returns
    -------
    int
        Windows peak working set, Linux ``ru_maxrss`` converted from KiB,
        or macOS ``ru_maxrss`` in its native byte units.

    Notes
    -----
    Kept outside torchlens so raw-forward benchmark subprocesses stay independent
    of TorchLens imports and wrapper installation.
    """
    if sys.platform == "win32":
        import psutil

        return int(psutil.Process().memory_info().peak_wset)
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024
