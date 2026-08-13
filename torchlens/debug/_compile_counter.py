"""Count Dynamo compilation events around a block of user code.

The torch.compile coexistence contract (compile verdict, 2026-08-12) makes two
countable promises: ZERO new compiles happen while a TorchLens capture holds
the ``force_eager`` stance, and TorchLens's wrapper install/uninstall costs at
most ONE bounded recompile on the next compiled call after capture. This
helper exposes the measurement so users can verify both promises on their own
models instead of taking them on faith.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

from ..utils import _torch_compat


class CompileCountsUnavailableError(RuntimeError):
    """Raised when the running torch does not expose Dynamo's compile counters."""


class CompileCounts:
    """Live view of Dynamo frame compilations observed since a start point.

    Attributes are computed on read, so the same object can be inspected both
    inside and after the measured block.
    """

    def __init__(self, counters: object) -> None:
        """Snapshot the current compilation count as the baseline.

        Parameters
        ----------
        counters:
            Live ``torch._dynamo.utils.counters`` mapping.
        """

        self._counters = counters
        self._start = self._total()

    def _total(self) -> int:
        """Return Dynamo's cumulative frame-compilation count.

        Returns
        -------
        int
            ``counters["frames"]["total"]``: increments on every frame
            compilation including recompiles, flat on warm cache hits and
            under an active ``force_eager`` stance.
        """

        frames = self._counters["frames"]  # type: ignore[index]
        return int(frames["total"])

    @property
    def frames_compiled(self) -> int:
        """Return the number of frame compilations since the baseline.

        Returns
        -------
        int
            Compilation events (including recompiles) observed so far in the
            measured block. ``0`` means Dynamo compiled nothing.
        """

        return self._total() - self._start


@contextlib.contextmanager
def count_compiles() -> Iterator[CompileCounts]:
    """Measure Dynamo compilation events across a block.

    Yields
    ------
    CompileCounts
        Live counter whose ``frames_compiled`` reads the number of Dynamo
        frame compilations (including recompiles) since the block started.

    Raises
    ------
    CompileCountsUnavailableError
        If this torch runtime does not expose ``torch._dynamo.utils.counters``.

    Examples
    --------
    Verify the coexistence contract on a compiled model::

        with tl.debug.count_compiles() as during:
            trace = tl.trace(model, x)
        assert during.frames_compiled == 0  # zero compiles during capture

        with tl.debug.count_compiles() as after:
            model(x)
        assert after.frames_compiled <= 1  # one bounded recompile after
    """

    counters = _torch_compat.get_dynamo_compile_counters(force_probe=True)
    if counters is None:
        raise CompileCountsUnavailableError(
            "torch._dynamo.utils.counters is unavailable in this torch runtime"
        )
    yield CompileCounts(counters)


__all__ = [
    "CompileCounts",
    "CompileCountsUnavailableError",
    "count_compiles",
]
