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
from collections.abc import Iterator

from ..utils import _torch_compat


class CompileCountsUnavailableError(RuntimeError):
    """Raised when the running torch does not expose Dynamo's compile counters."""


class CompileCounts:
    """Count of Dynamo frame compilations observed in a measured block.

    Reads are live while the block is running and FREEZE when it exits, so a
    count captured around one region never silently absorbs compilation events
    that happen after the block (e.g. the bounded post-capture recompile).
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
        self._frozen: int | None = None

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

    def _freeze(self) -> None:
        """Pin the count to the events observed up to now.

        Returns
        -------
        None
            Subsequent ``frames_compiled`` reads return the pinned value.
        """

        self._frozen = self._total() - self._start

    @property
    def frames_compiled(self) -> int:
        """Return the number of frame compilations in the measured block.

        Returns
        -------
        int
            Compilation events (including recompiles) observed in the block.
            ``0`` means Dynamo compiled nothing.
        """

        if self._frozen is not None:
            return self._frozen
        return self._total() - self._start


@contextlib.contextmanager
def count_compiles() -> Iterator[CompileCounts]:
    """Measure Dynamo compilation events across a block.

    Yields
    ------
    CompileCounts
        Counter whose ``frames_compiled`` reads the number of Dynamo frame
        compilations (including recompiles) since the block started; the
        value freezes when the block exits.

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
    counts = CompileCounts(counters)
    try:
        yield counts
    finally:
        counts._freeze()


__all__ = [
    "CompileCounts",
    "CompileCountsUnavailableError",
    "count_compiles",
]
