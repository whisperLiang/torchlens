"""W051 (AUD-CODE 2.16 / 2.18): foreign-thread touches settle against the owner's joins.

The FLAKEHUNT split routed every pure read made by a PRE-EXISTING thread into a
session-only disclosure so a benign background flusher could no longer ceiling an
honest capture. It over-corrected: host nondeterminism fed to the forward THROUGH a
pre-existing thread (a warm ``ThreadPoolExecutor``, a running asyncio loop) settled a
false ``verified`` because the only thread-agnostic mark was gone.

The tripwire now has two halves. Foreign-thread touches are still disclosed, never a
ceiling by themselves (the benign case PASSES). Every wait on a stdlib synchronization
primitive inside the capture's thread universe is classified by its counterpart thread,
and every disclosed foreign read is promoted to monitor UNCERTAINTY when some wait's
counterpart was outside that universe (an UNHOOKED join) or unresolvable (the malicious
case FAILS). A join whose counterparts are all in-window workers promotes nothing.
Private ``random.Random`` instance draws route through the same thread routing (2.18), so
a background ``tempfile`` name sequence no longer ceilings.

Named design fork, pinned as strict xfails below: an unhooked join with NO disclosed read
(``datetime.now`` -- an unpatchable C classmethod -- or a HELD clock builtin executed on a
pre-existing worker) is the contract's clause-(iv) residual and today settles clean; the
one-call switch that would settle it uncertain changes the r41 doctrine that a
digest-witnessed draw on a pre-existing worker is CERTAIN, so it is left for a ruling.
"""

from __future__ import annotations

import asyncio
import datetime
import os
import queue
import random
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import wraps
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._utils.rng import (
    HOST_NONDETERMINISM_REGISTRY,
    channels as _rng_channels,
    host_nondeterminism_monitor,
)


class _LinearNoBias(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


class _FeedIn(nn.Module):
    """``lin(x) * (1 + frac(value))`` with ``value`` obtained through ``fetch``."""

    def __init__(self, fetch: Callable[[], float]) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4, bias=False)
        self._fetch = fetch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x) * (1.0 + (float(self._fetch()) % 1.0))


@pytest.fixture
def warm_pool() -> Any:
    """A thread pool whose workers exist BEFORE any monitor window opens."""

    pool = ThreadPoolExecutor(max_workers=2)
    for future in [pool.submit(lambda: 1) for _ in range(4)]:
        future.result()
    try:
        yield pool
    finally:
        pool.shutdown(wait=True)


@pytest.fixture
def running_loop() -> Any:
    """An asyncio loop running on a thread that exists BEFORE any monitor window opens."""

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, name="w051-pre-loop", daemon=True)
    thread.start()
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=30)
        loop.close()


def _uncertain_reasons(result: Any, prefix: str) -> set[str]:
    return {reason for reason in result.uncertain_detail if reason.startswith(prefix)}


def _monitor_diagnostics(result: _rng_channels.HostRngMonitorResult) -> str:
    """Show the wait classification and channel evidence behind a monitor verdict."""

    return (
        f"uncertain_detail={result.uncertain_detail}\n"
        f"owner_thread_waits={sorted(result.owner_thread_waits)}\n"
        f"owner_unhooked_joins={sorted(result.owner_unhooked_joins)}\n"
        f"owner_unattributed_waits={sorted(result.owner_unattributed_waits)}\n"
        f"foreign_thread_reads={sorted(result.foreign_thread_reads)}\n"
        f"channels={sorted(result.channels)}"
    )


# ---- 2.16 malicious topologies FAIL -----------------------------------------------------


def test_prepool_clock_feed_in_settles_uncertain(warm_pool: ThreadPoolExecutor) -> None:
    """``PRE_POOL.submit(time.time).result()``: the unhooked join and the foreign read
    are both named; the capture is INCOMPLETE, never verified."""

    with host_nondeterminism_monitor(_LinearNoBias()) as result:
        warm_pool.submit(time.time).result()
    assert result.uncertain
    assert "time.time" in result.foreign_thread_reads
    assert "concurrent.futures.Future.result" in result.owner_unhooked_joins
    assert "owner_thread_waited:concurrent.futures.Future.result" in result.uncertain_detail
    assert "foreign_thread_read_joined:time.time" in result.uncertain_detail
    assert not result.channels, "foreign reads are promoted to uncertainty, never claimed consumed"


_UNHOOKED_JOIN_ALONE_FORK = (
    "design fork 'unhooked join alone ceilings' (W051 remainder): an unhooked join with no "
    "disclosed read is today the contract's clause-(iv) residual, disclosed on "
    "owner_unhooked_joins but not settled uncertain"
)


@pytest.mark.xfail(strict=True, reason=_UNHOOKED_JOIN_ALONE_FORK)
def test_prepool_datetime_now_feed_in_settles_uncertain(warm_pool: ThreadPoolExecutor) -> None:
    """``datetime.now`` is an unpatchable C classmethod invisible on an unhooked thread:
    only the join evidence exists (no foreign read can be disclosed). Strict xfail: the
    day the fork is ruled and the join alone settles uncertain, this test flips to a pin."""

    with host_nondeterminism_monitor(_LinearNoBias()) as result:
        warm_pool.submit(lambda: datetime.datetime.now().timestamp()).result()
    assert not result.foreign_thread_reads
    assert result.owner_unhooked_joins == {"concurrent.futures.Future.result"}
    assert result.uncertain


def test_prepool_datetime_now_join_is_disclosed(warm_pool: ThreadPoolExecutor) -> None:
    """The residual is DISCLOSED even while it does not settle: the unhooked join is named."""

    with host_nondeterminism_monitor(_LinearNoBias()) as result:
        warm_pool.submit(lambda: datetime.datetime.now().timestamp()).result()
    assert result.owner_unhooked_joins == {"concurrent.futures.Future.result"}
    assert not result.channels


def test_preexisting_loop_thread_feed_in_settles_uncertain(running_loop: Any) -> None:
    """``run_coroutine_threadsafe(...).result()`` on a pre-existing loop thread joins it."""

    async def _now() -> float:
        return time.time()

    with host_nondeterminism_monitor(_LinearNoBias()) as result:
        asyncio.run_coroutine_threadsafe(_now(), running_loop).result(timeout=30)
    assert result.uncertain
    assert "concurrent.futures.Future.result" in result.owner_unhooked_joins
    assert "foreign_thread_read_joined:time.time" in result.uncertain_detail


@pytest.mark.parametrize(
    "channel, reader",
    [
        ("os.urandom", lambda: os.urandom(1)[0]),
        ("time.perf_counter", lambda: time.perf_counter()),
        ("np.random.default_rng", lambda: __import__("numpy").random.default_rng().random()),
        ("_random.Random.random", lambda: random.Random().random()),
    ],
)
def test_prepool_entropy_and_instance_feed_ins_settle_uncertain(
    warm_pool: ThreadPoolExecutor, channel: str, reader: Callable[[], Any]
) -> None:
    """Entropy funnels, construction entropy, and PRIVATE instance draws (2.18) fed from a
    pre-existing worker all disclose the channel and settle uncertain through the join."""

    with host_nondeterminism_monitor(_LinearNoBias()) as result:
        warm_pool.submit(reader).result()
    assert result.uncertain
    assert channel in result.foreign_thread_reads
    assert f"foreign_thread_read_joined:{channel}" in result.uncertain_detail


@pytest.mark.xfail(strict=True, reason=_UNHOOKED_JOIN_ALONE_FORK)
def test_held_builtin_on_unhooked_worker_is_caught_by_the_join_alone(
    warm_pool: ThreadPoolExecutor,
) -> None:
    """A HELD clock builtin (``from time import perf_counter``) executed on an unhooked
    worker bypasses the module-attr patch and emits no observable event: nothing can be
    disclosed, so only the unhooked join could fail the capture (the named fork)."""

    held_perf_counter = time.perf_counter  # bound before the window: the original builtin
    with host_nondeterminism_monitor(_LinearNoBias()) as result:
        warm_pool.submit(held_perf_counter).result()
    assert not result.foreign_thread_reads
    assert "concurrent.futures.Future.result" in result.owner_unhooked_joins
    assert result.uncertain


def test_relay_through_in_window_worker_is_not_laundered(warm_pool: ThreadPoolExecutor) -> None:
    """An in-window worker fetches from the pre-existing pool and the owner joins ONLY the
    hooked worker: the worker's own unhooked join carries the promotion."""

    box: list[float] = []

    def _relay() -> None:
        box.append(warm_pool.submit(time.time).result())

    with host_nondeterminism_monitor(_LinearNoBias()) as result:
        worker = threading.Thread(target=_relay, name="w051-relay")
        worker.start()
        worker.join(timeout=30)
    assert result.uncertain
    assert "concurrent.futures.Future.result" in result.owner_unhooked_joins
    assert "foreign_thread_read_joined:time.time" in result.uncertain_detail


def test_event_handshake_with_reading_thread_is_a_join() -> None:
    """An Event carrying a pre-existing thread's RNG draw and clock read is a join."""

    start, done = threading.Event(), threading.Event()
    generator = random.Random(7)
    values: list[float] = []

    def _reader() -> None:
        start.wait(timeout=30)
        values.extend((time.time(), generator.random()))
        done.set()

    reader = threading.Thread(target=_reader, name="w051-reader", daemon=True)
    reader.start()
    try:
        with host_nondeterminism_monitor(_LinearNoBias()) as result:
            start.set()
            assert done.wait(timeout=30)
            _, draw = values
            assert 0 <= draw < 1
    finally:
        start.set()
        reader.join(timeout=30)
    assert result.uncertain
    assert "threading.Event.wait" in result.owner_unhooked_joins
    assert "foreign_thread_read_joined:time.time" in result.uncertain_detail
    assert "foreign_thread_read_joined:_random.Random.random" in result.uncertain_detail


def test_queue_get_from_preexisting_producer_is_a_join() -> None:
    """``queue.Queue.get`` resolves its counterpart through the queue's Condition stamp."""

    items: queue.Queue[float] = queue.Queue()
    start = threading.Event()

    def _producer() -> None:
        start.wait(timeout=30)
        items.put(time.time())

    producer = threading.Thread(target=_producer, name="w051-producer", daemon=True)
    producer.start()
    try:
        with host_nondeterminism_monitor(_LinearNoBias()) as result:
            start.set()
            items.get(timeout=30)
    finally:
        producer.join(timeout=30)
    assert result.uncertain
    assert "queue.Queue.get" in result.owner_unhooked_joins


def test_prebound_wait_alias_bypassing_class_patch_is_still_witnessed() -> None:
    """A pre-window bound method (``wait = event.wait``) calls the ORIGINAL function and
    bypasses the class patch; the held-code ``call`` layer records an unattributed wait, and
    a disclosed foreign read is promoted through it (fail-closed)."""

    start, done = threading.Event(), threading.Event()
    prebound_wait = done.wait

    def _reader() -> None:
        start.wait(timeout=30)
        time.time()
        done.set()

    reader = threading.Thread(target=_reader, name="w051-reader-alias", daemon=True)
    reader.start()
    try:
        with host_nondeterminism_monitor(_LinearNoBias()) as result:
            start.set()
            assert prebound_wait(timeout=30)
    finally:
        reader.join(timeout=30)
    assert result.uncertain
    assert "threading.Event.wait" in result.owner_thread_waits
    assert "foreign_thread_read_joined:time.time" in result.uncertain_detail


# ---- benign topologies PASS -----------------------------------------------------------------


def _spin_until(predicate: Callable[[], bool], budget_s: float = 30.0) -> None:
    """Busy-poll a plain shared flag -- NO synchronization primitive is entered."""

    deadline = time.monotonic() + budget_s  # monotonic is a monitored channel; owner marks it
    while not predicate():
        if time.monotonic() > deadline:  # pragma: no cover - failure path
            raise AssertionError("background reader never ran in-window")


def test_unjoined_background_reader_never_ceilings() -> None:
    """The FLAKEHUNT contract case, with the owner NEVER blocking on the reader: reads are
    disclosed only, no channel, no uncertainty."""

    reads = [0]
    stop = threading.Event()

    def _flusher() -> None:
        while not stop.is_set():
            time.time()
            os.urandom(2)
            reads[0] += 1

    flusher = threading.Thread(target=_flusher, name="w051-flusher", daemon=True)
    flusher.start()
    try:
        with host_nondeterminism_monitor(_LinearNoBias()) as result:
            seen = reads[0]
            _spin_until(lambda: reads[0] > seen + 5)
    finally:
        stop.set()
        flusher.join(timeout=30)
    assert not result.uncertain
    assert not result.owner_thread_waits
    assert {"time.time", "os.urandom"} <= result.foreign_thread_reads
    # The owner's own busy-poll clock reads keep ceiling exactly as before.
    assert result.channels == {"time.monotonic"}


def test_background_tempfile_loop_never_ceilings(tmp_path: Path) -> None:
    """2.18: stdlib ``tempfile`` draws from a process-global private ``random.Random``; a
    background thread making temp files in-window is disclosed, never a ceiling."""

    reads = [0]
    stop = threading.Event()

    def _writer() -> None:
        while not stop.is_set():
            fd, path = tempfile.mkstemp(dir=tmp_path)
            os.close(fd)
            os.unlink(path)
            reads[0] += 1

    writer = threading.Thread(target=_writer, name="w051-tempfile", daemon=True)
    writer.start()
    try:
        with host_nondeterminism_monitor(_LinearNoBias()) as result:
            seen = reads[0]
            _spin_until(lambda: reads[0] > seen + 3)
    finally:
        stop.set()
        writer.join(timeout=30)
    assert not result.uncertain
    assert "_random.Random.random" not in result.channels
    assert result.foreign_thread_reads & {"_random.Random.random", "_random.Random.getrandbits"}


def test_owner_and_in_window_instance_draws_still_ceiling() -> None:
    """The 2.18 routing narrows ONLY foreign threads: owner-thread and in-window-thread
    private instance draws keep ceiling (the tripwire is not weakened)."""

    with host_nondeterminism_monitor(_LinearNoBias()) as owner_result:
        random.Random(7).random()
    assert "_random.Random.random" in owner_result.channels

    with host_nondeterminism_monitor(_LinearNoBias()) as worker_result:
        worker = threading.Thread(target=lambda: random.Random(7).random())
        worker.start()
        worker.join(timeout=30)
    assert "_random.Random.random" in worker_result.channels


def test_join_on_own_in_window_worker_promotes_nothing() -> None:
    """Hooked-only counterparts (the model's OWN in-window worker) are disclosure only: a
    concurrent unjoined flusher's reads are NOT promoted through such a join."""

    reads = [0]
    stop = threading.Event()

    def _flusher() -> None:
        while not stop.is_set():
            time.time()
            reads[0] += 1

    flusher = threading.Thread(target=_flusher, name="w051-flusher-2", daemon=True)
    flusher.start()
    try:
        with host_nondeterminism_monitor(_LinearNoBias()) as result:
            seen = reads[0]
            _spin_until(lambda: reads[0] > seen + 5)
            box: list[int] = []
            worker = threading.Thread(target=lambda: box.append(sum(range(1000))))
            worker.start()
            worker.join(timeout=30)
    finally:
        stop.set()
        flusher.join(timeout=30)
    assert not result.uncertain, _monitor_diagnostics(result)
    assert "threading.Thread.join" in result.owner_thread_waits
    assert not result.owner_unhooked_joins
    assert not result.owner_unattributed_waits
    assert "time.time" in result.foreign_thread_reads


def test_thread_start_handshake_before_profile_registration_is_not_a_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finish Thread.start while its worker is paused before installing the profile hook."""

    stop = threading.Event()

    def flush() -> None:
        while not stop.is_set():
            time.time()

    flusher = threading.Thread(target=flush, name="w051-start-race-flusher")
    owner_waiting, release_worker, worker_ran = (threading.Event() for _ in range(3))
    worker = threading.Thread(target=worker_ran.set, name="w051-start-race-worker")
    started = worker._started
    set_started = started.set
    release_condition = started._cond._release_save

    def release_start_condition() -> Any:
        """Signal when start's Event.wait reaches its blocking Condition.wait."""

        owner_waiting.set()
        return release_condition()

    def pause_bootstrap_after_started() -> None:
        """Set the real startup flag, then suspend bootstrap before it can install the hook."""

        owner_waiting.wait(timeout=30)
        set_started()
        release_worker.wait(timeout=30)

    monkeypatch.setattr(started._cond, "_release_save", release_start_condition)
    monkeypatch.setattr(started, "set", pause_bootstrap_after_started)
    monitor = host_nondeterminism_monitor(_LinearNoBias())
    flusher.start()
    try:
        with monitor as result:
            _spin_until(lambda: "time.time" in result.foreign_thread_reads)
            worker.start()
            assert owner_waiting.is_set()
            assert worker.ident is not None
            assert worker.ident not in monitor._in_window_thread_idents
            release_worker.set()
            worker.join(timeout=30)
            assert worker_ran.is_set()
            assert worker.ident in monitor._in_window_thread_idents
    finally:
        owner_waiting.set()
        release_worker.set()
        stop.set()
        if worker.ident is not None:
            worker.join(timeout=30)
        flusher.join(timeout=30)
        assert not worker.is_alive()
        assert not flusher.is_alive()

    diagnostics = _monitor_diagnostics(result)
    assert not result.uncertain, diagnostics
    assert result.owner_thread_waits == {"threading.Thread.join"}, diagnostics
    assert not result.owner_unhooked_joins, diagnostics
    assert not result.owner_unattributed_waits, diagnostics
    assert "time.time" in result.foreign_thread_reads


@pytest.mark.parametrize("wrapped", [True, False], ids=["wrapped-start", "subclass-start"])
def test_user_start_wait_on_started_event_still_joins_foreign_reader(
    monkeypatch: pytest.MonkeyPatch, wrapped: bool
) -> None:
    """User start code that waits on its own _started Event can receive foreign data."""

    start_thread = threading.Thread.start
    ready = threading.Event()
    values: list[float] = []

    @wraps(start_thread)
    def start_with_clock(thread: threading.Thread) -> None:
        if thread is worker:
            assert thread._started.wait(timeout=30)
            assert len(values) == 1
            assert "threading.Event.wait" in result.owner_unhooked_joins
            thread._started.clear()
        start_thread(thread)

    class Worker(threading.Thread):
        """Thread subclass with a user synchronization in start."""

        start = start_with_clock

    worker = threading.Thread() if wrapped else Worker()

    def read_clock() -> None:
        ready.wait(timeout=30)
        values.append(time.time())
        worker._started.set()

    reader = threading.Thread(target=read_clock, name="w051-user-start-reader")
    reader.start()
    try:
        if wrapped:
            monkeypatch.setattr(threading.Thread, "start", start_with_clock)
        with host_nondeterminism_monitor(_LinearNoBias()) as result:
            ready.set()
            worker.start()
            worker.join(timeout=30)
    finally:
        ready.set()
        reader.join(timeout=30)
        if worker.ident is not None:
            worker.join(timeout=30)
        assert not reader.is_alive()
        assert not worker.is_alive()

    diagnostics = _monitor_diagnostics(result)
    assert result.uncertain, diagnostics
    assert "owner_thread_waited:threading.Event.wait" in result.uncertain_detail, diagnostics
    assert "foreign_thread_read_joined:time.time" in result.uncertain_detail, diagnostics


def test_started_event_wait_with_unknown_caller_still_counts() -> None:
    """A wait with an unidentifiable caller retains unattributed wait evidence."""

    monitor = host_nondeterminism_monitor(_LinearNoBias())
    started = threading.Thread()._started
    with monitor as result:
        _rng_channels.classify_owner_wait(
            monitor._owner_sync_session,
            _rng_channels.OwnerWait("threading.Event.wait", started, ("_cond",), None),
        )
    diagnostics = _monitor_diagnostics(result)
    assert result.owner_thread_waits == {"threading.Event.wait"}, diagnostics
    assert result.owner_unattributed_waits == {"threading.Event.wait"}, diagnostics


def test_torchlens_initiated_waits_are_not_joins() -> None:
    """A wait whose initiator frame is TorchLens-owned is machinery, never a join."""

    monitor = host_nondeterminism_monitor(_LinearNoBias())
    with monitor as result:
        # Simulate a TorchLens-owned initiator by resolving the wait from a frame whose
        # globals are a registered torchlens module's globals: exercise the classifier
        # directly with the rng package's own frame identity. The module object comes
        # from ``sys.modules``: an import statement inside the window would run the
        # ``sys.meta_path`` finders (pytest's assertion rewriter), whose frames seed the
        # frame-reachable inventory and reach other test modules' planted holders.
        session = monitor._owner_sync_session
        condition = threading.Condition()
        fake_frame = type("F", (), {})()
        fake_frame.f_globals = sys.modules["torchlens._utils.rng"].__dict__
        fake_frame.f_code = test_torchlens_initiated_waits_are_not_joins.__code__
        fake_frame.f_back = None
        _rng_channels.classify_owner_wait(
            session,
            _rng_channels.OwnerWait("threading.Condition.wait", condition, (), fake_frame),
        )
    assert not result.owner_thread_waits
    assert not result.uncertain


# ---- end to end through the runnable seam --------------------------------------------------


def test_prepool_feed_in_capture_is_not_verified(
    tmp_path: Path, warm_pool: ThreadPoolExecutor
) -> None:
    """End to end: the pre-existing-pool clock feed-in stamps monitor uncertainty on the
    seam, and the loaded replay reads unverifiable (was: verified with a divergent output)."""

    x = torch.randn(2, 4)
    model = _FeedIn(lambda: warm_pool.submit(time.time).result()).eval()
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))
    seam = trace._runnable
    assert seam.rng_monitor_uncertain
    assert "foreign_thread_read_joined:time.time" in seam.rng_monitor_uncertain_detail
    assert any(
        reason.startswith("owner_thread_waited:") for reason in seam.rng_monitor_uncertain_detail
    )
    bundle = tmp_path / "prepool.tlspec"
    trace.save(str(bundle), level="runnable", include_weights=True)
    result = tl.load(str(bundle)).run_transaction(inputs=x.clone())
    assert result.report.path_faithfulness.value == "unverifiable"


def test_lazy_pool_captures_never_verify_across_repeats() -> None:
    """The order-dependence corollary: a module that lazily builds a module-held pool was
    ceilinged on capture 0 (worker born in-window) and VERIFIED on captures 1 and 2 (same
    worker, now 'foreign'). Every capture must now be non-verified."""

    # Every pool the forward builds, not only ``model._pool``: the capture restores the
    # forward's attribute writes, so ``_pool`` reads None again after each capture and
    # the next capture builds a new pool. A pool left running is freed by a later
    # garbage collection, and its worker then exits inside an unrelated test's capture
    # window (``owner_thread_tripwire_changed`` there).
    pools: list[ThreadPoolExecutor] = []

    class _LazyPool(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 4, bias=False)
            self._pool: ThreadPoolExecutor | None = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=1)
                pools.append(self._pool)
            return self.lin(x) * (1.0 + (self._pool.submit(time.time).result() % 1.0))

    model = _LazyPool().eval()
    x = torch.randn(2, 4)
    try:
        for index in range(3):
            seam = tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))._runnable
            assert seam.host_rng_unreplayable or seam.rng_monitor_uncertain, index
    finally:
        for pool in pools:
            pool.shutdown(wait=True)


@pytest.mark.heavy  # measured 7.1-11.5s across CI rows (round-2 CI triage, 2026-10-01):
# consistently over the smoke 5s ceiling under a hammering unjoined thread plus a full
# runnable save/load/run round trip, not a one-off load spike.
def test_unjoined_flusher_capture_replays_verified(tmp_path: Path) -> None:
    """The FLAKEHUNT end-to-end pin, unchanged: a hammering pre-existing thread the owner
    never waits on leaves the runnable capture clean and the replay verified."""

    stop = threading.Event()

    def _hammer() -> None:
        while not stop.is_set():
            time.time()
            time.monotonic()

    hammer = threading.Thread(target=_hammer, name="w051-hammer", daemon=True)
    hammer.start()
    try:
        x = torch.randn(2, 4)
        trace = tl.trace(
            _LinearNoBias().eval(), x, capture=tl.options.CaptureOptions(replayable=True)
        )
        seam = trace._runnable
        assert not seam.host_rng_unreplayable
        assert not seam.rng_monitor_uncertain
        bundle = tmp_path / "flusher.tlspec"
        trace.save(str(bundle), level="runnable", include_weights=True)
        assert tl.load(str(bundle)).run_transaction(
            inputs=x.clone()
        ).report.path_faithfulness.value == ("verified")
    finally:
        stop.set()
        hammer.join(timeout=30)


# ---- vocabulary tripwires ---------------------------------------------------------------------


def test_owner_sync_vocabulary_resolves_completely() -> None:
    """Every owner-sync row resolves to a real stdlib function on this Python -- a stdlib
    rename is a failing test, never a silent under-witness."""

    assert _rng_channels.unresolved_owner_sync_rows() == ()
    displays = {row[0] for row in _rng_channels.OWNER_SYNC_PRIMITIVES}
    assert {
        "threading.Event.wait",
        "threading.Condition.wait",
        "threading.Thread.join",
        "concurrent.futures.Future.result",
        "queue.Queue.get",
        "asyncio.BaseEventLoop.run_until_complete",
    } <= displays
    assert _rng_channels.owner_sync_c_call_primitive(queue.SimpleQueue(), "get") == (
        "queue.SimpleQueue.get"
    )


def test_owner_sync_patches_restore_exactly() -> None:
    """The class patches on the wait/notify primitives restore identity-exact after the window."""

    originals = {
        (threading.Event, "wait"): threading.Event.wait,
        (threading.Condition, "notify"): threading.Condition.notify,
        (threading.Condition, "notify_all"): threading.Condition.notify_all,
        (Future, "result"): Future.result,
        (queue.Queue, "get"): queue.Queue.get,
        (threading.Thread, "join"): threading.Thread.join,
    }
    with host_nondeterminism_monitor(_LinearNoBias()) as result:
        for (holder, name), original in originals.items():
            assert getattr(holder, name) is not original, (holder, name)
    for (holder, name), original in originals.items():
        assert getattr(holder, name) is original, (holder, name)
    assert not result.uncertain


def test_registry_keeps_pure_read_rows_thread_agnostic_in_declaration() -> None:
    """Registry rows are unchanged by the join: the clock/entropy rows still declare
    thread scope ``any`` (observation is process-wide; routing is a settlement concern)."""

    clock_rows = [row for row in HOST_NONDETERMINISM_REGISTRY if row.family == "clock"]
    assert clock_rows and all(row.thread_scope in {"any", "hooked"} for row in clock_rows)
