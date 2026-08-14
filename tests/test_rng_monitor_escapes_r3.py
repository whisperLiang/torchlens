"""Round-3 RNG monitor closures: uuid1's C funnel, raw ``_thread`` spawns,
and bounded ``uncertain_detail`` growth.

Three hunt-b8 findings on ``utils/rng.py``:

* ``uuid.uuid1()`` escaped the seal completely on Linux -- the libuuid
  ``uuid._generate_time_safe`` C path (wall clock + clock-seq entropy + node)
  touches no monitored Python surface, so an in-window ``uuid1()`` reported
  ``channels=[]`` / ``uncertain=False``: a clean false-VERIFIED. The Python
  fallback path was already caught.
* Threads started IN-WINDOW via raw ``_thread.start_new_thread`` were never
  profile-hooked (``threading.setprofile`` rides only ``threading.Thread``'s
  bootstrap), so a raw thread drawing an externally-held numpy generator was
  unwitnessed -- outside the documented residual, which covers PRE-EXISTING
  threads only.
* ``_flag_uncertain`` grew ``uncertain_detail`` by a full tuple copy per call
  with no dedupe or cap -- measured O(N^2); a persistently-raising profiled
  object turned a capture into an effective hang.
"""

from __future__ import annotations

import threading
import uuid

import numpy as np
import pytest
import torch
from torch import nn

from torchlens.utils.rng import (
    _UNCERTAIN_DETAIL_CAP,
    host_nondeterminism_monitor,
)


@pytest.mark.smoke
def test_uuid1_c_path_is_marked_and_restored() -> None:
    """In-window ``uuid.uuid1()`` marks its channel through the C funnel too."""

    if getattr(uuid, "_generate_time_safe", None) is None and (
        getattr(uuid, "_UuidCreate", None) is None
    ):
        pytest.skip("no platform C uuid1 funnel to exercise; fallback path is covered")
    pre_window_funnel = getattr(uuid, "_generate_time_safe", None)

    with host_nondeterminism_monitor(nn.Identity()) as result:
        uuid.uuid1()
    assert "uuid.uuid1" in result.channels, (
        "uuid.uuid1() drew wall-clock/entropy through the libuuid C path without "
        f"marking any channel: {sorted(result.channels)!r}"
    )
    # Exact restoration: the module attr holds the pre-window funnel again.
    assert getattr(uuid, "_generate_time_safe", None) is pre_window_funnel


@pytest.mark.smoke
def test_uuid4_still_marks_through_os_urandom() -> None:
    """Sibling pin: uuid4's os.urandom feed stays witnessed."""

    with host_nondeterminism_monitor(nn.Identity()) as result:
        uuid.uuid4()
    assert "os.urandom" in result.channels


@pytest.mark.smoke
def test_raw_thread_spawned_in_window_is_profile_hooked() -> None:
    """A raw ``_thread.start_new_thread`` thread's host draws are witnessed."""

    import _thread

    external_generator = np.random.default_rng(7)  # seeded OUTSIDE the window
    pre_window_spawn = _thread.start_new_thread
    done = threading.Event()

    def draw_from_raw_thread() -> None:
        """Draw from an externally-held generator, then signal completion."""

        try:
            external_generator.random()
        finally:
            done.set()

    with host_nondeterminism_monitor(nn.Identity()) as result:
        _thread.start_new_thread(draw_from_raw_thread, ())
        assert done.wait(timeout=10.0), "raw thread never ran"
    assert result.channels, (
        "an in-window raw _thread.start_new_thread thread drew from an "
        "externally-held numpy generator with no channel marked: the spawn "
        "was never profile-hooked"
    )
    assert _thread.start_new_thread is pre_window_spawn, (
        "the _thread.start_new_thread patch did not restore"
    )


@pytest.mark.smoke
def test_threading_thread_in_window_draw_still_witnessed() -> None:
    """Sibling pin: the threading.Thread in-window positive control holds."""

    external_generator = np.random.default_rng(11)

    def draw() -> None:
        """Draw through a Python frame so the frame-digest belt can witness."""

        external_generator.random()

    with host_nondeterminism_monitor(nn.Identity()) as result:
        worker = threading.Thread(target=draw)
        worker.start()
        worker.join(timeout=10.0)
    assert result.channels


@pytest.mark.smoke
def test_flag_uncertain_detail_is_deduped_and_capped() -> None:
    """Per-event uncertainty reasons cannot grow the detail quadratically."""

    monitor = host_nondeterminism_monitor(nn.Identity())

    # Repeated identical reason: one retained entry, O(1) per repeat.
    for _ in range(10_000):
        monitor._flag_uncertain("profile_rng_state_read_failed:BrokenThing")
    assert monitor.result.uncertain is True
    assert monitor.result.uncertain_detail == ("profile_rng_state_read_failed:BrokenThing",)

    # Distinct reasons past the cap: retained set bounded, overflow disclosed.
    for index in range(10_000):
        monitor._flag_uncertain(f"reason:{index}")
    detail = monitor.result.uncertain_detail
    assert len(detail) <= _UNCERTAIN_DETAIL_CAP + 1
    assert detail[-1] == "uncertain_detail_capped"
    assert monitor.result.uncertain is True


@pytest.mark.smoke
def test_flag_uncertain_hot_loop_is_fast() -> None:
    """1e5 repeated flags finish in well under a second (was O(N^2) copies)."""

    import time

    monitor = host_nondeterminism_monitor(nn.Identity())
    start = time.perf_counter()
    for _ in range(100_000):
        monitor._flag_uncertain("profile_classifier_error:X")
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"_flag_uncertain hot loop took {elapsed:.2f}s for 1e5 calls"
    assert torch is not None
