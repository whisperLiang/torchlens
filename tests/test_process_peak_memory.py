"""Native peak-memory measurements preserve platform units and availability."""

import sys
from collections.abc import Callable
from types import SimpleNamespace

import pytest

from benchmarks._memory import peak_rss_bytes
from torchlens.validation.consolidated import _rss_high_water_bytes

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("measure", [_rss_high_water_bytes, peak_rss_bytes])
def test_windows_reports_peak_working_set(
    monkeypatch: pytest.MonkeyPatch, measure: Callable[[], int | None]
) -> None:
    """Return the native peak, even when the current working set is smaller."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "resource", None)
    info = SimpleNamespace(peak_wset=4096, rss=1024)
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(Process=lambda: SimpleNamespace(memory_info=lambda: info)),
    )
    assert measure() == 4096


@pytest.mark.parametrize(("platform", "expected"), [("linux", 4096), ("darwin", 4)])
@pytest.mark.parametrize("measure", [_rss_high_water_bytes, peak_rss_bytes])
def test_posix_peak_memory_units(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    expected: int,
    measure: Callable[[], int | None],
) -> None:
    """Scale Linux kilobytes while retaining macOS byte measurements."""
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setitem(
        sys.modules,
        "resource",
        SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda _: SimpleNamespace(ru_maxrss=4)),
    )
    assert measure() == expected


def test_missing_native_measurement_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing optional measurement support does not fabricate a zero peak."""
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "resource", None)
    monkeypatch.setitem(sys.modules, "psutil", None)
    assert _rss_high_water_bytes() is None
