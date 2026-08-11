"""Running budget for retained activation bytes, with a typed refusal.

``tl.trace(model, x)`` retains every operation's output by default. That is the
right default for the models TorchLens was designed around and a footgun at
frontier shapes: a first-time user pointing the default at a large model gets an
OOM kill (or an allocator ``RuntimeError`` from somewhere deep inside torch)
rather than an explanation.

This module makes that failure honest. Retained payload bytes are charged to a
per-device running total as they are committed, and when a device's total crosses
its budget capture stops with :class:`SaveBudgetExceededError`, naming the bytes
already committed, the budget and where the budget came from, the operation that
tripped it, and the remedies.

The accounting is deliberately a **lower bound, labelled as one**. At the moment
of the trip the forward is incomplete, so the true footprint of the finished
capture would have been larger; the error says exactly that instead of
extrapolating a total it cannot know. What it can state precisely -- committed
bytes, saved-activation count, mean bytes per saved activation, and the tripping
site -- it does.

Budgets are per-device because saved payloads follow the tensors they copy
(``output_device="same"`` by default), so a CUDA capture spends VRAM and a CPU
capture spends host RAM. Devices whose headroom cannot be measured are reported
as unbudgeted rather than silently assumed infinite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Union

import torch

from .errors._base import CaptureError

__all__ = [
    "DEFAULT_SAVE_BUDGET_FRACTION",
    "SaveBudget",
    "SaveBudgetExceededError",
    "SaveBudgetOption",
    "available_device_bytes",
    "format_bytes",
    "resolve_save_budget",
]

SaveBudgetOption = Union[str, int, float, None]
"""Accepted ``save_budget`` spellings: ``"auto"``, a fraction, bytes, or ``None``."""

DEFAULT_SAVE_BUDGET_FRACTION = 0.5
"""Fraction of a device's *available* memory the ``"auto"`` budget allows.

Half of free memory, not of total: the traced model's own parameters and
activations already occupy the difference. Half leaves headroom for the forward
pass itself, which allocates live intermediates alongside every payload
TorchLens retains.
"""

_BYTES_UNITS = (("TB", 1024**4), ("GB", 1024**3), ("MB", 1024**2), ("KB", 1024), ("B", 1))


class SaveBudgetExceededError(CaptureError, RuntimeError):
    """Raised when retained activation bytes cross the configured save budget.

    The structured accounting is retained on ``fields`` (``committed_bytes``,
    ``budget_bytes``, ``device``, ``num_saved``, ``label``) so callers branch on
    numbers rather than parsing the message.
    """


def format_bytes(num_bytes: float) -> str:
    """Render a byte count in the largest unit that keeps it readable.

    Parameters
    ----------
    num_bytes:
        Byte count.

    Returns
    -------
    str
        Human-readable size such as ``"1.44 GB"``.
    """

    for unit, scale in _BYTES_UNITS:
        if abs(num_bytes) >= scale or unit == "B":
            if unit == "B":
                return f"{int(num_bytes)} B"
            return f"{num_bytes / scale:.2f} {unit}"
    return f"{int(num_bytes)} B"


def _available_host_bytes() -> int | None:
    """Return available host RAM in bytes, or ``None`` when unmeasurable.

    Returns
    -------
    int | None
        Available bytes. Prefers ``MemAvailable`` from ``/proc/meminfo`` because
        it accounts for reclaimable cache, and falls back to the POSIX
        available-pages count.
    """

    try:
        with open("/proc/meminfo", encoding="ascii") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return int(pages) * int(page_size)
    except Exception:
        pass
    return None


def available_device_bytes(device: torch.device) -> int | None:
    """Return the memory headroom for one device, or ``None`` when unmeasurable.

    Parameters
    ----------
    device:
        Device whose free memory is queried.

    Returns
    -------
    int | None
        Free bytes, or ``None`` when this device type exposes no headroom query.
        ``None`` means *unbudgeted and reported as such*, never "assume
        infinite silently".
    """

    device_type = device.type
    if device_type == "cpu":
        return _available_host_bytes()
    if device_type == "cuda":
        try:
            free_bytes, _total = torch.cuda.mem_get_info(device)
            return int(free_bytes)
        except Exception:
            return None
    if device_type == "meta":
        # Meta tensors have no storage, so nothing is ever really committed.
        return None
    return None


@dataclass(frozen=True)
class _BudgetSpec:
    """Resolved budget policy.

    Parameters
    ----------
    fraction:
        Fraction of a device's available memory to allow, when the policy is
        headroom-relative.
    absolute_bytes:
        Fixed per-device byte cap, when the policy is absolute.
    source:
        Human-readable description of where the policy came from, quoted in the
        refusal so the user can tell a default from their own setting.
    """

    fraction: float | None
    absolute_bytes: int | None
    source: str


def resolve_save_budget(value: SaveBudgetOption) -> _BudgetSpec | None:
    """Resolve a ``save_budget`` option value into a budget policy.

    Parameters
    ----------
    value:
        ``"auto"`` for the default headroom fraction, a float in ``(0, 1]`` for a
        custom fraction of available memory, an int (``>= 1``) for an absolute
        per-device byte cap, or ``None`` to disable budgeting.

    Returns
    -------
    _BudgetSpec | None
        Resolved policy, or ``None`` when budgeting is disabled.

    Raises
    ------
    ValueError
        If ``value`` is not one of the documented spellings. Invalid budgets fail
        loudly rather than silently disabling the guard.
    """

    if value is None:
        return None
    if isinstance(value, str):
        if value != "auto":
            raise ValueError(
                f"save_budget string must be 'auto'; got {value!r}. Use a float in (0, 1] for a "
                "fraction of available memory, an int for absolute bytes, or None to disable."
            )
        return _BudgetSpec(
            fraction=DEFAULT_SAVE_BUDGET_FRACTION,
            absolute_bytes=None,
            source=(
                f"default save_budget='auto' "
                f"({DEFAULT_SAVE_BUDGET_FRACTION:.0%} of available memory)"
            ),
        )
    if isinstance(value, bool):
        raise ValueError(
            "save_budget does not accept bool; use None to disable or 'auto' for the default."
        )
    if isinstance(value, float):
        if not 0.0 < value <= 1.0:
            raise ValueError(
                f"save_budget float must be a fraction in (0, 1]; got {value!r}. "
                "Pass an int for an absolute byte cap."
            )
        return _BudgetSpec(
            fraction=value,
            absolute_bytes=None,
            source=f"save_budget={value!r} ({value:.0%} of available memory)",
        )
    if isinstance(value, int):
        if value < 1:
            raise ValueError(
                f"save_budget int must be at least 1 byte; got {value!r}. Use None to disable."
            )
        return _BudgetSpec(
            fraction=None,
            absolute_bytes=value,
            source=f"save_budget={value} ({format_bytes(value)} per device)",
        )
    raise ValueError(
        f"save_budget must be 'auto', a float in (0, 1], an int of bytes, or None; "
        f"got {type(value).__name__}"
    )


@dataclass
class _DeviceLedger:
    """Per-device running total and resolved limit."""

    committed_bytes: int = 0
    num_saved: int = 0
    limit_bytes: int | None = None
    available_bytes: int | None = None
    measured: bool = False


@dataclass
class SaveBudget:
    """Per-device running accountant for retained activation bytes.

    Parameters
    ----------
    spec:
        Resolved budget policy.

    Notes
    -----
    ``charge`` is on the capture hot path, once per retained payload. It is one
    dict lookup, one integer add, and one compare in the common case; device
    headroom is measured lazily on a device's first charge, so a capture that
    retains nothing pays nothing.
    """

    spec: _BudgetSpec
    ledgers: dict[str, _DeviceLedger] = field(default_factory=dict)
    tripped: bool = False

    @classmethod
    def from_option(cls, value: SaveBudgetOption) -> "SaveBudget | None":
        """Build a budget from a user-facing option value.

        Parameters
        ----------
        value:
            ``save_budget`` option value.

        Returns
        -------
        SaveBudget | None
            Accountant, or ``None`` when budgeting is disabled.
        """

        spec = resolve_save_budget(value)
        if spec is None:
            return None
        return cls(spec=spec)

    def _ledger_for(self, device: torch.device) -> _DeviceLedger:
        """Return (creating if needed) the ledger for one device.

        Parameters
        ----------
        device:
            Device whose ledger is requested.

        Returns
        -------
        _DeviceLedger
            Ledger with its limit resolved on first use.
        """

        key = str(device)
        ledger = self.ledgers.get(key)
        if ledger is not None:
            return ledger
        ledger = _DeviceLedger()
        if self.spec.absolute_bytes is not None:
            ledger.limit_bytes = self.spec.absolute_bytes
            ledger.measured = True
        else:
            available = available_device_bytes(device)
            ledger.available_bytes = available
            ledger.measured = available is not None
            if available is not None:
                fraction = self.spec.fraction or DEFAULT_SAVE_BUDGET_FRACTION
                ledger.limit_bytes = int(available * fraction)
        self.ledgers[key] = ledger
        return ledger

    def charge(self, label: str, device: torch.device, num_bytes: int) -> None:
        """Charge retained payload bytes and refuse when the budget is crossed.

        Parameters
        ----------
        label:
            Layer label of the operation whose payload is being retained, used to
            name the tripping site.
        device:
            Device the payload is retained on.
        num_bytes:
            Payload size in bytes.

        Raises
        ------
        SaveBudgetExceededError
            When this device's committed total crosses its budget.
        """

        if num_bytes <= 0:
            return
        ledger = self._ledger_for(device)
        ledger.committed_bytes += int(num_bytes)
        ledger.num_saved += 1
        limit = ledger.limit_bytes
        if limit is None or ledger.committed_bytes <= limit:
            return
        self.tripped = True
        raise SaveBudgetExceededError(
            self._message(label, device, ledger),
            committed_bytes=ledger.committed_bytes,
            budget_bytes=limit,
            available_bytes=ledger.available_bytes,
            device=str(device),
            num_saved=ledger.num_saved,
            label=label,
        )

    def unbudgeted_devices(self) -> tuple[str, ...]:
        """Return devices that were charged but could not be budgeted.

        Returns
        -------
        tuple[str, ...]
            Device strings whose headroom could not be measured, so no budget was
            enforced for them.
        """

        return tuple(
            key for key, ledger in self.ledgers.items() if not ledger.measured
        )

    def _message(self, label: str, device: torch.device, ledger: _DeviceLedger) -> str:
        """Build the refusal message.

        Parameters
        ----------
        label:
            Tripping layer label.
        device:
            Device whose budget was crossed.
        ledger:
            Ledger at the moment of the trip.

        Returns
        -------
        str
            Explanatory message naming the committed footprint and the remedies.
        """

        limit = ledger.limit_bytes or 0
        mean_bytes = ledger.committed_bytes / max(ledger.num_saved, 1)
        headroom = (
            f" (device had {format_bytes(ledger.available_bytes)} available at first save)"
            if ledger.available_bytes is not None
            else ""
        )
        return (
            "torchlens stopped capture: retained activations crossed the save budget on "
            f"{device}.\n"
            f"  committed so far: {format_bytes(ledger.committed_bytes)} across "
            f"{ledger.num_saved} saved activation(s), mean {format_bytes(mean_bytes)} each\n"
            f"  budget: {format_bytes(limit)} from {self.spec.source}{headroom}\n"
            f"  tripped while saving: {label}\n"
            "  This is a LOWER BOUND: the forward pass was still running, so a completed "
            "default capture of this model would have retained more than the figure above.\n"
            "  Remedies, cheapest first:\n"
            "    - save less: save=tl.func('relu') or save=tl.in_module('encoder') "
            "instead of the default save='all'\n"
            "    - stream payloads to disk: storage=tl.to_disk('run.tlspec')\n"
            "    - keep metadata only: layers_to_save='none' (the graph is still captured)\n"
            "    - raise or lift the budget deliberately: "
            "capture=tl.options.CaptureOptions(save_budget=<bytes|fraction>), or "
            "save_budget=None to disable it"
        )
