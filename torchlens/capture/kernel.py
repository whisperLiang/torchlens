"""DEPRECATED import-path shim for the removed capture kernel layer.

The fixed-order ``CaptureKernel``/``OpObservation`` pipeline was deleted in
the backend migration: production capture always ran the straight-line
per-op commit path in ``torchlens.backends.torch.ops``, and the kernel's
counters, stage ordering, and ledgers were an inert parallel lane. This
module keeps ``import torchlens.capture.kernel`` working for external code;
the classes are inert shells whose behavioral entry points raise.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, NoReturn

warnings.warn(
    "torchlens.capture.kernel is deprecated: the kernel/ledger/projector "
    "layer was removed and live capture commits ops on the straight-line "
    "backend path. This import shim will be dropped in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

InterventionTarget = Callable[[Any], Any]
ProducerTarget = Callable[..., None]
ObservationTarget = Callable[["OpObservation"], None]


@dataclass(slots=True)
class OpObservation:
    """DEPRECATED inert shim: transient live values for one backend operation.

    Nothing in TorchLens constructs or processes these anymore; the class
    exists only so the historical import keeps resolving.
    """

    operation_key: str
    value: Any
    normalize_metadata: ObservationTarget | None = None
    select: Callable[[OpObservation], Any] | None = None
    retain_payload: ObservationTarget | None = None
    append: ObservationTarget | None = None
    update_indexes_history: ObservationTarget | None = None
    evaluate_nonfinite_halt: ObservationTarget | None = None
    facts: dict[str, Any] = field(default_factory=dict)


class CaptureKernel:
    """DEPRECATED inert shim for the removed fixed-order capture kernel.

    Constructing it is allowed for import compatibility; the behavioral entry
    points raise because the stage pipeline they drove no longer exists.
    """

    __slots__ = ("_session",)

    def __init__(self, session: Any) -> None:
        """Store the session reference for repr/debug compatibility only."""

        self._session = session

    def _removed(self, entry_point: str) -> NoReturn:
        """Raise the uniform removed-layer error for one entry point."""

        raise RuntimeError(
            f"CaptureKernel.{entry_point} was removed with the kernel/ledger/"
            "projector layer: live capture commits each op on the straight-line "
            "backend path (torchlens.backends.torch.ops) instead."
        )

    def process(self, observation: OpObservation) -> None:
        """Raise: the fixed-order observation pipeline was removed."""

        self._removed("process")

    def emit(self, operation_key: str, producer: ProducerTarget, *producer_args: Any) -> None:
        """Raise: the legacy producer entry point was removed."""

        self._removed("emit")

    def apply_intervention(self, observation: OpObservation, target: InterventionTarget) -> Any:
        """Raise: kernel-mediated intervention was removed."""

        self._removed("apply_intervention")

    def begin_observation(self, operation_key: str) -> None:
        """Raise: kernel observation accounting was removed."""

        self._removed("begin_observation")

    def mark_metadata(self, *args: Any, **kwargs: Any) -> None:
        """Raise: kernel metadata staging was removed."""

        self._removed("mark_metadata")

    def mark_payload(self, *args: Any, **kwargs: Any) -> None:
        """Raise: kernel payload staging was removed."""

        self._removed("mark_payload")
