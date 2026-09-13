"""Empirical batch validation, deliberately distinct from universal proof."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .errors import SplitBoundaryError, SplitErrorContext


@dataclass(frozen=True)
class BatchProbeResult:
    """Result of comparing one native B=2 forward against generated replay."""

    traced_batch_size: int
    probe_batch_size: int | None
    status: Literal["passed", "failed", "unavailable"]
    reason: str | None = None
    atol: float = 1e-5
    rtol: float = 1e-4

    def as_dict(self) -> dict[str, Any]:
        """Return the sampled scope without implying untested batches are verified."""

        return {
            "policy": "single_probe",
            "mode": "sampled_extrapolation" if self.status == "passed" else "captured_only",
            "status": self.status,
            "traced_batch_size": self.traced_batch_size,
            "probe_batch_size": self.probe_batch_size,
            "reason": self.reason,
            "atol": self.atol,
            "rtol": self.rtol,
            "universal_proof": False,
            "limitation": (
                "Untested batches may take different Python branches and return incorrect results."
            ),
        }

    def require_batch(self, batch_size: int, *, backend: str, split_point: str) -> None:
        """Allow the captured batch, or extrapolation after the sample passed."""

        if batch_size == self.traced_batch_size or self.status == "passed":
            return
        raise SplitBoundaryError(
            f"Runtime batch {batch_size} differs from captured batch {self.traced_batch_size}; "
            f"the B=2 replay probe did not pass ({self.status}): {self.reason}",
            context=SplitErrorContext(
                backend=backend,
                split_point=split_point,
                reason="batch probe did not pass",
            ),
        )
