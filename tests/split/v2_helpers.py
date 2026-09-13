"""Small test helper for constructing typed split v2 requests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torchlens.split import PlacementPlan, SplitPoint, SplitRequest


def split_request(
    boundary: str | SplitPoint,
    *,
    backend: str | None = None,
    trainable: bool = False,
    boundary_cache: bool = False,
    validation: str = "strict",
    live_param_sources: bool | None = None,
    batch_axes: dict[str, int] | None = None,
    placement: PlacementPlan | None = None,
) -> SplitRequest:
    """Construct a v2 request from the test matrix's compact boundary spelling."""

    from torchlens.split import SplitFeatures, SplitPoint, SplitRequest

    point = boundary if isinstance(boundary, SplitPoint) else _point_from_text(boundary)
    extra: dict[str, Any] = {}
    if placement is not None:
        extra["placement"] = placement
    return SplitRequest(
        point=point,
        backend=backend,
        features=SplitFeatures(
            training=trainable,
            boundary_cache=boundary_cache,
            live_param_sources=live_param_sources,
            batch_axes=None if batch_axes is None else dict(batch_axes),
        ),
        validation=validation,  # type: ignore[arg-type]
        **extra,
    )


def _point_from_text(boundary: str) -> SplitPoint:
    """Parse compact fixture syntax without exposing a production shim."""

    from torchlens.split import after, before, percent

    if boundary.startswith("after:"):
        return after(boundary.split(":", 1)[1])
    if boundary.startswith("before:"):
        return before(boundary.split(":", 1)[1])
    percent_text = boundary.split(":", 1)[1] if boundary.startswith("percent:") else boundary
    if percent_text.endswith("%"):
        percent_text = percent_text[:-1]
    try:
        return percent(float(percent_text))
    except ValueError as exc:
        raise ValueError(f"invalid test split point {boundary!r}") from exc


__all__ = ["split_request"]
