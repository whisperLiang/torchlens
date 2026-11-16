"""Native Split Runtime construction API."""

from __future__ import annotations

from typing import Any, Literal

from ..backends.registry import TORCH_BACKEND_NAME, resolve_backend_spec
from .adapters import resolve_split_adapter
from .context import SplitGraph
from .errors import SplitErrorContext, SplitUnsupportedError
from .ir import GraphPartition, SplitFeatures, SplitModelProfile, SplitRequest
from .pipeline import (
    capture_canonical_model,
    normalize_to_split_ir,
)
from .placement import PlacementPlan, require_placement_support
from .profiles import resolve_model_profile
from .runtime import SplitRuntime

_DEFAULT_FEATURES = SplitFeatures()
_DEFAULT_PLACEMENT = PlacementPlan()


def _prepare_graph(
    model: Any,
    inputs: Any,
    request: SplitRequest,
    *,
    input_kwargs: dict[str, Any] | None = None,
) -> SplitGraph:
    """Capture a backend-neutral graph and shape program from typed policies.

    Parameters
    ----------
    model:
        Model or callable to capture.
    inputs:
        Example positional inputs.
    request:
        Canonical partition, backend, profile, and feature request. Torch inference
        defaults to a compact runtime without a diagnostic Trace. Request
        ``features=SplitFeatures(retain_trace=True)`` to retain saved activations.
        Training and other backends retain their complete captures by default.
        Torch training can opt into compact retention with ``retain_trace=False``;
        live state and replay metadata remain available for fresh autograd graphs.
    input_kwargs:
        Optional keyword inputs for the native capture.

    Returns
    -------
    SplitGraph
        Captured graph context without executable segments.
    """

    profile = resolve_model_profile(request.model_profile)
    backend_name = request.backend or (None if profile is None else profile.backend) or "torch"
    if profile is not None and profile.backend != backend_name:
        raise ValueError(
            f"split request backend {backend_name!r} does not match model profile "
            f"backend {profile.backend!r}"
        )

    input_tuple = inputs if isinstance(inputs, tuple) else (inputs,)
    backend_spec = resolve_backend_spec(backend_name, model, input_tuple, input_kwargs)
    adapter = resolve_split_adapter(backend_spec)
    if not request.features.replay or not adapter.supports_replay:
        raise SplitUnsupportedError(
            f"backend={backend_spec.name!r} does not support split replay.",
            context=SplitErrorContext(
                backend=str(backend_spec.name),
                split_point=request.boundary,
                reason="unsupported split replay",
            ),
            code=SplitUnsupportedError.code,
        )
    if request.trainable and not adapter.supports_training:
        raise SplitUnsupportedError(
            f"backend={backend_spec.name!r} does not support split training.",
            context=SplitErrorContext(
                backend=str(backend_spec.name),
                split_point=request.boundary,
                reason="unsupported split training",
            ),
            code=SplitUnsupportedError.code,
        )
    require_placement_support(adapter, request.placement, split_point=request.boundary)

    compact = adapter.name == TORCH_BACKEND_NAME and (
        request.features.retain_trace is False
        or (request.features.retain_trace is None and not request.trainable)
    )
    if request.features.retain_trace is False and not compact:
        raise SplitUnsupportedError(
            "retain_trace=False is supported only for Torch; other backends require "
            "their complete capture. Use retain_trace=True or None.",
            context=SplitErrorContext(
                backend=str(backend_spec.name),
                split_point=request.boundary,
                reason="compact_capture_unsupported",
            ),
            code=SplitUnsupportedError.code,
        )

    capture, canonical_inputs, canonical_kwargs, batch_spec = capture_canonical_model(
        model,
        input_tuple,
        request,
        input_kwargs=input_kwargs,
        adapter=adapter,
    )
    try:
        graph, graph_ir = normalize_to_split_ir(
            capture,
            request,
            inputs=canonical_inputs,
            input_kwargs=canonical_kwargs,
            adapter=adapter,
            model_profile=profile,
            model=model,
            batch_spec=batch_spec,
            _release_capture=compact,
        )
    finally:
        if compact:
            # Also release the owned diagnostic capture if normalization fails.
            # Successful normalization already releases it before the B=2 probe.
            capture.cleanup()
    return SplitGraph(
        model=model,
        trace=None if compact else capture,
        trace_graph=graph,
        request=request,
        adapter=adapter,
        graph_ir=graph_ir,
        model_profile=profile,
        prepared_input_kwargs=input_kwargs,
        batch_spec=batch_spec,
    )


def prepare_graph(
    model: Any,
    inputs: Any,
    *,
    backend: str | None = None,
    features: SplitFeatures = _DEFAULT_FEATURES,
    model_profile: str | SplitModelProfile | None = None,
    placement: PlacementPlan = _DEFAULT_PLACEMENT,
    validation: Literal["strict", "permissive"] = "strict",
    input_kwargs: dict[str, Any] | None = None,
) -> SplitGraph:
    """Capture once before choosing a canonical GraphPartition.

    Parameters
    ----------
    model:
        Backend model or callable.
    inputs:
        Example positional inputs; batch axes follow ``features.batch_axes``.
    backend, features, model_profile, placement, validation:
        Capture, shape, backend and execution policies.
    input_kwargs:
        Example keyword arguments.

    Returns
    -------
    SplitGraph
        Shared graph and shape metadata with analysis and materialization doors.
        No persistent executable segments or private parameter replicas are created.

    Examples
    --------
    >>> graph = tl.split.prepare_graph(torch.nn.Linear(3, 2).eval(), torch.ones(1, 3))
    >>> partition = graph.compute_dag().to_partition(())
    >>> runtime = graph.materialize(graph.analyze(partition))
    >>> runtime.replay(torch.ones(4, 3)).shape
    torch.Size([4, 2])
    """
    return _prepare_graph(
        model,
        inputs,
        SplitRequest(
            GraphPartition(frozenset()),
            backend=backend,
            features=features,
            model_profile=model_profile,
            placement=placement,
            validation=validation,
        ),
        input_kwargs=input_kwargs,
    )


def prepare(
    model: Any, inputs: Any, request: SplitRequest, *, input_kwargs: dict[str, Any] | None = None
) -> SplitRuntime:
    """Capture and materialize when canonical partition IDs are already known.

    Parameters
    ----------
    model, inputs:
        Backend model and example positional inputs.
    request:
        Canonical GraphPartition and capture/execution policies.
    input_kwargs:
        Optional example keyword inputs.

    Returns
    -------
    SplitRuntime
        Runtime produced by the same two-stage graph workflow.

    Examples
    --------
    >>> request = tl.SplitRequest(tl.split.GraphPartition(frozenset()))
    >>> runtime = tl.prepare(torch.nn.Linear(3, 2).eval(), torch.ones(1, 3), request)
    >>> runtime.replay(torch.ones(2, 3)).shape
    torch.Size([2, 2])
    """
    graph = _prepare_graph(model, inputs, request, input_kwargs=input_kwargs)
    return graph.materialize(graph.analyze(request.partition))


__all__ = ("prepare", "prepare_graph")
