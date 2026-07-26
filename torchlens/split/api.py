"""Public v2 split runtime construction API."""

from __future__ import annotations

from typing import Any

from ..backends import resolve_backend_spec
from .adapters import resolve_split_adapter
from .errors import SplitErrorContext, SplitUnsupportedError
from .ir import SplitRequest
from .pipeline import (
    analyze_split_capabilities,
    capture_model,
    execute_split_runtime,
    lower_split_program,
    normalize_to_split_ir,
)
from .planner import plan_split
from .profiles import resolve_model_profile
from .program import ensure_capability_report_supported
from .runtime import SplitRuntime


def _normalize_inputs(inputs: Any) -> tuple[Any, ...]:
    """Normalize public positional inputs into a tuple."""

    if isinstance(inputs, tuple):
        return inputs
    return (inputs,)


def prepare(
    model: Any,
    inputs: Any,
    request: SplitRequest,
    *,
    input_kwargs: dict[str, Any] | None = None,
) -> SplitRuntime:
    """Prepare a backend-neutral split runtime from a v2 request.

    Parameters
    ----------
    model:
        Model or callable to capture.
    inputs:
        Example positional inputs.
    request:
        Typed split point, backend, profile, and feature request.
    input_kwargs:
        Optional keyword inputs for the native capture.

    Returns
    -------
    SplitRuntime
        Prepared split runtime.
    """

    profile = resolve_model_profile(request.model_profile)
    backend_name = request.backend or (None if profile is None else profile.backend) or "torch"
    if profile is not None and profile.backend != backend_name:
        raise ValueError(
            f"split request backend {backend_name!r} does not match model profile "
            f"backend {profile.backend!r}"
        )

    input_tuple = _normalize_inputs(inputs)
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
        )
    if request.trainable and not adapter.supports_training:
        raise SplitUnsupportedError(
            f"backend={backend_spec.name!r} does not support split training.",
            context=SplitErrorContext(
                backend=str(backend_spec.name),
                split_point=request.boundary,
                reason="unsupported split training",
            ),
        )
    if request.dynamic_batch is not None and not adapter.supports_dynamic_batch:
        raise SplitUnsupportedError(
            f"backend={backend_spec.name!r} does not support dynamic-batch split replay.",
            context=SplitErrorContext(
                backend=str(backend_spec.name),
                split_point=request.boundary,
                reason="unsupported dynamic batch",
            ),
        )

    capture = capture_model(model, input_tuple, request, input_kwargs=input_kwargs)
    graph, graph_ir = normalize_to_split_ir(
        capture,
        request,
        inputs=input_tuple,
        input_kwargs=input_kwargs,
        adapter=adapter,
        model_profile=profile,
        model=model,
    )
    plan = plan_split(graph, request)
    prefix_program = lower_split_program(graph, plan, request, segment="prefix", adapter=adapter)
    suffix_program = lower_split_program(graph, plan, request, segment="suffix", adapter=adapter)
    capability_report = analyze_split_capabilities(
        adapter,
        graph,
        plan,
        request,
        prefix_program=prefix_program,
        suffix_program=suffix_program,
        graph_ir=graph_ir,
        model_profile=profile,
        features=request.features.as_dict(),
    )
    if request.validation == "strict":
        ensure_capability_report_supported(capability_report, request)
    segments = execute_split_runtime(adapter, graph, plan, request)
    return SplitRuntime(
        model=model,
        trace=capture,
        trace_graph=graph,
        request=request,
        plan=plan,
        adapter=adapter,
        segments=segments,
        capability_report=capability_report,
        prefix_program=prefix_program,
        suffix_program=suffix_program,
        graph_ir=graph_ir,
        model_profile=profile,
        prepared_input_kwargs=input_kwargs,
    )


__all__ = ["prepare"]
