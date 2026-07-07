"""Public split runtime construction APIs."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..backends import resolve_backend_spec
from .adapters import resolve_split_adapter
from .errors import SplitErrorContext, SplitUnsupportedError
from .graph import split_graph_from_trace
from .planner import plan_split
from .program import (
    build_capability_report,
    ensure_capability_report_supported,
    lower_replay_program,
)
from .runtime import SplitRuntime
from .spec import SplitSpec


def _normalize_example_inputs(example_inputs: Any) -> tuple[Any, ...]:
    """Normalize public example inputs into a tuple."""

    if isinstance(example_inputs, tuple):
        return example_inputs
    return (example_inputs,)


def _reject_unsupported_construction(spec: SplitSpec, backend_name: str) -> None:
    """Reject unsupported runtime construction options."""

    if spec.mode == "compiled":
        raise SplitUnsupportedError(
            "SplitSpec.mode='compiled' is accepted but not supported in v1.",
            context=SplitErrorContext(
                backend=backend_name,
                split_point=spec.boundary,
                module_path=None,
                op_type=None,
                layer_label=None,
                reason="compiled split mode unsupported",
            ),
        )


def _trace_kwargs_for_backend(
    backend_name: str,
    spec: SplitSpec,
    example_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return backend-compatible trace kwargs for split preparation."""

    common: dict[str, Any] = {
        "input_kwargs": example_kwargs,
        "layers_to_save": "all",
        "keep_orphans": True,
        "backend": backend_name,
    }
    if backend_name == "torch":
        return {
            **common,
            "intervention_ready": True,
            "capture_container_structure": True,
            "save_arg_values": True,
            "save_rng_states": True,
            "detach_saved_activations": False,
            "backward_ready": spec.trainable,
        }
    return common


def prepare_split(
    model: Any,
    example_inputs: Any,
    spec: SplitSpec,
    *,
    example_kwargs: dict[str, Any] | None = None,
) -> SplitRuntime:
    """Prepare a backend-neutral split runtime.

    Parameters
    ----------
    model:
        Model or callable to capture.
    example_inputs:
        Example positional inputs.
    spec:
        Split specification.
    example_kwargs:
        Optional keyword inputs for the trace capture.

    Returns
    -------
    SplitRuntime
        Prepared split runtime.
    """

    input_tuple = _normalize_example_inputs(example_inputs)
    backend_spec = resolve_backend_spec(spec.backend, model, input_tuple, example_kwargs)
    adapter = resolve_split_adapter(backend_spec)
    _reject_unsupported_construction(spec, str(backend_spec.name))
    if not adapter.supports_replay:
        raise SplitUnsupportedError(
            f"backend={backend_spec.name!r} does not support split replay.",
            context=SplitErrorContext(
                backend=str(backend_spec.name),
                split_point=spec.boundary,
                module_path=None,
                op_type=None,
                layer_label=None,
                reason="unsupported split replay",
            ),
        )
    if spec.trainable and not adapter.supports_training:
        raise SplitUnsupportedError(
            f"backend={backend_spec.name!r} does not support split training.",
            context=SplitErrorContext(
                backend=str(backend_spec.name),
                split_point=spec.boundary,
                module_path=None,
                op_type=None,
                layer_label=None,
                reason="unsupported split training",
            ),
        )
    if spec.dynamic_batch is not None and not adapter.supports_dynamic_batch:
        raise SplitUnsupportedError(
            f"backend={backend_spec.name!r} does not support dynamic-batch split replay.",
            context=SplitErrorContext(
                backend=str(backend_spec.name),
                split_point=spec.boundary,
                module_path=None,
                op_type=None,
                layer_label=None,
                reason="unsupported dynamic batch",
            ),
        )
    from ..user_funcs import trace as _trace

    trace = _trace(
        model,
        input_tuple,
        **_trace_kwargs_for_backend(str(backend_spec.name), spec, example_kwargs),
    )
    graph = split_graph_from_trace(
        trace,
        batch_symbol=spec.batch_symbol,
        dynamic_batch=spec.dynamic_batch,
    )
    plan = plan_split(graph, spec)
    prefix_program = lower_replay_program(graph, plan, spec, segment="prefix")
    suffix_program = lower_replay_program(graph, plan, spec, segment="suffix")
    capability_report = build_capability_report(
        adapter,
        graph,
        plan,
        spec,
        prefix_program=prefix_program,
        suffix_program=suffix_program,
    )
    ensure_capability_report_supported(capability_report, spec)
    segments = adapter.build_segments(graph, plan, spec)
    return SplitRuntime(
        model=model,
        trace=trace,
        trace_graph=graph,
        split_spec=spec,
        plan=plan,
        adapter=adapter,
        segments=segments,
        capability_report=capability_report,
        prefix_program=prefix_program,
        suffix_program=suffix_program,
    )


def prepare_split_replay(
    model: Any,
    example_inputs: Any,
    spec: SplitSpec,
    *,
    example_kwargs: dict[str, Any] | None = None,
) -> SplitRuntime:
    """Prepare an inference-only split replay runtime from ``spec``."""

    replay_spec = replace(spec, trainable=False)
    return prepare_split(
        model,
        example_inputs,
        replay_spec,
        example_kwargs=example_kwargs,
    )


__all__ = ["prepare_split", "prepare_split_replay"]
